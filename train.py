import os
import math
import time
import torch
import argparse
import numpy as np
import requests

from model.model import ShakGPT
from model.config import ModelConfig
from liger_kernel.transformers import LigerCrossEntropyLoss

# Training hyper-parameters
MAX_STEPS = 106809
EVAL_INTERVAL = 500
CHECKPOINT_INTERVAL = 1000
LOG_INTERVAL = 10
WARMUP_STEPS = 1000
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4
MILESTONE_STEPS = {27000, 53000, 80000, 106000}
KEEP_LAST_N = 1

# --- Notifications Setup ---
NTFY_TOPIC = "AlkBQR4I6Y0"

def notify(msg):
    """Fires a notification to ntfy.sh. Fails silently so it never crashes training."""
    if not NTFY_TOPIC:
        return
    try:
        requests.post(f"https://ntfy.sh/{NTFY_TOPIC}", data=msg.encode("utf-8"), timeout=5)
    except Exception:
        pass
# ---------------------------

class TrainDataLoader():
    def __init__(self, batch_size, seq_len, data_path="data/processed_v2/train.bin"):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.offset = 0
        self.stream = torch.cuda.Stream()
        self._prefetch()
    
    def _prefetch(self):
        total_tokens = self.batch_size * (self.seq_len + 1)
        if self.offset + total_tokens > len(self.data):
            remainder = (self.offset + total_tokens) - len(self.data)
            tokens = np.concatenate((self.data[self.offset:], self.data[:remainder]))
        else:
            tokens = self.data[self.offset:self.offset + total_tokens]
        tokens = tokens.astype(np.int64).reshape(self.batch_size, self.seq_len + 1)
        x = torch.from_numpy(tokens[:, :-1]).pin_memory()
        y = torch.from_numpy(tokens[:, 1:]).pin_memory()
        with torch.cuda.stream(self.stream):
            self.next_x = x.cuda(non_blocking=True)
            self.next_y = y.cuda(non_blocking=True)
        self.offset = (self.offset + total_tokens) % len(self.data)

    def next_batch(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        x, y = self.next_x, self.next_y
        self._prefetch()
        return x, y

class EvalDataLoader():
    def __init__(self, batch_size, seq_len, device, val_target_tokens, data_path="data/processed_v2/val.bin"):

        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.val_target_tokens = val_target_tokens

        self.total_tokens = len(self.data)
        self.tokens_per_batch = batch_size * (seq_len + 1)
        self.num_batches = self.val_target_tokens // self.tokens_per_batch
        assert self.num_batches >= 2, "EvalDataLoader requires num_batches >= 2"
        
        self.usable_end = self.total_tokens - self.tokens_per_batch
        self.stride = self.usable_end // (self.num_batches - 1)
        self.offsets = [i * self.stride for i in range(self.num_batches)]
    
    def _read_batch(self, offset):
        tokens = self.data[offset:offset + self.tokens_per_batch]
        tokens = tokens.astype(np.int64).reshape(self.batch_size, self.seq_len + 1)
        x = torch.from_numpy(tokens[:, :-1]).to(self.device)
        y = torch.from_numpy(tokens[:, 1:]).to(self.device)
        return x, y
    
    def __iter__(self):
        for offset in self.offsets:
            yield self._read_batch(offset)

def train_step(model, optimizer, loss_fn, x, y, dtype, device, is_last_accum):
    with torch.autocast(device_type=device, dtype=dtype):
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
    loss = loss / GRAD_ACCUM_STEPS
    loss.backward()
    if is_last_accum:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return loss.item() * GRAD_ACCUM_STEPS

def run_evaluation(model, eval_loader, dtype, device, loss_fn):
    torch.compiler.cudagraph_mark_step_begin()
    model.eval()
    losses = []
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        for x, y in eval_loader:
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss.item())
    model.train()
    mean_loss = sum(losses) / len(losses)
    return mean_loss, math.exp(mean_loss)

def get_lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    return 0.5 * (1 + math.cos(math.pi * progress))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()
    MAX_STEPS = args.max_steps

    os.makedirs("checkpoints", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    config = ModelConfig()
    model = ShakGPT(config).to(device)
    model = torch.compile(model, mode="reduce-overhead")
    loss_fn = LigerCrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1, betas=(0.9, 0.95), fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)

    train_loader = TrainDataLoader(batch_size=BATCH_SIZE, seq_len=config.max_seq_len, data_path="data/processed_v2/train.bin")
    val_loader = EvalDataLoader(batch_size=BATCH_SIZE, seq_len=config.max_seq_len, device=device, val_target_tokens=1_000_000, data_path="data/processed_v2/val.bin")
    print("Starting training...")   

    start_step = 0
    if os.path.exists("checkpoints/latest.pt"):
        checkpoint = torch.load("checkpoints/latest.pt", map_location=device)
        target_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        target_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = checkpoint["step"] + 1
        print(f"Resuming from step {start_step}")

    notify(f"Training started: MAX_STEPS={MAX_STEPS}, resuming from step {start_step}")

    step = start_step - 1 # ensures `step` is defined if loop never executes
    model.train()
    last_log_time = time.time()
    
    try:
        for step in range(start_step, MAX_STEPS):
            torch.compiler.cudagraph_mark_step_begin()      
            for accum_step in range(GRAD_ACCUM_STEPS):
                x, y = train_loader.next_batch()
                is_last_accum = (accum_step == GRAD_ACCUM_STEPS - 1)
                loss = train_step(model, optimizer, loss_fn, x, y, dtype, device, is_last_accum)

            if step > 0 and step % LOG_INTERVAL == 0:
                elapsed_time = time.time() - last_log_time
                tokens_per_sec = (LOG_INTERVAL * GRAD_ACCUM_STEPS * train_loader.batch_size * train_loader.seq_len) / elapsed_time
                last_log_time = time.time()
                print(f"Step {step}: loss = {loss:.4f} | {tokens_per_sec:.0f} tok/s")
            
            if step > 0 and step % EVAL_INTERVAL == 0:
                t0 = time.time()
                val_loss, val_perplexity = run_evaluation(model, val_loader, dtype, device, loss_fn)
                eval_elapsed = time.time() - t0
                print(f"Step {step}: val_loss = {val_loss:.4f}, val_perplexity = {val_perplexity:.2f} | eval took {eval_elapsed:.1f}s")
            
            if step > 0 and step % CHECKPOINT_INTERVAL == 0:
                t0 = time.time()
                full_ckpt = {
                    "step": step,
                    "model_state_dict": model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": loss,
                }
                torch.save(full_ckpt, "checkpoints/latest.pt")
                ckpt_elapsed = time.time() - t0
                print(f"Step {step}: checkpoint saved in {ckpt_elapsed:.1f}s") 
                
                if step in MILESTONE_STEPS:
                    torch.save(
                        {
                            "step": step,
                            "model_state_dict": model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
                        },
                        f"checkpoints/milestone_{step}.pt"
                    )
                else:
                    torch.save(full_ckpt, f"checkpoints/step_{step}.pt")
                    old_step = step - KEEP_LAST_N * CHECKPOINT_INTERVAL
                    old_path = f"checkpoints/step_{old_step}.pt"
                    if os.path.exists(old_path):
                        os.remove(old_path)
            
            scheduler.step()

        if step >= start_step:
            full_ckpt = {
                "step": step,
                "model_state_dict": model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": loss,
            }
            torch.save(full_ckpt, "checkpoints/latest.pt")
            
            # Notify Done with loss
            notify(f"Training complete at step {step}, final loss = {loss:.4f}")
        else:
            # Notify Done gracefully if loop skipped
            notify(f"Training already complete at step {step} (no steps run).")

    except Exception as e:
        notify(f"Training crashed at step {step}: {type(e).__name__}: {e}")
        raise