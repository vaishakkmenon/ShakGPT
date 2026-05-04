import os
import math
import time
import torch
import numpy as np

from model.model import ShakGPT
from model.config import ModelConfig
import torch.nn.functional as F
from liger_kernel.transformers import LigerCrossEntropyLoss

# Training hyper-parameters
MAX_STEPS = 106809
EVAL_INTERVAL = 500
CHECKPOINT_INTERVAL = 1000
LOG_INTERVAL = 10
WARMUP_STEPS = 1000
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4

class ShakGPTDataModule():
    def __init__(self, batch_size=8, seq_len=2048, data_path="data/processed/train.bin"):
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

def evaluate_step(model, x, y, dtype, device):
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        perplexity = torch.exp(loss)
    return loss.item(), perplexity.item()

def get_lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    return 0.5 * (1 + math.cos(math.pi * progress))

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    model = ShakGPT(ModelConfig()).to(device)
    model = torch.compile(model, mode="reduce-overhead")
    loss_fn = LigerCrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1, betas=(0.9, 0.95), fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)

    train_loader = ShakGPTDataModule(batch_size=BATCH_SIZE, data_path="data/processed/train.bin")
    val_loader = ShakGPTDataModule(batch_size=BATCH_SIZE, data_path="data/processed/val.bin")
    print("Starting training...")   

    start_step = 0
    if os.path.exists("checkpoints/latest.pt"):
        checkpoint = torch.load("checkpoints/latest.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = checkpoint["step"] + 1
        print(f"Resuming from step {start_step}")

    model.train()
    last_log_time = time.time()
    for step in range(start_step, MAX_STEPS):
        torch.compiler.cudagraph_mark_step_begin()      
        for accum_step in range(GRAD_ACCUM_STEPS):
            x, y = train_loader.next_batch()
            is_last_accum = (accum_step == GRAD_ACCUM_STEPS - 1)
            loss = train_step(model, optimizer, loss_fn, x, y, dtype, device, is_last_accum)

        if step % LOG_INTERVAL == 0 and step > 0:
            elapsed_time = time.time() - last_log_time
            tokens_per_sec = (LOG_INTERVAL * GRAD_ACCUM_STEPS * train_loader.batch_size * train_loader.seq_len) / elapsed_time
            last_log_time = time.time()
            print(f"Step {step}: loss = {loss:.4f} | {tokens_per_sec:.0f} tok/s")
        
        if step % EVAL_INTERVAL == 0:
            model.eval()
            x_val, y_val = val_loader.next_batch()
            val_loss, val_perplexity = evaluate_step(model, x_val, y_val, dtype, device)
            print(f"Step {step}: val_loss = {val_loss}, val_perplexity = {val_perplexity}")
            model.train()
        
        if step % CHECKPOINT_INTERVAL == 0:
            checkpoint = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": loss,
            }
            torch.save(checkpoint, f"checkpoints/step_{step}.pt")
            torch.save(checkpoint, "checkpoints/latest.pt")
        
        scheduler.step()
    