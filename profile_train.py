import os
import torch
import torch.nn.functional as F
import numpy as np

from model.model import ShakGPT
from model.config import ModelConfig

GRAD_ACCUM_STEPS = 2
BATCH_SIZE = 16
SEQ_LEN = 2048

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

def main():
    device = "cuda"
    dtype = torch.bfloat16
    
    model = ShakGPT(ModelConfig()).to(device)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1, betas=(0.9, 0.95), fused=True)
    train_loader = ShakGPTDataModule(batch_size=BATCH_SIZE, data_path="data/processed/train.bin")
    
    model.train()
    
    print("Warming up (compile)...")
    for warmup_step in range(50):
        for accum_step in range(GRAD_ACCUM_STEPS):
            x, y = train_loader.next_batch()
            with torch.autocast(device_type=device, dtype=dtype):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            if accum_step == GRAD_ACCUM_STEPS - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if warmup_step % 10 == 0:
            print(f"  warmup step {warmup_step}")
    
    torch.cuda.synchronize()
    print("Warmup done. Starting profile...")
    
    os.makedirs("./profile_logs", exist_ok=True)
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=2, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile_logs"),
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for step in range(7):
            for accum_step in range(GRAD_ACCUM_STEPS):
                x, y = train_loader.next_batch()
                with torch.autocast(device_type=device, dtype=dtype):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / GRAD_ACCUM_STEPS
                loss.backward()
                if accum_step == GRAD_ACCUM_STEPS - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            prof.step()
            print(f"  profile step {step}")
    
    print("Profile saved to ./profile_logs/")
    table_str = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
    print("\nTop 30 operations by CUDA time:")
    print(table_str)
    with open("profile_summary.txt", "w") as f:
        f.write(table_str)
    print("\nSaved full table to profile_summary.txt")

if __name__ == "__main__":
    main()