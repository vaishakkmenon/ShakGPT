import numpy as np
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer/shakgpt_tokenizer.json")
data = np.memmap("data/processed_v2/train.bin", dtype=np.uint16, mode='r')
print(f"Total tokens in train.bin: {len(data):,}\n")

# Check 1: Source variety at random offsets
offsets = [100, len(data)//5, 2*len(data)//5, 3*len(data)//5, 4*len(data)//5, len(data)-300]
for offset in offsets:
    chunk = data[offset:offset+200].tolist()
    text = tokenizer.decode(chunk)
    print(f"=== offset {offset:,} ===")
    print(text[:400])
    print()

# Check 2: BOS frequency — should appear regularly throughout
bos_positions = [i for i in range(len(data)) if data[i] == 1]
print(f"\nTotal BOS tokens: {len(bos_positions):,}")
print(f"Average gap between BOS tokens: {len(data) / len(bos_positions):.0f}")
print(f"First 10 BOS positions: {bos_positions[:10]}")
print(f"Max gap between consecutive BOS: {max(b - a for a, b in zip(bos_positions[:-1], bos_positions[1:]))}")