import numpy as np
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer/shakgpt_tokenizer.json")
data = np.memmap("data/processed/train.bin", dtype=np.uint16, mode='r')
print(f"Total tokens in train.bin: {len(data):,}")

for offset in [100, len(data)//4, len(data)//2, 3*len(data)//4, len(data)-300]:
    chunk = data[offset:offset+200].tolist()
    text = tokenizer.decode(chunk)
    print(f"\n=== offset {offset:,} ===")
    print(text[:500])