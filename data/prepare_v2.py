from collections import deque
import os
import numpy as np
import random
import json
from datetime import datetime

from datasets import load_dataset
from tokenizers import Tokenizer

MODEL_NAME = "shakgpt"
OUTPUT_DIR = "data/processed_v2"
TRAIN_FILE = "train.bin"
VALIDATION_FILE = "val.bin"

# Training Token Targets
FINEWEB_TARGET = 4_200_000_000      # 60% of 7B tokens
WIKIPEDIA_TARGET = 1_050_000_000    # 15% of 7B tokens
GUTENBERG_TARGET = 1_050_000_000    # 15% of 7B tokens
STARCODER_TARGET = 700_000_000      # 10% of 7B tokens

# Test Run Token Targets
# FINEWEB_TARGET = 10_000_000
# WIKIPEDIA_TARGET = 10_000_000
# GUTENBERG_TARGET = 10_000_000
# STARCODER_TARGET = 10_000_000

BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
DOCS_PER_REFILL = 64
TOKENS_PER_SLICE = 8192

DATASETS = [
    {
        "name": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-100BT",
        "text_field": "text",
        "data_dir": None,
        "token_quota": FINEWEB_TARGET,
    },
    {
        "name": "wikimedia/wikipedia",
        "subset": "20231101.en",
        "text_field": "text",
        "data_dir": None,
        "token_quota": WIKIPEDIA_TARGET,
    },
    {
        "name": "sedthh/gutenberg_english",
        "subset": None,
        "text_field": "TEXT",
        "data_dir": None,
        "token_quota": GUTENBERG_TARGET,
    },
    {
        "name": "bigcode/starcoderdata",
        "subset": None,
        "text_field": "content",
        "data_dir": "python",
        "token_quota": STARCODER_TARGET,
    }
]


def refill_buffer(source_idx, streams, chunk_buffers, tokenizer):
    """Pull DOCS_PER_REFILL documents from source, batch-tokenize, fill buffer."""
    docs = []
    while len(docs) < DOCS_PER_REFILL:
        try:
            doc = next(streams[source_idx])
            docs.append(doc[DATASETS[source_idx]["text_field"]])
        except StopIteration:
            DATASETS[source_idx]["finished"] = True
            break
    
    if not docs:
        return
    
    encoded_list = tokenizer.encode_batch(docs)
    for encoded_obj in encoded_list:
        chunk = [BOS_TOKEN_ID] + encoded_obj.ids + [EOS_TOKEN_ID]
        chunk_buffers[source_idx].extend(chunk)


if __name__ == "__main__":
    random.seed(42)

    quotas = [d["token_quota"] for d in DATASETS]
    consumed = [0, 0, 0, 0]
    weights = [0.6, 0.2, 0.15, 0.05]
    chunk_buffers = [deque() for _ in range(len(DATASETS))]

    streams = []
    for d in DATASETS:
        if d["data_dir"] is None:
            ds = load_dataset(d["name"], d["subset"], split="train", streaming=True)
        else:
            ds = load_dataset(d["name"], d["subset"], data_dir=d["data_dir"], split="train", streaming=True)
        ds = ds.select_columns([d["text_field"]]).shuffle(seed=42, buffer_size=10000)
        streams.append(iter(ds))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer = Tokenizer.from_file("tokenizer/shakgpt_tokenizer.json")

    with open(f'{OUTPUT_DIR}/{TRAIN_FILE}', "wb") as train_bf, open(f'{OUTPUT_DIR}/{VALIDATION_FILE}', "wb") as val_bf:
        while True:
            active_indices = [i for i in range(len(DATASETS)) 
                              if consumed[i] < quotas[i] and not DATASETS[i].get("finished")]
            if not active_indices:
                break
            
            active_weights = [weights[i] for i in active_indices]
            chosen = random.choices(active_indices, weights=active_weights, k=1)[0]
            
            # Refill buffer if needed
            if len(chunk_buffers[chosen]) < TOKENS_PER_SLICE:
                refill_buffer(chosen, streams, chunk_buffers, tokenizer)
            
            # If buffer still empty after refill attempt, source is exhausted
            if len(chunk_buffers[chosen]) < TOKENS_PER_SLICE: 
                DATASETS[chosen]["finished"] = True
                continue
            
            # Skip if would exceed quota
            if consumed[chosen] + TOKENS_PER_SLICE > quotas[chosen]:
                DATASETS[chosen]["finished"] = True
                continue

            # Pop one chunk
            chunk = [chunk_buffers[chosen].popleft() for _ in range(TOKENS_PER_SLICE)]
            assert len(chunk) == TOKENS_PER_SLICE
            
            arr = np.array(chunk, dtype=np.uint16)
            if consumed[chosen] >= 0.99 * quotas[chosen]: 
                arr.tofile(val_bf)
            else: 
                arr.tofile(train_bf)
            
            consumed[chosen] += TOKENS_PER_SLICE
            
            total_consumed = sum(consumed)
            if total_consumed // 100_000_000 > (total_consumed - TOKENS_PER_SLICE) // 100_000_000:
                pct = total_consumed / sum(quotas) * 100
                print(f"Progress: {pct:.1f}% — consumed: {consumed}")
    
    print("Both files have been processed and closed.")

    manifest = {
        "date": datetime.now().isoformat(),
        "sources": [
            {"name": DATASETS[i]["name"], "token_quota": quotas[i], "consumed_tokens": consumed[i]}
            for i in range(len(DATASETS))
        ]
    }

    with open('data/manifest_v2.json', 'w') as manifest_bf:
        json.dump(manifest, manifest_bf, indent=2)
    
    print("Manifest has been created and populated.")