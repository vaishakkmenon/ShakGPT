import os
import numpy as np
import random
import json
from datetime import datetime

from datasets import load_dataset
from tokenizers import Tokenizer

MODEL_NAME = "shakgpt"
OUTPUT_DIR = "data/processed"
TRAIN_FILE = "train.bin"
VALIDATION_FILE = "val.bin"

# Training Token Targets
FINEWEB_TARGET = 4_200_000_000      # 60% of 7B tokens
WIKIPEDIA_TARGET = 1_400_000_000    # 20% of 7B tokens
GUTENBERG_TARGET = 1_050_000_000    # 15% of 7B tokens
STARCODER_TARGET = 350_000_000      # 5% of 7B tokens

# Test Run Token Targets
# FINEWEB_TARGET = 10_000_000
# WIKIPEDIA_TARGET = 10_000_000
# GUTENBERG_TARGET = 10_000_000
# STARCODER_TARGET = 10_000_000

BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
DOCS_PER_REFILL = 64
MAX_DOC_TOKENS = 8192

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
    """Pull DOCS_PER_REFILL documents from source, batch-tokenize, chunk into buffer."""
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
        raw_tokens = encoded_obj.ids
        # Chunk each document into pieces of (MAX_DOC_TOKENS - 2) raw tokens,
        # then wrap each chunk with BOS and EOS to bring it up to MAX_DOC_TOKENS or less.
        for i in range(0, len(raw_tokens), MAX_DOC_TOKENS - 2):
            raw_chunk = raw_tokens[i:i + MAX_DOC_TOKENS - 2]
            chunk = [BOS_TOKEN_ID] + raw_chunk + [EOS_TOKEN_ID]
            chunk_buffers[source_idx].append(chunk)


if __name__ == "__main__":
    random.seed(42)

    quotas = [d["token_quota"] for d in DATASETS]
    consumed = [0, 0, 0, 0]
    weights = [0.6, 0.2, 0.15, 0.05]
    chunk_buffers = [[], [], [], []]

    streams = []
    for d in DATASETS:
        if d["data_dir"] is None:
            ds = load_dataset(d["name"], d["subset"], split="train", streaming=True)
        else:
            ds = load_dataset(d["name"], d["subset"], data_dir=d["data_dir"], split="train", streaming=True)
        ds = ds.select_columns([d["text_field"]]).shuffle(seed=42, buffer_size=1000)
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
            
            # Refill buffer if empty
            if not chunk_buffers[chosen]:
                refill_buffer(chosen, streams, chunk_buffers, tokenizer)
            
            # If buffer still empty after refill attempt, source is exhausted
            if not chunk_buffers[chosen]:
                continue
            
            # Pop one chunk
            chunk = chunk_buffers[chosen].pop(0)
            
            # Skip if would exceed quota
            if consumed[chosen] + len(chunk) > quotas[chosen]:
                DATASETS[chosen]["finished"] = True
                continue
            
            arr = np.array(chunk, dtype=np.uint16)
            if random.random() < 0.01:
                arr.tofile(val_bf)
            else:
                arr.tofile(train_bf)
            
            consumed[chosen] += len(chunk)
            
            total_consumed = sum(consumed)
            if total_consumed // 100_000_000 > (total_consumed - len(chunk)) // 100_000_000:
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

    with open('data/manifest.json', 'w') as manifest_bf:
        json.dump(manifest, manifest_bf, indent=2)
    
    print("Manifest has been created and populated.")