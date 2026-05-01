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

# FINEWEB_TARGET = 4_200_000_000      # 60% of 7B tokens
# WIKIPEDIA_TARGET = 1_400_000_000    # 20% of 7B tokens
# GUTENBERG_TARGET = 1_050_000_000    # 15% of 7B tokens
# STARCODER_TARGET = 350_000_000      # 5% of 7B tokens

FINEWEB_TARGET = 10_000_000      # 60% of 7B tokens
WIKIPEDIA_TARGET = 10_000_000    # 20% of 7B tokens
GUTENBERG_TARGET = 10_000_000    # 15% of 7B tokens
STARCODER_TARGET = 10_000_000      # 5% of 7B tokens

DATASETS = [
    {
        "name": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-100BT",
        "text_field": "text",
        "data_dir": None,
        "token_quota": FINEWEB_TARGET,
        "consumed_tokens": 0
    },
    {
        "name": "wikimedia/wikipedia",
        "subset": "20231101.en",
        "text_field": "text",
        "data_dir": None,
        "token_quota": WIKIPEDIA_TARGET,
        "consumed_tokens": 0
    },
    {
        "name": "sedthh/gutenberg_english",
        "subset": None,
        "text_field": "TEXT",
        "data_dir": None,
        "token_quota": GUTENBERG_TARGET,
        "consumed_tokens": 0
    },
    {
        "name": "bigcode/starcoderdata",
        "subset": None,
        "text_field": "content",
        "data_dir": "python",
        "token_quota": STARCODER_TARGET,
        "consumed_tokens": 0
    }
]

def process_source(dataset_dict, train_file, val_file, tokenizer):
    """
    Pull a dataset, encode the documents until the specified threshold is reached.
    Save the encoded tokens to a binary train/val file.
    """
    text_field = dataset_dict["text_field"]
    if dataset_dict["data_dir"] is None:
        ds = load_dataset(dataset_dict["name"], dataset_dict["subset"], split="train", streaming=True)
    else:
        ds = load_dataset(dataset_dict["name"], dataset_dict["subset"], data_dir=dataset_dict["data_dir"], split="train", streaming=True)
    ds = ds.select_columns([text_field])
    
    token_count = 0

    for document in ds:
        text = document[text_field]
        encoded_text = tokenizer.encode(text).ids
        encoded_text = [1] + encoded_text + [2]
        arr = np.array(encoded_text, dtype=np.uint16)
        full_count = token_count + len(arr)

        if full_count > dataset_dict["token_quota"]:
            dataset_dict["consumed_tokens"] = token_count
            break

        token_count = full_count

        if random.random() < 0.01:
            arr.tofile(val_file)
        else:
            arr.tofile(train_file)
        
        if token_count // 100_000_000 > (token_count - len(arr)) // 100_000_000:
            print(f"Dataset: {dataset_dict['name']}; {(token_count/dataset_dict['token_quota']) * 100:.1f}% tokens encoded...")
    
    dataset_dict["consumed_tokens"] = token_count
            
if __name__ == "__main__":
    random.seed(42)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer = Tokenizer.from_file("tokenizer/shakgpt_tokenizer.json")

    with open(f'{OUTPUT_DIR}/{TRAIN_FILE}', "wb") as train_bf, open(f'{OUTPUT_DIR}/{VALIDATION_FILE}', "wb") as val_bf:
        for dataset in DATASETS:
            process_source(dataset, train_bf, val_bf, tokenizer)
    
    print("Both files have been processed and closed.")

    manifest = {
        "date": datetime.now().isoformat(),
        "sources": [{"name": d["name"], "token_quota": d["token_quota"], "consumed_tokens": d["consumed_tokens"]} for d in DATASETS]    }

    with open('data/manifest.json', 'w') as manifest_bf:
        json.dump(manifest, manifest_bf, indent=2)
    
    print("Manifest has been created and populated.")