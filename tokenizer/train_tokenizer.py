import os
from datasets import load_dataset

from model.config import ModelConfig
from tokenizer.custom_bpe import ShakGPTTokenizer

FINEWEB_TARGET = 1_500_000_000 # 60%
WIKIPEDIA_TARGET = 500_000_000 # 20%
GUTENBERG_TARGET = 375_000_000 # 15%
STARCODER_TARGET = 125_000_000 # 5%

TOTAL_TARGET = FINEWEB_TARGET + WIKIPEDIA_TARGET + GUTENBERG_TARGET + STARCODER_TARGET
MODEL_NAME = "shakgpt"
SAVE_FILE = f"tokenizer/{MODEL_NAME}_tokenizer.json"

DATASETS = [
    {
        "name": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-100BT",
        "text_field": "text",
        "target_chars": FINEWEB_TARGET,
        "output_file": "tokenizer/training_data/fineweb_edu_sample.txt",
        "data_dir": None
    },
    {
        "name": "wikimedia/wikipedia",
        "subset": "20231101.en",
        "text_field": "text",
        "target_chars": WIKIPEDIA_TARGET,
        "output_file": "tokenizer/training_data/wikipedia.txt",
        "data_dir": None
    },
    {
        "name": "sedthh/gutenberg_english",
        "subset": None,
        "text_field": "TEXT",
        "target_chars": GUTENBERG_TARGET,
        "output_file": "tokenizer/training_data/gutenberg_english.txt",
        "data_dir": None
    },
    {
        "name": "bigcode/starcoderdata",
        "subset": None,
        "text_field": "content",
        "target_chars": STARCODER_TARGET,
        "output_file": "tokenizer/training_data/starcoderdata.txt",
        "data_dir": "python"
    }
]

def stream_and_save_dataset(dataset_dict):
    """
    Stream a dataset, select a subset of the data based on target_chars
    """
    chars_written = 0
    target_chars = dataset_dict["target_chars"]
    output_file = dataset_dict["output_file"]
    text_field = dataset_dict["text_field"]

    if os.path.exists(output_file):
        print(f"Skipping {output_file}, already exists")
        return

    if dataset_dict["data_dir"] is None:
        ds = load_dataset(dataset_dict["name"], dataset_dict["subset"], split="train", streaming=True)
    else:
        ds = load_dataset(dataset_dict["name"], dataset_dict["subset"], data_dir=dataset_dict["data_dir"], split="train", streaming=True)
    ds = ds.select_columns([text_field])
    with open(output_file, "w", encoding="utf-8") as f:
        for document in ds:
            text = document[text_field]
            f.write(text + "\n")
            chars_written += len(text)
            if chars_written % 10_000_000 == 0:
                print(f"{chars_written / 1_000_000:.0f}MB written...")
            if chars_written >= target_chars:
                break
    print(f"Finished writing {output_file}")
            
if __name__ == "__main__":
    os.makedirs("tokenizer/training_data", exist_ok=True)
    for dataset in DATASETS:
        stream_and_save_dataset(dataset)
    config = ModelConfig()
    tokenizer = ShakGPTTokenizer(config)
    tokenizer.train([dataset["output_file"] for dataset in DATASETS])
    tokenizer.save(SAVE_FILE)
