from model.config import ModelConfig
from tokenizer.custom_bpe import ShakGPTTokenizer

if __name__ == "__main__":
    config = ModelConfig()
    tokenizer = ShakGPTTokenizer(config)

    tokenizer.train(["./tests/sample.txt"])
    tokenizer.save("./tests/tokenizer.json")

    tokenized_output = tokenizer.encode("The quick brown fox jumps over the lazy dog.")
    print("Tokenized Output:", tokenized_output)

    decoded_output = tokenizer.decode(tokenized_output)
    print("Decoded Output:", decoded_output)

    if decoded_output.strip() != "The quick brown fox jumps over the lazy dog.":
        print("Error: Decoded output does not match original output")
    else:
        print("Success: Decoded output matches original output")

    tokenizer.load("./tests/tokenizer.json")

    tokenized_output_reloaded = tokenizer.encode("The quick brown fox jumps over the lazy dog.")
    print("Tokenized Output Reloaded:", tokenized_output_reloaded)
    decoded_output_reloaded = tokenizer.decode(tokenized_output_reloaded)
    print("Decoded Output Reloaded:", decoded_output_reloaded)

    assert decoded_output.strip() == "The quick brown fox jumps over the lazy dog.".strip(), "Decoded output does not match original output"
    assert decoded_output_reloaded.strip() == "The quick brown fox jumps over the lazy dog.".strip(), "Decoded output does not match original output"