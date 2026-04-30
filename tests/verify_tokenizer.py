from model.config import ModelConfig
from tokenizer.custom_bpe import ShakGPTTokenizer

if __name__ == "__main__":
    tokenizer = ShakGPTTokenizer(ModelConfig())
    tokenizer.load("tokenizer/shakgpt_tokenizer.json")
    print(tokenizer.vocab_size)

    text = "Hello, world!"
    tokens = tokenizer.encode(text)
    print(tokens)
    decoded = tokenizer.decode(tokens)
    print(decoded)

    print(tokenizer.tokenizer.token_to_id("[PAD]"))
    print(tokenizer.tokenizer.token_to_id("[BOS]"))
    print(tokenizer.tokenizer.token_to_id("[EOS]"))

    unicode_str = "café 日本語 🧬"
    print(tokenizer.encode(unicode_str))
    print(tokenizer.decode(tokenizer.encode(unicode_str)))

    code_str = "def hello_world():\n    print('Hello, world!')"
    print(tokenizer.encode(code_str))
    print(tokenizer.decode(tokenizer.encode(code_str)))