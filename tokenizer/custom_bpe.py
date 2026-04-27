from model.config import ModelConfig

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder



class ShakGPTTokenizer:
    def __init__(self, config: ModelConfig, special_tokens=None):
        if special_tokens is None:
            self.special_tokens = ["[PAD]", "[BOS]", "[EOS]"]
        else:
            self.special_tokens = special_tokens
            
        self.config = config
        self.vocab_size = config.vocab_size
        
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()
        self.tokenizer.add_special_tokens(self.special_tokens)

    def train(self, files: list[str], min_frequency: int=2):
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=min_frequency
        )
        self.tokenizer.train(files, trainer=trainer)

    def encode(self, text: str):
        return self.tokenizer.encode(text).ids

    def decode(self, tokens: list[int]):
        return self.tokenizer.decode(tokens)

    def save(self, path: str):
        self.tokenizer.save(path)

    def load(self, path: str):
        self.tokenizer = Tokenizer.from_file(path)

