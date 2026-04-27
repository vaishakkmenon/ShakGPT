from model.config import ModelConfig

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder



class ShakGPTTokenizer:
    """
    Tokenizer for the ShakGPT model.
    
    Attributes:
        tokenizer: Tokenizer
        vocab_size: Size of the vocabulary
        special_tokens: List of special tokens
    """
    
    def __init__(self, config: ModelConfig, special_tokens=None):
        """
        Initialize the tokenizer.

        Args:
            config: Model configuration
            special_tokens: List of special tokens to add to the tokenizer
        """
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
        """
        Train the tokenizer on the given files.

        Args:
            files: List of file paths to train the tokenizer on
            min_frequency: Minimum frequency of tokens to be included in the vocabulary
        """
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=min_frequency
        )
        self.tokenizer.train(files, trainer=trainer)

    def encode(self, text: str) -> list[int]:
        """
        Encode the given text into tokens.

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text).ids

    def decode(self, tokens: list[int]) -> str:
        """
        Decode the given tokens into text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text
        """
        return self.tokenizer.decode(tokens)

    def save(self, path: str):
        """
        Save the tokenizer to the given path.

        Args:
            path: Path to save the tokenizer to
        """
        self.tokenizer.save(path)

    def load(self, path: str):
        """
        Load the tokenizer from the given path.

        Args:
            path: Path to load the tokenizer from
        """
        self.tokenizer = Tokenizer.from_file(path)

