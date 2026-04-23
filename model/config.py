from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Dimensions required for the model
    d_model: int = 1024
    # Number of layers
    n_layers: int = 24
    # Vocabulary size
    vocab_size: int = 32768
    # Number of attention heads
    n_heads: int = 16
    # Number of key/value heads
    n_kv_heads: int = 8
    # Hidden dimension of the feed-forward network
    ffn_hidden: int = 2752
    # Maximum sequence length
    max_seq_len: int = 2048
    # Dropout probability
    dropout: float = 0.1
    # RoPE theta
    rope_theta: float = 10000.0

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

    @property
    def head_dim(self):
        return self.d_model // self.n_heads