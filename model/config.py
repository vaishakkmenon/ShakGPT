from dataclasses import dataclass

@dataclass
class ModelConfig:
    """
    Configuration for the model.
    
    Attributes:
        d_model: Dimension of the model
        n_layers: Number of layers
        vocab_size: Size of the vocabulary
        n_heads: Number of attention heads
        n_kv_heads: Number of key/value heads
        ffn_hidden: Hidden dimension of the feed-forward network
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        rope_theta: RoPE theta
    """
    
    d_model: int = 1024
    n_layers: int = 24
    vocab_size: int = 32768
    n_heads: int = 16
    n_kv_heads: int = 8
    ffn_hidden: int = 2752
    max_seq_len: int = 2048
    dropout: float = 0.1
    rope_theta: float = 10000.0

    def __post_init__(self):
        """
        Post-initialization checks for the model configuration.
        """
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

    @property
    def head_dim(self):
        """
        Get the dimension of the attention heads.
        """
        return self.d_model // self.n_heads