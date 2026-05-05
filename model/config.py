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
    ffn_hidden: int = 2752 # 8/3 × d_model ≈ 2730, rounded up to nearest multiple of 64 for GPU alignment (Shazeer 2020)
    max_seq_len: int = 2048
    dropout: float = 0.0
    rope_theta: float = 10000.0

    def __post_init__(self):
        """
        Post-initialization checks for the model configuration.
        """
        assert self.d_model > 0, "d_model must be greater than 0"
        assert self.n_layers > 0, "n_layers must be greater than 0"
        assert self.vocab_size > 0, "vocab_size must be greater than 0"
        assert self.n_heads > 0, "n_heads must be greater than 0"
        assert self.n_kv_heads > 0, "n_kv_heads must be greater than 0"
        assert self.ffn_hidden > 0, "ffn_hidden must be greater than 0"
        assert self.max_seq_len > 0, "max_seq_len must be greater than 0"
        assert 0.0 <= self.dropout <= 1.0, "dropout must be between 0.0 and 1.0"
        assert self.rope_theta > 0, "rope_theta must be greater than 0"
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.ffn_hidden % 64 == 0, "ffn_hidden must be a multiple of 64"

    @property
    def head_dim(self):
        """
        Get the dimension of the attention heads.
        """
        return self.d_model // self.n_heads