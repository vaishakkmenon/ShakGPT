import torch
import torch.nn as nn

from model.config import ModelConfig

class RoPE(nn.Module):
    def __init__(self, config: ModelConfig):
        """
        Initialize the RoPE layer.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.theta = config.rope_theta
        self.max_seq_len = config.max_seq_len

        # Pre-compute the RoPE embeddings
        cos, sin = self._precompute_freqs(config.max_seq_len)
        self.register_buffer("freqs_cos", cos)
        self.register_buffer("freqs_sin", sin)

    def _precompute_freqs(self, max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pre-compute the RoPE embeddings.

        Args:
            max_seq_len: Maximum sequence length

        Returns:
            RoPE embeddings of shape [max_seq_len, head_dim // 2]
        """
        # Compute frequencies: [0, 2, 4, ..., head_dim - 2]
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2) / self.head_dim))
        # Compute positions: [0, 1, 2, ..., max_seq_len - 1]
        positions = torch.arange(max_seq_len)
        
        # Outer product to get (max_seq_len, head_dim // 2)
        freqs_cis = torch.outer(positions, freqs)
        freqs_cis = torch.cat([freqs_cis, freqs_cis], dim=-1)
        return freqs_cis.cos(), freqs_cis.sin()
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate the input tensor by 90 degrees.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_heads, head_dim]

        Returns:
            Rotated tensor of shape [batch_size, seq_len, n_heads, head_dim]
        """
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE rotation to the input.

        Args:
            q: Query tensor of shape [batch_size, seq_len, n_heads, head_dim]
            k: Key tensor of shape [batch_size, seq_len, n_heads, head_dim]
            seq_len: Sequence length
            orig_dtype: Original dtype of the input tensors

        Returns:
            Tuple of RoPE-rotated query and key tensors of shape [batch_size, seq_len, n_heads, head_dim]
        """
        cos = self.freqs_cos[:seq_len].view(1, 1, seq_len, self.head_dim)
        sin = self.freqs_sin[:seq_len].view(1, 1, seq_len, self.head_dim)
        q_rotated = self.rotate_half(q)
        k_rotated = self.rotate_half(k)
        q = (q * cos) + (q_rotated * sin)
        k = (k * cos) + (k_rotated * sin)
        return q, k