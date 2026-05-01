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
        self.register_buffer("freqs_cis", self._precompute_freqs(config.max_seq_len))

    def _precompute_freqs(self, max_seq_len: int) -> torch.Tensor:
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
        return torch.polar(torch.ones_like(freqs_cis), freqs_cis)
    
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
        # Slice the precomputed frequencies to the current sequence length
        freqs_cis = self.freqs_cis[:seq_len]

        orig_dtype = q.dtype

        # Convert to float32 if not already
        if q.dtype != torch.float32:
            q = q.float()
        if k.dtype != torch.float32:
            k = k.float()

        # Reshape q to (batch_size, seq_len, n_heads, head_dim // 2, 2)
        q_view = q.view(q.shape[0], q.shape[1], q.shape[2], self.head_dim // 2, 2)
        # Convert to complex numbers
        q_complex = torch.view_as_complex(q_view)

        # Reshape k to (batch_size, seq_len, n_heads, head_dim // 2, 2)
        k_view = k.view(k.shape[0], k.shape[1], k.shape[2], self.head_dim // 2, 2)
        # Convert to complex numbers
        k_complex = torch.view_as_complex(k_view)

        # Broadcast frequencies to match q and k shapes
        freqs_cis_broadcast = freqs_cis.view(1, 1, seq_len, -1)
        
        # Apply RoPE rotation
        q_complex = q_complex * freqs_cis_broadcast
        k_complex = k_complex * freqs_cis_broadcast

        # View back to real numbers
        q_out = torch.view_as_real(q_complex)
        k_out = torch.view_as_real(k_complex)

        # Reshape to original shape
        q_out = q_out.contiguous().view(q.shape[0], q.shape[1], q.shape[2], self.head_dim)
        k_out = k_out.contiguous().view(k.shape[0], k.shape[1], k.shape[2], self.head_dim)

        q_out = q_out.to(orig_dtype)
        k_out = k_out.to(orig_dtype)   

        return q_out, k_out