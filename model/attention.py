import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig
from model.rope import RoPE

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        """
        Initialize the GroupedQueryAttention layer.

        Args:
            config: Model configuration
        """

        super().__init__()

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        self.o_proj.is_residual_projection = True
        self.dropout = nn.Dropout(config.dropout)

        self.rope = RoPE(config)

        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.n_kv_heads = config.n_kv_heads
        self.n_kv_groups_per_head = self.n_heads // self.n_kv_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for GroupedQueryAttention.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape

        # Step 1 Projecting embeddings to Q, K, V matrices
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Step 2 Reshaping the matrices to (batch_size, seq_len, n_heads, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1,2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1,2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1,2)

        # Step 3 Repeat KV heads to match Q heads
        k = torch.repeat_interleave(k, self.n_kv_groups_per_head, dim=1)
        v = torch.repeat_interleave(v, self.n_kv_groups_per_head, dim=1)

        # Step 4 RoPE Integration
        q, k = self.rope(q, k, seq_len)
        
        # Step 5 Scaled Dot-Product Attention with Dropout
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        output = self.dropout(output)

        # Step 6 Reshaping the matrices to (batch_size, seq_len, d_model)
        output = output.transpose(1,2).contiguous().view(batch_size, seq_len, self.n_heads * self.head_dim)
        
        # Step 7 Output Projection
        output = self.o_proj(output)

        return output

        