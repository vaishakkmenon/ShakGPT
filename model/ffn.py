import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig

class FeedForward(nn.Module):
    """
    FeedForward layer for the ShakGPT model.

    Attributes:
        gate_proj: First linear layer
        up_proj: Second linear layer
        down_proj: Third linear layer
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the FeedForward layer.

        Args:
            config: Model configuration
        """
        super().__init__()

        self.gate_proj = nn.Linear(config.d_model, config.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden, config.d_model, bias=False)
        self.down_proj.is_residual_projection = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for FeedForward.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))