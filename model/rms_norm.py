import torch
import torch.nn as nn
from model.config import ModelConfig

class RMSNorm(nn.Module):
    """
    RMSNorm layer for the ShakGPT model.
    
    Attributes:
        weights: Learnable weights for the layer
        eps: Epsilon value to prevent division by zero
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the RMSNorm layer.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.d_model))
        self.eps = 1e-6
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        rms = x_fp32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x_fp32 * rms).to(orig_dtype) * self.weight