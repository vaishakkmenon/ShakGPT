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
        self.weights = nn.Parameter(torch.ones(config.d_model))
        self.eps = 1e-6
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        x_squared_mean = torch.mean(x * x, dim=-1, keepdim=True)
        rms = x_squared_mean.sqrt()
        normalized = x / (rms + self.eps)
        return normalized * self.weights