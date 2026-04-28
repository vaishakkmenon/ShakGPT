import torch
import torch.nn as nn

from model.config import ModelConfig
from model.attention import GroupedQueryAttention
from model.ffn import FeedForward
from model.rms_norm import RMSNorm

class Block(nn.Module):
    """
    Transformer Block for the ShakGPT model.
    
    Attributes:
        attention_layer: GroupedQueryAttention layer
        feed_forward_layer: FeedForward layer
        norm1: First RMSNorm layer
        norm2: Second RMSNorm layer
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the Transformer Block.

        Args:
            config: Model configuration
        """
        super().__init__()

        # Define the self-attention layer
        self.attn = GroupedQueryAttention(config)

        # Define the feed-forward layer
        self.ffn = FeedForward(config)
        
        # Define the RMSNorm layers
        self.norm1 = RMSNorm(config)
        self.norm2 = RMSNorm(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Block.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        
        # Step 1: Apply RMSNorm and then self-attention with skip connection
        x = x + self.attn(self.norm1(x))

        # Step 2: Apply RMSNorm and then feed-forward with skip connection
        x = x + self.ffn(self.norm2(x))
        
        return x