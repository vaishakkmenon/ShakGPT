import torch
import torch.nn as nn
from model.config import ModelConfig

class ShakGPTEmbedding(nn.Module):
    """
    Embedding layer for the ShakGPT model.
    
    Attributes:
        token_embedding: Embedding matrix for tokens
        dropout: Dropout layer
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the embedding layer.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ShakGPTEmbedding.

        Args:
            x: Input tensor of shape [batch_size, seq_len]

        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        x = self.token_embedding(x)
        x = self.dropout(x)
        return x

