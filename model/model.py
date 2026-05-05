import math
import torch
import torch.nn as nn

from model.config import ModelConfig
from model.block import Block
from model.rms_norm import RMSNorm

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

class ShakGPT(nn.Module):
    """
    ShakGPT model for text generation.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the ShakGPT model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.embedding = ShakGPTEmbedding(config)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # self.lm_head.weight = self.embedding.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'is_residual_projection'):
                std = 0.02 / math.sqrt(2 * self.config.n_layers)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)