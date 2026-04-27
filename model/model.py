import torch
import torch.nn as nn
from model.config import ModelConfig

class ShakGPTEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    
    def forward(self, x: torch.Tensor):
        x = self.token_embedding(x)
        x = self.dropout(x)
        return x

