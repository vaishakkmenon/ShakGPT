from dataclasses import dataclass

@dataclass
class ModelConfig:
    d_model: int
    n_layers: int
    vocab_size: int
    