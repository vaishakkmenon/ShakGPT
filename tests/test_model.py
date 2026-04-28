import torch

from model.config import ModelConfig
from model.model import ShakGPT

def test_model():
    """
    Test the ShakGPT model.
    """
    
    # Create a model configuration
    config = ModelConfig()
    
    # Create the model
    model = ShakGPT(config)
    
    # Print the model
    print(model)

    # Create dummy input
    dummy = torch.randint(0, config.vocab_size, (1, config.max_seq_len))
    
    # Perform forward pass
    output = model(dummy)
    
    # Print the output shape
    print(output.shape)
    print(sum(p.numel() for p in model.parameters()))

if __name__ == "__main__":
    """
    Run the test model function.
    """
    test_model()