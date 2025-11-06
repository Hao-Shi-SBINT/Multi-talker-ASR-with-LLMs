import torch
import torch.nn as nn

def freeze_model(model: nn.Module):
    """
    Freeze all parameters in a PyTorch model so that they are not updated during training.

    Args:
        model (nn.Module): The model to freeze.

    Returns:
        None
    """
    for name, param in model.named_parameters():
        param.requires_grad = False  # Disable gradient computation for this parameter
        # Optional: print each frozen parameter
        # print(f"Froze parameter: {name}")

