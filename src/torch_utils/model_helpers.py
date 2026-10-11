import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Optional

def kaiming_initializer(module: nn.Module):
    """
    Recursively applies Kaiming Uniform initialization to Linear and Convolutional layers.

    This is the standard initialization for networks using ReLU or GELU (approximating ReLU).
    """
    for m in module.modules():
        # Handle Convolutional Layers (Conv1d, Conv2d, Conv3d)
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            # Kaiming (He) initialization for convolutional layers
            # mode='fan_in' is standard for preserving variance forward
            # nonlinearity='relu' is used even if the activation is GELU/LeakyReLU
            init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

            if m.bias is not None:
                init.constant_(m.bias, 0)

        # Handle Linear (Dense) Layers
        elif isinstance(m, nn.Linear):
            # Kaiming (He) initialization for linear layers
            init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

            if m.bias is not None:
                init.constant_(m.bias, 0)

        # Skip initialization for BatchNorm layers (they have their own standard setup)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            pass

def get_last_linear_out_features(model: nn.Module) -> Optional[int]:
    """
    Get the output feature dimension of the last nn.Linear layer in a PyTorch model.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to inspect.

    Returns
    -------
    out_features : int or None
        The number of output features of the last nn.Linear layer.
        Returns None if no nn.Linear layer is found.
    """
    for layer in reversed(list(model.modules())):
        if isinstance(layer, nn.Linear):
            return layer.out_features
    return None

def to_device_collate(batch, device):
    A, Y, U = zip(*batch)
    return (
        torch.stack(A).to(device),
        torch.stack(Y).to(device),
        torch.stack(U).to(device)
    )