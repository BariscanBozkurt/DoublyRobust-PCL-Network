import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch import nn
from typing import Callable, Optional, List, Tuple
from tqdm import tqdm
import numpy as np

def evaluate_loss_on_dataloader(
    net: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable,
    device: str
) -> float:
    """Computes the average loss over a given DataLoader in evaluation mode."""
    net.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ = batch[0].to(device)
            output_ = batch[1].to(device)

            predicted = net(input_)
            loss = loss_fn(predicted, output_)
            total_loss += loss.item()
            num_batches += 1

    net.train() # Restore training state
    return total_loss / num_batches if num_batches > 0 else 0.0