import torch
from torch import nn
from typing import Dict, Any

def get_loss_function(name: str, kwargs: Dict[str, Any] = None):
    if kwargs is None: kwargs = {}

    name = name.lower().strip()

    if name == "mse":
        return nn.MSELoss(**kwargs)
    elif name == "mae" or name == "l1":
        return nn.L1Loss(**kwargs)
    elif name == "huber":
        return nn.HuberLoss(**kwargs)
    elif name == "log_cosh":
        return LogCoshLoss(**kwargs)
    elif name == "quantile":
        return QuantileLoss(**kwargs)
    elif name == "asymmetric_mse" or name == "expectile":
        return AsymmetricMSELoss(**kwargs)
    elif name == "msle":
        return MSLELoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function name: {name}")

class QuantileLoss(nn.Module):
    """
    Quantile (Pinball) Loss.
    Used for Quantile Regression to estimate specific percentiles of the target distribution.

    Parameters
    ----------
    quantile : float, default=0.9
        The target quantile to estimate (must be between 0 and 1).
        - 0.5: Estimates the Median (equivalent to MAE).
        - 0.9: Estimates the 90th percentile.
    """
    def __init__(self, quantile: float = 0.9):
        super().__init__()
        self.quantile = quantile

    def forward(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        errors = target - y_pred
        # Logic: max((q-1)*error, q*error)
        # Note: (quantile - 1) is negative, so this handles overestimation logic
        loss = torch.max((self.quantile - 1) * errors, self.quantile * errors)
        return torch.mean(loss)

class AsymmetricMSELoss(nn.Module):
    """
    Asymmetric MSE (Expectile Loss).
    Penalizes underestimations (target > pred) significantly more than overestimations.
    Result: The model learns a 'shifted mean' that stays closer to the center
    than a raw quantile model, but still respects the high-end bias.
    """
    def __init__(self, quantile: float = 0.9):
        super().__init__()
        self.quantile = quantile

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = target - pred
        # If error > 0 (Underestimation), weight by q
        # If error < 0 (Overestimation), weight by (1-q)
        loss = torch.where(error > 0,
                           self.quantile * (error ** 2),
                           (1 - self.quantile) * (error ** 2))
        return torch.mean(loss)

class LogCoshLoss(nn.Module):
    """
    Log-Cosh Loss.
    Approximates (pred - target)^2 / 2 for small errors and |pred - target| - log(2) for large errors.
    Effectively a smooth version of Huber loss that is twice differentiable.
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.log(torch.cosh(y_pred - target))
        return torch.mean(loss)

class MSLELoss(nn.Module):
    """
    Mean Squared Logarithmic Error Loss.
    Calculates the mean squared error between log(prediction + 1) and log(target + 1).
    Useful for regression targets with exponential growth or large variance.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # We add 1 to avoid log(0) errors if inputs are non-negative
        return self.mse(torch.log1p(y_pred), torch.log1p(target))







