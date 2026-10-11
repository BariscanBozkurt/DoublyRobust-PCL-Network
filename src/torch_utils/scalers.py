import torch
import torch.nn as nn

class TorchStandardScaler(nn.Module):
    """
    A PyTorch equivalent of scikit-learn's StandardScaler.
    Standardizes features by removing the mean and scaling to unit variance.

    Parameters
    ----------
    with_mean : bool, default=True
        If True, center the data before scaling.
    with_std : bool, default=True
        If True, scale the data to unit variance.
    eps : float, default=1e-6
        Small constant added to the standard deviation for numerical stability.

    Attributes
    ----------
    mean_ : torch.Tensor
        Per-feature mean calculated from the training set.
    scale_ : torch.Tensor
        Per-feature standard deviation calculated from the training set.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True, eps: float = 1e-10):
        super().__init__()
        self.with_mean = with_mean
        self.with_std = with_std
        self.eps = eps
        self.register_buffer("mean_", None)
        self.register_buffer("scale_", None)
        self.is_fitted = False

    def fit(self, X: torch.Tensor):
        """
        Compute the mean and standard deviation to be used for later scaling.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : TorchStandardScaler
            Fitted scaler.
        """
        if self.with_mean:
            mean = X.mean(dim=0, keepdim=True)
        else:
            mean = torch.zeros(1, X.size(1), device=X.device)

        if self.with_std:
            scale = X.std(dim=0, unbiased=False, keepdim=True) + self.eps
        else:
            scale = torch.ones(1, X.size(1), device=X.device)

        self.mean_ = mean
        self.scale_ = scale
        self.is_fitted = True
        return self

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        X_tr : torch.Tensor of shape (n_samples, n_features)
            Transformed data.
        """
        return self.fit(X).transform(X)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Perform standardization by centering and scaling.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_tr : torch.Tensor
            Transformed data.
        """
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Undo the scaling of X according to the fitted mean and std.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data to inverse transform.

        Returns
        -------
        X_orig : torch.Tensor
            Original data in input space.
        """
        return X * self.scale_ + self.mean_


class TorchLogStandardScaler:
    """
    Applies Log transformation followed by Standard Scaling.
    y = (log(x + eps) - mean) / std

    Useful for heavy-tailed positive distributions like Density Ratios.
    """
    def __init__(self, eps: float = 1e-6, device="cpu"):
        self.mean_ = None
        self.scale_ = None
        self.eps = eps
        self.device = device

    def fit(self, x: torch.Tensor):
        # 1. Log Transform
        log_x = torch.log(x + self.eps)

        # 2. Compute Statistics in Log Space
        self.mean_ = log_x.mean(dim=0, keepdim=True).to(self.device)
        self.scale_ = log_x.std(dim=0, keepdim=True).to(self.device)

        # Handle constant values (std=0) to avoid division by zero
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, x: torch.Tensor):
        if self.mean_ is None:
            raise RuntimeError("Scaler must be fitted before transform.")

        # 1. Log Transform
        log_x = torch.log(x + self.eps)

        # 2. Standardize
        return (log_x - self.mean_) / self.scale_

    def fit_transform(self, x: torch.Tensor):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x_scaled: torch.Tensor):
        # 1. Inverse Standardize: y = x_scaled * std + mean
        log_x = x_scaled * self.scale_ + self.mean_

        # 2. Inverse Log (Exp): x = exp(y) - eps
        return torch.exp(log_x) - self.eps

    def to(self, device):
        self.device = device
        if self.mean_ is not None:
            self.mean_ = self.mean_.to(device)
            self.scale_ = self.scale_.to(device)
        return self


class TorchRobustNormalizer:
    """
    Robust Normalizer using Quantiles (Median and IQR).
    Scales data using statistics that are robust to outliers.

    Formula:
        x_scaled = (x - median) / (q75 - q25)

    This is akin to method='robust' in pytorch_forecasting.
    """
    def __init__(
        self,
        center: bool = True,
        quantile_range: tuple = (0.25, 0.75),
        eps: float = 1e-6,
        device: str = "cuda"
    ):
        self.center = center
        self.q_min, self.q_max = quantile_range
        self.eps = eps
        self.device = device

        # Buffers for statistics
        self.center_ = None  # Median
        self.scale_ = None   # IQR
        self.is_fitted = False

    def fit(self, x: torch.Tensor):
        """
        Computes the median and IQR of the input tensor x.
        """
        # Ensure x is on the correct device
        if x.device != torch.device(self.device):
            x = x.to(self.device)

        # 1. Compute Center (Median)
        if self.center:
            # torch.quantile requires float dtype
            self.center_ = torch.quantile(x.float(), 0.5, dim=0, keepdim=True)
        else:
            self.center_ = torch.zeros(1, x.shape[1], device=self.device)

        # 2. Compute Scale (IQR: q75 - q25)
        q_low = torch.quantile(x.float(), self.q_min, dim=0, keepdim=True)
        q_high = torch.quantile(x.float(), self.q_max, dim=0, keepdim=True)

        iqr = q_high - q_low

        # 3. Handle Constant Values (IQR = 0)
        # If IQR is 0, we set scale to 1.0 to avoid division by zero
        iqr[iqr == 0.0] = 1.0

        self.scale_ = iqr
        self.is_fitted = True
        return self

    def transform(self, x: torch.Tensor):
        """
        Applies robust scaling.
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform.")

        if x.device != torch.device(self.device):
            x = x.to(self.device)

        # Apply transformation: (x - median) / IQR
        return (x - self.center_) / (self.scale_ + self.eps)

    def fit_transform(self, x: torch.Tensor):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x_scaled: torch.Tensor):
        """
        Reverts the scaling.
        x = x_scaled * IQR + median
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform.")

        if x_scaled.device != torch.device(self.device):
            x_scaled = x_scaled.to(self.device)

        return x_scaled * (self.scale_ + self.eps) + self.center_

    def to(self, device):
        """Moves statistics to a new device."""
        self.device = device
        if self.center_ is not None:
            self.center_ = self.center_.to(device)
            self.scale_ = self.scale_.to(device)
        return self


class TorchIdentityTransformer(nn.Module):
    """
    Identity transformer for PyTorch.
    Does not change the input data.

    Useful when you want to keep the same API as TorchStandardScaler
    but avoid applying any transformation.
    """

    def __init__(self):
        super().__init__()
        self.is_fitted = True

    def fit(self, X: torch.Tensor):
        """
        Does nothing. Included for API compatibility.
        """
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns the input unchanged.
        """
        return X

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns the input unchanged (fit does nothing).
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns the input unchanged.
        """
        return X
