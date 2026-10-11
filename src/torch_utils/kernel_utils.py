import torch
import torch.nn as nn
import numpy as np
from scipy.special import loggamma
import sympy
from typing import Optional, Union, Tuple, Dict, List, Any
from abc import ABC, abstractmethod

# Import the PyTorch-compatible linalg functions
from .linear_algebra import pairwise_squared_distance, pairwise_absolute_distance
# Note:  If you want to use float64, uncomment the below line.
# torch.set_default_dtype(torch.float64)

# --- Abstract Base Classes ---

class Kernel(nn.Module, ABC):
    """
    Base class for all kernels.
    Inherits from torch.nn.Module to manage parameters and state.
    """
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self,
                 X: torch.Tensor,
                 Y: Optional[torch.Tensor] = None,
                 ) -> torch.Tensor:
        """Computes the kernel matrix K(X, Y)."""
        pass

    # Use __call__ to route to the standard PyTorch forward pass
    def __call__(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(X, Y)

    def _get_hyperparameters(self) -> Dict[str, Any]:
        """Collects non-module/non-tensor instance attributes for scikit-learn compatibility."""
        params = {}
        for name, value in self.__dict__.items():
            if name.startswith('_'): continue
            if isinstance(value, (nn.Module, torch.Tensor)): continue
            if name in ['training']: continue # Standard nn.Module attribute
            params[name] = value

        # Also include any registered parameters (e.g., length_scale)
        for name, param in self.named_parameters():
            # Convert single-element tensor to float/int for repr/params
            params[name] = param.item() if param.numel() == 1 else param.detach().cpu().numpy()

        return params

    def __repr__(self) -> str:
        """Return a string representation of the kernel."""
        param_str = ', '.join([f'{k}={v}' for k, v in self._get_hyperparameters().items()])
        return f"{self.__class__.__name__}({param_str})"

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the kernel (for scikit-learn compatibility)."""
        return self._get_hyperparameters()

    def set_params(self, **params):
        """Set the parameters of the kernel (for scikit-learn compatibility)."""
        for param_name, param_value in params.items():
            # Set regular attributes
            if hasattr(self, param_name) and not isinstance(getattr(self, param_name), nn.Parameter):
                setattr(self, param_name, param_value)
            # Set trainable parameters
            elif param_name in [n for n, p in self.named_parameters()]:
                param = getattr(self, param_name)
                with torch.no_grad():
                    if isinstance(param_value, (float, int)):
                        param.fill_(param_value)
                    elif isinstance(param_value, torch.Tensor):
                        param.copy_(param_value)
                    else:
                        raise TypeError(f"Cannot set parameter {param_name} with type {type(param_value)}")
            else:
                raise ValueError(f"Kernel {self.__class__.__name__} has no parameter named {param_name}")


# --- Basic Kernels ---

class StackOfKernels(Kernel):
    """
    Computes the product kernel K = k_1(X[:,1]) * k_2(X[:,2]) * ...
    Assumes kernels operate columnwise on corresponding feature dimensions.
    """
    def __init__(self, list_of_kernels: List[Kernel], **kwargs):
        super().__init__()
        # Use nn.ModuleList to register kernels as submodules
        self.list_of_kernels = nn.ModuleList(list_of_kernels)

    def forward(self,
                 X: torch.Tensor,
                 Y: Optional[torch.Tensor] = None,
                 ) -> torch.Tensor:
        if Y is None:
            Y = X

        data_shape_X = X.shape[0]
        data_shape_Y = Y.shape[0]
        device = X.device

        K = torch.ones((data_shape_X, data_shape_Y), device=device, dtype=torch.float64)

        if X.shape[1] != len(self.list_of_kernels):
             raise ValueError("Number of input features must match number of kernels.")

        for jj, kernel in enumerate(self.list_of_kernels):
            # Pass single columns as (N, 1) to the kernel
            K *= kernel(X[:, jj].view(-1, 1), Y[:, jj].view(-1, 1))

        return K


class BinaryKernel(Kernel):
    """
    Binary feature matching kernel (differentiable for binary inputs).
    K(x, y) = Matches(1) + Matches(0) = x^T y + (1-x)^T (1-y).

    NOTE: One can use CategoricalKernel below for computational efficiency as it is
    based on simple comparisons. However, this implementation of binary kernel is
    differentiable and should be used when gradients are necessary.
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if Y is None:
            Y = X

        X = X.to(torch.get_default_dtype())
        Y = Y.to(torch.get_default_dtype())
        # K = X @ Y.T (Matches for 1)
        res = X @ Y.T

        # K += (1.0 - X) @ (1.0 - Y.T) (Matches for 0)
        # Use 1.0 to ensure float precision if inputs are int
        res += (1.0 - X) @ (1.0 - Y.T)

        return res

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class CategoricalKernel(Kernel):
    """
    Indicator (Delta) Kernel for categorical data.
    K(x, y) = 1 if x == y, 0 otherwise.

    NOTE: This kernel uses direct comparison and is NOT differentiable.
    It should only be used in non-gradient-based contexts.
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if Y is None:
            Y = X

        # Ensure X and Y are treated as categorical/integer data,
        # and compare them across all dimensions (D).

        # 1. Expand X and Y for broadcasting: (N, 1, D) and (1, M, D)
        diff = X.unsqueeze(1) - Y.unsqueeze(0)

        # 2. Check for equality across all features: (N, M, D)
        # (diff == 0) results in a boolean tensor: True if X_i,d == Y_j,d

        # 3. Product over the feature dimension (-1): (N, M)
        # If the product is 1 (True), all features match.
        K = torch.prod((diff == 0).to(X.dtype), dim=-1)

        return K

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class LinearKernel(Kernel):

    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self,
                 X: torch.Tensor,
                 Y: Optional[torch.Tensor] = None,
                 ) -> torch.Tensor:
        if Y is None:
            Y = X
        # Replaced X.dot(Y.T) with the modern PyTorch operator @
        return X @ Y.T

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class PolynomialKernel(Kernel):

    def __init__(self,
                 degree: int = 3,
                 gamma: Optional[float] = None,
                 coef0: float = 1.0,
                 trainable: bool = False,
                 **kwargs) -> None:
        super().__init__()
        self.degree = degree
        # Gamma and coef0 are now registered as parameters if trainable=True
        self.log_gamma = nn.Parameter(torch.tensor(np.log(gamma or 1.0)), requires_grad=trainable)
        self.coef0 = nn.Parameter(torch.tensor(coef0), requires_grad=trainable)

    @property
    def gamma(self) -> torch.Tensor:
        return torch.exp(self.log_gamma)

    def forward(self,
                 X: torch.Tensor,
                 Y: Optional[torch.Tensor] = None,
                 ) -> torch.Tensor:
        if Y is None:
            Y = X

        # Default gamma: If gamma parameter is 1.0 (default), check if it should be 1/D
        if self.log_gamma.item() == 0.0 and self.log_gamma.requires_grad is False:
             gamma_val = 1.0 / X.shape[1]
        else:
             gamma_val = self.gamma

        # K(x, y) = (gamma * x^T y + coef0)^degree
        K = (gamma_val * (X @ Y.T) + self.coef0) ** self.degree
        return K

    def get_params(self) -> Dict[str, float]:
        # Overriding to include degree which is a plain attribute
        return {
            'degree': self.degree,
            'gamma': self.gamma.item(),
            'coef0': self.coef0.item()
        }


# --- RBF and Related Kernels ---

class RBF(nn.Module):
    """
    Radial Basis Function (RBF) / Gaussian kernel with efficient subset-based
    length scale heuristic for large datasets.
    """
    def __init__(self,
                 length_scale: Union[float, torch.Tensor] = 0.5,
                 use_length_scale_heuristic: bool = False,
                 length_scale_heuristic_quantile: float = 0.5,
                 heuristic_subset_limit: int = 5000, # Use 5000 points for a stable median
                 trainable: bool = False,
                 **kwargs) -> None:
        super().__init__()

        if isinstance(length_scale, (float, int)):
            length_scale = torch.as_tensor(length_scale)

        self.log_length_scale = nn.Parameter(
            torch.log(length_scale),
            requires_grad=trainable
        )

        self.use_length_scale_heuristic = use_length_scale_heuristic
        self.length_scale_heuristic_quantile = length_scale_heuristic_quantile
        self.heuristic_subset_limit = heuristic_subset_limit
        self._is_initialized = False

    @property
    def length_scale(self) -> torch.Tensor:
        return torch.exp(self.log_length_scale)

    def calculate_and_fix_length_scale(self, X: torch.Tensor):
        """
        Calculates the heuristic using a subset of X, updates the persistent Parameter,
        and locks the state.
        """
        N = X.shape[0]

        # --- Subset Sampling ---
        # If N exceeds the limit, sample a random subset for the heuristic
        if N > self.heuristic_subset_limit:
            indices = torch.randperm(N, device=X.device)[:self.heuristic_subset_limit]
            X_sub = X[indices]
        else:
            X_sub = X

        # Compute squared distances for the subset only
        # This prevents OOM on the full N x N matrix calculation
        dist_sq_sub = pairwise_squared_distance(X_sub, X_sub)

        # --- Efficient Median Calculation ---
        N_sub = X_sub.shape[0]
        # Only take the upper triangle indices for the subset
        upper_tri_indices = torch.triu_indices(N_sub, N_sub, offset=1, device=X.device)
        upper_tri_dists = dist_sq_sub[upper_tri_indices[0], upper_tri_indices[1]]

        if upper_tri_dists.numel() > 0:
            quantile_value = torch.quantile(upper_tri_dists, self.length_scale_heuristic_quantile)
            length_scale_sq = quantile_value / 2.0
        else:
            length_scale_sq = self.length_scale**2

        new_scale = torch.sqrt(length_scale_sq)

        # --- Persistence Update ---
        with torch.no_grad():
            self.log_length_scale.data.copy_(torch.log(new_scale.data))

        self.use_length_scale_heuristic = False
        self._is_initialized = True

    def forward(self,
                 X: torch.Tensor,
                 Y: Optional[torch.Tensor] = None,
                 ) -> torch.Tensor:

        is_self_kernel = (Y is None) or (X.data_ptr() == Y.data_ptr())

        # 1. Trigger Heuristic BEFORE computing the full distance matrix
        # This prevents computing the massive distances_sq twice or OOMing
        if self.use_length_scale_heuristic and is_self_kernel and not self._is_initialized:
            self.calculate_and_fix_length_scale(X)

        if Y is None:
            Y = X

        # 2. Compute the full distances matrix for the kernel calculation
        # NOTE: For N=75,000, ensure you have enough GPU memory for the resulting matrix
        distances_sq = pairwise_squared_distance(X, Y)

        # 3. Use the persistent length scale
        length_scale_sq = self.length_scale ** 2

        # K(x, y) = exp(- ||x-y||^2 / (2 * l^2))
        K = torch.exp(-distances_sq / (2.0 * length_scale_sq))
        return K

    def get_params(self) -> Dict[str, Any]:
        params = {
            'length_scale': self.length_scale.item(),
            'use_length_scale_heuristic': self.use_length_scale_heuristic,
            'length_scale_heuristic_quantile': self.length_scale_heuristic_quantile,
            'heuristic_subset_limit': self.heuristic_subset_limit,
            'trainable': self.log_length_scale.requires_grad
        }
        return params

    def __repr__(self) -> str:
        return f"RBF(length_scale={self.length_scale.item():.4f}, subset_limit={self.heuristic_subset_limit}, initialized={self._is_initialized})"

class SquaredExponential(RBF):
    """
    Squared Exponential kernel, equivalent to RBF but includes an amplitude parameter (theta).
    K(x, y) = theta^2 * RBF(x, y)
    """
    def __init__(self,
                 theta: float = 1,
                 length_scale: float = 0.5,
                 use_length_scale_heuristic: bool = False,
                 trainable: bool = False,
                 **kwargs) -> None:

        # Initialize RBF part for length_scale management
        super().__init__(
            length_scale=length_scale,
            use_length_scale_heuristic=use_length_scale_heuristic,
            trainable=trainable
        )

        # Register amplitude (theta) as a separate parameter
        self.log_theta = nn.Parameter(torch.tensor(np.log(theta)), requires_grad=trainable)

    @property
    def theta(self) -> torch.Tensor:
        """Returns the amplitude (exp-transformed)."""
        return torch.exp(self.log_theta)

    def forward(self,
                 X: torch.Tensor,
                 Y: Optional[torch.Tensor] = None,
                 ) -> torch.Tensor:

        # Calculate the RBF component using the base RBF class logic
        RBF_K = super().forward(X, Y)

        # Scale by the amplitude squared
        return (self.theta ** 2) * RBF_K

    def get_params(self) -> Dict[str, Any]:
        params = super().get_params()
        params['theta'] = self.theta.item()
        return params


class ColumnwiseRBF(Kernel):
    """
    Computes the product of RBF kernels columnwise (Automatic Relevance Determination - ARD).
    K(X, Y) = prod_j k_j(X_j, Y_j).

    The kernel intelligently manages the one-time length scale heuristic calculation
    for each individual feature dimension on the first call with training data.
    """
    def __init__(self,
                 length_scales: Union[float, List[float]] = 0.5,
                 use_length_scale_heuristic: bool = False,
                 length_scale_heuristic_quantile: float = 0.5,
                 trainable: bool = False,
                 **kwargs) -> None:
        super().__init__()

        self.use_length_scale_heuristic = use_length_scale_heuristic
        self.length_scale_heuristic_quantile = length_scale_heuristic_quantile
        self._initial_scales = length_scales # Stores the initial value(s)
        self.trainable = trainable

        # State attributes
        self.current_length_scales: List[float] = []
        self._is_initialized = False

        # Internal container for kernels (ModuleList to register children for PyTorch)
        self.base_kernels = nn.ModuleList()

    # ------------------ Initialization Logic ------------------

    def _initialize_kernels(self, column_size: int, device: torch.device, dtype: torch.dtype):
        """Initializes the base RBF kernels based on feature dimension and initial scales."""

        # Determine the scales for initialization
        if isinstance(self._initial_scales, float):
            scales = [self._initial_scales] * column_size
        elif len(self._initial_scales) == column_size:
            scales = self._initial_scales
        else:
            raise ValueError(f"Length scales list size must match feature size ({column_size}).")

        # Create and register the individual RBF kernels
        self.base_kernels = nn.ModuleList([
            RBF(length_scale=scales[jj],
                use_length_scale_heuristic=self.use_length_scale_heuristic,
                length_scale_heuristic_quantile=self.length_scale_heuristic_quantile,
                trainable=self.trainable)
            for jj in range(column_size)
        ]).to(device=device, dtype=dtype)

        # Set the current scales list based on initial values
        self.current_length_scales = scales
        self._is_initialized = True # Parent is initialized once children are created


    # ------------------ Core Forward Pass ------------------

    def forward(self,
                 X: torch.Tensor,
                 Y: Optional[torch.Tensor] = None) -> torch.Tensor:

        if Y is None:
            Y = X

        row_x_size, column_size = X.shape
        row_y_size, _ = Y.shape
        device = X.device
        dtype = X.dtype

        # 1. Initialize kernels if not yet done (runs once on the very first call)
        if not self._is_initialized or len(self.base_kernels) != column_size:
            self._initialize_kernels(column_size, device, dtype)

        K = torch.ones((row_x_size, row_y_size), device=device, dtype=dtype)
        updated_scales: List[float] = []

        # 2. Compute product of kernels
        for jj in range(column_size):
            # Pass single columns as (N, 1) to the child RBF kernel
            X_ = X[:, jj].reshape(-1, 1)
            Y_ = Y[:, jj].reshape(-1, 1)

            # The RBF call handles its own one-time heuristic calculation internally
            # if X_=Y_ and the heuristic flag is true.
            K_ = self.base_kernels[jj](X_, Y_)
            K *= K_

            # Retrieve the current persistent length scale value from the child module
            updated_scales.append(self.base_kernels[jj].length_scale.item())

        # 3. Update the parent's persistent state
        self.current_length_scales = updated_scales

        return K

    # ------------------ Parameter Management ------------------

    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the kernel, prioritizing the current fixed/trained scales.
        """
        # This implementation requires the base Kernel to have a functional _get_hyperparameters
        # For simplicity here, we manually construct the most relevant parameters.
        params = {
            'use_length_scale_heuristic': self.use_length_scale_heuristic,
            'length_scale_heuristic_quantile': self.length_scale_heuristic_quantile,
            'trainable': self.trainable,
        }

        # Use the most current list of scales
        params['length_scales'] = self.current_length_scales if self.current_length_scales else self._initial_scales

        return params

    def __repr__(self) -> str:
        """Return a string representation."""
        scale_str = ', '.join([f'{s:.3f}' for s in self.current_length_scales]) if self.current_length_scales else 'Initial'
        return f"ColumnwiseRBF(scales=[{scale_str}], trainable={self.trainable})"


# --- Other Kernels ---

class NuclearRBF(Kernel):
    """
    Nuclear RBF kernel. A complex combination of RBFs.

    NOTE: This is currently experimental. Requires some debugging probably.
    NOTE: Implemented following the paper: ...
    TODO: Check this kernel more carefully.

    """
    def __init__(self,
                 length_scale: float = 1.0,
                 eta: float = 1.0,
                 trainable: bool = False) -> None:
        super().__init__()
        # Use log parameters for stability
        self.log_length_scale = nn.Parameter(torch.tensor(np.log(length_scale)), requires_grad=trainable)
        self.log_eta = nn.Parameter(torch.tensor(np.log(eta)), requires_grad=trainable)

    @property
    def length_scale(self) -> torch.Tensor:
        return torch.exp(self.log_length_scale)

    @property
    def eta(self) -> torch.Tensor:
        return torch.exp(self.log_eta)

    def forward(self,
                 X: torch.Tensor,
                 Y: Optional[torch.Tensor] = None,
                 ) -> torch.Tensor:

        if Y is None:
            Y = X

        data_shape = X.shape[1]
        l = self.length_scale
        eta = self.eta

        X_Y_distances = pairwise_squared_distance(X, Y)
        # Pairwise squared distance between X/2 and -Y/2
        X_minus_Y_distance_divided_by_4 = pairwise_squared_distance(X / 2.0, -1.0 * Y / 2.0)

        K_1 = torch.exp( - X_Y_distances / (4.0 * l ** 2))
        K_2 = torch.exp( (- 0.5) * X_minus_Y_distance_divided_by_4 / (0.5 * l ** 2 + eta ** 2))

        K = K_1 * K_2

        # Scaling constant calculation
        constant_term = ((2.0 * np.pi) ** data_shape) * \
                        ((2.0 / (l ** 2) + 1 / (eta ** 2)) ** (- data_shape / 2.0))

        K *= constant_term
        return K

    def get_params(self) -> Dict[str, float]:
        params = super()._get_hyperparameters()
        params['length_scale'] = self.length_scale.item()
        params['eta'] = self.eta.item()
        return params


class FourthOrderGaussianKernel(Kernel):
    """
    Fourth-Order Gaussian kernel (window function). Source: https://users.ssc.wisc.edu/~bhansen/718/NonParametrics1.pdf

    Implements a self-locking heuristic where the length scale is calculated
    once on the first self-kernel call (training data) and then fixed.

    The core mathematical reason for using this kernel is to achieve a higher-order of bias reduction (specifically, fourth-order accuracy)
    in non-parametric estimators like Kernel Density Estimation (KDE) or Local Polynomial Regression (LPR). Standard kernels often have an
    asymptotic bias proportional to the second derivative of the estimated function; this kernel is designed to cancel out those lower-order bias terms.
    """

    def __init__(self,
                 length_scale: float = 0.5,
                 use_length_scale_heuristic: bool = False,
                 length_scale_heuristic_quantile: float = 0.5,
                 trainable: bool = False,
                 ) -> None:
        super().__init__()

        # Use log-length-scale for stability and nn.Parameter for trainability
        if isinstance(length_scale, (float, int)):
            length_scale = torch.as_tensor(length_scale)

        self.log_length_scale = nn.Parameter(
            torch.log(length_scale),
            requires_grad=trainable
        )

        # State flags and heuristic parameters
        self.use_length_scale_heuristic = use_length_scale_heuristic
        self.length_scale_heuristic_quantile = length_scale_heuristic_quantile
        self._is_initialized = False # CRITICAL FLAG: Tracks if heuristic has been calculated

    @property
    def length_scale(self) -> torch.Tensor:
        """Returns the actual length scale (exp-transformed)."""
        return torch.exp(self.log_length_scale)

    def calculate_and_fix_length_scale(self, distances_sq: torch.Tensor):
        """Calculates the median/quantile heuristic, updates the parameter, and locks the state."""

        N = distances_sq.shape[0]
        # Heuristic uses upper triangle including diagonal (k=0) as per original code
        upper_tri_indices = torch.triu_indices(N, N, offset=0)
        upper_tri_dists = distances_sq[upper_tri_indices[0], upper_tri_indices[1]]

        if upper_tri_dists.numel() > 0:
            # --- Robust Quantile Calculation using NumPy/CPU for stability ---
            with torch.no_grad():
                numpy_dists = upper_tri_dists.cpu().numpy()
                quantile_value_float = np.quantile(numpy_dists, self.length_scale_heuristic_quantile)

                # Apply the kernel's specific scaling: divided by 2
                length_scale_sq = quantile_value_float / 2.0

                new_scale_float = np.sqrt(length_scale_sq)

                # Create new tensor from the Python float, ensuring correct type/device
                new_scale = torch.tensor(new_scale_float,
                                         dtype=distances_sq.dtype,
                                         device=distances_sq.device)

            # --- Persistence Update ---
            with torch.no_grad():
                self.log_length_scale.data.copy_(torch.log(new_scale.data))

        # --- Lock State: Executed regardless of whether samples existed or not ---
        self.use_length_scale_heuristic = False
        self._is_initialized = True

    def forward(self,
                 X: torch.Tensor,
                 Y: Optional[torch.Tensor] = None,
                 ) -> torch.Tensor:

        # 1. Determine if this is a self-kernel call
        # is_self_kernel=True ensures the heuristic ONLY runs on the training data.
        is_self_kernel = (Y is None) or (X.data_ptr() == Y.data_ptr())
        if Y is None:
            Y = X

        distances_sq = pairwise_squared_distance(X, Y)

        # 2. Heuristic Check & Lock (This block prevents IndexError on test data)
        if self.use_length_scale_heuristic and is_self_kernel and not self._is_initialized:
            self.calculate_and_fix_length_scale(distances_sq)

        # 3. Use the persistent length scale for kernel calculation
        length_scale = self.length_scale
        length_scale_sq = length_scale ** 2

        # kernel computation
        K = torch.exp(-distances_sq / (2.0 * length_scale_sq)) * (3.0 - (distances_sq / length_scale_sq)) / 2.0
        return K

    def get_params(self) -> Dict[str, Any]:
        """Get parameters, including the current length scale and fixed state."""
        return {
            'length_scale': self.length_scale.item(),
            'use_length_scale_heuristic': self.use_length_scale_heuristic,
            'length_scale_heuristic_quantile': self.length_scale_heuristic_quantile,
            'trainable': self.log_length_scale.requires_grad
        }


class EpanechnikovKernel(Kernel):
    """
    Epanechnikov kernel, often used as a weight function.
    It computes the product of Epanechnikov kernels columnwise.
    """
    def __init__(self,
                 length_scales: Union[float, List[float]] = 0.5,
                 use_length_scale_heuristic: bool = True,
                 columnwise: bool = True,
                 c: float = 1.0,
                 **kwargs,
                 ):
        super().__init__()

        # NOTE: length_scales is stored as a list/float attribute, NOT a nn.Parameter.
        self._initial_length_scales = length_scales # Preserve initial value
        self.length_scales = length_scales          # Active value
        self.use_length_scale_heuristic = use_length_scale_heuristic
        self.columnwise = columnwise
        self.c = c

        self._is_initialized = False # NEW: Flag to track one-time heuristic execution

    def _calculate_and_fix_scales(self, X: torch.Tensor, n_data: int, columnsize: int, device: torch.device, dtype: torch.dtype):
        """Calculates the heuristic scales and locks the state."""

        # Bandwidth heuristic: c * (std(X) / N^0.2)
        # Calculates the bandwidth vector h of shape (D,)
        std_X = torch.std(X, dim = 0)
        h = self.c * (std_X / (n_data ** 0.2))

        # Reshape to (1, 1, D) for broadcasting
        length_scales_tensor = h.to(device=device, dtype=dtype).reshape(1, 1, -1)

        # --- Lock State ---
        with torch.no_grad():
            # Update the persistent Python attribute with the final list of floats
            self.length_scales = length_scales_tensor.squeeze().tolist()
            self.use_length_scale_heuristic = False
            self._is_initialized = True

        return length_scales_tensor


    def forward(self,
                 X: torch.Tensor,
                 Y: Optional[torch.Tensor] = None) -> torch.Tensor:

        # 1. Determine if this is a self-kernel call (Essential Fix)
        is_self_kernel = (Y is None) or (X.data_ptr() == Y.data_ptr())
        if Y is None:
            Y = X

        n_data, columnsize = X.shape
        device = X.device
        dtype = X.dtype

        # Handle row-norm case (no change here)
        if not self.columnwise:
            X = (X ** 2).sum(-1).reshape(-1, 1)
            Y = (Y ** 2).sum(-1).reshape(-1, 1)
            columnsize = 1

        # 2. Heuristic Check and Initialization
        if self.use_length_scale_heuristic and is_self_kernel and not self._is_initialized:
            # Calculate the scales and update self.length_scales (which locks it)
            length_scales_tensor = self._calculate_and_fix_scales(X, n_data, columnsize, device, dtype)
        else:
            # 3. Use the currently saved, fixed scales
            if isinstance(self.length_scales, float):
                scales = [self.length_scales] * columnsize
            else:
                scales = self.length_scales # Should be a fixed List[float]
                if len(scales) != columnsize:
                     raise ValueError("Fixed length_scales size must match feature dimension.")

            length_scales_tensor = torch.tensor(scales, device=device, dtype=dtype).reshape(1, 1, -1)

        # K(x, y) = max(0, 1 - (x-y)^2 / h^2) * (3/4) / h
        # distances is (X_i - Y_j)^2 / h^2: shape (N, M, D)
        distances = ((X.unsqueeze(1) - Y.unsqueeze(0)) ** 2) / length_scales_tensor.to(X.device)

        # Epanechnikov kernel part: (1.0 - distances) * (3/4) / h
        K = torch.maximum((1.0 - distances) * (3 / 4) / length_scales_tensor.to(X.device), torch.tensor(0.0, device=device, dtype=dtype))

        # Product over feature dimension (D)
        K = torch.prod(K, dim = -1)

        return K

    # Optional: Update get_params to show the locked state
    def get_params(self) -> Dict[str, Any]:
        params = super()._get_hyperparameters()
        params['length_scales'] = self.length_scales
        params['heuristic_locked'] = not self.use_length_scale_heuristic
        return params


class MaternKernel(Kernel):
    """
    Matern kernel for half-integer smoothness parameter p.

    The kernel function is derived via SymPy and compiled to a PyTorch function.
    It features a self-locking quantile heuristic for length scale determination.
    """
    def __init__(self,
                 p: int, # Corresponds to nu = p + 0.5 in the general formula
                 length_scale: float = 1.0,
                 use_length_scale_heuristic: bool = False,
                 length_scale_heuristic_quantile: float = 0.5,
                 trainable: bool = False,
                 **kwargs) -> None:
        super().__init__()

        # --- Parameter Setup ---
        self.p = p
        self.log_length_scale = nn.Parameter(torch.tensor(np.log(length_scale)), requires_grad=trainable)
        self.use_length_scale_heuristic = use_length_scale_heuristic
        self.length_scale_heuristic_quantile = length_scale_heuristic_quantile
        self._is_initialized = False # Flag to track one-time heuristic execution

        # --- Sympy Derivation and Fix ---

        dist = sympy.symbols("d")
        length_scale_ = sympy.symbols("l2")
        matern = 0.0

        def log_factorial(k):
            return loggamma(k + 1)

        for i in range(p + 1):
            # Calculate the coefficient using log-domain for stability
            const_log = log_factorial(p + i) - log_factorial(p - i) - log_factorial(i) \
                      + log_factorial(p) - log_factorial(2 * p)
            const = sympy.exp(const_log)

            # The term (2 * sqrt(2p+1) * d / l^2)^(p-i)
            # Note: We use 2*p+1 in the sqrt, corresponding to 2*nu in the general formula where nu = p + 0.5
            term_power = (2 * sympy.sqrt(2 * p + 1) * dist / length_scale_)**(p - i)
            matern = matern + const * term_power

        # Final Matern formula: (Sum) * exp(-sqrt(2p+1) * d / l^2)
        matern = matern * sympy.exp( - sympy.sqrt(2 * p + 1) * dist / length_scale_)

        # FIX: Define the explicit PyTorch namespace for lambdify
        torch_namespace = {
            'ImmutableDenseMatrix': torch.Tensor,
            'exp': torch.exp,
            'sqrt': torch.sqrt,
            'log': torch.log,
            'pi': np.pi,
            'Abs': torch.abs,
        }

        # lambdify should use [numpy, torch_namespace] to map math functions to torch.
        f_matern = sympy.utilities.lambdify(
            [dist, length_scale_],
            matern,
            modules=["numpy", torch_namespace]
        )
        self.f_matern = f_matern

    @property
    def length_scale(self) -> torch.Tensor:
        """Returns the actual length scale (exp-transformed)."""
        return torch.exp(self.log_length_scale)

    def calculate_and_fix_length_scale(self, distances: torch.Tensor):
        """Calculates the quantile heuristic, updates the parameter, and locks the state."""

        N = distances.shape[0]
        upper_tri_indices = torch.triu_indices(N, N, offset=1)
        upper_tri_dists = distances[upper_tri_indices[0], upper_tri_indices[1]]

        if upper_tri_dists.numel() > 0:
            # --- Robust Quantile Calculation using NumPy ---
            with torch.no_grad():
                numpy_dists = upper_tri_dists.cpu().numpy()
                # Use torch.quantile with the stored parameter
                # new_scale_val = np.quantile(numpy_dists, self.length_scale_heuristic_quantile)
                new_scale_val = torch.quantile(upper_tri_dists, self.length_scale_heuristic_quantile).item()
                # new_scale_val = torch.quantile(distances, self.length_scale_heuristic_quantile)

                # Convert back to tensor
                new_scale = torch.tensor(new_scale_val, dtype=distances.dtype, device=distances.device)
        else:
            new_scale = self.length_scale.data # Fallback to current value

        # --- Persistence Update ---
        with torch.no_grad():
            self.log_length_scale.data.copy_(torch.log(new_scale.data))

        # --- Lock State ---
        self.use_length_scale_heuristic = False
        self._is_initialized = True

    def forward(self,
                 X: torch.Tensor,
                 Y: Optional[torch.Tensor] = None,) -> torch.Tensor:

        # --- Self-Kernel Check ---
        is_self_kernel = (Y is None) or (X.data_ptr() == Y.data_ptr())
        if Y is None:
            Y = X

        distances_sq = pairwise_squared_distance(X, Y)
        # Matern requires the non-squared distance
        distances = torch.sqrt(distances_sq)

        # --- Heuristic Check ---
        if self.use_length_scale_heuristic and is_self_kernel and not self._is_initialized:
            # Runs only on the first self-kernel call (training data)
            self.calculate_and_fix_length_scale(distances)

        # The scale used is always the persistent value
        length_scale = self.length_scale

        # Call the pre-compiled Torch function
        # The function expects d (distance) and l2 (length_scale_squared)
        K = self.f_matern(distances, length_scale)
        return K

    def get_params(self) -> Dict[str, Any]:
        """Get parameters, including the smoothness and heuristic state."""
        # For simplicity, returning just the core parameters here.
        return {
            'length_scale': self.length_scale.item(),
            'p': self.p,
            'length_scale_heuristic_quantile': self.length_scale_heuristic_quantile,
            'heuristic_locked': not self.use_length_scale_heuristic
        }


class ColumnwiseMaternKernel(Kernel):
    """
    Computes the product of Matern kernels columnwise (Automatic Relevance Determination - ARD).
    Allows both length_scales and the smoothness parameter 'p' to vary per dimension.
    """
    def __init__(self,
                 p: Union[int, List[int]] = 1, # NOW ACCEPTS int OR List[int]
                 length_scales: Union[float, List[float]] = 0.5,
                 use_length_scale_heuristic: bool = False,
                 length_scale_heuristic_quantile: float = 0.5,
                 trainable: bool = False,
                 **kwargs) -> None:
        super().__init__()

        # Store initial p value(s)
        self._initial_p = p
        self._initial_scales = length_scales

        self.use_length_scale_heuristic = use_length_scale_heuristic
        self.length_scale_heuristic_quantile = length_scale_heuristic_quantile
        self.trainable = trainable

        # State attributes
        self.current_length_scales: List[float] = []
        self.current_p: List[int] = [] # NEW: To store the active list of p values
        self._is_initialized = False

        self.base_kernels = nn.ModuleList()

    # ------------------ Initialization Logic ------------------

    def _initialize_kernels(self, column_size: int, device: torch.device, dtype: torch.dtype):
        """Initializes the base Matern kernels, mapping scales and p values to dimensions."""

        # --- 1. Map p values to dimensions ---
        if isinstance(self._initial_p, int):
            p_values = [self._initial_p] * column_size
        elif len(self._initial_p) == column_size:
            p_values = self._initial_p
        else:
            raise ValueError(f"Smoothness parameter 'p' list size ({len(self._initial_p)}) must match feature size ({column_size}).")

        # --- 2. Map length scales to dimensions ---
        if isinstance(self._initial_scales, float):
            scales = [self._initial_scales] * column_size
        elif len(self._initial_scales) == column_size:
            scales = self._initial_scales
        else:
            raise ValueError(f"Length scales list size ({len(self._initial_scales)}) must match feature size ({column_size}).")

        # --- 3. Create and register Matern kernels ---
        self.base_kernels = nn.ModuleList([
            MaternKernel(p=p_values[jj], # Use the dimension-specific p
                         length_scale=scales[jj],
                         use_length_scale_heuristic=self.use_length_scale_heuristic,
                         length_scale_heuristic_quantile=self.length_scale_heuristic_quantile,
                         trainable=self.trainable)
            for jj in range(column_size)
        ]).to(device=device, dtype=dtype)

        # --- 4. Update Parent State ---
        self.current_length_scales = scales
        self.current_p = p_values # Store the active list of p values
        self._is_initialized = True


    # ------------------ Core Forward Pass ------------------

    def forward(self,
                 X: torch.Tensor,
                 Y: Optional[torch.Tensor] = None) -> torch.Tensor:

        if Y is None:
            Y = X

        row_x_size, column_size = X.shape
        row_y_size, _ = Y.shape
        device = X.device
        dtype = X.dtype

        # 1. Initialize kernels if not yet done (runs once on the very first call)
        if not self._is_initialized or len(self.base_kernels) != column_size:
            self._initialize_kernels(column_size, device, dtype)

        K = torch.ones((row_x_size, row_y_size), device=device, dtype=dtype)
        updated_scales: List[float] = []

        # 2. Compute product of kernels
        for jj in range(column_size):
            X_ = X[:, jj].reshape(-1, 1)
            Y_ = Y[:, jj].reshape(-1, 1)

            K_ = self.base_kernels[jj](X_, Y_)
            K *= K_

            # Retrieve the current persistent length scale value from the child module
            updated_scales.append(self.base_kernels[jj].length_scale.item())

            # NOTE: The 'p' value is fixed in __init__ for MaternKernel and does not need retrieval.

        # 3. Update the parent's persistent scale state
        self.current_length_scales = updated_scales

        return K

    # ------------------ Parameter Management ------------------

    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the kernel, including the anisotropic p values.
        """
        params = {
            'p': self.current_p if self.current_p else self._initial_p, # Use the active list of p
            'use_length_scale_heuristic': self.use_length_scale_heuristic,
            'length_scale_heuristic_quantile': self.length_scale_heuristic_quantile,
            'trainable': self.trainable,
        }

        params['length_scales'] = self.current_length_scales if self.current_length_scales else self._initial_scales

        return params

    def __repr__(self) -> str:
        """Return a string representation."""
        p_str = str(self.current_p) if self.current_p else str(self._initial_p)
        scale_str = ', '.join([f'{s:.3f}' for s in self.current_length_scales]) if self.current_length_scales else 'Initial'
        return f"ColumnwiseMaternKernel(p={p_str}, scales=[{scale_str}], trainable={self.trainable})"


class LaplacianKernel(RBF):
    """
    Laplacian kernel (Exponential kernel) using L1 distance.

    K(x, y) = exp(- ||x-y||_1 / (2 * l^2)). Uses RBF's parameter management.
    """
    def __init__(self,
                 length_scale: float = 0.5,
                 use_length_scale_heuristic: bool = False,
                 length_scale_heuristic_quantile: float = 0.5,
                 trainable: bool = False,
                 **kwargs) -> None:
        # Calls RBF.__init__ to set up parameters and flags (_is_initialized=False)
        super().__init__(
            length_scale=length_scale,
            use_length_scale_heuristic=use_length_scale_heuristic,
            length_scale_heuristic_quantile=length_scale_heuristic_quantile,
            trainable=trainable
        )

    def calculate_and_fix_length_scale(self, distances_abs: torch.Tensor):
        """Calculates the quantile heuristic on L1 distances, updates the parameter, and locks the state."""

        N = distances_abs.shape[0]
        upper_tri_indices = torch.triu_indices(N, N, offset=1)
        upper_tri_dists = distances_abs[upper_tri_indices[0], upper_tri_indices[1]]

        if upper_tri_dists.numel() > 0:
            # --- Use torch.quantile (No NumPy transfer) ---
            with torch.no_grad():
                # torch.quantile returns a tensor, so we detach and use.item
                # to get the standard Python float and avoid the UserWarning.
                quantile_value = torch.quantile(upper_tri_dists, self.length_scale_heuristic_quantile)
                quantile_value_float = quantile_value.item()

                # Apply the original kernel scaling to the result
                length_scale_sq = quantile_value_float / 2.0

                new_scale_float = np.sqrt(length_scale_sq)

                # Create new tensor from the Python float
                new_scale = torch.tensor(new_scale_float,
                                         dtype=distances_abs.dtype,
                                         device=distances_abs.device)

            # --- Persistence Update ---
            with torch.no_grad():
                self.log_length_scale.data.copy_(torch.log(new_scale.data))

        # --- Lock State ---
        self.use_length_scale_heuristic = False
        self._is_initialized = True

    def forward(self,
                 X: torch.Tensor,
                 Y: Optional[torch.Tensor] = None,
                 ) -> torch.Tensor:

        # 1. Determine if this is a self-kernel call
        is_self_kernel = (Y is None) or (X.data_ptr() == Y.data_ptr())
        if Y is None:
            Y = X

        # 2. Compute L1 distance
        distances_abs = pairwise_absolute_distance(X, Y)

        # 3. Check for one-time heuristic initialization
        if self.use_length_scale_heuristic and is_self_kernel and not self._is_initialized:
            self.calculate_and_fix_length_scale(distances_abs)

        # 4. Use the persistent length scale for kernel calculation
        length_scale = self.length_scale
        length_scale_sq = length_scale ** 2

        # K(x, y) = exp(- ||x-y||_1 / (2 * l^2)) (Using your original kernel form)
        K = torch.exp(-distances_abs / (2.0 * length_scale_sq))
        return K
