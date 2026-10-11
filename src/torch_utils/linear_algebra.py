# utils/linalg.py

import torch
from torch.jit import script
import numpy as np
from typing import Optional, Sequence, Union, Any

# Set default tensor type to float64 for high-precision kernel methods
# torch.set_default_dtype(torch.float64)

def add_const_col(mat: torch.Tensor):
    """

    Parameters
    ----------
    mat : torch.Tensor[n_data, n_col]

    Returns
    -------
    res : torch.Tensor[n_data, n_col+1]
        add one column only contains 1.

    """
    assert mat.dim() == 2
    n_data = mat.size()[0]
    device = mat.device
    return torch.cat([mat, torch.ones((n_data, 1), device=device)], dim=1)

def pairwise_squared_distance(X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the pairwise squared Euclidean distances: ||X_i - Y_j||^2.

    This is calculated using the formula ||x||^2 + ||y||^2 - 2x^T y, which is efficient
    for GPU computation via highly optimized matrix operations.

    Parameters
    -----------
    X: torch.Tensor of shape (n_samples_X, n_features)
    Y: torch.Tensor of shape (n_samples_Y, n_features), optional
        If not provided, defaults to X.

    Returns
    -------
    torch.Tensor
        Pairwise squared distances of shape (n_samples_X, n_samples_Y).
    """
    if Y is None:
        Y = X

    # Ensure inputs are on the same device and dtype
    if X.dtype != torch.get_default_dtype():
        X = X.to(torch.get_default_dtype())
    if Y.dtype != torch.get_default_dtype() or Y.device != X.device:
        Y = Y.to(dtype=X.dtype, device=X.device)

    # X_norm_sq: (N, 1) - Squared L2 norm of each row in X
    X_norm_sq = (X ** 2).sum(dim=-1, keepdim=True)

    # Y_norm_sq: (1, M) - Squared L2 norm of each row in Y (transposed for broadcasting)
    Y_norm_sq = (Y ** 2).sum(dim=-1, keepdim=True).T

    # Cross term: -2 * X @ Y.T
    cross_term = 2.0 * (X @ Y.T)

    # dist_sq = ||x||^2 + ||y||^2 - 2x^T y
    dist_sq = X_norm_sq + Y_norm_sq - cross_term

    # Clamp to zero for numerical stability (distances must be non-negative)
    return torch.clamp(dist_sq, min=0.0)


def pairwise_absolute_distance(X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the pairwise absolute (Manhattan/L1) distances: sum(|X_i - Y_j|).

    Parameters
    -----------
    X: torch.Tensor of shape (n_samples_X, n_features)
    Y: torch.Tensor of shape (n_samples_Y, n_features), optional
        If not provided, defaults to X.

    Returns
    -------
    torch.Tensor
        Pairwise absolute distances of shape (n_samples_X, n_samples_Y).
    """
    if Y is None:
        Y = X

    # Ensure inputs are on the same device and dtype
    if X.dtype != torch.get_default_dtype():
        X = X.to(torch.get_default_dtype())
    if Y.dtype != torch.get_default_dtype() or Y.device != X.device:
        Y = Y.to(dtype=X.dtype, device=X.device)

    # Broadcasted difference: X[:, None] - Y[None, :] is (N, M, D)
    diff = X.unsqueeze(1) - Y.unsqueeze(0)

    # Sum the absolute difference over the feature dimension (-1)
    return torch.sum(torch.abs(diff), dim=-1)


def make_psd(A: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Ensure the matrix is Positive Semi-Definite (PSD) by symmetrizing and adding
    a small epsilon (jitter) to the diagonal. This is a common regularization technique
    for kernel matrices.

    Parameters:
    - A (torch.Tensor): Input matrix of shape (n, n).
    - eps (float): Jitter value added to the diagonal.

    Returns:
    - torch.Tensor: Positive semi-definite matrix.
    """
    n = A.shape[0]
    # Symmetrize
    sym_A = (A + A.T) / 2

    # Add jitter to the diagonal
    return sym_A + eps * torch.eye(n, dtype=A.dtype, device=A.device)


def cartesian_product(*arrays: Sequence[Union[np.ndarray, torch.Tensor]]) -> torch.Tensor:
    """
    Compute the Cartesian product of input arrays/tensors.

    This utility is useful for generating all combinations of indices or feature values.
    The core logic uses efficient NumPy indexing before converting the result to a PyTorch tensor.

    Args:
        *arrays: Variable number of input arrays (NumPy or PyTorch).

    Returns:
        torch.Tensor: Cartesian product tensor of shape (prod(len(arr)), len(arrays)).

    Raises:
        ValueError: If the input arrays are empty.
    """
    if not arrays:
        raise ValueError("No arrays provided for Cartesian product.")

    # Convert all inputs to numpy arrays for reliable indexing operations
    np_arrays = [a.cpu().numpy() if isinstance(a, torch.Tensor) else a for a in arrays]

    la = len(np_arrays)
    # Use numpy.ix_ to efficiently compute the product indices
    arr = np.empty([len(a) for a in np_arrays] + [la], dtype=np.result_type(*np_arrays))
    for i, a in enumerate(np.ix_(*np_arrays)):
        arr[...,i] = a

    # Convert the resulting numpy array back to a PyTorch tensor
    return torch.from_numpy(arr.reshape(-1, la))


def remove_diagonal_elements(A: torch.Tensor) -> torch.Tensor:
    """
    Sets the diagonal elements of a square matrix to zero.

    Parameters:
    - A (torch.Tensor): Input matrix of shape (n, n).

    Returns:
    - torch.Tensor: Matrix A with zeroed diagonal.
    """
    # Subtract the diagonal from A
    return A - torch.diag_embed(torch.diag(A))


def columns_mean_excluding_self(A: torch.Tensor) -> torch.Tensor:
    """
    Computes the column mean of A, excluding the element on the main diagonal
    for each column.

    This is equivalent to multiplying A by an averaging matrix B = (J - I) / (n - 1),
    where J is the matrix of ones and I is the identity matrix.

    Parameters:
    - A (torch.Tensor): Input matrix of shape (n_samples, n_samples).

    Returns:
    - torch.Tensor: Result of A multiplied by the off-diagonal averaging matrix.
    """
    n = A.shape[1]

    if n <= 1:
        # If the matrix has one column, the mean excluding self is ill-defined (or zero)
        return torch.zeros_like(A)

    # Create the averaging matrix B: (Ones - Identity) / (n - 1)
    ones = torch.ones((n, n), dtype=A.dtype, device=A.device)
    I = torch.eye(n, dtype=A.dtype, device=A.device)

    B = (ones - I) / (n - 1)

    # Result is A @ B
    return A @ B


def outer_prod(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """
    Compute the batched outer product between two tensors.

    Parameters
    ----------
    mat1 : torch.Tensor of shape (n_batch, d1, d2, ..., dk)
        First batch of tensors.
    mat2 : torch.Tensor of shape (n_batch, e1, e2, ..., em)
        Second batch of tensors.

    Returns
    -------
    res : torch.Tensor of shape (n_batch, d1, d2, ..., dk, e1, e2, ..., em)
        Batched outer product of `mat1` and `mat2`.

    Notes
    -----
    - This function generalizes the outer product to higher-dimensional tensors.
    - The first dimension (`n_batch`) must match for both inputs.
    """
    mat1_shape = tuple(mat1.size())
    mat2_shape = tuple(mat2.size())
    assert mat1_shape[0] == mat2_shape[0], "Batch dimensions must match."

    nData = mat1_shape[0]
    aug_mat1_shape = mat1_shape + (1,) * (len(mat2_shape) - 1)
    aug_mat1 = torch.reshape(mat1, aug_mat1_shape)
    aug_mat2_shape = (nData,) + (1,) * (len(mat1_shape) - 1) + mat2_shape[1:]
    aug_mat2 = torch.reshape(mat2, aug_mat2_shape)

    return aug_mat1 * aug_mat2


def outer_prod_batch(*mats: torch.Tensor) -> torch.Tensor:
    """
    Compute the batched outer product between multiple tensors.

    Parameters
    ----------
    *mats : list of torch.Tensor
        Each tensor should have shape (n_batch, d1, d2, ..., dk).
        All tensors must share the same batch size.

    Returns
    -------
    res : torch.Tensor
        Batched outer product of all input tensors.
        Shape = (n_batch, d1, ..., dk, e1, ..., em, f1, ..., fn, ...)
    """
    if len(mats) < 2:
        raise ValueError("At least two tensors are required for outer product.")

    # Check batch sizes
    batch_size = mats[0].shape[0]
    for m in mats:
        if m.shape[0] != batch_size:
            raise ValueError("All tensors must have the same batch size.")

    # Start with the first tensor
    res = mats[0]

    for mat in mats[1:]:
        mat1_shape = tuple(res.size())
        mat2_shape = tuple(mat.size())

        aug_mat1_shape = mat1_shape + (1,) * (len(mat2_shape) - 1)
        aug_mat1 = res.reshape(aug_mat1_shape)

        aug_mat2_shape = (batch_size,) + (1,) * (len(mat1_shape) - 1) + mat2_shape[1:]
        aug_mat2 = mat.reshape(aug_mat2_shape)

        res = aug_mat1 * aug_mat2

    return res


def outer_prod_cross_batch(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """
    Compute the outer product of every possible pair between two batches.

    If mat1 has N samples (rows) and mat2 has M samples, the output will have
    N * M feature vectors, representing the Cartesian product of the two batches.

    Parameters
    ----------
    mat1 : torch.Tensor of shape (N, D1, ...)
        First batch of tensors (e.g., features from treatment A).
    mat2 : torch.Tensor of shape (M, D2, ...)
        Second batch of tensors (e.g., features from proxy W).
        (Note: N and M are typically equal to the batch size).

    Returns
    -------
    res : torch.Tensor of shape (N * M, D1, ..., D2, ...)
        Cartesian outer product of all pairs, flattened into N*M samples.
    """

    N = mat1.shape[0]
    M = mat2.shape[0]

    # --- 1. Reshape and Expand for Cross-Product ---

    # mat1_reshaped: (N, 1, D1,...)
    # Expands mat1 along a new dimension (axis 1) to prepare for broadcasting against M samples.
    mat1_expanded = mat1.unsqueeze(1).unsqueeze(-1)

    # mat2_reshaped: (1, M, D2,...)
    # Expands mat2 along a new dimension (axis 0) to prepare for broadcasting against N samples.
    mat2_expanded = mat2.unsqueeze(0).unsqueeze(-2)

    # --- 2. Element-wise Multiplication (Broadcasting) ---

    # The multiplication (N, M, D1, D2,...) automatically calculates the outer product
    # for all N*M pairs via PyTorch's broadcasting mechanism.
    cross_product_tensor = mat1_expanded * mat2_expanded

    # --- 3. Flatten into (N*M, D_combined) ---

    # The final step is to combine the batch dimensions (N and M) and flatten the
    # feature dimensions (D1, D2, etc.) into a single, combined feature vector.

    # The first two dimensions (N, M) are the samples. The remaining dimensions are the features.

    # Combine N and M dimensions into a single sample dimension (N * M)
    res = cross_product_tensor#.reshape(N * M, -1)

    return res


def fit_linear(
    target: torch.Tensor, feature: torch.Tensor, reg: float = 0.0
) -> torch.Tensor:
    """
    Fit a ridge linear regression model using `torch.linalg.solve`
    (avoids explicit matrix inversion).

    Parameters
    ----------
    target : torch.Tensor of shape (n_batch, d1, d2, ..., dk)
        Target tensor (labels).
    feature : torch.Tensor of shape (n_batch, n_features)
        Input feature matrix.
    reg : float, default=0.0
        L2 regularization strength (ridge penalty).

    Returns
    -------
    weight : torch.Tensor of shape (n_features, d1, d2, ..., dk)
        Regression weights solving the ridge regression problem:

        min_w ||Y - Xw||^2 + reg * n_batch * ||w||^2
    """
    assert feature.dim() == 2, "Feature must be 2D (n_batch, n_features)."
    assert target.dim() >= 2, "Target must be at least 2D."

    nData, nDim = feature.size()
    device = feature.device

    # Compute A = X^T X + reg * nData * I
    A = torch.matmul(feature.t(), feature) + reg * nData * torch.eye(nDim, device=device)

    # Compute b = X^T Y
    if target.dim() == 2:
        b = torch.matmul(feature.t(), target)
    else:
        b = torch.einsum("nd,n...->d...", feature, target)

    # Solve A @ weight = b
    weight = torch.linalg.solve(A, b)

    return weight

# @script
def fit_linear_proximal(
    target: torch.Tensor,
    feature: torch.Tensor,
    W_previous: torch.Tensor,
    reg_lambda: float = 0.0, # Proximal coefficient lambda
    fit_type:str = "ridge"
) -> torch.Tensor:
    r"""
    Fit the optimal closed-form solution W* for the proximal ridge linear regression.

    This loss minimizes the prediction error on the current batch while regularizing
    the solution towards the previous estimate W_previous (W_t)[cite: 1171, 1174].

    The objective minimized is the proximal loss:

    $$
    \mathcal{L}^{prox}(W) = \sum_{i \in \mathcal{B}_t} ||Y_i - W\phi(X_i)||^2 + \lambda ||W - W_{t}||_F^2
    $$

    The closed-form solution W* is given by:
    $$
    W^{*} = (\mathbf{\Phi}^\top \mathbf{\Phi} + \lambda \mathbf{I})^{-1} (\mathbf{\Phi}^\top \mathbf{Y} + \lambda \mathbf{W}_{t}^{\top})
    $$

    Parameters
    ----------
    target : torch.Tensor
        Target vector/matrix Y of shape (n_batch, output_dim, ...).
    feature : torch.Tensor
        Input feature matrix $\mathbf{\Phi}$ of shape (n_batch, n_features).
    W_previous : torch.Tensor
        The weight matrix W_t from the previous iteration, shape (output_dim, n_features).
    reg_lambda : float, default=0.0
        Proximal coefficient $\lambda$ (controls regularization strength towards $W_{t}$)[cite: 1172].

    Returns
    -------
    weight : torch.Tensor
        The optimal proximal weight matrix W*, shape (n_features, output_dim).
    """
    assert feature.dim() == 2, "Feature must be 2D (n_batch, n_features)."
    assert target.dim() >= 2, "Target must be at least 2D."

    nData, nFeat = feature.size()

    if fit_type == "ridge":
        # 1. Compute LHS matrix (A = Feature^T Feature + lambda * I)
        # This matrix is the kernel matrix of the features + proximal penalty on the diagonal.
        A = torch.matmul(feature.t(), feature) + nData * reg_lambda * torch.eye(nFeat, device=feature.device)

        # 2. Compute RHS vector (b = Feature^T Target + lambda * W_previous^T)

        # Term 1: Feature^T Target (Standard regression term)
        if target.dim() == 2:
            b_target = torch.matmul(feature.t(), target)
        else:
            # Use einsum for high-dimensional targets
            b_target = torch.einsum("nd,n...->d...", feature, target)

        # Term 2: reg_lambda * W_previous^T (Proximal prior term)
        # W_previous is (output_dim, n_features). W_previous^T is (n_features, output_dim).
        b_proximal = b_target + nData * reg_lambda * W_previous

        # 3. Solve for weight: weight = inv(A) @ b
        weight = torch.linalg.solve(A, b_proximal)
    elif fit_type == "kernel_ridge":
        Gram_mat = torch.matmul(feature, feature.t())
        A2 = Gram_mat + nData * reg_lambda * torch.eye(nData, device=feature.device)
        b2 = torch.matmul(Gram_mat, target) + nData * reg_lambda * torch.matmul(feature, W_previous)
        alpha = torch.linalg.solve(torch.matmul(Gram_mat, A2), b2)
        weight = torch.matmul(feature.t(), alpha)

    return weight


def linear_reg_pred(feature: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Predict targets from features and learned regression weights.

    Parameters
    ----------
    feature : torch.Tensor of shape (n_batch, n_features)
        Input feature matrix.
    weight : torch.Tensor of shape (n_features, d1, d2, ..., dk)
        Regression weights.

    Returns
    -------
    pred : torch.Tensor of shape (n_batch, d1, d2, ..., dk)
        Predicted targets.
    """
    assert weight.dim() >= 2, "Weight must be at least 2D."

    if weight.dim() == 2:
        return torch.matmul(feature, weight)
    else:
        return torch.einsum("nd,d...->n...", feature, weight)


def linear_reg_loss(target: torch.Tensor, feature: torch.Tensor, reg: float) -> torch.Tensor:
    """
    Compute the ridge regression loss.

    Parameters
    ----------
    target : torch.Tensor of shape (n_batch, d1, d2, ..., dk)
        True target tensor (labels).
    feature : torch.Tensor of shape (n_batch, n_features)
        Input feature matrix.
    reg : float
        L2 regularization strength (ridge penalty).

    Returns
    -------
    loss : torch.Tensor (scalar)
        Ridge regression loss:

        ||Y - Xw||^2 + reg * n_batch * ||w||^2
    """
    weight = fit_linear(target, feature, reg)
    pred = linear_reg_pred(feature, weight)
    nData, _ = feature.size()
    return torch.norm((target - pred)) ** 2 + reg * torch.norm(weight) ** 2 * nData, weight


def linear_reg_proxy_loss(target: torch.Tensor, feature: torch.Tensor, W_previous: torch.Tensor, reg: float) -> torch.Tensor:
    """
    Compute the ridge regression loss.

    Parameters
    ----------
    target : torch.Tensor of shape (n_batch, d1, d2, ..., dk)
        True target tensor (labels).
    feature : torch.Tensor of shape (n_batch, n_features)
        Input feature matrix.
    reg : float
        L2 regularization strength (ridge penalty).

    Returns
    -------
    loss : torch.Tensor (scalar)
        Ridge regression loss:

        ||Y - Xw||^2 + reg * n_batch * ||w||^2
    """
    weight = fit_linear_proximal(target, feature, W_previous, reg)
    pred = linear_reg_pred(feature, weight)
    nData, _ = feature.size()
    return torch.norm((target - pred)) ** 2 + reg * torch.norm(weight) ** 2 * nData, weight
