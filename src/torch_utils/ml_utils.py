# utils/ml_utils.py

import torch
from typing import Tuple, Optional, Union, Dict
import torch.nn.functional as F

# Assuming access to make_psd from linalg_utils
from .linear_algebra import make_psd


def lambda_objective_loocv(
    lambda_: torch.Tensor,
    K_XX: torch.Tensor,
    Y_or_KYY: torch.Tensor, # Renamed input for clarity
    label_variance_in_lambda_opt: float = 0.0,
    make_psd_eps: float = 1e-9
) -> torch.Tensor:
    """
    Computes the Leave-One-Out Cross-Validation (LOOCV) objective
    for KRR (scalar Y) or CME (kernel K_YY) regularization optimization.

    Parameters
    ----------
    lambda_ : torch.Tensor (scalar)
        Regularization parameter.
    K_XX : torch.Tensor
        Input kernel matrix K(X, X) of shape (n, n).
    Y_or_KYY : torch.Tensor
        Target values vector Y (n, 1) or target kernel matrix K_YY (n, n).
    label_variance_in_lambda_opt : float
        Regularization weight for the label variance terms.
    make_psd_eps : float
        Epsilon value for ensuring the matrix is positive semi-definite (jitter).

    Returns
    -------
    torch.Tensor (scalar)
        The LOOCV objective value.
    """
    n = K_XX.shape[0]
    device = K_XX.device
    dtype = K_XX.dtype

    # 1. Input Validation and Preparation
    is_kernel_input = (Y_or_KYY.ndim == 2 and Y_or_KYY.shape[1] == n)

    if not isinstance(lambda_, torch.Tensor):
        lambda_ = torch.tensor(lambda_, dtype=dtype, device=device)

    identity_matrix = torch.eye(n, dtype=dtype, device=device)

    # A = K_XX + n * lambda_ * I
    A = K_XX + n * lambda_ * identity_matrix

    # Ensure A is PSD before solving (crucial for stability)
    A_psd = make_psd(A, eps=make_psd_eps)

    # --- 2. Calculate Influence Matrix Components ---

    # R: K_XX @ (K_XX + n*lambda*I)^-1
    R = torch.linalg.solve(A_psd.T, K_XX.T).T
    H_alpha = identity_matrix - R # Influence/Residual Matrix H = I - R

    # H_tilde_alpha_inv = diag(1 / diag(H_alpha))
    H_alpha_diag = torch.diag(H_alpha)
    H_tilde_alpha_inv_diag = 1.0 / H_alpha_diag
    H_tilde_alpha_inv = torch.diag_embed(H_tilde_alpha_inv_diag)

    # --- 3. Conditional Loss Calculation (KRR vs. CME) ---

    if is_kernel_input:
        # Case: K_YY (Kernel Target - CME Objective)
        K_YY = Y_or_KYY

        # Weighted Residual Covariance: H_tilde_alpha_inv @ H_alpha @ K_YY @ H_alpha @ H_tilde_alpha_inv
        # Note: Your formula is missing the transpose on the second H_alpha, but the CME LOOCV loss
        # is generally defined as trace(S @ Residual_Cov), where Residual_Cov = H @ K_YY @ H^T.
        # Since H_alpha is symmetric (it's I - R, and R is symmetric for K_XX), we use:
        weighted_residual_cov = H_tilde_alpha_inv @ H_alpha @ K_YY @ H_alpha @ H_tilde_alpha_inv

        loss = (1/n) * torch.trace(weighted_residual_cov)

    else:
        # Case: Y (Vector Target - Standard KRR Objective)
        Y_vec = Y_or_KYY.reshape(-1, 1) # Ensure (n, 1) shape

        # Weighted Error: H_tilde_alpha_inv @ H_alpha @ Y
        weighted_error = H_tilde_alpha_inv @ H_alpha @ Y_vec

        # loss = (1 / n) * || weighted_error ||^2
        loss = (1 / n) * torch.linalg.norm(weighted_error) ** 2

    # --- 4. Add Label Variance Regularization Term ---

    if label_variance_in_lambda_opt > 0.0:
        # These terms are common to both KRR and CME objectives
        loss += label_variance_in_lambda_opt * torch.trace(R)
        loss += (1 / n) * label_variance_in_lambda_opt * torch.sum((H_alpha_diag - 1) / H_alpha_diag)
        loss += (1 / n) * label_variance_in_lambda_opt * torch.trace(R @ H_tilde_alpha_inv @ R.T)

    return loss


def cme_lambda_objective_loocv(
    lambda_: torch.Tensor,
    K_AA: torch.Tensor,  # Kernel matrix for the input variable (A)
    K_YY: torch.Tensor,  # Kernel matrix for the output variable (Y)
    make_psd_eps: float = 1e-9
) -> torch.Tensor:
    """
    Computes the Leave-One-Out Cross-Validation (LOOCV) objective for optimizing
    the regularization parameter (lambda_) for the Conditional Mean Embedding (CME) estimator.

    This loss is used when the output (Y) is high-dimensional or in an RKHS (e.g., K_YY is K_XX).

    See algorithm 7 in https://arxiv.org/abs/2012.10315
    Kernel Methods for Unobserved Confounding: Negative Controls, Proxies, and Instruments by Rahul Singh

    Parameters
    ----------
    lambda_ : torch.Tensor (scalar)
        Regularization parameter (lambda2_).
    K_AA : torch.Tensor
        Input kernel matrix K(A, A) of shape (n, n).
    K_YY : torch.Tensor
        Output kernel matrix K(Y, Y) of shape (n, n).
    make_psd_eps : float
        Epsilon value for ensuring the matrix is positive semi-definite (jitter).

    Returns
    -------
    torch.Tensor (scalar)
        The LOOCV objective value (proportional to squared prediction error in the RKHS).
    """
    n = K_AA.shape[0]
    device = K_AA.device
    dtype = K_AA.dtype
    identity = torch.eye(n, dtype=dtype, device=device)

    # Ensure lambda_ is a tensor
    if not isinstance(lambda_, torch.Tensor):
        lambda_ = torch.tensor(lambda_, dtype=dtype, device=device)

    # 1. R = K_AA @ inv(make_psd(K_AA) + n * lambda * I)
    A_matrix = K_AA + n * lambda_ * identity
    A_psd = make_psd(A_matrix, eps=make_psd_eps)
    # A_psd_inv = torch.linalg.pinv(A_psd) # Use pinv for robust KRR inversion
    R = torch.linalg.solve(A_psd, K_AA).T # K_AA @ A_psd_inv

    # 2. S = diag((1 / (1 - diag(R))) ** 2)
    R_diag = torch.diag(R)
    S_diag_inv_sq = (1.0 - R_diag) ** 2
    S_diag = 1.0 / S_diag_inv_sq
    S = torch.diag_embed(S_diag) # Diagonal matrix S

    # 3. T = S @ (K_YY - 2 * K_YY @ R.T + R @ K_YY @ R.T)
    # This is the expected LOOCV residual term for a vector-valued KRR (CME)

    # K_YY @ R.T term
    K_YY_R_T = K_YY @ R.T

    # K_YY - 2 * K_YY @ R.T + R @ K_YY @ R.T  = K_YY @ (I - R)^T @ (I - R) +...
    # This matrix is the residual covariance.
    Residual_Cov = K_YY - 2 * K_YY_R_T + R @ K_YY_R_T

    T = S @ Residual_Cov

    # 4. cost = trace(T)
    cost = torch.trace(T)
    return cost