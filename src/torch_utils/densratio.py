import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
try:
    import zuko  # only needed by HighDimCausalDensityRatioEstimator (normalizing-flow density ratios)
except ImportError:  # pragma: no cover - optional dependency
    zuko = None
from densratio import densratio
# from sklearn.model_selection import GridSearchCV
# from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from typing import List, Optional
import warnings

import sys
import os
# sys.path.append("..")
# torch_utils_root = os.path.abspath("..")
# if torch_utils_root not in sys.path:
#     sys.path.insert(0, torch_utils_root)

from .kernel_utils import RBF
########## Kernel Density Estimation Module ##########

class CausalKDEDensRatioTorch:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Store the training data and optimal bandwidths
        self.train_A = None
        self.train_WX = None
        self.train_AWX = None

        self.bw_a = None
        self.bw_wx = None
        self.bw_awx = None

    def _get_bandwidth_grid(self):
        # Log-space grid similar to your numpy code: np.logspace(-1, 0, 20)
        # Adjust range if necessary based on data standardization
        return torch.logspace(-1, 2, 20).to(self.device)

    def _gaussian_kernel_log_score(self, x, data, bandwidth):
        """
        Computes log(p(x)) for a set of query points 'x' against reference 'data'.
        Using Gaussian Kernel: K(u) = (1 / sqrt(2pi)) * exp(-0.5 * u^2)
        """
        N, D = data.shape
        M = x.shape[0]

        # 1. Compute Pairwise Euclidean Distances squared: ||x_i - data_j||^2
        # x: (M, D), data: (N, D) -> dists: (M, N)
        # Using efficient PyTorch cdist
        dists_sq = torch.cdist(x, data, p=2) ** 2

        # 2. Compute Log Kernel Values
        # log(K((x-y)/h)) = -0.5 * (d^2 / h^2) - log(h^D) - D/2 * log(2pi)
        log_h = torch.log(bandwidth)
        log_normalizer = -0.5 * D * np.log(2 * np.pi) - D * log_h

        log_kernels = -0.5 * (dists_sq / (bandwidth ** 2)) + log_normalizer

        # 3. Average over N reference points (in log domain)
        # log(1/N * sum(exp(log_kernels))) = logsumexp(log_kernels) - log(N)
        log_density = torch.logsumexp(log_kernels, dim=1) - np.log(N)

        return log_density

    def _select_bandwidth_cv(self, data, grid_points=20):
        """
        Selects optimal bandwidth using Leave-One-Out Cross-Validation (LOO-CV).
        Maximizes sum of log-likelihoods.
        """
        N, D = data.shape
        bandwidths = self._get_bandwidth_grid()
        best_ll = -float('inf')
        best_bw = bandwidths[0]

        # Optimization: Use a random subset for CV if N is huge (>10k) to save time
        if N > 2000:
            perm = torch.randperm(N)[:2000]
            eval_data = data[perm]
        else:
            eval_data = data

        for h in bandwidths:
            # Efficient LOO-CV Approximation:
            # Compute full density, then subtract the contribution of the point itself.
            # However, for speed/stability in high-dim, standard K-fold or hold-out is often preferred.
            # Here we implement a simple hold-out validation (train on 80%, val on 20%)
            # which is faster and often more robust than pure LOO for bandwidths.

            n_val = int(0.2 * eval_data.shape[0])
            train_sub = eval_data[n_val:]
            val_sub = eval_data[:n_val]

            with torch.no_grad():
                log_prob = self._gaussian_kernel_log_score(val_sub, train_sub, h)
                total_ll = log_prob.sum()

            if total_ll > best_ll:
                best_ll = total_ll
                best_bw = h

        return best_bw

    def fit(self, A, WX):
        """
        Fits the estimators by storing data and selecting bandwidths.
        Args:
            A: Tensor (N, dim_A)
            WX: Tensor (N, dim_W + dim_X)
        """
        # Ensure Inputs are Tensors
        if not torch.is_tensor(A): A = torch.tensor(A, dtype=torch.float32).to(self.device)
        if not torch.is_tensor(WX): WX = torch.tensor(WX, dtype=torch.float32).to(self.device)

        self.train_A = A
        self.train_WX = WX
        self.train_AWX = torch.cat([A, WX], dim=1)

        # 1. Fit Marginal P(A)
        self.bw_a = self._select_bandwidth_cv(self.train_A)

        # 2. Fit Marginal P(W, X)
        self.bw_wx = self._select_bandwidth_cv(self.train_WX)

        # 3. Fit Joint P(A, W, X)
        self.bw_awx = self._select_bandwidth_cv(self.train_AWX)

        return self

    def predict_ratio(self, A, WX, clip_min=1e-4, clip_max=100.0):
        """
        Computes w = p(A) * p(WX) / p(A,W,X)
        """
        if not torch.is_tensor(A): A = torch.tensor(A, dtype=torch.float32).to(self.device)
        if not torch.is_tensor(WX): WX = torch.tensor(WX, dtype=torch.float32).to(self.device)
        AWX = torch.cat([A, WX], dim=1)

        # Compute Log Probabilities
        # Note: We compute log-probs first for numerical stability
        log_p_a = self._gaussian_kernel_log_score(A, self.train_A, self.bw_a)
        log_p_wx = self._gaussian_kernel_log_score(WX, self.train_WX, self.bw_wx)
        log_p_awx = self._gaussian_kernel_log_score(AWX, self.train_AWX, self.bw_awx)

        # Ratio in log space: log(A) + log(WX) - log(AWX)
        log_ratio = log_p_a + log_p_wx - log_p_awx

        # Exponentiate and Clip
        ratio = torch.exp(log_ratio)
        ratio = torch.clamp(ratio, min=clip_min, max=clip_max)

        # Reshape to (N, 1) for broadcasting
        return ratio.view(-1, 1)


class HeterogeneousCausalKDEDensRatioTorch:
    """
    KDE-based density ratio estimator for Heterogeneous Proxy Causal Learning.
    Computes: r = [p(A, V) * p(V, X, W)] / [p(A, V, X, W) * p(V)]
    """
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Training reference data
        self.train_data = {}
        self.bandwidths = {}

    def _get_bandwidth_grid(self):
        return torch.logspace(-1, 0, 20).to(self.device)

    def _gaussian_kernel_log_score(self, x, data, bandwidth):
        """Computes log(p(x)) using Gaussian Kernel Density Estimation."""
        N, D = data.shape
        # Compute Pairwise Euclidean Distances squared
        dists_sq = torch.cdist(x, data, p=2) ** 2

        # log(K) normalizer
        log_h = torch.log(bandwidth)
        log_normalizer = -0.5 * D * np.log(2 * np.pi) - D * log_h

        log_kernels = -0.5 * (dists_sq / (bandwidth ** 2)) + log_normalizer

        # Log-sum-exp trick for averaging
        return torch.logsumexp(log_kernels, dim=1) - np.log(N)

    def _select_bandwidth_cv(self, data):
        """Bandwidth selection via hold-out validation (80/20 split)."""
        N, D = data.shape
        bandwidths = self._get_bandwidth_grid()
        best_ll = -float('inf')
        best_bw = bandwidths[0]

        eval_data = data[torch.randperm(N)[:10000]] if N > 10000 else data
        n_val = int(0.2 * eval_data.shape[0])
        train_sub = eval_data[n_val:]
        val_sub = eval_data[:n_val]

        for h in bandwidths:
            with torch.no_grad():
                log_prob = self._gaussian_kernel_log_score(val_sub, train_sub, h)
                total_ll = log_prob.sum()

            if total_ll > best_ll:
                best_ll = total_ll
                best_bw = h
        return best_bw

    def fit(self, A, V, XW):
        """
        Fits KDE for components of the heterogeneous ratio.
        XW is the concatenated tensor of (X, W).
        """
        if not torch.is_tensor(A): A = torch.tensor(A, dtype=torch.float32).to(self.device)
        if not torch.is_tensor(V): V = torch.tensor(V, dtype=torch.float32).to(self.device)
        if not torch.is_tensor(XW): XW = torch.tensor(XW, dtype=torch.float32).to(self.device)

        # Store joint sets
        self.train_data['v'] = V
        self.train_data['av'] = torch.cat([A, V], dim=1)
        self.train_data['vxw'] = torch.cat([V, XW], dim=1)
        self.train_data['avxw'] = torch.cat([A, V, XW], dim=1)

        # Select bandwidths for all 4 densities
        for key in self.train_data.keys():
            self.bandwidths[key] = self._select_bandwidth_cv(self.train_data[key])

        return self

    def predict_ratio(self, A, V, XW, clip_min=1e-4, clip_max=100.0):
        """
        Computes the heterogeneous density ratio.
        r = exp(log_p_av + log_p_vxw - log_p_avxw - log_p_v)
        """
        if not torch.is_tensor(A): A = torch.tensor(A, dtype=torch.float32).to(self.device)
        if not torch.is_tensor(V): V = torch.tensor(V, dtype=torch.float32).to(self.device)
        if not torch.is_tensor(XW): XW = torch.tensor(XW, dtype=torch.float32).to(self.device)

        # Form query tensors
        query_av = torch.cat([A, V], dim=1)
        query_vxw = torch.cat([V, XW], dim=1)
        query_avxw = torch.cat([A, V, XW], dim=1)

        # Compute Log Probabilities
        log_p_v = self._gaussian_kernel_log_score(V, self.train_data['v'], self.bandwidths['v'])
        log_p_av = self._gaussian_kernel_log_score(query_av, self.train_data['av'], self.bandwidths['av'])
        log_p_vxw = self._gaussian_kernel_log_score(query_vxw, self.train_data['vxw'], self.bandwidths['vxw'])
        log_p_avxw = self._gaussian_kernel_log_score(query_avxw, self.train_data['avxw'], self.bandwidths['avxw'])

        # Decomposed Ratio in log space
        log_ratio = log_p_av + log_p_vxw - log_p_avxw - log_p_v

        # Exponentiate and Clip
        ratio = torch.exp(log_ratio)
        ratio = torch.clamp(ratio, min=clip_min, max=clip_max)

        return ratio.view(-1, 1)

######### KLIEP Algorithm ############

class KLIEP:
    """
    Direct density estimation implementing the original KLIEP algorithm[cite: 5].
    """

    def __init__(self, max_iter=5000, num_params=[.1,.2], epsilon=1e-4, cv=3, sigmas=[.01,.1,.25,.5,.75,1], random_state=0, verbose=0, device='cpu'):
        self.max_iter = max_iter
        self.num_params = num_params
        self.epsilon = epsilon
        self.verbose = verbose
        self.sigmas = sigmas
        self.cv = cv
        self.random_state = random_state
        self.device = device

        # Internal state
        self._fitted = False
        self._phi_fitted = False
        self._alpha = None
        self._test_vectors = None
        self._sigma = None
        self._num_parameters = None

    def fit(self, X_train, X_test, alpha_0=None):
        """ Cross validation to select sigma as in the original paper (LCV)[cite: 8]. """
        if not torch.is_tensor(X_train): X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(X_test): X_test = torch.tensor(X_test, dtype=torch.float32, device=self.device)

        cv = self.cv
        chunk = int(X_test.shape[0] / float(cv))

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        # Shuffle X_test
        perm = torch.randperm(X_test.shape[0])
        X_test_shuffled = X_test[perm]

        j_scores = {}

        sigmas = self.sigmas if isinstance(self.sigmas, list) else [self.sigmas]
        num_params = self.num_params if isinstance(self.num_params, list) else [self.num_params]

        if len(sigmas) * len(num_params) > 1:
            if self.verbose > 0:
                print(f"Starting LCV for {len(sigmas) * len(num_params)} combinations...")

            for num_param in num_params:
                for sigma in sigmas:
                    j_scores[(num_param, sigma)] = torch.zeros(cv, device=self.device)
                    for k in range(1, cv + 1):
                        if self.verbose > 0:
                            print(f'Training: num_param: {num_param}, sigma: {sigma}    Fold: {k}/{cv}')

                        X_test_fold = X_test_shuffled[(k-1)*chunk : k*chunk, :]
                        j_scores[(num_param, sigma)][k-1] = self._fit(
                            X_train=X_train,
                            X_test=X_test_fold,
                            num_parameters=num_param,
                            sigma=sigma
                        )

                    mean_j = torch.mean(j_scores[(num_param, sigma)]).item()
                    j_scores[(num_param, sigma)] = mean_j

                    if self.verbose > 0:
                        print(f"Mean J-score for (num_param: {num_param}, sigma: {sigma}): {mean_j:.6f}")

            # Filter for finite values
            finite_scores = {k: v for k, v in j_scores.items() if torch.isfinite(torch.tensor(v))}
            if not finite_scores:
                warnings.warn('LCV failed to converge for all values of sigma.')
                return self

            best_params = max(finite_scores, key=finite_scores.get)
            self._num_parameters, self._sigma = best_params

            if self.verbose > 0:
                print(f"Best parameters found: num_param={self._num_parameters}, sigma={self._sigma} (J={finite_scores[best_params]:.6f})")
        else:
            self._sigma = sigmas[0]
            self._num_parameters = num_params[0]
            if self.verbose > 0:
                print(f"Single parameter set provided. Using sigma={self._sigma}, num_param={self._num_parameters}")

        if self.verbose > 0:
            print("Final fit on full dataset...")

        self._j = self._fit(X_train=X_train, X_test=X_test_shuffled, num_parameters=self._num_parameters, sigma=self._sigma)

        if self.verbose > 0:
            print(f"Fit complete. Final J-score: {self._j:.6f}")

        return self

    def _fit(self, X_train, X_test, num_parameters, sigma, alpha_0=None):
        if isinstance(num_parameters, float):
            num_parameters = int(X_test.shape[0] * num_parameters)

        self._select_param_vectors(X_test=X_test, sigma=sigma, num_parameters=num_parameters)

        X_train_reshaped = self._reshape_X(X_train)
        X_test_reshaped = self._reshape_X(X_test)

        if alpha_0 is None:
            alpha_0 = torch.ones((num_parameters, 1), device=self.device) / float(num_parameters)

        self._find_alpha(
            alpha_0=alpha_0,
            X_train=X_train_reshaped,
            X_test=X_test_reshaped,
            num_parameters=num_parameters,
            epsilon=self.epsilon,
            sigma=sigma
        )
        return self._calculate_j(X_test_reshaped, sigma=sigma)

    def _calculate_j(self, X_test, sigma):
        # [cite_start]Implementation of J score [cite: 10]
        preds = self.predict(X_test, sigma=sigma)
        return torch.log(preds).sum() / X_test.shape[0]

    def score(self, X_test):
        if not torch.is_tensor(X_test): X_test = torch.tensor(X_test, dtype=torch.float32, device=self.device)
        return self._calculate_j(X_test=self._reshape_X(X_test), sigma=self._sigma)

    @staticmethod
    def _reshape_X(X):
        if len(X.shape) != 3:
            return X.reshape((X.shape[0], 1, X.shape[1]))
        return X

    def _select_param_vectors(self, X_test, sigma, num_parameters):
        indices = torch.randperm(X_test.shape[0])[:num_parameters]
        self._test_vectors = X_test[indices, :].clone()
        self._phi_fitted = True

    def _phi(self, X, sigma=None):
        if sigma is None:
            sigma = self._sigma
        if self._phi_fitted:
            dist_sq = torch.sum((X - self._test_vectors)**2, dim=-1)
            return torch.exp(-dist_sq / (2 * sigma**2))
        raise Exception('Phi not fitted.')

    def _find_alpha(self, alpha_0, X_train, X_test, num_parameters, sigma, epsilon):
        A = self._phi(X_test, sigma)
        b = self._phi(X_train, sigma).sum(dim=0) / X_train.shape[0]
        b = b.reshape((num_parameters, 1))

        out = alpha_0.clone()
        for k in range(self.max_iter):
            grad = torch.mm(A.t(), 1.0 / torch.mm(A, out))
            out += epsilon * grad

            constraint_adj = (1.0 - torch.mm(b.t(), out)) / torch.mm(b.t(), b)
            out += b * constraint_adj

            out = torch.clamp(out, min=0)
            out /= torch.mm(b.t(), out)

        self._alpha = out
        self._fitted = True

    def predict(self, X, sigma=None):
        if not torch.is_tensor(X): X = torch.tensor(X, dtype=torch.float32, device=self.device)
        X = self._reshape_X(X)
        if not self._fitted:
            raise Exception('Not fitted!')

        res = torch.mm(self._phi(X, sigma=sigma), self._alpha)
        return res.reshape((X.shape[0],))

class CausalKLIEPDensRatio:
    def __init__(self,
                 subset_size = 2000,
                 max_iter=5000,
                 num_params=[.1,.2],
                 epsilon=1e-4,
                 cv=3,
                 sigmas=[.01,.1,.25,.5,.75,1],
                 random_state=0,
                 verbose=0,
                 device='cpu'):
        self.subset_size = subset_size
        self.dens_ratio_estimator = KLIEP(max_iter=max_iter,
                                          num_params=num_params,
                                          epsilon=epsilon,
                                          cv=cv,
                                          sigmas=sigmas,
                                          random_state=random_state,
                                          verbose=verbose,
                                          device=device)

    def fit(self, A: torch.Tensor, WX: torch.Tensor):
        nx = A.shape[0]
        self.subset_size = np.min([self.subset_size, nx]).item()
        dens_ratio_index = np.random.choice(A.shape[0], self.subset_size, replace=False)
        perm = torch.randperm(nx, device="cpu")
        A_tilde = A[perm]

        x_samples = torch.cat([A_tilde, WX], dim=1) # Numerator (p) [cite: 46]
        y_samples = torch.cat([A, WX], dim=1)      # Denominator (q) [cite: 46]
        self.dens_ratio_estimator.fit(y_samples[dens_ratio_index], x_samples[dens_ratio_index])

    def predict_ratio(self, A: torch.Tensor, WX: torch.Tensor):
        dens_ratio_tensor = self.dens_ratio_estimator.predict(torch.cat([A, WX], dim=1))
        return dens_ratio_tensor

######### RuLSIF Module #########

class CausalRuLSIFTorch:
    def __init__(
        self,
        sigma_range: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0],
        lambda_range: List[float] = [1e-3, 1e-2, 1e-1, 1.0],
        alpha: float = 0.0,
        kernel_num: int = 500, # Increased for better expressivity
        device: str = 'cuda'
    ):
        self.sigma_range = sigma_range
        self.lambda_range = lambda_range
        self.alpha = alpha
        self.kernel_num = kernel_num
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.centers = None
        self.theta = None
        self.opt_sigma = None
        self.opt_lambda = None

    def compute_kernel_gaussian(self, x, centers, sigma):
        """Standard Gaussian RBF Kernel evaluation [cite: 632]"""
        dist_sq = torch.cdist(x, centers)**2
        return torch.exp(-dist_sq / (2 * sigma**2))

    def fit(self, A: torch.Tensor, WX: torch.Tensor):
        nx = A.shape[0]
        # Aligning with causal logic: p(A)p(WX) / p(A, WX)
        perm = torch.randperm(nx, device=self.device)
        A_tilde = A[perm]

        x_samples = torch.cat([A_tilde, WX], dim=1)  # Numerator (p) [cite: 46]
        y_samples = torch.cat([A, WX], dim=1)       # Denominator (q) [cite: 46]

        ny = y_samples.shape[0]
        n_min = min(nx, ny) #[cite: 838]

        # Center selection from numerator samples [cite: 569]
        k_num = min(self.kernel_num, nx)
        center_indices = torch.randperm(nx, device=self.device)[:k_num]
        self.centers = x_samples[center_indices]

        best_score = float('inf')
        eye_k = torch.eye(k_num, device=self.device)

        for sigma in self.sigma_range:
            phi_x = self.compute_kernel_gaussian(x_samples, self.centers, sigma) # (nx, k)
            phi_y = self.compute_kernel_gaussian(y_samples, self.centers, sigma) # (ny, k)

            # H and h construction
            H = self.alpha * (phi_x.T @ phi_x / nx) + (1 - self.alpha) * (phi_y.T @ phi_y / ny)
            h = phi_x.mean(dim=0, keepdim=True).T # (k, 1)

            # Analytical LOOCV matrices
            phi_x_loocv = phi_x[:n_min].T
            phi_y_loocv = phi_y[:n_min].T

            for lam in self.lambda_range:
                # Regularization scaling as used in original search_sigma_and_lambda
                reg_val = lam * (ny - 1) / ny
                B = H + eye_k * reg_val

                try:
                    # Analytical LOOCV update using Sherman-Morrison logic
                    B_inv_phi_y = torch.linalg.solve(B, phi_y_loocv)
                    X_B_inv_X = phi_y_loocv * B_inv_phi_y
                    denom = ny - X_B_inv_X.sum(dim=0) # (n_min,)

                    # B0 and B1 represent the 'leave-one-out' solutions
                    h_ones = h @ torch.ones((1, n_min), device=self.device)
                    B0 = torch.linalg.solve(B, h_ones) + B_inv_phi_y * ((h.T @ B_inv_phi_y) / denom)
                    B1 = torch.linalg.solve(B, phi_x_loocv) + B_inv_phi_y * ((phi_x_loocv * B_inv_phi_y).sum(dim=0) / denom)

                    # Compute coefficients B2
                    B2 = (ny - 1) * (nx * B0 - B1) / (ny * (nx - 1))
                    B2 = torch.clamp(B2, min=0) # Non-negative coefficients

                    # Prediction components for squared loss
                    r_y = (phi_y_loocv * B2).sum(dim=0)
                    r_x = (phi_x_loocv * B2).sum(dim=0)

                    # Squared loss (directly related to negative PE-divergence)
                    score = (torch.mean(r_y**2) / 2 - torch.mean(r_x))

                    if score < best_score:
                        best_score = score
                        self.opt_sigma = sigma
                        self.opt_lambda = lam
                except RuntimeError:
                    continue

        # Final parameters with best sigma/lambda
        phi_x = self.compute_kernel_gaussian(x_samples, self.centers, self.opt_sigma)
        phi_y = self.compute_kernel_gaussian(y_samples, self.centers, self.opt_sigma)
        H = self.alpha * (phi_x.T @ phi_x / nx) + (1 - self.alpha) * (phi_y.T @ phi_y / ny)
        h = phi_x.mean(dim=0, keepdim=True).T
        self.theta = torch.linalg.solve(H + eye_k * self.opt_lambda, h).squeeze()
        self.theta = torch.clamp(self.theta, min=0)

        print(f"PyTorch RuLSIF: Found Opt Sigma = {self.opt_sigma:.4f}, Lambda = {self.opt_lambda:.4f}")

    def predict_ratio(self, A: torch.Tensor, WX: torch.Tensor) -> torch.Tensor:
        """Estimates the relative density ratio p(A,WX)/q_alpha(A,WX)"""
        samples = torch.cat([A, WX], dim=1)
        phi = self.compute_kernel_gaussian(samples, self.centers, self.opt_sigma)
        return phi @ self.theta


class PropensityCausalDensRatioTorch:
    """Density ratio ``r(a, w, x) = p(a) / p(a | w, x)`` for a **binary** treatment A in {0, 1}.

    This is the binary-treatment analogue of the continuous-A estimators (KDE / RuLSIF / normalizing
    flow) used in the abortion and grade-retention experiments, which are all inappropriate for a
    binary A. Instead of a density model we fit a **propensity classifier** ``p(a=1 | w, x)`` (a small
    MLP with BCE loss and a held-out early-stopping split), then form

        r_i = p(A_i) / p(A_i | W_i, X_i),   with   p(1)=mean(A),  p(0)=1-mean(A).

    A useful property: if the propensity is correct then ``E[r | A=a] = p(a) * E[1/p(a|W,X) | A=a] = 1``
    holds **exactly** for each treatment level, so (unlike the continuous-A case) the bridge needs no
    conditional re-normalisation -- the conditional identity is satisfied by construction. The class
    mirrors :class:`HighDimCausalDensityRatioEstimator`'s ``fit(A, W, X)`` / ``predict_ratio(A, W, X)``
    signature so it is a drop-in replacement in the proxy-causal pipeline.
    """

    def __init__(self, hidden_dims=(64, 64), lr=1e-3, weight_decay=1e-4, n_epochs=300,
                 batch_size=256, dropout=0.1, val_split=0.2, patience=25,
                 prob_clip=1e-3, ratio_clip=(1e-2, 50.0), device="cpu", verbose=False):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.val_split = val_split
        self.patience = patience
        self.prob_clip = prob_clip
        self.ratio_clip = ratio_clip
        self.device = device
        self.verbose = verbose
        self.net = None
        self.p_marginal = None   # P(A=1)

    @staticmethod
    def _as_tensor(t, device):
        if not torch.is_tensor(t):
            t = torch.tensor(np.asarray(t), dtype=torch.float32)
        return t.to(device).float()

    def _context(self, W, X):
        W = self._as_tensor(W, self.device)
        if X is not None:
            X = self._as_tensor(X, self.device)
            return torch.cat([W, X], dim=1)
        return W

    def _build_net(self, in_dim):
        layers, d = [], in_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(self.dropout)]
            d = h
        layers += [nn.Linear(d, 1)]           # logit of p(A=1 | w, x)
        return nn.Sequential(*layers).to(self.device)

    def fit(self, A, W, X=None):
        A = self._as_tensor(A, self.device).reshape(-1, 1)
        C = self._context(W, X)
        self.p_marginal = float(A.mean().item())

        n = A.shape[0]
        perm = torch.randperm(n)
        n_val = max(1, int(n * self.val_split))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        A_tr, C_tr, A_va, C_va = A[tr_idx], C[tr_idx], A[val_idx], C[val_idx]

        self.net = self._build_net(C.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        bce = nn.BCEWithLogitsLoss()

        best_val, best_state, waited = float("inf"), None, 0
        for epoch in range(self.n_epochs):
            self.net.train()
            for b in range(0, A_tr.shape[0], self.batch_size):
                idx = slice(b, b + self.batch_size)
                opt.zero_grad()
                loss = bce(self.net(C_tr[idx]), A_tr[idx])
                loss.backward(); opt.step()
            self.net.eval()
            with torch.no_grad():
                v = bce(self.net(C_va), A_va).item()
            if v < best_val - 1e-6:
                best_val, best_state, waited = v, {k: t.detach().clone() for k, t in self.net.state_dict().items()}, 0
            else:
                waited += 1
                if waited >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        if self.verbose:
            print(f"[Propensity] best val BCE={best_val:.4f}  P(A=1)={self.p_marginal:.3f}")
        return self

    def predict_prop(self, W, X=None):
        """P(A=1 | w, x), clipped to [prob_clip, 1-prob_clip]."""
        C = self._context(W, X)
        self.net.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net(C)).reshape(-1, 1)
        return p.clamp(self.prob_clip, 1.0 - self.prob_clip)

    def predict_ratio(self, A, W, X=None, clip_min=None, clip_max=None):
        A = self._as_tensor(A, self.device).reshape(-1, 1)
        p1 = self.predict_prop(W, X)                       # p(A=1 | w,x)
        p_cond = torch.where(A > 0.5, p1, 1.0 - p1)        # p(A_i | w,x)
        p_marg = torch.where(A > 0.5,
                             torch.full_like(A, self.p_marginal),
                             torch.full_like(A, 1.0 - self.p_marginal))
        r = p_marg / p_cond.clamp_min(self.prob_clip)
        lo = self.ratio_clip[0] if clip_min is None else clip_min
        hi = self.ratio_clip[1] if clip_max is None else clip_max
        return r.clamp(lo, hi).to("cpu")


########## NORMALIZING FLOW MODULES ##########

# The following method needs to be investigated more to make sure it correctly estimates the density ratios!!!
class HighDimCausalDensityRatioEstimator:
    def __init__(self,
                 features_dim: int,
                 context_dim: int = 0,
                 hidden_features=(128, 128), # Increased for 16-D latents
                 transforms=4,               # Increased for expressivity
                 flow_type="nsf",
                 activation=nn.Tanh,         # NEW: Customizable activation
                 lr=5e-4,                    # Slightly lower for stability
                 weight_decay=1e-5,          # NEW: L2 regularization
                 n_epochs=100,
                 batch_size=256,
                 device="cuda",
                 verbose=True):

        self.device = device
        self.verbose = verbose
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.features_dim = features_dim
        self.hidden_features = hidden_features
        self.transforms = transforms
        self.flow_type = flow_type.lower()
        self.activation = activation

        self.marginal_loss_history = []
        self.marginal_val_loss_history = []
        self.conditional_loss_history = []
        self.conditional_val_loss_history = []

        self.marginal_flow = None
        self.conditional_flow = None

    def _build_flow(self, features, context):
        """Builds the flow using the specified activation function."""
        if self.flow_type == "maf":
            return zuko.flows.MAF(features=features, context=context,
                                  hidden_features=self.hidden_features,
                                  transforms=self.transforms,
                                  activation=self.activation).to(self.device)
        elif self.flow_type == "realnvp":
            return zuko.flows.RealNVP(features=features, context=context,
                                      hidden_features=self.hidden_features,
                                      transforms=self.transforms,
                                      activation=self.activation).to(self.device)
        else: # nsf
            return zuko.flows.NSF(features=features, context=context,
                                  hidden_features=self.hidden_features,
                                  transforms=self.transforms,
                                  activation=self.activation).to(self.device)

    def fit(self, A, W, X=None, val_split=0.2): # <--- New Argument
        # 1. Prepare Data
        if not torch.is_tensor(A): A = torch.tensor(A, dtype=torch.float32).to(self.device)
        if not torch.is_tensor(W): W = torch.tensor(W, dtype=torch.float32).to(self.device)

        if X is not None:
            if not torch.is_tensor(X): X = torch.tensor(X, dtype=torch.float32).to(self.device)
            C = torch.cat([W, X], dim=1)
        else:
            C = W

        # --- NEW: Validation Split ---
        n_samples = A.shape[0]
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val

        # Random permutation to split data
        perm = torch.randperm(n_samples)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]

        A_train, C_train = A[train_idx], C[train_idx]
        A_val, C_val = A[val_idx], C[val_idx]

        # 2. Build Flows
        if self.marginal_flow is None:
            self.marginal_flow = self._build_flow(self.features_dim, 1) # Dummy context
        if self.conditional_flow is None:
            self.conditional_flow = self._build_flow(self.features_dim, C.shape[1])

        # 3. Train Marginal Flow p(A)
        # Dummy context (zeros) must also be split to match sizes
        dummy_train = torch.zeros(A_train.shape[0], 1).to(self.device)
        dummy_val = torch.zeros(A_val.shape[0], 1).to(self.device)

        print(f"Training Marginal Flow p(A) | Train: {n_train}, Val: {n_val}")
        self._train_flow(
            self.marginal_flow,
            A_train, dummy_train,
            A_val, dummy_val,   # Pass Validation Data
            self.marginal_loss_history,
            self.marginal_val_loss_history
        )

        # 4. Train Conditional Flow p(A|W)
        print(f"Training Conditional Flow p(A|W) | Train: {n_train}, Val: {n_val}")
        self._train_flow(
            self.conditional_flow,
            A_train, C_train,
            A_val, C_val,       # Pass Validation Data
            self.conditional_loss_history,
            self.conditional_val_loss_history
        )
        if self.verbose:
            # Auto-plot results
            self.plot_loss()

    def _train_flow(self, flow, train_data, train_context, val_data, val_context, train_hist, val_hist, verbose = True):
        optimizer = torch.optim.Adam(flow.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.n_epochs)

        # Train Loader
        train_dataset = TensorDataset(train_data, train_context)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Val Loader (Batch size can be larger for eval)
        val_dataset = TensorDataset(val_data, val_context)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size * 2, shuffle=False)

        for epoch in range(self.n_epochs):
            # --- TRAIN LOOP ---
            flow.train()
            epoch_loss = 0
            for x_batch, c_batch in train_loader:
                # Tiny noise for stability
                x_batch = x_batch + torch.randn_like(x_batch) * 1e-6

                optimizer.zero_grad()
                loss = -flow(c_batch).log_prob(x_batch).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(flow.parameters(), 100.0)
                optimizer.step()
                epoch_loss += loss.item()

            # Store average train loss
            train_hist.append(epoch_loss / len(train_loader))

            # --- VAL LOOP ---
            flow.eval()
            val_epoch_loss = 0
            with torch.no_grad():
                for x_val, c_val in val_loader:
                    v_loss = -flow(c_val).log_prob(x_val).mean()
                    val_epoch_loss += v_loss.item()

            # Store average val loss
            val_hist.append(val_epoch_loss / len(val_loader))

            scheduler.step()

            if self.verbose:
                # Optional: Print progress every 10 epochs
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}: Train NLL={train_hist[-1]:.3f}, Val NLL={val_hist[-1]:.3f}")

    def plot_loss(self):
        """Helper to visualize overfitting"""
        plt.figure(figsize=(12, 5))

        # Plot Marginal
        plt.subplot(1, 2, 1)
        plt.plot(self.marginal_loss_history, label="Train")
        plt.plot(self.marginal_val_loss_history, label="Validation", linestyle="--")
        plt.title("Marginal p(A) Loss")
        plt.xlabel("Epoch")
        plt.ylabel("NLL")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot Conditional
        plt.subplot(1, 2, 2)
        plt.plot(self.conditional_loss_history, label="Train")
        plt.plot(self.conditional_val_loss_history, label="Validation", linestyle="--")
        plt.title("Conditional p(A|W) Loss")
        plt.xlabel("Epoch")
        plt.ylabel("NLL")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.show()

    def predict_ratio(self, A, W, X=None, clip_min=1e-4, clip_max=100.0):
        # (Same as before)
        self.marginal_flow.eval()
        self.conditional_flow.eval()
        with torch.no_grad():
            if not torch.is_tensor(A): A = torch.tensor(A, dtype=torch.float32).to(self.device)
            if not torch.is_tensor(W): W = torch.tensor(W, dtype=torch.float32).to(self.device)

            if X is not None:
                if not torch.is_tensor(X): X = torch.tensor(X, dtype=torch.float32).to(self.device)
                C = torch.cat([W, X], dim=1)
            else:
                C = W

            dummy_context = torch.zeros(A.shape[0], 1).to(self.device)

            log_marg = self.marginal_flow(dummy_context).log_prob(A)
            log_cond = self.conditional_flow(C).log_prob(A)

            ratio = torch.exp(log_marg - log_cond)
            return torch.clamp(ratio, min=clip_min, max=clip_max)


class AnchoredATTDensRatio:
    """
    Builds the ATT density ratio from an already-fitted ATE density ratio estimator.

    If base estimator returns:
        r_ate(a, wx) ~= p(a) p(wx) / p(a, wx) = p(a) / p(a | wx),
    then for a fixed anchor a':
        r_att(a, a'; wx) = p(wx | a') / p(wx | a)
                         = r_ate(a, wx) / r_ate(a', wx).
    """
    def __init__(
        self,
        base_ratio_estimator,
        eps: float = 1e-6,
        clip_min: float = 1e-4,
        clip_max: float = 100.0,
    ):
        self.base = base_ratio_estimator
        self.eps = eps
        self.clip_min = clip_min
        self.clip_max = clip_max

    def fit(self, A: torch.Tensor, WX: torch.Tensor):
        self.base.fit(A, WX)
        return self

    def predict_ratio(
        self,
        A_obs: torch.Tensor,
        WX: torch.Tensor,
        A_anchor: torch.Tensor,
    ):
        """
        Parameters
        ----------
        A_obs : (n, d_a)
            Observed treatments in the training set.
        WX : (n, d_wx)
            Outcome-side conditioning variables. Use W in the no-X case,
            or torch.cat([X, W], dim=1) in the with-X case.
        A_anchor : (d_a,) or (1, d_a)
            Fixed anchor treatment a'.

        Returns
        -------
        r_att : (n, 1)
            ATT density ratios p(WX|a') / p(WX|A_i).
        r_obs : (n, 1)
            Base ATE ratios evaluated at observed A_i.
        r_anchor : (n, 1)
            Base ATE ratios evaluated at repeated anchor a'.
        """
        if not torch.is_tensor(A_obs):
            A_obs = torch.tensor(A_obs, dtype=torch.float32)
        if not torch.is_tensor(WX):
            WX = torch.tensor(WX, dtype=torch.float32)
        if not torch.is_tensor(A_anchor):
            A_anchor = torch.tensor(A_anchor, dtype=torch.float32)

        A_obs = A_obs.to(torch.float32)
        WX = WX.to(torch.float32)
        A_anchor = A_anchor.to(torch.float32)

        if A_anchor.ndim == 1:
            A_anchor = A_anchor.unsqueeze(0)

        if A_anchor.shape[0] == 1:
            A_anchor_rep = A_anchor.expand(A_obs.shape[0], -1)
        elif A_anchor.shape[0] == A_obs.shape[0]:
            A_anchor_rep = A_anchor
        else:
            raise ValueError(
                f"A_anchor must have shape (d_a,) or (1, d_a) or (n, d_a), "
                f"but got {tuple(A_anchor.shape)}."
            )

        r_obs = self.base.predict_ratio(A_obs, WX).reshape(-1, 1)
        r_anchor = self.base.predict_ratio(A_anchor_rep, WX).reshape(-1, 1)

        r_att = r_obs / (r_anchor + self.eps)
        r_att = torch.clamp(r_att, min=self.clip_min, max=self.clip_max)

        return r_att, r_obs, r_anchor

#### Old Codes

# class CausalRuLSIFTorch:
#     def __init__(self,
#                  sigma_range=[0.1, 0.3, 0.5, 0.7, 1, 1.5, 2],
#                  lambda_range = [1e-1, 1e-2, 1e-3, 1e-4],
#                  alpha = 0,
#                  kernel_num = 100,
#                  device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
#                 ):
#         self.sigma_range = sigma_range
#         self.lambda_range = lambda_range
#         self.kernel_num = kernel_num
#         self.alpha = alpha
#         self.densratio_obj = None
#         self.device = device

#     def fit(self, A, WX):
#         print("density ratio estimation using RuLSIF...")
#         A_np = A.detach.cpu.numpy
#         WX_np = WX.detach.cpu.numpy

#         perm = np.random.permutation(np.arange(A_np.shape[0]))
#         A_tilde = A_np[perm]

#         x = np.hstack((A_tilde, WX_np))
#         y = np.hstack((A_np, WX_np))
#         densratio_obj = densratio(x, y,
#                                   alpha=self.alpha,
#                                   sigma_range=self.sigma_range,
#                                   lambda_range = self.lambda_range,
#                                   kernel_num = self.kernel_num,
#                                   verbose = False)
#         self.densratio_obj = densratio_obj

#     def predict_ratio(self, A, WX):
#         A_np = A.detach.cpu.numpy
#         WX_np = WX.detach.cpu.numpy
#         y = np.hstack((A_np, WX_np))
#         pred = self.densratio_obj.compute_density_ratio(y)
#         pred_torch = torch.Tensor(pred).to(self.device)
#         return pred_torch

# class CausalDensityRatioEstimator:
#     def __init__(self,
#                  marginal_hidden_dims = (64, 64),
#                  conditional_hidden_dims = (64, 64),
#                  marginal_n_layers=4,
#                  conditional_n_layers=4,
#                  lr=1e-4,
#                  n_epochs=500,
#                  batch_size = 512,
#                  weight_decay = 1e-5,
#                 #  activation = "tanh",
#                  device="cuda"):
#         """
#         Estimates w(A, W, X) = p(A) / p(A | W, X) using two Normalizing Flows.
#         """
#         self.device = device

#         # 1. Marginal Model: p(A)
#         # Condition size is 0 because it's unconditional
#         self.marginal_flow = CNF(
#             DEVICE=device,
#             n_layers=marginal_n_layers,
#             hidden=marginal_hidden_dims,
#             batch_size=batch_size,
#             n_epochs=n_epochs,
#             lr=lr,
#             weight_decay = weight_decay,
#             # activation = activation
#         )

#         # 2. Conditional Model: p(A | W, X)
#         # Condition size is dim(W) + dim(X)
#         self.conditional_flow = CNF(
#             DEVICE=device,
#             n_layers=conditional_n_layers,
#             hidden=conditional_hidden_dims,
#             batch_size=batch_size,
#             n_epochs=n_epochs,
#             lr=lr,
#             weight_decay = weight_decay,
#             # activation = activation
#         )

#     def fit(self, A, W, X=None):
#         """
#         Trains both flows.
#         A, W, X: torch Tensors or numpy arrays
#         """
#         # Ensure inputs are tensors on the right device
#         if not torch.is_tensor(A): A = torch.tensor(A, dtype=torch.float32).to(self.device)
#         if not torch.is_tensor(W): W = torch.tensor(W, dtype=torch.float32).to(self.device)

#         # Prepare Conditions
#         if X is not None:
#             if not torch.is_tensor(X): X = torch.tensor(X, dtype=torch.float32).to(self.device)
#             C_joint = torch.cat([W, X], dim=1)
#         else:
#             C_joint = W

#         print("Training Marginal Flow p(A)...")
#         # For marginal p(A), condition is None
#         self.marginal_flow.fit(A, C=None)

#         print("Training Conditional Flow p(A | W, X)...")
#         self.conditional_flow.fit(A, C=C_joint)

#     def predict_ratio(self, A, W, X=None, clip_min=1e-4, clip_max=100.0):
#         """
#         Computes p(A) / p(A | W, X)
#         """
#         # Ensure eval mode
#         self.marginal_flow.nf.eval
#         self.conditional_flow.nf.eval

#         with torch.no_grad:
#             if not torch.is_tensor(A): A = torch.tensor(A, dtype=torch.float32).to(self.device)
#             if not torch.is_tensor(W): W = torch.tensor(W, dtype=torch.float32).to(self.device)

#             if X is not None:
#                 if not torch.is_tensor(X): X = torch.tensor(X, dtype=torch.float32).to(self.device)
#                 C_joint = torch.cat([W, X], dim=1)
#             else:
#                 C_joint = W

#             # 1. Compute Log Probs
#             # Note: We access.nf.log_prob directly for numerical stability
#             # instead of dividing standard probabilities
#             log_p_marginal = self.marginal_flow.nf.log_prob(A, None)
#             log_p_conditional = self.conditional_flow.nf.log_prob(A, C_joint)

#             # 2. Compute Ratio in Log Space
#             # log(p(A) / p(A|C)) = log(p(A)) - log(p(A|C))
#             log_ratio = log_p_marginal - log_p_conditional

#             # 3. Exponentiate and Clip
#             ratio = torch.exp(log_ratio)
#             ratio = torch.clamp(ratio, min=clip_min, max=clip_max)

#             return ratio


# def gen_network(n_inputs, n_outputs, hidden=(10,), activation='tanh'):
#     """
#     Generates a neural network with specified input, output, hidden layers, and activation function.

#     Args:
#         n_inputs (int): Number of input features.
#         n_outputs (int): Number of output features.
#         hidden (tuple, optional): Number of neurons in hidden layers. Default is (10,).
#         activation (str, optional): Activation function. Possible values: 'tanh', 'relu', 'leakyrelu'. Default is 'tanh'.

#     Returns:
#         model (nn.Sequential): The constructed neural network.
#     """
#     model = nn.Sequential
#     for i in range(len(hidden)):

#         # add layer
#         if i == 0:
#             alayer = nn.Linear(n_inputs, hidden[i])
#         else:
#             alayer = nn.Linear(hidden[i-1], hidden[i])
#         model.append(alayer)
#         model.append(nn.Dropout(0.2))

#         # add activation
#         if activation == 'tanh':
#             act = nn.Tanh
#         elif activation == 'relu':
#             act = nn.ReLU
#         elif activation == 'leakyrelu':
#             act = nn.LeakyReLU
#         else:
#             act = nn.ReLU
#         model.append(act)

#     # output layer
#     model.append(nn.Linear(hidden[-1], n_outputs))

#     return model


# class NormalizingFlow(nn.Module):
#     """
#     Normalizing Flow model interface.
#     """

#     def __init__(self, layers, prior):
#         """
#         Initializes the Normalizing Flow model with the specified layers and prior distribution.

#         Args:
#             layers (list): List of `InvertibleLayer` objects.
#             prior (torch.distributions.Distribution): The prior distribution for the latent variable.
#         """
#         super(NormalizingFlow, self).__init__

#         self.layers = nn.ModuleList(layers)
#         self.prior = prior

#     def log_prob(self, X, C):
#         """
#         Calculates the loss function.

#         Args:
#             X (Tensor): torch.Tensor of shape [batch_size, var_size] Input sample to transform.
#             C (Tensor): torch.Tensor of shape [batch_size, cond_size] or None Condition values.

#         Returns:
#             log_likelihood (Tensor): Calculated log likelihood.
#         """
#         log_likelihood = None

#         for layer in self.layers:
#             X, change = layer.f(X, C)
#             if log_likelihood is not None:
#                 log_likelihood = log_likelihood + change
#             else:
#                 log_likelihood = change
#         log_likelihood = log_likelihood + self.prior.log_prob(X)

#         return log_likelihood

#     def sample(self, C):
#         """
#         Sample new objects based on the give conditions.

#         Args:
#             C (Tensor): torch.Tensor of shape [batch_size, cond_size] or Int Condition values or number of samples to generate.

#         Returns:
#             X (Tensor): torch.Tensor of shape [batch_size, var_size] Generated sample.
#         """
#         if type(C) == type(1):
#             n = C
#             C = None
#         else:
#             n = len(C)

#         X = self.prior.sample((n,))
#         for layer in self.layers[::-1]:
#             X = layer.g(X, C)

#         return X


# class CNFLayer(nn.Module):
#     """
#     Invertible RealNVP function for RealNVP normalizing flow model.
#     """

#     def __init__(self, DEVICE, var_size, cond_size, mask, hidden=(10,), activation='tanh'):
#         """
#         Initializes the Normalizing Flow model.

#         Args:
#             DEVICE (str): Device to run the model ('cpu' or 'cuda').
#             var_size (int): Input vector size.
#             cond_size (int): Conditional vector size.
#             mask (Tensor): Tensor of {0, 1} to separate input vector components into two groups. Example: [0, 1, 0, 1].
#             hidden (tuple, optional): Number of neurons in hidden layers. Example: (10, 20, 15).
#             activation (str, optional): Activation function of the hidden neurons. Possible values: 'tanh', 'relu'.
#         """
#         super(CNFLayer, self).__init__

#         self.mask = mask.to(DEVICE)
#         self.nn_t = gen_network(var_size + cond_size,
#                                 var_size, hidden, activation)
#         self.nn_s = gen_network(var_size + cond_size,
#                                 var_size, hidden, activation)

#     def f(self, X, C=None):
#         """
#         Implementation of forward pass.

#         Args:
#             X (Tensor): torch.Tensor of shape [batch_size, var_size] Input sample to transform.
#             C (Tensor, optional): torch.Tensor of shape [batch_size, cond_size] or None Condition values.


#         Returns:
#             new_X (Tensor): torch.Tensor of shape [batch_size, var_size] Transformed X.
#             log_det (Tensor): torch.Tensor of shape [batch_size] Logarithm of the Jacobian determinant.
#         """
#         if C is not None:
#             XC = torch.cat((X * self.mask[None, :], C), dim=1)
#         else:
#             XC = X * self.mask[None, :]

#         T = self.nn_t(XC)
#         S = self.nn_s(XC)

#         X_new = (X * torch.exp(S) + T) * \
#             (1 - self.mask[None, :]) + X * self.mask[None, :]
#         log_det = (S * (1 - self.mask[None, :])).sum(dim=-1)
#         return X_new, log_det

#     def g(self, X, C=None):
#         """
#         Implementation of backward (inverse) pass.

#         Args:
#             X (Tensor): torch.Tensor of shape [batch_size, var_size] Input sample to transform.
#             C (Tensor, optional): torch.Tensor of shape [batch_size, cond_size] or None Condition values.

#         Returns:
#             new_X (Tensor): torch.Tensor of shape [batch_size, var_size] Transformed X.
#         """
#         if C is not None:
#             XC = torch.cat((X * self.mask[None, :], C), dim=1)
#         else:
#             XC = X * self.mask[None, :]

#         T = self.nn_t(XC)
#         S = self.nn_s(XC)

#         X_new = ((X - T) * torch.exp(-S)) * \
#             (1 - self.mask[None, :]) + X * self.mask[None, :]
#         return X_new


# class CNF:
#     """
#     RealNVP-based normalizing flow model.
#     """

#     def __init__(self, n_layers=8, hidden=(10,), activation='tanh',
#                  batch_size=32, n_epochs=10, lr=0.0001, weight_decay=0,
#                  DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')):
#         """
#         Initializes the RealNVP normalizing flow model.

#         Args:
#             DEVICE (str): Device to run the model ('cpu' or 'cuda').
#             n_layers (int): Number of RealNVP layers.
#             hidden (tuple, optional): Number of neurons in hidden layers. Example: (10,).
#             activation (str, optional): Activation function of the hidden neurons. Possible values: 'tanh', 'relu'.
#             batch_size (int, optional): Batch size. Default is 32.
#             n_epochs (int, optional): Number of epoches for fitting the model. Default is 10.
#             lr (float, optional): Learning rate. Default is 0.0001.
#             weight_decay (float, optional): L2 regularization coefficient. Default is 0.
#         """

#         self.n_layers = n_layers
#         self.hidden = hidden
#         self.activation = activation
#         self.batch_size = batch_size
#         self.n_epochs = n_epochs
#         self.lr = lr
#         self.weight_decay = weight_decay
#         self.DEVICE = DEVICE

#         self.prior = None
#         self.nf = None
#         self.opt = None

#         self.loss_history = []
#         self.val_loss = []

#     def _model_init(self, X, C):
#         """
#         Trains the model on the given data.

#         Args:
#             X (Tensor): Input sample tensor.
#             C (Tensor): Condition tensor.
#         """

#         var_size = X.shape[1]
#         if C is not None:
#             cond_size = C.shape[1]
#         else:
#             cond_size = 0

#         # init prior
#         if self.prior is None:
#             self.prior = torch.distributions.MultivariateNormal(torch.zeros(var_size, device=self.DEVICE),
#                                                                 torch.eye(var_size, device=self.DEVICE))
#         # init NF model and optimizer
#         if self.nf is None:

#             layers = []
#             for i in range(self.n_layers):
#                 alayer = CNFLayer(DEVICE=self.DEVICE, var_size=var_size,
#                                   cond_size=cond_size,
#                                   mask=((torch.arange(var_size) + i) % 2),
#                                   hidden=self.hidden,
#                                   activation=self.activation)
#                 layers.append(alayer)

#             self.nf = NormalizingFlow(
#                 layers=layers, prior=self.prior).to(self.DEVICE)
#             self.opt = torch.optim.Adam(self.nf.parameters,
#                                         lr=self.lr,
#                                         weight_decay=self.weight_decay)

#     def fit(self, X, C=None):
#         """
#         Fit the model.

#         Args:
#             X (ndarray): Input sample to transform.
#             C (ndarray, optional): Condition values.
#         """

#         # model init
#         self._model_init(X, C)

#         # numpy to tensor, tensor to dataset
#         if C is not None:
#             dataset = TensorDataset(X, C)
#         else:
#             dataset = TensorDataset(X)

#         for epoch in range(self.n_epochs):
#             for batch in DataLoader(dataset, batch_size=self.batch_size, shuffle=True):
#                 self.nf.train
#                 X_batch = batch[0].to(self.DEVICE)

#                 X_batch += torch.randn_like(X_batch) * 0.001

#                 if C is not None:
#                     C_batch = batch[1].to(self.DEVICE)
#                 else:
#                     C_batch = None

#                 # calculate loss
#                 loss = -self.nf.log_prob(X_batch, C_batch).mean

#                 # optimization step
#                 self.opt.zero_grad
#                 loss.backward
#                 self.opt.step

#                 # caiculate and store loss
#                 self.loss_history.append(loss.detach.cpu.item)

#     def pob(self, X, C=None):
#         """
#         Sample new objects based on the give conditions.

#         Args:
#             X (ndarray): Condition values or number of samples to generate.
#             C (ndarray, optional): Condition values or number of samples to generate.

#         Returns:
#             X (ndarray): Generated sample.
#         """
#         X, C = X.to(self.DEVICE), C.to(self.DEVICE)
#         self.nf.eval
#         log_pob = self.nf.log_prob(X, C)
#         pob = torch.exp(log_pob).cpu.detach

#         return pob

#     def sample(self, C=100):
#         """
#         Sample new objects based on the give conditions.

#         Args:
#             C (int, optional): Condition values or number of samples to generate.

#         Returns:
#             X (ndarray): Generated sample.
#         """
#         if type(C) != type(1):
#             C = torch.tensor(C, dtype=torch.float32, device=self.DEVICE)
#         X = self.nf.sample(C).cpu.detach.numpy
#         return X



#### The following is taken from
# def numpy_conversion(func):
#     """
#     Decorator to convert torch tensors to numpy arrays before calling the function.

#     Args:
#         func (callable): The function to be decorated.

#     Returns:
#         callable: The wrapper function that converts tensors to numpy arrays.
#     """
#     def wrapper(*args, **kwargs):
#         processed_args = []
#         for arg in args:
#             if isinstance(arg, torch.Tensor):
#                 arg = arg.cpu.detach.numpy
#             processed_args.append(arg)
#         return func(*processed_args, **kwargs)
#     return wrapper


# class CausalKDEDensRatioTorch:

#     def __init__(self,
#                 device = torch.device('cuda' if torch.cuda.is_available else 'cpu')):
#         self.device = device

#     @numpy_conversion
#     def kde_f_a(self, A):
#         """
#         Estimates the density function of the treatment variable A using Kernel Density Estimation (KDE).

#         Args:
#             A (np.ndarray): Treatment variable.

#         Returns:
#             np.ndarray: The estimated density function values for A.
#         """
#         bandwidths = {'bandwidth': np.logspace(-1, 0, 20)}
#         grid_a = GridSearchCV(KernelDensity, bandwidths)
#         grid_a.fit(A)
#         bandwidth_est_wx = grid_a.best_estimator_.bandwidth

#         kde_a = KernelDensity(kernel='gaussian', bandwidth=bandwidth_est_wx)
#         kde_a.fit(A)

#         # f_a = np.exp(kde_a.score_samples(A))
#         # return f_a
#         self.kde_a = kde_a

#     @numpy_conversion
#     def kde_gps(self, A, WX):
#         """
#         Estimates the Generalized Propensity Score (GPS) using Kernel Density Estimation (KDE).

#         Args:
#             A (np.ndarray): Treatment variable.
#             W (np.ndarray): Outcome proxy variable.
#             X (np.ndarray, optional): Additional backdoor variables. Defaults to None.

#         Returns:
#             np.ndarray: The estimated Generalized Propensity Score (GPS).
#         """
#         AWX = np.concatenate([A, WX], axis=1)

#         bandwidths = {'bandwidth': np.logspace(-1, 0, 20)}

#         grid_wx = GridSearchCV(KernelDensity, bandwidths)
#         grid_wx.fit(WX)
#         bandwidth_est_wx = grid_wx.best_estimator_.bandwidth

#         grid_awx = GridSearchCV(KernelDensity, bandwidths)
#         grid_awx.fit(AWX)
#         bandwidth_est_awx = grid_awx.best_estimator_.bandwidth

#         kde_wx = KernelDensity(kernel='gaussian', bandwidth=bandwidth_est_wx)
#         kde_wx.fit(WX)

#         # f_wx = np.exp(kde_wx.score_samples(WX))

#         kde_awx = KernelDensity(kernel='gaussian', bandwidth=bandwidth_est_awx)
#         kde_awx.fit(AWX)

#         # f_awx = np.exp(kde_awx.score_samples(AWX))
#         # gps = f_wx/f_awx
#         # return gps
#         self.kde_awx = kde_awx
#         self.kde_wx = kde_wx

#     @numpy_conversion
#     def predict_ratio(self, A, WX):
#         AWX = np.concatenate([A, WX], axis=1)
#         f_a = np.exp(self.kde_a.score_samples(A))
#         f_wx = np.exp(self.kde_wx.score_samples(WX))
#         f_awx = np.exp(self.kde_awx.score_samples(AWX))
#         dens_ratio = f_a * f_wx / f_awx
#         dens_ratio_torch = torch.Tensor(dens_ratio).to(self.device)
#         return dens_ratio_torch

#     def fit(self, A, WX):
#         self.kde_f_a(A)
#         self.kde_gps(A, WX)


