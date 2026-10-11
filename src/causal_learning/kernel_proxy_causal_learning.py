import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm import tqdm

from typing import Callable, Tuple, Optional, Union, Dict
import numpy as np

import os
import sys
# Get the path of the current script, go up one level to the root
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from torch_utils.ml_utils import lambda_objective_loocv, cme_lambda_objective_loocv
from torch_utils.linear_algebra import make_psd, columns_mean_excluding_self, remove_diagonal_elements
from torch_utils.kernel_utils import RBF
from torch_utils.scalers import TorchIdentityTransformer

##################################################################################
##################### DOSE RESPONSE ALGORITHMS ###################################
##################################################################################

class KernelAlternativeProxyDoseResponse(BaseEstimator, RegressorMixin, nn.Module):

    def __init__(self,
                 kernel_A: Callable,
                 kernel_W: Callable,
                 kernel_Z: Callable,
                 kernel_X: Optional[Callable] = None,
                 lambda1_: float = 0.1,
                 eta: float = 0.1,
                 lambda2_: float = 0.1,
                 device = None,
                 **kwargs) -> None:

        nn.Module.__init__(self)
        # --- Store Parameters ---
        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_X = kernel_X if kernel_X is not None else RBF() # Default RBF

        #... (Kernel setting logic remains the same, assuming it checks/sets params)...
        # NOTE: Using a simplified check for callable kernels now:
        if not isinstance(self.kernel_X, Callable) or not isinstance(kernel_Z, Callable) or not isinstance(kernel_W, Callable) or not isinstance(kernel_A, Callable):
            raise TypeError("Kernels for A, Z, W, or X must be callable (nn.Module).")

        self.lambda1_, self.eta, self.lambda2_ = lambda1_, eta, lambda2_
        self.optimize_lambda_parameters = kwargs.pop('optimize_lambda_parameters', True)
        self.optimize_eta_parameter = kwargs.pop('optimize_eta_parameter', True)
        self.lambda_optimization_range = kwargs.pop('lambda_optimization_range', (1e-7, 1.0))
        self.eta_optimization_range = kwargs.pop('eta_optimization_range', (1e-7, 1.0))
        self.regularization_grid_points = kwargs.pop('regularization_grid_points', 25)
        self.make_psd_eps = kwargs.pop('make_psd_eps', 1e-7)
        self.stage1_perc = kwargs.pop('stage1_perc', 0.5)
        self.label_variance_in_lambda_opt = kwargs.pop('label_variance_in_lambda_opt', 0.0)
        self.label_variance_in_eta_opt = kwargs.pop('label_variance_in_eta_opt', 0.0)

        # --- Model Components (Will be populated in fit) ---
        self.ATrain, self.WTrain, self.XTrain, self.ZTrain = [None] * 4
        self.alpha = None
        self.B, self.B_bar = None, None
        self.third_stage_KRR_weights = None
        self.train_indices, self.stage1_idx, self.stage2_idx = [None] * 3
        self.K_ZZ = None
        self.ones_divided_by_m = None

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _eta_objective(eta, L, L_sub, M, N, L2, M2, stage1_data_size, label_variance_in_eta_opt = 0.0, make_psd_eps = 1e-9):
        stage2_data_size = L.shape[0] - 1
        alpha = torch.linalg.solve(make_psd(L / stage2_data_size + eta * N, make_psd_eps), M)
        cost = ((1 / stage1_data_size) * (alpha.T @ make_psd(L2, make_psd_eps) @ alpha) - 2 * (alpha.T @ M2))
        cost += label_variance_in_eta_opt * (2 / stage2_data_size) * torch.trace(torch.linalg.solve(make_psd(L + stage2_data_size * eta * N, make_psd_eps), L))
        return cost.reshape(())

    @staticmethod
    def _predict_structural_function(alpha: torch.Tensor,
                                     B: torch.Tensor,
                                     B_bar: torch.Tensor,
                                     third_stage_KRR_weights: torch.Tensor,
                                     K_ATraina: torch.Tensor,
                                     K_ATildea: torch.Tensor,
                                     ones_divided_by_m: torch.Tensor) -> torch.Tensor:
        """
        Predict the structural function.

        Parameters:
        - alpha (torch.Tensor): Coefficient array.
        - B (torch.Tensor): Matrix B from second stage.
        - B_bar (torch.Tensor): Matrix B_bar from second stage.
        - third_stage_KRR_weights (torch.Tensor): Weights from third stage kernel ridge regression.
        - K_ATraina (torch.Tensor): Kernel matrix between training set A and a test point.
        - K_ATildea (torch.Tensor): Kernel matrix between stage 2 set A and a test point.
        - ones_divided_by_m (torch.Tensor): Array of ones divided by stage 2 data size.

        Returns:
        - torch.Tensor: Predicted values.
        """
        pred = (alpha[:-1].T @ ((B.T @ (third_stage_KRR_weights @ K_ATraina)) * K_ATildea))
        pred += (alpha[-1] * ((B_bar.T @ (third_stage_KRR_weights @ K_ATraina)) * K_ATildea) @ ones_divided_by_m)
        return pred

    def _predict_bridge_func(self, A_test : torch.Tensor, Z_test : torch.Tensor, X_test = None):
        if A_test.ndim != 2:
            A_test = A_test.reshape(-1, 1)
        if Z_test.ndim != 2:
            Z_test = Z_test.reshape(-1, 1)
        K_ZZTest = self.kernel_Z(self.ZTrain[self.stage1_idx, :], Z_test)
        K_ATildeATest = self.kernel_A(self.ATrain[self.stage2_idx, :], A_test)
        alpha, B, B_bar = self.alpha, self.B, self.B_bar
        ones_divided_by_m = self.ones_divided_by_m
        bridge_function = torch.vstack([alpha[:-1].T @ ((B.T @ K_ZZTest) * K_ATildeATest[:, jj].reshape(-1, 1))
                                        + alpha[-1] * ones_divided_by_m.T @ ((B_bar.T @ K_ZZTest) * K_ATildeATest[:, jj].reshape(-1,1))
                                        for jj in range(K_ATildeATest.shape[1])])
        return bridge_function

    def _predict_density_ratio(self, A_test: torch.Tensor, W_test: torch.Tensor, X_test : Optional[torch.Tensor] = None):
        if A_test.ndim != 2:
            A_test = A_test.reshape(-1, 1)
        if W_test.ndim != 2:
            W_test = W_test.reshape(-1, 1)
        K_WWTest = self.kernel_W(self.WTrain[self.stage1_idx, :], W_test)
        K_AATest = self.kernel_A(self.ATrain[self.stage1_idx, :], A_test)
        if X_test is None:
            K_XXTest = torch.ones_like(K_AATest)
        K_ATildeATest = self.kernel_A(self.ATrain[self.stage2_idx, :], A_test)
        B_test = torch.linalg.solve(make_psd(self.stage1_ridge_weights, self.make_psd_eps), (K_WWTest * K_XXTest * K_AATest))
        ones_divided_by_m = self.ones_divided_by_m
        dens_ratio = self.alpha[:-1].T @ ((self.B.T @ self.K_ZZ @ B_test) * K_ATildeATest) + self.alpha[-1] * ones_divided_by_m.T @ ((self.B_bar.T @ self.K_ZZ @ B_test) * K_ATildeATest)
        return dens_ratio.T

    def fit(self,
            AZWX: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
            Y: torch.Tensor) -> None:
        kernel_A, kernel_W, kernel_Z, kernel_X = self.kernel_A, self.kernel_W, self.kernel_Z, self.kernel_X
        lambda1_, eta, lambda2_ = self.lambda1_, self.eta, self.lambda2_
        optimize_lambda_parameters = self.optimize_lambda_parameters
        optimize_eta_parameter = self.optimize_eta_parameter
        lambda_optimization_range = self.lambda_optimization_range
        eta_optimization_range = self.eta_optimization_range
        stage1_perc = self.stage1_perc
        regularization_grid_points = self.regularization_grid_points
        make_psd_eps = self.make_psd_eps
        label_variance_in_lambda_opt = self.label_variance_in_lambda_opt
        label_variance_in_eta_opt = self.label_variance_in_eta_opt
        dtype = torch.get_default_dtype()

        if len(AZWX) == 4:
            A, Z, W, X = AZWX
        elif len(AZWX) == 3:
            A, Z, W = AZWX
            X = None

        K_ATrainATrain = kernel_A(A, A)
        K_WTrainWTrain = kernel_W(W, W)
        K_ZTrainZTrain = kernel_Z(Z, Z)
        if X is None:
            K_XTrainXTrain = torch.ones((W.shape[0], W.shape[0]), device=self.device)
        else:
            K_XTrainXTrain = make_psd(kernel_X(X, X), make_psd_eps)

        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_X = kernel_X

        ############################# SPLIT DATA IN STAGE 1 AND STAGE 2 #####################################
        train_data_size = A.shape[0]
        train_indices = np.random.permutation(train_data_size)

        if (stage1_perc > 0.) & (stage1_perc < 1.):
            stage1_data_size = int(train_data_size * stage1_perc)
            stage2_data_size = train_data_size - stage1_data_size
            stage1_idx, stage2_idx = train_indices[:stage1_data_size], train_indices[stage1_data_size:]
        else:
            stage1_data_size, stage2_data_size = train_data_size, train_data_size
            stage1_idx, stage2_idx = train_indices, train_indices

        # 2. Split Kernel Matrices (using tensor indexing)
        K_AA = K_ATrainATrain[stage1_idx, :][:, stage1_idx]
        K_AATilde = K_ATrainATrain[stage1_idx, :][:, stage2_idx]
        K_ATildeATilde = K_ATrainATrain[stage2_idx, :][:, stage2_idx]

        K_WW = K_WTrainWTrain[stage1_idx, :][:, stage1_idx]
        K_WWTilde = K_WTrainWTrain[stage1_idx, :][:, stage2_idx]

        K_ZZ = K_ZTrainZTrain[stage1_idx, :][:, stage1_idx] # K(Z1, Z1)

        K_XX = K_XTrainXTrain[stage1_idx, :][:, stage1_idx]
        K_XXTilde = K_XTrainXTrain[stage1_idx, :][:, stage2_idx]

        # 3. Stage 1: Optimize Lambda and Solve for B, B_bar
        I_n = torch.eye(stage1_data_size, device=self.device)
        K_AWX = K_AA * K_WW * K_XX

        if optimize_lambda_parameters:
            l_range = self.lambda_optimization_range
            grid_points = self.regularization_grid_points

            log_min = torch.log(torch.tensor(l_range[0], dtype=dtype, device=self.device))
            log_max = torch.log(torch.tensor(l_range[1], dtype=dtype, device=self.device))
            lambda_list = torch.exp(torch.linspace(log_min, log_max, grid_points, device=self.device))

            objective_list = torch.zeros(grid_points, dtype=dtype, device=self.device)

            with torch.no_grad():
                for i, lambda_val in tqdm(enumerate(lambda_list), total=grid_points, desc="Tuning Lambda 1"):
                    objective_list[i] = lambda_objective_loocv(lambda_val, K_AWX, K_ZZ, self.label_variance_in_lambda_opt, self.make_psd_eps)

            best_lambda = lambda_list[torch.argmin(objective_list).item()]
            self.lambda1_ = best_lambda.item()

        ########### FIRST AND SECOND STAGE REGRESSION ########################################
        stage1_ridge_weights = (K_AWX + stage1_data_size * self.lambda1_ * I_n)
        self.stage1_ridge_weights = stage1_ridge_weights
        B = torch.linalg.solve(make_psd(stage1_ridge_weights, make_psd_eps), (K_WWTilde * K_XXTilde * K_AATilde))
        B_bar = torch.linalg.solve(make_psd(stage1_ridge_weights, make_psd_eps),  (columns_mean_excluding_self(K_WWTilde * K_XXTilde) * K_AATilde))

        block_component1 = (B.T @ K_ZZ @ B) * K_ATildeATilde
        block_component2 = (B.T @ K_ZZ @ B_bar) * K_ATildeATilde
        block_component4 = (B_bar.T @ K_ZZ @ B_bar) * K_ATildeATilde
        ones_divided_by_m = torch.ones((stage2_data_size, 1), dtype=K_AWX.dtype, device=self.device) / stage2_data_size

        L_sub = torch.vstack((block_component1, ones_divided_by_m.T @ block_component2.T))
        L = L_sub @ L_sub.T
        M = torch.vstack(((block_component2 @ ones_divided_by_m).reshape(-1, 1), (ones_divided_by_m.T @ block_component4 @ ones_divided_by_m).reshape(-1, 1)))

        P = torch.hstack((block_component1, (block_component2 @ ones_divided_by_m).reshape(-1, 1)))
        R = torch.hstack(((ones_divided_by_m.T @ block_component2.T).reshape(1, -1), (ones_divided_by_m.T @ block_component4 @ ones_divided_by_m).reshape(-1, 1)))
        N = torch.vstack((P, R))

        if optimize_eta_parameter:
            K_ATildeA = K_AATilde.T
            B2 = torch.linalg.solve(make_psd(stage1_ridge_weights, make_psd_eps), K_AWX)
            B2_bar = torch.linalg.solve(make_psd(stage1_ridge_weights, make_psd_eps),  (columns_mean_excluding_self(K_WW * K_XX) * K_AA))
            ones_divided_by_n = torch.ones((stage1_data_size, 1), dtype=K_AWX.dtype, device=self.device) / stage1_data_size

            block_component12 = (B2.T @ K_ZZ @ B) * K_AATilde
            block_component22 = (B2.T @ K_ZZ @ B_bar) * K_AATilde
            block_component32 = (B.T @ K_ZZ @ B2_bar) * K_ATildeA
            block_component42 = (B_bar.T @ K_ZZ @ B2_bar) * K_ATildeA

            L2_sub = torch.vstack((block_component12.T, ones_divided_by_m.T @ block_component22.T))
            L2 = L2_sub @ L2_sub.T
            M2 = torch.vstack(((block_component32 @ ones_divided_by_n).reshape(-1, 1), (ones_divided_by_m.T @ block_component42 @ ones_divided_by_n).reshape(-1, 1)))

            eta_range = self.eta_optimization_range
            grid_points = self.regularization_grid_points

            log_min = torch.log(torch.tensor(eta_range[0], dtype=dtype, device=self.device))
            log_max = torch.log(torch.tensor(eta_range[1], dtype=dtype, device=self.device))
            eta_list = torch.exp(torch.linspace(log_min, log_max, grid_points, device=self.device))

            objective_list = torch.zeros(grid_points, dtype=dtype, device=self.device)

            with torch.no_grad():
                for i, eta_val in tqdm(enumerate(eta_list), total=grid_points, desc="Tuning Eta"):
                    objective_list[i] = self._eta_objective(eta_val, L, L_sub, M, N, L2, M2, stage1_data_size, self.label_variance_in_eta_opt, self.make_psd_eps)
            best_eta = eta_list[torch.argmin(objective_list).item()]
            self.eta = best_eta.item()
            del K_ATildeA, B2, B2_bar, ones_divided_by_n, block_component12, block_component22, block_component32, block_component42, L2_sub

        alpha = torch.linalg.solve(make_psd(L / stage2_data_size + self.eta * N, make_psd_eps), M)

        if optimize_lambda_parameters:
            objective_list = torch.zeros(grid_points, dtype=dtype, device=self.device)

            with torch.no_grad():
                for i, lambda_val in tqdm(enumerate(lambda_list), total=grid_points, desc="Tuning Lambda 2"):
                    objective_list[i] = lambda_objective_loocv(lambda_val, K_ATrainATrain, K_ZTrainZTrain * (Y @ Y.T), self.label_variance_in_lambda_opt, self.make_psd_eps)

            best_lambda = lambda_list[torch.argmin(objective_list).item()]
            self.lambda2_ = best_lambda.item()

        K_ZZTrain = K_ZTrainZTrain[stage1_idx, :][:, train_indices] # Shape (N1, N_full)
        K_ATrainATrain_ = K_ATrainATrain[train_indices, :][:, train_indices] # Shape (N_full, N_full)
        I_N_full = torch.eye(train_data_size, dtype=K_ATrainATrain.dtype, device=self.device)
        # LHS (Regularized Gram Matrix)
        KRR_LHS = make_psd(
            K_ATrainATrain_ + train_data_size * self.lambda2_ * I_N_full,
            eps=self.make_psd_eps
        )
        KRR_RHS = K_ZZTrain.T * Y[train_indices]
        third_stage_KRR_weights = torch.linalg.solve(KRR_LHS, KRR_RHS).T

        self.alpha = alpha
        self.B, self.B_bar = B, B_bar
        self.third_stage_KRR_weights = third_stage_KRR_weights
        self.ones_divided_by_m = ones_divided_by_m
        self.ATrain, self.WTrain, self.ZTrain, self.XTrain = A, W, Z, X
        self.K_ZZ = K_ZZ
        self.train_indices = train_indices
        self.stage1_idx, self.stage2_idx = stage1_idx, stage2_idx

    def predict(self, A: torch.Tensor, outcome_transformer: nn.Module = TorchIdentityTransformer()) -> torch.Tensor:

        if A.ndim != 2:
            A_test = A.reshape(-1, 1)
        else:
            A_test = A
        K_ATrainATest = self.kernel_A(self.ATrain, A_test)

        test_indices = np.arange(A_test.shape[0])

        K_ATrainATest_ = K_ATrainATest[self.train_indices, :][:,test_indices]
        K_ATildeATest = K_ATrainATest[self.stage2_idx, :][:,test_indices]

        ones_divided_by_m = self.ones_divided_by_m
        alpha = self.alpha
        B, B_bar = self.B, self.B_bar
        third_stage_KRR_weights = self.third_stage_KRR_weights

        f_struct_pred = torch.vstack([self._predict_structural_function(alpha, B, B_bar, third_stage_KRR_weights,
                                                                    K_ATrainATest_[:, jj], K_ATildeATest[:, jj],
                                                                    ones_divided_by_m) for jj in range(K_ATildeATest.shape[1])])
        f_struct_pred = outcome_transformer.inverse_transform(f_struct_pred.squeeze(-1))
        return f_struct_pred


class KernelProxyVariableDoseResponse(BaseEstimator, RegressorMixin, nn.Module):

    def __init__(self,
                 kernel_A: Callable,
                 kernel_W: Callable,
                 kernel_Z: Callable,
                 kernel_X: Optional[Callable] = None,
                 lambda1_: float = 0.1,
                 lambda2_: float = 0.1,
                 optimize_lambda1_parameter: bool = True,
                 optimize_lambda2_parameter: bool = True,
                 lambda1_optimization_range: Tuple[float, float] = (1e-5, 1.0),
                 lambda2_optimization_range: Tuple[float, float] = (1e-5, 1.0),
                 device = None,
                 **kwargs) -> None:

        nn.Module.__init__(self)

        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_X = kernel_X if kernel_X is not None else RBF()

        self.lambda1_ = lambda1_
        self.lambda2_ = lambda2_
        self.optimize_lambda1_parameter = optimize_lambda1_parameter
        self.optimize_lambda2_parameter = optimize_lambda2_parameter
        self.lambda1_optimization_range = lambda1_optimization_range
        self.lambda2_optimization_range = lambda2_optimization_range

        # Hyperparameters from kwargs
        self.stage1_perc = kwargs.pop('stage1_perc', 0.5)
        self.regularization_grid_points = kwargs.pop('regularization_grid_points', 25)
        self.make_psd_eps = kwargs.pop('make_psd_eps', 1e-9)
        self.label_variance_in_lambda_opt = kwargs.pop('label_variance_in_lambda_opt', 0.0)

        # Model weights and indices
        self.alpha = None
        self.B = None
        self.w_mean_vec = None
        self.x_mean_vec = None
        self.upweight = None
        self.stage1_idx = None
        self.stage2_idx = None

        # Training data storage (for prediction)
        self.ATilde = None
        self.XTilde = None
        self.W = None

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self,
            AZWX: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
            Y: torch.Tensor) -> None:

        dtype = torch.get_default_dtype()
        if len(AZWX) == 4:
            A, Z, W, X = AZWX
        else:
            A, Z, W = AZWX
            X = None

        # --- 1. Precompute Full Kernel Matrices ---
        K_ATrainATrain = self.kernel_A(A, A)
        K_WTrainWTrain = self.kernel_W(W, W)
        K_ZTrainZTrain = self.kernel_Z(Z, Z)
        if X is None:
            K_XTrainXTrain = torch.ones((W.shape[0], W.shape[0]), device=self.device)
        else:
            K_XTrainXTrain = make_psd(self.kernel_X(X, X), self.make_psd_eps)

        # --- 2. Data Splitting ---
        n_total = A.shape[0]
        indices = np.random.permutation(n_total)

        if 0.0 < self.stage1_perc < 1.0:
            n1 = int(n_total * self.stage1_perc)
            self.stage1_idx, self.stage2_idx = indices[:n1], indices[n1:]
        else:
            n1 = n_total
            self.stage1_idx, self.stage2_idx = indices, indices

        n2 = n_total - n1 if n1 < n_total else n1

        # Sub-matrices
        K_AA = K_ATrainATrain[self.stage1_idx][:, self.stage1_idx]
        K_AATilde = K_ATrainATrain[self.stage1_idx][:, self.stage2_idx]
        K_ATildeATilde = K_ATrainATrain[self.stage2_idx][:, self.stage2_idx]

        K_WW = K_WTrainWTrain[self.stage1_idx][:, self.stage1_idx]
        K_ZZ = K_ZTrainZTrain[self.stage1_idx][:, self.stage1_idx]
        K_ZZTilde = K_ZTrainZTrain[self.stage1_idx][:, self.stage2_idx]

        K_XX = K_XTrainXTrain[self.stage1_idx][:, self.stage1_idx]
        K_XXTilde = K_XTrainXTrain[self.stage1_idx][:, self.stage2_idx]
        K_XTildeXTilde = K_XTrainXTrain[self.stage2_idx][:, self.stage2_idx]

        # --- 3. Stage 1: Optimize Lambda 1 & Solve for B ---
        K_ZAX = K_ZZ * K_AA * K_XX
        I_n = torch.eye(n1, device=self.device)

        if self.optimize_lambda1_parameter:
            l1_min, l1_max = self.lambda1_optimization_range
            lambda_list = torch.exp(torch.linspace(np.log(l1_min), np.log(l1_max), self.regularization_grid_points, device=self.device))

            obj_list = torch.zeros(self.regularization_grid_points, device=self.device)
            with torch.no_grad():
                for i, l_val in tqdm(enumerate(lambda_list), total=len(lambda_list), desc="Tuning Lambda 1"):
                    obj_list[i] = lambda_objective_loocv(l_val, K_ZAX, K_WW, self.label_variance_in_lambda_opt, self.make_psd_eps)
            self.lambda1_ = lambda_list[torch.argmin(obj_list)].item()

        # Solve for B (Stage 1 Weights)
        ridge_w1 = make_psd(K_ZAX + n1 * self.lambda1_ * I_n, self.make_psd_eps)
        K_ZAX_Tilde = K_ZZTilde * K_AATilde * K_XXTilde
        self.B = torch.linalg.solve(ridge_w1, K_ZAX_Tilde)

        # --- 4. Stage 2: Optimize Lambda 2 & Solve for Alpha ---
        # Identification formula: Stage 2 features are defined by the Stage 1 CME
        stage2_ridge_weights = K_ATildeATilde * (self.B.T @ K_WW @ self.B) * K_XTildeXTilde
        I_m = torch.eye(n2, device=self.device)
        YTilde = Y[self.stage2_idx]

        if self.optimize_lambda2_parameter:
            l2_min, l2_max = self.lambda2_optimization_range
            lambda_list = torch.exp(torch.linspace(np.log(l2_min), np.log(l2_max), self.regularization_grid_points, device=self.device))

            K_YTilde = YTilde @ YTilde.T
            obj_list = torch.zeros(self.regularization_grid_points, device=self.device)
            with torch.no_grad():
                for i, l_val in tqdm(enumerate(lambda_list), total=len(lambda_list), desc="Tuning Lambda 2"):
                    obj_list[i] = lambda_objective_loocv(l_val, stage2_ridge_weights, K_YTilde, self.label_variance_in_lambda_opt, self.make_psd_eps)
            self.lambda2_ = lambda_list[torch.argmin(obj_list)].item()

        # Solve for Alpha (Outcome Bridge Weights)
        ridge_w2 = make_psd(stage2_ridge_weights + n2 * self.lambda2_ * I_m, self.make_psd_eps)
        self.alpha = torch.linalg.solve(ridge_w2, YTilde)

        # --- 5. Precompute items for prediction ---
        self.w_mean_vec = torch.mean(K_WW, dim=0, keepdim=True).T
        self.x_mean_vec = torch.mean(K_XXTilde, dim=0, keepdim=True).T
        self.upweight = torch.mean((self.B.T @ K_WW) * K_XXTilde.T, dim=1, keepdim=True)

        self.ATilde = A[self.stage2_idx]
        self.XTilde = X[self.stage2_idx] if X is not None else None
        self.W = W[self.stage1_idx]

    def predict(self, A: torch.Tensor, outcome_transformer: nn.Module = TorchIdentityTransformer()) -> torch.Tensor:
        """Estimate the population Dose-Response (ATE)."""
        A_test = A.reshape(-1, 1) if A.ndim != 2 else A
        K_ATildeATest = self.kernel_A(self.ATilde, A_test)

        # Upweight logic: (K_test * weight).T @ alpha
        f_struct_pred = (K_ATildeATest * self.upweight).T @ self.alpha
        f_struct_pred = outcome_transformer.inverse_transform(f_struct_pred.squeeze(-1))
        return f_struct_pred

    def _predict_bridge_func(self, A_test: torch.Tensor, W_test: torch.Tensor, X_test: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict the outcome bridge function h(a, w, x)."""
        A_test = A_test.reshape(-1, 1) if A_test.ndim != 2 else A_test
        W_test = W_test.reshape(-1, 1) if W_test.ndim != 2 else W_test

        K_ATildeATest = self.kernel_A(self.ATilde, A_test)
        K_WWTest = self.kernel_W(self.W, W_test)

        if X_test is not None and self.XTilde is not None:
            K_XTildeXTest = self.kernel_X(self.XTilde, X_test)
        else:
            K_XTildeXTest = torch.ones((self.ATilde.shape[0], W_test.shape[0]), device=self.device)

        # Implementing the list comprehension logic from the reference implementation into a stacked tensor operation
        bridge_preds = []
        for jj in range(A_test.shape[0]):
            # Features for a specific (a, w, x) combination
            feat = (K_ATildeATest[:, jj].reshape(-1, 1) * ((self.B.T @ K_WWTest) * K_XTildeXTest))
            bridge_preds.append(feat.T @ self.alpha)

        return torch.stack(bridge_preds).squeeze(-1)


class KernelProxyVariableConditionalResponse(BaseEstimator, RegressorMixin, nn.Module):

    def __init__(self,
                 kernel_A: Callable,
                 kernel_W: Callable,
                 kernel_Z: Callable,
                 kernel_X: Optional[Callable] = None,
                 lambda1_: float = 0.1,
                 lambda2_: float = 0.1,
                 zeta: float = 0.1,
                 optimize_lambda1_parameter: bool = True,
                 optimize_lambda2_parameter: bool = True,
                 optimize_zeta_parameter: bool = True,
                 lambda1_optimization_range: Tuple[float, float] = (1e-5, 1.0),
                 lambda2_optimization_range: Tuple[float, float] = (1e-5, 1.0),
                 zeta_optimization_range: Tuple[float, float] = (1e-5, 1.0),
                 device=None,
                 **kwargs) -> None:

        nn.Module.__init__(self)

        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_X = kernel_X if kernel_X is not None else RBF()

        self.lambda1_ = lambda1_
        self.lambda2_ = lambda2_
        self.zeta = zeta

        self.optimize_lambda1_parameter = optimize_lambda1_parameter
        self.optimize_lambda2_parameter = optimize_lambda2_parameter
        self.optimize_zeta_parameter = optimize_zeta_parameter

        self.lambda1_optimization_range = lambda1_optimization_range
        self.lambda2_optimization_range = lambda2_optimization_range
        self.zeta_optimization_range = zeta_optimization_range

        # Hyperparameters
        self.stage1_perc = kwargs.pop('stage1_perc', 0.5)
        self.regularization_grid_points = kwargs.pop('regularization_grid_points', 25)
        self.make_psd_eps = kwargs.pop('make_psd_eps', 1e-9)
        self.label_variance_in_lambda_opt = kwargs.pop('label_variance_in_lambda_opt', 0.0)

        # Model weights
        self.alpha = None
        self.B = None
        self.stage1_idx = None
        self.stage2_idx = None

        # Stored components for ATT prediction
        self.K_ATildeATilde = None
        self.K_XTildeXTilde = None
        self.K_WWTilde = None

        # Data storage
        self.ATilde = None
        self.XTilde = None
        self.W = None

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _zeta_objective(zeta: float, K_AA: torch.Tensor, K_XX_WW: torch.Tensor, make_psd_eps: float = 1e-9) -> torch.Tensor:
        """Objective function for zeta optimization (LOOCV for treated group projection)."""
        n = K_AA.shape[0]
        # Ridge weights for the projection
        ridge_w = make_psd(K_AA + n * zeta * torch.eye(n, device=K_AA.device), make_psd_eps)
        R = torch.linalg.solve(ridge_w, K_AA).T

        # Cross-validation scaling factor
        S_vec = (1.0 / (1.0 - torch.diagonal(R))) ** 2

        # Error calculation using trace logic
        T_diag = S_vec * (torch.diagonal(K_XX_WW) -
                          2 * torch.diagonal(K_XX_WW @ R.T) +
                          torch.diagonal(R @ K_XX_WW @ R.T))

        return torch.sum(T_diag)

    def fit(self,
            AZWX: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
            Y: torch.Tensor) -> None:

        # Move inputs to device to prevent "stuck" kernels
        AZWX = tuple(t.to(self.device) if t is not None else None for t in AZWX)
        Y = Y.to(self.device)

        if len(AZWX) == 4:
            A, Z, W, X = AZWX
        else:
            A, Z, W = AZWX
            X = None

        # --- 1. Compute Kernels ---
        K_ATrainATrain = self.kernel_A(A, A)
        K_WTrainWTrain = self.kernel_W(W, W)
        K_ZTrainZTrain = self.kernel_Z(Z, Z)
        if X is None:
            K_XTrainXTrain = torch.ones((W.shape[0], W.shape[0]), device=self.device)
        else:
            K_XTrainXTrain = make_psd(self.kernel_X(X, X), self.make_psd_eps)

        # --- 2. Data Splitting ---
        n_total = A.shape[0]
        indices = np.random.permutation(n_total)

        if 0.0 < self.stage1_perc < 1.0:
            n1 = int(n_total * self.stage1_perc)
            self.stage1_idx, self.stage2_idx = indices[:n1], indices[n1:]
        else:
            n1 = n_total
            self.stage1_idx, self.stage2_idx = indices, indices

        n2 = n_total - n1 if n1 < n_total else n1

        # Extract Sub-matrices
        K_AA = K_ATrainATrain[self.stage1_idx][:, self.stage1_idx]
        K_AATilde = K_ATrainATrain[self.stage1_idx][:, self.stage2_idx]
        K_ATildeATilde = K_ATrainATrain[self.stage2_idx][:, self.stage2_idx]

        K_WW = K_WTrainWTrain[self.stage1_idx][:, self.stage1_idx]
        K_WWTilde = K_WTrainWTrain[self.stage1_idx][:, self.stage2_idx]
        K_WTildeWTilde = K_WTrainWTrain[self.stage2_idx][:, self.stage2_idx]

        K_ZZ = K_ZTrainZTrain[self.stage1_idx][:, self.stage1_idx]
        K_ZZTilde = K_ZTrainZTrain[self.stage1_idx][:, self.stage2_idx]

        K_XX = K_XTrainXTrain[self.stage1_idx][:, self.stage1_idx]
        K_XXTilde = K_XTrainXTrain[self.stage1_idx][:, self.stage2_idx]
        K_XTildeXTilde = K_XTrainXTrain[self.stage2_idx][:, self.stage2_idx]

        # --- 3. Stage 1: Solve for B (CME) ---
        K_ZAX = K_ZZ * K_AA * K_XX
        I_n = torch.eye(n1, device=self.device)

        if self.optimize_lambda1_parameter:
            l1_min, l1_max = self.lambda1_optimization_range
            lambda_list = torch.exp(torch.linspace(np.log(l1_min), np.log(l1_max), self.regularization_grid_points, device=self.device))
            obj_list = torch.zeros(len(lambda_list), device=self.device)
            with torch.no_grad():
                for i, l_val in tqdm(enumerate(lambda_list), total=len(lambda_list), desc="Tuning Lambda 1"):
                    obj_list[i] = lambda_objective_loocv(l_val, K_ZAX, K_WW, self.label_variance_in_lambda_opt, self.make_psd_eps)
            self.lambda1_ = lambda_list[torch.argmin(obj_list)].item()

        ridge_w1 = make_psd(K_ZAX + n1 * self.lambda1_ * I_n, self.make_psd_eps)
        K_ZAX_Tilde = K_ZZTilde * K_AATilde * K_XXTilde
        self.B = torch.linalg.solve(ridge_w1, K_ZAX_Tilde)

        # --- 4. Stage 2: Solve for Alpha (Bridge Function) ---
        stage2_ridge_weights = K_ATildeATilde * (self.B.T @ K_WW @ self.B) * K_XTildeXTilde
        I_m = torch.eye(n2, device=self.device)
        YTilde = Y[self.stage2_idx]

        if self.optimize_lambda2_parameter:
            l2_min, l2_max = self.lambda2_optimization_range
            lambda_list = torch.exp(torch.linspace(np.log(l2_min), np.log(l2_max), self.regularization_grid_points, device=self.device))
            K_YTilde = YTilde @ YTilde.T
            obj_list = torch.zeros(len(lambda_list), device=self.device)
            with torch.no_grad():
                for i, l_val in tqdm(enumerate(lambda_list), total=len(lambda_list), desc="Tuning Lambda 2"):
                    obj_list[i] = lambda_objective_loocv(l_val, stage2_ridge_weights, K_YTilde, self.label_variance_in_lambda_opt, self.make_psd_eps)
            self.lambda2_ = lambda_list[torch.argmin(obj_list)].item()

        # --- 5. ATT Specific: Zeta Optimization for Treated Projection ---
        if self.optimize_zeta_parameter:
            K_XX_WW_Tilde = K_XTildeXTilde * K_WTildeWTilde
            z_min, z_max = self.zeta_optimization_range
            zeta_list = torch.exp(torch.linspace(np.log(z_min), np.log(z_max), self.regularization_grid_points, device=self.device))
            obj_list = torch.zeros(len(zeta_list), device=self.device)
            with torch.no_grad():
                for i, z_val in tqdm(enumerate(zeta_list), total=len(zeta_list), desc="Tuning Zeta"):
                    obj_list[i] = self._zeta_objective(z_val, K_ATildeATilde, K_XX_WW_Tilde, self.make_psd_eps)
            self.zeta = zeta_list[torch.argmin(obj_list)].item()

        ridge_w2 = make_psd(stage2_ridge_weights + n2 * self.lambda2_ * I_m, self.make_psd_eps)
        self.alpha = torch.linalg.solve(ridge_w2, YTilde)

        # Save for prediction
        self.ATilde = A[self.stage2_idx]
        self.W = W[self.stage1_idx]
        self.K_WWTilde = K_WWTilde
        self.K_ATildeATilde = K_ATildeATilde
        self.K_XTildeXTilde = K_XTildeXTilde
        self.XTilde = X[self.stage2_idx] if X is not None else None

    def _predict_structural_function(self,
                                     alpha: torch.Tensor,
                                     B: torch.Tensor,
                                     KATilde_a: torch.Tensor,
                                     K_XTildeXTilde: torch.Tensor,
                                     K_WWTilde: torch.Tensor,
                                     CME_weights_: torch.Tensor) -> torch.Tensor:
        """Helper for the identification formula calculation of ATT."""
        # alpha.T @ (KATilde_a * ((K_XT_XT.T * (B.T @ K_WW)) @ CME_weights_))
        feat = KATilde_a * ((K_XTildeXTilde.T * (B.T @ K_WWTilde)) @ CME_weights_)
        return alpha.T @ feat

    def predict(self, A: torch.Tensor, aprime: torch.Tensor, outcome_transformer: nn.Module = TorchIdentityTransformer()) -> torch.Tensor:
        """Estimate the Average Treatment Effect on the Treated (ATT)."""
        A_test = A.reshape(-1, 1) if A.ndim != 2 else A
        aprime_test = aprime.reshape(-1, 1) if aprime.ndim != 2 else aprime

        K_ATildeaprime = self.kernel_A(self.ATilde, aprime_test)
        m = K_ATildeaprime.shape[0]

        # Calculate CME weights to project the Treated group onto the bridge space
        I_m = torch.eye(m, device=self.device)
        ridge_cme = make_psd(self.K_ATildeATilde + m * self.zeta * I_m, self.make_psd_eps)
        CME_weights_ = torch.linalg.solve(ridge_cme, K_ATildeaprime)

        K_ATildeATest = self.kernel_A(self.ATilde, A_test)
        n_test = K_ATildeATest.shape[1]
        preds = torch.zeros(n_test, device=self.device)

        for jj in range(n_test):
            res = self._predict_structural_function(
                self.alpha,
                self.B,
                K_ATildeATest[:, jj].view(-1, 1),
                self.K_XTildeXTilde,
                self.K_WWTilde,
                CME_weights_
            )
            preds[jj] = res.item()

        return outcome_transformer.inverse_transform(preds)

    def _predict_bridge_func(self, A_test: torch.Tensor, W_test: torch.Tensor, X_test: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict the outcome bridge function h(a, w, x)."""
        A_test = A_test.reshape(-1, 1) if A_test.ndim != 2 else A_test
        W_test = W_test.reshape(-1, 1) if W_test.ndim != 2 else W_test

        K_ATildeATest = self.kernel_A(self.ATilde, A_test)
        K_WWTest = self.kernel_W(self.W, W_test)

        if X_test is not None and self.XTilde is not None:
            K_XTildeXTest = self.kernel_X(self.XTilde, X_test)
        else:
            K_XTildeXTest = torch.ones((self.ATilde.shape[0], W_test.shape[0]), device=self.device)

        bridge_preds = []
        for jj in range(A_test.shape[0]):
            feat = (K_ATildeATest[:, jj].view(-1, 1) * ((self.B.T @ K_WWTest) * K_XTildeXTest))
            bridge_preds.append(feat.T @ self.alpha)

        return torch.stack(bridge_preds).squeeze(-1)


class KernelProxyVariableHeterogeneousResponse(BaseEstimator, RegressorMixin, nn.Module):

    def __init__(self,
                 kernel_A: Callable,
                 kernel_W: Callable,
                 kernel_Z: Callable,
                 kernel_V: Callable,
                 kernel_X: Optional[Callable] = None,
                 lambda1_: float = 0.1,
                 lambda2_: float = 0.1,
                 zeta: float = 0.1,
                 optimize_lambda1_parameter: bool = True,
                 optimize_lambda2_parameter: bool = True,
                 optimize_zeta_parameter: bool = True,
                 lambda1_optimization_range: Tuple[float, float] = (1e-5, 1.0),
                 lambda2_optimization_range: Tuple[float, float] = (1e-5, 1.0),
                 zeta_optimization_range: Tuple[float, float] = (1e-5, 1.0),
                 device=None,
                 **kwargs) -> None:

        nn.Module.__init__(self)

        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_V = kernel_V
        self.kernel_X = kernel_X if kernel_X is not None else RBF()

        self.lambda1_ = lambda1_
        self.lambda2_ = lambda2_
        self.zeta = zeta

        self.optimize_lambda1_parameter = optimize_lambda1_parameter
        self.optimize_lambda2_parameter = optimize_lambda2_parameter
        self.optimize_zeta_parameter = optimize_zeta_parameter

        self.lambda1_optimization_range = lambda1_optimization_range
        self.lambda2_optimization_range = lambda2_optimization_range
        self.zeta_optimization_range = zeta_optimization_range

        # Hyperparameters
        self.stage1_perc = kwargs.pop('stage1_perc', 0.5)
        self.regularization_grid_points = kwargs.pop('regularization_grid_points', 25)
        self.make_psd_eps = kwargs.pop('make_psd_eps', 1e-9)
        self.label_variance_in_lambda_opt = kwargs.pop('label_variance_in_lambda_opt', 0.0)

        # Model weights
        self.alpha = None
        self.B = None
        self.stage1_idx = None
        self.stage2_idx = None

        # Stored matrices for CATE prediction
        self.K_XTildeXTilde = None
        self.K_WWTilde = None
        self.K_VTildeVTilde = None

        # Training data storage
        self.ATilde = None
        self.VTilde = None
        self.XTilde = None
        self.W = None

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _zeta_objective(zeta: float, K_VV: torch.Tensor, K_XX_WW: torch.Tensor, make_psd_eps: float = 1e-9) -> torch.Tensor:
        """Objective function for zeta optimization (regularization for CME)."""
        n = K_VV.shape[0]
        # R = K_VV @ inv(K_VV + n*zeta*I)
        ridge_w = make_psd(K_VV + n * zeta * torch.eye(n, device=K_VV.device), make_psd_eps)
        R = torch.linalg.solve(ridge_w, K_VV).T

        # S = diag((1 / (1 - diag(R)))**2)
        diag_R = torch.diagonal(R)
        S_vec = (1.0 / (1.0 - diag_R)) ** 2

        # T = S @ (K_XX_WW - 2 * K_XX_WW @ R.T + R @ K_XX_WW @ R.T)
        # Note: We only need the trace, so we can compute it efficiently
        T_diag = S_vec * (torch.diagonal(K_XX_WW) -
                          2 * torch.diagonal(K_XX_WW @ R.T) +
                          torch.diagonal(R @ K_XX_WW @ R.T))

        return torch.sum(T_diag)

    def fit(self,
            AZWVX: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
            Y: torch.Tensor) -> None:

        if len(AZWVX) == 5:
            A, Z, W, V, X = AZWVX
        else:
            A, Z, W, V = AZWVX
            X = None

        # --- 1. Full Kernel Matrices ---
        K_ATrainATrain = self.kernel_A(A, A)
        K_WTrainWTrain = self.kernel_W(W, W)
        K_ZTrainZTrain = self.kernel_Z(Z, Z)
        K_VTrainVTrain = self.kernel_V(V, V)
        if X is None:
            K_XTrainXTrain = torch.ones((W.shape[0], W.shape[0]), device=self.device)
        else:
            K_XTrainXTrain = make_psd(self.kernel_X(X, X), self.make_psd_eps)

        # --- 2. Data Splitting ---
        n_total = A.shape[0]
        indices = np.random.permutation(n_total)

        if 0.0 < self.stage1_perc < 1.0:
            n1 = int(n_total * self.stage1_perc)
            self.stage1_idx, self.stage2_idx = indices[:n1], indices[n1:]
        else:
            n1 = n_total
            self.stage1_idx, self.stage2_idx = indices, indices

        n2 = n_total - n1 if n1 < n_total else n1

        # Sub-matrices (Stage 1 and 2)
        K_AA = K_ATrainATrain[self.stage1_idx][:, self.stage1_idx]
        K_AATilde = K_ATrainATrain[self.stage1_idx][:, self.stage2_idx]
        K_ATildeATilde = K_ATrainATrain[self.stage2_idx][:, self.stage2_idx]

        K_VV = K_VTrainVTrain[self.stage1_idx][:, self.stage1_idx]
        K_VVTilde = K_VTrainVTrain[self.stage1_idx][:, self.stage2_idx]
        K_VTildeVTilde = K_VTrainVTrain[self.stage2_idx][:, self.stage2_idx]

        K_WW = K_WTrainWTrain[self.stage1_idx][:, self.stage1_idx]
        K_WWTilde = K_WTrainWTrain[self.stage1_idx][:, self.stage2_idx]
        K_WTildeWTilde = K_WTrainWTrain[self.stage2_idx][:, self.stage2_idx]

        K_ZZ = K_ZTrainZTrain[self.stage1_idx][:, self.stage1_idx]
        K_ZZTilde = K_ZTrainZTrain[self.stage1_idx][:, self.stage2_idx]

        K_XX = K_XTrainXTrain[self.stage1_idx][:, self.stage1_idx]
        K_XXTilde = K_XTrainXTrain[self.stage1_idx][:, self.stage2_idx]
        K_XTildeXTilde = K_XTrainXTrain[self.stage2_idx][:, self.stage2_idx]

        # --- 3. Stage 1: Lambda 1 & Solve B ---
        K_ZAVX = K_ZZ * K_AA * K_VV * K_XX
        I_n = torch.eye(n1, device=self.device)

        if self.optimize_lambda1_parameter:
            l1_min, l1_max = self.lambda1_optimization_range
            lambda_list = torch.exp(torch.linspace(np.log(l1_min), np.log(l1_max), self.regularization_grid_points, device=self.device))
            obj_list = torch.zeros(self.regularization_grid_points, device=self.device)
            with torch.no_grad():
                for i, l_val in tqdm(enumerate(lambda_list), total=len(lambda_list), desc="Tuning Lambda 1"):
                    obj_list[i] = lambda_objective_loocv(l_val, K_ZAVX, K_WW, self.label_variance_in_lambda_opt, self.make_psd_eps)
            self.lambda1_ = lambda_list[torch.argmin(obj_list)].item()

        ridge_w1 = make_psd(K_ZAVX + n1 * self.lambda1_ * I_n, self.make_psd_eps)
        K_ZAVX_Tilde = K_ZZTilde * K_AATilde * K_VVTilde * K_XXTilde
        self.B = torch.linalg.solve(ridge_w1, K_ZAVX_Tilde)

        # --- 4. Stage 2: Lambda 2 & Solve Alpha ---
        stage2_ridge_weights = K_ATildeATilde * (self.B.T @ K_WW @ self.B) * K_VTildeVTilde * K_XTildeXTilde
        I_m = torch.eye(n2, device=self.device)
        YTilde = Y[self.stage2_idx]

        if self.optimize_lambda2_parameter:
            l2_min, l2_max = self.lambda2_optimization_range
            lambda_list = torch.exp(torch.linspace(np.log(l2_min), np.log(l2_max), self.regularization_grid_points, device=self.device))
            K_YTilde = YTilde @ YTilde.T
            obj_list = torch.zeros(self.regularization_grid_points, device=self.device)
            with torch.no_grad():
                for i, l_val in tqdm(enumerate(lambda_list), total=len(lambda_list), desc="Tuning Lambda 2"):
                    obj_list[i] = lambda_objective_loocv(l_val, stage2_ridge_weights, K_YTilde, self.label_variance_in_lambda_opt, self.make_psd_eps)
            self.lambda2_ = lambda_list[torch.argmin(obj_list)].item()

        ridge_w2 = make_psd(stage2_ridge_weights + n2 * self.lambda2_ * I_m, self.make_psd_eps)
        self.alpha = torch.linalg.solve(ridge_w2, YTilde)

        # --- 5. Stage 3: Zeta Optimization (CME Regularization) ---
        if self.optimize_zeta_parameter:
            K_XXWW_Tilde = K_XTildeXTilde * K_WTildeWTilde
            z_min, z_max = self.zeta_optimization_range
            zeta_list = torch.exp(torch.linspace(np.log(z_min), np.log(z_max), self.regularization_grid_points, device=self.device))
            obj_list = torch.zeros(self.regularization_grid_points, device=self.device)
            with torch.no_grad():
                for i, z_val in tqdm(enumerate(zeta_list), total=len(zeta_list), desc="Tuning Zeta"):
                    obj_list[i] = self._zeta_objective(z_val, K_VTildeVTilde, K_XXWW_Tilde, self.make_psd_eps)
            self.zeta = zeta_list[torch.argmin(obj_list)].item()

        # Storage for Prediction
        self.ATilde, self.VTilde = A[self.stage2_idx], V[self.stage2_idx]
        self.K_XTildeXTilde = K_XTildeXTilde
        self.K_WWTilde = K_WWTilde
        self.K_VTildeVTilde = K_VTildeVTilde
        self.XTilde = X[self.stage2_idx] if X is not None else None
        self.W = W[self.stage1_idx]

    def _predict_structural_function(self,
                                     alpha: torch.Tensor,
                                     B: torch.Tensor,
                                     K_ATildea: torch.Tensor,
                                     K_VTildev: torch.Tensor,
                                     K_XTildeXTilde: torch.Tensor,
                                     K_WWTilde: torch.Tensor,
                                     CME_weights_: torch.Tensor) -> torch.Tensor:
        """Helper to calculate CATE for a specific test point."""
        # Equivalent to the reference implementation logic: alpha.T @ ((K_Aa * K_Vv) * ((K_XX * (B.T @ K_WW)) @ CME_weights))
        feat = (K_ATildea * K_VTildev) * ((K_XTildeXTilde * (B.T @ K_WWTilde)) @ CME_weights_)
        return alpha.T @ feat

    def predict(self, A: torch.Tensor, V: torch.Tensor, outcome_transformer: nn.Module = TorchIdentityTransformer()) -> torch.Tensor:
        """Estimate the Conditional Average Treatment Effect (CATE)."""
        A_test = A.reshape(-1, 1) if A.ndim != 2 else A
        V_test = V # Moderator test points

        K_ATildeATest = self.kernel_A(self.ATilde, A_test)
        K_VTildeVTest = self.kernel_V(self.VTilde, V_test)

        m = self.ATilde.shape[0]
        I_m = torch.eye(m, device=self.device)

        # CME_weights: Weights to project moderators onto the RKHS
        # CME_weights = (K_VTildeVTilde + m * zeta * I)^-1 @ K_VTildeVTest
        ridge_cme = make_psd(self.K_VTildeVTilde + m * self.zeta * I_m, self.make_psd_eps)
        CME_weights_ = torch.linalg.solve(ridge_cme, K_VTildeVTest)

        n_test = A_test.shape[0]
        preds = torch.zeros(n_test, device=self.device)

        # Calculate CATE point by point to match the structured function logic
        for jj in range(n_test):
            res = self._predict_structural_function(
                self.alpha,
                self.B,
                K_ATildeATest[:, jj].view(-1, 1),
                K_VTildeVTest[:, jj].view(-1, 1),
                self.K_XTildeXTilde,
                self.K_WWTilde,
                CME_weights_[:, jj].view(-1, 1)
            )
            preds[jj] = res.item()

        return outcome_transformer.inverse_transform(preds)

    def _predict_bridge_func(self,
                             A_test: torch.Tensor,
                             W_test: torch.Tensor,
                             V_test: torch.Tensor,
                             X_test: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict the outcome bridge function h(a, w, v, x)."""
        A_test = A_test.reshape(-1, 1) if A_test.ndim != 2 else A_test
        W_test = W_test.reshape(-1, 1) if W_test.ndim != 2 else W_test
        V_test = V_test.reshape(-1, 1) if V_test.ndim != 2 else V_test

        K_ATildeATest = self.kernel_A(self.ATilde, A_test)
        K_VTildeVTest = self.kernel_V(self.VTilde, V_test)
        K_WWTest = self.kernel_W(self.W, W_test)

        if X_test is not None and self.XTilde is not None:
            K_XTildeXTest = self.kernel_X(self.XTilde, X_test)
        else:
            K_XTildeXTest = torch.ones((self.ATilde.shape[0], W_test.shape[0]), device=self.device)

        bridge_preds = []
        for jj in range(A_test.shape[0]):
            # (K_Aa * K_Vv) * ((B.T @ K_WW) * K_XX)
            feat = (K_ATildeATest[:, jj] * K_VTildeVTest[:, jj]).view(-1, 1) * ((self.B.T @ K_WWTest) * K_XTildeXTest)
            bridge_preds.append(feat.T @ self.alpha)

        return torch.stack(bridge_preds).squeeze(-1)


##################################################################################
##################### ATT (AVERAGE TREATMENT EFFECT ON THE TREATED) ##############
##################################################################################
# ATT estimators `KernelProxyVariableATT` and `KernelAlternativeProxyATT`. The estimand is
#
#       f_ATT(a, a') = E[ Y^(a) | A = a' ]
#
# -- the dose response at treatment level `a` for the sub-population that actually received `a'`.
# The FIRST argument is the intervention; the SECOND indexes the factual reference population and is
# NOT a second intervention. At a = a' this rides the OBSERVATIONAL curve E[Y|A=a], not the ATE curve.
#
# THE ATE -> ATT DELTA IS ONE SUBSTITUTION in each family: wherever the ATE averages the (W, X)
# argument over its empirical marginal, the ATT averages it over the conditional mean embedding of
# P(W, X | A = a'), obtained by a zeta-regularised kernel ridge on A. Everything else -- stage 1, the
# block algebra, the alpha solve, the third stage -- is unchanged.
#
# IMPLEMENTATION NOTES (differences from the reference, all deliberate):
#   * DATA TUPLE ORDER IS (A, Z, W[, X]), this repo's convention. Other implementations take
#     (A, W, Z[, X]); verbatim reference code runs WITHOUT ERROR here and silently swaps the proxies.
#   * float64 is required: the reference implementation runs in double precision and the jitter is 1e-9/1e-10, which is
#     below float32 resolution. Call torch.set_default_dtype(torch.float64) before use.
#   * torch.linalg.solve everywhere (general LU), matching torch.linalg.solve -- never Cholesky, since
#     several of these matrices are only NEARLY PSD.
#   * The reference's JITTER PLACEMENT IS INCONSISTENT and is reproduced literally so the port can be
#     validated exactly: B / B_bar / B2_bar take the RAW ridge weights with no make_psd, while
#     B2 uses make_psd(..., make_psd_eps); several make_psd calls rely on the DEFAULT eps=1e-10 rather
#     than make_psd_eps. Every such site is commented below. Do not "tidy" these.
#   * The reference's per-test-point Python loops in predict are vectorised here; the algebra is a
#     plain batched contraction and the outputs agree to float64 round-off.
#   * The objectives are re-implemented as static methods rather than reusing
#     torch_utils.ml_utils.lambda_objective_loocv, because the reference's traces differ in transpose
#     placement between objectives (S @ K @ S vs S @ K @ S.T) and S is NOT symmetric.


class KernelProxyVariableATT(BaseEstimator, RegressorMixin, nn.Module):
    """KPV outcome bridge for the ATT.

    h solves the KPV moment equation E[h(A,W,X) | A,Z,X] = E[Y | A,Z,X] exactly as in the ATE case --
    `fit` takes no a' and ONE fit serves every anchor. a' enters only through `predict`, which
    replaces the ATE's uniform average of h over the empirical (W,X) with the CME-weighted average
    over P(W,X | A=a'):

        f_ATT(a, a') = sum_{k in stage2} theta_k * h(a, W_k, X_k),
        theta = (K_AtildeAtilde + m*zeta*I)^-1 k_A(Atilde, a').
    """

    def __init__(self,
                 kernel_A: Callable,
                 kernel_W: Callable,
                 kernel_Z: Callable,
                 kernel_X: Optional[Callable] = None,
                 lambda1_: float = 0.1,
                 lambda2_: float = 0.1,
                 zeta: float = 0.1,
                 optimize_lambda1_parameter: bool = True,
                 optimize_lambda2_parameter: bool = True,
                 optimize_zeta_parameter: bool = True,
                 lambda1_optimization_range: Tuple[float, float] = (1e-5, 1.0),
                 lambda2_optimization_range: Tuple[float, float] = (1e-5, 1.0),
                 zeta_optimization_range: Tuple[float, float] = (1e-5, 1.0),
                 device=None,
                 **kwargs) -> None:

        nn.Module.__init__(self)

        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_X = kernel_X if kernel_X is not None else RBF()

        self.lambda1_ = lambda1_
        self.lambda2_ = lambda2_
        self.zeta = zeta
        self.optimize_lambda1_parameter = optimize_lambda1_parameter
        self.optimize_lambda2_parameter = optimize_lambda2_parameter
        self.optimize_zeta_parameter = optimize_zeta_parameter
        self.lambda1_optimization_range = lambda1_optimization_range
        self.lambda2_optimization_range = lambda2_optimization_range
        self.zeta_optimization_range = zeta_optimization_range

        self.stage1_perc = kwargs.pop('stage1_perc', 0.5)
        self.regularization_grid_points = kwargs.pop('regularization_grid_points', 25)
        self.make_psd_eps = kwargs.pop('make_psd_eps', 1e-9)

        self.alpha = None
        self.B = None
        self.ATilde = None
        self.XTilde = None
        self.W = None
        self.stage1_idx = None
        self.stage2_idx = None

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ objectives
    @staticmethod
    def _lambda1_objective(lambda_, K_ZAX, K_WW, identity_matrix, make_psd_eps=1e-9):
        n = K_ZAX.shape[0]
        ridge_weights = make_psd(K_ZAX + n * lambda_ * identity_matrix, eps=make_psd_eps)
        R = torch.linalg.solve(ridge_weights, K_ZAX).T
        H1 = identity_matrix - R
        H1_diag = torch.diag(H1)
        H1_tilde_inv = torch.diag(1.0 / H1_diag)
        H1_tilde_inv_times_H1 = H1_tilde_inv @ H1
        # NOTE: no transpose on the trailing factor -- matches the reference exactly.
        return (1.0 / n) * torch.trace(H1_tilde_inv_times_H1 @ K_WW @ H1_tilde_inv_times_H1)

    @staticmethod
    def _lambda2_objective(lambda_, second_stage_ridge_weights, K_YTilde, identity_matrix,
                           make_psd_eps=1e-9):
        n = second_stage_ridge_weights.shape[0]
        ridge_weights = make_psd(second_stage_ridge_weights + n * lambda_ * identity_matrix,
                                 eps=make_psd_eps)
        R = torch.linalg.solve(ridge_weights, second_stage_ridge_weights).T
        H1 = identity_matrix - R
        H1_diag = torch.diag(H1)
        H1_tilde_inv = torch.diag(1.0 / H1_diag)
        H1_tilde_inv_times_H1 = H1_tilde_inv @ H1
        return (1.0 / n) * torch.trace(H1_tilde_inv_times_H1 @ K_YTilde @ H1_tilde_inv_times_H1)

    @staticmethod
    def zeta_objective(zeta, K_AA, K_XX_WW, make_psd_eps=1e-9):
        """CME LOOCV for E[phi_W(x) phi_X(x) | A]. Public name (no underscore) as in the reference."""
        n = K_AA.shape[0]
        eye = torch.eye(n, dtype=K_AA.dtype, device=K_AA.device)
        R = torch.linalg.solve(make_psd(K_AA + n * zeta * eye, make_psd_eps), K_AA).T
        S = torch.diag((1.0 / (1.0 - torch.diag(R))) ** 2)
        T = S @ (K_XX_WW - 2 * K_XX_WW @ R.T + R @ K_XX_WW @ R.T)
        return torch.trace(T)

    @staticmethod
    def _geomspace(lo, hi, n, dtype, device):
        """A logspace grid(log(lo), log(hi), n, base=e) == a geometric grid lo -> hi."""
        return torch.logspace(float(np.log(lo)), float(np.log(hi)), n,
                              base=float(np.exp(1.0)), dtype=dtype, device=device)

    # ------------------------------------------------------------------ fit
    def fit(self,
            AZWX: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
            Y: torch.Tensor) -> None:
        """Fit the a'-free outcome bridge. Tuple order is (A, Z, W[, X]) -- NOT the reference order (A, W, Z)."""
        if len(AZWX) == 4:
            A, Z, W, X = AZWX
        else:
            A, Z, W = AZWX
            X = None

        dtype, device = A.dtype, A.device
        K_ATrainATrain = self.kernel_A(A, A)
        K_WTrainWTrain = self.kernel_W(W, W)
        K_ZTrainZTrain = self.kernel_Z(Z, Z)
        if X is None:
            K_XTrainXTrain = torch.ones((W.shape[0], W.shape[0]), dtype=dtype, device=device)
        else:
            K_XTrainXTrain = make_psd(self.kernel_X(X, X), self.make_psd_eps)

        # the split is drawn from the GLOBAL numpy RNG, exactly as in the reference: seed immediately
        # before fit, and again after any DGP that consumes the stream.
        train_data_size = A.shape[0]
        train_indices = np.random.permutation(train_data_size)
        if (self.stage1_perc > 0.0) and (self.stage1_perc < 1.0):
            n = int(train_data_size * self.stage1_perc)
            m = train_data_size - n
            stage1_idx, stage2_idx = train_indices[:n], train_indices[n:]
        else:
            n, m = train_data_size, train_data_size
            stage1_idx, stage2_idx = train_indices, train_indices

        sub = lambda K, i1, i2: K[i1][:, i2]
        K_AA = sub(K_ATrainATrain, stage1_idx, stage1_idx)
        K_AATilde = sub(K_ATrainATrain, stage1_idx, stage2_idx)
        K_ATildeATilde = sub(K_ATrainATrain, stage2_idx, stage2_idx)
        K_WW = sub(K_WTrainWTrain, stage1_idx, stage1_idx)
        K_WWTilde = sub(K_WTrainWTrain, stage1_idx, stage2_idx)
        K_WTildeWTilde = sub(K_WTrainWTrain, stage2_idx, stage2_idx)
        K_ZZ = sub(K_ZTrainZTrain, stage1_idx, stage1_idx)
        K_ZZTilde = sub(K_ZTrainZTrain, stage1_idx, stage2_idx)
        K_XX = sub(K_XTrainXTrain, stage1_idx, stage1_idx)
        K_XXTilde = sub(K_XTrainXTrain, stage1_idx, stage2_idx)
        K_XTildeXTilde = sub(K_XTrainXTrain, stage2_idx, stage2_idx)

        for kernel_ in [self.kernel_A, self.kernel_W, self.kernel_Z, self.kernel_X]:
            if hasattr(kernel_, 'use_length_scale_heuristic'):
                kernel_.use_length_scale_heuristic = False

        # ---- stage 1: CME of phi_W(W) given (Z, A, X) ----
        K_ZAX = K_ZZ * K_AA * K_XX
        I_n = torch.eye(n, dtype=dtype, device=device)
        I_m = torch.eye(m, dtype=dtype, device=device)
        YTilde = Y[stage2_idx]

        if self.optimize_lambda1_parameter:
            grid = self._geomspace(*self.lambda1_optimization_range,
                                   self.regularization_grid_points, dtype, device)
            obj = torch.stack([self._lambda1_objective(l, K_ZAX, K_WW, I_n, self.make_psd_eps)
                               for l in grid])
            self.lambda1_ = float(grid[int(torch.argmin(obj))])
            self.lambda1_objective_list = obj

        stage1_ridge_weights = K_ZAX + n * self.lambda1_ * I_n
        K_ZAX_ZAXTilde = K_ZZTilde * K_AATilde * K_XXTilde
        B = torch.linalg.solve(make_psd(stage1_ridge_weights), K_ZAX_ZAXTilde)   # DEFAULT eps 1e-10

        # ---- stage 2: KRR of Y on phi_A(Atilde) (x) mu_W (x) phi_X(Xtilde) ----
        stage2_ridge_weights = K_ATildeATilde * (B.T @ K_WW @ B) * K_XTildeXTilde

        if self.optimize_lambda2_parameter:
            K_YTilde = YTilde @ YTilde.T
            grid = self._geomspace(*self.lambda2_optimization_range,
                                   self.regularization_grid_points, dtype, device)
            obj = torch.stack([self._lambda2_objective(l, stage2_ridge_weights, K_YTilde, I_m,
                                                       self.make_psd_eps) for l in grid])
            self.lambda2_ = float(grid[int(torch.argmin(obj))])
            self.lambda2_objective_list = obj

        if self.optimize_zeta_parameter:                      # ATT-ONLY block
            K_XX_WW_Tilde = K_XTildeXTilde * K_WTildeWTilde
            grid = self._geomspace(*self.zeta_optimization_range,
                                   self.regularization_grid_points, dtype, device)
            obj = torch.stack([self.zeta_objective(z, K_ATildeATilde, K_XX_WW_Tilde) for z in grid])
            self.zeta = float(grid[int(torch.argmin(obj))])
            self.zeta_objective_list = obj

        stage2_ridge_weights = stage2_ridge_weights + m * self.lambda2_ * I_m
        alpha = torch.linalg.solve(make_psd(stage2_ridge_weights), YTilde)       # DEFAULT eps 1e-10

        self.alpha = alpha
        self.B = B
        self.ATilde = A[stage2_idx]
        self.W = W[stage1_idx]
        self.K_WWTilde = K_WWTilde
        self.K_ATildeATilde = K_ATildeATilde
        self.K_XTildeXTilde = K_XTildeXTilde
        self.XTilde = X[stage2_idx] if X is not None else None
        self.stage1_idx, self.stage2_idx = stage1_idx, stage2_idx

    # ------------------------------------------------------------------ predict
    def predict(self, A: torch.Tensor, aprime, outcome_transformer=None) -> torch.Tensor:
        """f_ATT(a, a') on the grid `A`, for the single anchor `aprime`.

        `outcome_transformer`, when given, inverse-transforms INTERNALLY (this repo's convention);
        the reference instead inverse-transforms outside with the sklearn Y scaler.
        """
        A_test = A if A.ndim == 2 else A.reshape(-1, 1)
        if not torch.is_tensor(aprime):
            aprime = torch.tensor(aprime, dtype=A_test.dtype, device=A_test.device)
        aprime = aprime.reshape(-1, 1).to(dtype=A_test.dtype, device=A_test.device)

        K_ATildeaprime = self.kernel_A(self.ATilde, aprime)                       # (m, 1)
        m = K_ATildeaprime.shape[0]
        eye = torch.eye(m, dtype=K_ATildeaprime.dtype, device=K_ATildeaprime.device)
        # NO make_psd here in the reference -- the most fragile solve in the file (small zeta plus
        # near-duplicate A values is ill-conditioned). Reproduced as-is.
        CME_weights_ = torch.linalg.solve(self.K_ATildeATilde + m * self.zeta * eye, K_ATildeaprime)
        K_ATildeATest = self.kernel_A(self.ATilde, A_test)                        # (m, T)

        # vectorised form of the reference's per-test-point loop
        v = (self.K_XTildeXTilde.T * (self.B.T @ self.K_WWTilde)) @ CME_weights_  # (m, 1)
        pred = (K_ATildeATest * (self.alpha * v)).sum(dim=0).reshape(-1, 1)       # (T, 1)

        if outcome_transformer is not None:
            pred = outcome_transformer.inverse_transform(pred)
        return pred.reshape(-1)

    def _predict_bridge_func(self, A_test: torch.Tensor, W_test: torch.Tensor, X_test=None):
        """h_hat(a, w[, x]) evaluated on the test grids; shape (T_A, T_W)."""
        if A_test.ndim != 2:
            A_test = A_test.reshape(-1, 1)
        K_ATildeATest = self.kernel_A(self.ATilde, A_test)                        # (m, T_A)
        K_WWTest = self.kernel_W(self.W, W_test)                                  # (n, T_W)
        if (X_test is not None) and (self.XTilde is not None):
            K_XTildeXTest = self.kernel_X(self.XTilde, X_test)                    # (m, T_W)
        else:
            K_XTildeXTest = torch.ones((self.ATilde.shape[0], W_test.shape[0]),
                                       dtype=A_test.dtype, device=A_test.device)
        base = (self.B.T @ K_WWTest) * K_XTildeXTest                              # (m, T_W)
        return torch.einsum('mt,mw,mo->tw', K_ATildeATest, base, self.alpha)


class KernelAlternativeProxyATT(BaseEstimator, RegressorMixin, nn.Module):
    """KAP (alternative-proxy / treatment-bridge) estimator for the ATT, density-ratio-free.

    The bridge solves  E[phi_{a'}(A,Z) | A=a, W=w] = p(w|a') / p(w|a)  -- the ATE's target measure
    p(w) replaced by p(w|a'), which is where the "on the treated" conditioning enters. The third
    stage then forms  f_ATT(a, a') = E[ Y * phi_{a'}(a, Z) | A = a ].

    a' is baked into B_bar at FIT time, so sweeping anchors requires a refit (as in the reference).
    Only B_bar/alpha depend on a'; B, the Grams and the third stage do not, so a caller sweeping many
    anchors could cache those -- not done here, to keep the port faithful.
    """

    def __init__(self,
                 kernel_A: Callable,
                 kernel_W: Callable,
                 kernel_Z: Callable,
                 kernel_X: Optional[Callable] = None,
                 lambda_: float = 0.1,
                 eta: float = 0.1,
                 lambda2_: float = 0.1,
                 zeta: float = 0.1,
                 optimize_lambda_parameters: bool = True,
                 optimize_eta_parameter: bool = True,
                 optimize_zeta_parameter: bool = True,
                 lambda_optimization_range: Tuple[float, float] = (1e-7, 1.0),
                 eta_optimization_range: Tuple[float, float] = (1e-7, 1.0),
                 zeta_optimization_range: Tuple[float, float] = (1e-7, 1.0),
                 device=None,
                 **kwargs) -> None:

        nn.Module.__init__(self)

        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_X = kernel_X if kernel_X is not None else RBF()

        self.lambda_ = lambda_
        self.eta = eta
        self.lambda2_ = lambda2_
        self.zeta = zeta
        self.optimize_lambda_parameters = optimize_lambda_parameters
        self.optimize_eta_parameter = optimize_eta_parameter
        self.optimize_zeta_parameter = optimize_zeta_parameter
        self.lambda_optimization_range = lambda_optimization_range
        self.eta_optimization_range = eta_optimization_range
        self.zeta_optimization_range = zeta_optimization_range

        self.stage1_perc = kwargs.pop('stage1_perc', 0.5)
        self.regularization_grid_points = kwargs.pop('regularization_grid_points', 25)
        self.label_variance_in_lambda_opt = kwargs.pop('label_variance_in_lambda_opt', 0.0)
        self.label_variance_in_eta_opt = kwargs.pop('label_variance_in_eta_opt', 0.0)
        self.make_psd_eps = kwargs.pop('make_psd_eps', 1e-9)
        self.lambda2_optimization_range = kwargs.pop('lambda2_optimization_range', None)

        self.alpha = None
        self.B = None
        self.B_bar = None
        self.third_stage_KRR_weights = None

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ objectives
    @staticmethod
    def _lambda_objective(lambda_, K_AWX, K_ZZ, identity_matrix, label_variance_in_lambda_opt,
                          make_psd_eps=1e-9):
        n = K_AWX.shape[0]
        ridge_weights = make_psd(K_AWX + n * lambda_ * identity_matrix, eps=make_psd_eps)
        R = torch.linalg.solve(ridge_weights, K_AWX).T
        H1 = identity_matrix - R
        H1_diag = torch.diag(H1)
        H1_tilde_inv = torch.diag(1.0 / H1_diag)
        H1_tilde_inv_times_H1 = H1_tilde_inv @ H1
        objective = (1.0 / n) * torch.trace(H1_tilde_inv_times_H1 @ K_ZZ @ H1_tilde_inv_times_H1)
        objective = objective + label_variance_in_lambda_opt * torch.trace(R)
        objective = objective + (1.0 / n) * label_variance_in_lambda_opt * torch.sum(
            (H1_diag - 1.0) / H1_diag)
        objective = objective + (1.0 / n) * label_variance_in_lambda_opt * torch.trace(
            R @ H1_tilde_inv @ R.T)
        return objective

    @staticmethod
    def _zeta_objective(zeta, K_ATildeATilde, K_WXTilde):
        m = K_ATildeATilde.shape[0]
        eye = torch.eye(m, dtype=K_ATildeATilde.dtype, device=K_ATildeATilde.device)
        # make_psd at its DEFAULT eps (1e-10), not make_psd_eps -- reference behaviour.
        R = torch.linalg.solve(make_psd(K_ATildeATilde + m * zeta * eye), K_ATildeATilde)
        S = torch.diag((1.0 / (1.0 - torch.diag(R))) ** 2)
        T = S @ (K_WXTilde - 2 * K_WXTilde @ R.T + R @ K_WXTilde @ R.T)
        return torch.trace(T)

    @staticmethod
    def _eta_objective(eta, L, M, N, L2, M2, stage1_data_size, label_variance_in_eta_opt,
                       make_psd_eps=1e-9):
        m = L.shape[0] - 1
        alpha = torch.linalg.solve(make_psd(L / m + eta * N, make_psd_eps), M)
        cost = (1.0 / stage1_data_size) * (alpha.T @ make_psd(L2, make_psd_eps) @ alpha) \
            - 2.0 * (alpha.T @ M2)
        cost = cost + label_variance_in_eta_opt * (2.0 / m) * torch.trace(
            torch.linalg.solve(make_psd(L + m * eta * N, make_psd_eps), L))
        return cost.reshape(())

    @staticmethod
    def _lambda2_objective(lambda2_, K_AA, K_ZZ, Y, label_variance_in_lambda_opt,
                           make_psd_eps=1e-9):
        n = K_AA.shape[0]
        identity_matrix = torch.eye(n, dtype=K_AA.dtype, device=K_AA.device)
        ridge_weights = make_psd(K_AA + n * lambda2_ * identity_matrix, eps=make_psd_eps)
        R = torch.linalg.solve(ridge_weights, K_AA).T
        H2 = identity_matrix - R
        H2_diag = torch.diag(H2)
        H2_tilde_inv = torch.diag(1.0 / H2_diag)
        kernel_output = K_ZZ * (Y @ Y.T)
        H2_tilde_inv_times_H2 = H2_tilde_inv @ H2
        # NOTE the trailing.T here, ABSENT in _lambda_objective. S is not symmetric, so these are
        # genuinely different objectives; both are reproduced as written.
        objective = (1.0 / n) * torch.trace(
            H2_tilde_inv_times_H2 @ kernel_output @ H2_tilde_inv_times_H2.T)
        objective = objective + label_variance_in_lambda_opt * torch.trace(R)
        objective = objective + (1.0 / n) * label_variance_in_lambda_opt * torch.sum(
            (H2_diag - 1.0) / H2_diag)
        objective = objective + (1.0 / n) * label_variance_in_lambda_opt * torch.trace(
            R @ H2_tilde_inv @ R.T)
        return objective

    @staticmethod
    def _geomspace(lo, hi, n, dtype, device):
        """A logspace grid(log(lo), log(hi), n, base=e) == a geometric grid lo -> hi."""
        return torch.logspace(float(np.log(lo)), float(np.log(hi)), n,
                              base=float(np.exp(1.0)), dtype=dtype, device=device)

    # ------------------------------------------------------------------ fit
    def fit(self,
            AZWX: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
            Y: torch.Tensor,
            aprime) -> None:
        """Fit the treatment bridge for ONE anchor a'. Tuple order (A, Z, W[, X])."""
        if len(AZWX) == 4:
            A, Z, W, X = AZWX
        else:
            A, Z, W = AZWX
            X = None

        dtype, device = A.dtype, A.device
        if not torch.is_tensor(aprime):
            aprime = torch.tensor(aprime, dtype=dtype, device=device)
        aprime = aprime.reshape(-1, 1).to(dtype=dtype, device=device)

        K_ATrainATrain = self.kernel_A(A, A)
        K_WTrainWTrain = self.kernel_W(W, W)
        K_ZTrainZTrain = self.kernel_Z(Z, Z)
        if X is None:
            K_XTrainXTrain = torch.ones((W.shape[0], W.shape[0]), dtype=dtype, device=device)
        else:
            K_XTrainXTrain = self.kernel_X(X, X)          # NO make_psd here (reference ATT behaviour)

        train_data_size = A.shape[0]
        train_indices = np.random.permutation(train_data_size)     # GLOBAL numpy RNG
        if (self.stage1_perc > 0.0) and (self.stage1_perc < 1.0):
            n = int(train_data_size * self.stage1_perc)
            m = train_data_size - n
            stage1_idx, stage2_idx = train_indices[:n], train_indices[n:]
        else:
            n, m = train_data_size, train_data_size
            stage1_idx, stage2_idx = train_indices, train_indices

        sub = lambda K, i1, i2: K[i1][:, i2]
        K_AA = sub(K_ATrainATrain, stage1_idx, stage1_idx)
        K_AATilde = sub(K_ATrainATrain, stage1_idx, stage2_idx)
        K_ATildeA = K_AATilde.T
        K_ATildeATilde = sub(K_ATrainATrain, stage2_idx, stage2_idx)
        K_WW = sub(K_WTrainWTrain, stage1_idx, stage1_idx)
        K_WWTilde = sub(K_WTrainWTrain, stage1_idx, stage2_idx)
        K_WTildeWTilde = sub(K_WTrainWTrain, stage2_idx, stage2_idx)
        K_ZZ = sub(K_ZTrainZTrain, stage1_idx, stage1_idx)
        K_XX = sub(K_XTrainXTrain, stage1_idx, stage1_idx)
        K_XXTilde = sub(K_XTrainXTrain, stage1_idx, stage2_idx)
        K_XTildeXTilde = sub(K_XTrainXTrain, stage2_idx, stage2_idx)

        for kernel_ in [self.kernel_A, self.kernel_W, self.kernel_Z, self.kernel_X]:
            if hasattr(kernel_, 'use_length_scale_heuristic'):
                kernel_.use_length_scale_heuristic = False

        K_ATildeaprime = self.kernel_A(A[stage2_idx], aprime)                     # (m, 1)
        I_n = torch.eye(n, dtype=dtype, device=device)
        K_AWX = K_AA * K_WW * K_XX

        if self.optimize_lambda_parameters:
            grid = self._geomspace(*self.lambda_optimization_range,
                                   self.regularization_grid_points, dtype, device)
            obj = torch.stack([self._lambda_objective(l, K_AWX, K_ZZ, I_n,
                                                      self.label_variance_in_lambda_opt,
                                                      self.make_psd_eps) for l in grid])
            self.lambda_ = float(grid[int(torch.argmin(obj))])
            self.lambda_objective_list, self.lambda_list = obj, grid

        zeta_list = self._geomspace(*self.zeta_optimization_range,
                                    self.regularization_grid_points, dtype, device)
        if self.optimize_zeta_parameter:
            K_WX_Tilde = K_WTildeWTilde * K_XTildeXTilde
            obj = torch.stack([self._zeta_objective(z, K_ATildeATilde, K_WX_Tilde) for z in zeta_list])
            self.zeta = float(zeta_list[int(torch.argmin(obj))])
            self.zeta_objective_list, self.zeta_list = obj, zeta_list

        # ---- stages 1 and 2 -- the a'-conditional mean embedding is the whole ATE->ATT delta ----
        stage1_ridge_weights = K_AWX + n * self.lambda_ * I_n
        eye_m = torch.eye(m, dtype=dtype, device=device)
        # make_psd at DEFAULT eps 1e-10, with m*zeta*I added OUTSIDE -- reference behaviour.
        theta = torch.linalg.solve(make_psd(K_ATildeATilde) + m * self.zeta * eye_m, K_ATildeaprime)
        ones_matrix_m = remove_diagonal_elements(torch.ones((m, m), dtype=dtype, device=device))
        # B / B_bar solve against the RAW ridge weights: NO make_psd (the ATE wraps both).
        B = torch.linalg.solve(stage1_ridge_weights, K_WWTilde * K_XXTilde * K_AATilde)
        B_bar = torch.linalg.solve(
            stage1_ridge_weights,
            ((K_WWTilde * K_XXTilde) @ (theta * ones_matrix_m)) * K_AATilde)

        block_component1 = (B.T @ K_ZZ @ B) * K_ATildeATilde
        block_component2 = (B.T @ K_ZZ @ B_bar) * K_ATildeATilde
        block_component4 = (B_bar.T @ K_ZZ @ B_bar) * K_ATildeATilde
        ones_divided_by_m = torch.ones(m, dtype=dtype, device=device) / m

        bc2_om = (block_component2 @ ones_divided_by_m).reshape(-1, 1)             # (m, 1)
        bc4_omom = (ones_divided_by_m @ block_component4 @ ones_divided_by_m).reshape(-1, 1)
        L = torch.vstack((block_component1,
                          (ones_divided_by_m @ block_component2.T).reshape(1, -1))) \
            @ torch.hstack((block_component1, bc2_om))
        M = torch.vstack((bc2_om, bc4_omom))
        P = torch.hstack((block_component1, bc2_om))
        R_blk = torch.hstack(((ones_divided_by_m @ block_component2.T).reshape(1, -1), bc4_omom))
        N = torch.vstack((P, R_blk))

        if self.optimize_eta_parameter:
            # eta is selected on the STAGE-1 fold used as validation, which needs its own CME.
            K_WX = K_WW * K_XX
            zeta2_list = self._geomspace(*self.zeta_optimization_range,
                                         self.regularization_grid_points, dtype, device)
            obj2 = torch.stack([self._zeta_objective(z, K_AA, K_WX) for z in zeta2_list])
            # the reference indexes zeta_list (not zeta2_list) here; the values are identical.
            zeta2 = float(zeta_list[int(torch.argmin(obj2))])
            K_Aaprime = self.kernel_A(A[stage1_idx], aprime)
            theta2 = torch.linalg.solve(make_psd(K_AA) + n * zeta2 * I_n, K_Aaprime)
            ones_matrix_n = remove_diagonal_elements(torch.ones((n, n), dtype=dtype, device=device))
            B2 = torch.linalg.solve(make_psd(stage1_ridge_weights, self.make_psd_eps), K_AWX)
            B2_bar = torch.linalg.solve(stage1_ridge_weights,
                                        (K_WX @ (theta2 * ones_matrix_n)) * K_AA)
            ones_divided_by_n = torch.ones(n, dtype=dtype, device=device) / n

            block_component12 = (B2.T @ K_ZZ @ B) * K_AATilde
            block_component22 = (B2.T @ K_ZZ @ B_bar) * K_AATilde
            block_component32 = (B.T @ K_ZZ @ B2_bar) * K_ATildeA
            block_component42 = (B_bar.T @ K_ZZ @ B2_bar) * K_ATildeA

            L2_sub = torch.vstack((block_component12.T,
                                   (ones_divided_by_m @ block_component22.T).reshape(1, -1)))
            L2 = L2_sub @ L2_sub.T
            M2 = torch.vstack(((block_component32 @ ones_divided_by_n).reshape(-1, 1),
                               (ones_divided_by_m @ block_component42 @ ones_divided_by_n).reshape(-1, 1)))

            eta_list = self._geomspace(*self.eta_optimization_range,
                                       self.regularization_grid_points, dtype, device)
            obj = torch.stack([self._eta_objective(e, L, M, N, L2, M2, n,
                                                   self.label_variance_in_eta_opt,
                                                   self.make_psd_eps) for e in eta_list])
            self.eta = float(eta_list[int(torch.argmin(obj))])
            self.eta_objective_list, self.eta_list = obj, eta_list

        alpha = torch.linalg.solve(make_psd(L / m + self.eta * N, self.make_psd_eps), M)

        # ---- third stage: KRR of the H_Z-valued target Y*phi_Z(Z) on A, over ALL training rows ----
        if self.optimize_lambda_parameters:
            rng2 = self.lambda2_optimization_range or self.lambda_optimization_range
            grid = self._geomspace(*rng2, self.regularization_grid_points, dtype, device)
            obj = torch.stack([self._lambda2_objective(l, K_ATrainATrain, K_ZTrainZTrain, Y,
                                                       self.label_variance_in_lambda_opt,
                                                       self.make_psd_eps) for l in grid])
            self.lambda2_ = float(grid[int(torch.argmin(obj))])
            self.lambda2_objective_list = obj

        K_ZZtrain = sub(K_ZTrainZTrain, stage1_idx, train_indices)                # (n, N)
        K_ATrainATrain_ = sub(K_ATrainATrain, train_indices, train_indices)       # (N, N)
        eye_N = torch.eye(train_data_size, dtype=dtype, device=device)
        third_stage_KRR_weights = torch.linalg.solve(                             # DEFAULT eps 1e-10
            make_psd(K_ATrainATrain_ + train_data_size * self.lambda2_ * eye_N),
            K_ZZtrain.T * Y[train_indices]).T                                     # (n, N)

        self.alpha = alpha
        self.B, self.B_bar = B, B_bar
        self.third_stage_KRR_weights = third_stage_KRR_weights
        self.ones_divided_by_m = ones_divided_by_m
        self.ATrain, self.ZTrain = A, Z
        self.train_indices = train_indices
        self.stage1_idx, self.stage2_idx = stage1_idx, stage2_idx
        self.aprime = aprime

    # ------------------------------------------------------------------ predict
    def predict(self, A: torch.Tensor, outcome_transformer=None) -> torch.Tensor:
        """f_ATT(a, a') on the grid `A`; a' is the anchor baked in at fit time."""
        A_test = A if A.ndim == 2 else A.reshape(-1, 1)
        K_ATrainATest = self.kernel_A(self.ATrain, A_test)                        # (N, T)
        K_ATrainATest_ = K_ATrainATest[self.train_indices]                        # (N, T)
        K_ATildeATest = K_ATrainATest[self.stage2_idx]                            # (m, T)

        # vectorised form of the reference's per-test-point loop
        V = self.third_stage_KRR_weights @ K_ATrainATest_                         # (n, T)
        term1 = (self.B.T @ V) * K_ATildeATest                                    # (m, T)
        term2 = (self.B_bar.T @ V) * K_ATildeATest                                # (m, T)
        pred = (self.alpha[:-1].T @ term1).reshape(-1) \
            + self.alpha[-1] * (self.ones_divided_by_m @ term2)                   # (T,)

        if outcome_transformer is not None:
            pred = outcome_transformer.inverse_transform(pred.reshape(-1, 1))
        return pred.reshape(-1)

    def _predict_bridge_func(self, A_test: torch.Tensor, Z_test: torch.Tensor, X_test=None):
        """phi_hat(a, z) evaluated on the test grids; shape (T_A, T_Z). X_test is ignored, as in the
        reference (the tensor-product X slot is not implemented in the bridge evaluator)."""
        if A_test.ndim != 2:
            A_test = A_test.reshape(-1, 1)
        K_ZZTest = self.kernel_Z(self.ZTrain[self.stage1_idx, :], Z_test)         # (n, T_Z)
        K_ATildeATest = self.kernel_A(self.ATrain[self.stage2_idx, :], A_test)    # (m, T_A)
        base1 = self.B.T @ K_ZZTest                                               # (m, T_Z)
        base2 = self.B_bar.T @ K_ZZTest                                           # (m, T_Z)
        out1 = torch.einsum('mt,mz,mo->tz', K_ATildeATest, base1, self.alpha[:-1])
        out2 = torch.einsum('mt,mz,m->tz', K_ATildeATest, base2, self.ones_divided_by_m)
        return out1 + self.alpha[-1] * out2
