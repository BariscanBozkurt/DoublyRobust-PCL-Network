import numpy as np
import torch
from torch import nn
from typing import Callable, Optional, Tuple
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm import tqdm
from typing import Callable, Tuple, Optional, Union, Dict, Any

import os
import sys
# Get the path of the current script, go up one level to the root
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from torch_utils.scalers import TorchIdentityTransformer
from torch_utils.linear_algebra import make_psd, columns_mean_excluding_self

import copy
from torch_utils.kernel_utils import RBF
from torch_utils.ml_utils import lambda_objective_loocv   # deploy port: was called at 4 sites but never imported (latent NameError when any optimize_* flag is set)

def _as_2d_tensor(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = x.to(device)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def _ix(K: torch.Tensor, rows, cols) -> torch.Tensor:
    rows = torch.as_tensor(rows, dtype=torch.long, device=K.device)
    cols = torch.as_tensor(cols, dtype=torch.long, device=K.device)
    return K.index_select(0, rows).index_select(1, cols)


def _take_rows(X: torch.Tensor, rows) -> torch.Tensor:
    rows = torch.as_tensor(rows, dtype=torch.long, device=X.device)
    return X.index_select(0, rows)


class KernelAlternativeProxyDoseResponseNystrom(BaseEstimator, RegressorMixin, nn.Module):
    """
    Nyström version of KernelAlternativeProxyDoseResponse.

    Approximation is used in:
      - Stage 1 conditional embedding solve.
      - Third-stage KRR solve.

    The main alpha / eta second-stage system is kept exact over the Stage-2 sample,
    matching your stated plan.
    """

    def __init__(
        self,
        kernel_A: Callable,
        kernel_W: Callable,
        kernel_Z: Callable,
        kernel_X: Optional[Callable] = None,
        lambda1_: float = 0.1,
        eta: float = 0.1,
        lambda2_: float = 0.1,
        nystrom_first_stage_m: int = 500,
        nystrom_third_stage_m: int = 500,
        device=None,
        **kwargs,
    ) -> None:
        nn.Module.__init__(self)

        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_X = kernel_X if kernel_X is not None else RBF()

        self.lambda1_ = lambda1_
        self.eta = eta
        self.lambda2_ = lambda2_

        self.nystrom_first_stage_m = nystrom_first_stage_m
        self.nystrom_third_stage_m = nystrom_third_stage_m

        self.optimize_lambda_parameters = kwargs.pop("optimize_lambda_parameters", False)
        self.optimize_eta_parameter = kwargs.pop("optimize_eta_parameter", False)
        self.lambda_optimization_range = kwargs.pop("lambda_optimization_range", (1e-7, 1.0))
        self.eta_optimization_range = kwargs.pop("eta_optimization_range", (1e-7, 1.0))
        self.regularization_grid_points = kwargs.pop("regularization_grid_points", 25)
        self.make_psd_eps = kwargs.pop("make_psd_eps", 1e-7)
        self.stage1_perc = kwargs.pop("stage1_perc", 0.5)
        self.model_seed = kwargs.pop("model_seed", 0)
        self.label_variance_in_lambda_opt = kwargs.pop("label_variance_in_lambda_opt", 0.0)
        self.label_variance_in_eta_opt = kwargs.pop("label_variance_in_eta_opt", 0.0)

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.alpha = None
        self.B = None
        self.B_bar = None
        self.third_stage_KRR_weights = None
        self.ones_divided_by_m = None

        self.ATrain = None
        self.WTrain = None
        self.ZTrain = None
        self.XTrain = None

        self.K_ZZ = None
        self.train_indices = None
        self.stage1_idx = None
        self.stage2_idx = None

        self.nystrom_stage1_landmarks = None
        self.K_AWX_nm = None
        self.stage1_ridge_weights = None

    def sample_landmarks(self, original_size: int, m: int, seed: int = 0):
        if m > original_size:
            m = original_size
        rng = np.random.default_rng(seed)
        return rng.choice(original_size, m, replace=False)

    @staticmethod
    def _eta_objective(
        eta,
        L,
        L_sub,
        M,
        N,
        L2,
        M2,
        stage1_data_size,
        label_variance_in_eta_opt=0.0,
        make_psd_eps=1e-9,
    ):
        stage2_data_size = L.shape[0] - 1
        alpha = torch.linalg.solve(make_psd(L / stage2_data_size + eta * N, make_psd_eps), M)
        cost = (1.0 / stage1_data_size) * (alpha.T @ make_psd(L2, make_psd_eps) @ alpha) - 2.0 * (alpha.T @ M2)
        cost += label_variance_in_eta_opt * (2.0 / stage2_data_size) * torch.trace(
            torch.linalg.solve(make_psd(L + stage2_data_size * eta * N, make_psd_eps), L)
        )
        return cost.reshape(())

    @staticmethod
    def _predict_structural_function(
        alpha: torch.Tensor,
        B: torch.Tensor,
        B_bar: torch.Tensor,
        third_stage_KRR_weights: torch.Tensor,
        K_ATraina: torch.Tensor,
        K_ATildea: torch.Tensor,
        ones_divided_by_m: torch.Tensor,
    ) -> torch.Tensor:
        K_ATraina = K_ATraina.reshape(-1, 1)
        K_ATildea = K_ATildea.reshape(-1, 1)

        u = third_stage_KRR_weights @ K_ATraina

        term1 = (B.T @ u) * K_ATildea
        pred = alpha[:-1].T @ term1

        term2 = (B_bar.T @ u) * K_ATildea
        pred = pred + alpha[-1].reshape(1, 1) * (term2.T @ ones_divided_by_m)

        return pred.reshape(())

    def fit(
        self,
        AZWX: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        Y: torch.Tensor,
    ) -> None:
        if len(AZWX) == 4:
            A, Z, W, X = AZWX
        elif len(AZWX) == 3:
            A, Z, W = AZWX
            X = None
        else:
            raise ValueError("AZWX must be either (A, Z, W) or (A, Z, W, X).")

        A = _as_2d_tensor(A, self.device)
        Z = _as_2d_tensor(Z, self.device)
        W = _as_2d_tensor(W, self.device)
        Y = _as_2d_tensor(Y, self.device)
        X = _as_2d_tensor(X, self.device) if X is not None else None

        dtype = torch.get_default_dtype()

        K_ATrainATrain = self.kernel_A(A, A)
        K_WTrainWTrain = self.kernel_W(W, W)
        K_ZTrainZTrain = self.kernel_Z(Z, Z)

        if X is None:
            K_XTrainXTrain = torch.ones((W.shape[0], W.shape[0]), dtype=K_ATrainATrain.dtype, device=self.device)
        else:
            K_XTrainXTrain = make_psd(self.kernel_X(X, X), self.make_psd_eps)

        for kernel_ in [self.kernel_A, self.kernel_W, self.kernel_Z, self.kernel_X]:
            if hasattr(kernel_, "use_length_scale_heuristic"):
                kernel_.use_length_scale_heuristic = False

        train_data_size = A.shape[0]
        train_indices = np.random.permutation(train_data_size)

        if 0.0 < self.stage1_perc < 1.0:
            stage1_data_size = int(train_data_size * self.stage1_perc)
            stage2_data_size = train_data_size - stage1_data_size
            stage1_idx = train_indices[:stage1_data_size]
            stage2_idx = train_indices[stage1_data_size:]
        else:
            stage1_data_size = train_data_size
            stage2_data_size = train_data_size
            stage1_idx = train_indices
            stage2_idx = train_indices

        nystrom_stage1_landmarks = self.sample_landmarks(
            stage1_data_size,
            self.nystrom_first_stage_m,
            self.model_seed,
        )
        nystrom_stage3_landmarks = self.sample_landmarks(
            train_data_size,
            self.nystrom_third_stage_m,
            self.model_seed + 1,
        )

        K_AA = _ix(K_ATrainATrain, stage1_idx, stage1_idx)
        K_AATilde = _ix(K_ATrainATrain, stage1_idx, stage2_idx)
        K_ATildeATilde = _ix(K_ATrainATrain, stage2_idx, stage2_idx)

        K_WW = _ix(K_WTrainWTrain, stage1_idx, stage1_idx)
        K_WWTilde = _ix(K_WTrainWTrain, stage1_idx, stage2_idx)

        K_ZZ = _ix(K_ZTrainZTrain, stage1_idx, stage1_idx)

        K_XX = _ix(K_XTrainXTrain, stage1_idx, stage1_idx)
        K_XXTilde = _ix(K_XTrainXTrain, stage1_idx, stage2_idx)

        K_AWX = K_AA * K_WW * K_XX

        if self.optimize_lambda_parameters:
            l_min, l_max = self.lambda_optimization_range
            lambda_list = torch.exp(
                torch.linspace(
                    torch.log(torch.tensor(l_min, dtype=dtype, device=self.device)),
                    torch.log(torch.tensor(l_max, dtype=dtype, device=self.device)),
                    self.regularization_grid_points,
                    device=self.device,
                )
            )
            objective_list = torch.zeros(self.regularization_grid_points, dtype=dtype, device=self.device)
            with torch.no_grad():
                for i, lambda_val in tqdm(enumerate(lambda_list), total=len(lambda_list), desc="Tuning Lambda 1"):
                    objective_list[i] = lambda_objective_loocv(
                        lambda_val,
                        K_AWX,
                        K_ZZ,
                        self.label_variance_in_lambda_opt,
                        self.make_psd_eps,
                    )
            self.lambda1_ = lambda_list[torch.argmin(objective_list)].item()

        K_AWX_nm = K_AWX[:, nystrom_stage1_landmarks]
        K_AWX_mm = K_AWX[nystrom_stage1_landmarks][:, nystrom_stage1_landmarks]

        stage1_ridge_weights = K_AWX_nm.T @ K_AWX_nm + stage1_data_size * self.lambda1_ * K_AWX_mm
        stage1_ridge_weights = make_psd(stage1_ridge_weights, self.make_psd_eps)

        K_A_m_ATilde = K_AATilde[nystrom_stage1_landmarks, :]
        K_WX_m_WXTilde = (K_WWTilde * K_XXTilde)[nystrom_stage1_landmarks, :]

        B = K_AWX_nm @ torch.linalg.solve(stage1_ridge_weights, K_WX_m_WXTilde * K_A_m_ATilde)
        B_bar = K_AWX_nm @ torch.linalg.solve(
            stage1_ridge_weights,
            columns_mean_excluding_self(K_WX_m_WXTilde) * K_A_m_ATilde,
        )

        block_component1 = (B.T @ K_ZZ @ B) * K_ATildeATilde
        block_component2 = (B.T @ K_ZZ @ B_bar) * K_ATildeATilde
        block_component4 = (B_bar.T @ K_ZZ @ B_bar) * K_ATildeATilde

        ones_divided_by_m = torch.ones((stage2_data_size, 1), dtype=K_AWX.dtype, device=self.device) / stage2_data_size

        L_sub = torch.vstack((block_component1, ones_divided_by_m.T @ block_component2.T))
        L = L_sub @ L_sub.T

        M = torch.vstack(
            (
                (block_component2 @ ones_divided_by_m).reshape(-1, 1),
                (ones_divided_by_m.T @ block_component4 @ ones_divided_by_m).reshape(-1, 1),
            )
        )

        P = torch.hstack((block_component1, (block_component2 @ ones_divided_by_m).reshape(-1, 1)))
        R = torch.hstack(
            (
                (ones_divided_by_m.T @ block_component2.T).reshape(1, -1),
                (ones_divided_by_m.T @ block_component4 @ ones_divided_by_m).reshape(-1, 1),
            )
        )
        N = torch.vstack((P, R))

        if self.optimize_eta_parameter:
            K_ATildeA = K_AATilde.T

            B2 = K_AWX_nm @ torch.linalg.solve(stage1_ridge_weights, K_AWX[nystrom_stage1_landmarks, :])

            K_WX = K_WW * K_XX
            B2_bar_target = columns_mean_excluding_self(K_WX) * K_AA
            B2_bar = K_AWX_nm @ torch.linalg.solve(stage1_ridge_weights, B2_bar_target[nystrom_stage1_landmarks, :])

            ones_divided_by_n = torch.ones((stage1_data_size, 1), dtype=K_AWX.dtype, device=self.device) / stage1_data_size

            block_component12 = (B2.T @ K_ZZ @ B) * K_AATilde
            block_component22 = (B2.T @ K_ZZ @ B_bar) * K_AATilde
            block_component32 = (B.T @ K_ZZ @ B2_bar) * K_ATildeA
            block_component42 = (B_bar.T @ K_ZZ @ B2_bar) * K_ATildeA

            L2_sub = torch.vstack((block_component12.T, ones_divided_by_m.T @ block_component22.T))
            L2 = L2_sub @ L2_sub.T
            M2 = torch.vstack(
                (
                    (block_component32 @ ones_divided_by_n).reshape(-1, 1),
                    (ones_divided_by_m.T @ block_component42 @ ones_divided_by_n).reshape(-1, 1),
                )
            )

            e_min, e_max = self.eta_optimization_range
            eta_list = torch.exp(
                torch.linspace(
                    torch.log(torch.tensor(e_min, dtype=dtype, device=self.device)),
                    torch.log(torch.tensor(e_max, dtype=dtype, device=self.device)),
                    self.regularization_grid_points,
                    device=self.device,
                )
            )
            objective_list = torch.zeros(self.regularization_grid_points, dtype=dtype, device=self.device)

            with torch.no_grad():
                for i, eta_val in tqdm(enumerate(eta_list), total=len(eta_list), desc="Tuning Eta"):
                    objective_list[i] = self._eta_objective(
                        eta_val,
                        L,
                        L_sub,
                        M,
                        N,
                        L2,
                        M2,
                        stage1_data_size,
                        self.label_variance_in_eta_opt,
                        self.make_psd_eps,
                    )
            self.eta = eta_list[torch.argmin(objective_list)].item()

        alpha = torch.linalg.solve(make_psd(L / stage2_data_size + self.eta * N, self.make_psd_eps), M)

        if self.optimize_lambda_parameters:
            objective_list = torch.zeros(self.regularization_grid_points, dtype=dtype, device=self.device)
            K_Y = Y @ Y.T
            with torch.no_grad():
                for i, lambda_val in tqdm(enumerate(lambda_list), total=len(lambda_list), desc="Tuning Lambda 2"):
                    objective_list[i] = lambda_objective_loocv(
                        lambda_val,
                        K_ATrainATrain,
                        K_ZTrainZTrain * K_Y,
                        self.label_variance_in_lambda_opt,
                        self.make_psd_eps,
                    )
            self.lambda2_ = lambda_list[torch.argmin(objective_list)].item()

        K_ZZTrain = _ix(K_ZTrainZTrain, stage1_idx, train_indices)
        K_ATrainATrain_ = _ix(K_ATrainATrain, train_indices, train_indices)
        K_ATrainATrain_nm = K_ATrainATrain_[:, nystrom_stage3_landmarks]
        K_ATrainATrain_mm = K_ATrainATrain_[nystrom_stage3_landmarks][:, nystrom_stage3_landmarks]

        KRR_LHS = make_psd(
            K_ATrainATrain_nm.T @ K_ATrainATrain_nm
            + train_data_size * self.lambda2_ * K_ATrainATrain_mm,
            self.make_psd_eps,
        )
        KRR_RHS = K_ATrainATrain_nm.T @ (K_ZZTrain.T * Y[train_indices])
        third_stage_KRR_weights = torch.linalg.solve(KRR_LHS, KRR_RHS).T

        self.alpha = alpha
        self.B = B
        self.B_bar = B_bar
        self.third_stage_KRR_weights = third_stage_KRR_weights
        self.ones_divided_by_m = ones_divided_by_m

        self.ATrain = A
        self.WTrain = W
        self.ZTrain = Z
        self.XTrain = X

        self.K_ZZ = K_ZZ
        self.train_indices = train_indices[nystrom_stage3_landmarks]
        self.stage1_idx = stage1_idx
        self.stage2_idx = stage2_idx

        self.nystrom_stage1_landmarks = nystrom_stage1_landmarks
        self.K_AWX_nm = K_AWX_nm
        self.stage1_ridge_weights = stage1_ridge_weights

    def predict(self, A: torch.Tensor, outcome_transformer: nn.Module = TorchIdentityTransformer()) -> torch.Tensor:
        A_test = _as_2d_tensor(A, self.device)

        K_ATrainATest = self.kernel_A(self.ATrain, A_test)
        K_ATrainATest_ = _ix(K_ATrainATest, self.train_indices, np.arange(A_test.shape[0]))
        K_ATildeATest = _ix(K_ATrainATest, self.stage2_idx, np.arange(A_test.shape[0]))

        f_struct_pred = torch.vstack(
            [
                self._predict_structural_function(
                    self.alpha,
                    self.B,
                    self.B_bar,
                    self.third_stage_KRR_weights,
                    K_ATrainATest_[:, jj],
                    K_ATildeATest[:, jj],
                    self.ones_divided_by_m,
                )
                for jj in range(K_ATildeATest.shape[1])
            ]
        )

        f_struct_pred = outcome_transformer.inverse_transform(f_struct_pred.squeeze(-1))
        return f_struct_pred

    def _predict_bridge_func(self, A_test: torch.Tensor, Z_test: torch.Tensor, X_test=None):
        A_test = _as_2d_tensor(A_test, self.device)
        Z_test = _as_2d_tensor(Z_test, self.device)

        K_ZZTest = self.kernel_Z(_take_rows(self.ZTrain, self.stage1_idx), Z_test)
        K_ATildeATest = self.kernel_A(_take_rows(self.ATrain, self.stage2_idx), A_test)

        bridge_function = torch.vstack(
            [
                self.alpha[:-1].T @ ((self.B.T @ K_ZZTest) * K_ATildeATest[:, jj].reshape(-1, 1))
                + self.alpha[-1].reshape(1, 1)
                * (self.ones_divided_by_m.T @ ((self.B_bar.T @ K_ZZTest) * K_ATildeATest[:, jj].reshape(-1, 1)))
                for jj in range(K_ATildeATest.shape[1])
            ]
        )
        return bridge_function

    def _predict_density_ratio(self, A_test: torch.Tensor, W_test: torch.Tensor, X_test: Optional[torch.Tensor] = None):
        A_test = _as_2d_tensor(A_test, self.device)
        W_test = _as_2d_tensor(W_test, self.device)
        X_test = _as_2d_tensor(X_test, self.device) if X_test is not None else None

        A_stage1 = _take_rows(self.ATrain, self.stage1_idx)
        W_stage1 = _take_rows(self.WTrain, self.stage1_idx)

        K_WWTest = self.kernel_W(W_stage1, W_test)
        K_AATest = self.kernel_A(A_stage1, A_test)

        if X_test is None or self.XTrain is None:
            K_XXTest = torch.ones_like(K_AATest)
        else:
            X_stage1 = _take_rows(self.XTrain, self.stage1_idx)
            K_XXTest = self.kernel_X(X_stage1, X_test)

        target = K_WWTest * K_XXTest * K_AATest
        B_test = self.K_AWX_nm @ torch.linalg.solve(
            self.stage1_ridge_weights,
            target[self.nystrom_stage1_landmarks, :],
        )

        K_ATildeATest = self.kernel_A(_take_rows(self.ATrain, self.stage2_idx), A_test)

        dens_ratio = (
            self.alpha[:-1].T @ ((self.B.T @ self.K_ZZ @ B_test) * K_ATildeATest)
            + self.alpha[-1].reshape(1, 1)
            * (self.ones_divided_by_m.T @ ((self.B_bar.T @ self.K_ZZ @ B_test) * K_ATildeATest))
        )
        return dens_ratio.T


class KernelProxyVariableDoseResponseNystrom(BaseEstimator, RegressorMixin, nn.Module):
    """
    Nyström version of KernelProxyVariableDoseResponse.

    Approximation is used in:
      - Stage 1 conditional mean embedding solve.
      - Stage 2 outcome bridge solve.
    """

    def __init__(
        self,
        kernel_A: Callable,
        kernel_W: Callable,
        kernel_Z: Callable,
        kernel_X: Optional[Callable] = None,
        lambda1_: float = 0.1,
        lambda2_: float = 0.1,
        nystrom_first_stage_m: int = 500,
        nystrom_second_stage_m: int = 500,
        optimize_lambda1_parameter: bool = False,
        optimize_lambda2_parameter: bool = False,
        lambda1_optimization_range: Tuple[float, float] = (1e-5, 1.0),
        lambda2_optimization_range: Tuple[float, float] = (1e-5, 1.0),
        device=None,
        **kwargs,
    ) -> None:
        nn.Module.__init__(self)

        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_X = kernel_X if kernel_X is not None else RBF()

        self.lambda1_ = lambda1_
        self.lambda2_ = lambda2_

        self.nystrom_first_stage_m = nystrom_first_stage_m
        self.nystrom_second_stage_m = nystrom_second_stage_m

        self.optimize_lambda1_parameter = optimize_lambda1_parameter
        self.optimize_lambda2_parameter = optimize_lambda2_parameter
        self.lambda1_optimization_range = lambda1_optimization_range
        self.lambda2_optimization_range = lambda2_optimization_range

        self.stage1_perc = kwargs.pop("stage1_perc", 0.5)
        self.regularization_grid_points = kwargs.pop("regularization_grid_points", 25)
        self.make_psd_eps = kwargs.pop("make_psd_eps", 1e-9)
        self.label_variance_in_lambda_opt = kwargs.pop("label_variance_in_lambda_opt", 0.0)
        self.model_seed = kwargs.pop("model_seed", 0)

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.alpha = None
        self.B = None
        self.w_mean_vec = None
        self.x_mean_vec = None
        self.upweight = None

        self.stage1_idx = None
        self.stage2_idx = None
        self.nystrom_stage2_landmarks = None

        self.ATilde = None
        self.XTilde = None
        self.W = None

    def sample_landmarks(self, original_size: int, m: int, seed: int = 0):
        if m > original_size:
            m = original_size
        rng = np.random.default_rng(seed)
        return rng.choice(original_size, m, replace=False)

    def fit(
        self,
        AZWX: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        Y: torch.Tensor,
    ) -> None:
        if len(AZWX) == 4:
            A, Z, W, X = AZWX
        elif len(AZWX) == 3:
            A, Z, W = AZWX
            X = None
        else:
            raise ValueError("AZWX must be either (A, Z, W) or (A, Z, W, X).")

        A = _as_2d_tensor(A, self.device)
        Z = _as_2d_tensor(Z, self.device)
        W = _as_2d_tensor(W, self.device)
        Y = _as_2d_tensor(Y, self.device)
        X = _as_2d_tensor(X, self.device) if X is not None else None

        dtype = torch.get_default_dtype()

        K_ATrainATrain = self.kernel_A(A, A)
        K_WTrainWTrain = self.kernel_W(W, W)
        K_ZTrainZTrain = self.kernel_Z(Z, Z)

        if X is None:
            K_XTrainXTrain = torch.ones((W.shape[0], W.shape[0]), dtype=K_ATrainATrain.dtype, device=self.device)
        else:
            K_XTrainXTrain = make_psd(self.kernel_X(X, X), self.make_psd_eps)

        for kernel_ in [self.kernel_A, self.kernel_W, self.kernel_Z, self.kernel_X]:
            if hasattr(kernel_, "use_length_scale_heuristic"):
                kernel_.use_length_scale_heuristic = False

        n_total = A.shape[0]
        indices = np.random.permutation(n_total)

        if 0.0 < self.stage1_perc < 1.0:
            n1 = int(n_total * self.stage1_perc)
            stage1_idx = indices[:n1]
            stage2_idx = indices[n1:]
        else:
            n1 = n_total
            stage1_idx = indices
            stage2_idx = indices

        n2 = len(stage2_idx)

        nystrom_stage1_landmarks = self.sample_landmarks(n1, self.nystrom_first_stage_m, self.model_seed)
        nystrom_stage2_landmarks = self.sample_landmarks(n2, self.nystrom_second_stage_m, self.model_seed + 1)

        K_AATilde = _ix(K_ATrainATrain, stage1_idx, stage2_idx)
        K_ATildeATilde = _ix(K_ATrainATrain, stage2_idx, stage2_idx)

        K_WW = _ix(K_WTrainWTrain, stage1_idx, stage1_idx)

        K_ZZTilde = _ix(K_ZTrainZTrain, stage1_idx, stage2_idx)

        K_XXTilde = _ix(K_XTrainXTrain, stage1_idx, stage2_idx)
        K_XTildeXTilde = _ix(K_XTrainXTrain, stage2_idx, stage2_idx)

        K_ZAX = _ix(K_ZTrainZTrain * K_ATrainATrain * K_XTrainXTrain, stage1_idx, stage1_idx)

        if self.optimize_lambda1_parameter:
            l_min, l_max = self.lambda1_optimization_range
            lambda_list = torch.exp(
                torch.linspace(
                    torch.log(torch.tensor(l_min, dtype=dtype, device=self.device)),
                    torch.log(torch.tensor(l_max, dtype=dtype, device=self.device)),
                    self.regularization_grid_points,
                    device=self.device,
                )
            )
            obj_list = torch.zeros(self.regularization_grid_points, dtype=dtype, device=self.device)
            with torch.no_grad():
                for i, l_val in tqdm(enumerate(lambda_list), total=len(lambda_list), desc="Tuning Lambda 1"):
                    obj_list[i] = lambda_objective_loocv(
                        l_val,
                        K_ZAX,
                        K_WW,
                        self.label_variance_in_lambda_opt,
                        self.make_psd_eps,
                    )
            self.lambda1_ = lambda_list[torch.argmin(obj_list)].item()

        K_ZAX_nm = K_ZAX[:, nystrom_stage1_landmarks]
        K_ZAX_mm = K_ZAX[nystrom_stage1_landmarks][:, nystrom_stage1_landmarks]

        stage1_ridge_weights = make_psd(
            K_ZAX_nm.T @ K_ZAX_nm + n1 * self.lambda1_ * K_ZAX_mm,
            self.make_psd_eps,
        )

        K_ZAX_Tilde = K_ZZTilde * K_AATilde * K_XXTilde
        K_ZAX_m_ZAXTilde = K_ZAX_Tilde[nystrom_stage1_landmarks, :]

        B = K_ZAX_nm @ torch.linalg.solve(stage1_ridge_weights, K_ZAX_m_ZAXTilde)
        self.B = B

        stage2_ridge_weights = K_ATildeATilde * (B.T @ K_WW @ B) * K_XTildeXTilde
        YTilde = Y[stage2_idx]

        if self.optimize_lambda2_parameter:
            l_min, l_max = self.lambda2_optimization_range
            lambda_list = torch.exp(
                torch.linspace(
                    torch.log(torch.tensor(l_min, dtype=dtype, device=self.device)),
                    torch.log(torch.tensor(l_max, dtype=dtype, device=self.device)),
                    self.regularization_grid_points,
                    device=self.device,
                )
            )
            K_YTilde = YTilde @ YTilde.T
            obj_list = torch.zeros(self.regularization_grid_points, dtype=dtype, device=self.device)
            with torch.no_grad():
                for i, l_val in tqdm(enumerate(lambda_list), total=len(lambda_list), desc="Tuning Lambda 2"):
                    obj_list[i] = lambda_objective_loocv(
                        l_val,
                        stage2_ridge_weights,
                        K_YTilde,
                        self.label_variance_in_lambda_opt,
                        self.make_psd_eps,
                    )
            self.lambda2_ = lambda_list[torch.argmin(obj_list)].item()

        stage2_ridge_weights_mn = stage2_ridge_weights[nystrom_stage2_landmarks, :]
        stage2_ridge_weights_mm = stage2_ridge_weights[nystrom_stage2_landmarks][:, nystrom_stage2_landmarks]

        alpha = torch.linalg.solve(
            make_psd(
                stage2_ridge_weights_mn @ stage2_ridge_weights_mn.T
                + n2 * self.lambda2_ * stage2_ridge_weights_mm,
                self.make_psd_eps,
            ),
            stage2_ridge_weights_mn @ YTilde,
        )

        self.alpha = alpha
        self.w_mean_vec = torch.mean(K_WW, dim=0, keepdim=True).T
        self.x_mean_vec = torch.mean(K_XXTilde, dim=0, keepdim=True).T

        self.ATilde = _take_rows(A[stage2_idx], nystrom_stage2_landmarks)
        self.XTilde = _take_rows(X[stage2_idx], nystrom_stage2_landmarks) if X is not None else None
        self.W = A.new_tensor(W[stage1_idx].detach().cpu().numpy()).to(self.device)

        self.upweight = torch.mean(
            (B[:, nystrom_stage2_landmarks].T @ K_WW) * K_XXTilde[:, nystrom_stage2_landmarks].T,
            dim=1,
            keepdim=True,
        )

        self.stage1_idx = stage1_idx
        self.stage2_idx = stage2_idx
        self.nystrom_stage2_landmarks = nystrom_stage2_landmarks

        self.kernel_A = self.kernel_A
        self.kernel_W = self.kernel_W
        self.kernel_Z = self.kernel_Z
        self.kernel_X = self.kernel_X

    def predict(self, A: torch.Tensor, outcome_transformer: nn.Module = TorchIdentityTransformer()) -> torch.Tensor:
        A_test = _as_2d_tensor(A, self.device)

        K_ATildeATest = self.kernel_A(self.ATilde, A_test)
        pred = (K_ATildeATest * self.upweight).T @ self.alpha

        pred = outcome_transformer.inverse_transform(pred.squeeze(-1))
        return pred

    def _predict_bridge_func(
        self,
        A_test: torch.Tensor,
        W_test: torch.Tensor,
        X_test: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        A_test = _as_2d_tensor(A_test, self.device)
        W_test = _as_2d_tensor(W_test, self.device)
        X_test = _as_2d_tensor(X_test, self.device) if X_test is not None else None

        K_ATildeATest = self.kernel_A(self.ATilde, A_test)
        K_WWTest = self.kernel_W(self.W, W_test)

        if X_test is not None and self.XTilde is not None:
            K_XTildeXTest = self.kernel_X(self.XTilde, X_test)
        else:
            K_XTildeXTest = torch.ones((self.ATilde.shape[0], W_test.shape[0]), dtype=K_ATildeATest.dtype, device=self.device)

        B_sub = self.B[:, self.nystrom_stage2_landmarks]

        bridge_preds = []
        for jj in range(A_test.shape[0]):
            feat = K_ATildeATest[:, jj].reshape(-1, 1) * ((B_sub.T @ K_WWTest) * K_XTildeXTest)
            bridge_preds.append(feat.T @ self.alpha)

        return torch.stack(bridge_preds).squeeze(-1)


class DoublyRobustKernelProxyDoseResponseNystrom(BaseEstimator, RegressorMixin, nn.Module):
    """
    Nyström doubly robust kernel proxy estimator for dose-response estimation.

    This estimator combines:
      - KernelAlternativeProxyDoseResponseNystrom for the treatment bridge;
      - KernelProxyVariableDoseResponseNystrom for the outcome bridge;
      - a Nyström KRR smoother for the interaction/slack term.

    The fit tuple follows the torch convention:
        (A, Z, W) or (A, Z, W, X).

    Regularization parameters are fixed by we. No tuning is performed.
    """

    def __init__(
        self,
        treatment_bridge_params: Optional[Dict[str, Any]] = None,
        outcome_bridge_params: Optional[Dict[str, Any]] = None,
        lambda_DR: float = 1e-3,
        nystrom_m: int = 500,
        device=None,
        **kwargs,
    ) -> None:
        nn.Module.__init__(self)

        self.lambda_DR = lambda_DR
        self.nystrom_m = nystrom_m
        self.model_seed = kwargs.pop("model_seed", 0)
        self.make_psd_eps = kwargs.pop("make_psd_eps", 5e-9)

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if treatment_bridge_params is None:
            treatment_bridge_params = {
                "kernel_A": RBF(use_length_scale_heuristic=True),
                "kernel_W": RBF(use_length_scale_heuristic=True),
                "kernel_Z": RBF(use_length_scale_heuristic=True),
                "lambda1_": 1e-3,
                "eta": 1e-3,
                "lambda2_": 1e-3,
                "nystrom_first_stage_m": 500,
                "nystrom_third_stage_m": 500,
                "stage1_perc": 0.5,
                "model_seed": self.model_seed,
                "make_psd_eps": self.make_psd_eps,
                "device": self.device,
            }

        if outcome_bridge_params is None:
            outcome_bridge_params = {
                "algorithm_name": "Kernel_Proxy_Variable",
                "kernel_A": RBF(use_length_scale_heuristic=True),
                "kernel_W": RBF(use_length_scale_heuristic=True),
                "kernel_Z": RBF(use_length_scale_heuristic=True),
                "lambda1_": 0.1,
                "lambda2_": 0.1,
                "nystrom_first_stage_m": 500,
                "nystrom_second_stage_m": 500,
                "stage1_perc": 0.5,
                "model_seed": self.model_seed + 1,
                "make_psd_eps": self.make_psd_eps,
                "device": self.device,
            }

        self.treatment_bridge_params = self._clean_treatment_params(treatment_bridge_params)
        self.outcome_bridge_params = self._clean_outcome_params(outcome_bridge_params)

        self.treatment_bridge_algo = KernelAlternativeProxyDoseResponseNystrom(
            **self.treatment_bridge_params
        )

        algorithm_name = outcome_bridge_params.get("algorithm_name", "Kernel_Proxy_Variable")
        if algorithm_name == "Kernel_Proxy_Variable":
            self.outcome_bridge_algo = KernelProxyVariableDoseResponseNystrom(
                **self.outcome_bridge_params
            )
        else:
            raise NotImplementedError(
                f"Only Kernel_Proxy_Variable is implemented for the outcome bridge, got {algorithm_name}."
            )

        self.A = None
        self.Z = None
        self.W = None
        self.X = None
        self.Y = None
        self.kernel_A = None

        self.nystrom_landmarks = None
        self.DR_KRR_weights = None
        self.slack_prediction_raw = None
        self.treatment_bridge_algo_pred_raw = None
        self.outcome_bridge_algo_pred_raw = None

    @staticmethod
    def _clean_treatment_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove tuning-related keys and normalize parameter names.

        Some implementations name this lambda_, while this KAP class uses lambda1_.
        """
        params = copy.deepcopy(params)

        # Map old reference-style name to current torch-style name.
        if "lambda_" in params and "lambda1_" not in params:
            params["lambda1_"] = params.pop("lambda_")
        else:
            params.pop("lambda_", None)

        # Remove all tuning-related keys. Regularization is fixed.
        for key in [
            "optimize_lambda_parameters",
            "optimize_eta_parameter",
            "lambda_optimization_range",
            "eta_optimization_range",
            "regularization_grid_points",
            "label_variance_in_lambda_opt",
            "label_variance_in_eta_opt",
            "large_eta_option",
            "selecting_biggest_eta_tol",
            "zeta",
            "kernel_V",
        ]:
            params.pop(key, None)

        return params

    @staticmethod
    def _clean_outcome_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove tuning-related keys and metadata. Regularization is fixed.
        """
        params = copy.deepcopy(params)

        for key in [
            "algorithm_name",
            "optimize_lambda1_parameter",
            "optimize_lambda2_parameter",
            "lambda1_optimization_range",
            "lambda2_optimization_range",
            "regularization_grid_points",
            "label_variance_in_lambda_opt",
            "zeta",
            "kernel_V",
        ]:
            params.pop(key, None)

        return params

    def sample_landmarks(self, original_size: int, m: int, seed: int = 0):
        if m > original_size:
            m = original_size
        rng = np.random.default_rng(seed)
        return rng.choice(original_size, m, replace=False)

    @staticmethod
    def _ensure_bridge_shape(
        bridge_values: torch.Tensor,
        n_test: int,
        n_train: int,
        name: str,
    ) -> torch.Tensor:
        """
        Ensure bridge predictions have shape (n_test, n_train).
        """
        bridge_values = bridge_values.squeeze()

        if bridge_values.ndim == 1:
            if n_test == 1 and bridge_values.shape[0] == n_train:
                bridge_values = bridge_values.reshape(1, n_train)
            else:
                raise ValueError(
                    f"{name} bridge prediction has invalid 1D shape {bridge_values.shape}; "
                    f"expected ({n_test}, {n_train})."
                )

        if bridge_values.shape == (n_train, n_test):
            bridge_values = bridge_values.T

        if bridge_values.shape != (n_test, n_train):
            raise ValueError(
                f"{name} bridge prediction has shape {bridge_values.shape}; "
                f"expected ({n_test}, {n_train})."
            )

        return bridge_values

    def fit(
        self,
        AZWX: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        Y: torch.Tensor,
    ):
        if len(AZWX) == 4:
            A, Z, W, X = AZWX
        elif len(AZWX) == 3:
            A, Z, W = AZWX
            X = None
        else:
            raise ValueError("AZWX must be either (A, Z, W) or (A, Z, W, X).")

        A = _as_2d_tensor(A, self.device)
        Z = _as_2d_tensor(Z, self.device)
        W = _as_2d_tensor(W, self.device)
        Y = _as_2d_tensor(Y, self.device)
        X = _as_2d_tensor(X, self.device) if X is not None else None

        self.treatment_bridge_algo.fit((A, Z, W) if X is None else (A, Z, W, X), Y)
        self.outcome_bridge_algo.fit((A, Z, W) if X is None else (A, Z, W, X), Y)

        self.A = A
        self.Z = Z
        self.W = W
        self.X = X
        self.Y = Y
        self.kernel_A = self.treatment_bridge_algo.kernel_A

        return self

    def _compute_dr_krr_weights(self, A_test: torch.Tensor) -> torch.Tensor:
        """
        Compute Nyström KRR smoothing weights for E[phi(A_i) | A=a].

        Returns
        -------
        DR_KRR_weights:
            Tensor of shape (n_train, n_test).
        """
        n_train = self.A.shape[0]

        nystrom_landmarks = self.sample_landmarks(
            original_size=n_train,
            m=self.nystrom_m,
            seed=self.model_seed,
        )

        A_landmarks = _take_rows(self.A, nystrom_landmarks)

        K_AA_nm = self.kernel_A(self.A, A_landmarks)
        K_AA_mm = self.kernel_A(A_landmarks, A_landmarks)
        K_Aland_A_test = self.kernel_A(A_landmarks, A_test)

        lhs = make_psd(
            K_AA_nm.T @ K_AA_nm + n_train * self.lambda_DR * K_AA_mm,
            self.make_psd_eps,
        )

        DR_KRR_weights = K_AA_nm @ torch.linalg.solve(lhs, K_Aland_A_test)

        self.nystrom_landmarks = nystrom_landmarks
        self.DR_KRR_weights = DR_KRR_weights

        return DR_KRR_weights

    def predict(
        self,
        A: torch.Tensor,
        outcome_transformer: nn.Module = TorchIdentityTransformer(),
    ) -> torch.Tensor:
        """
        Predict the doubly robust dose-response curve.

        Important:
        The component predictions and the slack term are first computed in the
        scale on which the bridge algorithms were fitted. The final DR curve is
        then inverse-transformed once.
        """
        A_test = _as_2d_tensor(A, self.device)
        n_test = A_test.shape[0]
        n_train = self.A.shape[0]

        # Component predictions in the fitted outcome scale.
        treatment_bridge_algo_pred_raw = self.treatment_bridge_algo.predict(
            A_test,
            TorchIdentityTransformer(),
        ).reshape(-1)

        outcome_bridge_algo_pred_raw = self.outcome_bridge_algo.predict(
            A_test,
            TorchIdentityTransformer(),
        ).reshape(-1)

        self.treatment_bridge_algo_pred_raw = treatment_bridge_algo_pred_raw
        self.outcome_bridge_algo_pred_raw = outcome_bridge_algo_pred_raw

        # Bridge values on the training data.
        treatment_bridge_values = self.treatment_bridge_algo._predict_bridge_func(
            A_test,
            self.Z,
            self.X,
        )
        outcome_bridge_values = self.outcome_bridge_algo._predict_bridge_func(
            A_test,
            self.W,
            self.X,
        )

        treatment_bridge_values = self._ensure_bridge_shape(
            treatment_bridge_values,
            n_test=n_test,
            n_train=n_train,
            name="Treatment",
        )
        outcome_bridge_values = self._ensure_bridge_shape(
            outcome_bridge_values,
            n_test=n_test,
            n_train=n_train,
            name="Outcome",
        )

        # Nyström KRR weights for the DR interaction/slack term.
        DR_KRR_weights = self._compute_dr_krr_weights(A_test)  # (n_train, n_test)

        slack_prediction_raw = (
            treatment_bridge_values
            * outcome_bridge_values
            * DR_KRR_weights.T
        ).sum(dim=1)

        self.slack_prediction_raw = slack_prediction_raw

        f_struct_pred_raw = (
            treatment_bridge_algo_pred_raw
            + outcome_bridge_algo_pred_raw
            - slack_prediction_raw
        )

        f_struct_pred = outcome_transformer.inverse_transform(f_struct_pred_raw)

        return f_struct_pred

    def predict_components(
        self,
        A: torch.Tensor,
        outcome_transformer: nn.Module = TorchIdentityTransformer(),
    ) -> Dict[str, torch.Tensor]:
        """
        Return DR prediction and its three components.

        Components are returned after applying the same final inverse-transform
        convention only to the final DR prediction. The raw components are also
        included for diagnostic use.
        """
        A_test = _as_2d_tensor(A, self.device)

        f_dr = self.predict(A_test, outcome_transformer=outcome_transformer)

        return {
            "dr_prediction": f_dr,
            "treatment_prediction_raw": self.treatment_bridge_algo_pred_raw,
            "outcome_prediction_raw": self.outcome_bridge_algo_pred_raw,
            "slack_prediction_raw": self.slack_prediction_raw,
            "dr_krr_weights": self.DR_KRR_weights,
        }


# =========================================================================================================
# ATT (dose response ON THE TREATED) -- Nystrom counterparts of the EXACT classes
# `causal_learning/kernel_proxy_causal_learning.py::KernelProxyVariableATT / KernelAlternativeProxyATT`.
#
# WHY (design decision): the exact ATT kernel baseline does not reach the reporting axis. Its Grams are
# O(n^2) float64 and its solves O(n^3), measured at 31.5 GB (KPV) / 64.3 GB (KAP) at n=20000 -- and KAP bakes
# the anchor a' into B_bar at FIT time, so it pays that FOUR times per (seed, repeat). These classes keep the
# estimator and every hyperparameter name identical and replace the large solves by their Nystrom forms, so
# the ATT rows can be reported on the same {2,5,10,15,20}x1000 axis as every other benchmark.
#
# WHERE THE APPROXIMATION SITS (same places as the ATE Nystrom classes above, plus the ATT's own CME):
#   KPV-ATT : stage-1 CME solve; stage-2 outcome-bridge KRR (its representer set becomes the stage-2
#             landmarks); and the a'-CME of predict, which is estimated on that same landmark subsample.
#   KAP-ATT : stage-1 CME solve (B, B_bar, and the eta block's B2, B2_bar); the a'-CME theta; the stage-2
#             alpha/eta system, whose representer set becomes the stage-2 landmarks (the ATE Nystrom KAP
#             keeps that system exact -- for the ATT it is the binding cost, because block_component{1,2,4}
#             are (n2 x n2) and n2 = n/2; pass nystrom_second_stage_m=0 to recover the exact system); and
#             the third-stage KRR.
#
# EXACTNESS. Every Nystrom solve here is the standard  K_nm (K_mn K_nm + n lam K_mm)^-1 T_m  form, which is
# ALGEBRAICALLY EQUAL to the exact  (K_nn + n lam I)^-1 T  when the landmark set is the whole sample (K and
# (K + n lam I)^-1 commute). The m = n path is therefore a genuine regression test of the port -- it is the
# number, so agreement at m = n is to solver tolerance, not to machine precision.
#
# LOOCV IS OFF BY DEFAULT AND SHOULD STAY OFF. The exact classes select every regulariser by the hat-matrix
# LOOCV identity  h_i = (I - R)_ii  with the 1/diag(I - R) leave-one-out correction. That identity is derived
# for the EXACT ridge solution; under a low-rank Nystrom solve those leverage terms are no longer the
# leave-one-out residuals of the estimator actually being fitted, so the objective is NOT well defined (design decision
#; the dSprite and demand Nystrom sims already fix their regularisers by hand for the same
# reason). The `optimize_*` flags are therefore expected to be FALSE in reporting runs and the regularisers
# passed in; they are kept because they ARE exact -- and so meaningful -- when the landmark set is the whole
# O(n^3) per grid point, so it is restricted to a random subsample of `loocv_subsample` rows (default 2000;
# <= 0 or a value above the sample size means the full sample). That subsample is drawn from `model_seed`,
# never from the global RNG.
#
# PAIRING. As in the exact classes the stage-1/stage-2 split is drawn from the GLOBAL numpy RNG inside fit,
# so seeding immediately before each fit keeps KPV and every KAP anchor on the same split. Landmarks and
# LOOCV subsamples come from `model_seed` instead, so they are identical across anchors by construction.
# =========================================================================================================


def _att_landmarks(original_size: int, m: Optional[int], seed: int = 0):
    """Landmark indices into [0, original_size). m <= 0 or m >= original_size -> ALL indices, in order
    (an identity selection, so the m = n path is bit-comparable with the exact classes rather than a
    permutation of them)."""
    if (m is None) or (m <= 0) or (m >= original_size):
        return np.arange(original_size)
    return np.sort(np.random.default_rng(seed).choice(original_size, m, replace=False))


def _att_ones_gram(n: int, m: int, ref: torch.Tensor) -> torch.Tensor:
    return torch.ones((n, m), dtype=ref.dtype, device=ref.device)


def _theta_weighted_offdiag(K: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """K @ (theta 1^T - diag(theta)), i.e. column j becomes sum_{l != j} theta_l K[:, l].

    The exact ATT classes build the dense (m x m) factor `theta * remove_diagonal_elements(ones)`; this is
    the same matrix product without materialising it (the ATT's B_bar target is the only place it is used,
    and at n=20000 that factor alone is 800 MB)."""
    theta = theta.reshape(-1)
    return (K @ theta).reshape(-1, 1) @ torch.ones((1, K.shape[1]), dtype=K.dtype, device=K.device) \
        - K * theta.reshape(1, -1)


class KernelProxyVariableATTNystrom(BaseEstimator, RegressorMixin, nn.Module):
    """Nystrom KPV outcome bridge for the ATT.

    f_ATT(a, a') = sum_j alpha_j k_A(ATilde_j, a) * v_j,  v = ((B_l^T K_W(W, WTilde_l)) * K_X(XTilde_l, XTilde_l)) theta,
    theta = (K_A(ATilde_l, ATilde_l) + m_l zeta I)^-1 k_A(ATilde_l, a'),  l = the stage-2 landmark set.

    `fit` takes no a' (h solves the a'-free KPV moment equation) so ONE fit serves every anchor; a' enters
    only through `predict`, exactly as in the exact `KernelProxyVariableATT`.
    """

    def __init__(self,
                 kernel_A: Callable,
                 kernel_W: Callable,
                 kernel_Z: Callable,
                 kernel_X: Optional[Callable] = None,
                 lambda1_: float = 0.1,
                 lambda2_: float = 0.1,
                 zeta: float = 0.1,
                 nystrom_first_stage_m: int = 500,
                 nystrom_second_stage_m: int = 500,
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

        self.nystrom_first_stage_m = nystrom_first_stage_m
        self.nystrom_second_stage_m = nystrom_second_stage_m

        self.optimize_lambda1_parameter = optimize_lambda1_parameter
        self.optimize_lambda2_parameter = optimize_lambda2_parameter
        self.optimize_zeta_parameter = optimize_zeta_parameter
        self.lambda1_optimization_range = lambda1_optimization_range
        self.lambda2_optimization_range = lambda2_optimization_range
        self.zeta_optimization_range = zeta_optimization_range

        self.stage1_perc = kwargs.pop("stage1_perc", 0.5)
        self.regularization_grid_points = kwargs.pop("regularization_grid_points", 25)
        self.make_psd_eps = kwargs.pop("make_psd_eps", 1e-9)
        self.label_variance_in_lambda_opt = kwargs.pop("label_variance_in_lambda_opt", 0.0)
        self.model_seed = kwargs.pop("model_seed", 0)
        self.loocv_subsample = kwargs.pop("loocv_subsample", 2000)

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.alpha = None
        self.B = None
        self.ATilde = None
        self.XTilde = None
        self.W = None
        self.stage1_idx = None
        self.stage2_idx = None
        self.nystrom_stage2_landmarks = None

    @staticmethod
    def _geomspace(lo, hi, n, dtype, device):
        return torch.logspace(float(np.log(lo)), float(np.log(hi)), n,
                              base=float(np.exp(1.0)), dtype=dtype, device=device)

    @staticmethod
    def _lambda1_objective(lambda_, K_ZAX, K_WW, identity_matrix, make_psd_eps=1e-9):
        """The exact KernelProxyVariableATT objective, verbatim (note: NO transpose on the trailing
        factor -- `torch_utils.ml_utils.lambda_objective_loocv`, which the ATE Nystrom classes call,
        places it differently and is NOT the same functional)."""
        n = K_ZAX.shape[0]
        ridge_weights = make_psd(K_ZAX + n * lambda_ * identity_matrix, eps=make_psd_eps)
        R = torch.linalg.solve(ridge_weights, K_ZAX).T
        H1 = identity_matrix - R
        H1_diag = torch.diag(H1)
        H1_tilde_inv = torch.diag(1.0 / H1_diag)
        H1_tilde_inv_times_H1 = H1_tilde_inv @ H1
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
        """CME LOOCV for E[phi_W phi_X | A] -- identical to the exact class's public objective."""
        n = K_AA.shape[0]
        eye = torch.eye(n, dtype=K_AA.dtype, device=K_AA.device)
        R = torch.linalg.solve(make_psd(K_AA + n * zeta * eye, make_psd_eps), K_AA).T
        S = torch.diag((1.0 / (1.0 - torch.diag(R))) ** 2)
        T = S @ (K_XX_WW - 2 * K_XX_WW @ R.T + R @ K_XX_WW @ R.T)
        return torch.trace(T)

    def _loocv_idx(self, size: int, seed_offset: int):
        s = self.loocv_subsample
        if (s is None) or (s <= 0) or (s >= size):
            return np.arange(size)
        return np.sort(np.random.default_rng(self.model_seed + seed_offset).choice(size, s, replace=False))

    def fit(self,
            AZWX: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
            Y: torch.Tensor) -> None:
        """Tuple order is (A, Z, W[, X]) -- NOT the reference implementation's reference's (A, W, Z)."""
        if len(AZWX) == 4:
            A, Z, W, X = AZWX
        elif len(AZWX) == 3:
            A, Z, W = AZWX
            X = None
        else:
            raise ValueError("AZWX must be either (A, Z, W) or (A, Z, W, X).")

        A = _as_2d_tensor(A, self.device)
        Z = _as_2d_tensor(Z, self.device)
        W = _as_2d_tensor(W, self.device)
        Y = _as_2d_tensor(Y, self.device)
        X = _as_2d_tensor(X, self.device) if X is not None else None

        K_ATrainATrain = self.kernel_A(A, A)
        K_WTrainWTrain = self.kernel_W(W, W)
        K_ZTrainZTrain = self.kernel_Z(Z, Z)
        dtype = K_ATrainATrain.dtype
        has_X = X is not None
        if has_X:
            K_XTrainXTrain = make_psd(self.kernel_X(X, X), self.make_psd_eps)

        for kernel_ in [self.kernel_A, self.kernel_W, self.kernel_Z, self.kernel_X]:
            if hasattr(kernel_, "use_length_scale_heuristic"):
                kernel_.use_length_scale_heuristic = False

        n_total = A.shape[0]
        indices = np.random.permutation(n_total)                   # GLOBAL numpy RNG, as in the exact class
        if 0.0 < self.stage1_perc < 1.0:
            n1 = int(n_total * self.stage1_perc)
            stage1_idx, stage2_idx = indices[:n1], indices[n1:]
        else:
            n1 = n_total
            stage1_idx, stage2_idx = indices, indices
        n2 = len(stage2_idx)

        l1 = _att_landmarks(n1, self.nystrom_first_stage_m, self.model_seed)
        l2 = _att_landmarks(n2, self.nystrom_second_stage_m, self.model_seed + 1)

        K_AA = _ix(K_ATrainATrain, stage1_idx, stage1_idx)
        K_AATilde = _ix(K_ATrainATrain, stage1_idx, stage2_idx)
        K_WW = _ix(K_WTrainWTrain, stage1_idx, stage1_idx)
        K_WWTilde = _ix(K_WTrainWTrain, stage1_idx, stage2_idx)
        K_ZZ = _ix(K_ZTrainZTrain, stage1_idx, stage1_idx)
        K_ZZTilde = _ix(K_ZTrainZTrain, stage1_idx, stage2_idx)
        if has_X:
            K_XX = _ix(K_XTrainXTrain, stage1_idx, stage1_idx)
            K_XXTilde = _ix(K_XTrainXTrain, stage1_idx, stage2_idx)
        else:                                                      # the all-ones Gram is skipped, not built
            K_XX = None
            K_XXTilde = None

        K_ZAX = K_ZZ * K_AA if not has_X else K_ZZ * K_AA * K_XX

        # ---- stage 1: CME of phi_W(W) given (Z, A, X) ----
        if self.optimize_lambda1_parameter:
            sub = self._loocv_idx(n1, 11)
            grid = self._geomspace(*self.lambda1_optimization_range,
                                   self.regularization_grid_points, dtype, self.device)
            K_a, K_b = _ix(K_ZAX, sub, sub), _ix(K_WW, sub, sub)
            I_sub = torch.eye(len(sub), dtype=dtype, device=self.device)
            obj = torch.stack([self._lambda1_objective(l, K_a, K_b, I_sub, self.make_psd_eps)
                               for l in grid])
            self.lambda1_ = float(grid[int(torch.argmin(obj))])
            self.lambda1_objective_list = obj

        K_ZAX_nm = K_ZAX[:, l1]
        K_ZAX_mm = K_ZAX[l1][:, l1]
        stage1_ridge_weights = make_psd(K_ZAX_nm.T @ K_ZAX_nm + n1 * self.lambda1_ * K_ZAX_mm,
                                        self.make_psd_eps)
        K_ZAX_Tilde = K_ZZTilde * K_AATilde if not has_X else K_ZZTilde * K_AATilde * K_XXTilde
        B = K_ZAX_nm @ torch.linalg.solve(stage1_ridge_weights, K_ZAX_Tilde[l1, :])     # (n1, n2)

        # ---- stage 2: KRR of Y on phi_A(ATilde) (x) mu_W (x) phi_X(XTilde), representers = landmarks ----
        K_ATildeATilde = _ix(K_ATrainATrain, stage2_idx, stage2_idx)
        if has_X:
            K_XTildeXTilde = _ix(K_XTrainXTrain, stage2_idx, stage2_idx)
            stage2_ridge_weights = K_ATildeATilde * (B.T @ K_WW @ B) * K_XTildeXTilde
        else:
            K_XTildeXTilde = None
            stage2_ridge_weights = K_ATildeATilde * (B.T @ K_WW @ B)
        YTilde = Y[stage2_idx]

        if self.optimize_lambda2_parameter:
            sub = self._loocv_idx(n2, 12)
            grid = self._geomspace(*self.lambda2_optimization_range,
                                   self.regularization_grid_points, dtype, self.device)
            Ys = YTilde[sub]
            I_sub = torch.eye(len(sub), dtype=dtype, device=self.device)
            obj = torch.stack([self._lambda2_objective(l, _ix(stage2_ridge_weights, sub, sub),
                                                       Ys @ Ys.T, I_sub, self.make_psd_eps)
                               for l in grid])
            self.lambda2_ = float(grid[int(torch.argmin(obj))])
            self.lambda2_objective_list = obj

        srw_mn = stage2_ridge_weights[l2, :]
        srw_mm = stage2_ridge_weights[l2][:, l2]
        alpha = torch.linalg.solve(
            make_psd(srw_mn @ srw_mn.T + n2 * self.lambda2_ * srw_mm, self.make_psd_eps),
            srw_mn @ YTilde)                                                             # (m_l, 1)

        # ---- the ATT's own CME, on the stage-2 landmark subsample ----
        K_All = K_ATildeATilde[l2][:, l2]
        K_Wll = _ix(K_WTrainWTrain, stage2_idx[l2], stage2_idx[l2])
        K_Xll = (K_XTildeXTilde[l2][:, l2] if has_X
                 else _att_ones_gram(len(l2), len(l2), K_All))
        if self.optimize_zeta_parameter:
            grid = self._geomspace(*self.zeta_optimization_range,
                                   self.regularization_grid_points, dtype, self.device)
            obj = torch.stack([self.zeta_objective(z, K_All, K_Xll * K_Wll) for z in grid])
            self.zeta = float(grid[int(torch.argmin(obj))])
            self.zeta_objective_list = obj

        self.alpha = alpha
        self.B = B
        self.B_sub = B[:, l2]                                                            # (n1, m_l)
        self.K_WWTilde_l = K_WWTilde[:, l2]                                              # (n1, m_l)
        self.K_All = K_All
        self.K_Xll = K_Xll
        self.ATilde = _take_rows(A, stage2_idx[l2])
        self.XTilde = _take_rows(X, stage2_idx[l2]) if has_X else None
        self.W = _take_rows(W, stage1_idx)
        self.stage1_idx, self.stage2_idx = stage1_idx, stage2_idx
        self.nystrom_stage1_landmarks, self.nystrom_stage2_landmarks = l1, l2

    def predict(self, A: torch.Tensor, aprime, outcome_transformer=None) -> torch.Tensor:
        """f_ATT(a, a') on the grid `A`, for the single anchor `aprime`."""
        A_test = _as_2d_tensor(A, self.device)
        if not torch.is_tensor(aprime):
            aprime = torch.tensor(aprime, dtype=A_test.dtype, device=self.device)
        aprime = aprime.reshape(-1, 1).to(dtype=A_test.dtype, device=self.device)

        m_l = self.K_All.shape[0]
        eye = torch.eye(m_l, dtype=self.K_All.dtype, device=self.K_All.device)
        K_ATildeaprime = self.kernel_A(self.ATilde, aprime)                              # (m_l, 1)
        # NO make_psd here, as in the exact class (the most fragile solve of the ATT path).
        theta = torch.linalg.solve(self.K_All + m_l * self.zeta * eye, K_ATildeaprime)
        K_ATildeATest = self.kernel_A(self.ATilde, A_test)                               # (m_l, T)

        v = ((self.B_sub.T @ self.K_WWTilde_l) * self.K_Xll) @ theta                     # (m_l, 1)
        pred = (K_ATildeATest * (self.alpha * v)).sum(dim=0).reshape(-1, 1)              # (T, 1)
        if outcome_transformer is not None:
            pred = outcome_transformer.inverse_transform(pred)
        return pred.reshape(-1)

    def _predict_bridge_func(self, A_test: torch.Tensor, W_test: torch.Tensor, X_test=None):
        """h_hat(a, w[, x]) on the test grids; shape (T_A, T_W)."""
        A_test = _as_2d_tensor(A_test, self.device)
        W_test = _as_2d_tensor(W_test, self.device)
        K_ATildeATest = self.kernel_A(self.ATilde, A_test)                               # (m_l, T_A)
        K_WWTest = self.kernel_W(self.W, W_test)                                         # (n1, T_W)
        if (X_test is not None) and (self.XTilde is not None):
            K_XTildeXTest = self.kernel_X(self.XTilde, _as_2d_tensor(X_test, self.device))
        else:
            K_XTildeXTest = _att_ones_gram(self.ATilde.shape[0], W_test.shape[0], K_ATildeATest)
        base = (self.B_sub.T @ K_WWTest) * K_XTildeXTest                                 # (m_l, T_W)
        return torch.einsum("mt,mw,mo->tw", K_ATildeATest, base, self.alpha)


class KernelAlternativeProxyATTNystrom(BaseEstimator, RegressorMixin, nn.Module):
    """Nystrom KAP (alternative-proxy / treatment-bridge) estimator for the ATT, density-ratio-free.

    The bridge solves E[phi_{a'}(A,Z) | A=a, W=w] = p(w|a')/p(w|a) and the third stage forms
    f_ATT(a, a') = E[Y phi_{a'}(a, Z) | A = a]. a' is baked into B_bar at FIT time, so each anchor needs its
    own fit -- which is exactly why the exact class does not reach n = 15000/20000 and this one exists.

    Every objective (`_lambda_objective`, `_zeta_objective`, `_eta_objective`, `_lambda2_objective`) is the
    exact class's, verbatim, including its asymmetries: the trailing `.T` that `_lambda2_objective` carries
    and `_lambda_objective` does not, and the two different make_psd epsilons. They are evaluated on a
    subsample of `loocv_subsample` rows (see the module note above).
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
                 nystrom_first_stage_m: int = 500,
                 nystrom_second_stage_m: int = 500,
                 nystrom_third_stage_m: int = 500,
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

        self.nystrom_first_stage_m = nystrom_first_stage_m
        self.nystrom_second_stage_m = nystrom_second_stage_m
        self.nystrom_third_stage_m = nystrom_third_stage_m

        self.optimize_lambda_parameters = optimize_lambda_parameters
        self.optimize_eta_parameter = optimize_eta_parameter
        self.optimize_zeta_parameter = optimize_zeta_parameter
        self.lambda_optimization_range = lambda_optimization_range
        self.eta_optimization_range = eta_optimization_range
        self.zeta_optimization_range = zeta_optimization_range

        self.stage1_perc = kwargs.pop("stage1_perc", 0.5)
        self.regularization_grid_points = kwargs.pop("regularization_grid_points", 25)
        self.label_variance_in_lambda_opt = kwargs.pop("label_variance_in_lambda_opt", 0.0)
        self.label_variance_in_eta_opt = kwargs.pop("label_variance_in_eta_opt", 0.0)
        self.make_psd_eps = kwargs.pop("make_psd_eps", 1e-9)
        self.lambda2_optimization_range = kwargs.pop("lambda2_optimization_range", None)
        self.model_seed = kwargs.pop("model_seed", 0)
        self.loocv_subsample = kwargs.pop("loocv_subsample", 2000)

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.alpha = None
        self.B = None
        self.B_bar = None
        self.third_stage_KRR_weights = None

    # ------------------------------------------------------------------ objectives (exact class, verbatim)
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
    def _lambda2_objective(lambda2_, K_AA, K_ZZ, Y, label_variance_in_lambda_opt, make_psd_eps=1e-9):
        n = K_AA.shape[0]
        identity_matrix = torch.eye(n, dtype=K_AA.dtype, device=K_AA.device)
        ridge_weights = make_psd(K_AA + n * lambda2_ * identity_matrix, eps=make_psd_eps)
        R = torch.linalg.solve(ridge_weights, K_AA).T
        H2 = identity_matrix - R
        H2_diag = torch.diag(H2)
        H2_tilde_inv = torch.diag(1.0 / H2_diag)
        kernel_output = K_ZZ * (Y @ Y.T)
        H2_tilde_inv_times_H2 = H2_tilde_inv @ H2
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
        return torch.logspace(float(np.log(lo)), float(np.log(hi)), n,
                              base=float(np.exp(1.0)), dtype=dtype, device=device)

    def _loocv_idx(self, size: int, seed_offset: int):
        s = self.loocv_subsample
        if (s is None) or (s <= 0) or (s >= size):
            return np.arange(size)
        return np.sort(np.random.default_rng(self.model_seed + seed_offset).choice(size, s, replace=False))

    # ------------------------------------------------------------------ fit
    def fit(self,
            AZWX: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
            Y: torch.Tensor,
            aprime) -> None:
        """Fit the treatment bridge for ONE anchor a'. Tuple order (A, Z, W[, X])."""
        if len(AZWX) == 4:
            A, Z, W, X = AZWX
        elif len(AZWX) == 3:
            A, Z, W = AZWX
            X = None
        else:
            raise ValueError("AZWX must be either (A, Z, W) or (A, Z, W, X).")

        A = _as_2d_tensor(A, self.device)
        Z = _as_2d_tensor(Z, self.device)
        W = _as_2d_tensor(W, self.device)
        Y = _as_2d_tensor(Y, self.device)
        X = _as_2d_tensor(X, self.device) if X is not None else None

        K_ATrainATrain = self.kernel_A(A, A)
        K_WTrainWTrain = self.kernel_W(W, W)
        K_ZTrainZTrain = self.kernel_Z(Z, Z)
        dtype = K_ATrainATrain.dtype
        has_X = X is not None
        if has_X:
            K_XTrainXTrain = self.kernel_X(X, X)        # NO make_psd here -- exact ATT class behaviour

        if not torch.is_tensor(aprime):
            aprime = torch.tensor(aprime, dtype=dtype, device=self.device)
        aprime = aprime.reshape(-1, 1).to(dtype=dtype, device=self.device)

        for kernel_ in [self.kernel_A, self.kernel_W, self.kernel_Z, self.kernel_X]:
            if hasattr(kernel_, "use_length_scale_heuristic"):
                kernel_.use_length_scale_heuristic = False

        N_total = A.shape[0]
        train_indices = np.random.permutation(N_total)             # GLOBAL numpy RNG, as in the exact class
        if 0.0 < self.stage1_perc < 1.0:
            n1 = int(N_total * self.stage1_perc)
            stage1_idx, stage2_idx = train_indices[:n1], train_indices[n1:]
        else:
            n1 = N_total
            stage1_idx, stage2_idx = train_indices, train_indices
        n2 = len(stage2_idx)

        l1 = _att_landmarks(n1, self.nystrom_first_stage_m, self.model_seed)
        l2 = _att_landmarks(n2, self.nystrom_second_stage_m, self.model_seed + 1)
        l3 = _att_landmarks(N_total, self.nystrom_third_stage_m, self.model_seed + 2)
        m1, m2r, m3 = len(l1), len(l2), len(l3)

        K_AA = _ix(K_ATrainATrain, stage1_idx, stage1_idx)
        K_AATilde = _ix(K_ATrainATrain, stage1_idx, stage2_idx)
        K_WW = _ix(K_WTrainWTrain, stage1_idx, stage1_idx)
        K_WWTilde = _ix(K_WTrainWTrain, stage1_idx, stage2_idx)
        K_ZZ = _ix(K_ZTrainZTrain, stage1_idx, stage1_idx)
        if has_X:
            K_XX = _ix(K_XTrainXTrain, stage1_idx, stage1_idx)
            K_XXTilde = _ix(K_XTrainXTrain, stage1_idx, stage2_idx)
            K_AWX = K_AA * K_WW * K_XX
            K_WX = K_WW * K_XX
            K_WX_Tilde_l = (K_WWTilde * K_XXTilde)[l1][:, l2]
        else:
            K_AWX = K_AA * K_WW
            K_WX = K_WW
            K_WX_Tilde_l = K_WWTilde[l1][:, l2]

        # ---- stage-1 regulariser ----
        I_sub = None
        if self.optimize_lambda_parameters:
            sub = self._loocv_idx(n1, 11)
            I_sub = torch.eye(len(sub), dtype=dtype, device=self.device)
            grid = self._geomspace(*self.lambda_optimization_range,
                                   self.regularization_grid_points, dtype, self.device)
            K_a, K_b = _ix(K_AWX, sub, sub), _ix(K_ZZ, sub, sub)
            obj = torch.stack([self._lambda_objective(l, K_a, K_b, I_sub,
                                                      self.label_variance_in_lambda_opt,
                                                      self.make_psd_eps) for l in grid])
            self.lambda_ = float(grid[int(torch.argmin(obj))])
            self.lambda_objective_list, self.lambda_list = obj, grid

        # ---- the ATT's own CME weights theta over the stage-2 representer set ----
        K_A_l2_l2 = _ix(K_ATrainATrain, stage2_idx[l2], stage2_idx[l2])
        zeta_list = self._geomspace(*self.zeta_optimization_range,
                                    self.regularization_grid_points, dtype, self.device)
        if self.optimize_zeta_parameter:
            if has_X:
                K_WX_l2 = _ix(K_WTrainWTrain, stage2_idx[l2], stage2_idx[l2]) \
                    * _ix(K_XTrainXTrain, stage2_idx[l2], stage2_idx[l2])
            else:
                K_WX_l2 = _ix(K_WTrainWTrain, stage2_idx[l2], stage2_idx[l2])
            obj = torch.stack([self._zeta_objective(z, K_A_l2_l2, K_WX_l2) for z in zeta_list])
            self.zeta = float(zeta_list[int(torch.argmin(obj))])
            self.zeta_objective_list, self.zeta_list = obj, zeta_list

        eye_m = torch.eye(m2r, dtype=dtype, device=self.device)
        K_A_l2_aprime = self.kernel_A(_take_rows(A, stage2_idx[l2]), aprime)                 # (m2r, 1)
        theta = torch.linalg.solve(make_psd(K_A_l2_l2) + m2r * self.zeta * eye_m, K_A_l2_aprime)

        # ---- stage 1 (Nystrom): B and the a'-tilted B_bar ----
        K_AWX_nm = K_AWX[:, l1]
        K_AWX_mm = K_AWX[l1][:, l1]
        stage1_ridge_weights = make_psd(K_AWX_nm.T @ K_AWX_nm + n1 * self.lambda_ * K_AWX_mm,
                                        self.make_psd_eps)
        K_A_l1_l2 = K_AATilde[l1][:, l2]
        B = K_AWX_nm @ torch.linalg.solve(stage1_ridge_weights, K_WX_Tilde_l * K_A_l1_l2)    # (n1, m2r)
        B_bar = K_AWX_nm @ torch.linalg.solve(
            stage1_ridge_weights, _theta_weighted_offdiag(K_WX_Tilde_l, theta) * K_A_l1_l2)  # (n1, m2r)

        # ---- stage 2: the alpha / eta system on the stage-2 representer set ----
        block_component1 = (B.T @ K_ZZ @ B) * K_A_l2_l2
        block_component2 = (B.T @ K_ZZ @ B_bar) * K_A_l2_l2
        block_component4 = (B_bar.T @ K_ZZ @ B_bar) * K_A_l2_l2
        ones_divided_by_m = torch.ones(m2r, dtype=dtype, device=self.device) / m2r

        bc2_om = (block_component2 @ ones_divided_by_m).reshape(-1, 1)
        bc4_omom = (ones_divided_by_m @ block_component4 @ ones_divided_by_m).reshape(-1, 1)
        L = torch.vstack((block_component1,
                          (ones_divided_by_m @ block_component2.T).reshape(1, -1))) \
            @ torch.hstack((block_component1, bc2_om))
        M = torch.vstack((bc2_om, bc4_omom))
        P_blk = torch.hstack((block_component1, bc2_om))
        R_blk = torch.hstack(((ones_divided_by_m @ block_component2.T).reshape(1, -1), bc4_omom))
        N_blk = torch.vstack((P_blk, R_blk))

        if self.optimize_eta_parameter:
            # eta is selected on the STAGE-1 fold used as validation, which needs its own a'-CME.
            K_A_l1_l1 = K_AA[l1][:, l1]
            obj2 = torch.stack([self._zeta_objective(z, K_A_l1_l1, K_WX[l1][:, l1]) for z in zeta_list])
            zeta2 = float(zeta_list[int(torch.argmin(obj2))])

            K_A_n1_l1 = K_AA[:, l1]
            K_A_l1_aprime = self.kernel_A(_take_rows(A, stage1_idx[l1]), aprime)             # (m1, 1)
            theta2 = K_A_n1_l1 @ torch.linalg.solve(                                          # (n1, 1)
                make_psd(K_A_n1_l1.T @ K_A_n1_l1 + n1 * zeta2 * K_A_l1_l1, self.make_psd_eps),
                K_A_l1_aprime)

            B2 = K_AWX_nm @ torch.linalg.solve(stage1_ridge_weights, K_AWX[l1, :])            # (n1, n1)
            B2_bar = K_AWX_nm @ torch.linalg.solve(
                stage1_ridge_weights, _theta_weighted_offdiag(K_WX[l1, :], theta2) * K_AA[l1, :])
            ones_divided_by_n = torch.ones(n1, dtype=dtype, device=self.device) / n1

            K_A_n1_l2 = K_AATilde[:, l2]
            block_component12 = (B2.T @ K_ZZ @ B) * K_A_n1_l2
            block_component22 = (B2.T @ K_ZZ @ B_bar) * K_A_n1_l2
            block_component32 = (B.T @ K_ZZ @ B2_bar) * K_A_n1_l2.T
            block_component42 = (B_bar.T @ K_ZZ @ B2_bar) * K_A_n1_l2.T

            L2_sub = torch.vstack((block_component12.T,
                                   (ones_divided_by_m @ block_component22.T).reshape(1, -1)))
            L2 = L2_sub @ L2_sub.T
            M2 = torch.vstack(((block_component32 @ ones_divided_by_n).reshape(-1, 1),
                               (ones_divided_by_m @ block_component42 @ ones_divided_by_n).reshape(-1, 1)))

            eta_list = self._geomspace(*self.eta_optimization_range,
                                       self.regularization_grid_points, dtype, self.device)
            obj = torch.stack([self._eta_objective(e, L, M, N_blk, L2, M2, n1,
                                                   self.label_variance_in_eta_opt,
                                                   self.make_psd_eps) for e in eta_list])
            self.eta = float(eta_list[int(torch.argmin(obj))])
            self.eta_objective_list, self.eta_list = obj, eta_list

        alpha = torch.linalg.solve(make_psd(L / m2r + self.eta * N_blk, self.make_psd_eps), M)

        # ---- third stage (Nystrom): KRR of the H_Z-valued target Y phi_Z(Z) on A, over ALL train rows ----
        if self.optimize_lambda_parameters:
            rng2 = self.lambda2_optimization_range or self.lambda_optimization_range
            sub = self._loocv_idx(N_total, 13)
            grid = self._geomspace(*rng2, self.regularization_grid_points, dtype, self.device)
            obj = torch.stack([self._lambda2_objective(l, _ix(K_ATrainATrain, sub, sub),
                                                       _ix(K_ZTrainZTrain, sub, sub), Y[sub],
                                                       self.label_variance_in_lambda_opt,
                                                       self.make_psd_eps) for l in grid])
            self.lambda2_ = float(grid[int(torch.argmin(obj))])
            self.lambda2_objective_list = obj

        train_l3 = train_indices[l3]
        K_ZZtrain = _ix(K_ZTrainZTrain, stage1_idx, train_indices)                            # (n1, N)
        K_A_N_l3 = _ix(K_ATrainATrain, train_indices, train_l3)                               # (N, m3)
        K_A_l3_l3 = _ix(K_ATrainATrain, train_l3, train_l3)                                   # (m3, m3)
        KRR_LHS = make_psd(K_A_N_l3.T @ K_A_N_l3 + N_total * self.lambda2_ * K_A_l3_l3,
                           self.make_psd_eps)
        KRR_RHS = K_A_N_l3.T @ (K_ZZtrain.T * Y[train_indices])                               # (m3, n1)
        third_stage_KRR_weights = torch.linalg.solve(KRR_LHS, KRR_RHS).T                      # (n1, m3)

        self.alpha = alpha
        self.B, self.B_bar = B, B_bar
        self.third_stage_KRR_weights = third_stage_KRR_weights
        self.ones_divided_by_m = ones_divided_by_m
        self.ATrain, self.ZTrain = A, Z
        self.A_third = _take_rows(A, train_l3)
        self.A_second = _take_rows(A, stage2_idx[l2])
        self.train_indices, self.train_l3 = train_indices, train_l3
        self.stage1_idx, self.stage2_idx = stage1_idx, stage2_idx
        self.nystrom_stage1_landmarks = l1
        self.nystrom_stage2_landmarks = l2
        self.nystrom_stage3_landmarks = l3
        self.aprime = aprime

    # ------------------------------------------------------------------ predict
    def predict(self, A: torch.Tensor, outcome_transformer=None) -> torch.Tensor:
        """f_ATT(a, a') on the grid `A`; a' is the anchor baked in at fit time."""
        A_test = _as_2d_tensor(A, self.device)
        K_A_l3_ATest = self.kernel_A(self.A_third, A_test)                                    # (m3, T)
        K_A_l2_ATest = self.kernel_A(self.A_second, A_test)                                   # (m2r, T)

        V = self.third_stage_KRR_weights @ K_A_l3_ATest                                       # (n1, T)
        term1 = (self.B.T @ V) * K_A_l2_ATest
        term2 = (self.B_bar.T @ V) * K_A_l2_ATest
        pred = (self.alpha[:-1].T @ term1).reshape(-1) \
            + self.alpha[-1] * (self.ones_divided_by_m @ term2)
        if outcome_transformer is not None:
            pred = outcome_transformer.inverse_transform(pred.reshape(-1, 1))
        return pred.reshape(-1)

    def _predict_bridge_func(self, A_test: torch.Tensor, Z_test: torch.Tensor, X_test=None):
        """phi_hat(a, z) on the test grids; shape (T_A, T_Z). X_test ignored, as in the exact class."""
        A_test = _as_2d_tensor(A_test, self.device)
        Z_test = _as_2d_tensor(Z_test, self.device)
        K_ZZTest = self.kernel_Z(_take_rows(self.ZTrain, self.stage1_idx), Z_test)            # (n1, T_Z)
        K_A_l2_ATest = self.kernel_A(self.A_second, A_test)                                   # (m2r, T_A)
        base1 = self.B.T @ K_ZZTest
        base2 = self.B_bar.T @ K_ZZTest
        out1 = torch.einsum("mt,mz,mo->tz", K_A_l2_ATest, base1, self.alpha[:-1])
        out2 = torch.einsum("mt,mz,m->tz", K_A_l2_ATest, base2, self.ones_divided_by_m)
        return out1 + self.alpha[-1] * out2


##################################################################################
############# CATE / HETEROGENEOUS RESPONSE (Nystrom KPV outcome bridge) #########
##################################################################################
# Nystrom port of the exact `KernelProxyVariableHeterogeneousResponse`
# (kernel_proxy_causal_learning.py). KPV ONLY -- KAP and therefore DRKPV are not published for the
# heterogeneous response, so no treatment-bridge / doubly-robust counterpart exists here.
#
# Estimand:  f_CATE(a, v) = E[ Y^(a) | V = v ]
#
# What the heterogeneous class adds over the dose-response one:
#   * a covariate/moderator kernel `kernel_V`;
#   * V joins the CONDITIONING set of the stage-1 CME   (K_ZZ * K_AA * K_VV [* K_XX]);
#   * V joins the ARGUMENTS of the stage-2 bridge        (h(a, w, v[, x]));
#   * the dose-response's unconditional empirical mean over the (W, X) slot is replaced by a
#     CONDITIONAL mean embedding given V = v -- a kernel ridge on V with its own regulariser `zeta`.
#     `predict(A, V)` therefore takes a ROW-ALIGNED (a_j, v_j) grid and rebuilds the marginalisation
#     weights for every test point.


def _het_lock_length_scale(kernel_, X: torch.Tensor) -> None:
    """Lock a stateful kernel's median-heuristic length scale on the FULL sample, without
    materialising the n x n self-Gram.

    `RBF.forward` calls `calculate_and_fix_length_scale(X)` on its first SELF-kernel call and then
    self-disables (`use_length_scale_heuristic = False`, `_is_initialized = True`). Calling that
    method directly here is the same call with the same argument -- same subsample limit, same torch
    RNG draw -- so the locked length scale is identical to the one the exact class obtains from
    `kernel(X, X)`; we just skip building the n x n matrix we would immediately throw away. Kernels
    without the heuristic (e.g. `BinaryKernel`) are left untouched.
    """
    if kernel_ is None:
        return
    if getattr(kernel_, "use_length_scale_heuristic", False) and not getattr(kernel_, "_is_initialized", False):
        if hasattr(kernel_, "calculate_and_fix_length_scale"):
            kernel_.calculate_and_fix_length_scale(X)
    if hasattr(kernel_, "use_length_scale_heuristic"):
        kernel_.use_length_scale_heuristic = False


class KernelProxyVariableHeterogeneousResponseNystrom(BaseEstimator, RegressorMixin, nn.Module):
    """Nystrom KPV outcome bridge for the heterogeneous (CATE) response.

        f_CATE(a, v) = sum_{j in l2} alpha_j * k_A(ATilde_j, a) * k_V(VTilde_j, v) * u_j(v)
        u(v)         = ( K_X(XTilde_l, X_S) * (B_l' K_W(W_stage1, W_S)) ) @ cme(v)

    with `l2` the stage-2 landmark set (the representers `alpha` lives on) and `S` the sample the
    V-conditional mean embedding averages over.

    Three Nystrom solves, each of the form  K_nm (K_mn K_nm + n lam K_mm)^-1 T_m :
      1. stage-1 conditional mean embedding of phi_W given (Z, A, V [, X]);
      2. stage-2 kernel ridge of Y on phi_A (x) mu_W (x) phi_V [(x) phi_X];
      3. the third-stage V-CME, controlled by `cme_on_landmarks`:
           False (DEFAULT) -- S = ALL n2 stage-2 rows, the CME solved by Nystrom with landmarks l2
                              (only the SOLVE is approximated; every stage-2 row still contributes);
           True            -- S = l2 only (the pattern `KernelProxyVariableATTNystrom` uses for its
                              a'-CME: cheaper, but the conditional expectation is estimated from m
                              rows instead of n2).
    At landmarks = all rows both choices collapse to the exact class, because
    K (K K + n lam K)^-1 T = (K + n lam I)^-1 T.

    MEASURED (n=2000, the CATE benchmark): the DEFAULT full-S CME is the right one.
    Landmark ladder, causal MSE against an exact-class 0.00054242 --
        m=100  full-S 0.00078  vs landmark-CME 0.00481
        m=250  full-S 0.00049  vs landmark-CME 0.00570
        m=500  full-S 0.00054  vs landmark-CME 0.00152
        m=1000 (= n2, both exact)          0.00054
    The ATT port's landmark-CME pattern costs 2.8x-11x here; estimating E[phi_W phi_X | V=v] from m
    rows instead of n2 is a real statistical loss, and it is free to avoid.

    `make_psd_eps` IS NOT NEUTRAL IN A NYSTROM SOLVE -- measured, and it is the whole of the residual
    m = n gap. The Nystrom system K_mn K_nm + n lam K_mm equals K (K + n lam I) at m = n, and
    jittering THAT is not jittering K + n lam I: along a Gram direction with singular value sigma the
    jitter acts as an extra ridge of order eps / sigma. Against the exact class at m = n (n=400, the
    CATE benchmark's LOOCV-selected regularisers) the max relative curve difference is
        eps 5e-8 -> 5.31e-04 | 1e-9 -> 4.17e-05 | 1e-12 -> 5.05e-08 | 0 -> 8.65e-10
    i.e. exactly linear in eps down to the float64 conditioning floor of ~1e-9. Confirmed to be
    algebra and not rounding by redoing both solves at 60 digits (mpmath): the gap is IDENTICAL at 60
    digits and collapses at eps = 0. This is the convention of EVERY pre-existing Nystrom class in
    this file, not something introduced here, and it is statistically irrelevant at a working
    landmark count (m=500, n=2000: 0.00053990 at eps 1e-9 vs 0.00053994 at eps 5e-8). Keep eps small
    (1e-9, the default), and never read an m = n equivalence number without knowing its eps.

    Fit tuple order is (A, Z, W, V[, X]) -- NOT the reference implementation's order (A, W, Z, V).

    `optimize_*` default to False: the LOOCV hat-matrix identity is derived for the EXACT ridge
    solution and its leverage terms are not the leave-one-out residuals of a low-rank solve. Pass the
    regularisers in. The selection code is retained because it IS exact at m = n, which is what the
    acceptance test uses.
    """

    def __init__(self,
                 kernel_A: Callable,
                 kernel_W: Callable,
                 kernel_Z: Callable,
                 kernel_V: Callable,
                 kernel_X: Optional[Callable] = None,
                 lambda1_: float = 0.1,
                 lambda2_: float = 0.1,
                 zeta: float = 0.1,
                 nystrom_first_stage_m: int = 500,
                 nystrom_second_stage_m: int = 500,
                 optimize_lambda1_parameter: bool = False,
                 optimize_lambda2_parameter: bool = False,
                 optimize_zeta_parameter: bool = False,
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

        self.nystrom_first_stage_m = nystrom_first_stage_m
        self.nystrom_second_stage_m = nystrom_second_stage_m

        self.optimize_lambda1_parameter = optimize_lambda1_parameter
        self.optimize_lambda2_parameter = optimize_lambda2_parameter
        self.optimize_zeta_parameter = optimize_zeta_parameter
        self.lambda1_optimization_range = lambda1_optimization_range
        self.lambda2_optimization_range = lambda2_optimization_range
        self.zeta_optimization_range = zeta_optimization_range

        self.stage1_perc = kwargs.pop("stage1_perc", 0.5)
        self.regularization_grid_points = kwargs.pop("regularization_grid_points", 25)
        self.make_psd_eps = kwargs.pop("make_psd_eps", 1e-9)
        self.label_variance_in_lambda_opt = kwargs.pop("label_variance_in_lambda_opt", 0.0)
        self.model_seed = kwargs.pop("model_seed", 0)
        self.loocv_subsample = kwargs.pop("loocv_subsample", 2000)
        self.cme_on_landmarks = bool(kwargs.pop("cme_on_landmarks", False))

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.alpha = None
        self.B = None
        self.B_sub = None
        self.base = None
        self.ATilde = None
        self.VTilde = None
        self.XTilde = None
        self.W = None
        self.K_V_Sl = None
        self.K_V_ll = None
        self.n_cme = None
        self.stage1_idx = None
        self.stage2_idx = None
        self.nystrom_stage1_landmarks = None
        self.nystrom_stage2_landmarks = None

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _geomspace(lo, hi, n, dtype, device):
        return torch.logspace(float(np.log(lo)), float(np.log(hi)), n,
                              base=float(np.exp(1.0)), dtype=dtype, device=device)

    @staticmethod
    def _zeta_objective(zeta: float, K_VV: torch.Tensor, K_XX_WW: torch.Tensor,
                        make_psd_eps: float = 1e-9) -> torch.Tensor:
        """CME LOOCV for E[phi_W phi_X | V] -- the exact class's objective, verbatim."""
        n = K_VV.shape[0]
        ridge_w = make_psd(K_VV + n * zeta * torch.eye(n, dtype=K_VV.dtype, device=K_VV.device),
                           make_psd_eps)
        R = torch.linalg.solve(ridge_w, K_VV).T
        diag_R = torch.diagonal(R)
        S_vec = (1.0 / (1.0 - diag_R)) ** 2
        T_diag = S_vec * (torch.diagonal(K_XX_WW)
                          - 2 * torch.diagonal(K_XX_WW @ R.T)
                          + torch.diagonal(R @ K_XX_WW @ R.T))
        return torch.sum(T_diag)

    def _loocv_idx(self, size: int, seed_offset: int):
        s = self.loocv_subsample
        if (s is None) or (s <= 0) or (s >= size):
            return np.arange(size)
        return np.sort(np.random.default_rng(self.model_seed + seed_offset).choice(size, s, replace=False))

    # ------------------------------------------------------------------ fit

    def fit(self,
            AZWVX: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
            Y: torch.Tensor) -> None:
        """Tuple order is (A, Z, W, V[, X]) -- NOT the reference implementation's reference's (A, W, Z, V)."""
        if len(AZWVX) == 5:
            A, Z, W, V, X = AZWVX
        elif len(AZWVX) == 4:
            A, Z, W, V = AZWVX
            X = None
        else:
            raise ValueError("AZWVX must be either (A, Z, W, V) or (A, Z, W, V, X).")

        A = _as_2d_tensor(A, self.device)
        Z = _as_2d_tensor(Z, self.device)
        W = _as_2d_tensor(W, self.device)
        V = _as_2d_tensor(V, self.device)
        Y = _as_2d_tensor(Y, self.device)
        X = _as_2d_tensor(X, self.device) if X is not None else None
        has_X = X is not None

        # Lock every stateful kernel's length scale on the FULL sample (see _het_lock_length_scale),
        # then build stage blocks directly -- never the four n x n self-Grams.
        _het_lock_length_scale(self.kernel_A, A)
        _het_lock_length_scale(self.kernel_W, W)
        _het_lock_length_scale(self.kernel_Z, Z)
        _het_lock_length_scale(self.kernel_V, V)
        if has_X:
            _het_lock_length_scale(self.kernel_X, X)

        n_total = A.shape[0]
        indices = np.random.permutation(n_total)                   # GLOBAL numpy RNG, as in the exact class
        if 0.0 < self.stage1_perc < 1.0:
            n1 = int(n_total * self.stage1_perc)
            stage1_idx, stage2_idx = indices[:n1], indices[n1:]
        else:
            n1 = n_total
            stage1_idx, stage2_idx = indices, indices
        n2 = len(stage2_idx)

        l1 = _att_landmarks(n1, self.nystrom_first_stage_m, self.model_seed)
        l2 = _att_landmarks(n2, self.nystrom_second_stage_m, self.model_seed + 1)

        A1, A2 = _take_rows(A, stage1_idx), _take_rows(A, stage2_idx)
        Z1, Z2 = _take_rows(Z, stage1_idx), _take_rows(Z, stage2_idx)
        W1, W2 = _take_rows(W, stage1_idx), _take_rows(W, stage2_idx)
        V1, V2 = _take_rows(V, stage1_idx), _take_rows(V, stage2_idx)
        X1 = _take_rows(X, stage1_idx) if has_X else None
        X2 = _take_rows(X, stage2_idx) if has_X else None

        K_AA = self.kernel_A(A1, A1)
        K_AATilde = self.kernel_A(A1, A2)
        K_ATildeATilde = self.kernel_A(A2, A2)
        dtype = K_AA.dtype

        K_ZZ = self.kernel_Z(Z1, Z1)
        K_ZZTilde = self.kernel_Z(Z1, Z2)

        K_WW = self.kernel_W(W1, W1)
        K_WWTilde = self.kernel_W(W1, W2)

        K_VV = self.kernel_V(V1, V1)
        K_VVTilde = self.kernel_V(V1, V2)
        K_VTildeVTilde = self.kernel_V(V2, V2)

        if has_X:
            K_XX = make_psd(self.kernel_X(X1, X1), self.make_psd_eps)
            K_XXTilde = self.kernel_X(X1, X2)
            K_XTildeXTilde = make_psd(self.kernel_X(X2, X2), self.make_psd_eps)
        else:                                                      # the all-ones Gram is skipped, not built
            K_XX = K_XXTilde = K_XTildeXTilde = None

        # ---- stage 1: CME of phi_W(W) given (Z, A, V [, X]) ----
        K_ZAVX = K_ZZ * K_AA * K_VV
        if has_X:
            K_ZAVX = K_ZAVX * K_XX

        if self.optimize_lambda1_parameter:
            sub = self._loocv_idx(n1, 11)
            grid = self._geomspace(*self.lambda1_optimization_range,
                                   self.regularization_grid_points, dtype, self.device)
            K_a, K_b = _ix(K_ZAVX, sub, sub), _ix(K_WW, sub, sub)
            obj = torch.zeros(self.regularization_grid_points, dtype=dtype, device=self.device)
            with torch.no_grad():
                for i, l_val in tqdm(enumerate(grid), total=len(grid), desc="Tuning Lambda 1"):
                    obj[i] = lambda_objective_loocv(l_val, K_a, K_b,
                                                    self.label_variance_in_lambda_opt, self.make_psd_eps)
            self.lambda1_ = float(grid[int(torch.argmin(obj))])
            self.lambda1_objective_list = obj

        K_ZAVX_nm = K_ZAVX[:, l1]
        K_ZAVX_mm = K_ZAVX[l1][:, l1]
        stage1_ridge_weights = make_psd(K_ZAVX_nm.T @ K_ZAVX_nm + n1 * self.lambda1_ * K_ZAVX_mm,
                                        self.make_psd_eps)
        K_ZAVX_Tilde = K_ZZTilde * K_AATilde * K_VVTilde
        if has_X:
            K_ZAVX_Tilde = K_ZAVX_Tilde * K_XXTilde
        B = K_ZAVX_nm @ torch.linalg.solve(stage1_ridge_weights, K_ZAVX_Tilde[l1, :])     # (n1, n2)

        # ---- stage 2: KRR of Y on phi_A(ATilde) (x) mu_W (x) phi_V(VTilde) [(x) phi_X(XTilde)] ----
        stage2_ridge_weights = K_ATildeATilde * (B.T @ K_WW @ B) * K_VTildeVTilde
        if has_X:
            stage2_ridge_weights = stage2_ridge_weights * K_XTildeXTilde
        YTilde = _take_rows(Y, stage2_idx)

        if self.optimize_lambda2_parameter:
            sub = self._loocv_idx(n2, 12)
            grid = self._geomspace(*self.lambda2_optimization_range,
                                   self.regularization_grid_points, dtype, self.device)
            Ys = _take_rows(YTilde, sub)
            srw_sub = _ix(stage2_ridge_weights, sub, sub)
            K_YTilde = Ys @ Ys.T
            obj = torch.zeros(self.regularization_grid_points, dtype=dtype, device=self.device)
            with torch.no_grad():
                for i, l_val in tqdm(enumerate(grid), total=len(grid), desc="Tuning Lambda 2"):
                    obj[i] = lambda_objective_loocv(l_val, srw_sub, K_YTilde,
                                                    self.label_variance_in_lambda_opt, self.make_psd_eps)
            self.lambda2_ = float(grid[int(torch.argmin(obj))])
            self.lambda2_objective_list = obj

        srw_mn = stage2_ridge_weights[l2, :]
        srw_mm = stage2_ridge_weights[l2][:, l2]
        alpha = torch.linalg.solve(
            make_psd(srw_mn @ srw_mn.T + n2 * self.lambda2_ * srw_mm, self.make_psd_eps),
            srw_mn @ YTilde)                                                              # (m2, 1)

        # ---- stage 3: the V-conditional mean embedding's ridge ----
        if self.optimize_zeta_parameter:
            sub = self._loocv_idx(n2, 13)
            W_sub = _take_rows(W2, sub)
            K_XX_WW = self.kernel_W(W_sub, W_sub)
            if has_X:
                K_XX_WW = _ix(K_XTildeXTilde, sub, sub) * K_XX_WW
            grid = self._geomspace(*self.zeta_optimization_range,
                                   self.regularization_grid_points, dtype, self.device)
            K_VV_sub = _ix(K_VTildeVTilde, sub, sub)
            obj = torch.zeros(self.regularization_grid_points, dtype=dtype, device=self.device)
            with torch.no_grad():
                for i, z_val in tqdm(enumerate(grid), total=len(grid), desc="Tuning Zeta"):
                    obj[i] = self._zeta_objective(z_val, K_VV_sub, K_XX_WW, self.make_psd_eps)
            self.zeta = float(grid[int(torch.argmin(obj))])
            self.zeta_objective_list = obj

        # ---- prediction-time state ----
        # NOTE `self.B` is deliberately NOT retained: it is (n1, n2) = 0.8 GB of float64 at n=20000
        # and nothing downstream reads it -- prediction needs only its landmark columns. The
        # dose-response and ATT Nystrom classes keep both; this one keeps `B_sub` only.
        self.alpha = alpha
        self.B_sub = B[:, l2].contiguous()                                                # (n1, m2)
        del B
        self.ATilde = _take_rows(A2, l2)                                                  # (m2, dA)
        self.VTilde = _take_rows(V2, l2)                                                  # (m2, dV)
        self.XTilde = _take_rows(X2, l2) if has_X else None
        self.W = W1                                                                       # stage-1 W

        if self.cme_on_landmarks:
            # S = the landmark rows only.
            base = self.B_sub.T @ K_WWTilde[:, l2]                                        # (m2, m2)
            if has_X:
                base = base * _ix(K_XTildeXTilde, l2, l2)
            self.K_V_ll = _ix(K_VTildeVTilde, l2, l2)                                     # (m2, m2)
            self.K_V_Sl = None
            self.n_cme = len(l2)
        else:
            # S = every stage-2 row; only the CME SOLVE is approximated.
            base = self.B_sub.T @ K_WWTilde                                               # (m2, n2)
            if has_X:
                base = base * K_XTildeXTilde[l2, :]
            self.K_V_Sl = K_VTildeVTilde[:, l2]                                           # (n2, m2)
            self.K_V_ll = _ix(K_VTildeVTilde, l2, l2)                                     # (m2, m2)
            self.n_cme = n2
        self.base = base

        self.stage1_idx, self.stage2_idx = stage1_idx, stage2_idx
        self.nystrom_stage1_landmarks, self.nystrom_stage2_landmarks = l1, l2

    # ------------------------------------------------------------------ predict

    def _cme_weights(self, V_test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (k_V(VTilde_l, V_test), CME weights over the marginalisation sample S)."""
        K_V_lt = self.kernel_V(self.VTilde, V_test)                                       # (m2, T)
        if self.cme_on_landmarks:
            m = self.K_V_ll.shape[0]
            eye = torch.eye(m, dtype=self.K_V_ll.dtype, device=self.K_V_ll.device)
            ridge_cme = make_psd(self.K_V_ll + m * self.zeta * eye, self.make_psd_eps)
            cme = torch.linalg.solve(ridge_cme, K_V_lt)                                   # (m2, T)
        else:
            ridge_cme = make_psd(self.K_V_Sl.T @ self.K_V_Sl + self.n_cme * self.zeta * self.K_V_ll,
                                 self.make_psd_eps)
            cme = self.K_V_Sl @ torch.linalg.solve(ridge_cme, K_V_lt)                     # (n2, T)
        return K_V_lt, cme

    def predict(self, A: torch.Tensor, V: torch.Tensor,
                outcome_transformer: nn.Module = TorchIdentityTransformer()) -> torch.Tensor:
        """f_CATE(a_j, v_j) on the ROW-ALIGNED grid (A, V); returns a 1-D tensor of length len(A)."""
        A_test = _as_2d_tensor(A, self.device)
        V_test = _as_2d_tensor(V, self.device)
        if A_test.shape[0] != V_test.shape[0]:
            raise ValueError("predict expects row-aligned (a, v) pairs: "
                             f"got {A_test.shape[0]} treatments and {V_test.shape[0]} moderators.")

        K_A_lt = self.kernel_A(self.ATilde, A_test)                                       # (m2, T)
        K_V_lt, cme = self._cme_weights(V_test)
        U = self.base @ cme                                                               # (m2, T)
        pred = ((K_A_lt * K_V_lt * U) * self.alpha).sum(dim=0)                            # (T,)
        return outcome_transformer.inverse_transform(pred).reshape(-1)

    def _predict_bridge_func(self,
                             A_test: torch.Tensor,
                             W_test: torch.Tensor,
                             V_test: torch.Tensor,
                             X_test: Optional[torch.Tensor] = None) -> torch.Tensor:
        """h_hat(a, w, v[, x]); (A_test, V_test) are row-aligned, W_test is a separate grid.
        Shape (T_AV, T_W)."""
        A_test = _as_2d_tensor(A_test, self.device)
        W_test = _as_2d_tensor(W_test, self.device)
        V_test = _as_2d_tensor(V_test, self.device)

        K_A_lt = self.kernel_A(self.ATilde, A_test)                                       # (m2, T_AV)
        K_V_lt = self.kernel_V(self.VTilde, V_test)                                       # (m2, T_AV)
        K_WWTest = self.kernel_W(self.W, W_test)                                          # (n1, T_W)
        base_w = self.B_sub.T @ K_WWTest                                                  # (m2, T_W)
        if (X_test is not None) and (self.XTilde is not None):
            base_w = base_w * self.kernel_X(self.XTilde, _as_2d_tensor(X_test, self.device))
        return torch.einsum("mt,mt,mw,mo->tw", K_A_lt, K_V_lt, base_w, self.alpha)
