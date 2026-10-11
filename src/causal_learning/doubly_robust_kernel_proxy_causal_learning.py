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
from torch_utils.linear_algebra import make_psd, columns_mean_excluding_self
from torch_utils.kernel_utils import RBF
from causal_learning.kernel_proxy_causal_learning import KernelAlternativeProxyDoseResponse, KernelProxyVariableDoseResponse
from torch_utils.scalers import TorchIdentityTransformer

# Default parameter dictionaries for the sub-algorithms
TREATMENT_BRIDGE_DEFAULTS = {
    "kernel_A" : RBF(use_length_scale_heuristic = True, use_jit_call = True),
    "kernel_W" : RBF(use_length_scale_heuristic = True, use_jit_call = True),
    "kernel_Z" : RBF(use_length_scale_heuristic = True, use_jit_call = True),
    "kernel_X" : RBF(use_length_scale_heuristic = True, use_jit_call = True),
    "lambda_": 1e-3,
    "eta": 1e-3,
    "lambda2_": 1e-3,
    "optimize_lambda_parameters": True,
    "optimize_eta_parameter": True,
    "lambda_optimization_range": (1e-5, 1.0),
    "eta_optimization_range": (1e-5, 1.0),
    "stage1_perc": 0.5,
    "regularization_grid_points": 50,
    "make_psd_eps": 1e-9,
}

OUTCOME_BRIDGE_DEFAULTS = {
    "algorithm_name": "Kernel_Proxy_Variable",
    "kernel_A" : RBF(use_length_scale_heuristic = True, use_jit_call = True),
    "kernel_W" : RBF(use_length_scale_heuristic = True, use_jit_call = True),
    "kernel_Z" : RBF(use_length_scale_heuristic = True, use_jit_call = True),
    "kernel_X" : RBF(use_length_scale_heuristic = True, use_jit_call = True),
    "lambda1_": 0.1,
    "lambda2_": 0.1,
    "optimize_lambda1_parameter": True,
    "optimize_lambda2_parameter": True,
    "lambda1_optimization_range": (1e-5, 1.0),
    "lambda2_optimization_range": (1e-5, 1.0),
    "stage1_perc": 0.5,
    "regularization_grid_points": 25,
    "make_psd_eps": 1e-9,
}

class DoublyRobustKernelProxyATE(BaseEstimator, RegressorMixin, nn.Module):

    def __init__(self,
                 treatment_bridge_params: Dict = TREATMENT_BRIDGE_DEFAULTS,
                 outcome_bridge_params: Dict = OUTCOME_BRIDGE_DEFAULTS,
                 lambda_DR: float = 1e-3,
                 optimize_lambda_DR_parameter: bool = True,
                 lambda_DR_optimization_range: Tuple[float, float] = (1e-5, 1.0),
                 device = None,
                 **kwargs):

        nn.Module.__init__(self)

        self.treatment_bridge_params = treatment_bridge_params
        self.outcome_bridge_params = outcome_bridge_params

        self.lambda_DR = lambda_DR
        self.optimize_lambda_DR_parameter = optimize_lambda_DR_parameter
        self.lambda_DR_optimization_range = lambda_DR_optimization_range
        self.regularization_grid_points = kwargs.pop('regularization_grid_points', 25)
        self.label_variance_in_lambda_DR_opt = kwargs.pop('label_variance_in_lambda_DR_opt', 0.0)
        self.make_psd_eps = kwargs.pop('make_psd_eps', 5e-6)

        # Initialize sub-algorithms

        self.treatment_bridge_algo = KernelAlternativeProxyDoseResponse(
            **treatment_bridge_params,
            device = device
        )

        self.outcome_bridge_algo = KernelProxyVariableDoseResponse(
            **outcome_bridge_params,
            device = device
        )

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self,
            AZWX: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
            Y: torch.Tensor) -> None:

        if len(AZWX) == 4:
            A, Z, W, X = AZWX
        else:
            A, Z, W = AZWX
            X = None

        # 1. Fit individual components
        self.treatment_bridge_algo.fit(AZWX, Y)
        self.outcome_bridge_algo.fit(AZWX, Y)

        # 2. Extract outcome kernel Gram matrix
        self.K_AA = self.outcome_bridge_algo.kernel_A(A, A)

        # 3. Optimize DR lambda using the external LOOCV function
        if self.optimize_lambda_DR_parameter:
            l_min, l_max = self.lambda_DR_optimization_range
            lambda_list = torch.exp(torch.linspace(np.log(l_min), np.log(l_max), self.regularization_grid_points, device=self.device))

            # Feature kernel for DR interaction: K_ZW = K_ZZ * K_WW
            K_ZW = self.treatment_bridge_algo.kernel_Z(Z, Z) * self.outcome_bridge_algo.kernel_W(W, W)

            obj_list = torch.zeros(self.regularization_grid_points, device=self.device)
            with torch.no_grad():
                for i, l_val in tqdm(enumerate(lambda_list), total=len(lambda_list), desc="Tuning DR Lambda"):
                    # REPLACED manual logic with lambda_objective_loocv
                    obj_list[i] = lambda_objective_loocv(
                        l_val,
                        self.K_AA,
                        K_ZW,
                        self.label_variance_in_lambda_DR_opt,
                        self.make_psd_eps
                    )

            self.lambda_DR = lambda_list[torch.argmin(obj_list)].item()

        # Save context for prediction
        self.W, self.Z, self.X, self.A = W, Z, X, A

    def predict(self, A_test: torch.Tensor, outcome_transformer: nn.Module = TorchIdentityTransformer()) -> torch.Tensor:
        """Standard DR estimation: f = bridge_treatment + bridge_outcome - slack"""
        A_test = A_test.reshape(-1, 1) if A_test.ndim != 2 else A_test
        n = self.A.shape[0]

        # Component predictions
        treatment_pred = self.treatment_bridge_algo.predict(A_test, outcome_transformer).reshape(-1)
        outcome_pred = self.outcome_bridge_algo.predict(A_test, outcome_transformer).reshape(-1)

        # Bridge function evaluations
        phi_eval = self.treatment_bridge_algo._predict_bridge_func(A_test, self.Z, self.X)
        h_eval = self.outcome_bridge_algo._predict_bridge_func(A_test, self.W, self.X)

        # Solve for interaction weights
        krr_lhs = make_psd(self.K_AA + n * self.lambda_DR * torch.eye(n, device=self.device), eps=self.make_psd_eps)
        krr_rhs = self.outcome_bridge_algo.kernel_A(self.A, A_test)
        dr_krr_weights = torch.linalg.solve(krr_lhs, krr_rhs)

        # Correction term
        slack_prediction = outcome_transformer.inverse_transform(((phi_eval * h_eval) * dr_krr_weights.T).sum(dim=1, keepdims = True)).squeeze(-1)

        return treatment_pred + outcome_pred - slack_prediction