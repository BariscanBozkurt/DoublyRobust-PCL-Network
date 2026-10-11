import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Optional, Tuple, Callable, Union
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler # Use abstract base class for type hint
from torch.optim.lr_scheduler import ExponentialLR

import copy
import numpy as np
from tqdm import tqdm
import pylab as pl
from IPython.display import clear_output, display

import sys
sys.path.append("..")
from torch_utils.scalers import TorchIdentityTransformer
from torch_utils.linear_algebra import add_const_col, fit_linear_proximal, outer_prod, outer_prod_batch
from torch_utils.model_helpers import get_last_linear_out_features
from torch_utils.torch_eval import evaluate_loss_on_dataloader
from torch_utils.networks import EnsembleConditionalMeanMLP


class TreatmentBridgePCLNET(nn.Module):

    def __init__(self,
                 first_stage_featurizer: nn.Module, # Neural network featurizer for the first stage that uses (A, W, X: Optional)
                 treatment_backdoor_featurizer: nn.Module, # Neural network for (A, X: Optional) in the second stage
                 treatment_proxy_featurizer: nn.Module, # Neural network for Z in the second stage
                 final_layer_first_stage_weight: Optional[torch.Tensor] = None,
                 final_layer_second_stage_weight: Optional[torch.Tensor] = None,
                 outcome_dim: int = 1,
                 dens_ratio_transformer: Optional[nn.Module] = TorchIdentityTransformer(),
                 device: str = "cuda",
                 **kwargs,):
        super().__init__()

        self.first_stage_featurizer = first_stage_featurizer.to(device)
        self.treatment_proxy_featurizer = treatment_proxy_featurizer.to(device)
        self.treatment_backdoor_featurizer = treatment_backdoor_featurizer.to(device)

        treatment_proxy_feature_dim = get_last_linear_out_features(treatment_proxy_featurizer)

        def init_first_stage_final_layer_weight(final_layer_first_stage_weight_val):
            if final_layer_first_stage_weight_val is None:
                first_stage_in_dim = get_last_linear_out_features(first_stage_featurizer)
                final_layer_first_stage_weight_val = torch.zeros(treatment_proxy_feature_dim,
                                                                 first_stage_in_dim + 1,
                                                                 device=device)
                final_layer_first_stage_weight_val.data[:, -1] = 1.0 / (first_stage_in_dim)
            return final_layer_first_stage_weight_val


        final_layer_first_stage_weight_val = init_first_stage_final_layer_weight(final_layer_first_stage_weight)
        final_layer_first_stage_inside_second_stage_weight_val = init_first_stage_final_layer_weight(final_layer_first_stage_weight)

        if final_layer_second_stage_weight is None:
            treatment_backdoor_featurizer_dim = get_last_linear_out_features(treatment_backdoor_featurizer)
            final_layer_second_stage_weight = torch.zeros(outcome_dim,
                                                          treatment_proxy_feature_dim * treatment_backdoor_featurizer_dim + 1,
                                                          device=device)
            final_layer_second_stage_weight.data[:, -1] = 0.0

        # Registering the final linear weights! This ensures they are included in model.state_dict
        self.register_buffer("final_layer_first_stage_weight", final_layer_first_stage_weight_val)
        self.register_buffer("final_layer_first_stage_inside_second_stage_weight", final_layer_first_stage_inside_second_stage_weight_val)
        self.register_buffer("final_layer_second_stage_weight", final_layer_second_stage_weight)
        self.final_layer_first_stage_weight.requires_grad = False
        self.final_layer_second_stage_weight.requires_grad = False
        self.final_layer_first_stage_inside_second_stage_weight.requires_grad = False
        self.dens_ratio_transformer = dens_ratio_transformer.to(device)
        self.device = device

    # =========================================================================
    # Helpers: Data Unpacking & Feature Construction
    # =========================================================================

    def _unpack_batch(self, batch: Tuple[torch.Tensor, ...]):
        """
        Unpacks batch tuple into named tensors.
        Assumes structure: (A, Y_scaled, Z, W, [X], [dens_ratio])
        Returns: A, outcome_proxy (Y), treatment_proxy (Z), backdoor (W/X), dens_ratio
        """
        batch = [x.to(self.device) for x in batch]

        if len(batch) > 5:
            # Case: With X (High Dim / Confounder)
            # A, Y, Z, W, X, dens_ratio
            treatment, _, treatment_proxy, outcome_proxy, backdoor, dens_ratio = batch
        else:
            # Case: No X
            # A, Y, Z, W, dens_ratio
            treatment, _, treatment_proxy, outcome_proxy, dens_ratio = batch
            backdoor = None

        return treatment, treatment_proxy, outcome_proxy, backdoor, dens_ratio

    def _compute_first_stage_features(self, treatment, outcome_proxy, backdoor=None):
        """
        Computes phi_1(A, W, X).
        """
        if backdoor is not None:
            # Single network takes concatenation
            inp = torch.hstack([treatment, backdoor, outcome_proxy])
        else:
            inp = torch.hstack([treatment, outcome_proxy])

        feat = self.first_stage_featurizer(inp)
        return add_const_col(feat)

    def _compute_second_stage_features(self, treatment, outcome_proxy, backdoor, w1_weight_matrix):
        """
        Computes the outer product features for the second stage.
        Psi(A, X) (outer) E[phi(Z) | A, W, X]
        """
        with torch.no_grad():
            # 1. Get E[phi(Z) | A, W, X] using Stage 1 weights
            feat_stage1 = self._compute_first_stage_features(treatment, outcome_proxy, backdoor)

        pred_cond_t_proxy_feat = torch.matmul(feat_stage1, w1_weight_matrix)
        # 2. Get Psi(A, X)
        if backdoor is not None:
            inp_backdoor = torch.hstack([treatment, backdoor])
        else:
            inp_backdoor = treatment
        feat_backdoor = self.treatment_backdoor_featurizer(inp_backdoor)
        # 3. Outer Product & Flatten
        combined_feat = outer_prod(feat_backdoor, pred_cond_t_proxy_feat).flatten(start_dim=1)
        return add_const_col(combined_feat)

    # =========================================================================
    # Forward & Predictions
    # =========================================================================

    def forward(self, treatment, outcome_proxy, backdoor: Optional = None):
        """
        Forward method predicts density ratios given the input features. This is used during evaluation and prediction, not training updates.
        """
        feat_stage2 = self._compute_second_stage_features(treatment, outcome_proxy, backdoor, self.final_layer_first_stage_weight.T)
        pred_raw = torch.matmul(feat_stage2, self.final_layer_second_stage_weight.T)
        return pred_raw

    def first_stage_forward(self, treatment, outcome_proxy, backdoor: Optional = None):
        first_stage_feature = self._compute_first_stage_features(treatment, outcome_proxy, backdoor)
        predicted_conditional_treatment_proxy_feature = torch.matmul(first_stage_feature, self.final_layer_first_stage_weight.T)
        return predicted_conditional_treatment_proxy_feature

    def predict_dens_ratio(self, treatment, outcome_proxy, backdoor=None):
        """
        Predicts r(A, X, W) (density ratios) using Stage 2 weights.
        """
        # We use the 'inside_second_stage' weights logic usually, or the main weights?
        # Typically for inference we use the standard flow.
        feat_stage2 = self._compute_second_stage_features(treatment, outcome_proxy, backdoor, self.final_layer_first_stage_weight.T)
        pred_raw = torch.matmul(feat_stage2, self.final_layer_second_stage_weight.T)
        dens_ratio_pred = self.dens_ratio_transformer.inverse_transform(pred_raw)
        return dens_ratio_pred

    def predict_treatment_bridge_function(self, treatment, treatment_proxy, backdoor=None):
        """
        Predicts the treatment bridge function h(A, W, X) using the first stage final layer weights.
        This is not the main prediction target but can be useful for diagnostics.
        """
        # 1. Get Psi(Z)
        phi_Z = self.treatment_proxy_featurizer(treatment_proxy)
        # 2. Get Psi(A, X)
        if backdoor is not None:
            inp_backdoor = torch.hstack([treatment, backdoor])
        else:
            inp_backdoor = treatment
        feat_backdoor = self.treatment_backdoor_featurizer(inp_backdoor)
        feauture = add_const_col(outer_prod_batch(feat_backdoor, phi_Z).flatten(start_dim=1))
        treatment_bridge_pred = torch.matmul(feauture, self.final_layer_second_stage_weight.T)
        # treatment_bridge_pred = self.dens_ratio_transformer.inverse_transform(pred_raw)
        return treatment_bridge_pred

    # =========================================================================
    # Training / Updates
    # =========================================================================

    def update_first_stage_final_layer_with_batch_regression(
        self,
        data_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        weight_regularizer: float,
        consider_prev_weight: bool = True,
        update_type: str = "ridge"):
        """Computes first stage final layer with regression."""
        ##############################################################################
        # Extract data from the input tuple.
        ##############################################################################
        treatment, treatment_proxy, outcome_proxy, backdoor, _ = self._unpack_batch(data_tuple)

        ##############################################################################
        # Construct First Stage Feature.
        ##############################################################################
        first_stage_feature = self._compute_first_stage_features(treatment, outcome_proxy, backdoor)

        ##############################################################################
        # Construct output for regression (AKA y).
        ##############################################################################
        treatment_proxy_feature = self.treatment_proxy_featurizer(treatment_proxy)

        ##############################################################################
        # Get previous weights w_{t}
        ##############################################################################
        weight_previous = self.final_layer_first_stage_weight.detach()

        ##############################################################################
        # Do the weight update w_{t+1} = REG_SOLUTION(w_t, x,y)
        ##############################################################################
        new_weight = fit_linear_proximal(
            # output (y)
            treatment_proxy_feature,
            # input (x)
            first_stage_feature,
            # prev weight (can use 0 instead -> becomes ridge)
            int(consider_prev_weight) * (weight_previous.T),
            # weight regularizer
            weight_regularizer,
            # update type
            update_type)
        ##############################################################################
        # Return the result
        ##############################################################################
        return new_weight.T

    def update_second_stage_final_layer_with_batch_regression(
        self,
        data_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        weight_regularizer: float,
        consider_prev_weight: bool = True):
        """Update final layer weights using linear regression on batch data."""
        ##############################################################################
        # EXTRACT DATA
        ##############################################################################
        treatment, _, outcome_proxy, backdoor, dens_ratio = self._unpack_batch(data_tuple)

        ##############################################################################
        # Compute SECOND stage features and OUTCOMES.
        ##############################################################################
        # INCREDIBLY IMPORTANT
        FIRST_STAGE_INSIDE_SECOND_STAGE_WEIGHT = self.final_layer_first_stage_inside_second_stage_weight.data.T
        second_stage_feature = self._compute_second_stage_features(treatment, outcome_proxy, backdoor, FIRST_STAGE_INSIDE_SECOND_STAGE_WEIGHT)

        ##############################################################################
        # Extract the previous weight for final layer and update it with closed form.
        ##############################################################################
        weight_previous = self.final_layer_second_stage_weight .detach()
        weight = fit_linear_proximal(dens_ratio, second_stage_feature, int(consider_prev_weight) * (weight_previous.T), weight_regularizer, "ridge")
        self.final_layer_second_stage_weight.data = weight.T

    def update_final_layer_with_batch_regression(self, data_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                                                 weight_regularizer: float,
                                                 stage: str,
                                                 consider_prev_weight: bool = True,
                                                 update_type: str = "ridge",
                                                 ):
        """
        Update final layer weights using linear regression on batch data.
        """
        data_tuple = [x.to(self.device) for x in data_tuple]
        if stage == "first":
            self.final_layer_first_stage_weight.data = self.update_first_stage_final_layer_with_batch_regression(
                data_tuple=data_tuple,
                weight_regularizer=weight_regularizer,
                consider_prev_weight=consider_prev_weight,
                update_type=update_type)

        elif stage == "second":
            self.update_second_stage_final_layer_with_batch_regression(
                data_tuple=data_tuple,
                weight_regularizer=weight_regularizer,
                consider_prev_weight=consider_prev_weight)
        else:
            raise ValueError(f"Unknown stage = {stage}")

    def first_stage_loss(self, first_stage_data_tuple, loss_fn = None):
        if loss_fn is None: loss_fn = nn.MSELoss()
        ##############################################################################
        # Extract data from the input tuple.
        ##############################################################################
        treatment, treatment_proxy, outcome_proxy, backdoor, _ = self._unpack_batch(first_stage_data_tuple)

        ##############################################################################
        # Construct the Estimate of Conditional Treatment Proxy Feature.
        ##############################################################################
        # first_stage_feature = self._compute_first_stage_features(treatment, outcome_proxy, backdoor)
        # predicted_conditional_treatment_proxy_feature = torch.matmul(first_stage_feature, self.final_layer_first_stage_weight.T)
        predicted_conditional_treatment_proxy_feature = self.first_stage_forward(treatment, outcome_proxy, backdoor)
        with torch.no_grad():
            treatment_proxy_feature = self.treatment_proxy_featurizer(treatment_proxy)
        loss = loss_fn(treatment_proxy_feature, predicted_conditional_treatment_proxy_feature)
        return loss

    def second_stage_loss(self, first_stage_tuple, second_stage_tuple,
                          loss_fn = None,
                          second_stage_first_final_layer_regularizer = 1e-2,
                          negative_penalty: float = 0.0):
        if loss_fn is None: loss_fn = nn.MSELoss()
        ##############################################################################
        # Extract first stage data from the input tuple.
        ##############################################################################
        treatment, treatment_proxy, outcome_proxy, backdoor, _ = self._unpack_batch(first_stage_tuple)

        ######################################################################################################
        # Construct First Stage Feature and first stage final layer weight for inside second stage.
        ######################################################################################################
        with torch.no_grad():
            first_stage_feature = self._compute_first_stage_features(treatment, outcome_proxy, backdoor)
            weight_previous = self.final_layer_first_stage_inside_second_stage_weight.detach()

        treatment_proxy_feature = self.treatment_proxy_featurizer(treatment_proxy)
        new_weight = fit_linear_proximal(treatment_proxy_feature, first_stage_feature, weight_previous.T,
                                         second_stage_first_final_layer_regularizer)

        ##############################################################################
        # Extract second stage data from the input tuple.
        ##############################################################################
        treatment, treatment_proxy, outcome_proxy, backdoor, dens_ratios = self._unpack_batch(second_stage_tuple)
        # ##############################################################################
        # # Compute SECOND stage features
        # ##############################################################################
        second_stage_feature = self._compute_second_stage_features(treatment, outcome_proxy, backdoor, new_weight)

        dens_ratio_preds = torch.matmul(second_stage_feature, self.final_layer_second_stage_weight.T)
        loss = loss_fn(dens_ratio_preds, dens_ratios)
        loss += (nn.ReLU()(-self.dens_ratio_transformer.inverse_transform(dens_ratio_preds)) ** 2).mean() * negative_penalty
        with torch.no_grad():
            self.final_layer_first_stage_inside_second_stage_weight.data = new_weight.detach().T
        return loss

    # =========================================================================
    # Evaluation Helpers
    # =========================================================================

    def evaluate_loss(self, dataloader, stage="first"):
        self.eval()
        total_loss = 0.0
        n_total = 0
        loss_fn = nn.MSELoss()

        with torch.no_grad():
            for batch in dataloader:
                bs = batch[0].shape[0]
                n_total += bs

                if stage == "first":
                    # Evaluate Stage 1 reconstruction loss
                    loss = self.first_stage_loss(batch, loss_fn)
                elif stage == "second":
                    # Evaluate Stage 2 density prediction loss against ground truth
                    # We pass the same batch for s1 and s2 arg because we just want the loss calculation
                    # Note: We skip the W1_in_2 update during eval!
                    # So we manually compute pred vs target
                    A_batch, Z_batch, W_batch, X_batch, dens_ratio_batch = self._unpack_batch(batch)
                    if X_batch is not None: X_batch = X_batch.to(self.device)
                    pred = self.forward(A_batch.to(self.device),
                                        W_batch.to(self.device),
                                        X_batch)
                    loss = loss_fn(pred, dens_ratio_batch)

                total_loss += loss.item() * bs

        return total_loss / n_total if n_total > 0 else 0.0

    def freeze_featurizer_params(self, stage: str):
        if stage == "first_stage":
            self.treatment_featurizer_first_stage.train(False)
            self.outcome_proxy_featurizer.train(False)
            if (self.backdoor_featurizer_first_stage is not None):
                self.backdoor_featurizer_first_stage.train(False)

        elif stage == "second_stage":
            self.treatment_featurizer_second_stage.train(False)
            self.treatment_proxy_featurizer.train(False)
            if (self.backdoor_featurizer_second_stage is not None):
                self.backdoor_featurizer_second_stage.train(False)

    def unfreeze_featurizer_params(self, stage: str):
        if stage == "first_stage":
            self.treatment_featurizer_first_stage.train(True)
            self.outcome_proxy_featurizer.train(True)
            if (self.backdoor_featurizer_first_stage is not None):
                self.backdoor_featurizer_first_stage.train(True)

        elif stage == "second_stage":
            self.treatment_featurizer_second_stage.train(True)
            self.treatment_proxy_featurizer.train(True)
            if (self.backdoor_featurizer_second_stage is not None):
                self.backdoor_featurizer_second_stage.train(True)


class HeterogeneousTreatmentBridgePCLNET(nn.Module):

    def __init__(self,
                 first_stage_featurizer: nn.Module, # Neural network featurizer for the first stage that uses (A, W, X: Optional)
                 treatment_backdoor_featurizer: nn.Module, # Neural network for (A, X: Optional) in the second stage
                 treatment_proxy_featurizer: nn.Module, # Neural network for Z in the second stage
                 final_layer_first_stage_weight: Optional[torch.Tensor] = None,
                 final_layer_second_stage_weight: Optional[torch.Tensor] = None,
                 outcome_dim: int = 1,
                 dens_ratio_transformer: Optional[nn.Module] = TorchIdentityTransformer(),
                 device: str = "cuda",
                 **kwargs,):
        super().__init__()

        self.first_stage_featurizer = first_stage_featurizer.to(device)
        self.treatment_proxy_featurizer = treatment_proxy_featurizer.to(device)
        self.treatment_backdoor_featurizer = treatment_backdoor_featurizer.to(device)

        treatment_proxy_feature_dim = get_last_linear_out_features(treatment_proxy_featurizer)

        def init_first_stage_final_layer_weight(final_layer_first_stage_weight_val):
            if final_layer_first_stage_weight_val is None:
                first_stage_in_dim = get_last_linear_out_features(first_stage_featurizer)
                final_layer_first_stage_weight_val = torch.zeros(treatment_proxy_feature_dim,
                                                                 first_stage_in_dim + 1,
                                                                 device=device)
                final_layer_first_stage_weight_val.data[:, -1] = 1.0 / (first_stage_in_dim)
            return final_layer_first_stage_weight_val


        final_layer_first_stage_weight_val = init_first_stage_final_layer_weight(final_layer_first_stage_weight)
        final_layer_first_stage_inside_second_stage_weight_val = init_first_stage_final_layer_weight(final_layer_first_stage_weight)

        if final_layer_second_stage_weight is None:
            treatment_backdoor_featurizer_dim = get_last_linear_out_features(treatment_backdoor_featurizer)
            final_layer_second_stage_weight = torch.zeros(outcome_dim,
                                                          treatment_proxy_feature_dim * treatment_backdoor_featurizer_dim + 1,
                                                          device=device)
            final_layer_second_stage_weight.data[:, -1] = 0.0

        # Registering the final linear weights! This ensures they are included in model.state_dict
        self.register_buffer("final_layer_first_stage_weight", final_layer_first_stage_weight_val)
        self.register_buffer("final_layer_first_stage_inside_second_stage_weight", final_layer_first_stage_inside_second_stage_weight_val)
        self.register_buffer("final_layer_second_stage_weight", final_layer_second_stage_weight)
        self.final_layer_first_stage_weight.requires_grad = False
        self.final_layer_second_stage_weight.requires_grad = False
        self.final_layer_first_stage_inside_second_stage_weight.requires_grad = False
        self.dens_ratio_transformer = dens_ratio_transformer.to(device)
        self.device = device

    # =========================================================================
    # Helpers: Data Unpacking & Feature Construction
    # =========================================================================

    def _unpack_batch(self, batch: Tuple[torch.Tensor, ...]):
        """
        Unpacks batch tuple into named tensors.
        Assumes structure: (A, Y_scaled, Z, W, [X], [dens_ratio])
        Returns: A, outcome_proxy (Y), treatment_proxy (Z), backdoor (W/X), dens_ratio
        """
        batch = [x.to(self.device) for x in batch]

        if len(batch) > 6:
            # Case: With X (High Dim / Confounder)
            # A, Y, Z, W, X, dens_ratio
            treatment, _, treatment_proxy, outcome_proxy, covariate, backdoor, dens_ratio = batch
        else:
            # Case: No X
            # A, Y, Z, W, dens_ratio
            treatment, _, treatment_proxy, outcome_proxy, covariate, dens_ratio = batch
            backdoor = None

        return treatment, treatment_proxy, outcome_proxy, covariate, backdoor, dens_ratio

    def _compute_first_stage_features(self, treatment, covariate, outcome_proxy, backdoor=None):
        """
        Computes phi_1(A, W, X).
        """
        if backdoor is not None:
            # Single network takes concatenation
            inp = torch.hstack([treatment, covariate, backdoor, outcome_proxy])
        else:
            inp = torch.hstack([treatment, covariate, outcome_proxy])

        feat = self.first_stage_featurizer(inp)
        return add_const_col(feat)

    def _compute_second_stage_features(self, treatment, covariate, outcome_proxy, backdoor, w1_weight_matrix):
        """
        Computes the outer product features for the second stage.
        Psi(A, X) (outer) E[phi(Z) | A, W, X]
        """
        with torch.no_grad():
            # 1. Get E[phi(Z) | A, W, X] using Stage 1 weights
            feat_stage1 = self._compute_first_stage_features(treatment, covariate, outcome_proxy, backdoor)

        pred_cond_t_proxy_feat = torch.matmul(feat_stage1, w1_weight_matrix)
        # 2. Get Psi(A, V, X)
        if backdoor is not None:
            inp_backdoor = torch.hstack([treatment, covariate, backdoor])
        else:
            inp_backdoor = torch.hstack([treatment, covariate])
        feat_backdoor = self.treatment_backdoor_featurizer(inp_backdoor)
        # 3. Outer Product & Flatten
        combined_feat = outer_prod(feat_backdoor, pred_cond_t_proxy_feat).flatten(start_dim=1)
        return add_const_col(combined_feat)

    # =========================================================================
    # Forward & Predictions
    # =========================================================================

    def forward(self, treatment, covariate, outcome_proxy, backdoor: Optional = None):
        """
        Forward method predicts density ratios given the input features. This is used during evaluation and prediction, not training updates.
        """
        feat_stage2 = self._compute_second_stage_features(treatment, covariate, outcome_proxy, backdoor, self.final_layer_first_stage_weight.T)
        pred_raw = torch.matmul(feat_stage2, self.final_layer_second_stage_weight.T)
        return pred_raw

    def first_stage_forward(self, treatment, covariate, outcome_proxy, backdoor: Optional = None):
        first_stage_feature = self._compute_first_stage_features(treatment, covariate, outcome_proxy, backdoor)
        predicted_conditional_treatment_proxy_feature = torch.matmul(first_stage_feature, self.final_layer_first_stage_weight.T)
        return predicted_conditional_treatment_proxy_feature

    def predict_dens_ratio(self, treatment, covariate, outcome_proxy, backdoor=None):
        """
        Predicts r(A, X, W) (density ratios) using Stage 2 weights.
        """
        # We use the 'inside_second_stage' weights logic usually, or the main weights?
        # Typically for inference we use the standard flow.
        feat_stage2 = self._compute_second_stage_features(treatment, covariate, outcome_proxy, backdoor, self.final_layer_first_stage_weight.T)
        pred_raw = torch.matmul(feat_stage2, self.final_layer_second_stage_weight.T)
        dens_ratio_pred = self.dens_ratio_transformer.inverse_transform(pred_raw)
        return dens_ratio_pred

    def predict_treatment_bridge_function(self, treatment, covariate, treatment_proxy, backdoor=None):
        """
        Predicts the treatment bridge function h(A, W, X) using the first stage final layer weights.
        This is not the main prediction target but can be useful for diagnostics.
        """
        # 1. Get Psi(Z)
        phi_Z = self.treatment_proxy_featurizer(treatment_proxy)
        # 2. Get Psi(A, X)
        if backdoor is not None:
            inp_backdoor = torch.hstack([treatment, covariate, backdoor])
        else:
            inp_backdoor = torch.hstack([treatment, covariate])
        feat_backdoor = self.treatment_backdoor_featurizer(inp_backdoor)
        feauture = add_const_col(outer_prod_batch(feat_backdoor, phi_Z).flatten(start_dim=1))
        treatment_bridge_pred = torch.matmul(feauture, self.final_layer_second_stage_weight.T)
        # treatment_bridge_pred = self.dens_ratio_transformer.inverse_transform(pred_raw)
        return treatment_bridge_pred

    # =========================================================================
    # Training / Updates
    # =========================================================================

    def update_first_stage_final_layer_with_batch_regression(
        self,
        data_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        weight_regularizer: float,
        consider_prev_weight: bool = True,
        update_type: str = "ridge"):
        """Computes first stage final layer with regression."""
        ##############################################################################
        # Extract data from the input tuple.
        ##############################################################################
        treatment, treatment_proxy, outcome_proxy, covariate, backdoor, _ = self._unpack_batch(data_tuple)

        ##############################################################################
        # Construct First Stage Feature.
        ##############################################################################
        first_stage_feature = self._compute_first_stage_features(treatment, covariate, outcome_proxy, backdoor)

        ##############################################################################
        # Construct output for regression (AKA y).
        ##############################################################################
        treatment_proxy_feature = self.treatment_proxy_featurizer(treatment_proxy)

        ##############################################################################
        # Get previous weights w_{t}
        ##############################################################################
        weight_previous = self.final_layer_first_stage_weight.detach()

        ##############################################################################
        # Do the weight update w_{t+1} = REG_SOLUTION(w_t, x,y)
        ##############################################################################
        new_weight = fit_linear_proximal(
            # output (y)
            treatment_proxy_feature,
            # input (x)
            first_stage_feature,
            # prev weight (can use 0 instead -> becomes ridge)
            int(consider_prev_weight) * (weight_previous.T),
            # weight regularizer
            weight_regularizer,
            # update type
            update_type)
        ##############################################################################
        # Return the result
        ##############################################################################
        return new_weight.T

    def update_second_stage_final_layer_with_batch_regression(
        self,
        data_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        weight_regularizer: float,
        consider_prev_weight: bool = True):
        """Update final layer weights using linear regression on batch data."""
        ##############################################################################
        # EXTRACT DATA
        ##############################################################################
        treatment, _, outcome_proxy, covariate, backdoor, dens_ratio = self._unpack_batch(data_tuple)

        ##############################################################################
        # Compute SECOND stage features and OUTCOMES.
        ##############################################################################
        # INCREDIBLY IMPORTANT
        FIRST_STAGE_INSIDE_SECOND_STAGE_WEIGHT = self.final_layer_first_stage_inside_second_stage_weight.data.T
        second_stage_feature = self._compute_second_stage_features(treatment, covariate, outcome_proxy, backdoor, FIRST_STAGE_INSIDE_SECOND_STAGE_WEIGHT)

        ##############################################################################
        # Extract the previous weight for final layer and update it with closed form.
        ##############################################################################
        weight_previous = self.final_layer_second_stage_weight .detach()
        weight = fit_linear_proximal(dens_ratio, second_stage_feature, int(consider_prev_weight) * (weight_previous.T), weight_regularizer, "ridge")
        self.final_layer_second_stage_weight.data = weight.T

    def update_final_layer_with_batch_regression(self, data_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                                                 weight_regularizer: float,
                                                 stage: str,
                                                 consider_prev_weight: bool = True,
                                                 update_type: str = "ridge",
                                                 ):
        """
        Update final layer weights using linear regression on batch data.
        """
        data_tuple = [x.to(self.device) for x in data_tuple]
        if stage == "first":
            self.final_layer_first_stage_weight.data = self.update_first_stage_final_layer_with_batch_regression(
                data_tuple=data_tuple,
                weight_regularizer=weight_regularizer,
                consider_prev_weight=consider_prev_weight,
                update_type=update_type)

        elif stage == "second":
            self.update_second_stage_final_layer_with_batch_regression(
                data_tuple=data_tuple,
                weight_regularizer=weight_regularizer,
                consider_prev_weight=consider_prev_weight)
        else:
            raise ValueError(f"Unknown stage = {stage}")

    def first_stage_loss(self, first_stage_data_tuple, loss_fn = None):
        if loss_fn is None: loss_fn = nn.MSELoss()
        ##############################################################################
        # Extract data from the input tuple.
        ##############################################################################
        treatment, treatment_proxy, outcome_proxy, covariate, backdoor, _ = self._unpack_batch(first_stage_data_tuple)

        ##############################################################################
        # Construct the Estimate of Conditional Treatment Proxy Feature.
        ##############################################################################
        # first_stage_feature = self._compute_first_stage_features(treatment, outcome_proxy, backdoor)
        # predicted_conditional_treatment_proxy_feature = torch.matmul(first_stage_feature, self.final_layer_first_stage_weight.T)
        predicted_conditional_treatment_proxy_feature = self.first_stage_forward(treatment, covariate, outcome_proxy, backdoor)
        with torch.no_grad():
            treatment_proxy_feature = self.treatment_proxy_featurizer(treatment_proxy)
        loss = loss_fn(treatment_proxy_feature, predicted_conditional_treatment_proxy_feature)
        return loss

    def second_stage_loss(self, first_stage_tuple, second_stage_tuple,
                          loss_fn = None,
                          second_stage_first_final_layer_regularizer = 1e-2,
                          negative_penalty: float = 0.0):
        if loss_fn is None: loss_fn = nn.MSELoss()
        ##############################################################################
        # Extract first stage data from the input tuple.
        ##############################################################################
        treatment, treatment_proxy, outcome_proxy, covariate, backdoor, _ = self._unpack_batch(first_stage_tuple)

        ######################################################################################################
        # Construct First Stage Feature and first stage final layer weight for inside second stage.
        ######################################################################################################
        with torch.no_grad():
            first_stage_feature = self._compute_first_stage_features(treatment, covariate, outcome_proxy, backdoor)
            weight_previous = self.final_layer_first_stage_inside_second_stage_weight.detach()

        treatment_proxy_feature = self.treatment_proxy_featurizer(treatment_proxy)
        new_weight = fit_linear_proximal(treatment_proxy_feature, first_stage_feature, weight_previous.T,
                                         second_stage_first_final_layer_regularizer)

        ##############################################################################
        # Extract second stage data from the input tuple.
        ##############################################################################
        treatment, treatment_proxy, outcome_proxy, covariate, backdoor, dens_ratios = self._unpack_batch(second_stage_tuple)
        # ##############################################################################
        # # Compute SECOND stage features
        # ##############################################################################
        second_stage_feature = self._compute_second_stage_features(treatment, covariate, outcome_proxy, backdoor, new_weight)

        dens_ratio_preds = torch.matmul(second_stage_feature, self.final_layer_second_stage_weight.T)
        loss = loss_fn(dens_ratio_preds, dens_ratios)
        loss += (nn.ReLU()(-self.dens_ratio_transformer.inverse_transform(dens_ratio_preds)) ** 2).mean() * negative_penalty
        with torch.no_grad():
            self.final_layer_first_stage_inside_second_stage_weight.data = new_weight.detach().T
        return loss

    # =========================================================================
    # Evaluation Helpers
    # =========================================================================

    def evaluate_loss(self, dataloader, stage="first"):
        self.eval()
        total_loss = 0.0
        n_total = 0
        loss_fn = nn.MSELoss()

        with torch.no_grad():
            for batch in dataloader:
                bs = batch[0].shape[0]
                n_total += bs

                if stage == "first":
                    # Evaluate Stage 1 reconstruction loss
                    loss = self.first_stage_loss(batch, loss_fn)
                elif stage == "second":
                    # Evaluate Stage 2 density prediction loss against ground truth
                    # We pass the same batch for s1 and s2 arg because we just want the loss calculation
                    # Note: We skip the W1_in_2 update during eval!
                    # So we manually compute pred vs target
                    A_batch, Z_batch, W_batch, V_batch, X_batch, dens_ratio_batch = self._unpack_batch(batch)
                    if X_batch is not None: X_batch = X_batch.to(self.device)
                    pred = self.forward(A_batch.to(self.device),
                                        V_batch.to(self.device),
                                        W_batch.to(self.device),
                                        X_batch)
                    loss = loss_fn(pred, dens_ratio_batch)

                total_loss += loss.item() * bs

        return total_loss / n_total if n_total > 0 else 0.0

    def freeze_featurizer_params(self, stage: str):
        if stage == "first_stage":
            self.treatment_featurizer_first_stage.train(False)
            self.outcome_proxy_featurizer.train(False)
            if (self.backdoor_featurizer_first_stage is not None):
                self.backdoor_featurizer_first_stage.train(False)

        elif stage == "second_stage":
            self.treatment_featurizer_second_stage.train(False)
            self.treatment_proxy_featurizer.train(False)
            if (self.backdoor_featurizer_second_stage is not None):
                self.backdoor_featurizer_second_stage.train(False)

    def unfreeze_featurizer_params(self, stage: str):
        if stage == "first_stage":
            self.treatment_featurizer_first_stage.train(True)
            self.outcome_proxy_featurizer.train(True)
            if (self.backdoor_featurizer_first_stage is not None):
                self.backdoor_featurizer_first_stage.train(True)

        elif stage == "second_stage":
            self.treatment_featurizer_second_stage.train(True)
            self.treatment_proxy_featurizer.train(True)
            if (self.backdoor_featurizer_second_stage is not None):
                self.backdoor_featurizer_second_stage.train(True)


def get_annealed_value_for_reg(start_val, end_val, current_step, total_steps, method="linear"):
    """
    Calculates the hyperparameter value for the current step.
    Supports 'linear', 'cosine', and 'exponential' schedules.
    """
    # Clamp step to avoid going out of bounds
    current_step = min(current_step, total_steps)

    if current_step >= total_steps:
        return end_val

    progress = current_step / total_steps

    if method == "linear":
        return start_val + (end_val - start_val) * progress

    elif method == "cosine":
        # Cosine annealing (Standard): Starts slow, accelerates, slows down.
        # Ideally suited for learning rates, but works for Reg too.
        # This implementation goes from start -> end.
        cosine_factor = 0.5 * (1 - np.cos(np.pi * progress)) # 0 -> 1 curve
        return start_val + (end_val - start_val) * cosine_factor

    elif method == "exponential":
        # Geometric progression (Log-Linear)
        # Formula: y = start * (end/start)^progress
        # Useful for sweeping magnitudes (e.g. 1e-4 -> 1e-1)
        if start_val <= 0 or end_val <= 0:
            # Fallback to linear if values aren't positive (log undefined)
            return start_val + (end_val - start_val) * progress

        ratio = end_val / start_val
        return start_val * (ratio ** progress)

    return start_val # Default constant


def train_treatment_pcl_net_ate_model(
    model: nn.Module,
    first_stage_train_dataloader,
    second_stage_train_dataloader,
    stage1_optimizers: List[torch.optim.Optimizer],
    stage2_optimizers: List[torch.optim.Optimizer],
    n_epochs: int = 10,
    stage1_iter: int = 1,
    stage2_iter: int = 1,
    first_stage_final_layer_regularizer: Union[float, Tuple[float, float]] = 1e-2,
    second_stage_final_layer_regularizer: Union[float, Tuple[float, float]] = 1e-2,
    second_stage_first_final_layer_regularizer: Union[float, Tuple[float, float]] = 1e-2,
    regularizer_annealing_method: str = "linear", # "linear" or "cosine"
    consider_prev_weight: bool = True,
    negative_penalty: float = 0.0,
    log_per_epoch: int = 1,
    validation_dataloader = None,
    plot_loss: bool = True,
    stage1_schedulers: Optional[List[Scheduler]] = None,
    stage2_schedulers: Optional[List[Scheduler]] = None,
    **kwargs
):
    loss_fn = nn.MSELoss()
    if validation_dataloader is None: validation_dataloader = first_stage_train_dataloader

    if plot_loss: pl.figure(figsize=(32, 12), dpi=80)
    # Check if schedulers are provided
    use_schedulers_stage1 = stage1_schedulers is not None
    use_schedulers_stage2 = stage2_schedulers is not None

    device = model.device
    stage1_loss_hist, stage1_val_loss_hist, stage2_loss_hist, stage2_val_loss_hist = [], [], [], []
    # Initial evaluation before training
    model.eval()
    stage2_loss_hist.append(model.evaluate_loss(second_stage_train_dataloader, "second"))
    stage2_val_loss_hist.append(model.evaluate_loss(validation_dataloader, "second"))
    # --- 1. Parse Regularization Schedules ---
    def parse_reg_schedule(reg_input):
        if isinstance(reg_input, (tuple, list)):
            return reg_input[0], reg_input[1]
        return reg_input, reg_input # Constant if single float provided

    reg1_start, reg1_end = parse_reg_schedule(first_stage_final_layer_regularizer)
    reg2_start, reg2_end = parse_reg_schedule(second_stage_final_layer_regularizer)
    reg12_start, reg12_end = parse_reg_schedule(second_stage_first_final_layer_regularizer)

    # --- HELPER: Infinite Iterator for the smaller dataset ---
    # This prevents 'zip' from cutting off the larger dataset
    def cycle(iterable):
        while True:
            for x in iterable: yield x

    # Identify which loader is longer to define "One Epoch"
    len_1 = len(first_stage_train_dataloader)
    len_2 = len(second_stage_train_dataloader)
    batches_per_epoch = max(len_1, len_2)

    for epoch in tqdm(range(n_epochs)):
        epoch_stage1_loss = 0
        counter = 0
        model.train()

        # --- 2. CALL ANNEALING FUNCTION ---
        curr_reg_1 = get_annealed_value_for_reg(reg1_start, reg1_end, epoch, n_epochs, method=regularizer_annealing_method)
        curr_reg_2 = get_annealed_value_for_reg(reg2_start, reg2_end, epoch, n_epochs, method=regularizer_annealing_method)
        curr_reg_12 = get_annealed_value_for_reg(reg12_start, reg12_end, epoch, n_epochs, method=regularizer_annealing_method)

        # Create Iterators (Cycle the shorter one)
        if len_1 >= len_2:
            iter_1 = iter(first_stage_train_dataloader)
            iter_2 = cycle(second_stage_train_dataloader)
        else:
            iter_1 = cycle(first_stage_train_dataloader)
            iter_2 = iter(second_stage_train_dataloader)

        for _ in range(batches_per_epoch):
            # Fetch Batches
            first_stage_data_tuple = next(iter_1)
            second_stage_data_tuple = next(iter_2)

            model.first_stage_featurizer.train(True)
            model.treatment_backdoor_featurizer.train(False)
            model.treatment_proxy_featurizer.train(False)

            for ii in range(stage1_iter):
                for opt in stage1_optimizers:
                    opt.zero_grad()

                ## First stage
                for opt in stage1_optimizers:
                    opt.zero_grad()

                first_stage_loss = model.first_stage_loss(first_stage_data_tuple, loss_fn = loss_fn)
                first_stage_loss.backward()

                for opt in stage1_optimizers:
                    opt.step()

                with torch.no_grad():
                    model.update_final_layer_with_batch_regression( first_stage_data_tuple,
                                                                    curr_reg_1, stage = "first",
                                                                    consider_prev_weight = consider_prev_weight)

            epoch_stage1_loss += first_stage_loss.item()

            model.first_stage_featurizer.train(False)
            model.treatment_backdoor_featurizer.train(True)
            model.treatment_proxy_featurizer.train(True)

            ## Second stage
            for ii in range(stage2_iter):
                for opt in stage2_optimizers:
                    opt.zero_grad()

                second_stage_loss = model.second_stage_loss(first_stage_data_tuple, second_stage_data_tuple,
                                                            loss_fn,
                                                            second_stage_first_final_layer_regularizer = curr_reg_12,
                                                            negative_penalty = negative_penalty)
                second_stage_loss.backward()

                for opt in stage2_optimizers:
                    opt.step()

                with torch.no_grad():
                    model.update_final_layer_with_batch_regression(second_stage_data_tuple,
                                                                   curr_reg_2,
                                                                   stage = "second", consider_prev_weight = consider_prev_weight)
            counter += 1
        # ==========================
        # Scheduler Step (End of Epoch)
        # ==========================
        if stage1_schedulers:
            for sched in stage1_schedulers:
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Step with validation metric if available
                    if len(stage1_val_loss_hist) > 0: sched.step(stage1_val_loss_hist[-1])
                else:
                    sched.step()

        if stage2_schedulers:
            for sched in stage2_schedulers:
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if len(stage2_val_loss_hist) > 0: sched.step(stage2_val_loss_hist[-1])
                else:
                    sched.step()

        if (epoch % log_per_epoch == 0) | (epoch == n_epochs - 1):
            model.eval()
            stage1_loss_hist.append(epoch_stage1_loss / counter)
            stage1_val_loss_hist.append(model.evaluate_loss(first_stage_train_dataloader, "first"))
            stage2_loss_hist.append(model.evaluate_loss(second_stage_train_dataloader, "second"))
            stage2_val_loss_hist.append(model.evaluate_loss(validation_dataloader, "second"))
            if plot_loss:
                val_loss_x_axis = np.arange(1, len(stage1_val_loss_hist) + 1)

                pl.clf()
                pl.subplot(1, 2, 1)
                pl.plot(val_loss_x_axis, stage1_loss_hist, linewidth=5, label = 'Training Averaged Loss')
                pl.plot(val_loss_x_axis, stage1_val_loss_hist, linewidth=5, label = 'Validation Loss')
                pl.xlabel(f"Number of Epochs / {log_per_epoch}", fontsize=20)
                pl.ylabel("MSE", fontsize=20)
                pl.title("Averaged Stage 1 Loss: {}".format(stage1_loss_hist[-1]), fontsize=20)
                pl.grid()
                pl.legend(fontsize=15)
                pl.xticks(fontsize=20)
                pl.yticks(fontsize=20)

                val_loss_x_axis = np.arange(len(stage2_val_loss_hist))
                pl.subplot(1, 2, 2)
                pl.plot(val_loss_x_axis, stage2_loss_hist, linewidth=5, label = 'Training Loss')
                pl.plot(val_loss_x_axis, stage2_val_loss_hist, linewidth=5, label = 'Validation Loss')
                pl.xlabel(f"Number of Epochs / {log_per_epoch}", fontsize=20)
                pl.ylabel(r"$\mathcal{L}_2$", fontsize=20)
                pl.title("Training Stage 2 Loss: {}\n Validation Stage 2 Loss: {}\n".format(stage2_loss_hist[-1],
                                                                                            stage2_val_loss_hist[-1]), fontsize=20)
                pl.grid()
                pl.legend(fontsize=15)
                pl.xticks(fontsize=20)
                pl.yticks(fontsize=20)
                clear_output(wait=True)
                display(pl.gcf())

    return model.eval()


def create_third_stage_dataset_for_treatment_pcl_net_ate(
    pcl_model: nn.Module,
    pcl_dataloader: DataLoader,
    pcl_val_dataloader: Optional[DataLoader] = None,
    outcome_transformer: Optional[nn.Module] = TorchIdentityTransformer(),
    input_transformer: Optional[nn.Module] = TorchIdentityTransformer(),
    treatment_featurizer = torch.nn.Identity(),
    dens_ratio_pred_tolerance: float = 1.5,
    device: str = "cuda",
) -> Tuple[TensorDataset, Optional[TensorDataset]]:
    """
    Generates a new dataset for the Third Stage regression.

    The regression problem is: Psi(A) = E[Composite_Target | A]

    Input: A
    Target: Composite_Target = Y * varphi_ZA (where varphi_ZA is the output of the final layer 2)

    Returns: TensorDataset(A_all, Composite_Target_all)
    """

    pcl_model.eval().to(device) # Ensure featurizers are frozen and in evaluation mode

    # Check if the model has a second stage final layer (required for varphi_ZA)
    if not hasattr(pcl_model, 'final_layer_second_stage_weight') or pcl_model.final_layer_second_stage_weight is None:
        return None, None # Cannot proceed without the final layer weight

    # Get the inverse transformer for Y
    y_transformer = pcl_dataloader.dataset.transformers[1].to(device)
    dens_ratio_transformer = copy.deepcopy(pcl_dataloader.dataset.dens_ratio_transformer).to(device)

    # --- Helper Function to Process Loaders (DRY) ---
    def _process_single_loader(loader, desc_text):
        phi_A_list = []
        Composite_Target_list = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc_text):
                batch = [batch[i].to(device) for i in range(len(batch))]

                # Unpack Batch
                # A, Y, Z, W, [X]
                if len(batch) > 5:
                    A_batch, Y_batch_scaled, Z_batch, W_batch, X_batch, dens_ratio_batch = batch
                    dens_ratio_batch = dens_ratio_transformer.inverse_transform(dens_ratio_batch) # Just to ensure it's on the right scale
                else:
                    A_batch, Y_batch_scaled, Z_batch, W_batch, dens_ratio_batch = batch
                    dens_ratio_batch = dens_ratio_transformer.inverse_transform(dens_ratio_batch) # Just to ensure it's on the right scale
                    X_batch = None

                # Predict Density Ratio
                dens_ratio_pred_batch = pcl_model.predict_dens_ratio(A_batch, W_batch, X_batch)

                # Identify Inliers
                inlier_indices = (
                    (torch.abs(dens_ratio_pred_batch - dens_ratio_batch) < dens_ratio_pred_tolerance) &
                    (dens_ratio_pred_batch > 0.0)
                ).view(-1)

                if inlier_indices.sum() == 0:
                    continue

                # --- Step 1: Inverse Transform Y ---
                Y_batch_unscaled = y_transformer.inverse_transform(Y_batch_scaled[inlier_indices])

                # --- Step 2: Compute Bridge Function Output (varphi_ZA) ---
                # # phi_Z_features (N, D_Z_feat)
                # phi_Z_features = pcl_model.treatment_proxy_featurizer(Z_batch[inlier_indices])

                # # phi_AX_features (N, D_backdoor_feat)
                # if X_batch is not None:
                #     second_stage_input = torch.hstack([A_batch[inlier_indices], X_batch[inlier_indices]])
                # else:
                #     second_stage_input = A_batch[inlier_indices]

                # phi_AX_features = pcl_model.treatment_backdoor_featurizer(second_stage_input)

                # # Outer Product Batch
                # second_stage_feature = outer_prod_batch(phi_AX_features, phi_Z_features).flatten(start_dim=1)

                # # Prediction Feature for final_layer_second_stage (N, D_input_2)
                # second_stage_feature = add_const_col(second_stage_feature)

                # # varphi_ZA: The predicted scalar output of the final layer (N, 1)
                # varphi_AXZ = torch.matmul(second_stage_feature, pcl_model.final_layer_second_stage_weight.T)
                varphi_AXZ = pcl_model.predict_treatment_bridge_function(A_batch[inlier_indices], Z_batch[inlier_indices], X_batch[inlier_indices] if X_batch is not None else None)
                varphi_AXZ_unscaled = dens_ratio_transformer.inverse_transform(varphi_AXZ)

                # --- Step 3: Compute Composite Target (Y * varphi_ZA) ---
                composite_target = Y_batch_unscaled * varphi_AXZ_unscaled

                # --- Step 4: Store Data ---
                # Apply the treatment featurizer (likely Identity or Kernel)
                A_stored = treatment_featurizer(A_batch)

                phi_A_list.append(A_stored[inlier_indices].cpu())
                Composite_Target_list.append(composite_target.cpu())

        if len(phi_A_list) == 0:
            return None, None

        return torch.cat(phi_A_list), torch.cat(Composite_Target_list)

    # 1. Process Training Data
    phi_A_all, Composite_Target_all = _process_single_loader(pcl_dataloader, "Creating Stage 3 Dataset")

    if phi_A_all is None:
        raise ValueError("No inliers found in training set.")

    # 2. Fit Transformers on Train
    phi_A_all = input_transformer.fit_transform(phi_A_all)
    Composite_Target_all = outcome_transformer.fit_transform(Composite_Target_all)

    third_stage_dataset = TensorDataset(phi_A_all, Composite_Target_all)
    third_stage_dataset.outcome_transformer = outcome_transformer
    third_stage_dataset.input_transformer = input_transformer

    # 3. Process Validation Data
    third_stage_dataset_val = None
    if pcl_val_dataloader is not None:
        phi_A_val, Composite_Target_val = _process_single_loader(pcl_val_dataloader, "Creating Stage 3 Val Dataset")

        if phi_A_val is not None:
            # Transform Val using Train statistics
            phi_A_val = input_transformer.transform(phi_A_val)
            Composite_Target_val = outcome_transformer.transform(Composite_Target_val)

            third_stage_dataset_val = TensorDataset(phi_A_val, Composite_Target_val)

    return third_stage_dataset, third_stage_dataset_val



def create_third_stage_dataset_for_treatment_pcl_net_cate(
    pcl_model: nn.Module,
    pcl_dataloader: DataLoader,
    pcl_val_dataloader: Optional[DataLoader] = None,
    outcome_transformer: Optional[nn.Module] = TorchIdentityTransformer(),
    input_transformer: Optional[nn.Module] = TorchIdentityTransformer(),
    dens_ratio_pred_tolerance: float = 1.5,
    device: str = "cuda",
) -> Tuple[TensorDataset, Optional[TensorDataset]]:
    """
    Generates a new dataset for the Third Stage regression.

    The regression problem is: Psi(A) = E[Composite_Target | A]

    Input: A
    Target: Composite_Target = Y * varphi_ZA (where varphi_ZA is the output of the final layer 2)

    Returns: TensorDataset(A_all, Composite_Target_all)
    """

    pcl_model.eval().to(device) # Ensure featurizers are frozen and in evaluation mode

    # Check if the model has a second stage final layer (required for varphi_ZA)
    if not hasattr(pcl_model, 'final_layer_second_stage_weight') or pcl_model.final_layer_second_stage_weight is None:
        return None, None # Cannot proceed without the final layer weight

    # Get the inverse transformer for Y
    y_transformer = pcl_dataloader.dataset.transformers[1].to(device)
    dens_ratio_transformer = copy.deepcopy(pcl_dataloader.dataset.dens_ratio_transformer).to(device)

    # --- Helper Function to Process Loaders (DRY) ---
    def _process_single_loader(loader, desc_text):
        phi_A_list = []
        Composite_Target_list = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc_text):
                batch = [batch[i].to(device) for i in range(len(batch))]

                # Unpack Batch
                # A, Y, Z, W, [X]
                if len(batch) > 6:
                    A_batch, Y_batch_scaled, Z_batch, W_batch, V_batch, X_batch, dens_ratio_batch = batch
                    dens_ratio_batch = dens_ratio_transformer.inverse_transform(dens_ratio_batch) # Just to ensure it's on the right scale
                else:
                    A_batch, Y_batch_scaled, Z_batch, W_batch, V_batch, dens_ratio_batch = batch
                    dens_ratio_batch = dens_ratio_transformer.inverse_transform(dens_ratio_batch) # Just to ensure it's on the right scale
                    X_batch = None

                # Predict Density Ratio
                dens_ratio_pred_batch = pcl_model.predict_dens_ratio(A_batch, V_batch, W_batch, X_batch)

                # Identify Inliers
                inlier_indices = (
                    (torch.abs(dens_ratio_pred_batch - dens_ratio_batch) < dens_ratio_pred_tolerance) &
                    (dens_ratio_pred_batch > 0.0)
                ).view(-1)

                if inlier_indices.sum() == 0:
                    continue

                # --- Step 1: Inverse Transform Y ---
                Y_batch_unscaled = y_transformer.inverse_transform(Y_batch_scaled[inlier_indices])

                # # varphi_ZA: The predicted scalar output of the final layer (N, 1)
                # varphi_AXZ = torch.matmul(second_stage_feature, pcl_model.final_layer_second_stage_weight.T)
                varphi_AXZ = pcl_model.predict_treatment_bridge_function(A_batch[inlier_indices], V_batch[inlier_indices], Z_batch[inlier_indices], X_batch[inlier_indices] if X_batch is not None else None)
                varphi_AXZ_unscaled = dens_ratio_transformer.inverse_transform(varphi_AXZ)

                # --- Step 3: Compute Composite Target (Y * varphi_ZA) ---
                composite_target = Y_batch_unscaled * varphi_AXZ_unscaled

                # --- Step 4: Store Data ---
                pseudo_outcome = torch.hstack([A_batch, V_batch])

                phi_A_list.append(pseudo_outcome[inlier_indices].cpu())
                Composite_Target_list.append(composite_target.cpu())

        if len(phi_A_list) == 0:
            return None, None

        return torch.cat(phi_A_list), torch.cat(Composite_Target_list)

    # 1. Process Training Data
    phi_A_all, Composite_Target_all = _process_single_loader(pcl_dataloader, "Creating Stage 3 Dataset")

    if phi_A_all is None:
        raise ValueError("No inliers found in training set.")

    # 2. Fit Transformers on Train
    phi_A_all = input_transformer.fit_transform(phi_A_all)
    Composite_Target_all = outcome_transformer.fit_transform(Composite_Target_all)

    third_stage_dataset = TensorDataset(phi_A_all, Composite_Target_all)
    third_stage_dataset.outcome_transformer = outcome_transformer
    third_stage_dataset.input_transformer = input_transformer

    # 3. Process Validation Data
    third_stage_dataset_val = None
    if pcl_val_dataloader is not None:
        phi_A_val, Composite_Target_val = _process_single_loader(pcl_val_dataloader, "Creating Stage 3 Val Dataset")

        if phi_A_val is not None:
            # Transform Val using Train statistics
            phi_A_val = input_transformer.transform(phi_A_val)
            Composite_Target_val = outcome_transformer.transform(Composite_Target_val)

            third_stage_dataset_val = TensorDataset(phi_A_val, Composite_Target_val)

    return third_stage_dataset, third_stage_dataset_val


def train_third_stage_treatment_pcl_net(
    third_stage_net: nn.Module,
    dataloader: DataLoader,
    val_dataloader: DataLoader,  # Validation is now required for this logic
    optimizer: torch.optim.Adam,
    scheduler: Optional[ExponentialLR] = None,
    loss_fn: Callable = nn.MSELoss(),
    clip_grad_norm: Optional[float] = 100.0,
    n_epochs: int = 100,
    device: str = "cuda",
    log_per_epoch: int = 10,
    gap_penalty_weight: float = 0.0,
    verbose = False,
) -> Tuple[nn.Module, List[float], List[float]]:

    third_stage_net.to(device)

    train_loss_history = []
    val_loss_history = []

    # Initialize best tracking
    best_composite_score = float('inf')
    best_model_state = copy.deepcopy(third_stage_net.state_dict())

    for epoch in tqdm(range(n_epochs), desc="Stage 3 CME Training"):
        # --- TRAINING ---
        third_stage_net.train()
        running_loss = 0.0
        counter = 0

        for batch in dataloader:
            phi_A_batch = batch[0].to(device)
            composite_target = batch[1].to(device)

            optimizer.zero_grad()
            predicted_mean = third_stage_net(phi_A_batch)
            loss = loss_fn(predicted_mean, composite_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(third_stage_net.parameters(), clip_grad_norm)
            optimizer.step()

            running_loss += loss.item()
            counter += 1

        if scheduler: scheduler.step()

        avg_train_loss = running_loss / counter if counter > 0 else 0.0
        train_loss_history.append(avg_train_loss)

        # --- VALIDATION & SELECTION ---
        if (epoch + 1) % log_per_epoch == 0 or epoch == n_epochs - 1:
            # We calculate exact losses for fair comparison
            avg_train_loss_eval = evaluate_loss_on_dataloader(third_stage_net, dataloader, loss_fn, device)
            avg_val_loss = evaluate_loss_on_dataloader(third_stage_net, val_dataloader, loss_fn, device)
            val_loss_history.append(avg_val_loss)

            # 1. Calculate Overfitting Gap (ensure non-negative)
            overfitting_gap = max(0, avg_val_loss - avg_train_loss_eval)

            # 2. Calculate Composite Score
            # Lower is better. We add the gap weighted by lambda.
            composite_score = avg_val_loss + (gap_penalty_weight * overfitting_gap)

            # 3. Selection Logic
            if composite_score < best_composite_score:
                best_composite_score = composite_score
                best_model_state = copy.deepcopy(third_stage_net.state_dict())
                if verbose:
                    print(f"Epoch {epoch+1}: New Best! Val: {avg_val_loss:.5f} | Train-Val Loss Gap: {overfitting_gap:.5f} | Score: {composite_score:.5f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}: | Train Loss: {avg_train_loss_eval:.6f} | Val Loss: {avg_val_loss:.6f} | Train-Val Loss Gap: {overfitting_gap:.5f}")

    # Restore and return the single best model
    final_model = copy.deepcopy(third_stage_net)
    final_model.load_state_dict(best_model_state)
    final_model.eval()

    return final_model, train_loss_history, val_loss_history


def train_third_stage_treatment_pcl_net_ensemble(
    model_class: Callable,         # Pass the class (e.g. ConditionalMeanEstimator), not an instance
    dataloader: DataLoader,
    val_dataloader: DataLoader,  # Validation is now required for this logic
    n_members: int = 5,
    device: str = "cuda",
    **train_kwargs                 # Pass lr, n_epochs, etc. here
) -> nn.Module:

    trained_models = []
    print(f"Training Ensemble of {n_members} models...")

    for i in range(n_members):
        print(f"--- Training Model {i+1}/{n_members} ---")

        # 1. Instantiate a FRESH model
        # Crucial: Must be a new instance to get new random initialization
        model = model_class(train_kwargs.get('input_dim', 1),
                            train_kwargs.get('output_dim', 1),
                            train_kwargs.get('hidden_dims', [32, 64]),
                            train_kwargs.get('dropout_rate', 0.05)).to(device)

        # 2. Setup Optimizer for this specific model
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_kwargs.get('lr', 1e-3),
            weight_decay=train_kwargs.get('weight_decay', 1e-6)
        )

        # 4. Train
        # We reuse your existing training function
        best_model, _, _ = train_third_stage_treatment_pcl_net(
            third_stage_net=model,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            loss_fn=train_kwargs.get('loss_fn', nn.MSELoss()),
            n_epochs=train_kwargs.get('n_epochs', 100),
            gap_penalty_weight=train_kwargs.get('gap_penalty_weight', 0.0),
            log_per_epoch=train_kwargs.get('log_per_epoch', 10),
            device=device
        )

        trained_models.append(best_model)

    print("Ensemble Training Complete.")
    return EnsembleConditionalMeanMLP(trained_models).eval()