from xml.parsers.expat import model

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Optional, Tuple, Callable, Union
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler # Use abstract base class for type hint

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

import sys
sys.path.append("..")
from torch_utils.scalers import TorchIdentityTransformer
from torch_utils.linear_algebra import fit_linear_proximal, outer_prod, outer_prod_batch, add_const_col
from torch_utils.model_helpers import get_last_linear_out_features


class OutcomeBridgePCLNET(nn.Module):

    def __init__(self,
                 first_stage_featurizer: nn.Module,
                 treatment_featurizer: nn.Module,
                 outcome_proxy_featurizer: nn.Module,
                 backdoor_featurizer: Optional[nn.Module] = None,
                 final_layer_first_stage_weight: Optional[torch.Tensor] = None,
                 final_layer_second_stage_weight: Optional[torch.Tensor] = None,
                 outcome_dim: int = 1,
                 device: str = "cuda",
                 **kwargs,):
        super().__init__()
        self.first_stage_featurizer = first_stage_featurizer.to(device)
        self.treatment_featurizer = treatment_featurizer.to(device)
        self.outcome_proxy_featurizer = outcome_proxy_featurizer.to(device)
        self.backdoor_featurizer = backdoor_featurizer.to(device) if backdoor_featurizer is not None else None

        outcome_proxy_feature_dim = get_last_linear_out_features(outcome_proxy_featurizer)

        def init_first_stage_final_layer_weight(final_layer_first_stage_weight_val):
            if final_layer_first_stage_weight_val is None:
                first_stage_in_dim = get_last_linear_out_features(first_stage_featurizer)
                final_layer_first_stage_weight_val = torch.zeros(outcome_proxy_feature_dim,
                                                                 first_stage_in_dim + 1,
                                                                 device=device)
                final_layer_first_stage_weight_val.data[:, -1] = 1.0 / (first_stage_in_dim)
            return final_layer_first_stage_weight_val

        final_layer_first_stage_weight_val = init_first_stage_final_layer_weight(final_layer_first_stage_weight)
        final_layer_first_stage_inside_second_stage_weight_val = init_first_stage_final_layer_weight(final_layer_first_stage_weight)

        if final_layer_second_stage_weight is None:
            if backdoor_featurizer is not None:
                backdoor_featurizer_dim = get_last_linear_out_features(backdoor_featurizer)
            else:
                backdoor_featurizer_dim = 1
            treatment_featurizer_dim = get_last_linear_out_features(treatment_featurizer)
            final_layer_second_stage_weight = torch.zeros(outcome_dim,
                                                          outcome_proxy_feature_dim * treatment_featurizer_dim * backdoor_featurizer_dim + 1,
                                                          device=device)
            final_layer_second_stage_weight.data[:, -1] = 0.0

        # Registering the final linear weights! This ensures they are included in model.state_dict
        self.register_buffer("final_layer_first_stage_weight", final_layer_first_stage_weight_val)
        self.register_buffer("final_layer_first_stage_inside_second_stage_weight", final_layer_first_stage_inside_second_stage_weight_val)
        self.register_buffer("final_layer_second_stage_weight", final_layer_second_stage_weight)
        self.final_layer_first_stage_weight.requires_grad = False
        self.final_layer_second_stage_weight.requires_grad = False
        self.final_layer_first_stage_inside_second_stage_weight.requires_grad = False
        self.device = device
        # self.register_buffer("mean_outcome_proxy_feature", None)
        self.mean_outcome_proxy_feature = None

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

        if self.backdoor_featurizer is not None:
            # Case: With X (High Dim / Confounder)
            # A, Y, Z, W, X
            treatment, Y, treatment_proxy, outcome_proxy, backdoor = batch[:5]
        else:
            # Case: No X
            # A, Y, Z, W
            treatment, Y, treatment_proxy, outcome_proxy = batch[:4]
            backdoor = None

        return treatment, Y, treatment_proxy, outcome_proxy, backdoor

    def _compute_first_stage_features(self, treatment, treatment_proxy, backdoor=None):
        """
        Computes phi_1(A, W, X).
        """
        if backdoor is not None:
            # Single network takes concatenation
            inp = torch.hstack([treatment, backdoor, treatment_proxy])
        else:
            inp = torch.hstack([treatment, treatment_proxy])

        feat = self.first_stage_featurizer(inp)
        return add_const_col(feat)

    def _compute_second_stage_features(self, treatment, treatment_proxy, backdoor, w1_weight_matrix):
        """
        Computes the outer product features for the second stage.
        Psi_2(A, X) (outer) E[phi_2(W) | A, Z, X]
        """
        with torch.no_grad():
            # 1. Get E[phi(Z) | A, W, X] using Stage 1 weights
            feat_stage1 = self._compute_first_stage_features(treatment, treatment_proxy, backdoor)

        pred_cond_o_proxy_feat = torch.matmul(feat_stage1, w1_weight_matrix)
        # 2. Get Psi(A, X)
        treatment_feature = self.treatment_featurizer(treatment)
        if backdoor is not None and self.backdoor_featurizer is not None:
            backdoor_feature = self.backdoor_featurizer(backdoor)
            second_stage_feature = outer_prod_batch(
                treatment_feature,
                backdoor_feature,
                pred_cond_o_proxy_feat
            ).flatten(start_dim = 1)
        else:
            second_stage_feature = outer_prod(
                treatment_feature,
                pred_cond_o_proxy_feat
            ).flatten(start_dim = 1)
        second_stage_feature = add_const_col(second_stage_feature)
        return second_stage_feature

    def forward(self, treatment, treatment_proxy, backdoor: Optional = None):
        second_stage_feature = self._compute_second_stage_features(treatment, treatment_proxy, backdoor, self.final_layer_first_stage_weight.T)
        pred = torch.matmul(second_stage_feature, self.final_layer_second_stage_weight.T)
        return pred

    def first_stage_forward(self, treatment, treatment_proxy, backdoor: Optional = None):
        first_stage_feature = self._compute_first_stage_features(treatment, treatment_proxy, backdoor)
        predicted_conditional_outcome_proxy_feature = torch.matmul(first_stage_feature, self.final_layer_first_stage_weight.T)
        return predicted_conditional_outcome_proxy_feature

    def first_stage_loss(self, first_stage_data_tuple, loss_fn = None):
        if loss_fn is None: loss_fn = nn.MSELoss()
        ##############################################################################
        # Extract data from the input tuple.
        ##############################################################################
        treatment, _, treatment_proxy, outcome_proxy, backdoor = self._unpack_batch(first_stage_data_tuple)
        with torch.no_grad():
            outcome_proxy_feature = self.outcome_proxy_featurizer(outcome_proxy)
        predicted_conditional_outcome_proxy_feature = self.first_stage_forward(treatment,
                                                                               treatment_proxy,
                                                                               backdoor)
        loss = loss_fn(predicted_conditional_outcome_proxy_feature, outcome_proxy_feature)
        return loss

    def second_stage_loss(self, first_stage_tuple, second_stage_tuple,
                          loss_fn = None,
                          second_stage_first_final_layer_regularizer = 1e-1,
                          ):
        if loss_fn is None: loss_fn = nn.MSELoss()
        ##############################################################################
        # Extract first stage data from the input tuple.
        ##############################################################################
        treatment, Y, treatment_proxy, outcome_proxy, backdoor = self._unpack_batch(first_stage_tuple)

        with torch.no_grad():
            ##############################################################################
            # Construct First Stage Feature.
            ##############################################################################
            first_stage_feature = self._compute_first_stage_features(treatment, treatment_proxy, backdoor)
            weight_previous = self.final_layer_first_stage_inside_second_stage_weight.detach()

        outcome_proxy_feature = self.outcome_proxy_featurizer(outcome_proxy)
        new_weight = fit_linear_proximal(outcome_proxy_feature, first_stage_feature, weight_previous.T,
                                         second_stage_first_final_layer_regularizer)

        ##############################################################################
        # Extract second stage data from the input tuple.
        ##############################################################################
        treatment, Y, treatment_proxy, outcome_proxy, backdoor = self._unpack_batch(second_stage_tuple)
        second_stage_feature = self._compute_second_stage_features(treatment, treatment_proxy, backdoor, new_weight)

        Y_preds = torch.matmul(second_stage_feature, self.final_layer_second_stage_weight.T)
        loss = loss_fn(Y_preds, Y)
        with torch.no_grad():
            self.final_layer_first_stage_inside_second_stage_weight = new_weight.T
        return loss

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
        treatment, _, treatment_proxy, outcome_proxy, backdoor = self._unpack_batch(data_tuple)
        ##############################################################################
        # Construct First Stage Feature.
        ##############################################################################
        first_stage_feature = self._compute_first_stage_features(treatment, treatment_proxy, backdoor)

        ##############################################################################
        # Construct output for regression (AKA y).
        ##############################################################################
        outcome_proxy_feature = self.outcome_proxy_featurizer(outcome_proxy)

        ##############################################################################
        # Get previous weights w_{t}
        ##############################################################################
        weight_previous = self.final_layer_first_stage_weight.detach()

        ##############################################################################
        # Do the weight update w_{t+1} = REG_SOLUTION(w_t, x,y)
        ##############################################################################
        new_weight = fit_linear_proximal(
            # output (y)
            outcome_proxy_feature,
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
        treatment, Y, treatment_proxy, _, backdoor = self._unpack_batch(data_tuple)

        ##############################################################################
        # Compute SECOND stage features
        ##############################################################################
        # INCREDIBLY IMPORTANT
        FIRST_STAGE_INSIDE_SECOND_STAGE_WEIGHT = self.final_layer_first_stage_inside_second_stage_weight.T
        second_stage_feature = self._compute_second_stage_features(treatment, treatment_proxy, backdoor, FIRST_STAGE_INSIDE_SECOND_STAGE_WEIGHT)
        # extract the previous last layer
        weight_previous = self.final_layer_second_stage_weight .detach()
        weight = fit_linear_proximal(Y, second_stage_feature, int(consider_prev_weight) * (weight_previous.T), weight_regularizer, "ridge")
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

    def predict_outcome_bridge_function(self, treatment, outcome_proxy, backdoor: Optional = None):
        training_state = self.training
        self.eval()
        treatment_feature = self.treatment_featurizer(treatment.to(self.device))
        outcome_proxy_feature = self.outcome_proxy_featurizer(outcome_proxy.to(self.device))
        if (backdoor is not None) & (self.backdoor_featurizer is not None):
            backdoor_feature = self.backdoor_featurizer(backdoor.to(self.device))
            feature = outer_prod_batch(
                treatment_feature,
                backdoor_feature,
                outcome_proxy_feature,
            ).flatten(start_dim=1)
        else:
            feature = outer_prod(
                treatment_feature,
                outcome_proxy_feature,
            ).flatten(start_dim=1)
        feature = add_const_col(feature)
        pred = torch.matmul(feature, self.final_layer_second_stage_weight.T)
        # --- Restore original training state ---
        if training_state:
            self.train()
        return pred

    def pred_structural_function(self,
                                 treatment,
                                 treatment_transformer: nn.Module = TorchIdentityTransformer(),
                                 outcome_transformer: nn.Module = TorchIdentityTransformer()
        ):
        training_state = self.training
        self.eval()
        treatment_transformer = treatment_transformer.to(self.device)
        outcome_transformer = outcome_transformer.to(self.device)
        treatment_transformed = treatment_transformer.transform(treatment.to(self.device))
        treatment_feature = self.treatment_featurizer(treatment_transformed)
        n_data = treatment_feature.shape[0]
        mean_outcome_proxy_mat = self.mean_outcome_proxy_feature.expand(n_data, -1)
        feature = outer_prod_batch(
                            treatment_feature,
                            mean_outcome_proxy_mat,
                        ).flatten(start_dim=1)
        feature = add_const_col(feature)
        f_struct_pred_transformed = torch.matmul(feature, self.final_layer_second_stage_weight.T)
        f_struct_pred = outcome_transformer.inverse_transform(f_struct_pred_transformed)
        # --- Restore original training state ---
        if training_state:
            self.train()
        return f_struct_pred

    def pred_conditional_structural_function(
        self,
        treatment: torch.Tensor,
        anchor_treatment: torch.Tensor,
        third_stage_net: nn.Module,
        treatment_transformer: nn.Module = TorchIdentityTransformer(),
        outcome_transformer: nn.Module = TorchIdentityTransformer(),
        use_raw_treatment_for_third_stage: bool = True,
    ):
        """
        Predict the ATT curve:
            f_ATT(a, a') ≈ h_head^T [ phi_A(a) ⊗ E[c | A=a'] ]

        where
            c = phi_W(W)                    if no backdoor featurizer is used
            c = phi_X(X) ⊗ phi_W(W)         if backdoor_featurizer is used

        The third_stage_net is trained to map either:
            - raw transformed treatment A   (recommended), or
            - phi_A(A)
        to the conditional embedding E[c | A=a'].
        """
        if third_stage_net is None:
            raise ValueError("ATT estimation requires a trained third-stage network.")

        training_state = self.training
        third_stage_training_state = third_stage_net.training
        self.eval()
        third_stage_net.eval()

        treatment_transformer = treatment_transformer.to(self.device)
        outcome_transformer = outcome_transformer.to(self.device)

        with torch.no_grad():
            treatment = treatment.to(self.device)
            anchor_treatment = anchor_treatment.to(self.device)

            if treatment.ndim == 1:
                treatment = treatment.unsqueeze(0)
            if anchor_treatment.ndim == 1:
                anchor_treatment = anchor_treatment.unsqueeze(0)

            treatment_transformed = treatment_transformer.transform(treatment)
            anchor_treatment_transformed = treatment_transformer.transform(anchor_treatment)

            # phi_A(a) is still used in the final ATT formula
            phi_A = self.treatment_featurizer(treatment_transformed)

            # Third-stage input is now raw transformed A by default
            if use_raw_treatment_for_third_stage:
                third_stage_input = anchor_treatment_transformed
            else:
                third_stage_input = self.treatment_featurizer(anchor_treatment_transformed)

            conditional_embedding = third_stage_net(third_stage_input)

            if conditional_embedding.ndim == 1:
                conditional_embedding = conditional_embedding.unsqueeze(0)

            if conditional_embedding.shape[0] == 1 and phi_A.shape[0] > 1:
                conditional_embedding = conditional_embedding.expand(phi_A.shape[0], -1)
            elif conditional_embedding.shape[0] != phi_A.shape[0]:
                raise ValueError(
                    "Mismatch between number of intervention points and conditional embedding rows. "
                    f"Got phi_A.shape[0]={phi_A.shape[0]} and conditional_embedding.shape[0]={conditional_embedding.shape[0]}."
                )

            feature = outer_prod_batch(phi_A, conditional_embedding).flatten(start_dim=1)
            feature = add_const_col(feature)

            f_att_pred_transformed = torch.matmul(feature, self.final_layer_second_stage_weight.T)
            f_att_pred = outcome_transformer.inverse_transform(f_att_pred_transformed)

        if training_state:
            self.train()
        if third_stage_training_state:
            third_stage_net.train()

        return f_att_pred

    def compute_mean_outcome_proxy_feature(self, train_dataloader):
        training_state = self.training
        self.eval()
        with torch.no_grad():
            outcome_proxy_feature_list = []
            for batch in train_dataloader:
                outcome_proxy_feature = self.outcome_proxy_featurizer(batch[3].to(self.device))
                if self.backdoor_featurizer is not None:
                    backdoor_feature = self.backdoor_featurizer(batch[4].to(self.device))
                    outcome_proxy_feature_list.append(
                        outer_prod(backdoor_feature, outcome_proxy_feature).flatten(start_dim = 1)
                    )
                else:
                    outcome_proxy_feature_list.append(
                        outcome_proxy_feature
                    )

            outcome_proxy_feature = torch.vstack(outcome_proxy_feature_list)
            self.mean_outcome_proxy_feature = torch.mean(outcome_proxy_feature, dim=0)
        # --- Restore original training state ---
        if training_state:
            self.train()

    def evaluate_second_stage_loss(self, data_loader: DataLoader, loss_fn = nn.MSELoss()):
        loss = 0.0
        self.eval()
        with torch.no_grad():
            for data_tuple in data_loader:
                data_tuple = [data_tuple[i].to(self.device) for i in range(len(data_tuple))]
                batch_size = data_tuple[0].shape[0]
                predicted_outcome = self.forward(data_tuple[0],
                                                 data_tuple[2],
                                                 backdoor = data_tuple[4] if (self.backdoor_featurizer is not None) else None)
                batch_loss = loss_fn(predicted_outcome, data_tuple[1])
                loss += batch_loss.item() * batch_size
        loss /= len(data_loader.dataset)
        return loss

    def freeze_featurizer_params(self, stage: str):
        if stage == "first_stage":
            self.first_stage_featurizer.train(False)

        elif stage == "second_stage":
            self.treatment_featurizer.train(False)
            self.outcome_proxy_featurizer.train(False)
            if (self.backdoor_featurizer is not None):
                self.backdoor_featurizer.train(False)

    def unfreeze_featurizer_params(self, stage: str):
        if stage == "first_stage":
            self.first_stage_featurizer.train(True)
        elif stage == "second_stage":
            self.treatment_featurizer.train(True)
            self.outcome_proxy_featurizer.train(True)
            if (self.backdoor_featurizer is not None):
                self.backdoor_featurizer.train(True)


class HeterogeneousOutcomeBridgePCLNET(nn.Module):

    def __init__(self,
                 first_stage_featurizer: nn.Module,
                 treatment_featurizer: nn.Module,
                 covariate_featurizer: nn.Module,
                 outcome_proxy_featurizer: nn.Module,
                 backdoor_featurizer: Optional[nn.Module] = None,
                 final_layer_first_stage_weight: Optional[torch.Tensor] = None,
                 final_layer_second_stage_weight: Optional[torch.Tensor] = None,
                 outcome_dim: int = 1,
                 device: str = "cuda",
                 **kwargs,):
        super().__init__()
        self.first_stage_featurizer = first_stage_featurizer.to(device)
        self.treatment_featurizer = treatment_featurizer.to(device)
        self.covariate_featurizer = covariate_featurizer.to(device)
        self.outcome_proxy_featurizer = outcome_proxy_featurizer.to(device)
        self.backdoor_featurizer = backdoor_featurizer.to(device) if backdoor_featurizer is not None else None

        outcome_proxy_feature_dim = get_last_linear_out_features(outcome_proxy_featurizer)

        def init_first_stage_final_layer_weight(final_layer_first_stage_weight_val):
            if final_layer_first_stage_weight_val is None:
                first_stage_in_dim = get_last_linear_out_features(first_stage_featurizer)
                final_layer_first_stage_weight_val = torch.zeros(outcome_proxy_feature_dim,
                                                                 first_stage_in_dim + 1,
                                                                 device=device)
                final_layer_first_stage_weight_val.data[:, -1] = 1.0 / (first_stage_in_dim)
            return final_layer_first_stage_weight_val

        final_layer_first_stage_weight_val = init_first_stage_final_layer_weight(final_layer_first_stage_weight)
        final_layer_first_stage_inside_second_stage_weight_val = init_first_stage_final_layer_weight(final_layer_first_stage_weight)

        if final_layer_second_stage_weight is None:
            if backdoor_featurizer is not None:
                backdoor_featurizer_dim = get_last_linear_out_features(backdoor_featurizer)
            else:
                backdoor_featurizer_dim = 1
            treatment_featurizer_dim = get_last_linear_out_features(treatment_featurizer)
            covariate_featurizer_dim = get_last_linear_out_features(covariate_featurizer)
            final_layer_second_stage_out_dim = outcome_proxy_feature_dim * treatment_featurizer_dim * covariate_featurizer_dim * backdoor_featurizer_dim + 1
            final_layer_second_stage_weight = torch.zeros(outcome_dim,
                                                          final_layer_second_stage_out_dim,
                                                          device=device)
            final_layer_second_stage_weight.data[:, -1] = 0.0

        # Registering the final linear weights! This ensures they are included in model.state_dict
        self.register_buffer("final_layer_first_stage_weight", final_layer_first_stage_weight_val)
        self.register_buffer("final_layer_first_stage_inside_second_stage_weight", final_layer_first_stage_inside_second_stage_weight_val)
        self.register_buffer("final_layer_second_stage_weight", final_layer_second_stage_weight)
        self.final_layer_first_stage_weight.requires_grad = False
        self.final_layer_second_stage_weight.requires_grad = False
        self.final_layer_first_stage_inside_second_stage_weight.requires_grad = False
        self.device = device
        # self.register_buffer("mean_outcome_proxy_feature", None)
        self.mean_outcome_proxy_feature = None

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

        if self.backdoor_featurizer is not None:
            # Case: With X (High Dim / Confounder)
            # A, Y, Z, W, X
            treatment, Y, treatment_proxy, outcome_proxy, covariate, backdoor = batch[:6]
        else:
            # Case: No X
            # A, Y, Z, W
            treatment, Y, treatment_proxy, outcome_proxy, covariate = batch[:5]
            backdoor = None

        return treatment, Y, treatment_proxy, outcome_proxy, covariate, backdoor

    def _compute_first_stage_features(self, treatment, covariate, treatment_proxy, backdoor=None):
        """
        Computes phi_1(A, W, X).
        """
        if backdoor is not None:
            # Single network takes concatenation
            inp = torch.hstack([treatment, covariate, backdoor, treatment_proxy])
        else:
            inp = torch.hstack([treatment, covariate, treatment_proxy])

        feat = self.first_stage_featurizer(inp)
        return add_const_col(feat)

    def _compute_second_stage_features(self, treatment, covariate, treatment_proxy, backdoor, w1_weight_matrix):
        """
        Computes the outer product features for the second stage.
        Psi_2(A, X) (outer) E[phi_2(W) | A, Z, X]
        """
        with torch.no_grad():
            # 1. Get E[phi(Z) | A, W, X] using Stage 1 weights
            feat_stage1 = self._compute_first_stage_features(treatment, covariate, treatment_proxy, backdoor)

        pred_cond_o_proxy_feat = torch.matmul(feat_stage1, w1_weight_matrix)
        # 2. Get Psi(A, V, X)
        treatment_feature = self.treatment_featurizer(treatment)
        covariate_feature = self.covariate_featurizer(covariate)
        if backdoor is not None and self.backdoor_featurizer is not None:
            backdoor_feature = self.backdoor_featurizer(backdoor)
            second_stage_feature = outer_prod_batch(
                treatment_feature,
                covariate_feature,
                backdoor_feature,
                pred_cond_o_proxy_feat
            ).flatten(start_dim = 1)
        else:
            second_stage_feature = outer_prod_batch(
                treatment_feature,
                covariate_feature,
                pred_cond_o_proxy_feat
            ).flatten(start_dim = 1)
        second_stage_feature = add_const_col(second_stage_feature)
        return second_stage_feature

    def forward(self, treatment, treatment_proxy, covariate, backdoor: Optional = None):
        second_stage_feature = self._compute_second_stage_features(treatment, covariate, treatment_proxy, backdoor, self.final_layer_first_stage_weight.T)
        pred = torch.matmul(second_stage_feature, self.final_layer_second_stage_weight.T)
        return pred

    def first_stage_forward(self, treatment, covariate, treatment_proxy, backdoor: Optional = None):
        first_stage_feature = self._compute_first_stage_features(treatment, covariate, treatment_proxy, backdoor)
        predicted_conditional_outcome_proxy_feature = torch.matmul(first_stage_feature, self.final_layer_first_stage_weight.T)
        return predicted_conditional_outcome_proxy_feature

    def first_stage_loss(self, first_stage_data_tuple, loss_fn = None):
        if loss_fn is None: loss_fn = nn.MSELoss()
        ##############################################################################
        # Extract data from the input tuple.
        ##############################################################################
        treatment, _, treatment_proxy, outcome_proxy, covariate, backdoor = self._unpack_batch(first_stage_data_tuple)
        with torch.no_grad():
            outcome_proxy_feature = self.outcome_proxy_featurizer(outcome_proxy)
        predicted_conditional_outcome_proxy_feature = self.first_stage_forward(treatment,
                                                                               covariate,
                                                                               treatment_proxy,
                                                                               backdoor)
        loss = loss_fn(predicted_conditional_outcome_proxy_feature, outcome_proxy_feature)
        return loss

    def second_stage_loss(self, first_stage_tuple, second_stage_tuple,
                          loss_fn = None,
                          second_stage_first_final_layer_regularizer = 1e-1,
                          ):
        if loss_fn is None: loss_fn = nn.MSELoss()
        ##############################################################################
        # Extract first stage data from the input tuple.
        ##############################################################################
        treatment, Y, treatment_proxy, outcome_proxy, covariate,backdoor = self._unpack_batch(first_stage_tuple)

        with torch.no_grad():
            ##############################################################################
            # Construct First Stage Feature.
            ##############################################################################
            first_stage_feature = self._compute_first_stage_features(treatment, covariate, treatment_proxy, backdoor)
            weight_previous = self.final_layer_first_stage_inside_second_stage_weight.detach()

        outcome_proxy_feature = self.outcome_proxy_featurizer(outcome_proxy)
        new_weight = fit_linear_proximal(outcome_proxy_feature, first_stage_feature, weight_previous.T,
                                         second_stage_first_final_layer_regularizer)

        ##############################################################################
        # Extract second stage data from the input tuple.
        ##############################################################################
        treatment, Y, treatment_proxy, outcome_proxy, covariate, backdoor = self._unpack_batch(second_stage_tuple)
        second_stage_feature = self._compute_second_stage_features(treatment, covariate, treatment_proxy, backdoor, new_weight)

        Y_preds = torch.matmul(second_stage_feature, self.final_layer_second_stage_weight.T)
        loss = loss_fn(Y_preds, Y)
        with torch.no_grad():
            self.final_layer_first_stage_inside_second_stage_weight = new_weight.T
        return loss

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
        treatment, _, treatment_proxy, outcome_proxy, covariate, backdoor = self._unpack_batch(data_tuple)
        ##############################################################################
        # Construct First Stage Feature.
        ##############################################################################
        first_stage_feature = self._compute_first_stage_features(treatment, covariate, treatment_proxy, backdoor)

        ##############################################################################
        # Construct output for regression (AKA y).
        ##############################################################################
        outcome_proxy_feature = self.outcome_proxy_featurizer(outcome_proxy)

        ##############################################################################
        # Get previous weights w_{t}
        ##############################################################################
        weight_previous = self.final_layer_first_stage_weight.detach()

        ##############################################################################
        # Do the weight update w_{t+1} = REG_SOLUTION(w_t, x,y)
        ##############################################################################
        new_weight = fit_linear_proximal(
            # output (y)
            outcome_proxy_feature,
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
        treatment, Y, treatment_proxy, _, covariate, backdoor = self._unpack_batch(data_tuple)

        ##############################################################################
        # Compute SECOND stage features
        ##############################################################################
        # INCREDIBLY IMPORTANT
        FIRST_STAGE_INSIDE_SECOND_STAGE_WEIGHT = self.final_layer_first_stage_inside_second_stage_weight.T
        second_stage_feature = self._compute_second_stage_features(treatment, covariate, treatment_proxy, backdoor, FIRST_STAGE_INSIDE_SECOND_STAGE_WEIGHT)
        # extract the previous last layer
        weight_previous = self.final_layer_second_stage_weight .detach()
        weight = fit_linear_proximal(Y, second_stage_feature, int(consider_prev_weight) * (weight_previous.T), weight_regularizer, "ridge")
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

    def predict_outcome_bridge_function(self, treatment, covariate, outcome_proxy, backdoor: Optional = None):
        training_state = self.training
        self.eval()
        treatment_feature = self.treatment_featurizer(treatment.to(self.device))
        covariate_feature = self.covariate_featurizer(covariate.to(self.device))
        outcome_proxy_feature = self.outcome_proxy_featurizer(outcome_proxy.to(self.device))
        if (backdoor is not None) & (self.backdoor_featurizer is not None):
            backdoor_feature = self.backdoor_featurizer(backdoor.to(self.device))
            feature = outer_prod_batch(
                treatment_feature,
                covariate_feature,
                backdoor_feature,
                outcome_proxy_feature,
            ).flatten(start_dim=1)
        else:
            feature = outer_prod_batch(
                treatment_feature,
                covariate_feature,
                outcome_proxy_feature,
            ).flatten(start_dim=1)
        feature = add_const_col(feature)
        pred = torch.matmul(feature, self.final_layer_second_stage_weight.T)
        # --- Restore original training state ---
        if training_state:
            self.train()
        return pred

    def pred_structural_function(self,
                                 treatment,
                                 covariate,
                                 third_stage_net: Optional[nn.Module] = None,
                                 treatment_transformer: nn.Module = TorchIdentityTransformer(),
                                 covariate_transformer: nn.Module = TorchIdentityTransformer(),
                                 outcome_transformer: nn.Module = TorchIdentityTransformer()
        ):
        if third_stage_net is None:
            print("Heterogeneous dose-response curve estimation requires training a third stage network!")
            return torch.zeros(treatment.shape)
        training_state = self.training
        self.eval()
        third_stage_net.eval()
        treatment_normalized = treatment_transformer.transform(treatment.to(self.device))
        covariate_transformed = covariate_transformer.transform(covariate.to(self.device))
        phi_V = self.covariate_featurizer(covariate_transformed)
        phi_A = self.treatment_featurizer(treatment_normalized)
        CME_WX = third_stage_net(phi_V)
        conditional_feature = add_const_col(outer_prod_batch(phi_A, phi_V, CME_WX).flatten(start_dim = 1))
        f_struct_pred = torch.matmul(conditional_feature, self.final_layer_second_stage_weight.T)
        f_struct_pred = outcome_transformer.inverse_transform(f_struct_pred)
        if training_state:
            self.train()
        return f_struct_pred

    def compute_mean_outcome_proxy_feature(self, train_dataloader):
        pass

    def evaluate_second_stage_loss(self, data_loader: DataLoader, loss_fn = nn.MSELoss()):
        loss = 0.0
        self.eval()
        with torch.no_grad():
            for data_tuple in data_loader:
                data_tuple = [data_tuple[i].to(self.device) for i in range(len(data_tuple))]
                batch_size = data_tuple[0].shape[0]
                predicted_outcome = self.forward(data_tuple[0],
                                                 data_tuple[2],
                                                 data_tuple[4],
                                                 backdoor = data_tuple[5] if (self.backdoor_featurizer is not None) else None)
                batch_loss = loss_fn(predicted_outcome, data_tuple[1])
                loss += batch_loss.item() * batch_size
        loss /= len(data_loader.dataset)
        return loss

    def freeze_featurizer_params(self, stage: str):
        if stage == "first_stage":
            self.first_stage_featurizer.train(False)

        elif stage == "second_stage":
            self.treatment_featurizer.train(False)
            self.covariate_featurizer.train(False)
            self.outcome_proxy_featurizer.train(False)
            if (self.backdoor_featurizer is not None):
                self.backdoor_featurizer.train(False)

    def unfreeze_featurizer_params(self, stage: str):
        if stage == "first_stage":
            self.first_stage_featurizer.train(True)
        elif stage == "second_stage":
            self.treatment_featurizer.train(True)
            self.covariate_featurizer.train(True)
            self.outcome_proxy_featurizer.train(True)
            if (self.backdoor_featurizer is not None):
                self.backdoor_featurizer.train(True)


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


def train_deep_feature_proxy_closed_form_ate_model(
        pcl_model: nn.Module,
        first_stage_train_dataloader,
        second_stage_train_dataloader,
        stage1_optimizers,
        stage2_optimizers,
        stage1_schedulers: Optional[List[Scheduler]] = None,
        stage2_schedulers: Optional[List[Scheduler]] = None,
        first_stage_final_layer_regularizer: Union[float, Tuple[float, float]] = 1e-2,
        second_stage_final_layer_regularizer: Union[float, Tuple[float, float]] = 1e-2,
        second_stage_first_final_layer_regularizer: Union[float, Tuple[float, float]] = 1e-2,
        regularizer_annealing_method: str = "exponential", # "linear" or "cosine"
        consider_prev_weight: bool = True,
        n_epochs = 1,
        stage1_iter: int = 1,
        stage2_iter: int = 1,
        validation_dataloader: Optional[DataLoader] = None,
        log_per_epoch: int = 25,
        **kwargs,
        ):

    loss_fn = nn.MSELoss()
    if validation_dataloader is None:
        validation_dataloader = first_stage_train_dataloader

    plot_loss = kwargs.pop('plot_loss', False)
    do_A = kwargs.pop('do_A', None) # These are only for debugging purposes in the case that interventional curves are available!
    EY_do_A = kwargs.pop('EY_do_A', None)

    # Check if we have ground truth and if the model is the standard (non-heterogeneous) version
    is_standard_outcome_model = isinstance(pcl_model, OutcomeBridgePCLNET)
    if plot_loss:
        plt.figure(figsize=(18, 14) if ((do_A is not None) and (EY_do_A is not None) and is_standard_outcome_model) else (12, 8))

    # --- Helper: Infinite Iterator ---
    def cycle(iterable):
        while True:
            for x in iterable: yield x

    len_1 = len(first_stage_train_dataloader)
    len_2 = len(second_stage_train_dataloader)
    batches_per_epoch = max(len_1, len_2)

    device = pcl_model.device

    stage1_loss_hist = []
    stage2_loss_hist = []
    stage2_val_loss_hist = []

    # --- 1. Parse Regularization Schedules ---
    def parse_reg_schedule(reg_input):
        if isinstance(reg_input, (tuple, list)):
            return reg_input[0], reg_input[1]
        return reg_input, reg_input # Constant if single float provided

    reg1_start, reg1_end = parse_reg_schedule(first_stage_final_layer_regularizer)
    reg2_start, reg2_end = parse_reg_schedule(second_stage_final_layer_regularizer)
    reg12_start, reg12_end = parse_reg_schedule(second_stage_first_final_layer_regularizer)

    if (do_A is not None) and (EY_do_A is not None) and is_standard_outcome_model: causal_mse_list, mu_W_norm_hist = [], []

    for epoch in tqdm(range(n_epochs)):
        epoch_stage1_loss = 0
        counter = 0
        pcl_model.train()

        # --- 2. CALL ANNEALING FUNCTION ---
        curr_reg_1 = get_annealed_value_for_reg(reg1_start, reg1_end, epoch, n_epochs, method=regularizer_annealing_method)
        curr_reg_2 = get_annealed_value_for_reg(reg2_start, reg2_end, epoch, n_epochs, method=regularizer_annealing_method)
        curr_reg_12 = get_annealed_value_for_reg(reg12_start, reg12_end, epoch, n_epochs, method=regularizer_annealing_method)

        # Iterators
        if len_1 >= len_2:
            iter_1 = iter(first_stage_train_dataloader)
            iter_2 = cycle(second_stage_train_dataloader)
        else:
            iter_1 = cycle(first_stage_train_dataloader)
            iter_2 = iter(second_stage_train_dataloader)

        for _ in range(batches_per_epoch):
            first_stage_data_tuple = next(iter_1)
            second_stage_data_tuple = next(iter_2)

            pcl_model.freeze_featurizer_params("second_stage")
            pcl_model.unfreeze_featurizer_params("first_stage")

            for ii in range(stage1_iter):
                for opt in stage1_optimizers:
                    opt.zero_grad()
                loss_stage1 = pcl_model.first_stage_loss(first_stage_data_tuple,
                                                         loss_fn = loss_fn)
                loss_stage1.backward()
                for opt in stage1_optimizers:
                    opt.step()
                with torch.no_grad():
                    pcl_model.update_final_layer_with_batch_regression(first_stage_data_tuple, curr_reg_1,
                                                                        "first", consider_prev_weight)
            epoch_stage1_loss += loss_stage1.item()
            # Stage 2 update
            pcl_model.freeze_featurizer_params("first_stage")
            pcl_model.unfreeze_featurizer_params("second_stage")

            for i in range(stage2_iter):
                for opt in stage2_optimizers:
                    opt.zero_grad()

                loss_stage2 = pcl_model.second_stage_loss(first_stage_data_tuple,
                                                          second_stage_data_tuple,
                                                          loss_fn = loss_fn,
                                                          second_stage_first_final_layer_regularizer = curr_reg_12)
                loss_stage2.backward()
                for opt in stage2_optimizers:
                    opt.step()
                with torch.no_grad():
                    pcl_model.update_final_layer_with_batch_regression(second_stage_data_tuple, curr_reg_2,
                                                                        "second", consider_prev_weight)
            counter += 1
        # Schedulers (Epoch Level)
        if stage1_schedulers:
            for s in stage1_schedulers: s.step()
        if stage2_schedulers:
            for s in stage2_schedulers: s.step()
        stage1_loss_hist.append(epoch_stage1_loss / counter)

        pcl_model.eval()
        if (epoch % log_per_epoch == 0) | (epoch == n_epochs - 1):
            stage2_loss_hist.append(pcl_model.evaluate_second_stage_loss(second_stage_train_dataloader, loss_fn=loss_fn))
            stage2_val_loss_hist.append(pcl_model.evaluate_second_stage_loss(validation_dataloader, loss_fn=loss_fn))
            if plot_loss:
                n_plots = 2 + 2 * int(((do_A is not None) and (EY_do_A is not None) and is_standard_outcome_model))
                plt.clf()
                plt.subplot(int(n_plots / 2), 2, 1)
                plt.plot(np.arange(epoch + 1), stage1_loss_hist, linewidth=5)
                plt.xlabel("Number of Epochs", fontsize=20)
                plt.ylabel("MSE", fontsize=20)
                plt.title("Averaged Stage 1 Loss: {}".format(stage1_loss_hist[-1]), fontsize=20)
                plt.grid()
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)

                val_loss_x_axis = np.arange(len(stage2_val_loss_hist)) * log_per_epoch
                plt.subplot(int(n_plots / 2), 2, 2)
                plt.plot(val_loss_x_axis, stage2_loss_hist, linewidth=5, label = 'Training Loss')
                plt.plot(val_loss_x_axis, stage2_val_loss_hist, linewidth=5, label = 'Validation Loss')
                plt.xlabel("Number of Epochs", fontsize=20)
                plt.ylabel("MSE", fontsize=20)
                plt.title("Training Stage 2 Loss: {}\n Validation Stage 2 Loss: {}\n".format(stage2_loss_hist[-1],
                                                                                            stage2_val_loss_hist[-1]), fontsize=20)
                plt.grid()
                plt.legend()
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)

                if (do_A is not None) and (EY_do_A is not None) and is_standard_outcome_model:
                    pcl_model.compute_mean_outcome_proxy_feature(second_stage_train_dataloader)
                    # 2. Get the mean feature vector
                    mean_proxy_feature = pcl_model.mean_outcome_proxy_feature.detach()

                    # 3. Calculate L2 Norm and store history
                    mu_W_norm = torch.linalg.norm(mean_proxy_feature).item()
                    mu_W_norm_hist.append(mu_W_norm)

                    f_struct_pred = pcl_model.pred_structural_function( do_A,
                                                                        second_stage_train_dataloader.dataset.transformers[0],
                                                                        second_stage_train_dataloader.dataset.transformers[1]
                                              ).detach().cpu().numpy()
                    structured_pred_mse = (np.mean((f_struct_pred.reshape(-1, 1) - EY_do_A.detach().cpu().numpy().reshape(-1, 1)) ** 2))
                    causal_mse_list.append(structured_pred_mse)

                    plt.subplot(int(n_plots / 2), 2, 3)
                    plt.plot(np.arange(len(causal_mse_list)), causal_mse_list, linewidth=5)
                    plt.xlabel("Number of Epochs / {}".format(log_per_epoch), fontsize=20)
                    plt.ylabel("Causal MSE", fontsize=20)
                    plt.title("Causal MSE: {}".format(structured_pred_mse), fontsize=20)
                    plt.grid()
                    plt.xticks(fontsize=20)
                    plt.yticks(fontsize=20)

                    # Plot 4: Feature Magnitude (Diagnostic)
                    plt.subplot(int(n_plots / 2), 2, 4)
                    plt.plot(np.arange(len(mu_W_norm_hist)), mu_W_norm_hist, linewidth=5)
                    plt.title(f"Mean Proxy Feature Norm", fontsize=20)
                    plt.xlabel("Number of Epochs / {}".format(log_per_epoch), fontsize=20)
                    plt.ylabel("L2 Norm", fontsize=20)
                    plt.grid()
                    plt.xticks(fontsize=20)
                    plt.yticks(fontsize=20)

                clear_output(wait=True)
                display(plt.gcf())

    pcl_model.compute_mean_outcome_proxy_feature(second_stage_train_dataloader)
    return pcl_model.eval()


def create_third_stage_dataset_for_outcome_pcl_net_cate(
    pcl_model: nn.Module,
    pcl_dataloader: DataLoader,
    pcl_val_dataloader: Optional[DataLoader] = None,
    device: str = "cuda",
) -> Tuple[TensorDataset, Optional[TensorDataset]]:
    """
    Creates the dataset for the third stage regression to learn CATE.
    """

    pcl_model.eval().to(device) # Ensure featurizers are frozen and in evaluation mode

    # Check if the model has a second stage final layer (required for varphi_ZA)
    if not hasattr(pcl_model, 'final_layer_second_stage_weight') or pcl_model.final_layer_second_stage_weight is None:
        return None, None # Cannot proceed without the final layer weight

    # --- Helper Function to Process Loaders (DRY) ---
    def _process_single_loader(loader, desc_text):
        phi_V_list = []
        Composite_Target_list = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc_text):
                batch = [batch[i].to(device) for i in range(len(batch))]

                # Unpack Batch
                # A, Y, Z, W, [X]
                if pcl_model.backdoor_featurizer is not None:
                    A_batch, Y_batch_scaled, Z_batch, W_batch, V_batch, X_batch = batch[:6]
                else:
                    A_batch, Y_batch_scaled, Z_batch, W_batch, V_batch = batch[:5]
                    X_batch = None

                # --- Step 3: Compute Composite Target (Y * varphi_ZA) ---
                composite_target = pcl_model.outcome_proxy_featurizer(W_batch)
                if X_batch is not None and pcl_model.backdoor_featurizer is not None:
                    backdoor_feature = pcl_model.backdoor_featurizer(X_batch)
                    composite_target = outer_prod_batch(
                        backdoor_feature,
                        composite_target
                    ).flatten(start_dim=1)

                # --- Step 4: Store Data ---
                covariate_feature = pcl_model.covariate_featurizer(V_batch)

                phi_V_list.append(covariate_feature.cpu())
                Composite_Target_list.append(composite_target.cpu())

        if len(phi_V_list) == 0:
            return None, None

        return torch.cat(phi_V_list), torch.cat(Composite_Target_list)

    # 1. Process Training Data
    phi_V_all, Composite_Target_all = _process_single_loader(pcl_dataloader, "Creating Stage 3 Dataset")

    third_stage_dataset = TensorDataset(phi_V_all, Composite_Target_all)

    # 3. Process Validation Data
    third_stage_dataset_val = None
    if pcl_val_dataloader is not None:
        phi_V_val, Composite_Target_val = _process_single_loader(pcl_val_dataloader, "Creating Stage 3 Val Dataset")

        if phi_V_val is not None:
            third_stage_dataset_val = TensorDataset(phi_V_val, Composite_Target_val)

    return third_stage_dataset, third_stage_dataset_val


def create_third_stage_dataset_for_outcome_pcl_net_att(
    pcl_model: nn.Module,
    pcl_dataloader: DataLoader,
    pcl_val_dataloader: Optional[DataLoader] = None,
    device: str = "cuda",
    use_raw_treatment: bool = True,
) -> Tuple[TensorDataset, Optional[TensorDataset]]:
    """
    Creates the dataset for the third-stage regression used in ATT estimation.

    Third-stage target:
        c_i = phi_W(W_i)                           if no backdoor featurizer is used
        c_i = phi_X(X_i) ⊗ phi_W(W_i)             if backdoor_featurizer is used

    Third-stage input:
        - raw transformed treatment A_i           if use_raw_treatment=True
        - phi_A(A_i)                              otherwise

    The learned network approximates:
        E[c | A = a']
    """

    pcl_model.eval().to(device)

    def _process_single_loader(loader, desc_text):
        treatment_input_list = []
        composite_target_list = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc_text):
                batch = [batch[i].to(device) for i in range(len(batch))]

                if pcl_model.backdoor_featurizer is not None:
                    # A, Y, Z, W, X
                    A_batch, _, _, W_batch, X_batch = batch[:5]
                else:
                    # A, Y, Z, W
                    A_batch, _, _, W_batch = batch[:4]
                    X_batch = None

                # --- Third-stage input ---
                if use_raw_treatment:
                    # Use treatment as it appears in the dataloader.
                    # If the treatment transformer is identity, this is the raw image.
                    treatment_input = A_batch
                else:
                    treatment_input = pcl_model.treatment_featurizer(A_batch)

                # --- Third-stage target ---
                outcome_proxy_feature = pcl_model.outcome_proxy_featurizer(W_batch)

                if X_batch is not None and pcl_model.backdoor_featurizer is not None:
                    backdoor_feature = pcl_model.backdoor_featurizer(X_batch)
                    composite_target = outer_prod_batch(
                        backdoor_feature,
                        outcome_proxy_feature
                    ).flatten(start_dim=1)
                else:
                    composite_target = outcome_proxy_feature

                treatment_input_list.append(treatment_input.cpu())
                composite_target_list.append(composite_target.cpu())

        if len(treatment_input_list) == 0:
            return None, None

        return torch.cat(treatment_input_list), torch.cat(composite_target_list)

    treatment_input_all, composite_target_all = _process_single_loader(
        pcl_dataloader,
        "Creating ATT Stage 3 Dataset"
    )
    third_stage_dataset = TensorDataset(treatment_input_all, composite_target_all)

    third_stage_dataset_val = None
    if pcl_val_dataloader is not None:
        treatment_input_val, composite_target_val = _process_single_loader(
            pcl_val_dataloader,
            "Creating ATT Stage 3 Val Dataset"
        )
        if treatment_input_val is not None:
            third_stage_dataset_val = TensorDataset(treatment_input_val, composite_target_val)

    return third_stage_dataset, third_stage_dataset_val