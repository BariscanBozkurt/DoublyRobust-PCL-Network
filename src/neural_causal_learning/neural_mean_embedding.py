import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Callable
from torch.optim import Optimizer

import sys
sys.path.append("..")
from torch_utils.scalers import TorchIdentityTransformer
from torch_utils.linear_algebra import fit_linear_proximal, fit_linear, outer_prod, linear_reg_loss
# from torch_utils.dataset_classes import BackdoorDataset
from torch_utils.model_helpers import get_last_linear_out_features
from tqdm import tqdm


class NMEBackdoorATE(nn.Module):
    """
    Neural Mean Embedding (NME) model for estimating Average Treatment Effect (ATE)
    under backdoor adjustment.

    Parameters
    ----------
    treatment_featurizer : nn.Module
        A PyTorch model mapping treatment A to feature space.
    backdoor_featurizer : nn.Module
        A PyTorch model mapping backdoor variables U to feature space.
    device : str
        Device for computation, e.g., 'cpu' or 'cuda'.
    final_layer : nn.Linear, optional
        Optional final linear layer for combining treatment and backdoor features.
    """
    def __init__(self,
                 treatment_featurizer: nn.Module,
                 backdoor_featurizer: nn.Module,
                 device: str = "cuda",
                 **kwargs):
        super().__init__()
        self.treatment_featurizer = treatment_featurizer.to(device)
        self.backdoor_featurizer = backdoor_featurizer.to(device)
        self.backdoor_feature_dim = get_last_linear_out_features(backdoor_featurizer)
        self.treatment_feature_dim = get_last_linear_out_features(treatment_featurizer)
        self.mean_backdoor_feature = torch.zeros(self.backdoor_feature_dim, requires_grad=False).to(device)

        self.final_layer = kwargs.pop('final_layer', None)
        if self.final_layer is None:
            self.final_layer = nn.Linear(
                self.treatment_feature_dim * self.backdoor_feature_dim,
                1, bias=False
            ).to(device)

        self.device = device

    def forward(self, treatment: torch.Tensor, backdoor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predicts outcome using treatment and backdoor features.
        """
        n_test = treatment.shape[0]
        test_treatment_feature = self.treatment_featurizer(treatment.to(self.device))
        test_backdoor_feature = self.backdoor_featurizer(backdoor.to(self.device))
        feature = outer_prod(test_treatment_feature, test_backdoor_feature).reshape((n_test, -1))
        return self.final_layer(feature)

    def pred_structural_function(
        self,
        treatment: torch.Tensor,
        treatment_transformer: nn.Module = TorchIdentityTransformer(),
        outcome_transformer: nn.Module = TorchIdentityTransformer()
    ) -> torch.Tensor:
        """
        Predict structural function E[Y | do(A = a)] using mean backdoor feature.
        """
        training_state = self.training
        self.eval()
        with torch.no_grad():
            n_test = treatment.shape[0]
            test_treatment_feature = self.treatment_featurizer(
                treatment_transformer.transform(treatment)
            )
            feature = outer_prod(
                test_treatment_feature, self.mean_backdoor_feature.repeat(n_test, 1)
            ).reshape((n_test, -1))
            pred = outcome_transformer.inverse_transform(self.final_layer(feature))

        # --- Restore original training state ---
        if training_state:
            self.train()
        return pred

    def compute_mean_backdoor_feature(self, train_dataloader: DataLoader):
        """
        Compute mean backdoor feature from the training data.
        """
        with torch.no_grad():
            backdoor_feature_list = []
            for batch in train_dataloader:
                backdoor_feature_list.append(
                    self.backdoor_featurizer(batch[2].to(self.device))
                )
            backdoor_feature = torch.cat(backdoor_feature_list)
            self.mean_backdoor_feature = torch.mean(backdoor_feature, dim=0)

    def freeze_featurizer_params(self):
        """Freeze parameters of treatment and backdoor featurizers."""
        for param in self.backdoor_featurizer.parameters():
            param.requires_grad = False
        for param in self.treatment_featurizer.parameters():
            param.requires_grad = False

    def unfreeze_featurizer_params(self):
        """Unfreeze parameters of treatment and backdoor featurizers."""
        for param in self.backdoor_featurizer.parameters():
            param.requires_grad = True
        for param in self.treatment_featurizer.parameters():
            param.requires_grad = True

    def freeze_final_layer_params(self):
        """Freeze parameters of the final linear layer."""
        for param in self.final_layer.parameters():
            param.requires_grad = False

    def unfreeze_final_layer_params(self):
        """Unfreeze parameters of the final linear layer."""
        for param in self.final_layer.parameters():
            param.requires_grad = True

    def update_final_layer_with_batch_regression(
        self, data_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        weight_decay: float = 0.0
    ):
        """
        Update final layer weights using linear regression on batch data.
        """
        with torch.no_grad():
            A_tensor, Y_tensor, X_tensor = [x.to(self.device) for x in data_tuple]
            backdoor_feature = self.backdoor_featurizer(X_tensor)
            treatment_feature = self.treatment_featurizer(A_tensor)
            feature = outer_prod(treatment_feature, backdoor_feature).reshape((backdoor_feature.shape[0], -1))
            weight_previous = self.final_layer.weight.data
            weight = fit_linear_proximal(Y_tensor, feature, weight_previous, weight_decay)
            self.final_layer.weight.data = weight

    def evaluate_nme_backdoor_regression(
        self, val_dataloader: DataLoader, loss_fn: Callable
    ) -> float:
        """
        Evaluate average loss on given data loader.
        """
        training_state = self.training
        self.eval()

        device = self.device
        loss_sum = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in val_dataloader:
                A_batch, Y_batch, U_batch = [x.to(device) for x in batch]
                pred = self.forward(A_batch, U_batch)
                loss_sum += loss_fn(pred, Y_batch).item()
                num_batches += 1

        # --- Restore original training state ---
        if training_state:
            self.train()

        return loss_sum / num_batches


def train_nme_backdoor_ate_model(
    nme_model: torch.nn.Module,
    train_dataloader: DataLoader,
    backdoor_optimizer: Optimizer,
    treatment_optimizer: Optional[Optimizer],
    finallayer_optimizer: Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    n_epochs: int = 100,
    use_full_data_to_update_final_layer: bool = True,
    device: str = "cuda",
    verbose: bool = True,
    log_per_epoch: int = 25
) -> torch.nn.Module:
    """
    Train a Neural Mean Embedding (NME) model for backdoor adjustment in ATE estimation.

    The training consists of two phases per epoch:
    1. Update featurizers (backdoor and treatment) per batch.
    2. Update the final layer using either the full dataset or per-batch regression.

    Args:
        nme_model (torch.nn.Module): Neural Mean Embedding model.
        train_dataloader (DataLoader): DataLoader providing training batches (A, Y, U).
        backdoor_optimizer (Optimizer): Optimizer for the backdoor featurizer parameters.
        treatment_optimizer (Optional[Optimizer]): Optimizer for the treatment featurizer parameters. Can be None.
        finallayer_optimizer (Optimizer): Optimizer for the final layer parameters.
        loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Loss function (e.g., MSELoss).
        n_epochs (int, optional): Number of training epochs. Defaults to 100.
        use_full_data_to_update_final_layer (bool, optional): Whether to update the final layer using the full dataset. Defaults to True.
        device (str, optional): Device for tensors ('cuda' or 'cpu'). Defaults to "cuda".
        verbose (bool, optional): If True, print training progress. Defaults to True.
        log_per_epoch (int, optional): Frequency of logging loss (every `log_per_epoch` epochs). Defaults to 25.

    Returns:
        torch.nn.Module: Trained NME model with updated featurizers and final layer.
    """
    # Unfreeze final layer
    nme_model.unfreeze_final_layer_params()
    # Unfreeze featurizers
    nme_model.unfreeze_featurizer_params()

    for i in tqdm(range(n_epochs)):

        # ---- Phase 1: Update featurizers per batch ----
        for batch in train_dataloader:
            A_batch, Y_batch, U_batch = [x.to(device) for x in batch]

            backdoor_optimizer.zero_grad()
            if treatment_optimizer is not None:
                treatment_optimizer.zero_grad()
            finallayer_optimizer.zero_grad()

            pred = nme_model(A_batch, U_batch)
            loss = loss_fn(pred, Y_batch) + finallayer_optimizer.param_groups[0].get("weight_decay", 0.0) * torch.linalg.norm(nme_model.final_layer.weight.data) ** 2
            loss.backward()

            backdoor_optimizer.step()
            if treatment_optimizer is not None:
                treatment_optimizer.step()
            finallayer_optimizer.step()
            # # ---- Phase 2: Update final layer over the entire dataset ----
            # # Freeze featurizers
            # nme_model.freeze_featurizer_params

            # if use_full_data_to_update_final_layer:
            #     nme_model.update_final_layer_with_batch_regression(
            #         (train_dataloader.dataset.A, train_dataloader.dataset.Y, train_dataloader.dataset.U), finallayer_optimizer.param_groups[0].get("weight_decay", 0.0)
            #     )
            # else:
            #     # Unfreeze final layer
            #     nme_model.unfreeze_final_layer_params

            #     for batch_ in train_dataloader:
            #         A_batch_, Y_batch_, U_batch_ = [x.to(device) for x in batch_]

            #         finallayer_optimizer.zero_grad

            #         backdoor_feature_ = nme_model.backdoor_featurizer(U_batch_)
            #         treatment_feature_ = nme_model.treatment_featurizer(A_batch_)
            #         feature_ = outer_prod(treatment_feature_, backdoor_feature_).reshape(
            #             (backdoor_feature_.shape[0], -1)
            #         )

            #         pred = nme_model.final_layer(feature_)
            #         loss = loss_fn(pred, Y_batch_)
            #         loss.backward
            #         finallayer_optimizer.step

        if verbose and i % log_per_epoch == 0:
            nme_loss_avg = nme_model.evaluate_nme_backdoor_regression(train_dataloader, loss_fn)
            print(f"Iteration {i+1}: NME Backdoor model training loss = {nme_loss_avg:.6f}")

    nme_model.backdoor_featurizer.eval()
    nme_model.treatment_featurizer.eval()
    nme_model.final_layer.eval()

    nme_model.compute_mean_backdoor_feature(train_dataloader)
    return nme_model

def train_nme_backdoor_closed_form_ate_model(
    nme_model: torch.nn.Module,
    train_dataloader: DataLoader,
    backdoor_optimizer: Optimizer,
    treatment_optimizer: Optional[Optimizer],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    final_layer_weight_decay: float = 1e-3,
    n_epochs: int = 100,
    device: str = "cuda",
    verbose: bool = True,
    log_per_epoch: int = 25
) -> torch.nn.Module:
    """
    Train a Neural Mean Embedding (NME) model for backdoor adjustment in ATE estimation.

    The training consists of two phases per epoch:
    1. Update featurizers (backdoor and treatment) per batch.
    2. Update the final layer using either the full dataset or per-batch regression.

    Args:
        nme_model (torch.nn.Module): Neural Mean Embedding model.
        train_dataloader (DataLoader): DataLoader providing training batches (A, Y, U).
        backdoor_optimizer (Optimizer): Optimizer for the backdoor featurizer parameters.
        treatment_optimizer (Optional[Optimizer]): Optimizer for the treatment featurizer parameters. Can be None.
        loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Loss function (e.g., MSELoss).
        n_epochs (int, optional): Number of training epochs. Defaults to 100.
        device (str, optional): Device for tensors ('cuda' or 'cpu'). Defaults to "cuda".
        verbose (bool, optional): If True, print training progress. Defaults to True.
        log_per_epoch (int, optional): Frequency of logging loss (every `log_per_epoch` epochs). Defaults to 25.

    Returns:
        torch.nn.Module: Trained NME model with updated featurizers and final layer.
    """
    # Freeze final layer
    nme_model.freeze_final_layer_params()
    # Unfreeze featurizers
    nme_model.unfreeze_featurizer_params()

    for i in tqdm(range(n_epochs)):

        # ---- Phase 1: Update featurizers per batch ----
        for batch in train_dataloader:
            A_batch, Y_batch, U_batch = [x.to(device) for x in batch]

            nme_model.update_final_layer_with_batch_regression((A_batch, Y_batch, U_batch), final_layer_weight_decay)

            backdoor_optimizer.zero_grad()
            if treatment_optimizer is not None:
                treatment_optimizer.zero_grad()

            pred = nme_model(A_batch, U_batch)
            loss = loss_fn(pred, Y_batch) + 1e-3 * torch.linalg.norm(nme_model.final_layer.weight.data) ** 2
            loss.backward()

            backdoor_optimizer.step()
            if treatment_optimizer is not None:
                treatment_optimizer.step()

        if verbose and ((i % log_per_epoch == 0) or (i == n_epochs - 1)):
            nme_loss_avg = nme_model.evaluate_nme_backdoor_regression(train_dataloader, loss_fn)
            print(f"Iteration {i+1}: NME Backdoor model training loss = {nme_loss_avg:.6f}")

    nme_model.backdoor_featurizer.eval()
    nme_model.treatment_featurizer.eval()
    nme_model.final_layer.eval()

    nme_model.compute_mean_backdoor_feature(train_dataloader)
    return nme_model


def train_nme_backdoor_ate_model_full_batch(  nme_model: torch.nn.Module,
                                              train_dataloader: DataLoader,
                                              backdoor_optimizer: Optimizer,
                                              treatment_optimizer: Optional[Optimizer],
                                              final_layer_weight_decay: float,
                                              loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                              n_epochs: int = 100,
                                              use_full_data_to_update_final_layer: bool = True,
                                              device: str = "cuda",
                                              verbose: bool = True,
                                              log_per_epoch: int = 25):

    A_tensor, Y_tensor, X_tensor = train_dataloader.dataset.A, train_dataloader.dataset.Y, train_dataloader.dataset.U
    A_tensor, Y_tensor, X_tensor = A_tensor.to(device), Y_tensor.to(device), X_tensor.to(device)
    # Freeze final layer
    nme_model.freeze_final_layer_params()

    for i in tqdm(range(n_epochs)):
        nme_model.unfreeze_featurizer_params()
        backdoor_optimizer.zero_grad()
        if treatment_optimizer is not None:
            treatment_optimizer.zero_grad()

        backdoor_feature = nme_model.backdoor_featurizer(X_tensor)
        treatment_feature = nme_model.treatment_featurizer(A_tensor)

        feature = outer_prod(treatment_feature, backdoor_feature).reshape((backdoor_feature.shape[0], -1))
        loss, weight = linear_reg_loss(Y_tensor, feature, final_layer_weight_decay)
        loss.backward()

        backdoor_optimizer.step()
        if treatment_optimizer is not None:
            treatment_optimizer.step()

        # nme_model.final_layer.weight.data = weight.T
        # Freeze featurizers
        # nme_model.freeze_featurizer_params
        with torch.no_grad():
            backdoor_feature = nme_model.backdoor_featurizer(X_tensor)
            treatment_feature = nme_model.treatment_featurizer(A_tensor)

            feature = outer_prod(treatment_feature, backdoor_feature).reshape((backdoor_feature.shape[0], -1))
            _, weight = linear_reg_loss(Y_tensor, feature, final_layer_weight_decay)
            nme_model.final_layer.weight.data = weight.T

        if verbose and i % log_per_epoch == 0:
            nme_loss_avg = nme_model.evaluate_nme_backdoor_regression(train_dataloader, loss_fn)
            print(f"Iteration {i+1}: NME Backdoor model training loss = {nme_loss_avg:.6f}")

    nme_model.backdoor_featurizer.eval()
    nme_model.treatment_featurizer.eval()

    nme_model.compute_mean_backdoor_feature(train_dataloader)
    return nme_model
