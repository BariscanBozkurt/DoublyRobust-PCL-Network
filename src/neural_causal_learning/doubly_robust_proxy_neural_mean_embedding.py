import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Optional
from tqdm import tqdm

import sys
sys.path.append("..")
from torch_utils.scalers import TorchIdentityTransformer
from torch_utils.linear_algebra import outer_prod, outer_prod_batch, add_const_col

def create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_ate(
    outcome_pcl_model: nn.Module,
    treatment_pcl_model: nn.Module,
    pcl_dataloader: DataLoader,
    pcl_val_dataloader: Optional[DataLoader] = None,
    outcome_transformer: Optional[nn.Module] = None,
    input_transformer: Optional[nn.Module] = None,
    input_type: str = "raw", # "raw" -> returns A; "features" -> returns phi_A
    treatment_featurizer: Optional[nn.Module] = None,
    dens_ratio_pred_tolerance: float = 100.5,
    device: str = "cuda",
) -> Tuple[TensorDataset, Optional[TensorDataset]]:
    """
    Generates the dataset for the Third Stage Doubly Robust (DR) regression.

    The DR target is mathematically defined as:
    Y_DR = h(A, W, X) + varphi(A, Z, X) * (Y - h(A, W, X))
    """

    # 1. Setup & Safety Checks
    outcome_pcl_model.eval().to(device)
    treatment_pcl_model.eval().to(device)

    if not hasattr(treatment_pcl_model, 'final_layer_second_stage_weight') or treatment_pcl_model.final_layer_second_stage_weight is None:
        return None, None

    if treatment_featurizer is None:
        treatment_featurizer = outcome_pcl_model.treatment_featurizer_second_stage

    if outcome_transformer is None: outcome_transformer = TorchIdentityTransformer()
    if input_transformer is None: input_transformer = TorchIdentityTransformer()

    y_transformer = pcl_dataloader.dataset.transformers[1].to(device)
    dens_ratio_transformer = pcl_dataloader.dataset.dens_ratio_transformer.to(device)

    # --- DRY Helper Function ---
    def _process_single_loader(loader: DataLoader, desc_text: str):
        phi_A_list = []
        Composite_Target_list = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc_text):
                batch = [b.to(device) for b in batch]

                # Unpack Batch
                if len(batch) > 5:
                    A_batch, Y_batch_scaled, Z_batch, W_batch, X_batch, dens_ratio_batch = batch
                else:
                    A_batch, Y_batch_scaled, Z_batch, W_batch, dens_ratio_batch = batch
                    X_batch = None

                # 1. Density Ratio Inlier Filtering
                dens_ratio_pred = treatment_pcl_model.predict_dens_ratio(A_batch, W_batch, X_batch)
                dens_ratio_pred_unscaled = dens_ratio_transformer.inverse_transform(dens_ratio_pred)

                inlier_mask = (
                    (torch.abs(dens_ratio_pred - dens_ratio_batch) < dens_ratio_pred_tolerance) &
                    (dens_ratio_pred_unscaled > 0.0)
                ).view(-1)

                if inlier_mask.sum() == 0:
                    continue

                # Filter tensors to inliers
                A_in = A_batch[inlier_mask]
                Y_in_scaled = Y_batch_scaled[inlier_mask]
                Z_in = Z_batch[inlier_mask]
                W_in = W_batch[inlier_mask]
                X_in = X_batch[inlier_mask] if X_batch is not None else None

                # 2. Get Y and h(A, W, X)
                Y_unscaled = y_transformer.inverse_transform(Y_in_scaled)
                h_pred_scaled = outcome_pcl_model.predict_outcome_bridge_function(A_in, W_in, X_in)
                h_unscaled = y_transformer.inverse_transform(h_pred_scaled)

                # 3. Compute \varphi(A, Z, X)
                varphi_scaled = treatment_pcl_model.predict_treatment_bridge_function(A_in, Z_in, X_in)
                varphi_unscaled = dens_ratio_transformer.inverse_transform(varphi_scaled)

                # 4. Construct Doubly Robust Target
                # Target = \varphi(A,Z,X) * (Y - h(A,W,X))
                composite_target = varphi_unscaled * (Y_unscaled - h_unscaled)

                # 5. Process Input A
                if input_type == "features":
                    A_processed = treatment_featurizer(A_in)
                else:
                    A_processed = A_in

                phi_A_list.append(A_processed.cpu())
                Composite_Target_list.append(composite_target.cpu())

        if not phi_A_list:
            return None, None

        return torch.cat(phi_A_list), torch.cat(Composite_Target_list)

    # --- Execute for Train Set ---
    phi_A_train, Target_train = _process_single_loader(pcl_dataloader, "Creating Stage 3 Train DR Data")

    if phi_A_train is None:
        raise ValueError("No valid inliers found in the training dataset.")

    phi_A_train = input_transformer.fit_transform(phi_A_train)
    Target_train = outcome_transformer.fit_transform(Target_train)

    train_dataset = TensorDataset(phi_A_train, Target_train)
    train_dataset.outcome_transformer = outcome_transformer
    train_dataset.input_transformer = input_transformer

    # --- Execute for Validation Set ---
    val_dataset = None
    if pcl_val_dataloader is not None:
        phi_A_val, Target_val = _process_single_loader(pcl_val_dataloader, "Creating Stage 3 Val DR Data")

        if phi_A_val is not None:
            phi_A_val = input_transformer.transform(phi_A_val)
            Target_val = outcome_transformer.transform(Target_val)
            val_dataset = TensorDataset(phi_A_val, Target_val)

    return train_dataset, val_dataset


def create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_ate_v2(
    outcome_pcl_model: nn.Module,
    treatment_pcl_model: nn.Module,
    pcl_dataloader: DataLoader,
    pcl_val_dataloader: Optional[DataLoader] = None,
    outcome_transformer: Optional[nn.Module] = None,
    input_transformer: Optional[nn.Module] = None,
    input_type: str = "raw", # "raw" -> returns A; "features" -> returns phi_A
    treatment_featurizer: Optional[nn.Module] = None,
    dens_ratio_pred_tolerance: float = 100.5,
    device: str = "cuda",
) -> Tuple[TensorDataset, Optional[TensorDataset]]:
    """
    Generates the dataset for the Third Stage Doubly Robust (DR) regression.

    The DR target is mathematically defined as:
    Y_DR = varphi(A, Z, X) * (- h(A, W, X))
    """

    # 1. Setup & Safety Checks
    outcome_pcl_model.eval().to(device)
    treatment_pcl_model.eval().to(device)

    if not hasattr(treatment_pcl_model, 'final_layer_second_stage_weight') or treatment_pcl_model.final_layer_second_stage_weight is None:
        return None, None

    if treatment_featurizer is None:
        treatment_featurizer = outcome_pcl_model.treatment_featurizer_second_stage

    if outcome_transformer is None: outcome_transformer = TorchIdentityTransformer()
    if input_transformer is None: input_transformer = TorchIdentityTransformer()

    y_transformer = pcl_dataloader.dataset.transformers[1].to(device)
    dens_ratio_transformer = pcl_dataloader.dataset.dens_ratio_transformer.to(device)

    # --- DRY Helper Function ---
    def _process_single_loader(loader: DataLoader, desc_text: str):
        phi_A_list = []
        Composite_Target_list = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc_text):
                batch = [b.to(device) for b in batch]

                # Unpack Batch
                if len(batch) > 5:
                    A_batch, Y_batch_scaled, Z_batch, W_batch, X_batch, dens_ratio_batch = batch
                else:
                    A_batch, Y_batch_scaled, Z_batch, W_batch, dens_ratio_batch = batch
                    X_batch = None

                # 1. Density Ratio Inlier Filtering
                dens_ratio_pred = treatment_pcl_model.predict_dens_ratio(A_batch, W_batch, X_batch)
                dens_ratio_pred_unscaled = dens_ratio_transformer.inverse_transform(dens_ratio_pred)

                inlier_mask = (
                    (torch.abs(dens_ratio_pred - dens_ratio_batch) < dens_ratio_pred_tolerance) &
                    (dens_ratio_pred_unscaled > 0.0)
                ).view(-1)

                if inlier_mask.sum() == 0:
                    continue

                # Filter tensors to inliers
                A_in = A_batch[inlier_mask]
                Z_in = Z_batch[inlier_mask]
                W_in = W_batch[inlier_mask]
                X_in = X_batch[inlier_mask] if X_batch is not None else None

                # 2. Get Y and h(A, W, X)
                h_pred_scaled = outcome_pcl_model.predict_outcome_bridge_function(A_in, W_in, X_in)
                h_unscaled = y_transformer.inverse_transform(h_pred_scaled)

                # 3. Compute \varphi(A, Z, X)
                varphi_scaled = treatment_pcl_model.predict_treatment_bridge_function(A_in, Z_in, X_in)
                varphi_unscaled = dens_ratio_transformer.inverse_transform(varphi_scaled)

                # 4. Construct Doubly Robust Target
                # Target = \varphi(A,Z,X) * (h(A,W,X))
                composite_target = varphi_unscaled * (h_unscaled)

                # 5. Process Input A
                if input_type == "features":
                    A_processed = treatment_featurizer(A_in)
                else:
                    A_processed = A_in

                phi_A_list.append(A_processed.cpu())
                Composite_Target_list.append(composite_target.cpu())

        if not phi_A_list:
            return None, None

        return torch.cat(phi_A_list), torch.cat(Composite_Target_list)

    # --- Execute for Train Set ---
    phi_A_train, Target_train = _process_single_loader(pcl_dataloader, "Creating Stage 3 Train DR Data")

    if phi_A_train is None:
        raise ValueError("No valid inliers found in the training dataset.")

    phi_A_train = input_transformer.fit_transform(phi_A_train)
    Target_train = outcome_transformer.fit_transform(Target_train)

    train_dataset = TensorDataset(phi_A_train, Target_train)
    train_dataset.outcome_transformer = outcome_transformer
    train_dataset.input_transformer = input_transformer

    # --- Execute for Validation Set ---
    val_dataset = None
    if pcl_val_dataloader is not None:
        phi_A_val, Target_val = _process_single_loader(pcl_val_dataloader, "Creating Stage 3 Val DR Data")

        if phi_A_val is not None:
            phi_A_val = input_transformer.transform(phi_A_val)
            Target_val = outcome_transformer.transform(Target_val)
            val_dataset = TensorDataset(phi_A_val, Target_val)

    return train_dataset, val_dataset


def create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_cate(
    outcome_pcl_model: nn.Module,
    treatment_pcl_model: nn.Module,
    pcl_dataloader: DataLoader,
    pcl_val_dataloader: Optional[DataLoader] = None,
    outcome_transformer: Optional[nn.Module] = None,
    input_transformer: Optional[nn.Module] = None,
    dens_ratio_pred_tolerance: float = 100.5,
    device: str = "cuda",
) -> Tuple[TensorDataset, Optional[TensorDataset]]:
    """
    Generates the dataset for the Third Stage Doubly Robust (DR) regression.

    The DR target is mathematically defined as:
    Y_DR = h(A, V, W, X) + varphi(A, V, Z, X) * (Y - h(A, V, W, X))
    """

    # 1. Setup & Safety Checks
    outcome_pcl_model.eval().to(device)
    treatment_pcl_model.eval().to(device)

    if not hasattr(treatment_pcl_model, 'final_layer_second_stage_weight') or treatment_pcl_model.final_layer_second_stage_weight is None:
        return None, None

    if outcome_transformer is None: outcome_transformer = TorchIdentityTransformer()
    if input_transformer is None: input_transformer = TorchIdentityTransformer()

    y_transformer = pcl_dataloader.dataset.transformers[1].to(device)
    dens_ratio_transformer = pcl_dataloader.dataset.dens_ratio_transformer.to(device)

    # --- DRY Helper Function ---
    def _process_single_loader(loader: DataLoader, desc_text: str):
        input_list = []
        Composite_Target_list = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc_text):
                batch = [b.to(device) for b in batch]

                # Unpack Batch
                if len(batch) > 6:
                    A_batch, Y_batch_scaled, Z_batch, W_batch, V_batch, X_batch, dens_ratio_batch = batch
                    dens_ratio_batch = dens_ratio_transformer.inverse_transform(dens_ratio_batch) # Just to ensure it's on the right scale
                else:
                    A_batch, Y_batch_scaled, Z_batch, W_batch, V_batch, dens_ratio_batch = batch
                    dens_ratio_batch = dens_ratio_transformer.inverse_transform(dens_ratio_batch) # Just to ensure it's on the right scale
                    X_batch = None

                # 1. Density Ratio Inlier Filtering
                dens_ratio_pred_batch = treatment_pcl_model.predict_dens_ratio(A_batch, V_batch, W_batch, X_batch)

                inlier_mask = (
                    (torch.abs(dens_ratio_pred_batch - dens_ratio_batch) < dens_ratio_pred_tolerance) &
                    (dens_ratio_pred_batch > 0.0)
                ).view(-1)

                if inlier_mask.sum() == 0:
                    continue

                # Filter tensors to inliers
                A_in = A_batch[inlier_mask]
                Y_in_scaled = Y_batch_scaled[inlier_mask]
                Z_in = Z_batch[inlier_mask]
                W_in = W_batch[inlier_mask]
                V_in = V_batch[inlier_mask]
                X_in = X_batch[inlier_mask] if X_batch is not None else None

                # 2. Get Y and h(A, W, X)
                Y_unscaled = y_transformer.inverse_transform(Y_in_scaled)
                h_pred_scaled = outcome_pcl_model.predict_outcome_bridge_function(A_in, V_in, W_in, X_in)
                h_unscaled = y_transformer.inverse_transform(h_pred_scaled)

                # 3. Compute \varphi(A, Z, X)
                varphi_scaled = treatment_pcl_model.predict_treatment_bridge_function(A_in, V_in, Z_in, X_in)
                varphi_unscaled = dens_ratio_transformer.inverse_transform(varphi_scaled)

                # 4. Construct Doubly Robust Target
                # Target = \varphi(A,V,Z,X) * (Y - h(A,V,W,X))
                composite_target = varphi_unscaled * (Y_unscaled - h_unscaled)

                input_list.append(torch.hstack([A_in, V_in]).cpu())
                Composite_Target_list.append(composite_target.cpu())

        if not input_list:
            return None, None

        return torch.cat(input_list), torch.cat(Composite_Target_list)

    # --- Execute for Train Set ---
    phi_A_train, Target_train = _process_single_loader(pcl_dataloader, "Creating Stage 3 Train DR Data")

    if phi_A_train is None:
        raise ValueError("No valid inliers found in the training dataset.")

    phi_A_train = input_transformer.fit_transform(phi_A_train)
    Target_train = outcome_transformer.fit_transform(Target_train)

    train_dataset = TensorDataset(phi_A_train, Target_train)
    train_dataset.outcome_transformer = outcome_transformer
    train_dataset.input_transformer = input_transformer

    # --- Execute for Validation Set ---
    val_dataset = None
    if pcl_val_dataloader is not None:
        phi_A_val, Target_val = _process_single_loader(pcl_val_dataloader, "Creating Stage 3 Val DR Data")

        if phi_A_val is not None:
            phi_A_val = input_transformer.transform(phi_A_val)
            Target_val = outcome_transformer.transform(Target_val)
            val_dataset = TensorDataset(phi_A_val, Target_val)

    return train_dataset, val_dataset


def create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_cate_v2(
    outcome_pcl_model: nn.Module,
    treatment_pcl_model: nn.Module,
    pcl_dataloader: DataLoader,
    pcl_val_dataloader: Optional[DataLoader] = None,
    outcome_transformer: Optional[nn.Module] = None,
    input_transformer: Optional[nn.Module] = None,
    dens_ratio_pred_tolerance: float = 100.5,
    device: str = "cuda",
) -> Tuple[TensorDataset, Optional[TensorDataset]]:
    """
    Generates the dataset for the Third Stage Doubly Robust (DR) regression.

    The DR target is mathematically defined as:
    Y_DR = h(A, V, W, X) + varphi(A, V, Z, X) * (Y - h(A, V, W, X))
    """

    # 1. Setup & Safety Checks
    outcome_pcl_model.eval().to(device)
    treatment_pcl_model.eval().to(device)

    if not hasattr(treatment_pcl_model, 'final_layer_second_stage_weight') or treatment_pcl_model.final_layer_second_stage_weight is None:
        return None, None

    if outcome_transformer is None: outcome_transformer = TorchIdentityTransformer()
    if input_transformer is None: input_transformer = TorchIdentityTransformer()

    y_transformer = pcl_dataloader.dataset.transformers[1].to(device)
    dens_ratio_transformer = pcl_dataloader.dataset.dens_ratio_transformer.to(device)

    # --- DRY Helper Function ---
    def _process_single_loader(loader: DataLoader, desc_text: str):
        input_list = []
        Composite_Target_list = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc_text):
                batch = [b.to(device) for b in batch]

                # Unpack Batch
                if len(batch) > 6:
                    A_batch, Y_batch_scaled, Z_batch, W_batch, V_batch, X_batch, dens_ratio_batch = batch
                    dens_ratio_batch = dens_ratio_transformer.inverse_transform(dens_ratio_batch) # Just to ensure it's on the right scale
                else:
                    A_batch, Y_batch_scaled, Z_batch, W_batch, V_batch, dens_ratio_batch = batch
                    dens_ratio_batch = dens_ratio_transformer.inverse_transform(dens_ratio_batch) # Just to ensure it's on the right scale
                    X_batch = None

                # 1. Density Ratio Inlier Filtering
                dens_ratio_pred_batch = treatment_pcl_model.predict_dens_ratio(A_batch, V_batch, W_batch, X_batch)

                inlier_mask = (
                    (torch.abs(dens_ratio_pred_batch - dens_ratio_batch) < dens_ratio_pred_tolerance) &
                    (dens_ratio_pred_batch > 0.0)
                ).view(-1)

                if inlier_mask.sum() == 0:
                    continue

                # Filter tensors to inliers
                A_in = A_batch[inlier_mask]
                Z_in = Z_batch[inlier_mask]
                W_in = W_batch[inlier_mask]
                V_in = V_batch[inlier_mask]
                X_in = X_batch[inlier_mask] if X_batch is not None else None

                # 2. Get Y and h(A, W, X)
                h_pred_scaled = outcome_pcl_model.predict_outcome_bridge_function(A_in, V_in, W_in, X_in)
                h_unscaled = y_transformer.inverse_transform(h_pred_scaled)

                # 3. Compute \varphi(A, Z, X)
                varphi_scaled = treatment_pcl_model.predict_treatment_bridge_function(A_in, V_in, Z_in, X_in)
                varphi_unscaled = dens_ratio_transformer.inverse_transform(varphi_scaled)

                # 4. Construct Doubly Robust Target
                # Target = \varphi(A,V,Z,X) * h(A,V,W,X)
                composite_target = varphi_unscaled * (h_unscaled)

                input_list.append(torch.hstack([A_in, V_in]).cpu())
                Composite_Target_list.append(composite_target.cpu())

        if not input_list:
            return None, None

        return torch.cat(input_list), torch.cat(Composite_Target_list)

    # --- Execute for Train Set ---
    phi_A_train, Target_train = _process_single_loader(pcl_dataloader, "Creating Stage 3 Train DR Data")

    if phi_A_train is None:
        raise ValueError("No valid inliers found in the training dataset.")

    phi_A_train = input_transformer.fit_transform(phi_A_train)
    Target_train = outcome_transformer.fit_transform(Target_train)

    train_dataset = TensorDataset(phi_A_train, Target_train)
    train_dataset.outcome_transformer = outcome_transformer
    train_dataset.input_transformer = input_transformer

    # --- Execute for Validation Set ---
    val_dataset = None
    if pcl_val_dataloader is not None:
        phi_A_val, Target_val = _process_single_loader(pcl_val_dataloader, "Creating Stage 3 Val DR Data")

        if phi_A_val is not None:
            phi_A_val = input_transformer.transform(phi_A_val)
            Target_val = outcome_transformer.transform(Target_val)
            val_dataset = TensorDataset(phi_A_val, Target_val)

    return train_dataset, val_dataset