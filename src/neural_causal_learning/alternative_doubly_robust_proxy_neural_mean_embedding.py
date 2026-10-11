"""Doubly-robust third-stage datasets for the DENSITY-RATIO-FREE bridge (no inlier selection).

Density-ratio-free counterpart of doubly_robust_proxy_neural_mean_embedding.py. The two creators build
exactly the same DR targets as their density-ratio siblings,

    V1 (robust form):   slack_1 = E[ varphi(A,Z,X) * (Y - h(A,W,X)) | A = a ]   ->  f_dr1 = f_om + slack_1
    V2 (AIPW form):     slack_2 = E[ varphi(A,Z,X) *      h(A,W,X)  | A = a ]   ->  f_dr2 = f_om + f_tm - slack_2

with ONE difference: there is NO INLIER SELECTION -- every row is kept.

WHY. The shared creators drop rows on
    |predict_dens_ratio - stored_ratio| < tol   AND   predict_dens_ratio > 0.
For the density-ratio method that is a real calibration check: `stored_ratio` is an actual KDE / RuLSIF /
flow estimate and the comparison says "does the bridge agree with the ratio we estimated?". In the
density-ratio-free setting the stored ratio is a `ones` PLACEHOLDER and every notebook passes
tol = 100.5, so the first test is vacuous and the only live one is a sign test on the bridge's own
implied ratio E[varphi | A, W, X] -- the model censoring its own output on a criterion unrelated to the
data. The selection is also a function of (A, W, X), so the retained sample is not a clean draw.

The knob that genuinely controls this bridge's tail behaviour is the second-stage ridge
`reg_second_stage`, which every notebook sets explicitly.

MEASURED, so the change is not taken on faith (demand benchmark, bridge held fixed, third-stage seed
varied): the filter drops 1 of 1350 rows (0.07%, a value of -0.002, i.e. numerical zero), and the
treatment-bridge causal MSE is 80.1 +/- 11.6 filtered versus 81.9 +/- 4.3 unfiltered -- a difference far
inside the seed noise.

Pairs with create_third_stage_dataset_for_treatment_pcl_net_ate in
alternative_treatment_neural_mean_embedding.py, which does the same for the treatment-only third stage,
so a notebook can use one consistent unfiltered path end to end.
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, Tuple

from tqdm import tqdm

import sys
sys.path.append("..")
from torch_utils.scalers import TorchIdentityTransformer


def _create_third_stage_dataset_for_dr(
    outcome_pcl_model: nn.Module,
    treatment_pcl_model: nn.Module,
    pcl_dataloader: DataLoader,
    pcl_val_dataloader: Optional[DataLoader],
    outcome_transformer: Optional[nn.Module],
    input_transformer: Optional[nn.Module],
    input_type: str,
    treatment_featurizer: Optional[nn.Module],
    residual_target: bool,
    device: str,
) -> Tuple[TensorDataset, Optional[TensorDataset]]:
    """Shared body for the V1 and V2 creators; they differ only in the composite target.

    residual_target=True  -> varphi * (Y - h)   (V1)
    residual_target=False -> varphi *      h    (V2)
    """
    outcome_pcl_model.eval().to(device)
    treatment_pcl_model.eval().to(device)

    if not hasattr(treatment_pcl_model, "final_layer_second_stage_weight") or \
            treatment_pcl_model.final_layer_second_stage_weight is None:
        return None, None

    if outcome_transformer is None:
        outcome_transformer = TorchIdentityTransformer()
    if input_transformer is None:
        input_transformer = TorchIdentityTransformer()
    if treatment_featurizer is None:
        treatment_featurizer = torch.nn.Identity()

    y_transformer = pcl_dataloader.dataset.transformers[1].to(device)
    dens_ratio_transformer = copy.deepcopy(pcl_dataloader.dataset.dens_ratio_transformer).to(device)

    def _process_single_loader(loader: DataLoader, desc_text: str):
        input_list = []
        composite_target_list = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc_text):
                batch = [b.to(device) for b in batch]
                treatment, outcome_scaled = batch[0], batch[1]
                treatment_proxy, outcome_proxy = batch[2], batch[3]
                # Same covariate block the bridge was trained against (None when there is no backdoor).
                backdoor = treatment_pcl_model._backdoor_from(batch)

                outcome_unscaled = y_transformer.inverse_transform(outcome_scaled)
                h_unscaled = y_transformer.inverse_transform(
                    outcome_pcl_model.predict_outcome_bridge_function(treatment, outcome_proxy, backdoor))
                varphi_unscaled = dens_ratio_transformer.inverse_transform(
                    treatment_pcl_model.predict_treatment_bridge_function(treatment, treatment_proxy, backdoor))

                if residual_target:
                    composite_target = varphi_unscaled * (outcome_unscaled - h_unscaled)   # V1
                else:
                    composite_target = varphi_unscaled * h_unscaled                        # V2

                treatment_stored = treatment_featurizer(treatment) if input_type == "features" else treatment
                input_list.append(treatment_stored.cpu())
                composite_target_list.append(composite_target.cpu())

        if not input_list:
            return None, None
        return torch.cat(input_list), torch.cat(composite_target_list)

    input_train, target_train = _process_single_loader(pcl_dataloader, "Creating Stage 3 Train DR Data")
    if input_train is None:
        raise ValueError("Empty training dataset.")

    input_train = input_transformer.fit_transform(input_train)
    target_train = outcome_transformer.fit_transform(target_train)

    train_dataset = TensorDataset(input_train, target_train)
    train_dataset.outcome_transformer = outcome_transformer
    train_dataset.input_transformer = input_transformer

    val_dataset = None
    if pcl_val_dataloader is not None:
        input_val, target_val = _process_single_loader(pcl_val_dataloader, "Creating Stage 3 Val DR Data")
        if input_val is not None:
            val_dataset = TensorDataset(input_transformer.transform(input_val),
                                        outcome_transformer.transform(target_val))

    return train_dataset, val_dataset


def create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_ate(
    outcome_pcl_model: nn.Module,
    treatment_pcl_model: nn.Module,
    pcl_dataloader: DataLoader,
    pcl_val_dataloader: Optional[DataLoader] = None,
    outcome_transformer: Optional[nn.Module] = None,
    input_transformer: Optional[nn.Module] = None,
    input_type: str = "raw",
    treatment_featurizer: Optional[nn.Module] = None,
    device: str = "cuda",
) -> Tuple[TensorDataset, Optional[TensorDataset]]:
    """DR-V1 slack dataset, unfiltered: target varphi(A,Z,X) * (Y - h(A,W,X)), regressed on A.

    The correction acts on the RESIDUAL Y - h, so it is small whenever the outcome bridge is good and
    the estimator degrades gracefully toward f_om -- the robust variant.
    """
    return _create_third_stage_dataset_for_dr(
        outcome_pcl_model, treatment_pcl_model, pcl_dataloader, pcl_val_dataloader,
        outcome_transformer, input_transformer, input_type, treatment_featurizer,
        residual_target=True, device=device)


def create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_ate_v2(
    outcome_pcl_model: nn.Module,
    treatment_pcl_model: nn.Module,
    pcl_dataloader: DataLoader,
    pcl_val_dataloader: Optional[DataLoader] = None,
    outcome_transformer: Optional[nn.Module] = None,
    input_transformer: Optional[nn.Module] = None,
    input_type: str = "raw",
    treatment_featurizer: Optional[nn.Module] = None,
    device: str = "cuda",
) -> Tuple[TensorDataset, Optional[TensorDataset]]:
    """DR-V2 slack dataset, unfiltered: target varphi(A,Z,X) * h(A,W,X), regressed on A.

    Used as f_dr2 = f_om + f_tm - slack_2, the canonical AIPW form. Note this correction is a DIFFERENCE
    of two separately-regressed quantities, so its noise only cancels when the slack net is matched to
    the f_tm third-stage net (as observed on the abortion benchmark).
    """
    return _create_third_stage_dataset_for_dr(
        outcome_pcl_model, treatment_pcl_model, pcl_dataloader, pcl_val_dataloader,
        outcome_transformer, input_transformer, input_type, treatment_featurizer,
        residual_target=False, device=device)
