"""Density-ratio-FREE CATE from a single density-ratio-free ATE bridge.

Implements Chapter 21 of the PhD-upgrade audit ("Density-ratio-free CATE"). The point of that chapter
is that NO CATE-specific bridge is needed: fit one density-ratio-free ATE bridge on ALL covariates
X = (S, V) and recover the heterogeneous response by SELF-NORMALIZING it.

Write phi^ATE for the fitted ATE bridge, which satisfies (eq. 21.2)
    E[ phi^ATE(a, S, V, Z) | A = a, S, V, W ] = p_A(a) / p_{A|S,V,W}(a | S, V, W).
Its conditional normalizer (eq. 21.3-21.4) is
    D_0(a, v) := E[ phi^ATE(a, S, v, Z) | A = a, V = v ] = p_A(a) / p_{A|V}(a | v),
so phi^CATE = phi^ATE / D_0 is a valid CATE bridge (eq. 21.5) and (eq. 21.6)

    f_CATE(a, v) = E[ Y phi^ATE(a, S, v, Z) | A = a, V = v ]
                 / E[   phi^ATE(a, S, v, Z) | A = a, V = v ] .

WHY THE DENOMINATOR IS NOT OPTIONAL. The density-ratio CATE code in this repo
(create_third_stage_dataset_for_treatment_pcl_net_cate) regresses only the numerator, and is correct
only because HeterogeneousCausalKDEDensRatioTorch estimates the V-CONDITIONAL ratio
p(A|V)/p(A|W,V), for which E[phi | A=a, V=v] = 1 identically. The ATE bridge here targets the
MARGINAL-in-A ratio p(A)/p(A|W,X), so E[phi^ATE | A=a, V=v] = p(a)/p(a|v) != 1. Both the numerator and
the denominator carry that same factor, and only their ratio cancels it. Dropping the denominator would
leave the estimate multiplied by p(a)/p(a|v) -- precisely the confounding-strength-vs-V distortion the
estimator exists to remove.

The two estimators built here (algorithm 21.1) differ ONLY in the numerator label:
    treatment-only (eq. 21.6):   N_i = Y_i * phi_i          ->  f_CATE = N_hat / D_hat
    self-normalized DR (21.10):  K_i = phi_i * (Y_i - h_i)  ->  f_CATE = M_h + K_hat / D_hat
with D_i = phi_i in both cases. Both labels are regressed on the SAME input (A_i, V_i) by a SHARED
two-output network, which is what makes the ratio well posed: N (or K) and D must come from the same
rows of the same pass.
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Optional, Tuple

from tqdm import tqdm

import sys
sys.path.append("..")
from torch_utils.scalers import TorchIdentityTransformer


def create_third_stage_dataset_for_alternative_treatment_cate(
    pcl_model: nn.Module,
    pcl_dataloader: DataLoader,
    pcl_val_dataloader: Optional[DataLoader] = None,
    outcome_transformer: Optional[nn.Module] = TorchIdentityTransformer(),
    input_transformer: Optional[nn.Module] = TorchIdentityTransformer(),
    outcome_pcl_model: Optional[nn.Module] = None,
    numerator: str = "treatment",
    v_cols: Optional[List[int]] = None,
    device: str = "cuda",
) -> Tuple[TensorDataset, Optional[TensorDataset]]:
    """Third-stage dataset for the density-ratio-free CATE: input (A, V), target (numerator, phi).

    The regression problem is the SHARED two-output conditional mean
        [ N_hat(a, v), D_hat(a, v) ] = E[ (N_i, D_i) | A = a, V = v ]
    from which the caller forms N_hat / clip(D_hat) via `predict_cate_from_ratio_head`.

    numerator:
      "treatment" -> N_i = Y_i * phi_i                      (eq. 21.6, treatment-only CATE)
      "dr"        -> K_i = phi_i * (Y_i - h_i)              (eq. 21.10 numerator; needs outcome_pcl_model)
    The denominator label is D_i = phi_i in both cases, built in the SAME pass so that the two columns
    always describe the same rows -- which is what makes the self-normalized ratio well posed.

    NO inlier filtering. Every row is kept: the numerator and the denominator carry the same
    phi, so a common distortion cancels in the ratio, and selection on a function of (A, W, X) would be
    correlated with V and could thin specific (a, v) cells.

    Batches come from a HeterogeneousProxyDataset: (A, Y, Z, W, V, [X], dens_ratio), so slot 4 is the
    covariate block that the density-ratio-free bridge was ALSO trained against as its backdoor. To
    guarantee the creator conditions the bridge on exactly what training used, the backdoor is taken
    from `pcl_model._backdoor_from(batch)` rather than re-derived here. NOTE there is NO concatenation
    step: the bridge's `backdoor` argument IS that whole block. In the synthetic CATE benchmark S is
    empty, so the block is exactly V and `backdoor_in` is literally V.

    WHERE S GOES. Chapter 21 fits ONE bridge over ALL of X = (S, V), so S must live in the SAME slot the
    bridge reads -- slot 4 -- with `v_cols` selecting the effect-modifier columns for the third-stage
    input. Putting S in the dataset's separate X slot instead would leave it in slot 5, which
    `_backdoor_from` never reads: training and this creator would agree with each other but would both
    condition on V alone, quietly violating eq. (21.2). That configuration is rejected below rather than
    silently mis-fitted.
    """
    if numerator not in ("treatment", "dr"):
        raise ValueError(f"numerator must be 'treatment' or 'dr', got {numerator!r}")
    if numerator == "dr" and outcome_pcl_model is None:
        raise ValueError("numerator='dr' needs outcome_pcl_model to evaluate h(a, v, w).")

    # Guard the one configuration that would be wrong but silent: a covariate parked in the dataset's
    # separate X slot (slot 5) is invisible to the bridge, which only ever reads slot 4.
    base_dataset = pcl_dataloader.dataset
    if (getattr(base_dataset, "V", None) is not None) and (getattr(base_dataset, "X", None) is not None):
        raise ValueError(
            "This dataset fills BOTH the V slot and the X slot. The density-ratio-free bridge reads only "
            "slot 4, so the X slot would be silently dropped and the bridge would condition on V alone -- "
            "violating eq. (21.2), which requires the bridge to see all of X = (S, V). Build the dataset "
            "with the FULL covariate block (S and V concatenated) in the V slot, and pass v_cols to select "
            "the effect-modifier columns for the third-stage input."
        )

    pcl_model.eval().to(device)
    if outcome_pcl_model is not None:
        outcome_pcl_model.eval().to(device)

    if not hasattr(pcl_model, "final_layer_second_stage_weight") or pcl_model.final_layer_second_stage_weight is None:
        return None, None

    y_transformer = pcl_dataloader.dataset.transformers[1].to(device)
    dens_ratio_transformer = copy.deepcopy(pcl_dataloader.dataset.dens_ratio_transformer).to(device)

    def _process_single_loader(loader, desc_text):
        input_list = []
        target_list = []
        n_rows = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc_text):
                batch = [b.to(device) for b in batch]
                A_batch, Y_batch_scaled, Z_batch, W_batch = batch[0], batch[1], batch[2], batch[3]
                # The covariate block the bridge was trained on (X = (S, V)); slot 4 of the batch.
                backdoor_batch = pcl_model._backdoor_from(batch)
                V_full = backdoor_batch if backdoor_batch is not None else batch[4]
                n_rows += A_batch.shape[0]

                A_in, W_in, Z_in = A_batch, W_batch, Z_batch
                V_in_full = V_full
                backdoor_in = backdoor_batch

                Y_unscaled = y_transformer.inverse_transform(Y_batch_scaled)
                # The ATE bridge, 3-arg density-ratio-free signature: varphi(A, Z, X).
                varphi = pcl_model.predict_treatment_bridge_function(A_in, Z_in, backdoor_in)
                varphi_unscaled = dens_ratio_transformer.inverse_transform(varphi)

                if numerator == "treatment":
                    numerator_label = Y_unscaled * varphi_unscaled                 # N_i = Y_i * phi_i
                else:
                    h_unscaled = y_transformer.inverse_transform(
                        outcome_pcl_model.predict_outcome_bridge_function(A_in, V_in_full, W_in, None))
                    numerator_label = varphi_unscaled * (Y_unscaled - h_unscaled)  # K_i = phi_i (Y_i - h_i)

                # (n, 2): the numerator and the normalizer, from the SAME rows.
                target_list.append(torch.hstack([numerator_label, varphi_unscaled]).cpu())
                V_in = V_in_full if v_cols is None else V_in_full[:, v_cols]
                input_list.append(torch.hstack([A_in, V_in]).cpu())

        if len(input_list) == 0:
            return None, None, 0
        return torch.cat(input_list), torch.cat(target_list), n_rows

    input_all, target_all, n_rows = _process_single_loader(pcl_dataloader, "Creating Stage 3 CATE Dataset")
    if input_all is None:
        raise ValueError("Empty training set.")
    print(f"[CATE stage 3 | numerator={numerator}] {n_rows} rows "
          f"| input dim {input_all.shape[1]} | target dim {target_all.shape[1]}")

    input_all = input_transformer.fit_transform(input_all)
    target_all = outcome_transformer.fit_transform(target_all)

    third_stage_dataset = TensorDataset(input_all, target_all)
    third_stage_dataset.outcome_transformer = outcome_transformer
    third_stage_dataset.input_transformer = input_transformer

    third_stage_dataset_val = None
    if pcl_val_dataloader is not None:
        input_val, target_val, _ = _process_single_loader(pcl_val_dataloader, "Creating Stage 3 CATE Val Dataset")
        if input_val is not None:
            input_val = input_transformer.transform(input_val)
            target_val = outcome_transformer.transform(target_val)
            third_stage_dataset_val = TensorDataset(input_val, target_val)

    return third_stage_dataset, third_stage_dataset_val


def predict_cate_from_ratio_head(
    third_stage_dataset: TensorDataset,
    third_stage_net: nn.Module,
    treatment_scaled: torch.Tensor,
    covariate_scaled: torch.Tensor,
    denominator_floor: float = 0.1,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Evaluate the self-normalized ratio on an (a, v) grid.

    CRITICAL ORDER: the two-output target was standardized PER COLUMN (TorchStandardScaler keeps a
    separate mean/scale for the numerator and the normalizer), so the network output must be
    inverse-transformed BEFORE the division. Dividing the standardized columns is meaningless.

    Returns (ratio, denominator, clipped_fraction). `clipped_fraction` is the share of grid points whose
    estimated denominator fell below `denominator_floor` -- per section 21.6 the clip encodes the overlap
    region on which the ratio is stable, so it should be reported, not hidden.
    """
    third_stage_net.eval()
    with torch.no_grad():
        grid = torch.hstack([treatment_scaled.to(device), covariate_scaled.to(device)])
        grid = third_stage_dataset.input_transformer.to(device).transform(grid)
        out = third_stage_dataset.outcome_transformer.to(device).inverse_transform(third_stage_net(grid))
        numerator, denominator = out[:, :1], out[:, 1:2]
        clipped_fraction = float((denominator < denominator_floor).float().mean())
        ratio = numerator / denominator.clamp_min(denominator_floor)
    return ratio, denominator, clipped_fraction
