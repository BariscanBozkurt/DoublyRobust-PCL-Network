"""Sample-average third stage for a DISCRETE treatment.

The third stage of every estimator in this repo is a conditional mean in the treatment,

    f_tm(a)   = E[ varphi(A,Z,X) * Y            | A = a ]
    slack_1(a)= E[ varphi(A,Z,X) * (Y - h)      | A = a ]        ->  f_dr1 = f_om + slack_1
    slack_2(a)= E[ varphi(A,Z,X) *      h       | A = a ]        ->  f_dr2 = f_om + f_tm - slack_2

and everywhere else in the repo it is fitted by an ensemble MLP (`ConditionalMeanMLP` via
`train_third_stage_treatment_pcl_net_ensemble`), because A is continuous and the conditional mean has
to be interpolated between observed treatment values.

When A is DISCRETE that regression is unnecessary. Conditioning on A = a is just selecting the rows with
A_i = a, so the conditional mean is the plain sample average over that group,

    f_hat(a) = (1 / n_a) * sum_{i : A_i = a} target_i ,

which is the nonparametric MLE, is unbiased for the group mean, and has no learning rate, no epoch
count, no ensemble and no validation split to get wrong. It also removes a genuinely awkward regression:
in the ColorMNIST / OCTMNIST experiments the outcome is an IMAGE, so the network is asked to map a
single binary input to a 784- or 2352-dimensional target -- a map with exactly two distinct input
values and hundreds of thousands of parameters, fitted from two groups of rows.

A consequence worth knowing (asserted by `check_dr_variants_agree`): under this estimator the two DR
variants are ALGEBRAICALLY IDENTICAL. Since the same rows enter all three averages,

    f_om + f_tm - slack_2 = f_om + mean_a[varphi*Y] - mean_a[varphi*h]
                          = f_om + mean_a[varphi*(Y - h)] = f_om + slack_1 .

They differ in the neural version only because f_tm and slack_2 are fitted by two separate networks
whose errors do not cancel -- which is exactly the failure mode observed on the abortion benchmark
("V2's correction is a difference of two separately-regressed quantities"). The sample-average form
makes that failure mode structurally impossible.

The composite targets are built to match `create_third_stage_dataset_for_treatment_pcl_net_ate` and
`_create_third_stage_dataset_for_dr` row for row, including the transformer handling (Y and h are
inverse-transformed to their natural scale; varphi goes through the dens-ratio transformer), so the two
third stages are a controlled comparison on identical pseudo-outcomes.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Tuple

from tqdm import tqdm

import sys
sys.path.append("..")
from torch_utils.scalers import TorchIdentityTransformer


def assign_rows_to_treatment_levels(
    treatment: torch.Tensor,
    treatment_levels: torch.Tensor,
    tolerance: Optional[float] = 1e-4,
) -> torch.Tensor:
    """Index of the nearest row of `treatment_levels` for every row of `treatment`.

    `treatment_levels` is the evaluation grid (`do_A`) on the SAME scale as the data, i.e. already put
    through the treatment transformer. With a genuinely discrete treatment every observation coincides
    with a grid point, so `tolerance` is a guard rather than a knob: if some row sits further than
    `tolerance` from every level, the treatment is not discrete on this grid and a sample average is the
    wrong estimator. Pass tolerance=None to skip the check.
    """
    distances = torch.cdist(treatment.reshape(treatment.shape[0], -1).float(),
                            treatment_levels.reshape(treatment_levels.shape[0], -1).float())
    nearest = distances.argmin(dim=1)
    if tolerance is not None:
        worst = float(distances.gather(1, nearest.unsqueeze(1)).max())
        if worst > tolerance:
            raise ValueError(
                "treatment is not discrete on this grid: the worst row sits %.6g from its nearest "
                "level (tolerance %.6g). Use the neural third stage instead." % (worst, tolerance))
    return nearest


def _group_average(
    loader: DataLoader,
    targets_fn,
    treatment_levels: torch.Tensor,
    device: str,
    tolerance: Optional[float],
    desc: str,
) -> Tuple[dict, np.ndarray]:
    """One pass over `loader`, accumulating a per-level sum of every target `targets_fn` returns.

    `targets_fn(batch)` returns a dict of name -> (N, dim) tensor, so the bridge and the outcome model
    are evaluated ONCE per batch and every target is accumulated over identical rows -- which is what
    makes the two DR variants coincide exactly rather than approximately.

    Sums are accumulated in float64: an image target has thousands of columns and tens of thousands of
    rows, and float32 accumulation loses low-order bits at that length.
    """
    n_levels = treatment_levels.shape[0]
    sums = {}
    counts = torch.zeros(n_levels, dtype=torch.float64, device=device)

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            batch = [b.to(device) for b in batch]
            level_index = assign_rows_to_treatment_levels(batch[0], treatment_levels, tolerance)
            counts.index_add_(0, level_index,
                              torch.ones(level_index.shape[0], dtype=torch.float64, device=device))
            for name, target in targets_fn(batch).items():
                target = target.to(torch.float64)
                if name not in sums:
                    sums[name] = torch.zeros(n_levels, target.shape[1],
                                             dtype=torch.float64, device=device)
                sums[name].index_add_(0, level_index, target)

    if bool((counts == 0).any()):
        raise ValueError("no observations at treatment level(s) %s -- the group mean is undefined there"
                         % np.flatnonzero(counts.cpu().numpy() == 0).tolist())

    averages = {name: (total / counts.unsqueeze(1)).cpu().numpy().astype(np.float32)
                for name, total in sums.items()}
    return averages, counts.cpu().numpy()


def estimate_structural_functions_by_group_average(
    treatment_pcl_model: nn.Module,
    pcl_dataloader: DataLoader,
    treatment_levels: torch.Tensor,
    outcome_pcl_model: Optional[nn.Module] = None,
    device: str = "cuda",
    tolerance: Optional[float] = 1e-4,
    hajek: bool = False,
    desc: str = "Stage 3 group average",
) -> dict:
    """Sample-average third stage. Returns f_tm, and the DR slacks when an outcome model is given.

    Parameters mirror the dataset creators this replaces: `pcl_dataloader` supplies the rows the
    conditional mean is taken over (the stage-2 split in every notebook here), and `treatment_levels`
    is the evaluation grid on the data's scale (`do_A` after the treatment transformer).

    Returns a dict with
        "f_tm"        E[varphi Y | A = a]                          (n_levels, outcome_dim)
        "slack_v1"    E[varphi (Y - h) | A = a]                    -- only if outcome_pcl_model given
        "slack_v2"    E[varphi h | A = a]                          -- only if outcome_pcl_model given
        "phi_mean"    E[varphi | A = a]                            -- the normalization diagnostic
        "counts"      n_a per level
    """
    treatment_pcl_model.eval().to(device)
    if outcome_pcl_model is not None:
        outcome_pcl_model.eval().to(device)

    if not hasattr(treatment_pcl_model, "final_layer_second_stage_weight") or \
            treatment_pcl_model.final_layer_second_stage_weight is None:
        raise ValueError("the treatment bridge has no fitted second-stage head")

    y_transformer = pcl_dataloader.dataset.transformers[1].to(device)
    dens_ratio_transformer = copy.deepcopy(pcl_dataloader.dataset.dens_ratio_transformer).to(device)
    treatment_levels = treatment_levels.to(device)

    def targets_fn(batch):
        """Every composite target for one batch, on the natural scale the creators use."""
        treatment, outcome_scaled, treatment_proxy, outcome_proxy = batch[0], batch[1], batch[2], batch[3]
        backdoor = treatment_pcl_model._backdoor_from(batch)
        varphi = dens_ratio_transformer.inverse_transform(
            treatment_pcl_model.predict_treatment_bridge_function(treatment, treatment_proxy, backdoor))
        outcome = y_transformer.inverse_transform(outcome_scaled)

        targets = {"f_tm": varphi * outcome, "phi_mean": varphi}
        if outcome_pcl_model is not None:
            bridge = y_transformer.inverse_transform(
                outcome_pcl_model.predict_outcome_bridge_function(treatment, outcome_proxy, backdoor))
            targets["slack_v1"] = varphi * (outcome - bridge)     # V1
            targets["slack_v2"] = varphi * bridge                 # V2
        return targets

    averages, counts = _group_average(pcl_dataloader, targets_fn, treatment_levels,
                                      device, tolerance, desc)
    averages["counts"] = counts
    if hajek:
        # HAJEK (self-normalised) form: divide each level's average by that level's
        # REALISED mean of varphi on this loader,
        #     f(a) = sum_{i:A_i=a} varphi_i y_i / sum_{i:A_i=a} varphi_i,
        # instead of the Horvitz-Thompson form that trusts E[varphi | A=a] = 1. Any normalisation of
        # varphi -- trained-in constraint or post-hoc -- holds exactly only on the rows it was computed
        # on; on HELD-OUT rows the per-level mean drifts to some m(a) != 1, and the plain average is
        # then m(a) times the correct value. With an outcome on the scale of RHC's t3d30 (~23 days) a
        # 5-10% drift costs 1-2 days of ATE, the size of the effect itself. Dividing by the realised
        # mean cancels the drift EXACTLY, whatever its source; it uses only (A, Z, X), never Y, so it
        # is legitimate on a cross-fitting estimation fold. All three targets share the same per-level
        # divisor, so DR-V1 == DR-V2 is preserved.
        m = averages["phi_mean"].reshape(-1, 1)
        if (m <= 0).any():
            raise ValueError("Hajek normalisation undefined: mean varphi <= 0 at level(s) %s"
                             % np.flatnonzero((m <= 0).reshape(-1)).tolist())
        for key in ("f_tm", "slack_v1", "slack_v2"):
            if key in averages:
                averages[key] = averages[key] / m
    return averages


def check_dr_variants_agree(f_om: np.ndarray, group_averages: dict, atol: float = 1e-5) -> float:
    """Assert f_om + slack_v1 == f_om + f_tm - slack_v2 and return the largest absolute gap.

    This holds by construction for the sample-average third stage (see the module docstring), so a
    nonzero gap means the three averages did NOT run over the same rows -- e.g. a shuffled loader was
    consumed twice, or the bridge was re-normalized in between.
    """
    v1 = f_om + group_averages["slack_v1"]
    v2 = f_om + group_averages["f_tm"] - group_averages["slack_v2"]
    gap = float(np.abs(v1 - v2).max())
    if gap > atol:
        raise AssertionError(
            "DR-V1 and DR-V2 should coincide exactly under the sample-average third stage, but differ "
            "by %.3g -- the three group averages did not run over the same rows." % gap)
    return gap
