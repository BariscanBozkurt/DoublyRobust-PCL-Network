"""Scenario O: exact outcome-bridge amplification under correct specification.

This is the direct oracle-level mirror of Scenario T in
``2-ScenarioB_CompleteMoment_TreatmentDegradation.ipynb``.  The treatment law,
the outcome law, and the treatment proxy Z are fixed.  Only the reliability of
the S-coordinate of W changes.  Consequently,

* the exact outcome bridge and its evaluation variance depend on ``lambda_ws``;
* the exact treatment bridge and its oracle weighting variance do not;
* both bridge equations remain correctly specified.

The neural pipeline intentionally matches Scenario T: one density-ratio-free
TreatmentNet shared across the three response coordinates, three scalar
OutcomeNets, the same overlapping stage-1/stage-2 split, and held-out arm
averages.  Run one (lambda_ws, seed) cell per process so paired cells can be
parallelized without concurrent writes to one CSV.

Example
-------
python \
    scenario_o_outcome_bridge_amplification.py \
    --lambda_ws 0.9 --data_seed 1 --n_sample 4000
"""

import argparse
import copy
import csv
import hashlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from neural_causal_learning.alternative_treatment_neural_mean_embedding import (
    AlternativeTreatmentBridgePCLNET,
    train_alternative_treatment_pcl_net_ate_model,
)
from neural_causal_learning.proxy_outcome_neural_mean_embedding import (
    OutcomeBridgePCLNET,
    train_deep_feature_proxy_closed_form_ate_model,
)
from python_utils.helpers import slice_tuple
from simulation_utils.alternative_treatment_pcl_net_configs import (
    AlternativeTreatmentPCLConfig,
)
from simulation_utils.alternative_treatment_pcl_net_optimizers import (
    build_adam_optimizers_for_alternative_treatment_pcl_net,
)
from simulation_utils.outcome_pcl_net_configs import OutcomePCLConfig
from simulation_utils.outcome_pcl_net_optimizers import (
    build_adam_optimizers_for_outcome_pcl_net,
)
from simulation_utils.treatment_pcl_net_nn_structures import (
    build_alternative_treatment_nets,
)
from torch_utils.dataset_classes import ProxyDataset
from torch_utils.scalers import TorchIdentityTransformer, TorchStandardScaler


KAPPA_C = 0.1
KAPPA_S = 0.2
KAPPA_CS = 0.1
BETA = np.array([1.0, 1.0, 1.0])
SIGMA_Y = 0.25
LAM_ZC = 0.8
LAM_ZS = 0.6
LAM_WC = 0.8
DO_A = np.array([-1.0, 1.0])
LATENT_STATES = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda_ws", type=float, default=0.9)
    parser.add_argument("--data_seed", type=int, default=1)
    parser.add_argument(
        "--train_seed",
        type=int,
        default=-1,
        help="network/split seed; -1 uses data_seed, matching Scenario T",
    )
    parser.add_argument("--n_sample", type=int, default=4000)
    parser.add_argument("--om_epochs", type=int, default=100)
    parser.add_argument("--tm_epochs", type=int, default=100)
    parser.add_argument("--val_fraction", type=float, default=0.25)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(
            REPO, "Simulations", "Results", "BridgeRoutePreference", "scenarioO"
        ),
    )
    parser.add_argument("--tag", type=str, default="neural_nme")
    parser.add_argument(
        "--no_write",
        action="store_true",
        help="run and print diagnostics without creating a result CSV",
    )
    return parser


def resolve_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def f_true(a):
    a = np.asarray(a, dtype=float)
    return np.stack([a, a, a], axis=-1)


EY_DO_A = f_true(DO_A)


def eta(c, s):
    return KAPPA_C * c + KAPPA_S * s + KAPPA_CS * c * s


def walsh_coefficients(a):
    """Coefficients of r_a(c,s) = 1 / {1 + a eta(c,s)}."""
    weights = {
        state: 1.0 / (1.0 + a * eta(state[0], state[1]))
        for state in LATENT_STATES
    }
    characters = {
        "0": lambda c, s: 1.0,
        "C": lambda c, s: c,
        "S": lambda c, s: s,
        "CS": lambda c, s: c * s,
    }
    return {
        key: 0.25
        * sum(weights[state] * character(*state) for state in LATENT_STATES)
        for key, character in characters.items()
    }


def phi0(a, z_c, z_s):
    """Exact treatment bridge; deliberately independent of lambda_ws."""
    d = walsh_coefficients(a)
    return (
        d["0"]
        + d["C"] * z_c / LAM_ZC
        + d["S"] * z_s / LAM_ZS
        + d["CS"] * z_c * z_s / (LAM_ZC * LAM_ZS)
    )


def h0(a, w_c, w_s, lambda_ws):
    """Exact three-coordinate outcome bridge."""
    target = f_true(a)
    return np.stack(
        [
            target[..., 0] + BETA[0] * w_c / LAM_WC,
            target[..., 1] + BETA[1] * w_s / lambda_ws,
            target[..., 2]
            + BETA[2] * w_c * w_s / (LAM_WC * lambda_ws),
        ],
        axis=-1,
    )


def bit_flip(latent, reliability, uniforms):
    sign = np.where(uniforms < (1.0 + reliability) / 2.0, 1.0, -1.0)
    return latent * sign


def generate_data(n, lambda_ws, data_seed):
    """Generate paired data; only W_S changes when lambda_ws changes."""
    rng = np.random.RandomState(1000 + data_seed)
    latent_index = rng.randint(0, 4, size=n)
    c = np.array([LATENT_STATES[i][0] for i in latent_index], dtype=float)
    s = np.array([LATENT_STATES[i][1] for i in latent_index], dtype=float)

    treatment_uniform = rng.rand(n)
    a = np.where(treatment_uniform < (1.0 + eta(c, s)) / 2.0, 1.0, -1.0)
    epsilon = SIGMA_Y * rng.randn(n, 3)
    y = f_true(a) + np.stack([c, s, c * s], axis=-1) + epsilon

    z_c = bit_flip(c, LAM_ZC, rng.rand(n))
    z_s = bit_flip(s, LAM_ZS, rng.rand(n))
    w_c = bit_flip(c, LAM_WC, rng.rand(n))
    w_s = bit_flip(s, lambda_ws, rng.rand(n))
    return (
        a.reshape(-1, 1),
        y,
        np.stack([z_c, z_s], axis=1),
        np.stack([w_c, w_s], axis=1),
    )


def proxy_kernel(lambda_c, lambda_s):
    kernel = np.zeros((4, 4))
    for i, (c, s) in enumerate(LATENT_STATES):
        for j, (v_c, v_s) in enumerate(LATENT_STATES):
            kernel[i, j] = (
                (1.0 + lambda_c * v_c * c)
                * (1.0 + lambda_s * v_s * s)
                / 4.0
            )
    return kernel


def oracle_diagnostics(lambda_ws):
    """Exact moments and standardized conditional-expectation gains."""
    by_arm = {}
    k_z = proxy_kernel(LAM_ZC, LAM_ZS)
    k_w = proxy_kernel(LAM_WC, lambda_ws)
    for a in (-1, 1):
        p_cs = np.array(
            [(1.0 + a * eta(c, s)) / 4.0 for c, s in LATENT_STATES]
        )
        e_phi = 0.0
        lambda_phi = 0.0
        for i, _state in enumerate(LATENT_STATES):
            for j, (z_c, z_s) in enumerate(LATENT_STATES):
                probability = p_cs[i] * k_z[i, j]
                bridge_value = phi0(a, z_c, z_s)
                e_phi += probability * bridge_value
                lambda_phi += probability * bridge_value**2

        joint_zw = k_z.T @ np.diag(p_cs) @ k_w
        d_z = np.diag(1.0 / np.sqrt(joint_zw.sum(axis=1)))
        d_w = np.diag(1.0 / np.sqrt(joint_zw.sum(axis=0)))
        standardized = d_z @ joint_zw @ d_w
        singular_values = np.linalg.svd(standardized, compute_uv=False)
        by_arm[a] = {
            "E_phi": float(e_phi),
            "Lambda_phi": float(lambda_phi),
            "smallest_gain": float(singular_values[-1]),
        }

    v_h = (
        BETA[0] ** 2 / LAM_WC**2
        + BETA[1] ** 2 / lambda_ws**2
        + BETA[2] ** 2 / (LAM_WC**2 * lambda_ws**2)
    )
    for arm in by_arm:
        by_arm[arm]["V_phi"] = 2.0 * by_arm[arm]["Lambda_phi"] * (
            np.sum(BETA**2) + 3.0 * SIGMA_Y**2
        )
    return v_h, by_arm


def verify_oracles(lambda_ws):
    """Exact finite-state checks; raises before neural training on any mismatch."""
    if not 0.0 < lambda_ws <= 1.0:
        raise ValueError("lambda_ws must lie in (0, 1]")
    if abs(KAPPA_C) + abs(KAPPA_S) + abs(KAPPA_CS) >= 1.0:
        raise ValueError("treatment mechanism violates strict overlap")

    v_h, diagnostics = oracle_diagnostics(lambda_ws)
    expected_v_h = 1.5625 + 2.5625 / lambda_ws**2
    if not np.isclose(v_h, expected_v_h, atol=1e-12):
        raise AssertionError((v_h, expected_v_h))
    for arm in (-1, 1):
        if not np.isclose(diagnostics[arm]["E_phi"], 1.0, atol=1e-12):
            raise AssertionError(diagnostics)
    return v_h, diagnostics


def build_outcome_nets(device, first_stage_input_dim, w_input_dim):
    first_stage_output = 128
    w_output = 16
    a_output = 8
    width = 128
    dropout = 0.05

    class FeatureNet(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width, 2 * width),
                nn.LayerNorm(2 * width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(2 * width, output_dim),
                nn.LayerNorm(output_dim),
            )
            self.activation = nn.GELU()

        def forward(self, value):
            return self.activation(self.net(value))

    return (
        FeatureNet(first_stage_input_dim, first_stage_output).to(device),
        FeatureNet(1, a_output).to(device),
        FeatureNet(w_input_dim, w_output).to(device),
    )


def train_outcome_model(dl_stage1, dl_stage2, dl_validation, device, epochs):
    config = OutcomePCLConfig.low_dim()
    config.n_epochs = epochs
    config.device = device
    first_stage, treatment_stage2, proxy_stage2 = build_outcome_nets(device, 3, 2)
    model = OutcomeBridgePCLNET(
        first_stage, treatment_stage2, proxy_stage2, None, None, device=device
    )
    optimizers1, optimizers2, schedulers1, schedulers2 = (
        build_adam_optimizers_for_outcome_pcl_net(
            model,
            config.lr_first_stage,
            config.lr_second_stage_ax,
            config.lr_second_stage_w,
            config.weight_decay,
            gamma=config.scheduler_gamma,
        )
    )
    reg1, reg2_inner, reg2_outer = config.regularizers
    return train_deep_feature_proxy_closed_form_ate_model(
        pcl_model=model,
        first_stage_train_dataloader=dl_stage1,
        second_stage_train_dataloader=dl_stage2,
        stage1_optimizers=optimizers1,
        stage2_optimizers=optimizers2,
        stage1_schedulers=schedulers1,
        stage2_schedulers=schedulers2,
        n_epochs=epochs,
        stage1_iter=config.stage1_iter,
        stage2_iter=config.stage2_iter,
        first_stage_final_layer_regularizer=reg1,
        second_stage_final_layer_regularizer=reg2_outer,
        second_stage_first_final_layer_regularizer=reg2_inner,
        regularizer_annealing_method=config.reg_annealing_method,
        consider_prev_weight=config.consider_prev_weight,
        log_per_epoch=10**9,
        plot_loss=False,
        validation_dataloader=dl_validation,
        do_A=torch.tensor(DO_A.reshape(-1, 1), dtype=torch.float32),
        EY_do_A=torch.tensor(EY_DO_A[:, 0:1], dtype=torch.float32),
    )


def normalize_treatment_bridge(model, a_raw, stage2_dataset, a_transformer, device):
    """Use the same per-arm post-hoc normalization as Scenario T."""
    with torch.no_grad():
        values = model.predict_treatment_bridge_function(
            stage2_dataset.A.to(device), stage2_dataset.Z.to(device), None
        )
        values = values.cpu().numpy().reshape(-1)
    a_flat = a_raw.reshape(-1)
    raw_arm_mean = {
        arm: float(np.mean(values[a_flat == arm]))
        for arm in (-1.0, 1.0)
    }
    normalization_denominator = {
        arm: float(np.clip(raw_arm_mean[arm], 0.15, None))
        for arm in (-1.0, 1.0)
    }
    original_predict = model.predict_treatment_bridge_function
    a_transformer_cpu = copy.deepcopy(a_transformer).to("cpu")

    def normalized_predict(treatment, treatment_proxy, backdoor=None):
        output = original_predict(treatment, treatment_proxy, backdoor)
        treatment_raw = (
            a_transformer_cpu.inverse_transform(treatment.detach().cpu())
            .numpy()
            .reshape(-1)
        )
        denominator = np.where(
            treatment_raw >= 0.0,
            normalization_denominator[1.0],
            normalization_denominator[-1.0],
        )
        return output / torch.as_tensor(
            denominator, dtype=output.dtype, device=output.device
        ).reshape(output.shape)

    model.predict_treatment_bridge_function = normalized_predict
    return raw_arm_mean, normalization_denominator


def vector_mse(estimate):
    return float(np.mean((np.asarray(estimate) - EY_DO_A) ** 2))


def run_neural_cell(args, device, **kwargs):
    """Default protocol "single": one 75/25 split (the reported BridgeRoute grid), unchanged.
    Split "overlap" (default): stage 1 = first 75% of the training rows, stage 2 = last 75%, so the two
    stages share 50% of them. Split "independent" (opt-in): disjoint halves, no shared observation.
    Protocol "crossfit" (opt-in): three disjoint thirds of the sample, rotated so each third serves
    once as stage-1, stage-2 and estimation fold (no validation rows); the per-arm estimates of the three
    rotations are averaged and scored, as in the paper's other cross-fitted experiments."""
    if getattr(args, "protocol", "single") == "single":
        return _run_neural_cell_once(args, device, **kwargs)
    rotations = [_run_neural_cell_once(args, device, fold_rotation=r, **kwargs) for r in range(3)]
    metrics = {key: float(np.mean([m[key] for m in rotations])) for key, value in rotations[0].items()
               if isinstance(value, (int, float)) and not isinstance(value, bool)}
    metrics["seconds"] = float(sum(m["seconds"] for m in rotations))
    metrics["phi_prediction_sha256"] = rotations[0]["phi_prediction_sha256"]
    metrics["curves"] = {key: np.mean([np.asarray(m["curves"][key]) for m in rotations], axis=0).tolist()
                         for key in rotations[0]["curves"]}
    for key, curve in metrics["curves"].items():
        metrics[key] = vector_mse(curve)
    return metrics


def _run_neural_cell_once(
    args,
    device,
    generate_data_function=generate_data,
    outcome_bridge_function=h0,
    treatment_bridge_function=phi0,
    verify_oracles_function=verify_oracles,
    treatment_projection_function=None,
    fold_rotation=None,
):
    start = time.time()
    train_seed = args.data_seed if args.train_seed < 0 else args.train_seed
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    a, y, z, w = generate_data_function(
        args.n_sample, args.lambda_ws, args.data_seed
    )
    n = len(a)
    tensors = tuple(torch.tensor(value, dtype=torch.float32) for value in (a, y, z, w))
    a_tensor, y_tensor, _z_tensor, _w_tensor = tensors
    density_ratio = torch.ones(n, 1, dtype=torch.float32)
    density_transformer = TorchIdentityTransformer()
    a_transformer = TorchStandardScaler()
    z_transformer = TorchStandardScaler()
    w_transformer = TorchStandardScaler()
    y_transformers = [TorchStandardScaler() for _ in range(3)]

    permutation = np.random.permutation(n)
    if fold_rotation is None:
        n_train = int(n * (1.0 - args.val_fraction))
        train_index = permutation[:n_train]
        validation_index = permutation[n_train:]
    else:
        folds = np.array_split(permutation, 3)
        stage1_index = folds[fold_rotation % 3]
        stage2_index = folds[(fold_rotation + 1) % 3]
        validation_index = folds[(fold_rotation + 2) % 3]
        train_index = np.concatenate([stage1_index, stage2_index])
    if len(validation_index) == 0:
        raise ValueError("validation split is empty")
    a_transformer.fit(a_tensor[train_index])
    z_transformer.fit(tensors[2][train_index])
    w_transformer.fit(tensors[3][train_index])
    density_transformer.fit(density_ratio[train_index])
    for coordinate in range(3):
        y_transformers[coordinate].fit(
            y_tensor[train_index, coordinate : coordinate + 1]
        )

    if fold_rotation is None:
        if getattr(args, "split", "overlap") == "independent":
            # disjoint halves of the training rows: stage 1 and stage 2 share NO observation
            half = len(train_index) // 2
            stage1_index = train_index[:half]
            stage2_index = train_index[half:]
        else:
            stage1_index = train_index[: int(len(train_index) * 0.75)]
            stage2_index = train_index[-int(len(train_index) * 0.75) :]

    def make_dataset(index, y_coordinate):
        return ProxyDataset(
            slice_tuple(
                (
                    a_tensor,
                    y_tensor[:, y_coordinate : y_coordinate + 1],
                    tensors[2],
                    tensors[3],
                ),
                index,
            ),
            dens_ratio=density_ratio[index],
            dens_ratio_transformer=density_transformer,
            transformers=[
                a_transformer,
                y_transformers[y_coordinate],
                z_transformer,
                w_transformer,
            ],
            device="cpu",
        )

    batch_size = n
    stage1_tm = make_dataset(stage1_index, 0)
    stage2_tm = make_dataset(stage2_index, 0)
    validation_tm = make_dataset(validation_index, 0)
    treatment_config = AlternativeTreatmentPCLConfig.low_dim()
    treatment_config.device = device
    treatment_model = AlternativeTreatmentBridgePCLNET(
        *build_alternative_treatment_nets(
            device, z_dim=2, w_dim=2, a_dim=1
        ),
        device=device,
    )
    optimizers1, optimizers2, schedulers1, schedulers2 = (
        build_adam_optimizers_for_alternative_treatment_pcl_net(
            treatment_model,
            lr_first_stage=treatment_config.lr_first_stage,
            lr_second_stage_ax=treatment_config.lr_second_stage_ax,
            lr_second_stage_z=treatment_config.lr_second_stage_z,
            weight_decay=treatment_config.weight_decay,
            gamma=treatment_config.scheduler_gamma,
        )
    )
    treatment_model, _, _ = train_alternative_treatment_pcl_net_ate_model(
        model=treatment_model,
        first_stage_train_dataloader=DataLoader(
            stage1_tm, batch_size=batch_size, shuffle=True
        ),
        second_stage_train_dataloader=DataLoader(
            stage2_tm, batch_size=batch_size, shuffle=True
        ),
        stage1_optimizers=optimizers1,
        stage2_optimizers=optimizers2,
        stage1_schedulers=schedulers1,
        stage2_schedulers=schedulers2,
        n_epochs=args.tm_epochs,
        first_stage_final_layer_regularizer=treatment_config.reg_first_stage,
        second_stage_final_layer_regularizer=treatment_config.reg_second_stage,
        second_stage_first_final_layer_regularizer=(
            treatment_config.reg_second_stage_first
        ),
        validation_dataloader=DataLoader(
            validation_tm, batch_size=batch_size, shuffle=False
        ),
        log_per_epoch=10**9,
        plot_loss=False,
    )
    raw_arm_means, normalization_denominators = normalize_treatment_bridge(
        treatment_model,
        a[stage2_index],
        stage2_tm,
        a_transformer,
        device,
    )

    a_validation = a[validation_index].reshape(-1)
    y_validation = y[validation_index]
    z_validation = z[validation_index]
    w_validation = w[validation_index]
    with torch.no_grad():
        phi_validation = treatment_model.predict_treatment_bridge_function(
            validation_tm.A.to(device), validation_tm.Z.to(device), None
        ).cpu().numpy().reshape(-1)
    phi_oracle = treatment_bridge_function(
        a_validation, z_validation[:, 0], z_validation[:, 1]
    )
    with torch.no_grad():
        projected_phi_validation = treatment_model.predict_dens_ratio(
            validation_tm.A.to(device), validation_tm.W.to(device), None
        ).cpu().numpy().reshape(-1)
    projection_denominator = np.where(
        a_validation >= 0.0,
        normalization_denominators[1.0],
        normalization_denominators[-1.0],
    )
    projected_phi_validation = projected_phi_validation / projection_denominator
    if treatment_projection_function is None:
        projected_phi_oracle = np.full_like(projected_phi_validation, np.nan)
        projected_phi_mse = np.nan
    else:
        projected_phi_oracle = treatment_projection_function(
            a_validation,
            w_validation[:, 0],
            w_validation[:, 1],
            args.lambda_ws,
        )
        projected_phi_mse = float(
            np.mean((projected_phi_validation - projected_phi_oracle) ** 2)
        )

    a_transformer_device = copy.deepcopy(a_transformer).to(device)
    w_validation_scaled = validation_tm.W.to(device)
    y_transformers_device = [copy.deepcopy(item).to(device) for item in y_transformers]
    treatment_columns = {
        arm: a_transformer_device.transform(
            torch.full((len(validation_index), 1), arm, device=device)
        )
        for arm in (-1.0, 1.0)
    }

    f_outcome = np.zeros((2, 3))
    f_treatment = np.zeros((2, 3))
    f_dr = np.zeros((2, 3))
    f_dr_oracle_phi = np.zeros((2, 3))
    f_dr_oracle_h = np.zeros((2, 3))
    f_naive = np.zeros((2, 3))
    f_outcome_oracle = np.zeros((2, 3))
    f_treatment_oracle = np.zeros((2, 3))
    h_squared_error = np.zeros((2, 3))
    h_centered_second_moment = np.zeros((2, 3))

    for coordinate in range(3):
        outcome_model = train_outcome_model(
            DataLoader(
                make_dataset(stage1_index, coordinate),
                batch_size=batch_size,
                shuffle=True,
            ),
            DataLoader(
                make_dataset(stage2_index, coordinate),
                batch_size=batch_size,
                shuffle=True,
            ),
            DataLoader(
                make_dataset(validation_index, coordinate),
                batch_size=batch_size,
                shuffle=False,
            ),
            device,
            args.om_epochs,
        )
        for arm_index, arm in enumerate((-1.0, 1.0)):
            with torch.no_grad():
                h_prediction = y_transformers_device[coordinate].inverse_transform(
                    outcome_model.predict_outcome_bridge_function(
                        treatment_columns[arm], w_validation_scaled, None
                    )
                ).cpu().numpy().reshape(-1)

            arm_mask = a_validation == arm
            learned_denominator = np.sum(phi_validation[arm_mask])
            oracle_denominator = np.sum(phi_oracle[arm_mask])
            if abs(learned_denominator) < 1e-8 or abs(oracle_denominator) < 1e-8:
                raise FloatingPointError("treatment-weight denominator is numerically zero")

            f_outcome[arm_index, coordinate] = np.mean(h_prediction)
            f_treatment[arm_index, coordinate] = np.sum(
                phi_validation[arm_mask] * y_validation[arm_mask, coordinate]
            ) / learned_denominator
            f_dr[arm_index, coordinate] = f_outcome[arm_index, coordinate] + np.sum(
                phi_validation[arm_mask]
                * (y_validation[arm_mask, coordinate] - h_prediction[arm_mask])
            ) / learned_denominator
            f_naive[arm_index, coordinate] = np.mean(
                y_validation[arm_mask, coordinate]
            )

            h_oracle = outcome_bridge_function(
                arm,
                w_validation[:, 0],
                w_validation[:, 1],
                args.lambda_ws,
            )[:, coordinate]
            f_outcome_oracle[arm_index, coordinate] = np.mean(h_oracle)
            f_treatment_oracle[arm_index, coordinate] = np.sum(
                phi_oracle[arm_mask] * y_validation[arm_mask, coordinate]
            ) / oracle_denominator
            f_dr_oracle_phi[arm_index, coordinate] = f_outcome[
                arm_index, coordinate
            ] + np.sum(
                phi_oracle[arm_mask]
                * (y_validation[arm_mask, coordinate] - h_prediction[arm_mask])
            ) / oracle_denominator
            f_dr_oracle_h[arm_index, coordinate] = f_outcome_oracle[
                arm_index, coordinate
            ] + np.sum(
                phi_validation[arm_mask]
                * (y_validation[arm_mask, coordinate] - h_oracle[arm_mask])
            ) / learned_denominator
            h_squared_error[arm_index, coordinate] = np.mean(
                (h_prediction - h_oracle) ** 2
            )
            h_centered_second_moment[arm_index, coordinate] = np.mean(
                (h_prediction - EY_DO_A[arm_index, coordinate]) ** 2
            )

    phi_arm_means = {
        arm: float(np.mean(phi_validation[a_validation == arm]))
        for arm in (-1.0, 1.0)
    }
    phi_arm_second_moments = {
        arm: float(np.mean(phi_validation[a_validation == arm] ** 2))
        for arm in (-1.0, 1.0)
    }
    phi_oracle_arm_second_moments = {
        arm: float(np.mean(phi_oracle[a_validation == arm] ** 2))
        for arm in (-1.0, 1.0)
    }
    phi_arm_ess_fraction = {
        arm: float(
            np.sum(phi_validation[a_validation == arm]) ** 2
            / (
                np.sum(a_validation == arm)
                * np.sum(phi_validation[a_validation == arm] ** 2)
            )
        )
        for arm in (-1.0, 1.0)
    }
    phi_oracle_arm_ess_fraction = {
        arm: float(
            np.sum(phi_oracle[a_validation == arm]) ** 2
            / (
                np.sum(a_validation == arm)
                * np.sum(phi_oracle[a_validation == arm] ** 2)
            )
        )
        for arm in (-1.0, 1.0)
    }
    v_h, oracle = verify_oracles_function(args.lambda_ws)
    return {
        "OutcomeNet": vector_mse(f_outcome),
        "TreatmentNet": vector_mse(f_treatment),
        "DRPCLNET": vector_mse(f_dr),
        "DR_OraclePhi": vector_mse(f_dr_oracle_phi),
        "DR_OracleH": vector_mse(f_dr_oracle_h),
        "Naive": vector_mse(f_naive),
        "OracleOM": vector_mse(f_outcome_oracle),
        "OracleTM": vector_mse(f_treatment_oracle),
        "phi_mse": float(np.mean((phi_validation - phi_oracle) ** 2)),
        "phi_projected_mse": projected_phi_mse,
        "phi_projected_second_moment": float(
            np.mean(projected_phi_validation**2)
        ),
        "phi_projected_oracle_second_moment": float(
            np.mean(projected_phi_oracle**2)
        ),
        "phi_prediction_sha256": hashlib.sha256(
            np.ascontiguousarray(phi_validation).tobytes()
        ).hexdigest(),
        "phi_norm_max": max(abs(value - 1.0) for value in phi_arm_means.values()),
        "phi_mean_m1": phi_arm_means[-1.0],
        "phi_mean_p1": phi_arm_means[1.0],
        "phi_second_moment_m1": phi_arm_second_moments[-1.0],
        "phi_second_moment_p1": phi_arm_second_moments[1.0],
        "phi_oracle_second_moment_m1": phi_oracle_arm_second_moments[-1.0],
        "phi_oracle_second_moment_p1": phi_oracle_arm_second_moments[1.0],
        "phi_ess_fraction_m1": phi_arm_ess_fraction[-1.0],
        "phi_ess_fraction_p1": phi_arm_ess_fraction[1.0],
        "phi_oracle_ess_fraction_m1": phi_oracle_arm_ess_fraction[-1.0],
        "phi_oracle_ess_fraction_p1": phi_oracle_arm_ess_fraction[1.0],
        "phi_abs_max": float(np.max(np.abs(phi_validation))),
        "phi_abs_q95": float(np.quantile(np.abs(phi_validation), 0.95)),
        "phi_abs_q99": float(np.quantile(np.abs(phi_validation), 0.99)),
        "phi_raw_mean_m1": raw_arm_means[-1.0],
        "phi_raw_mean_p1": raw_arm_means[1.0],
        "phi_normalization_denominator_m1": normalization_denominators[-1.0],
        "phi_normalization_denominator_p1": normalization_denominators[1.0],
        "h_mse": float(np.mean(h_squared_error)),
        "h_centered_second_moment_per_coordinate": float(
            np.mean(h_centered_second_moment)
        ),
        "h_centered_second_moment_sum": float(
            np.mean(np.sum(h_centered_second_moment, axis=1))
        ),
        "h_mse_coord1": float(np.mean(h_squared_error[:, 0])),
        "h_mse_coord2": float(np.mean(h_squared_error[:, 1])),
        "h_mse_coord3": float(np.mean(h_squared_error[:, 2])),
        "V_h": v_h,
        "Lambda_phi_m1": oracle[-1]["Lambda_phi"],
        "Lambda_phi_p1": oracle[1]["Lambda_phi"],
        "V_phi_m1": oracle[-1]["V_phi"],
        "V_phi_p1": oracle[1]["V_phi"],
        "smallest_gain": min(
            oracle[-1]["smallest_gain"], oracle[1]["smallest_gain"]
        ),
        "seconds": time.time() - start,
        "curves": {
            "OutcomeNet": f_outcome.tolist(),
            "TreatmentNet": f_treatment.tolist(),
            "DRPCLNET": f_dr.tolist(),
            "DR_OraclePhi": f_dr_oracle_phi.tolist(),
            "DR_OracleH": f_dr_oracle_h.tolist(),
            "Naive": f_naive.tolist(),
            "OracleOM": f_outcome_oracle.tolist(),
            "OracleTM": f_treatment_oracle.tolist(),
        },
    }


def write_result(
    args,
    device,
    metrics,
    scenario="O",
    sweep_name="lambda_ws",
    file_prefix="scenarioO",
    file_sweep_label="lamws",
):
    os.makedirs(args.output_dir, exist_ok=True)
    train_seed = args.data_seed if args.train_seed < 0 else args.train_seed
    sweep_value = getattr(args, sweep_name)
    sweep_tag = f"{sweep_value:.3f}".replace(".", "p")
    output_path = os.path.join(
        args.output_dir,
        f"{file_prefix}_{file_sweep_label}{sweep_tag}_s{args.data_seed}_t{train_seed}_{args.tag}.csv",
    )
    algorithms = [
        "OutcomeNet",
        "TreatmentNet",
        "DRPCLNET",
        "DR_OraclePhi",
        "DR_OracleH",
        "Naive",
        "OracleOM",
        "OracleTM",
    ]
    shared = {
        "scenario": scenario,
        sweep_name: sweep_value,
        "data_seed": args.data_seed,
        "train_seed": train_seed,
        "n_sample": args.n_sample,
        "om_epochs": args.om_epochs,
        "tm_epochs": args.tm_epochs,
        "val_fraction": args.val_fraction,
        "device": str(device),
        "tag": args.tag,
        "V_h": metrics["V_h"],
        "Lambda_phi_m1": metrics["Lambda_phi_m1"],
        "Lambda_phi_p1": metrics["Lambda_phi_p1"],
        "V_phi_m1": metrics["V_phi_m1"],
        "V_phi_p1": metrics["V_phi_p1"],
        "smallest_gain": metrics["smallest_gain"],
        "phi_mse": metrics["phi_mse"],
        "phi_projected_mse": metrics["phi_projected_mse"],
        "phi_projected_second_moment": metrics[
            "phi_projected_second_moment"
        ],
        "phi_projected_oracle_second_moment": metrics[
            "phi_projected_oracle_second_moment"
        ],
        "phi_prediction_sha256": metrics["phi_prediction_sha256"],
        "phi_norm_max": metrics["phi_norm_max"],
        "phi_mean_m1": metrics["phi_mean_m1"],
        "phi_mean_p1": metrics["phi_mean_p1"],
        "phi_second_moment_m1": metrics["phi_second_moment_m1"],
        "phi_second_moment_p1": metrics["phi_second_moment_p1"],
        "phi_oracle_second_moment_m1": metrics["phi_oracle_second_moment_m1"],
        "phi_oracle_second_moment_p1": metrics["phi_oracle_second_moment_p1"],
        "phi_ess_fraction_m1": metrics["phi_ess_fraction_m1"],
        "phi_ess_fraction_p1": metrics["phi_ess_fraction_p1"],
        "phi_oracle_ess_fraction_m1": metrics["phi_oracle_ess_fraction_m1"],
        "phi_oracle_ess_fraction_p1": metrics["phi_oracle_ess_fraction_p1"],
        "phi_abs_max": metrics["phi_abs_max"],
        "phi_abs_q95": metrics["phi_abs_q95"],
        "phi_abs_q99": metrics["phi_abs_q99"],
        "phi_raw_mean_m1": metrics["phi_raw_mean_m1"],
        "phi_raw_mean_p1": metrics["phi_raw_mean_p1"],
        "phi_normalization_denominator_m1": metrics[
            "phi_normalization_denominator_m1"
        ],
        "phi_normalization_denominator_p1": metrics[
            "phi_normalization_denominator_p1"
        ],
        "h_mse": metrics["h_mse"],
        "h_centered_second_moment_per_coordinate": metrics[
            "h_centered_second_moment_per_coordinate"
        ],
        "h_centered_second_moment_sum": metrics[
            "h_centered_second_moment_sum"
        ],
        "h_mse_coord1": metrics["h_mse_coord1"],
        "h_mse_coord2": metrics["h_mse_coord2"],
        "h_mse_coord3": metrics["h_mse_coord3"],
        "seconds": metrics["seconds"],
    }
    if getattr(args, "protocol", "single") != "single":
        shared["protocol"] = args.protocol
    if getattr(args, "split", "overlap") != "overlap":
        shared["split"] = args.split
    if getattr(args, "sampling", "posterior") != "posterior":
        shared["sampling"] = args.sampling
    if "relevant_gain" in metrics:
        shared["relevant_gain"] = metrics["relevant_gain"]
    if "observed_input_sha256" in metrics:
        shared["observed_input_sha256"] = metrics["observed_input_sha256"]
    rows = []
    for algorithm in algorithms:
        curve = np.asarray(metrics["curves"][algorithm])
        rows.append(
            {
                **shared,
                "Algorithm": algorithm,
                "Causal_MSE": metrics[algorithm],
                "arm_m1_mse": float(np.mean((curve[0] - EY_DO_A[0]) ** 2)),
                "arm_p1_mse": float(np.mean((curve[1] - EY_DO_A[1]) ** 2)),
                "coord1_mse": float(np.mean((curve[:, 0] - EY_DO_A[:, 0]) ** 2)),
                "coord2_mse": float(np.mean((curve[:, 1] - EY_DO_A[:, 1]) ** 2)),
                "coord3_mse": float(np.mean((curve[:, 2] - EY_DO_A[:, 2]) ** 2)),
            }
        )
    temporary_path = f"{output_path}.tmp.{os.getpid()}"
    with open(temporary_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, output_path)
    return output_path


def main():
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    v_h, oracle = verify_oracles(args.lambda_ws)
    print(
        "oracle:",
        f"lambda_ws={args.lambda_ws:.3f}",
        f"V_h={v_h:.4f}",
        f"V_phi=(-1:{oracle[-1]['V_phi']:.4f}, +1:{oracle[1]['V_phi']:.4f})",
        f"smallest_gain={min(oracle[-1]['smallest_gain'], oracle[1]['smallest_gain']):.4f}",
        flush=True,
    )
    print(
        f"neural: n={args.n_sample} data_seed={args.data_seed} "
        f"train_seed={args.data_seed if args.train_seed < 0 else args.train_seed} "
        f"device={device}",
        flush=True,
    )
    metrics = run_neural_cell(args, device)
    print(
        "result:",
        f"OM={metrics['OutcomeNet']:.6g}",
        f"TM={metrics['TreatmentNet']:.6g}",
        f"DR={metrics['DRPCLNET']:.6g}",
        f"OracleOM={metrics['OracleOM']:.6g}",
        f"OracleTM={metrics['OracleTM']:.6g}",
        f"h_mse={metrics['h_mse']:.6g}",
        f"phi_mse={metrics['phi_mse']:.6g}",
        f"seconds={metrics['seconds']:.1f}",
        flush=True,
    )
    if not args.no_write:
        output_path = write_result(args, device, metrics)
        print("wrote", output_path, flush=True)


if __name__ == "__main__":
    main()
