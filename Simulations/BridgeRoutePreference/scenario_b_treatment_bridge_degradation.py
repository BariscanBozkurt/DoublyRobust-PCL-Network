"""Scenario B: treatment-bridge degradation under correct specification.

This is the script version of
``2-ScenarioB_CompleteMoment_TreatmentDegradation.ipynb``.  It preserves that
notebook's data-generating process exactly and uses the frozen Scenario O
neural pipeline so that the two route-stress experiments are directly
comparable.  The sweep changes only ``kappa_s`` in

    P(A=a | C=c,S=s) = {1 + a (kappa_c c + kappa_s s
                                      + kappa_cs c s)} / 2.

The exact vector-valued outcome bridge and its oracle evaluation variance are
fixed over the sweep.  The exact treatment bridge and its second moment change,
while the shared adjoint bridge operators become more ill-conditioned as
``kappa_s`` grows.  Both bridge equations remain correctly specified at every
admissible sweep value.

Each invocation trains one ``(kappa_s, data_seed, train_seed)`` cell and writes
one atomic, eight-algorithm CSV.  One invocation runs one cell of the paired grid.
"""

import argparse
import os

# Match the deterministic CUDA protocol used by compensated Scenario O.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

import scenario_o_outcome_bridge_amplification as base


KAPPA_C = 0.10
KAPPA_CS = 0.10
BETA = np.array([1.0, 1.0, 1.0])
SIGMA_Y = 0.25
LAMBDA_ZC = 0.80
LAMBDA_ZS = 0.60
LAMBDA_WC = 0.80
LAMBDA_WS = 0.80
KAPPA_S_GRID = (0.00, 0.20, 0.40, 0.60)
LATENT_STATES = tuple(base.LATENT_STATES)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kappa_s", "--kappa_S", type=float, default=0.0)
    parser.add_argument("--data_seed", type=int, default=10001)
    parser.add_argument(
        "--train_seed",
        type=int,
        default=-1,
        help="network/split seed; -1 reuses data_seed",
    )
    parser.add_argument("--n_sample", "--n", type=int, default=4000)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="set both OutcomeNet and TreatmentNet epochs",
    )
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
            base.REPO,
            "Simulations",
            "Results",
            "BridgeRoutePreference",
            "scenarioB_treatment_bridge_degradation",
        ),
    )
    parser.add_argument("--tag", type=str, default="paired_neural_nme")
    parser.add_argument("--no_write", action="store_true")
    parser.add_argument("--protocol", choices=["single", "crossfit"], default="single")
    parser.add_argument("--split", choices=["overlap", "independent"], default="overlap")
    return parser


def f_true(a):
    return base.f_true(a)


def treatment_score(c, s, kappa_s):
    return KAPPA_C * c + kappa_s * s + KAPPA_CS * c * s


def walsh_coefficients(a, kappa_s):
    """Walsh coefficients of ``1 / {1 + a eta(C,S)}``."""
    a = np.asarray(a, dtype=float)
    values = {
        state: 1.0
        / (
            1.0
            + a
            * treatment_score(state[0], state[1], kappa_s)
        )
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
        * sum(
            values[state] * character(*state)
            for state in LATENT_STATES
        )
        for key, character in characters.items()
    }


def phi0(a, z_c, z_s, kappa_s):
    """Exact treatment bridge from the notebook's Walsh expansion."""
    coefficients = walsh_coefficients(a, kappa_s)
    return (
        coefficients["0"]
        + coefficients["C"] * np.asarray(z_c) / LAMBDA_ZC
        + coefficients["S"] * np.asarray(z_s) / LAMBDA_ZS
        + coefficients["CS"]
        * np.asarray(z_c)
        * np.asarray(z_s)
        / (LAMBDA_ZC * LAMBDA_ZS)
    )


def h0(a, w_c, w_s, _kappa_s=None):
    """Exact outcome bridge; deliberately fixed over ``kappa_s``."""
    target = f_true(a)
    return np.stack(
        [
            target[..., 0] + BETA[0] * np.asarray(w_c) / LAMBDA_WC,
            target[..., 1] + BETA[1] * np.asarray(w_s) / LAMBDA_WS,
            target[..., 2]
            + BETA[2]
            * np.asarray(w_c)
            * np.asarray(w_s)
            / (LAMBDA_WC * LAMBDA_WS),
        ],
        axis=-1,
    )


def projected_phi0(a, w_c, w_s, kappa_s):
    """Exact ``E{phi_0(A,Z) | A,W}`` for the finite-state DGP."""
    conditional_score = (
        KAPPA_C * LAMBDA_WC * np.asarray(w_c)
        + kappa_s * LAMBDA_WS * np.asarray(w_s)
        + KAPPA_CS
        * LAMBDA_WC
        * LAMBDA_WS
        * np.asarray(w_c)
        * np.asarray(w_s)
    )
    return 1.0 / (1.0 + np.asarray(a) * conditional_score)


def generate_data(n, kappa_s, data_seed):
    """Generate the notebook DGP with its original RNG order unchanged."""
    validate_kappa_s(kappa_s)
    rng = np.random.RandomState(1000 + data_seed)
    latent_index = rng.randint(0, 4, size=n)
    c = np.array([LATENT_STATES[index][0] for index in latent_index], dtype=float)
    s = np.array([LATENT_STATES[index][1] for index in latent_index], dtype=float)

    score = treatment_score(c, s, kappa_s)
    a = np.where(rng.rand(n) < (1.0 + score) / 2.0, 1.0, -1.0)
    epsilon = SIGMA_Y * rng.randn(n, 3)
    y = f_true(a) + np.stack(
        [BETA[0] * c, BETA[1] * s, BETA[2] * c * s], axis=-1
    ) + epsilon

    z_c = base.bit_flip(c, LAMBDA_ZC, rng.rand(n))
    z_s = base.bit_flip(s, LAMBDA_ZS, rng.rand(n))
    w_c = base.bit_flip(c, LAMBDA_WC, rng.rand(n))
    w_s = base.bit_flip(s, LAMBDA_WS, rng.rand(n))
    return (
        a.reshape(-1, 1),
        y,
        np.stack([z_c, z_s], axis=1),
        np.stack([w_c, w_s], axis=1),
    )


def validate_kappa_s(kappa_s):
    if not np.isfinite(kappa_s):
        raise ValueError("kappa_s must be finite")
    if abs(KAPPA_C) + abs(kappa_s) + abs(KAPPA_CS) >= 1.0:
        raise ValueError(
            "strict overlap requires |kappa_c|+|kappa_s|+|kappa_cs| < 1"
        )


def proxy_kernel(lambda_c, lambda_s):
    kernel = np.zeros((4, 4))
    for latent_index, (c, s) in enumerate(LATENT_STATES):
        for proxy_index, (value_c, value_s) in enumerate(LATENT_STATES):
            kernel[latent_index, proxy_index] = (
                (1.0 + lambda_c * value_c * c)
                * (1.0 + lambda_s * value_s * s)
                / 4.0
            )
    return kernel


def oracle_diagnostics(kappa_s):
    """Compute exact bridge moments and finite-state operator gains."""
    validate_kappa_s(kappa_s)
    k_z = proxy_kernel(LAMBDA_ZC, LAMBDA_ZS)
    k_w = proxy_kernel(LAMBDA_WC, LAMBDA_WS)
    by_arm = {}
    residual_second_moment = np.sum(BETA**2) + 3.0 * SIGMA_Y**2

    for arm in (-1, 1):
        p_latent_given_arm = np.array(
            [
                (1.0 + arm * treatment_score(c, s, kappa_s)) / 4.0
                for c, s in LATENT_STATES
            ]
        )
        if not np.isclose(p_latent_given_arm.sum(), 1.0, atol=1e-12):
            raise AssertionError(p_latent_given_arm)

        e_phi = 0.0
        lambda_phi = 0.0
        for latent_index, _state in enumerate(LATENT_STATES):
            for proxy_index, (z_c, z_s) in enumerate(LATENT_STATES):
                probability = (
                    p_latent_given_arm[latent_index]
                    * k_z[latent_index, proxy_index]
                )
                bridge_value = phi0(arm, z_c, z_s, kappa_s)
                e_phi += probability * bridge_value
                lambda_phi += probability * bridge_value**2

        p_w_given_arm = p_latent_given_arm @ k_w
        projected_second_moment = sum(
            probability * projected_phi0(arm, w_c, w_s, kappa_s) ** 2
            for probability, (w_c, w_s) in zip(
                p_w_given_arm, LATENT_STATES
            )
        )
        joint_zw_given_arm = (
            k_z.T @ np.diag(p_latent_given_arm) @ k_w
        )
        standardized = (
            np.diag(1.0 / np.sqrt(joint_zw_given_arm.sum(axis=1)))
            @ joint_zw_given_arm
            @ np.diag(1.0 / np.sqrt(joint_zw_given_arm.sum(axis=0)))
        )
        singular_values = np.linalg.svd(standardized, compute_uv=False)
        by_arm[arm] = {
            "E_phi": float(e_phi),
            "Lambda_phi": float(lambda_phi),
            "projected_Lambda_phi": float(projected_second_moment),
            "V_phi": float(2.0 * lambda_phi * residual_second_moment),
            "smallest_gain": float(singular_values[-1]),
            "singular_values": singular_values,
        }

    v_h = (
        BETA[0] ** 2 / LAMBDA_WC**2
        + BETA[1] ** 2 / LAMBDA_WS**2
        + BETA[2] ** 2 / (LAMBDA_WC**2 * LAMBDA_WS**2)
    )
    return float(v_h), by_arm


def verify_oracles(kappa_s):
    """Exhaustively verify the exact bridges on the 16 finite-state cells."""
    validate_kappa_s(kappa_s)
    k_z = proxy_kernel(LAMBDA_ZC, LAMBDA_ZS)
    k_w = proxy_kernel(LAMBDA_WC, LAMBDA_WS)
    if np.linalg.matrix_rank(k_z) != 4 or np.linalg.matrix_rank(k_w) != 4:
        raise AssertionError("proxy kernels must be full rank")

    latent_features = np.array(
        [[c, s, c * s] for c, s in LATENT_STATES], dtype=float
    )
    latent_prior = np.full(4, 0.25)

    # The outcome bridge equation holds for every (a,z), and its marginal
    # identifying formula holds for every intervention arm.
    for arm in (-1, 1):
        treatment_probability = np.array(
            [
                (1.0 + arm * treatment_score(c, s, kappa_s)) / 2.0
                for c, s in LATENT_STATES
            ]
        )
        for z_index, _z in enumerate(LATENT_STATES):
            cell_weight = (
                latent_prior * treatment_probability * k_z[:, z_index]
            )
            denominator = cell_weight.sum()
            outcome_conditional_mean = f_true(arm) + (
                cell_weight[:, None] * BETA * latent_features
            ).sum(axis=0) / denominator
            bridge_conditional_mean = np.zeros(3)
            for latent_index in range(4):
                for w_index, (w_c, w_s) in enumerate(LATENT_STATES):
                    bridge_conditional_mean += (
                        cell_weight[latent_index]
                        * k_w[latent_index, w_index]
                        * h0(arm, w_c, w_s)
                        / denominator
                    )
            if not np.allclose(
                bridge_conditional_mean,
                outcome_conditional_mean,
                atol=1e-12,
            ):
                raise AssertionError(
                    ("outcome bridge", arm, _z, bridge_conditional_mean,
                     outcome_conditional_mean)
                )

        marginal_h = sum(
            latent_prior[latent_index]
            * k_w[latent_index, w_index]
            * h0(arm, w_c, w_s)
            for latent_index in range(4)
            for w_index, (w_c, w_s) in enumerate(LATENT_STATES)
        )
        if not np.allclose(marginal_h, f_true(arm), atol=1e-12):
            raise AssertionError(("outcome identifying formula", arm, marginal_h))

        # Treatment normalization and weighted identifying formula.
        p_latent_given_arm = 2.0 * latent_prior * treatment_probability
        e_phi = 0.0
        weighted_outcome = np.zeros(3)
        for latent_index in range(4):
            conditional_y = f_true(arm) + BETA * latent_features[latent_index]
            for z_index, (z_c, z_s) in enumerate(LATENT_STATES):
                probability = (
                    p_latent_given_arm[latent_index]
                    * k_z[latent_index, z_index]
                )
                bridge_value = phi0(arm, z_c, z_s, kappa_s)
                e_phi += probability * bridge_value
                weighted_outcome += (
                    probability * bridge_value * conditional_y
                )
        if not np.isclose(e_phi, 1.0, atol=1e-12):
            raise AssertionError(("treatment normalization", arm, e_phi))
        if not np.allclose(weighted_outcome, f_true(arm), atol=1e-12):
            raise AssertionError(
                ("treatment identifying formula", arm, weighted_outcome)
            )

        # The projected bridge is checked independently in every (a,w) cell.
        for w_index, (w_c, w_s) in enumerate(LATENT_STATES):
            denominator = np.sum(
                latent_prior * treatment_probability * k_w[:, w_index]
            )
            numerator = 0.0
            for latent_index in range(4):
                for z_index, (z_c, z_s) in enumerate(LATENT_STATES):
                    numerator += (
                        latent_prior[latent_index]
                        * treatment_probability[latent_index]
                        * k_w[latent_index, w_index]
                        * k_z[latent_index, z_index]
                        * phi0(arm, z_c, z_s, kappa_s)
                    )
            enumerated_projection = numerator / denominator
            closed_form_projection = projected_phi0(
                arm, w_c, w_s, kappa_s
            )
            if not np.isclose(
                enumerated_projection, closed_form_projection, atol=1e-12
            ):
                raise AssertionError(
                    (
                        "projected treatment bridge",
                        arm,
                        (w_c, w_s),
                        enumerated_projection,
                        closed_form_projection,
                    )
                )

    v_h, diagnostics = oracle_diagnostics(kappa_s)
    expected_v_h = 5.56640625
    if not np.isclose(v_h, expected_v_h, atol=1e-12):
        raise AssertionError((v_h, expected_v_h))
    for arm in (-1, 1):
        arm_diagnostics = diagnostics[arm]
        if not np.isclose(arm_diagnostics["E_phi"], 1.0, atol=1e-12):
            raise AssertionError(diagnostics)
        if arm_diagnostics["projected_Lambda_phi"] > (
            arm_diagnostics["Lambda_phi"] + 1e-12
        ):
            raise AssertionError("conditional projection increased the L2 norm")
        if arm_diagnostics["smallest_gain"] <= 0.0:
            raise AssertionError("conditional operator is not injective")
    return v_h, diagnostics


def run_cell(args):
    if args.epochs is not None:
        args.om_epochs = args.epochs
        args.tm_epochs = args.epochs
    device = base.resolve_device(args.device)
    torch.use_deterministic_algorithms(True)
    v_h, oracle = verify_oracles(args.kappa_s)
    print(
        "oracle:",
        f"kappa_s={args.kappa_s:.3f}",
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

    # The shared pipeline names its generic sweep argument ``lambda_ws``.
    # Supplying this private compatibility alias does not change the DGP.
    args.lambda_ws = args.kappa_s
    metrics = base.run_neural_cell(
        args,
        device,
        generate_data_function=generate_data,
        outcome_bridge_function=h0,
        treatment_bridge_function=(
            lambda a, z_c, z_s: phi0(a, z_c, z_s, args.kappa_s)
        ),
        verify_oracles_function=verify_oracles,
        treatment_projection_function=projected_phi0,
    )
    print(
        "result:",
        f"OM={metrics['OutcomeNet']:.6g}",
        f"TM={metrics['TreatmentNet']:.6g}",
        f"DR={metrics['DRPCLNET']:.6g}",
        f"OracleOM={metrics['OracleOM']:.6g}",
        f"OracleTM={metrics['OracleTM']:.6g}",
        f"h_mse={metrics['h_mse']:.6g}",
        f"phi_mse={metrics['phi_mse']:.6g}",
        f"phi_projected_mse={metrics['phi_projected_mse']:.6g}",
        f"seconds={metrics['seconds']:.1f}",
        flush=True,
    )
    if not args.no_write:
        output_path = base.write_result(
            args,
            device,
            metrics,
            scenario="B-treatment-bridge-degradation",
            sweep_name="kappa_s",
            file_prefix="scenarioBTreatmentDegradation",
            file_sweep_label="kappas",
        )
        print("wrote", output_path, flush=True)
    return metrics


def main():
    args = build_parser().parse_args()
    run_cell(args)


if __name__ == "__main__":
    main()
