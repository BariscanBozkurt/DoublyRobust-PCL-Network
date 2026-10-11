"""Scenario O with an invariant full-input TreatmentNet problem.

The treatment mechanism depends on a latent confounder ``C``.  A second,
independent latent sign ``S`` is prognostic.  Its proxy reliabilities satisfy

    lambda_ZS * lambda_WS = Q,

so the observed law of ``(A, Z, W)`` is exactly invariant when ``lambda_WS``
is swept.  Consequently, the unrestricted density-ratio-free TreatmentNet
training problem is fixed.  In contrast, the exact outcome bridge contains
``1 / lambda_WS`` and its oracle variance increases as ``lambda_WS`` falls.

The data generator samples the invariant observed proxy pair first and then
samples the lambda-specific latent ``S`` from its exact posterior.  With a
common data seed, the realized arrays ``(A, Z, W)`` are therefore numerically identical
at every sweep level; only ``S`` and the outcome can change.
"""

import argparse
import hashlib
import os

# Required by PyTorch when deterministic CUDA algorithms use cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

import scenario_o_outcome_bridge_amplification as base


KAPPA_C = 0.10
GAMMA_C = 0.25
BETA_S = 1.00
ALPHA_CS = 0.80
SIGMA_Y = 0.25
LAMBDA_ZC = 0.90
LAMBDA_WC = 0.90
PROXY_PRODUCT_S = 0.36
LAMBDA_GRID = (0.90, 0.70, 0.60, 0.50, 0.40)
LATENT_STATES = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda_ws", type=float, default=0.90)
    parser.add_argument("--data_seed", type=int, default=1)
    parser.add_argument("--train_seed", type=int, default=-1)
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
            base.REPO,
            "Simulations",
            "Results",
            "BridgeRoutePreference",
            "scenarioO_compensated_proxy_reliability",
        ),
    )
    parser.add_argument("--tag", type=str, default="paired_neural_nme")
    parser.add_argument("--no_write", action="store_true")
    parser.add_argument("--protocol", choices=["single", "crossfit"], default="single")
    parser.add_argument("--split", choices=["overlap", "independent"], default="overlap")
    parser.add_argument("--sampling", choices=["posterior", "forward"], default="posterior",
                        help="factorisation of P(S, Z_S, W_S); same joint law, different pairing "
                             "along the sweep (see generate_data)")
    return parser


def lambda_zs(lambda_ws):
    if not PROXY_PRODUCT_S <= lambda_ws <= 1.0:
        raise ValueError(
            "lambda_ws must lie in [PROXY_PRODUCT_S, 1] so lambda_zs <= 1"
        )
    return PROXY_PRODUCT_S / lambda_ws


def treatment_score(c):
    return KAPPA_C * np.asarray(c)


def phi0(a, z_c, z_s):
    """Exact treatment bridge; ``z_s`` is intentionally irrelevant."""
    del z_s
    return (1.0 - np.asarray(a) * KAPPA_C * np.asarray(z_c) / LAMBDA_ZC) / (
        1.0 - KAPPA_C**2
    )


def projected_phi0(a, w_c, w_s, lambda_ws):
    """Exact ``E[phi_0(A,Z) | A,W]``; fixed over the reliability sweep."""
    del w_s, lambda_ws
    return 1.0 / (
        1.0 + np.asarray(a) * KAPPA_C * LAMBDA_WC * np.asarray(w_c)
    )


def h0(a, w_c, w_s, lambda_ws):
    """Exact vector-valued outcome bridge."""
    target = base.f_true(a)
    return np.stack(
        [
            target[..., 0] + GAMMA_C * np.asarray(w_c) / LAMBDA_WC,
            target[..., 1] + BETA_S * np.asarray(w_s) / lambda_ws,
            target[..., 2]
            + ALPHA_CS
            * np.asarray(w_c)
            * np.asarray(w_s)
            / (LAMBDA_WC * lambda_ws),
        ],
        axis=-1,
    )


def generate_data(n, lambda_ws, data_seed, sampling="posterior"):
    """Two equivalent factorisations of the SAME joint law of (S, Z_S, W_S) -- verified to Monte-Carlo
    error over the eight cells -- differing only in what a fixed seed holds fixed along the sweep.

    "posterior" (the original): draw the observed pair (Z_S, W_S), whose law depends only on the pinned
    product Q = lambda_ZS * lambda_WS, then draw S from its exact posterior. (A,Z,W) is then BIT-IDENTICAL
    across the sweep, while S -- and hence Y -- is redrawn at each level.

    "forward" (the textbook order): draw the latent S first and pass it through the two conditionally
    independent bit-flip channels. A, C, S and Y are then numerically identical across the sweep and only the
    S-proxies move, so every quantity that ignores them -- the exact treatment bridge phi_0(a, z_c) and
    its oracle estimate -- is EXACTLY constant along the sweep rather than merely constant in expectation.
    """
    if sampling not in ("posterior", "forward"):
        raise ValueError("sampling must be 'posterior' or 'forward'")
    current_lambda_zs = lambda_zs(lambda_ws)
    rng = np.random.RandomState(1000 + data_seed)

    c = np.where(rng.rand(n) < 0.5, -1.0, 1.0)
    a = np.where(
        rng.rand(n) < (1.0 + treatment_score(c)) / 2.0,
        1.0,
        -1.0,
    )
    epsilon = rng.randn(n, 3)

    z_c = base.bit_flip(c, LAMBDA_ZC, rng.rand(n))
    w_c = base.bit_flip(c, LAMBDA_WC, rng.rand(n))

    # P(Z_S=z,W_S=w)=(1+Q*z*w)/4 depends only on Q, not on either
    # reliability separately.  Sampling this observed pair first makes the
    # TreatmentNet input exactly paired, rather than merely equal in law.
    if sampling == "forward":
        s = np.where(rng.rand(n) < 0.5, -1.0, 1.0)
        z_s = base.bit_flip(s, current_lambda_zs, rng.rand(n))
        w_s = base.bit_flip(s, lambda_ws, rng.rand(n))
        return _assemble(a, c, s, z_c, z_s, w_c, w_s, epsilon)
    z_s = np.where(rng.rand(n) < 0.5, -1.0, 1.0)
    same_sign = rng.rand(n) < (1.0 + PROXY_PRODUCT_S) / 2.0
    w_s = np.where(same_sign, z_s, -z_s)

    # Exact posterior mean under conditionally independent bit-flip channels:
    # E(S|z,w)=(lambda_ZS*z+lambda_WS*w)/(1+Q*z*w).
    latent_s_mean = (
        current_lambda_zs * z_s + lambda_ws * w_s
    ) / (1.0 + PROXY_PRODUCT_S * z_s * w_s)
    if np.max(np.abs(latent_s_mean)) > 1.0 + 1e-12:
        raise AssertionError("invalid latent-S posterior mean")
    s = np.where(
        rng.rand(n) < (1.0 + latent_s_mean) / 2.0,
        1.0,
        -1.0,
    )

    return _assemble(a, c, s, z_c, z_s, w_c, w_s, epsilon)


def _assemble(a, c, s, z_c, z_s, w_c, w_s, epsilon):
    """Outcome equation and array packing, shared by both sampling orders."""
    y = base.f_true(a) + np.stack(
        [
            GAMMA_C * c + SIGMA_Y * epsilon[:, 0],
            BETA_S * s + SIGMA_Y * epsilon[:, 1],
            ALPHA_CS * c * s + SIGMA_Y * epsilon[:, 2],
        ],
        axis=-1,
    )
    return (
        a.reshape(-1, 1),
        y,
        np.stack([z_c, z_s], axis=1),
        np.stack([w_c, w_s], axis=1),
    )


def proxy_kernel(lambda_c, lambda_s):
    kernel = np.zeros((4, 4))
    for latent_index, (c, s) in enumerate(LATENT_STATES):
        for proxy_index, (v_c, v_s) in enumerate(LATENT_STATES):
            kernel[latent_index, proxy_index] = (
                (1.0 + lambda_c * v_c * c)
                * (1.0 + lambda_s * v_s * s)
                / 4.0
            )
    return kernel


def oracle_diagnostics(lambda_ws):
    current_lambda_zs = lambda_zs(lambda_ws)
    k_z = proxy_kernel(LAMBDA_ZC, current_lambda_zs)
    k_w = proxy_kernel(LAMBDA_WC, lambda_ws)
    residual_second_moment = (
        GAMMA_C**2 + BETA_S**2 + ALPHA_CS**2 + 3.0 * SIGMA_Y**2
    )
    by_arm = {}
    for arm in (-1, 1):
        p_latent_given_arm = np.array(
            [(1.0 + arm * KAPPA_C * c) / 4.0 for c, _s in LATENT_STATES]
        )
        e_phi = 0.0
        lambda_phi = 0.0
        for latent_index, _state in enumerate(LATENT_STATES):
            for proxy_index, (z_c, z_s) in enumerate(LATENT_STATES):
                probability = (
                    p_latent_given_arm[latent_index]
                    * k_z[latent_index, proxy_index]
                )
                value = phi0(arm, z_c, z_s)
                e_phi += probability * value
                lambda_phi += probability * value**2

        joint_zw = k_z.T @ np.diag(p_latent_given_arm) @ k_w
        d_z = np.diag(1.0 / np.sqrt(joint_zw.sum(axis=1)))
        d_w = np.diag(1.0 / np.sqrt(joint_zw.sum(axis=0)))
        singular_values = np.linalg.svd(d_z @ joint_zw @ d_w, compute_uv=False)
        by_arm[arm] = {
            "E_phi": float(e_phi),
            "Lambda_phi": float(lambda_phi),
            "V_phi": float(2.0 * lambda_phi * residual_second_moment),
            "smallest_gain": float(singular_values[-1]),
            "relevant_gain": float(
                LAMBDA_ZC
                * LAMBDA_WC
                * (1.0 - KAPPA_C**2)
                / np.sqrt(
                    (1.0 - (LAMBDA_ZC * KAPPA_C) ** 2)
                    * (1.0 - (LAMBDA_WC * KAPPA_C) ** 2)
                )
            ),
        }

    v_h = (
        GAMMA_C**2 / LAMBDA_WC**2
        + BETA_S**2 / lambda_ws**2
        + ALPHA_CS**2 / (LAMBDA_WC**2 * lambda_ws**2)
    )
    return v_h, by_arm


def verify_oracles(lambda_ws):
    current_lambda_zs = lambda_zs(lambda_ws)
    v_h, diagnostics = oracle_diagnostics(lambda_ws)
    expected_lambda_phi = (
        1.0 - 2.0 * KAPPA_C**2 + KAPPA_C**2 / LAMBDA_ZC**2
    ) / (1.0 - KAPPA_C**2) ** 2
    expected_v_phi = 2.0 * expected_lambda_phi * (
        GAMMA_C**2 + BETA_S**2 + ALPHA_CS**2 + 3.0 * SIGMA_Y**2
    )
    expected_v_h = (
        GAMMA_C**2 / LAMBDA_WC**2
        + (BETA_S**2 + ALPHA_CS**2 / LAMBDA_WC**2) / lambda_ws**2
    )
    if not np.isclose(v_h, expected_v_h, atol=1e-12):
        raise AssertionError((v_h, expected_v_h))
    for arm in (-1, 1):
        current = diagnostics[arm]
        if not np.isclose(current["E_phi"], 1.0, atol=1e-12):
            raise AssertionError(diagnostics)
        if not np.isclose(current["Lambda_phi"], expected_lambda_phi, atol=1e-12):
            raise AssertionError(diagnostics)
        if not np.isclose(current["V_phi"], expected_v_phi, atol=1e-12):
            raise AssertionError(diagnostics)

        # Verify the projected treatment bridge by exact enumeration.
        for w_c in (-1, 1):
            for w_s in (-1, 1):
                numerator = 0.0
                denominator = 0.0
                for c, s in LATENT_STATES:
                    latent_probability = 0.25
                    arm_probability = (1.0 + arm * KAPPA_C * c) / 2.0
                    proxy_probability = (
                        (1.0 + LAMBDA_WC * w_c * c)
                        * (1.0 + lambda_ws * w_s * s)
                        / 4.0
                    )
                    expected_phi_given_latent = (
                        1.0 - arm * KAPPA_C * c
                    ) / (1.0 - KAPPA_C**2)
                    joint_probability = (
                        latent_probability * arm_probability * proxy_probability
                    )
                    denominator += joint_probability
                    numerator += joint_probability * expected_phi_given_latent
                enumerated = numerator / denominator
                closed_form = projected_phi0(arm, w_c, w_s, lambda_ws)
                if not np.isclose(enumerated, closed_form, atol=1e-12):
                    raise AssertionError((enumerated, closed_form))

        # The observable conditional-expectation operator is exactly fixed.
        p_latent_given_arm = np.array(
            [(1.0 + arm * KAPPA_C * c) / 4.0 for c, _s in LATENT_STATES]
        )
        joint_zw = (
            proxy_kernel(LAMBDA_ZC, current_lambda_zs).T
            @ np.diag(p_latent_given_arm)
            @ proxy_kernel(LAMBDA_WC, lambda_ws)
        )
        reference_lambda_ws = LAMBDA_GRID[0]
        reference_joint_zw = (
            proxy_kernel(LAMBDA_ZC, lambda_zs(reference_lambda_ws)).T
            @ np.diag(p_latent_given_arm)
            @ proxy_kernel(LAMBDA_WC, reference_lambda_ws)
        )
        if not np.allclose(joint_zw, reference_joint_zw, atol=1e-12):
            raise AssertionError("conditional Z-W operator changes over sweep")

        # Verify E[Y|A,Z]=E[h_0(A,W)|A,Z] coordinate by coordinate.
        for z_c in (-1, 1):
            for z_s in (-1, 1):
                denominator = 0.0
                outcome_numerator = np.zeros(3)
                bridge_numerator = np.zeros(3)
                for c, s in LATENT_STATES:
                    latent_probability = 0.25
                    arm_probability = (1.0 + arm * KAPPA_C * c) / 2.0
                    z_probability = (
                        (1.0 + LAMBDA_ZC * z_c * c)
                        * (1.0 + current_lambda_zs * z_s * s)
                        / 4.0
                    )
                    joint_probability = (
                        latent_probability * arm_probability * z_probability
                    )
                    conditional_y_mean = base.f_true(arm) + np.array(
                        [GAMMA_C * c, BETA_S * s, ALPHA_CS * c * s]
                    )
                    conditional_h_mean = np.zeros(3)
                    for w_c in (-1, 1):
                        for w_s in (-1, 1):
                            w_probability = (
                                (1.0 + LAMBDA_WC * w_c * c)
                                * (1.0 + lambda_ws * w_s * s)
                                / 4.0
                            )
                            conditional_h_mean += w_probability * h0(
                                arm, w_c, w_s, lambda_ws
                            )
                    denominator += joint_probability
                    outcome_numerator += joint_probability * conditional_y_mean
                    bridge_numerator += joint_probability * conditional_h_mean
                if not np.allclose(
                    outcome_numerator / denominator,
                    bridge_numerator / denominator,
                    atol=1e-12,
                ):
                    raise AssertionError("outcome bridge equation failed")

        # Verify both bridge identifying formulas, not only their Fredholm
        # equations.  Since P(A=arm)=1/2, the normalized treatment bridge
        # identifies E[Y(arm)] through an arm-conditional weighted mean.
        treatment_identified = np.zeros(3)
        outcome_identified = np.zeros(3)
        for latent_index, (c, s) in enumerate(LATENT_STATES):
            latent_probability = 0.25
            arm_conditional_probability = (1.0 + arm * KAPPA_C * c) / 4.0
            conditional_y_mean = base.f_true(arm) + np.array(
                [GAMMA_C * c, BETA_S * s, ALPHA_CS * c * s]
            )
            for proxy_index, (z_c, z_s) in enumerate(LATENT_STATES):
                treatment_identified += (
                    arm_conditional_probability
                    * proxy_kernel(
                        LAMBDA_ZC, current_lambda_zs
                    )[latent_index, proxy_index]
                    * phi0(arm, z_c, z_s)
                    * conditional_y_mean
                )
            for w_c, w_s in LATENT_STATES:
                w_probability = (
                    (1.0 + LAMBDA_WC * w_c * c)
                    * (1.0 + lambda_ws * w_s * s)
                    / 4.0
                )
                outcome_identified += (
                    latent_probability
                    * w_probability
                    * h0(arm, w_c, w_s, lambda_ws)
                )
        if not np.allclose(
            treatment_identified, base.f_true(arm), atol=1e-12
        ):
            raise AssertionError("treatment bridge identifying formula failed")
        if not np.allclose(
            outcome_identified, base.f_true(arm), atol=1e-12
        ):
            raise AssertionError("outcome bridge identifying formula failed")

    # The S-channel joint proxy law depends only on the product Q.
    for z_s in (-1, 1):
        for w_s in (-1, 1):
            enumerated = 0.0
            for s in (-1, 1):
                enumerated += (
                    0.5
                    * (1.0 + current_lambda_zs * z_s * s)
                    * (1.0 + lambda_ws * w_s * s)
                    / 4.0
                )
            closed_form = (1.0 + PROXY_PRODUCT_S * z_s * w_s) / 4.0
            if not np.isclose(enumerated, closed_form, atol=1e-12):
                raise AssertionError((enumerated, closed_form))
            posterior_mean = (
                current_lambda_zs * z_s + lambda_ws * w_s
            ) / (1.0 + PROXY_PRODUCT_S * z_s * w_s)
            for s in (-1, 1):
                observed_first_joint = closed_form * (
                    1.0 + s * posterior_mean
                ) / 2.0
                latent_first_joint = (
                    (1.0 + current_lambda_zs * z_s * s)
                    * (1.0 + lambda_ws * w_s * s)
                    / 8.0
                )
                if not np.isclose(
                    observed_first_joint, latent_first_joint, atol=1e-12
                ):
                    raise AssertionError(
                        (observed_first_joint, latent_first_joint)
                    )
    return v_h, diagnostics


def observed_input_sha256(n, lambda_ws, data_seed, sampling="posterior"):
    a, _y, z, w = generate_data(n, lambda_ws, data_seed, sampling)
    joined = np.ascontiguousarray(np.concatenate([a, z, w], axis=1))
    return hashlib.sha256(joined.tobytes()).hexdigest()


def verify_paired_generator(n=4000, data_seed=1, sampling="posterior"):
    """posterior: (A,Z,W) must be numerically identical along the sweep. forward: (A, C, S, Y) is what is held
    fixed instead, so the S-proxies -- and hence the hash -- move by construction; the invariant checked
    there is that the exact-bridge quantities still reproduce at every level."""
    hashes = {
        observed_input_sha256(n, lambda_ws, data_seed, sampling)
        for lambda_ws in LAMBDA_GRID
    }
    if sampling == "posterior" and len(hashes) != 1:
        raise AssertionError("(A,Z,W) is not numerically identical over lambda_ws")
    if sampling == "forward":
        first = generate_data(n, LAMBDA_GRID[0], data_seed, sampling)
        for lambda_ws in LAMBDA_GRID[1:]:
            other = generate_data(n, lambda_ws, data_seed, sampling)
            if not (np.array_equal(first[0], other[0]) and np.array_equal(first[1], other[1])):
                raise AssertionError("(A,Y) is not numerically identical over lambda_ws under forward sampling")
    for lambda_ws in LAMBDA_GRID:
        verify_oracles(lambda_ws)
    return sorted(hashes)[0]


def main():
    args = build_parser().parse_args()
    device = base.resolve_device(args.device)
    torch.use_deterministic_algorithms(True)
    paired_hash = verify_paired_generator(args.n_sample, args.data_seed, args.sampling)
    v_h, oracle = verify_oracles(args.lambda_ws)
    print(
        "oracle:",
        f"lambda_ws={args.lambda_ws:.3f}",
        f"lambda_zs={lambda_zs(args.lambda_ws):.3f}",
        f"V_h={v_h:.4f}",
        f"V_phi={oracle[-1]['V_phi']:.4f}",
        f"smallest_gain={min(oracle[-1]['smallest_gain'], oracle[1]['smallest_gain']):.4f}",
        f"AZW_sha256={paired_hash[:12]}",
        flush=True,
    )
    print(
        f"neural: n={args.n_sample} data_seed={args.data_seed} "
        f"train_seed={args.data_seed if args.train_seed < 0 else args.train_seed} "
        f"device={device}",
        flush=True,
    )
    metrics = base.run_neural_cell(
        args,
        device,
        generate_data_function=lambda n, lam, seed: generate_data(n, lam, seed, args.sampling),
        outcome_bridge_function=h0,
        treatment_bridge_function=phi0,
        verify_oracles_function=verify_oracles,
        treatment_projection_function=projected_phi0,
    )
    metrics["relevant_gain"] = oracle[-1]["relevant_gain"]
    metrics["observed_input_sha256"] = paired_hash
    print(
        "result:",
        f"OM={metrics['OutcomeNet']:.6g}",
        f"TM={metrics['TreatmentNet']:.6g}",
        f"DR={metrics['DRPCLNET']:.6g}",
        f"OracleOM={metrics['OracleOM']:.6g}",
        f"OracleTM={metrics['OracleTM']:.6g}",
        f"h_mse={metrics['h_mse']:.6g}",
        f"phi_mse={metrics['phi_mse']:.6g}",
        f"phi_sha256={metrics['phi_prediction_sha256'][:12]}",
        f"seconds={metrics['seconds']:.1f}",
        flush=True,
    )
    if not args.no_write:
        output_path = base.write_result(
            args,
            device,
            metrics,
            scenario="O-compensated-proxy-reliability",
            sweep_name="lambda_ws",
            file_prefix="scenarioOCompensatedProxyReliability",
            file_sweep_label="lamws",
        )
        print("wrote", output_path, flush=True)


if __name__ == "__main__":
    main()
