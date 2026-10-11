"""NYSTROM kernel baselines (KPV + KAP) for the ATT on the synthetic LOW-DIM benchmark, size sweep.

WHY THIS EXISTS: the EXACT ATT kernel sim
(`Synthetic_ATT_LowDim_KPV_KAP_Simulation.py`) does not reach the reporting axis. Its Grams are O(n^2)
float64 and its solves O(n^3): measured 31.5 GB (KPV) / 64.3 GB (KAP) at n=20000, and KAP bakes the anchor
a' into B_bar at FIT time so it pays that FOUR times per (data_seed, repeat). The paper reports the NYSTROM
family on every benchmark (dSprite V2/V3 and demand already do); this script supplies the matching ATT rows
so the ATT table is complete at {2,5,10,15,20}x1000 without ever running the exact method at large n.

WHAT IS APPROXIMATED (`causal_learning/approximate_kernel_proxy_causal_learning.py`):
  KPV_Nystrom : stage-1 CME solve; stage-2 outcome-bridge KRR (representers = stage-2 landmarks); and the
                a'-CME of predict, on that same landmark subsample.
  KAP_Nystrom : stage-1 CME solve (B, B_bar and the eta block's B2, B2_bar); the a'-CME theta; the stage-2
                alpha/eta system (representers = stage-2 landmarks); and the third-stage KRR.
REGULARISERS ARE FIXED BY HAND -- NO LOOCV (design decision, and the convention of the existing
Nystrom sims for dSprite and demand). The exact classes' LOOCV identity is the hat-matrix formula
h_i = (I - R)_{ii} with the 1/diag(I - R) leave-one-out correction, and that identity is derived for the
EXACT ridge solution: under a low-rank Nystrom solve those leverage terms are no longer the leave-one-out
residuals of the estimator being fitted, so the objective is not well defined. `--optimize 1` keeps the
selection available (it is exact, and therefore meaningful, only when the landmark set is the whole sample --
which is how the acceptance test below uses it), but the REPORTING DEFAULT is `--optimize 0` with the values
FROZEN at KPV 1e-4 / KAP 1e-3 by the screen on tuning seeds 100-102 and the confirmation on fresh seeds
103-105 (both rules declared before the runs). When
a LOOCV is run it is O(n^3) per grid point, so it is restricted to `--loocv_subsample` rows.

ACCEPTANCE: with the landmark set equal to
the whole sample and the LOOCV on the whole sample, both classes reproduce the EXACT ATT classes -- every
selected hyperparameter identical, max relative curve difference 2.1e-6 (KPV) and 2.2e-9 (KAP) at n=400
with the selections active, 4.2e-9 / 1.0e-7 with the regularisers fixed. At n=2000 with 1000 landmarks
(= all of the stage-2 sample) the agreement is 2.3e-7 / 4.4e-8.

THE ESTIMAND is unchanged: f_ATT(a, a') = E[Y^(a) | A = a'], ground truth
`benchmark.generate_att_curve(anchor_a=a')` -- NOT EY_do_A / the ATE curve, which is off by up to 0.94 here.

PROTOCOL. Identical to the exact ATT sim so the two are directly comparable: single split (stage1_perc 0.5)
drawn inside fit() from the GLOBAL numpy RNG, 10 data seeds x --n_repeats, train_seed = 1000*data_seed +
repeat, np.random.seed(train_seed) immediately before every fit so KPV and all four KAP anchors see the SAME
split. Landmarks and LOOCV subsamples are drawn from `model_seed` (= train_seed here) with an independent
generator, so they never disturb that stream and are identical across anchors.

  python Synthetic_ATT_LowDim_KPV_KAP_Nystrom_Simulation.py --data_seeds 11 --n_repeats 1 --n_sample 2000
  python Synthetic_ATT_LowDim_KPV_KAP_Nystrom_Simulation.py --data_seeds 0 --n_sample 20000 --nystrom_m 500
"""
import argparse, csv, os, socket, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))

from simulation_utils.data_generating_processes import SyntheticLowDimATTBenchmark
from torch_utils.scalers import TorchStandardScaler
from torch_utils.kernel_utils import ColumnwiseRBF, RBF
from causal_learning.approximate_kernel_proxy_causal_learning import (
    KernelAlternativeProxyATTNystrom, KernelProxyVariableATTNystrom)

torch.set_default_dtype(torch.float64)          # mandatory: make_psd_eps 1e-9 is below float32 resolution
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "Results", "Synthetic_ATT_LowDim_KPV_KAP_Nystrom"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--n_sample", type=int, default=2000)
    p.add_argument("--n_repeats", type=int, default=3)
    p.add_argument("--repeat_offset", type=int, default=0)
    p.add_argument("--anchors", type=float, nargs="+", default=[-1.0, -0.5, 0.25, 1.0])
    p.add_argument("--do_a_lo", type=float, default=-1.0)
    p.add_argument("--do_a_hi", type=float, default=2.0)
    p.add_argument("--do_a_size", type=int, default=100)
    p.add_argument("--algorithms", nargs="+", choices=["KPV", "KAP"], default=["KPV", "KAP"])
    # -- Nystrom knobs (the only addition over the exact sim) --
    p.add_argument("--nystrom_m", type=int, default=500,
                   help="landmarks per approximated solve; the per-slot flags below override it")
    p.add_argument("--kpv_nystrom_stage1_m", type=int, default=0, help="0 = --nystrom_m")
    p.add_argument("--kpv_nystrom_stage2_m", type=int, default=0)
    p.add_argument("--kap_nystrom_stage1_m", type=int, default=0)
    p.add_argument("--kap_nystrom_stage2_m", type=int, default=0,
                   help="0 = --nystrom_m; pass a value above n/2 to keep the alpha/eta system EXACT")
    p.add_argument("--kap_nystrom_stage3_m", type=int, default=0)
    p.add_argument("--loocv_subsample", type=int, default=2000,
                   help="rows used by every regulariser LOOCV (0 = all rows = the exact classes)")
    # -- FIXED regularisers (used when --optimize 0, the reporting default). FROZEN by the screen
    # declared before either ran). The two estimators take DIFFERENT values: on
    # the screen grid KPV's optimum was 1e-4 and KAP's 1e-3, and that split reproduced 3/3 on fresh seeds at
    # BOTH n=2000 and n=5000, which was the declared condition for adopting it. --
    p.add_argument("--kpv_lambda1", type=float, default=1e-4)
    p.add_argument("--kpv_lambda2", type=float, default=1e-4)
    p.add_argument("--kpv_zeta", type=float, default=1e-4)
    p.add_argument("--kap_lambda", type=float, default=1e-3)
    p.add_argument("--kap_eta", type=float, default=1e-3)
    p.add_argument("--kap_lambda2", type=float, default=1e-3)
    p.add_argument("--kap_zeta", type=float, default=1e-3)
    # -- LOOCV ranges, used only when --optimize 1 --
    p.add_argument("--kpv_lambda1_lo", type=float, default=5e-4)
    p.add_argument("--kpv_lambda1_hi", type=float, default=1.0)
    p.add_argument("--kpv_lambda2_lo", type=float, default=5e-4)
    p.add_argument("--kpv_lambda2_hi", type=float, default=1.0)
    p.add_argument("--kpv_zeta_lo", type=float, default=5e-5)
    p.add_argument("--kpv_zeta_hi", type=float, default=1.0)
    p.add_argument("--kap_lambda_lo", type=float, default=5e-5)
    p.add_argument("--kap_lambda_hi", type=float, default=1.0)
    p.add_argument("--kap_eta_lo", type=float, default=5e-5)
    p.add_argument("--kap_eta_hi", type=float, default=1.0)
    p.add_argument("--kap_zeta_lo", type=float, default=5e-5)
    p.add_argument("--kap_zeta_hi", type=float, default=1.0)
    p.add_argument("--kap_label_var_lambda", type=float, default=0.0)
    p.add_argument("--kap_label_var_eta", type=float, default=1.0)
    p.add_argument("--stage1_perc", type=float, default=0.5)
    p.add_argument("--grid_points", type=int, default=25)
    p.add_argument("--make_psd_eps", type=float, default=1e-9)
    p.add_argument("--optimize", type=int, default=0,
                   help="0 (DEFAULT) = use the fixed regularisers below; 1 = run the exact classes' LOOCV "
                        "selections, which are only well defined when the landmarks are the whole sample")
    p.add_argument("--tag", type=str, default="sizesweep")
    P = p.parse_args()
    for a in ("kpv_nystrom_stage1_m", "kpv_nystrom_stage2_m",
              "kap_nystrom_stage1_m", "kap_nystrom_stage2_m", "kap_nystrom_stage3_m"):
        if getattr(P, a) <= 0:
            setattr(P, a, P.nystrom_m)
    return P


HYPER_COLUMNS = ["n_sample", "do_a_lo", "do_a_hi", "do_a_size",
                 "nystrom_m", "kpv_nystrom_stage1_m", "kpv_nystrom_stage2_m",
                 "kap_nystrom_stage1_m", "kap_nystrom_stage2_m", "kap_nystrom_stage3_m",
                 "loocv_subsample",
                 "kpv_lambda1", "kpv_lambda2", "kpv_zeta",
                 "kap_lambda", "kap_eta", "kap_lambda2", "kap_zeta",
                 "kpv_lambda1_lo", "kpv_lambda1_hi", "kpv_lambda2_lo", "kpv_lambda2_hi",
                 "kpv_zeta_lo", "kpv_zeta_hi",
                 "kap_lambda_lo", "kap_lambda_hi", "kap_eta_lo", "kap_eta_hi",
                 "kap_zeta_lo", "kap_zeta_hi", "kap_label_var_lambda", "kap_label_var_eta",
                 "stage1_perc", "grid_points", "make_psd_eps", "optimize", "tag"]


def kpv_params(P, model_seed):
    """Fresh kernel objects every call -- the RBF median length scale locks on its first self-Gram."""
    return {
        "kernel_A": RBF(use_length_scale_heuristic=True),
        "kernel_W": RBF(use_length_scale_heuristic=True),
        "kernel_Z": RBF(use_length_scale_heuristic=True),
        "lambda1_": P.kpv_lambda1, "lambda2_": P.kpv_lambda2, "zeta": P.kpv_zeta,
        "nystrom_first_stage_m": P.kpv_nystrom_stage1_m,
        "nystrom_second_stage_m": P.kpv_nystrom_stage2_m,
        "optimize_lambda1_parameter": bool(P.optimize),
        "optimize_lambda2_parameter": bool(P.optimize),
        "optimize_zeta_parameter": bool(P.optimize),
        "lambda1_optimization_range": (P.kpv_lambda1_lo, P.kpv_lambda1_hi),
        "lambda2_optimization_range": (P.kpv_lambda2_lo, P.kpv_lambda2_hi),
        "zeta_optimization_range": (P.kpv_zeta_lo, P.kpv_zeta_hi),
        "stage1_perc": P.stage1_perc,
        "regularization_grid_points": P.grid_points,
        "make_psd_eps": P.make_psd_eps,
        "model_seed": model_seed,
        "loocv_subsample": P.loocv_subsample,
    }


def kap_params(P, model_seed):
    return {
        "kernel_A": RBF(use_length_scale_heuristic=True),
        "kernel_W": ColumnwiseRBF(use_length_scale_heuristic=True),   # per-column bandwidths on the 2-D W
        "kernel_Z": RBF(use_length_scale_heuristic=True),
        "lambda_": P.kap_lambda, "eta": P.kap_eta, "lambda2_": P.kap_lambda2, "zeta": P.kap_zeta,
        "nystrom_first_stage_m": P.kap_nystrom_stage1_m,
        "nystrom_second_stage_m": P.kap_nystrom_stage2_m,
        "nystrom_third_stage_m": P.kap_nystrom_stage3_m,
        "optimize_lambda_parameters": bool(P.optimize),
        "optimize_eta_parameter": bool(P.optimize),
        "optimize_zeta_parameter": bool(P.optimize),
        "lambda_optimization_range": (P.kap_lambda_lo, P.kap_lambda_hi),
        "eta_optimization_range": (P.kap_eta_lo, P.kap_eta_hi),
        "zeta_optimization_range": (P.kap_zeta_lo, P.kap_zeta_hi),
        "stage1_perc": P.stage1_perc,
        "regularization_grid_points": P.grid_points,
        "make_psd_eps": P.make_psd_eps,
        "label_variance_in_lambda_opt": P.kap_label_var_lambda,
        "label_variance_in_eta_opt": P.kap_label_var_eta,
        "model_seed": model_seed,
        "loocv_subsample": P.loocv_subsample,
    }


def run_once(data_seed, train_seed, P):
    """One single-split fit set; returns {(algorithm, anchor): (mse, mae)}, rows, seconds, peak GPU GB."""
    t0 = time.time()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    benchmark = SyntheticLowDimATTBenchmark()
    d = benchmark.generate_dataset(n=P.n_sample, seed=data_seed, anchor_a=P.anchors[0],
                                   do_A_range=(P.do_a_lo, P.do_a_hi), do_A_size=P.do_a_size)
    A, Y, Z, W, do_A = d["A"], d["Y"], d["Z"], d["W"], d["do_A"]
    truth = {a: benchmark.generate_att_curve(anchor_a=a, do_A_range=(P.do_a_lo, P.do_a_hi),
                                             do_A_size=P.do_a_size)[1] for a in P.anchors}

    to_t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float64, device=DEVICE)
    transformers = [TorchStandardScaler() for _ in range(4)]          # A, Y, Z, W
    A_t = transformers[0].fit_transform(to_t(A))
    Y_t = transformers[1].fit_transform(to_t(Y))
    Z_t = transformers[2].fit_transform(to_t(Z))
    W_t = transformers[3].fit_transform(to_t(W))
    do_A_t = transformers[0].transform(to_t(do_A))
    aprime_t = {a: transformers[0].transform(to_t(np.array([[a]]))) for a in P.anchors}

    def scores(pred, a):
        pred = np.asarray(pred).reshape(-1, 1)
        gt = np.asarray(truth[a]).reshape(-1, 1)
        return float(np.mean((pred - gt) ** 2)), float(np.mean(np.abs(pred - gt)))

    out = {}
    if "KPV" in P.algorithms:
        model = KernelProxyVariableATTNystrom(**kpv_params(P, train_seed), device=DEVICE)
        np.random.seed(train_seed)                       # fit() splits off the GLOBAL numpy RNG
        model.fit((A_t, Z_t, W_t), Y_t)                  # torch order (A, Z, W), NOT the reference order (A, W, Z)
        with torch.no_grad():
            for a in P.anchors:
                out[("KPV_Nystrom", a)] = scores(
                    model.predict(do_A_t, aprime_t[a], transformers[1]).cpu().numpy(), a)
        del model

    if "KAP" in P.algorithms:
        for a in P.anchors:                              # a' is baked in at fit time -> one fit per anchor
            model = KernelAlternativeProxyATTNystrom(**kap_params(P, train_seed), device=DEVICE)
            np.random.seed(train_seed)
            model.fit((A_t, Z_t, W_t), Y_t, aprime_t[a])
            with torch.no_grad():
                out[("KAP_Nystrom", a)] = scores(model.predict(do_A_t, transformers[1]).cpu().numpy(), a)
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    peak = (torch.cuda.max_memory_allocated() / 1024 ** 3) if DEVICE == "cuda" else 0.0
    return out, A_t.shape[0], time.time() - t0, peak


if __name__ == "__main__":
    P = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    sd = "s" + "-".join(str(s) for s in P.data_seeds) if len(P.data_seeds) <= 3 else \
         "s%d-%d" % (min(P.data_seeds), max(P.data_seeds))
    out = os.path.join(OUT_DIR, "KPVKAPNystrom_SynthATTLowDim_%s_n%d_%s_%s.csv"
                       % (P.tag, P.n_sample, sd, stamp))
    cols = ["Algorithm", "Anchor", "Data_Seed", "Repeat", "Train_Seed", "Data_Size",
            "Causal_MSE", "Causal_MAE", "Fit_Seconds", "Peak_GPU_GB", "hostname"] + HYPER_COLUMNS
    with open(out, "w", newline="") as f:
        csv.writer(f).writerow(cols)
    hyper = [getattr(P, c) for c in HYPER_COLUMNS]
    print("writing to", out, flush=True)
    for data_seed in P.data_seeds:
        for repeat in range(P.repeat_offset, P.repeat_offset + P.n_repeats):
            train_seed = 1000 * data_seed + repeat
            res, n, secs, peak = run_once(data_seed, train_seed, P)
            with open(out, "a", newline="") as f:        # reopened per cell => crash-safe partial CSVs
                w = csv.writer(f)
                for (algo, anchor), (mse, mae) in res.items():
                    w.writerow([algo, anchor, data_seed, repeat, train_seed, n, mse, mae,
                                round(secs, 2), round(peak, 3), socket.gethostname()] + hyper)
            summary = "  ".join(
                "%s %.5f" % (algo, np.mean([res[(algo + "_Nystrom", a)][0] for a in P.anchors]))
                for algo in P.algorithms)
            print("n=%d seed %d rep %d: %s  (%.0fs, %.2f GB)"
                  % (P.n_sample, data_seed, repeat, summary, secs, peak), flush=True)
    print("wrote", out)
