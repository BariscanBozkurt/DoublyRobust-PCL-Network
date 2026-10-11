"""KERNEL baselines (KPV + KAP + DRKPV) on Abortion & Crime -- the kernel-family counterpart of
DR-NME-PCL-Dens-Ratio-Free/Abortion_CrossFitting_Simulation.py, for the paired analysis notebooks.

PROTOCOL. Single-split (the kernel methods' published protocol -- no cross-fitting), 10 benchmark data
seeds x --n_repeats method repeats, train_seed = 1000*data_seed + repeat (the repo convention; the
method seed drives the stage-1/stage-2 split permutation inside every fit, which is drawn from the
GLOBAL numpy RNG). One row per (Algorithm, Data_Seed, Repeat) appended incrementally; each invocation
writes its OWN tag+seed+timestamped CSV (no parallel write collisions).

HYPERPARAMETERS. The dicts below are the DRKPV reference implementation's SIMULATION configuration
(the corresponding script of the reference implementation),
which produced the reference values (KPV 0.0204 / KAP 0.0271 / DRKPV 0.0185). It
differs from the reference NOTEBOOK configuration in three load-bearing ways -- all-RBF kernels (no
ColumnwiseRBF on Z), lambda floors at 5e-5 (the notebook's 1e-6 floor makes KAP blow up on 5/10 seeds:
seed 3 KAP=624), and grid 25 with label_variance_in_eta_opt=1.0. Do not "widen" the ranges.

IMPLEMENTATION NOTES (PyTorch): fit tuple order is (A, Z, W) -- the reference implementation's order (A, W, Z)
runs silently and corrupts; the KAP stage-1 ridge is `lambda1_` (reference implementation: `lambda_`); float64 because
make_psd_eps = 1e-9 is sub-resolution in float32; predictions come back in the TRANSFORMED Y space and
are inverse-transformed here with the sklearn Y scaler.
"""
import argparse
import csv
import os
import socket
import sys
import time

import numpy as np
import torch
from sklearn import preprocessing

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))

from simulation_utils.data_generating_processes import read_legalized_abortion_and_crime_dataset
from torch_utils.kernel_utils import RBF
from causal_learning.doubly_robust_kernel_proxy_causal_learning import DoublyRobustKernelProxyATE

torch.set_default_dtype(torch.float64)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = os.path.abspath(os.path.join(HERE, "..", "..", "data", "abortion"))
OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "Results", "Abortion_DRKPV"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_seeds", type=int, nargs="+", default=list(range(10)),
                   help="benchmark .npz split seeds (0-9)")
    p.add_argument("--n_repeats", type=int, default=3, help="method repeats per data seed")
    p.add_argument("--repeat_offset", type=int, default=0,
                   help="first repeat index (train_seed = 1000*data_seed + repeat)")
    # every scalar of the two bridge dicts, exposed and written to the CSV
    p.add_argument("--kap_lambda_lo", type=float, default=5e-5)
    p.add_argument("--kap_lambda_hi", type=float, default=1.0)
    p.add_argument("--kap_eta_lo", type=float, default=5e-5)
    p.add_argument("--kap_eta_hi", type=float, default=1.0)
    p.add_argument("--kap_label_var_lambda", type=float, default=0.0)
    p.add_argument("--kap_label_var_eta", type=float, default=1.0)
    p.add_argument("--kpv_lambda_lo", type=float, default=5e-5)
    p.add_argument("--kpv_lambda_hi", type=float, default=1.0)
    p.add_argument("--dr_lambda_lo", type=float, default=5e-5)
    p.add_argument("--dr_lambda_hi", type=float, default=1.0)
    p.add_argument("--stage1_perc", type=float, default=0.5)
    p.add_argument("--grid_points", type=int, default=25)
    p.add_argument("--make_psd_eps", type=float, default=1e-9)
    p.add_argument("--a_min", type=float, default=0.0,
                   help="treatment support lower bound; our neural sim uses A_RANGE = (0.0, 2.0)")
    p.add_argument("--a_max", type=float, default=2.0)
    p.add_argument("--tag", default="baseline", help="config label; goes in filename and CSV")
    return p.parse_args()


def data_transform(X):
    """= the DRKPV reference implementation's utils.ml_utils.data_transform('Standard')."""
    scaler = preprocessing.StandardScaler()
    if X.ndim == 1:
        return scaler.fit_transform(X.reshape(-1, 1)).squeeze(), scaler
    return scaler.fit_transform(X).squeeze(), scaler


def run_once(data_seed, train_seed, P):
    """One single-split kernel fit; returns {algorithm: (mse, mae)} plus the run time."""
    np.random.seed(train_seed)
    W, Z, A, Y, do_A, EY_do_A = read_legalized_abortion_and_crime_dataset(DATA_PATH, seed=data_seed)
    # A-range mask, identical to Abortion_CrossFitting_Simulation.py (A_RANGE = (0.0, 2.0), strict
    # inequalities): the benchmark's effective support, and the do_A grid lies inside it. Keeps
    # 1419 of 1500 rows. Added so the kernel and neural families fit the same support;
    keep = ((A > P.a_min) & (A < P.a_max)).reshape(-1)
    W, Z, A, Y = W[keep], Z[keep], A[keep], Y[keep]

    A_tr, A_scaler = data_transform(A)
    Z_tr, _ = data_transform(Z)
    W_tr, _ = data_transform(W)
    Y_tr, Y_scaler = data_transform(Y)
    n = A_tr.shape[0]
    to_t = lambda x: torch.tensor(np.asarray(x, dtype=np.float64).reshape(n, -1), device=DEVICE)
    A_t, Z_t, W_t, Y_t = to_t(A_tr), to_t(Z_tr), to_t(W_tr), to_t(Y_tr)
    do_A_t = torch.tensor(A_scaler.transform(do_A.reshape(-1, 1)), dtype=torch.float64, device=DEVICE)

    # ==== the DRKPV reference implementation's STABLE simulation dicts (fresh kernel objects every fit -- the RBF
    # median-heuristic length scale locks on its first self-Gram call) ==========================
    treatment_bridge_params = {                                   # KAP
        "kernel_A": RBF(use_length_scale_heuristic=True),
        "kernel_W": RBF(use_length_scale_heuristic=True),
        "kernel_Z": RBF(use_length_scale_heuristic=True),
        "lambda1_": 1e-3,                 # the reference implementation name: lambda_ (renamed in the torch port)
        "eta": 1e-3,
        "lambda2_": 1e-3,
        "optimize_lambda_parameters": True,
        "optimize_eta_parameter": True,
        "lambda_optimization_range": (P.kap_lambda_lo, P.kap_lambda_hi),
        "eta_optimization_range": (P.kap_eta_lo, P.kap_eta_hi),
        "stage1_perc": P.stage1_perc,
        "regularization_grid_points": P.grid_points,
        "make_psd_eps": P.make_psd_eps,
        "label_variance_in_lambda_opt": P.kap_label_var_lambda,
        "label_variance_in_eta_opt": P.kap_label_var_eta,
    }
    outcome_bridge_params = {                                     # KPV
        "kernel_A": RBF(use_length_scale_heuristic=True),
        "kernel_W": RBF(use_length_scale_heuristic=True),
        "kernel_Z": RBF(use_length_scale_heuristic=True),
        "lambda1_": 0.1,
        "lambda2_": 0.1,
        "optimize_lambda1_parameter": True,
        "optimize_lambda2_parameter": True,
        "lambda1_optimization_range": (P.kpv_lambda_lo, P.kpv_lambda_hi),
        "lambda2_optimization_range": (P.kpv_lambda_lo, P.kpv_lambda_hi),
        "stage1_perc": P.stage1_perc,
        "regularization_grid_points": P.grid_points,
        "make_psd_eps": P.make_psd_eps,
    }
    model_DR = DoublyRobustKernelProxyATE(
        treatment_bridge_params=treatment_bridge_params,
        outcome_bridge_params=outcome_bridge_params,
        lambda_DR=1e-3,
        optimize_lambda_DR_parameter=True,
        lambda_DR_optimization_range=(P.dr_lambda_lo, P.dr_lambda_hi),
        regularization_grid_points=P.grid_points,
        make_psd_eps=P.make_psd_eps,
        device=DEVICE,
    )

    t0 = time.time()
    np.random.seed(train_seed)         # the stage splits inside fit() draw from the GLOBAL numpy RNG
    model_DR.fit((A_t, Z_t, W_t), Y_t)  # reference tuple order (A, Z, W); the reference implementation order corrupts silently
    fit_secs = time.time() - t0

    gt = EY_do_A.reshape(-1, 1)
    def scores(pred_t):
        pred = Y_scaler.inverse_transform(pred_t.detach().cpu().numpy().reshape(-1, 1))
        return float(np.mean((pred - gt) ** 2)), float(np.mean(np.abs(pred - gt)))
    with torch.no_grad():
        out = {"KPV": scores(model_DR.outcome_bridge_algo.predict(do_A_t)),
               "KAP": scores(model_DR.treatment_bridge_algo.predict(do_A_t)),
               "DRKPV": scores(model_DR.predict(do_A_t))}
    return out, n, fit_secs


HYPER_COLUMNS = ["a_min", "a_max", "kap_lambda_lo", "kap_lambda_hi", "kap_eta_lo", "kap_eta_hi",
                 "kap_label_var_lambda", "kap_label_var_eta", "kpv_lambda_lo", "kpv_lambda_hi",
                 "dr_lambda_lo", "dr_lambda_hi", "stage1_perc", "grid_points", "make_psd_eps", "tag"]

if __name__ == "__main__":
    P = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    sd = "s" + "-".join(str(s) for s in P.data_seeds) if len(P.data_seeds) <= 3 else \
         "s%d-%d" % (min(P.data_seeds), max(P.data_seeds))
    out = os.path.join(OUT_DIR, "DRKPV_Abortion_%s_%s_%s.csv" % (P.tag, sd, stamp))
    cols = ["Algorithm", "Data_Seed", "Repeat", "Train_Seed", "Data_Size",
            "Causal_MSE", "Causal_MAE", "Fit_Seconds", "hostname"] + HYPER_COLUMNS
    with open(out, "w", newline="") as f:
        csv.writer(f).writerow(cols)
    hyper = [getattr(P, c) for c in HYPER_COLUMNS]
    for data_seed in P.data_seeds:
        for repeat in range(P.repeat_offset, P.repeat_offset + P.n_repeats):
            train_seed = 1000 * data_seed + repeat
            res, n, secs = run_once(data_seed, train_seed, P)
            with open(out, "a", newline="") as f:
                w = csv.writer(f)
                for algo, (mse, mae) in res.items():
                    w.writerow([algo, data_seed, repeat, train_seed, n, mse, mae,
                                round(secs, 2), socket.gethostname()] + hyper)
            print("seed %d rep %d: %s  (%.1fs)"
                  % (data_seed, repeat,
                     "  ".join("%s %.5f" % (a, m[0]) for a, m in res.items()), secs), flush=True)
    print("wrote", out)
