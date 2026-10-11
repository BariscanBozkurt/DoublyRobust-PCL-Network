"""NYSTROM kernel baseline (KPV) for the synthetic CATE benchmark, data-size sweep.

WHY THIS EXISTS: the synthetic CATE experiment had no kernel row at all. This
script supplies it, so the CATE table carries the same three families as every other benchmark --
neural (`Synthetic_CATE_CrossFitting_Simulation.py`), NMMR where it exists, and kernel.

KPV ONLY, AND THAT IS NOT AN OMISSION. The heterogeneous-response kernel family publishes only the
OUTCOME bridge (`KernelProxyVariableHeterogeneousResponse`). There is no published KAP -- hence no
DRKPV -- for a CATE estimand, which is exactly why this file lives in `Simulations/KPV/` and not in
`Simulations/DRKPV/` (the same reason the synthetic HIGH-DIM V2 KPV sim does: there, KAP is undefined
with a backdoor X). Do not add a KAP or DRKPV row here; it would have to be invented.

WHAT IS APPROXIMATED (`causal_learning/approximate_kernel_proxy_causal_learning.py::
KernelProxyVariableHeterogeneousResponseNystrom`): the stage-1 conditional-mean-embedding solve, the
stage-2 outcome-bridge solve, and the moderator CME of `predict` (the zeta-regularised projection of
V onto the stage-2 representers). The class still forms full n x n Grams; the landmarks shrink the
SOLVES, which is where the O(n^3) sits.

REGULARISERS ARE FIXED BY HAND -- NO LOOCV (design decision, and the convention of every
Nystrom sim in this repo: dSprite, demand, ATT). The exact class's selections are hat-matrix
leave-one-out identities, h_i = (I - R)_ii with the 1/diag(I - R) correction, derived for the EXACT
ridge solution; under a low-rank Nystrom solve those leverage terms are no longer the leave-one-out
residuals of the estimator being fitted, so the objective is not well defined. `--optimize 1` keeps
the selections reachable -- they ARE exact, and therefore meaningful, when the landmark set is the
whole sample, which is how the acceptance test below uses them -- but the REPORTING DEFAULT is
`--optimize 0` with the values frozen by the screen recorded in
a screen whose selection rule was declared in writing before any result was read.

ACCEPTANCE. Every Nystrom solve is
K_nm (K_mn K_nm + n lam K_mm)^-1 T_m, which at landmarks = the whole sample is algebraically the
exact (K_nn + n lam I)^-1 T. So at `--nystrom_m >= n` this class MUST reproduce
`KernelProxyVariableHeterogeneousResponse` to solver tolerance. See the README for the measured
number. Agreement is NOT expected at machine precision -- the Nystrom normal equations square the
Gram's condition number.

PAIRING WITH THE NEURAL CATE GRID -- verified by reading both scripts, not assumed:
  * same DGP call, `generate_synthetic_CATE_data(n, sigma, +noise_bound, -noise_bound, seed=data_seed)`;
  * same evaluation grid: the DGP's own `covariate_v_test` (100 points on [-0.5, 0.5]) at `do_A = 1`;
  * same ground truth `EY_do_A_CATE = v (1+2v)^2 (v-1)^2`, closed form;
  * same metric, in RAW Y units: mean((pred - EY)^2) over the 100 grid points. `predict` applies
    `outcome_transformer.inverse_transform` INTERNALLY, so passing the Y scaler is what puts the
    prediction back in raw units -- matching the neural sim's `pred_structural_function(..., tf_Y)`;
  * same seeds and repeats: 10 data seeds x 3 repeats, train_seed = 1000*data_seed + repeat;
  * same sizes {2,5,10,15,20}x1000 (the neural launcher's SIZES).
PROTOCOL CAVEAT TO STATE WHEREVER THESE ROWS ARE REPORTED: the kernel method uses ONE 50/50 split
(`--stage1_perc 0.5`) while the neural rows are 3-fold cross-fitting, so equal NOMINAL n is not equal
data per stage. Same caveat as the kernel rows in analysis notebooks 5 / 9 / 11.

TRAPS RESPECTED HERE (each previously paid for in this repo):
  * FIT TUPLE ORDER is the TORCH order `(A, Z, W, V)`. The reference the reference implementation class takes `(A, W, Z, V)`;
    verbatim reference code runs WITHOUT ERROR and silently swaps the proxies.
  * A is BINARY -> `BinaryKernel` (the notebook's choice). The other four kernels are RBF with the
    median heuristic. RBF kernels are STATEFUL -- the length scale locks on the first self-Gram call
    -- so `kpv_params()` builds FRESH kernel objects on every call; never reuse an instance.
  * `fit()` draws its stage split from the GLOBAL numpy RNG, so `np.random.seed(train_seed)` is called
    immediately before it. The DGP uses its own `default_rng(seed)` and does not touch that stream.
  * Landmarks are drawn from `model_seed` with an INDEPENDENT generator inside the class, so they
    never disturb the split stream.
  * An unknown ctor key is silently swallowed by `**kwargs`. `_assert_params_landed` therefore checks
    after construction that every value actually reached the model.

  python Synthetic_CATE_KPV_Nystrom_Simulation.py --data_seeds 42 --n_repeats 1 --n_sample 2000
  python Synthetic_CATE_KPV_Nystrom_Simulation.py --data_seeds 0 --n_sample 20000 --nystrom_m 500
"""
import argparse, csv, datetime, glob, os, socket, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))

from simulation_utils.data_generating_processes import generate_synthetic_CATE_data
from torch_utils.scalers import TorchStandardScaler
from torch_utils.kernel_utils import RBF, BinaryKernel
from causal_learning.approximate_kernel_proxy_causal_learning import (
    KernelProxyVariableHeterogeneousResponseNystrom)

ALGOS = ["KPV_Nystrom"]          # registered in visualization_utils as `KPV (Nystrom)`


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--n_repeats", type=int, default=3)
    p.add_argument("--repeat_offset", type=int, default=0)
    # -- data: the neural CATE sim's defaults, so the datasets are numerically identical at each data_seed --
    p.add_argument("--n_sample", type=int, default=2000)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--noise_bound", type=float, default=0.1,
                   help="uniform proxy noise in [-noise_bound, +noise_bound]")
    # -- Nystrom knobs --
    p.add_argument("--nystrom_m", type=int, default=500,
                   help="landmarks per approximated solve; the per-slot flags below override it")
    p.add_argument("--nystrom_stage1_m", type=int, default=0, help="0 = --nystrom_m")
    p.add_argument("--nystrom_stage2_m", type=int, default=0,
                   help="0 = --nystrom_m; also carries the moderator CME of predict")
    p.add_argument("--loocv_subsample", type=int, default=2000,
                   help="rows used by a regulariser LOOCV (0 = all rows = the exact class); "
                        "irrelevant at the default --optimize 0")
    # -- FIXED regularisers, used when --optimize 0 (the reporting default). FROZEN by the
    # CONFIRMATION on fresh seeds 103-105 at both sizes; every rule was declared in writing before any
    # result was read. Reporting seeds 0-9 were
    # never touched. HONEST CAVEAT, and it is the main finding of that screen: this estimator is
    # nearly INSENSITIVE to these values -- the whole top-10 of the n=2000 grid spans 0.000619-0.000682
    # (10%) while the spread ACROSS SEEDS within one cell is 0.0004-0.0010 (2.5x). Only badly mismatched
    # corners hurt (lambda1 1e-2 with lambda2 1e-5: 0.0039, a 6x penalty). So do not read these three
    # numbers as a finely tuned optimum; read them as "anywhere in 1e-4..1e-3 with zeta 1e-2". --
    p.add_argument("--lambda1", type=float, default=1e-4, help="stage-1 CME ridge")
    p.add_argument("--lambda2", type=float, default=1e-4, help="stage-2 outcome-bridge ridge")
    p.add_argument("--zeta", type=float, default=1e-2, help="moderator-CME ridge used by predict")
    p.add_argument("--lambda1_lo", type=float, default=1e-5)
    p.add_argument("--lambda1_hi", type=float, default=1.0)
    p.add_argument("--lambda2_lo", type=float, default=1e-5)
    p.add_argument("--lambda2_hi", type=float, default=1.0)
    p.add_argument("--zeta_lo", type=float, default=1e-5)
    p.add_argument("--zeta_hi", type=float, default=1.0)
    p.add_argument("--label_var_lambda", type=float, default=0.0)
    # -- shared kernel/solver settings: the notebook's values --
    p.add_argument("--stage1_perc", type=float, default=0.5)
    p.add_argument("--grid_points", type=int, default=25)
    p.add_argument("--make_psd_eps", type=float, default=5e-8)
    p.add_argument("--optimize", type=int, default=0,
                   help="0 (DEFAULT) = the fixed regularisers above; 1 = the exact class's LOOCV "
                        "selections, which are only well defined when the landmarks are the whole sample")
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64",
                   help="float64, AND DO NOT CHANGE IT -- float32 is BROKEN for the Nystrom solve "
                        "here (measured: causal MSE 0.00073 -> 0.00228 at n=2000 and 0.00025 -> "
                        "0.00045 at n=5000, i.e. 3.1x and 1.8x wrong, with 28%% / 7.5%% relative "
                        "curve error). The EXACT class IS dtype-insensitive on this benchmark (f32 vs "
                        "f64 agree to 5.06e-06), which is exactly the trap: it is the LOW-RANK solve "
                        "that is ill-conditioned, because the Nystrom normal equations square the "
                        "Gram's condition number. Same shape as the KERNEL KAP NEEDS FLOAT64 rule. "
                        "float32 is kept only as a last-resort memory escape, and any row produced "
                        "with it must be reported as a different estimator, not a cheaper one.")
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument("--tag", type=str, default="sizesweep")
    P = p.parse_args()
    for a in ("nystrom_stage1_m", "nystrom_stage2_m"):
        if getattr(P, a) <= 0:
            setattr(P, a, P.nystrom_m)
    return P


HYPER_COLUMNS = ["n_sample", "sigma", "noise_bound",
                 "nystrom_m", "nystrom_stage1_m", "nystrom_stage2_m", "loocv_subsample",
                 "lambda1", "lambda2", "zeta",
                 "lambda1_lo", "lambda1_hi", "lambda2_lo", "lambda2_hi", "zeta_lo", "zeta_hi",
                 "label_var_lambda", "stage1_perc", "grid_points", "make_psd_eps",
                 "optimize", "dtype", "tag"]


def kpv_params(P, model_seed):
    """FRESH kernel objects on every call -- an RBF's median length scale locks on its first self-Gram.

    A is BINARY, so it goes through BinaryKernel (the reference notebook's choice). Note that A is
    standardised first, so BinaryKernel's x y' + (1-x)(1-y)' is not literally a match indicator; on a
    two-point support it is still a full-rank rank-2 kernel, i.e. it spans every function of A. This
    is exactly what the reference notebook does and what its 0.000542 reproduces.
    """
    return {
        "kernel_A": BinaryKernel(),
        "kernel_W": RBF(use_length_scale_heuristic=True),
        "kernel_Z": RBF(use_length_scale_heuristic=True),
        "kernel_V": RBF(use_length_scale_heuristic=True),
        "kernel_X": RBF(use_length_scale_heuristic=True),        # inert: this benchmark has no backdoor X
        "lambda1_": P.lambda1, "lambda2_": P.lambda2, "zeta": P.zeta,
        "nystrom_first_stage_m": P.nystrom_stage1_m,
        "nystrom_second_stage_m": P.nystrom_stage2_m,
        "optimize_lambda1_parameter": bool(P.optimize),
        "optimize_lambda2_parameter": bool(P.optimize),
        "optimize_zeta_parameter": bool(P.optimize),
        "lambda1_optimization_range": (P.lambda1_lo, P.lambda1_hi),
        "lambda2_optimization_range": (P.lambda2_lo, P.lambda2_hi),
        "zeta_optimization_range": (P.zeta_lo, P.zeta_hi),
        "stage1_perc": P.stage1_perc,
        "regularization_grid_points": P.grid_points,
        "make_psd_eps": P.make_psd_eps,
        "label_variance_in_lambda_opt": P.label_var_lambda,
        "model_seed": model_seed,
        "loocv_subsample": P.loocv_subsample,
    }


def _assert_params_landed(model, params):
    """An unknown ctor key is SILENTLY SWALLOWED by **kwargs -- check the values actually arrived.

    Attribute names differ from the ctor keys for the trailing-underscore ridges only.
    """
    attr_of = {"lambda1_": "lambda1_", "lambda2_": "lambda2_", "zeta": "zeta",
               "nystrom_first_stage_m": "nystrom_first_stage_m",
               "nystrom_second_stage_m": "nystrom_second_stage_m",
               "stage1_perc": "stage1_perc", "regularization_grid_points": "regularization_grid_points",
               "make_psd_eps": "make_psd_eps", "model_seed": "model_seed",
               "loocv_subsample": "loocv_subsample",
               "label_variance_in_lambda_opt": "label_variance_in_lambda_opt",
               "optimize_lambda1_parameter": "optimize_lambda1_parameter",
               "optimize_lambda2_parameter": "optimize_lambda2_parameter",
               "optimize_zeta_parameter": "optimize_zeta_parameter"}
    missing = []
    for key, attr in attr_of.items():
        if key not in params:
            continue
        if not hasattr(model, attr):
            missing.append("%s (no attribute %s)" % (key, attr))
        elif getattr(model, attr) != params[key]:
            missing.append("%s: passed %r, model holds %r" % (key, params[key], getattr(model, attr)))
    if missing:
        raise RuntimeError("ctor keys did not land on the model (silently swallowed by **kwargs):\n  "
                           + "\n  ".join(missing))


def run_once(data_seed, train_seed, P, device, dtype):
    """One single-split fit; returns {algorithm: causal MSE}, seconds, peak GPU GB."""
    t0 = time.time()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # The DGP carries its own default_rng(seed) and does NOT consume the global numpy stream.
    U, W, Z, V, A, Y, v_test, do_A, EY = generate_synthetic_CATE_data(
        P.n_sample, P.sigma, P.noise_bound, -P.noise_bound, seed=data_seed)

    to_t = lambda x: torch.tensor(np.asarray(x), dtype=dtype, device=device).reshape(len(x), -1)
    tf = [TorchStandardScaler() for _ in range(5)]                       # A, Y, Z, W, V
    A_t = tf[0].fit_transform(to_t(A))
    Y_t = tf[1].fit_transform(to_t(Y))
    Z_t = tf[2].fit_transform(to_t(Z))
    W_t = tf[3].fit_transform(to_t(W))
    V_t = tf[4].fit_transform(to_t(V))
    doA_t = tf[0].transform(to_t(do_A))                                  # the DGP's grid: 100 x A=1
    vte_t = tf[4].transform(to_t(v_test))                                # 100 moderator points

    params = kpv_params(P, train_seed)
    model = KernelProxyVariableHeterogeneousResponseNystrom(**params, device=device)
    _assert_params_landed(model, params)

    np.random.seed(train_seed)               # fit() splits off the GLOBAL numpy RNG
    #... AND the TORCH RNG: `RBF.calculate_and_fix_length_scale` subsamples with `torch.randperm`
    # whenever the sample exceeds `heuristic_subset_limit` (5000), so at n >= 10000 -- THREE of the
    # five sweep sizes -- the median length scale of W, Z and V is a fresh random draw on every
    # invocation. MEASURED without this line: three identical invocations at n=10000 seed 3 gave
    # 9.3171e-05 / 9.2863e-05 / 9.3549e-05 (0.74% spread); with it they are numerically identical. Inert at
    # n <= 5000, where no torch RNG is consumed (landmarks use their own default_rng(model_seed) and
    # the stage split uses the global numpy stream) -- gated: n=2000 seed 42 rep 0 is 0.000732 either
    # way. The neural CATE sim seeds both streams in `build_data` for the same reason.
    torch.manual_seed(train_seed)
    model.fit((A_t, Z_t, W_t, V_t), Y_t)     # TORCH order (A, Z, W, V), NOT the reference order (A, W, Z, V)
    with torch.no_grad():
        # predict inverse-transforms with the Y scaler INTERNALLY -> raw Y units, as the neural sim
        pred = model.predict(doA_t, vte_t, tf[1]).detach().cpu().numpy().reshape(-1, 1)

    gt = np.asarray(EY, dtype=np.float64).reshape(-1, 1)
    mse = {"KPV_Nystrom": float(np.mean((pred.astype(np.float64) - gt) ** 2))}
    selected = (float(model.lambda1_), float(model.lambda2_), float(model.zeta))
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 3) if device == "cuda" else float("nan")
    del model, A_t, Y_t, Z_t, W_t, V_t
    if device == "cuda":
        torch.cuda.empty_cache()
    return mse, selected, time.time() - t0, peak


def main():
    P = parse_args()
    dtype = getattr(torch, P.dtype)
    torch.set_default_dtype(dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if P.dtype == "float32":
        print("*** WARNING: --dtype float32 is BROKEN for this Nystrom solve (measured 3.1x wrong "
              "causal MSE at n=2000). See the --dtype help. These rows are NOT comparable to the "
              "float64 grid. ***", flush=True)

    out_dir = P.out_dir or os.path.abspath(os.path.join(HERE, "..", "Results",
                                                        "Synthetic_CATE_KPV_Nystrom"))
    os.makedirs(out_dir, exist_ok=True)
    sd = ("s" + "-".join(str(s) for s in P.data_seeds)) if len(P.data_seeds) <= 3 else \
         ("s%d-%d" % (min(P.data_seeds), max(P.data_seeds)))
    prefix = "KPVNystrom_SyntheticCATE_%s_n%d_%s_" % (P.tag, P.n_sample, sd)
    stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    out = os.path.join(out_dir, prefix + stamp + ".csv")
    fields = (["Algorithm", "Data_Seed", "Repeat", "Train_Seed", "Data_Size", "Causal_MSE",
               "sel_lambda1", "sel_lambda2", "sel_zeta", "Fit_Seconds", "Peak_GPU_GB", "hostname"]
              + HYPER_COLUMNS)

    # this (tag, n, seeds) whose header matches, and skip every (data_seed, repeat) already complete.
    done, resumed = set(), False
    prev = sorted(glob.glob(os.path.join(out_dir, prefix + "*.csv")))
    if prev:
        with open(prev[-1], newline="") as f:
            rd = csv.DictReader(f)
            rows = list(rd) if rd.fieldnames == fields else None
        if rows is not None:
            seen = {}
            for r in rows:
                seen.setdefault((int(r["Data_Seed"]), int(r["Repeat"])), set()).add(r["Algorithm"])
            done = {k for k, v in seen.items() if set(ALGOS) <= v}
            out, resumed = prev[-1], True
            print("RESUME: appending to %s; finished cells skipped: %s"
                  % (os.path.basename(out), sorted(done)), flush=True)
    if not resumed:
        with open(out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    meta = {k: getattr(P, k) for k in HYPER_COLUMNS}
    host = socket.gethostname()
    print("CATE KPV-Nystrom | n %d | m %d/%d | dtype %s | device %s | optimize %d | seeds %s x %d repeats"
          % (P.n_sample, P.nystrom_stage1_m, P.nystrom_stage2_m, P.dtype, device, P.optimize,
             P.data_seeds, P.n_repeats), flush=True)
    print("writing to", out, flush=True)

    for data_seed in P.data_seeds:
        for repeat in range(P.repeat_offset, P.repeat_offset + P.n_repeats):
            train_seed = 1000 * data_seed + repeat
            if (data_seed, repeat) in done:
                print("skip data_seed %d rep %d (already in the CSV)" % (data_seed, repeat), flush=True)
                continue
            mse, sel, secs, peak = run_once(data_seed, train_seed, P, device, dtype)
            with open(out, "a", newline="") as f:          # reopened per cell => crash-safe partials
                w = csv.DictWriter(f, fieldnames=fields)
                for algo in ALGOS:
                    w.writerow(dict(Algorithm=algo, Data_Seed=data_seed, Repeat=repeat,
                                    Train_Seed=train_seed, Data_Size=P.n_sample,
                                    Causal_MSE=mse[algo], sel_lambda1=sel[0], sel_lambda2=sel[1],
                                    sel_zeta=sel[2], Fit_Seconds=round(secs, 2),
                                    Peak_GPU_GB=round(peak, 3), hostname=host, **meta))
            print("n=%d seed %d rep %d: %s  (%.1fs, %.2f GB)"
                  % (P.n_sample, data_seed, repeat,
                     "  ".join("%s %.6f" % (a, mse[a]) for a in ALGOS), secs, peak), flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
