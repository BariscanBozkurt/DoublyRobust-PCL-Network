"""Demand experiment dose-response with the NYSTROM doubly robust kernel proxy estimator (DRKPV-Nystrom) and its two
bridges (KPV-Nystrom outcome bridge, KAP-Nystrom treatment bridge). Sizes up to 20000 => the Nystrom classes
(src/causal_learning/approximate_kernel_proxy_causal_learning.py, ported) rather than the exact ones.

DATA: generate_demand_experiment_data(size, epsilon_std, do_A_range=(10,30), do_A_size=10, seed=data_seed) -- the SAME
draw (and Monte-Carlo ground truth) as Demand_CrossFitting_Simulation.py and Demand_NMMR_Simulation.py at the same
(data_seed, n_sample), so rows pair with them per (n_sample, Data_Seed, Repeat). A (n,1) price, Z (n,2) costs,
W (n,1) views, Y (n,1). Var(EY_do_A) ~ 47.

PROTOCOL: data seeds 0-9 x --n_repeats 3, train_seed = 1000*data_seed + repeat. The DGP seeds the GLOBAL numpy stream
itself (with data_seed), so np/torch are re-seeded with train_seed AFTER the DGP and right before fit (the abortion
kernel sim's convention): the two bridges' 50/50 stage splits come from that stream, the Nystrom landmarks from
model_seed = train_seed / +1 / +2. Repeats therefore differ in split + landmarks, never in the data.

RECIPE (mirrors dSprite_DRKPV_Nystrom_Simulation.py): TorchStandardScaler on A, Y, Z, W; RBF kernels with the
length-scale heuristic (median by default; --tm_quantile_aw for the treatment bridge's A/W kernels), optional
ColumnwiseRBF on the 2-D Z (--z_kernel colwise); fixed regularisers (no LOOCV -- the DR wrapper strips every tuning
key); --nystrom_m landmarks per stage; make_psd_eps 5e-6; float64 by default (KAP convention); fit tuple (A, Z, W).
HYPERPARAMETERS = the DFPV authors' (Xu et al., 2021; public repository) KPV baseline config for the demand design, configs/kpv.json: lam1 = lam2 = 1e-3
with the n*lambda ridge convention (same as ours), split_ratio 0.5, Gaussian bandwidth = median pairwise distance
(their K = exp(-d^2 / (2 * median(d)^2 * scale)) with scale 0.5 == our RBF median heuristic, quantile 0.5) applied
COLUMN-WISE on every variable (ColumnWiseGaussianKernel => our ColumnwiseRBF on the 2-D Z; RBF == column-wise on the
1-D A and W), standardised data, no LOOCV. The treatment bridge (KAP, not in the DFPV repo) and the DR ridge take the
same 1e-3 (the DRKPV paper's defaults). Kernel methods are known to do poorly on this design (DFPV and NMMR papers);
a screened alternative exists as an opt-in EXTRA, not the default.

Rows per (data_seed, repeat): DRKPV_Nystrom, KPV_Nystrom, KAP_Nystrom; Causal_MSE in RAW Y units over the 10-point
do_A grid vs EY_do_A. Requeue-safe: a restarted task appends to its own CSV and skips finished cells.

  python Demand_DRKPV_Nystrom_Simulation.py --data_seeds 100 --n_repeats 1 --n_sample 2000
"""
import argparse, csv, datetime, glob, os, socket, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
from simulation_utils.data_generating_processes import generate_demand_experiment_data     # noqa: E402
from torch_utils.scalers import TorchStandardScaler                                        # noqa: E402
from torch_utils.kernel_utils import RBF, ColumnwiseRBF                                    # noqa: E402
from causal_learning.approximate_kernel_proxy_causal_learning import (                     # noqa: E402
    DoublyRobustKernelProxyDoseResponseNystrom)

ALGOS = ["DRKPV_Nystrom", "KPV_Nystrom", "KAP_Nystrom"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--n_sample", type=int, default=2000)
    p.add_argument("--n_repeats", type=int, default=3)
    p.add_argument("--repeat_offset", type=int, default=0, help="first repeat index (train_seed = 1000*data_seed + repeat)")
    p.add_argument("--epsilon_std", type=float, default=1.0)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--nystrom_m", type=int, default=500)
    p.add_argument("--z_kernel", choices=["rbf", "colwise"], default="colwise",
                   help="kernel on the 2-D Z (both bridges): one RBF with the median heuristic, or ColumnwiseRBF (per-column bandwidths)")
    p.add_argument("--tm_lambda1", type=float, default=1e-3)
    p.add_argument("--tm_eta", type=float, default=1e-3)
    p.add_argument("--tm_lambda2", type=float, default=1e-3)
    p.add_argument("--tm_quantile_aw", type=float, default=0.5, help="length-scale quantile of the treatment bridge's A and W kernels")
    p.add_argument("--om_lambda1", type=float, default=1e-3)
    p.add_argument("--om_quantile_aw", type=float, default=0.5, help="length-scale quantile of the outcome bridge's A and W kernels")
    p.add_argument("--om_lambda2", type=float, default=1e-3)
    p.add_argument("--lambda_dr", type=float, default=1e-3)
    p.add_argument("--stage1_perc", type=float, default=0.5)
    p.add_argument("--make_psd_eps", type=float, default=5e-6)
    p.add_argument("--tag", default="sizesweep")
    p.add_argument("--out_dir", default=None)
    return p.parse_args(argv)


HYPER = ["n_sample", "epsilon_std", "dtype", "nystrom_m", "z_kernel", "tm_lambda1", "tm_eta", "tm_lambda2",
         "tm_quantile_aw", "om_lambda1", "om_quantile_aw", "om_lambda2", "lambda_dr", "stage1_perc", "make_psd_eps", "tag"]


def _zk(P):
    return ColumnwiseRBF(use_length_scale_heuristic=True) if P.z_kernel == "colwise" else RBF(use_length_scale_heuristic=True)


def run_once(data_seed, train_seed, P):
    dt = getattr(torch, P.dtype)
    U, W, Z, A, Y, do_A, EY = generate_demand_experiment_data(size=P.n_sample, epsilon_std=P.epsilon_std,
                                                              do_A_range=(10, 30), do_A_size=10, seed=data_seed)
    np.random.seed(train_seed); torch.manual_seed(train_seed)          # AFTER the self-seeding DGP: keys the stage splits
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed); torch.cuda.reset_peak_memory_stats()
    dev = P.device
    tt = lambda x: torch.tensor(np.asarray(x), dtype=dt).to(dev)
    A_t, Y_t, Z_t, W_t, do_A_t = tt(A), tt(Y), tt(Z), tt(W), tt(do_A)
    tf = [TorchStandardScaler() for _ in range(4)]
    A_s, Y_s, Z_s, W_s = tf[0].fit_transform(A_t), tf[1].fit_transform(Y_t), tf[2].fit_transform(Z_t), tf[3].fit_transform(W_t)
    do_A_s = tf[0].transform(do_A_t)
    tm = {"kernel_A": RBF(use_length_scale_heuristic=True, length_scale_heuristic_quantile=P.tm_quantile_aw),
          "kernel_W": RBF(use_length_scale_heuristic=True, length_scale_heuristic_quantile=P.tm_quantile_aw),
          "kernel_Z": _zk(P),
          "lambda1_": P.tm_lambda1, "eta": P.tm_eta, "lambda2_": P.tm_lambda2,
          "nystrom_first_stage_m": P.nystrom_m, "nystrom_third_stage_m": P.nystrom_m,
          "stage1_perc": P.stage1_perc, "model_seed": train_seed, "make_psd_eps": P.make_psd_eps, "device": dev}
    om = {"algorithm_name": "Kernel_Proxy_Variable",
          "kernel_A": RBF(use_length_scale_heuristic=True, length_scale_heuristic_quantile=P.om_quantile_aw),
          "kernel_W": RBF(use_length_scale_heuristic=True, length_scale_heuristic_quantile=P.om_quantile_aw),
          "kernel_Z": _zk(P),
          "lambda1_": P.om_lambda1, "lambda2_": P.om_lambda2, "stage1_perc": P.stage1_perc,
          "nystrom_first_stage_m": P.nystrom_m, "nystrom_second_stage_m": P.nystrom_m,
          "model_seed": train_seed + 1, "make_psd_eps": P.make_psd_eps, "device": dev}
    t0 = time.time()
    model = DoublyRobustKernelProxyDoseResponseNystrom(treatment_bridge_params=tm, outcome_bridge_params=om,
                                                       lambda_DR=P.lambda_dr, nystrom_m=P.nystrom_m,
                                                       model_seed=train_seed + 2, make_psd_eps=P.make_psd_eps, device=dev)
    model.fit((A_s, Z_s, W_s), Y_s)
    preds = {"DRKPV_Nystrom": model.predict(do_A_s, tf[1]),
             "KPV_Nystrom": model.outcome_bridge_algo.predict(do_A_s, tf[1]),
             "KAP_Nystrom": model.treatment_bridge_algo.predict(do_A_s, tf[1])}
    secs = time.time() - t0
    gt = np.asarray(EY, dtype=np.float64).reshape(-1, 1)
    mse = {k: float(np.mean((v.detach().cpu().numpy().reshape(-1, 1) - gt) ** 2)) for k, v in preds.items()}
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else float("nan")
    del model, A_t, Y_t, Z_t, W_t, A_s, Y_s, Z_s, W_s
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return mse, secs, peak


def main():
    P = parse_args()
    torch.set_default_dtype(getattr(torch, P.dtype))
    out_dir = P.out_dir or os.path.join(HERE, "..", "Results", "Demand_DRKPV_Nystrom")
    os.makedirs(out_dir, exist_ok=True)
    sd = "s" + "-".join(str(s) for s in P.data_seeds) if len(P.data_seeds) <= 3 else "s%d-%d" % (min(P.data_seeds), max(P.data_seeds))
    stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    prefix = "DRKPV_Nystrom_Demand_n%d_%s_%s_" % (P.n_sample, P.tag, sd)
    out = os.path.join(out_dir, prefix + stamp + ".csv")
    fields = ["Algorithm", "Data_Seed", "Repeat", "Train_Seed", "Causal_MSE", "Fit_Seconds", "Peak_GPU_GB", "hostname"] + HYPER
    done, resumed = set(), False
    prev = sorted(glob.glob(os.path.join(out_dir, prefix + "*.csv")))
    if prev:
        with open(prev[-1], newline="") as f:
            rd = csv.DictReader(f); rows = list(rd) if rd.fieldnames == fields else None
        if rows is not None:
            seen = {}
            for r in rows:
                seen.setdefault((int(r["Data_Seed"]), int(r["Repeat"])), set()).add(r["Algorithm"])
            done = {k for k, v in seen.items() if set(ALGOS) <= v}
            out, resumed = prev[-1], True
            print("RESUME: appending to %s; finished cells skipped: %s" % (os.path.basename(out), sorted(done)), flush=True)
    if not resumed:
        with open(out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
    meta = {k: getattr(P, k) for k in HYPER}; host = socket.gethostname()
    print("demand | n %d | dtype %s | device %s | seeds %s x repeats %d" % (P.n_sample, P.dtype, P.device, P.data_seeds, P.n_repeats), flush=True)
    for data_seed in P.data_seeds:
        for repeat in range(P.repeat_offset, P.repeat_offset + P.n_repeats):
            train_seed = 1000 * data_seed + repeat
            if (data_seed, repeat) in done:
                print("skip data_seed %d rep %d (already in the CSV)" % (data_seed, repeat), flush=True); continue
            mse, secs, peak = run_once(data_seed, train_seed, P)
            with open(out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                for algo in ALGOS:
                    w.writerow(dict(Algorithm=algo, Data_Seed=data_seed, Repeat=repeat, Train_Seed=train_seed,
                                    Causal_MSE=mse[algo], Fit_Seconds=round(secs, 1), Peak_GPU_GB=round(peak, 2), hostname=host, **meta))
            print("data_seed %d rep %d (%.1fs, peak %.2f GB): %s" % (data_seed, repeat, secs, peak,
                  "  ".join("%s %.4f" % (a, mse[a]) for a in ALGOS)), flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
