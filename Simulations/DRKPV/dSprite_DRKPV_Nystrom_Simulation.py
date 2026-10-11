"""dSprite dose-response with the NYSTROM doubly robust kernel proxy estimator (DRKPV-Nystrom), plus its two
bridges (KPV-Nystrom outcome bridge, KAP-Nystrom treatment bridge), on the repo's dSprite benchmarks.

PORT of an earlier internal version -- the script behind the 30-seed
Nystrom record (flat ~21.6 DRKPV_Nystrom at every size, float32, cuda). The estimator lives in
src/causal_learning/approximate_kernel_proxy_causal_learning.py (numerically identical to an earlier internal version except one added
import). Recipe transcribed VERBATIM: TorchStandardScaler on A (4096-D image), Y, Z, W (4096-D image); treatment
bridge RBF kernels with the 0.95-quantile length-scale heuristic on A and W and the median on Z; outcome bridge
median heuristics everywhere; fixed regularisers (no LOOCV -- the DR wrapper strips every tuning key); Nystrom
m=500 landmarks per stage; make_psd_eps 5e-6; fit tuple order (A, Z, W); model_seed = seed / seed+1 / seed+2 for
the treatment / outcome / DR landmark RNGs.

PROTOCOL (repo standard): data seeds 0-9 x --n_repeats 3, train_seed = 1000*data_seed + repeat. np/torch/cuda are
seeded with train_seed BEFORE the DGP (the DGP draws from its OWN default_rng(data_seed)), so the first global
draws are the two bridges' 50/50 stage splits and the length-scale heuristic subsample (n > 5000), exactly as
the old script -- data_seed 0, repeat 0 (train_seed 0) reproduces the old seed-0 run. --dtype float64 is the
DEFAULT (convention: any KAP row runs float64); --dtype float32 reproduces earlier float32 runs (~0.2% drift, cuda).

MEMORY: the "Nystrom" classes still materialise the full n x n Grams (A, W, Z and their products); landmarks only
shrink the linear solves. Peak GPU memory is written to every row (Peak_GPU_GB) -- use it to choose the GPU for each size.

Rows per (data_seed, repeat): DRKPV_Nystrom, KPV_Nystrom, KAP_Nystrom; Causal_MSE in RAW Y units over the 588-image
do_A grid vs EY_do_A (Var(EY_do_A) = 53.27 on V2 / V3). Requeue-safe: a restarted task appends to its own CSV and
skips finished (data_seed, repeat) cells.

  python dSprite_DRKPV_Nystrom_Simulation.py --dgp v2 --data_seeds 0 --n_repeats 1 --n_sample 2000 --dtype float32
"""
import argparse, csv, datetime, glob, os, socket, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
from simulation_utils.data_generating_processes import (dSprite_ProxyVariable_DatasetV2,   # noqa: E402
                                                       dSprite_ProxyVariable_DatasetV3)
from torch_utils.scalers import TorchStandardScaler                                       # noqa: E402
from torch_utils.kernel_utils import RBF                                                  # noqa: E402
from causal_learning.approximate_kernel_proxy_causal_learning import (                    # noqa: E402
    DoublyRobustKernelProxyDoseResponseNystrom)

DATA_PATH = os.path.join(REPO, "data", "dsprite")
ALGOS = ["DRKPV_Nystrom", "KPV_Nystrom", "KAP_Nystrom"]
OUT_BY_DGP = {"v2": "dSprite_DRKPV_Nystrom", "org": "dSprite_DRKPV_Nystrom_org", "v3": "dSprite_V3_DRKPV_Nystrom"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dgp", choices=["v2", "org", "v3"], default="v2",
                   help="v2 = dSprite_ProxyVariable_DatasetV2 (the benchmark of the neural dSprite sim), org = its "
                        "proceedings variant, v3 = dSprite_ProxyVariable_DatasetV3 (latent-confounder benchmark)")
    p.add_argument("--data_seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--n_sample", type=int, default=5000)
    p.add_argument("--n_repeats", type=int, default=3)
    p.add_argument("--repeat_offset", type=int, default=0, help="first repeat index (train_seed = 1000*data_seed + repeat)")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64",
                   help="float64 = convention for KAP-family rows; float32 = the earlier setting")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--nystrom_m", type=int, default=500, help="landmarks per Nystrom stage (all stages, all bridges, DR smoother)")
    p.add_argument("--tm_lambda1", type=float, default=1e-3)
    p.add_argument("--tm_eta", type=float, default=1e-3)
    p.add_argument("--tm_lambda2", type=float, default=1e-3)
    p.add_argument("--tm_quantile_aw", type=float, default=0.95, help="length-scale quantile of the treatment bridge's A and W kernels")
    p.add_argument("--om_lambda1", type=float, default=1e-3)
    p.add_argument("--om_lambda2", type=float, default=1e-3)
    p.add_argument("--lambda_dr", type=float, default=1e-3)
    p.add_argument("--stage1_perc", type=float, default=0.5)
    p.add_argument("--make_psd_eps", type=float, default=5e-6)
    p.add_argument("--tag", default="sizesweep")
    p.add_argument("--out_dir", default=None)
    return p.parse_args(argv)


HYPER = ["n_sample", "dgp", "dtype", "nystrom_m", "tm_lambda1", "tm_eta", "tm_lambda2", "tm_quantile_aw",
         "om_lambda1", "om_lambda2", "lambda_dr", "stage1_perc", "make_psd_eps", "tag"]


def run_once(data_seed, train_seed, P):
    dt = getattr(torch, P.dtype)
    np.random.seed(train_seed); torch.manual_seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed); torch.cuda.reset_peak_memory_stats()
    gen = dSprite_ProxyVariable_DatasetV3() if P.dgp == "v3" else dSprite_ProxyVariable_DatasetV2(variant=P.dgp)
    A, Y, Z, W, do_A, EY = gen.generate_dsprite_pv(DATA_PATH, n_sample=P.n_sample, generate_test=True, rand_seed=data_seed)
    dev = P.device
    tt = lambda x: torch.tensor(np.asarray(x), dtype=dt).to(dev)
    A_t, Y_t, Z_t, W_t, do_A_t = tt(A), tt(Y), tt(Z), tt(W), tt(do_A)
    tf = [TorchStandardScaler() for _ in range(4)]
    A_s, Y_s, Z_s, W_s = tf[0].fit_transform(A_t), tf[1].fit_transform(Y_t), tf[2].fit_transform(Z_t), tf[3].fit_transform(W_t)
    do_A_s = tf[0].transform(do_A_t)
    tm = {"kernel_A": RBF(use_length_scale_heuristic=True, length_scale_heuristic_quantile=P.tm_quantile_aw),
          "kernel_W": RBF(use_length_scale_heuristic=True, length_scale_heuristic_quantile=P.tm_quantile_aw),
          "kernel_Z": RBF(use_length_scale_heuristic=True),
          "lambda1_": P.tm_lambda1, "eta": P.tm_eta, "lambda2_": P.tm_lambda2,
          "nystrom_first_stage_m": P.nystrom_m, "nystrom_third_stage_m": P.nystrom_m,
          "stage1_perc": P.stage1_perc, "model_seed": train_seed, "make_psd_eps": P.make_psd_eps, "device": dev}
    om = {"algorithm_name": "Kernel_Proxy_Variable",
          "kernel_A": RBF(use_length_scale_heuristic=True), "kernel_W": RBF(use_length_scale_heuristic=True),
          "kernel_Z": RBF(use_length_scale_heuristic=True),
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
    out_dir = P.out_dir or os.path.join(HERE, "..", "Results", OUT_BY_DGP[P.dgp])
    os.makedirs(out_dir, exist_ok=True)
    sd = "s" + "-".join(str(s) for s in P.data_seeds) if len(P.data_seeds) <= 3 else "s%d-%d" % (min(P.data_seeds), max(P.data_seeds))
    stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    prefix = "DRKPV_Nystrom_dSprite%s_n%d_%s_%s_" % ("V3" if P.dgp == "v3" else ("Org" if P.dgp == "org" else ""), P.n_sample, P.tag, sd)
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
    print("dgp %s | n %d | dtype %s | device %s | seeds %s x repeats %d" % (P.dgp, P.n_sample, P.dtype, P.device, P.data_seeds, P.n_repeats), flush=True)
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
            print("data_seed %d rep %d (%.0fs, peak %.1f GB): %s" % (data_seed, repeat, secs, peak,
                  "  ".join("%s %.4f" % (a, mse[a]) for a in ALGOS)), flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
