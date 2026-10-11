"""Multi-seed, multi-hyperparameter simulation: density-ratio-FREE CATE with DISJOINT-fold
cross-fitting on the synthetic heterogeneous-effects benchmark -- data size as a sweep axis, and
BOTH normalization arms run PAIRED inside every (data_seed, repeat).

PROTOCOL. data_seed draws the dataset from the CATE DGP; train_seed = 1000 * data_seed + repeat drives
the val/fold permutation and every network init. Three mutually disjoint folds per loop (asserted),
roles rotating. val_perc held out before folds; monitoring only.

NORMALIZATION ARMS -- why this experiment pairs "none" with "kkt", not "posthoc" with "kkt".
The CATE third stage is a two-output (numerator, normalizer) head and the estimator is the RATIO
N_hat(a,v) / clip{D_hat(a,v), eps_D} -- it SELF-normalizes. A post-hoc rescaling of varphi by any
m(A) multiplies numerator and denominator targets alike and cancels in the ratio (up to head-fitting
noise and the floor), so the ATE-style post-hoc arm is mathematically inert here. The live question is
whether the TRAINING-TIME KKT constraint (E[varphi | A = a] = 1; A is binary, so the indicator basis;
V sits in the backdoor slot, and the constraint must NOT condition on V -- E[varphi|A,V] = p(a)/p(a|v)
is not 1) helps the bridge that the self-normalizing head then consumes. Default --normalization both
therefore runs [none, kkt] per (seed, repeat): the OUTCOME bridge is fitted once and shared (it does
not depend on the constraint), the treatment bridge is refit per arm from the SAME torch seed, and both
arms' rows land in the same CSV with their `normalization` value.

Examples:
  python Synthetic_CATE_CrossFitting_Simulation.py --n_sample 10000 --data_seeds 0 1 2 --n_repeats 3
  python Synthetic_CATE_CrossFitting_Simulation.py --n_sample 4000 --data_seeds 4 --normalization kkt
"""
import argparse, copy, csv, datetime, os, socket, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import torch
from torch import nn
from torch.utils.data import DataLoader

from simulation_utils.data_generating_processes import generate_synthetic_CATE_data
from torch_utils.scalers import TorchStandardScaler, TorchIdentityTransformer
from torch_utils.dataset_classes import HeterogeneousProxyDataset
from torch_utils.networks import ConditionalMeanMLP
from torch_utils.torch_loss import get_loss_function
from python_utils.helpers import slice_tuple

from simulation_utils.outcome_pcl_net_configs import OutcomePCLConfig
from simulation_utils.outcome_pcl_net_nn_structures import (
    build_nets_for_outcome_pcl_net_synthetic_cate_experiment)
from simulation_utils.outcome_pcl_net_optimizers import build_adam_optimizers_for_outcome_pcl_net
from neural_causal_learning.proxy_outcome_neural_mean_embedding import (
    HeterogeneousOutcomeBridgePCLNET, train_deep_feature_proxy_closed_form_ate_model,
    create_third_stage_dataset_for_outcome_pcl_net_cate)

from simulation_utils.alternative_treatment_pcl_net_configs import AlternativeTreatmentPCLConfig
from simulation_utils.treatment_pcl_net_nn_structures import (
    build_nets_for_alternative_treatment_pcl_net_synthetic_cate_experiment)
from simulation_utils.alternative_treatment_pcl_net_optimizers import (
    build_adam_optimizers_for_alternative_treatment_pcl_net)
from neural_causal_learning.alternative_treatment_neural_mean_embedding import (
    AlternativeTreatmentBridgePCLNET, train_alternative_treatment_pcl_net_ate_model)
from neural_causal_learning.normalization_constraint import ConditionalMeanConstraint
from neural_causal_learning.alternative_treatment_cate import (
    create_third_stage_dataset_for_alternative_treatment_cate, predict_cate_from_ratio_head)
from neural_causal_learning.proxy_treatment_neural_mean_embedding import (
    train_third_stage_treatment_pcl_net_ensemble)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ESTIMATORS = ["OutcomeNet", "TreatmentNet", "DR self-norm"]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--n_repeats", type=int, default=3)
    # -- data --
    p.add_argument("--n_sample", type=int, default=4000,
                   help="dataset size; the size-scaling axis (sweep: 2000..20000)")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--noise_bound", type=float, default=0.1,
                   help="uniform noise in [-noise_bound, +noise_bound]")
    # -- protocol --
    p.add_argument("--val_perc", type=float, default=0.1)
    p.add_argument("--n_folds", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=None, help="None = n_sample (one step/epoch)")
    # -- normalization arms --
    p.add_argument("--normalization", choices=["both", "none", "kkt"], default="both",
                   help="'both' runs none AND kkt paired per (seed, repeat); see module docstring "
                        "for why 'posthoc' is not an arm in the CATE experiment")
    # -- outcome bridge --
    p.add_argument("--om_epochs", type=int, default=50)
    # -- treatment bridge --
    p.add_argument("--tm_epochs", type=int, default=50)
    p.add_argument("--tm_reg2", type=float, default=10.0)
    # -- CATE heads --
    p.add_argument("--eps_d", type=float, default=0.1, help="denominator floor for the ratio head")
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--head_epochs", type=int, default=100)
    p.add_argument("--head_members", type=int, default=5)
    p.add_argument("--head_batch", type=int, default=0,
                   help="third-stage head loader batch (OM CME head + both ratio heads). "
                        "0 = inherit the shared --batch_size (shipped: batch = n_sample, i.e. ONE "
                        "optimizer step per epoch at every n). A fixed value makes the step count "
                        "grow with n, as the submitted density-ratio sim did (fixed 1024).")
    p.add_argument("--tag", default="baseline")
    p.add_argument("--out_dir", default="",
                   help="'' = Results/Synthetic_CATE_CrossFitting (the reporting grid)")
    return p


HYPER_COLUMNS = ["n_sample", "sigma", "noise_bound", "val_perc", "n_folds", "batch_size",
                 "om_epochs", "tm_epochs", "tm_reg2", "eps_d", "head_lr", "head_epochs",
                 "head_members", "head_batch", "tag"]


def build_data(data_seed, train_seed, P):
    np.random.seed(train_seed); torch.manual_seed(train_seed)
    U, W, Z, V, A, Y, v_test, do_A, EY = generate_synthetic_CATE_data(
        P.n_sample, P.sigma, P.noise_bound, -P.noise_bound, seed=data_seed)
    t = lambda v: torch.tensor(v, dtype=torch.float32)
    tens = [t(A), t(Y), t(Z), t(W), t(V)]
    transformers = [TorchStandardScaler() for _ in range(5)]
    drt = TorchIdentityTransformer(); dr = torch.ones(P.n_sample, 1)
    idx = np.random.permutation(P.n_sample)
    n_val = int(P.n_sample * P.val_perc)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    for i, x in enumerate(tens):
        transformers[i].fit(x[train_idx])
    drt.fit(dr[train_idx])
    mk = lambda ix: HeterogeneousProxyDataset(
        slice_tuple(tuple(tens), ix), dens_ratio=dr[ix],
        dens_ratio_transformer=copy.deepcopy(drt),
        transformers=copy.deepcopy(transformers), device="cpu")
    folds = np.array_split(train_idx, P.n_folds)

    def stage_indices(k):
        others = [folds[(k + o) % P.n_folds] for o in range(1, P.n_folds)]
        half = max(len(others) // 2, 1)
        return np.concatenate(others[:half]), np.concatenate(others[half:]), folds[k]

    assignment = [stage_indices(k) for k in range(P.n_folds)]
    for s1, s2, est in assignment:
        assert len(np.intersect1d(s1, s2)) == 0
        assert len(np.intersect1d(est, s1)) == 0 and len(np.intersect1d(est, s2)) == 0
        assert len(np.intersect1d(est, val_idx)) == 0
        assert len(np.union1d(np.union1d(s1, s2), est)) == len(train_idx)

    return dict(mk=mk, assignment=assignment, val=mk(val_idx),
                do_A=t(do_A), v_test=t(v_test), EY=EY,
                dev_tf=[copy.deepcopy(x).to(DEVICE) for x in transformers])


def fit_outcome(d, l1, l2, lv, P):
    c = OutcomePCLConfig.synthetic_cate(); c.device = DEVICE
    c.lr_first_stage_base = c.lr_second_stage_ax = c.lr_second_stage_w = 1e-4
    c.weight_decay = 1e-5; c.scheduler_gamma = 0.99; c.n_epochs = P.om_epochs
    c.reg_first_stage = (5e-3, 1e-1); c.reg_second_stage_first = (5e-3, 1e-1)
    c.reg_second_stage = (1e-2, 1.0); c.reg_annealing_method = "exponential"
    c.log_per_epoch, c.plot_loss = 10 ** 9, False
    nets = build_nets_for_outcome_pcl_net_synthetic_cate_experiment(c.device)
    m = HeterogeneousOutcomeBridgePCLNET(nets[0], nets[1], nets[2], nets[3], None, device=c.device)
    o1, o2, s1, s2 = build_adam_optimizers_for_outcome_pcl_net(
        m, c.lr_first_stage, c.lr_second_stage_ax, c.lr_second_stage_w, c.weight_decay,
        gamma=c.scheduler_gamma)
    r1, r2i, r2o = c.regularizers
    return train_deep_feature_proxy_closed_form_ate_model(
        pcl_model=m, first_stage_train_dataloader=l1, second_stage_train_dataloader=l2,
        stage1_optimizers=o1, stage2_optimizers=o2, stage1_schedulers=s1, stage2_schedulers=s2,
        n_epochs=c.n_epochs, stage1_iter=c.stage1_iter, stage2_iter=c.stage2_iter,
        first_stage_final_layer_regularizer=r1, second_stage_final_layer_regularizer=r2o,
        second_stage_first_final_layer_regularizer=r2i,
        regularizer_annealing_method=c.reg_annealing_method,
        consider_prev_weight=c.consider_prev_weight,
        log_per_epoch=c.log_per_epoch, plot_loss=c.plot_loss, validation_dataloader=lv)


def fit_treatment(d, l1, l2, lv, P, norm_mode):
    c = AlternativeTreatmentPCLConfig.synthetic_cate(); c.device = DEVICE
    c.lr_first_stage = 1e-3; c.lr_second_stage_ax = c.lr_second_stage_z = 1e-4
    c.weight_decay = 1e-6; c.scheduler_gamma = 1.0; c.n_epochs = P.tm_epochs
    c.reg_first_stage = c.reg_second_stage_first = 1e-2; c.reg_second_stage = P.tm_reg2
    # A is binary -> indicator basis (per-arm mean of varphi = 1). V rides in the backdoor slot; the
    # constraint conditions on A ONLY (E[varphi | A, V] = p(a)/p(a|v) != 1 -- see the module docstring).
    constraint = ConditionalMeanConstraint(basis="indicator") if norm_mode == "kkt" else None
    nets = build_nets_for_alternative_treatment_pcl_net_synthetic_cate_experiment(
        c.device, joint_backdoor_first_stage=False)
    m = AlternativeTreatmentBridgePCLNET(
        nets[0], nets[1], nets[2], nets[3],
        backdoor_featurizer_first_stage=nets[4], backdoor_featurizer_second_stage=nets[5],
        device=c.device, normalization_constraint=constraint)
    o1, o2, s1, s2 = build_adam_optimizers_for_alternative_treatment_pcl_net(
        m, lr_first_stage=c.lr_first_stage, lr_second_stage_ax=c.lr_second_stage_ax,
        lr_second_stage_z=c.lr_second_stage_z, weight_decay=c.weight_decay, gamma=c.scheduler_gamma)
    m, _, _ = train_alternative_treatment_pcl_net_ate_model(
        model=m, first_stage_train_dataloader=l1, second_stage_train_dataloader=l2,
        stage1_optimizers=o1, stage2_optimizers=o2, stage1_schedulers=s1, stage2_schedulers=s2,
        n_epochs=c.n_epochs, stage1_iter=c.stage1_iter, stage2_iter=c.stage2_iter,
        first_stage_final_layer_regularizer=c.reg_first_stage,
        second_stage_first_final_layer_regularizer=c.reg_second_stage_first,
        second_stage_final_layer_regularizer=c.reg_second_stage,
        consider_prev_weight=c.consider_prev_weight,
        loss_fn=get_loss_function(c.first_stage_loss_name, c.first_stage_loss_kwargs),
        validation_dataloader=lv, log_per_epoch=10 ** 9, plot_loss=False)
    return m


def estimate_cate(d, om, tm, le, lv, P):
    """The notebook's estimate step: OM CME head + the shared two-output ratio head, held-out fold."""
    om_head_kwargs = dict(lr=P.head_lr, weight_decay=1e-6, n_epochs=P.head_epochs,
                          loss_fn=nn.MSELoss(), gap_penalty_weight=0.0, log_per_epoch=10 ** 9,
                          input_dim=8, output_dim=16, hidden_dims=[32, 64], dropout_rate=0.01)
    cate_head_kwargs = dict(lr=P.head_lr, weight_decay=1e-6, n_epochs=P.head_epochs,
                            loss_fn=nn.MSELoss(), gap_penalty_weight=0.0, log_per_epoch=10 ** 9,
                            input_dim=2, output_dim=2, hidden_dims=[32, 64], dropout_rate=0.01)

    def head(ds, dsv, kwargs):
        return train_third_stage_treatment_pcl_net_ensemble(
            model_class=ConditionalMeanMLP,
            dataloader=DataLoader(ds, batch_size=P.head_batch or P.batch, shuffle=True),
            val_dataloader=DataLoader(dsv, batch_size=P.head_batch or P.batch, shuffle=True),
            n_members=P.head_members, device=DEVICE, **kwargs)

    om_ds, om_dsv = create_third_stage_dataset_for_outcome_pcl_net_cate(om, le, lv, device=DEVICE)
    om_net = head(om_ds, om_dsv, om_head_kwargs)
    f_om = om.pred_structural_function(
        d["do_A"], d["v_test"], om_net,
        d["dev_tf"][0], d["dev_tf"][4], d["dev_tf"][1]).detach().cpu().numpy()

    do_A_tr = d["dev_tf"][0].transform(d["do_A"].to(DEVICE))
    v_tr = d["dev_tf"][4].transform(d["v_test"].to(DEVICE))

    def two_output_head(numerator, outcome_pcl_model=None):
        ds, dsv = create_third_stage_dataset_for_alternative_treatment_cate(
            tm, le, lv, TorchStandardScaler(), TorchStandardScaler(),
            outcome_pcl_model=outcome_pcl_model, numerator=numerator, device=DEVICE)
        net = head(ds, dsv, cate_head_kwargs)
        return predict_cate_from_ratio_head(ds, net, do_A_tr, v_tr,
                                            denominator_floor=P.eps_d, device=DEVICE)

    ratio_tm, _, clipped_tm = two_output_head("treatment")
    corr_dr, _, clipped_dr = two_output_head("dr", outcome_pcl_model=om)
    return ({"OutcomeNet": f_om,
             "TreatmentNet": ratio_tm.detach().cpu().numpy(),
             "DR self-norm": f_om + corr_dr.detach().cpu().numpy()},
            {"clipped_tm": float(clipped_tm), "clipped_dr": float(clipped_dr)})


def run_once(data_seed, train_seed, P, norm_modes):
    """One (data_seed, train_seed): all norm arms PAIRED -- shared data, folds, outcome bridges."""
    P.batch = P.batch_size if P.batch_size is not None else P.n_sample
    d = build_data(data_seed, train_seed, P)
    lv = DataLoader(d["val"], batch_size=P.batch, shuffle=False)
    mse = lambda f: float(np.mean((np.asarray(f).reshape(-1, 1) - d["EY"].reshape(-1, 1)) ** 2))
    curves = {m: [] for m in norm_modes}
    clips = {m: [] for m in norm_modes}
    for s1, s2, est in d["assignment"]:
        ds1, ds2, dse = d["mk"](s1), d["mk"](s2), d["mk"](est)
        l1 = DataLoader(ds1, batch_size=P.batch, shuffle=True)
        l2 = DataLoader(ds2, batch_size=P.batch, shuffle=True)
        torch.manual_seed(train_seed)
        om = fit_outcome(d, l1, l2, lv, P)               # SHARED: does not depend on the constraint
        # PAIRING VIA RNG-STATE CAPTURE, not re-seeding: snapshot the generator right after the shared
        # outcome fit and RESTORE it before every arm. The first arm therefore continues the notebook's
        # natural RNG stream (the 'none' arm reproduces the notebook exactly), and every other
        # arm replays the exact same stream -- inits, dropout, loader shuffles all identical, so the
        # constraint is the ONLY difference between arms.
        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        for mode in norm_modes:
            torch.set_rng_state(rng_cpu)
            if rng_cuda is not None:
                torch.cuda.set_rng_state_all(rng_cuda)
            tm = fit_treatment(d, l1, l2, lv, P, mode)
            le = DataLoader(dse, batch_size=P.batch, shuffle=True)
            est_curves, diag = estimate_cate(d, om, tm, le, lv, P)
            curves[mode].append(est_curves)
            clips[mode].append(diag)
    out = {}
    for mode in norm_modes:
        cf = {n: np.mean([c[n] for c in curves[mode]], axis=0) for n in ESTIMATORS}
        out[mode] = {"mse": {n: mse(cf[n]) for n in ESTIMATORS},
                     "clipped_tm": float(np.mean([x["clipped_tm"] for x in clips[mode]])),
                     "clipped_dr": float(np.mean([x["clipped_dr"] for x in clips[mode]]))}
    return out


def main():
    P = build_parser().parse_args()
    norm_modes = ["none", "kkt"] if P.normalization == "both" else [P.normalization]
    out_dir = P.out_dir or os.path.join(HERE, "..", "Results", "Synthetic_CATE_CrossFitting")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    seed_part = "n%d_s%s" % (P.n_sample, "-".join(str(s) for s in P.data_seeds))
    out = os.path.join(out_dir, "DRNMEPCL_DensRatioFree_SyntheticCATECF_%s_%s_%s.csv"
                       % (P.tag, seed_part, stamp))
    fields = ["Algorithm", "Arm", "normalization", "Data_Seed", "Repeat", "Train_Seed",
              "Causal_MSE", "clipped_tm", "clipped_dr", "hostname"] + HYPER_COLUMNS
    with open(out, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    meta = {k: getattr(P, k) for k in HYPER_COLUMNS}
    meta["hostname"] = socket.gethostname()
    print("writing to %s   (norm arms: %s)" % (out, norm_modes), flush=True)

    for data_seed in P.data_seeds:
        for repeat in range(P.n_repeats):
            train_seed = 1000 * data_seed + repeat
            t0 = time.time()
            res = run_once(data_seed, train_seed, P, norm_modes)
            with open(out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                for mode in norm_modes:
                    for est in ESTIMATORS:
                        w.writerow(dict(Algorithm=est, Arm="cross_fitted", normalization=mode,
                                        Data_Seed=data_seed, Repeat=repeat, Train_Seed=train_seed,
                                        Causal_MSE=res[mode]["mse"][est],
                                        clipped_tm=res[mode]["clipped_tm"],
                                        clipped_dr=res[mode]["clipped_dr"], **meta))
            print("n=%d data_seed %d repeat %d (%ds) | %s"
                  % (P.n_sample, data_seed, repeat, time.time() - t0,
                     " || ".join("%s: %s" % (m, "  ".join("%s %.5f" % (e[:4], res[m]["mse"][e])
                                                          for e in ESTIMATORS))
                                 for m in norm_modes)), flush=True)


if __name__ == "__main__":
    main()
