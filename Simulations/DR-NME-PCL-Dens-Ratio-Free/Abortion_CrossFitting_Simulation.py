"""Multi-seed, multi-hyperparameter simulation: density-ratio-FREE DR-PCL with DISJOINT-fold
cross-fitting on the Legalized Abortion & Crime benchmark.

PROTOCOL (kernel-paper convention). Benchmark data seeds 0..9 select the shipped .npz splits; the
method seed train_seed = 1000 * data_seed + repeat drives the val/fold permutation and every network
init. Per cross-fitting loop the three roles use three MUTUALLY DISJOINT folds (stage-1 CME fold,
stage-2 head fold, estimation fold -- asserted at runtime), rotating cyclically so every fold serves
every role exactly once. A further val_perc share is held out BEFORE the folds are cut; it is used
as the bridges' validation monitor and the third-stage ensembles' validation loader -- it never
enters any estimate, with ONE flagged exception: under --dr_val_shrink 1 the val split additionally
selects the scalar shrinkage lam applied to the DR-V1 correction (a validated-shrinkage selector,
still disjoint from the estimation fold). The estimation fold is therefore
(1 - val_perc)/n_folds of the data (~30% at the defaults), NOT the 10% validation share.

HYPERPARAMETERS. Every CLI knob below is written into every CSV row, so a finished sweep can be
analysed by grouping the concatenated CSVs on those columns. Nothing is tuned inside this script.

PARALLELISM. Each invocation writes its OWN timestamped CSV whose name includes the tag and the seed
range, so any number of array tasks can run concurrently with zero write contention (a
same-second collision between two array tasks once destroyed results -- hence seed in the filename,
not just the timestamp). Concatenate the CSVs at analysis time.

Examples:
  # everything, serially
  python Abortion_CrossFitting_Simulation.py --data_seeds 0 1 2 3 4 5 6 7 8 9 --n_repeats 3
  # one array task: a single data seed
  python Abortion_CrossFitting_Simulation.py --data_seeds 4 --n_repeats 3 --tag baseline
  # a hyperparameter arm on the same seeds, for a paired comparison
  python Abortion_CrossFitting_Simulation.py --data_seeds 4 --n_repeats 3 \
      --normalization kkt --tag kkt_arm
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

from simulation_utils.data_generating_processes import read_legalized_abortion_and_crime_dataset
from torch_utils.scalers import TorchStandardScaler, TorchIdentityTransformer
from torch_utils.dataset_classes import ProxyDataset
from torch_utils.networks import ConditionalMeanMLP
from torch_utils.torch_loss import get_loss_function
from python_utils.helpers import slice_tuple

from simulation_utils.outcome_pcl_net_configs import OutcomePCLConfig
from simulation_utils.outcome_pcl_net_nn_structures import (
    build_nets_for_outcome_pcl_net_abortion_and_crime_experiment)
from simulation_utils.outcome_pcl_net_optimizers import build_adam_optimizers_for_outcome_pcl_net
from neural_causal_learning.proxy_outcome_neural_mean_embedding import (
    OutcomeBridgePCLNET, train_deep_feature_proxy_closed_form_ate_model)

from simulation_utils.alternative_treatment_pcl_net_configs import AlternativeTreatmentPCLConfig
from simulation_utils.treatment_pcl_net_nn_structures import (
    build_nets_for_alternative_treatment_pcl_net_abortion_and_crime_experiment)
from simulation_utils.alternative_treatment_pcl_net_optimizers import (
    build_adam_optimizers_for_alternative_treatment_pcl_net)
from neural_causal_learning.alternative_treatment_neural_mean_embedding import (
    AlternativeTreatmentBridgePCLNET, train_alternative_treatment_pcl_net_ate_model,
    create_third_stage_dataset_for_treatment_pcl_net_ate,
    train_alternative_treatment_pcl_net_ensemble,
)
from neural_causal_learning.normalization_constraint import ConditionalMeanConstraint
from neural_causal_learning.proxy_treatment_neural_mean_embedding import (
    train_third_stage_treatment_pcl_net_ensemble)
from neural_causal_learning.alternative_doubly_robust_proxy_neural_mean_embedding import (
    create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_ate as DR1,
    create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_ate_v2 as DR2)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ESTIMATORS = ["OutcomeNet", "TreatmentNet", "DRPCLNET-V1", "DRPCLNET-V2"]
A_RANGE = (0.0, 2.0)          # the benchmark's effective treatment support (the do_A grid lies inside)


# ================================================================ hyperparameters
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_seeds", type=int, nargs="+", default=list(range(10)),
                   help="benchmark .npz seeds (0..9)")
    p.add_argument("--n_repeats", type=int, default=3, help="method repeats per data seed")
    p.add_argument("--repeat_offset", type=int, default=0,
                   help="first repeat index (train_seed = 1000*data_seed + repeat_offset + i). 0 = "
                        "numerically identical historical behaviour; the tuning protocol confirms winners on "
                        "repeats 5,6 -- method seeds disjoint from the reporting repeats 0-2.")
    # -- protocol --
    p.add_argument("--val_perc", type=float, default=0.1,
                   help="validation share held out before folds (monitoring only, never estimation)")
    p.add_argument("--n_folds", type=int, default=3,
                   help="disjoint folds; estimation share = (1 - val_perc)/n_folds")
    p.add_argument("--batch_size", type=int, default=2048)
    # -- outcome bridge --
    p.add_argument("--om_epochs", type=int, default=None, help="None = preset")
    p.add_argument("--om_reg2", type=float, default=1e-1, help="outcome head ridge (constant)")
    # -- treatment bridge --
    p.add_argument("--tm_epochs", type=int, default=None, help="None = preset")
    p.add_argument("--tm_reg2", type=float, default=None,
                   help="KAP head ridge; None = preset (10.0). An INVERSE STEP SIZE, not a ridge.")
    # -- E[phi | A = a] = 1 normalization --
    p.add_argument("--tm_width", type=int, default=32, help="TM featurizer MLP width (historical 32)")
    p.add_argument("--tm_first_out", type=int, default=16, help="TM stage-1 A-feature dim")
    p.add_argument("--tm_z_out", type=int, default=8, help="TM Z-feature dim (head is z_out x a2_out)")
    p.add_argument("--tm_w_out", type=int, default=8, help="TM W-feature dim")
    p.add_argument("--tm_a2_out", type=int, default=8, help="TM stage-2 A-feature dim")
    p.add_argument("--tm_dropout", type=float, default=0.01, help="TM featurizer dropout")
    p.add_argument("--tm_members", type=int, default=1,
                   help="bridge ensemble size. 1 = the historical single bridge (numerically identical path); "
                        ">1 averages members via train_alternative_treatment_pcl_net_ensemble -- the "
                        "notebook used K=10 and every other benchmark found K>=3 materially better. "
                        "NOTE the ensemble trainer drives phi_A2 and phi_Z with ONE stage-2 lr "
                        "(lr_second_stage_ax), a minor difference from the single-bridge optimizers.")
    p.add_argument("--normalization", choices=["posthoc", "kkt", "none"], default="posthoc",
                   help="posthoc = the notebook's polynomial divide-out; kkt = training-time constraint")
    p.add_argument("--norm_degree", type=int, default=5,
                   help="polynomial degree (posthoc fit / kkt moment basis)")
    p.add_argument("--norm_floor", type=float, default=0.15, help="posthoc divisor floor")
    # -- third stage --
    p.add_argument("--third_lr", type=float, default=1e-3)
    p.add_argument("--third_epochs", type=int, default=100)
    # PER-HEAD third-stage budgets, same semantics as the demand sim: 0 = inherit
    # --third_epochs, so the default path is untouched. The third-stage loaders are FULL BATCH
    # (batch_size 2048 vs an estimation fold of ~450 rows), so one epoch is ONE optimizer update:
    # --third_epochs 100 gives the heads only 100 Adam updates. Measured on disjoint repeats,
    # 10 updates beat 100 for DRPCLNET-V1 (and for TreatmentNet) on this benchmark.
    p.add_argument("--tm_third_epochs", type=int, default=0,
                   help="TreatmentNet third-stage updates; 0 = inherit --third_epochs")
    p.add_argument("--dr_third_epochs", type=int, default=0,
                   help="DR slack third-stage updates; 0 = inherit --third_epochs")
    p.add_argument("--third_hidden", type=str, default="32,64")
    p.add_argument("--third_dropout", type=float, default=0.01)
    p.add_argument("--third_members", type=int, default=5)
    p.add_argument("--dr_val_shrink", type=int, default=0,
                   help="1 = validation-selected mean-preserving shrinkage of the DR-V1 correction's "
                        "A-varying part (lam = <t_dev,p_dev>/||p_dev||^2 on the held-out DR pseudo-"
                        "outcomes; noise -> lam -> 0 so DR falls back toward OM, signal -> lam -> 1). "
                        "Ported from the density-ratio pipeline's run_abortion.py, V1 only.")
    # -- bookkeeping --
    p.add_argument("--include_single_split", type=int, default=0,
                   help="1 = also run a DISJOINT single-split arm (thirds of the training rows)")
    # -- overlapping-split ablation (see the OVERLAPPING-SPLIT ABLATION block) --
    p.add_argument("--split_mode", choices=["crossfit", "overlap"], default="crossfit",
                   help="crossfit = the paper protocol (disjoint folds, rotated). overlap = ONE "
                        "split with stages that may share rows, as in an earlier internal version -- no rotation.")
    p.add_argument("--stage_perc", type=float, default=0.75,
                   help="overlap mode: each stage takes this share of the pool from opposite ends, "
                        "so the stages share max(0, 2p-1) of it. 0.75 = the legacy 50%% overlap.")
    p.add_argument("--est_mode", choices=["holdout", "stage2"], default="holdout",
                   help="overlap mode: holdout = estimation fold carved out FIRST and kept disjoint "
                        "(isolates training-stage reuse); stage2 = estimation rows ARE the stage-2 "
                        "rows, the legacy choice (adds estimation reuse on top).")
    p.add_argument("--rotate_overlap", type=int, default=0,
                   help="overlap mode: 1 = ALSO rotate the estimation fold over all n_folds and "
                        "average the curves, so reuse and cross-fitting are combined (only the two "
                        "TRAINING stages share rows; with est_mode=holdout the estimation fold stays "
                        "disjoint from both within each rotation). 0 = one split, the legacy shape.")
    p.add_argument("--tag", default="baseline", help="config label; goes in filename and CSV")
    return p


HYPER_COLUMNS = ["split_mode", "stage_perc", "est_mode", "rotate_overlap", "val_perc", "n_folds", "batch_size", "om_epochs", "om_reg2", "tm_epochs", "tm_reg2",
                 "tm_third_epochs", "dr_third_epochs",
                 "tm_width", "tm_first_out", "tm_z_out", "tm_w_out", "tm_a2_out", "tm_dropout",
                 "tm_members",
                 "normalization", "norm_degree", "norm_floor", "third_lr", "third_epochs",
                 "third_hidden", "third_dropout", "third_members", "dr_val_shrink", "tag"]


# ================================================================ OVERLAPPING-SPLIT ABLATION
# The paper's protocol uses MUTUALLY DISJOINT folds. In practice one is tempted to reuse rows across
# stages to buy a larger sample per stage, and the previous density-ratio implementation
# (an earlier internal version*) did exactly that:
#
#     stage1_perc = stage2_perc = 0.75
#     stage1_idx, stage2_idx = train_indices[:n1], train_indices[-n2:]     # 50% OVERLAP
#     create_third_stage_dataset(..., second_stage_train_dataloader,...)  # ESTIMATION = STAGE 2
#
# i.e. TWO separate departures -- overlapping training stages, AND an estimation set that is the
# stage-2 set itself. They are different mechanisms (the first affects how well the bridges fit, the
# second is what breaks the DR correction's orthogonality), so this ablation varies them
# INDEPENDENTLY rather than reproducing the legacy protocol as a single lump:
#
#   --est_mode stage2    estimation rows ARE the stage-2 rows (legacy). Training pool = all train rows.
#   --est_mode holdout   an estimation fold of the SAME SIZE as cross-fitting's ((1-val)/n_folds) is
#                        carved out FIRST and stays disjoint from both stages, so the arms differ only
#                        in how the REMAINING pool is reused between the two training stages.
#   --stage_perc p       stage1 = pool[:p|pool|], stage2 = pool[-p|pool|:], so |s1 cap s2|/|pool| =
#                        max(0, 2p-1): p=0.5 disjoint, p=0.75 the legacy 50%, p=1.0 both stages see
#                        the whole pool.
#
# --split_mode overlap is a SINGLE split with NO rotation (as the legacy code was), which is why the
# p=0.5/holdout cell is the right control for the cross-fitted reference: it isolates the value of
# ROTATION at an unchanged disjointness premise.
#
# --rotate_overlap 1 adds the cell the single-split ladder leaves out: ROTATION *AND* REUSE. The
# estimation fold still rotates over all K folds and is still disjoint from both training stages
# WITHIN each rotation (so the DR correction keeps its orthogonality per rotation, and the K curves
# are still averaged), but the two TRAINING stages are drawn from the same pool and may share rows.
# This is what DML actually does -- it requires the nuisance fit to be disjoint from the EVALUATION
# fold, not the two nuisance stages to be disjoint from EACH OTHER; the latter is an extra demand of
# the two-stage regression. At K=3, p=1.0 each bridge sees 2/3 of the training rows instead of 1/3
# while the estimation fold stays clean.
#   CONSISTENCY CHECK, and it is exact: at p=0.5 with est_mode=holdout and rotation, the pool
#   (folds k+1, k+2) is cut into halves, which IS the crossfit assignment -- so that cell must
#   reproduce the reference arm, and it is asserted to.
#
# CAUTION, and it is the reason every arm must be run at one batch_size: reuse makes the stages
# BIGGER, and the head solve is a proximal iteration whose effective ridge is lam_eff = reg2/T with
# T = batches_per_epoch * n_epochs. If a larger stage silently crosses batch_size, batches_per_epoch
# rises, lam_eff falls, and the arm differs in REGULARISATION as well as in splitting. Keep
# batch_size >= the largest stage in the whole comparison so batches_per_epoch == 1 everywhere.
def _one_overlap_split(pool, est, P):
    """Cut one pool from both ends. est=None means the legacy choice "estimation IS stage 2".

    FLOOR, not round, and an explicit no-overlap branch: with an odd pool, rounding both stages up at
    p=0.5 leaves them sharing one row, which would make the "disjoint" cell not actually disjoint.
    When 2*n1 <= m the second stage takes the exact COMPLEMENT of the first, so p <= 0.5 partitions
    the pool and overlap_rows is exactly 0.
    """
    m = len(pool)
    n1 = max(1, min(int(m * P.stage_perc), m))
    s1 = pool[:n1]
    s2 = pool[n1:] if 2 * n1 <= m else pool[m - n1:]
    return s1, s2, (s2 if est is None else est)


def overlap_assignment(train_idx, P):
    """Return ([(s1, s2, est), ...], diagnostics).

    --rotate_overlap 0 (default): ONE split, no rotation -- the legacy shape.
    --rotate_overlap 1: K rotations, estimation fold disjoint from both stages within each rotation.
    """
    rotate = bool(getattr(P, "rotate_overlap", 0))
    folds = np.array_split(train_idx, P.n_folds)
    triples = []
    if not rotate:
        if P.est_mode == "stage2":
            pool, est = train_idx, None
        else:
            k = len(train_idx) // P.n_folds      # same estimation size as a cross-fitting fold
            est, pool = train_idx[:k], train_idx[k:]
        triples.append(_one_overlap_split(pool, est, P))
    else:
        for k in range(P.n_folds):
            pool = np.concatenate([folds[(k + o) % P.n_folds] for o in range(1, P.n_folds)])
            est = None if P.est_mode == "stage2" else folds[k]
            triples.append(_one_overlap_split(pool, est, P))
    s1, s2, est = triples[0]
    m = len(np.union1d(s1, s2)) if rotate else len(
        train_idx if P.est_mode == "stage2" else train_idx[len(train_idx) // P.n_folds:])
    inter = np.intersect1d(s1, s2)
    diag = dict(stage1_rows=len(s1), stage2_rows=len(s2), est_rows=len(est),
                overlap_rows=len(inter),
                overlap_of_pool=round(len(inter) / max(m, 1), 4),
                est_reused_frac=round(len(np.intersect1d(est, np.union1d(s1, s2)))
                                      / max(len(est), 1), 4))
    return triples, diag


# ================================================================ data and folds
def build_data(data_seed, train_seed, P):
    """Load a benchmark split, restrict to the treatment support, hold out validation, cut folds."""
    np.random.seed(train_seed); torch.manual_seed(train_seed)
    W, Z, A, Y, do_A, EY = read_legalized_abortion_and_crime_dataset(
        os.path.join(REPO, "data", "abortion"), seed=data_seed)
    t = lambda v: torch.tensor(v, dtype=torch.float32)
    A_t, Y_t, Z_t, W_t = t(A), t(Y), t(Z), t(W)
    mask = ((A_t > A_RANGE[0]) & (A_t < A_RANGE[1])).view(-1)
    A_t, Y_t, Z_t, W_t = A_t[mask], Y_t[mask], Z_t[mask], W_t[mask]
    n = A_t.shape[0]

    transformers = [TorchStandardScaler() for _ in range(4)]
    drt = TorchIdentityTransformer(); dr = torch.ones(n, 1)
    idx = np.random.permutation(n)
    n_val = int(n * P.val_perc)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    for i, x in enumerate([A_t, Y_t, Z_t, W_t]):
        transformers[i].fit(x[train_idx])
    drt.fit(dr[train_idx])

    mk = lambda ix: ProxyDataset(slice_tuple((A_t, Y_t, Z_t, W_t), ix), dens_ratio=dr[ix],
                                 dens_ratio_transformer=copy.deepcopy(drt),
                                 transformers=copy.deepcopy(transformers), device="cpu")
    folds = np.array_split(train_idx, P.n_folds)

    def stage_indices(k):
        """DISJOINT fold per role: estimation = fold k; the others split evenly between the stages."""
        others = [folds[(k + o) % P.n_folds] for o in range(1, P.n_folds)]
        half = max(len(others) // 2, 1)
        return np.concatenate(others[:half]), np.concatenate(others[half:]), folds[k]

    if P.split_mode == "crossfit":
        assignment = [stage_indices(k) for k in range(P.n_folds)]
        split_diag = dict(stage1_rows=len(assignment[0][0]), stage2_rows=len(assignment[0][1]),
                          est_rows=len(assignment[0][2]), overlap_rows=0, overlap_of_pool=0.0,
                          est_reused_frac=0.0)
        for s1, s2, est in assignment:                   # the premise, enforced, every run
            assert len(np.intersect1d(s1, s2)) == 0
            assert len(np.intersect1d(est, s1)) == 0 and len(np.intersect1d(est, s2)) == 0
            assert len(np.intersect1d(est, val_idx)) == 0
            assert len(np.union1d(np.union1d(s1, s2), est)) == len(train_idx)
    else:
        assignment, split_diag = overlap_assignment(train_idx, P)
        for s1, s2, est in assignment:                   # what still HAS to hold under reuse
            assert len(np.intersect1d(est, val_idx)) == 0
            assert max(len(s1), len(s2)) <= P.batch_size, (
                "batch_size must cover the largest stage or lam_eff = reg2/T changes with the arm")
            if P.est_mode == "holdout":
                assert len(np.intersect1d(est, np.union1d(s1, s2))) == 0

    return dict(mk=mk, assignment=assignment, val=mk(val_idx), train_idx=train_idx,
                split_diag=split_diag,
                do_A=t(do_A), EY=EY, dev_tf=[copy.deepcopy(x).to(DEVICE) for x in transformers])


# ================================================================ bridges
def fit_outcome(d, l1, l2, lv, P):
    c = OutcomePCLConfig.abortion_and_crime(); c.device = DEVICE
    c.lr_first_stage_base = c.lr_second_stage_ax = c.lr_second_stage_w = 1e-4
    c.weight_decay = 1e-7; c.stage1_iter = 10; c.consider_prev_weight = True
    c.reg_second_stage = (P.om_reg2, P.om_reg2)
    if P.om_epochs is not None:
        c.n_epochs = P.om_epochs
    c.log_per_epoch, c.plot_loss = 10 ** 9, False
    nets = build_nets_for_outcome_pcl_net_abortion_and_crime_experiment(c.device)
    m = OutcomeBridgePCLNET(nets[0], nets[1], nets[2], None, None, device=c.device)
    o1, o2, s1, s2 = build_adam_optimizers_for_outcome_pcl_net(
        m, c.lr_first_stage, c.lr_second_stage_ax, c.lr_second_stage_w, c.weight_decay,
        gamma=c.scheduler_gamma)
    r1, r2i, r2o = c.regularizers
    m = train_deep_feature_proxy_closed_form_ate_model(
        pcl_model=m, first_stage_train_dataloader=l1, second_stage_train_dataloader=l2,
        stage1_optimizers=o1, stage2_optimizers=o2, stage1_schedulers=s1, stage2_schedulers=s2,
        n_epochs=c.n_epochs, stage1_iter=c.stage1_iter, stage2_iter=c.stage2_iter,
        first_stage_final_layer_regularizer=r1, second_stage_final_layer_regularizer=r2o,
        second_stage_first_final_layer_regularizer=r2i,
        regularizer_annealing_method=c.reg_annealing_method,
        consider_prev_weight=c.consider_prev_weight,
        log_per_epoch=c.log_per_epoch, plot_loss=c.plot_loss, validation_dataloader=lv)
    m.mean_outcome_proxy_feature = None
    return m


def posthoc_normalize(model, ds2, P):
    """The notebook's E[phi|A]=1 repair: divide out a poly fit from the stage-2 fold, floored, clipped."""
    with torch.no_grad():
        A_sc, Z_sc = ds2.A.to(DEVICE), ds2.Z.to(DEVICE)
        phi = model.predict_treatment_bridge_function(A_sc, Z_sc, None).cpu().numpy().reshape(-1)
    a = A_sc.cpu().numpy().reshape(-1)
    coef = np.polyfit(a, phi, P.norm_degree)
    lo, hi = float(a.min()), float(a.max())
    orig = model.predict_treatment_bridge_function

    def normalized(treatment, treatment_proxy, backdoor=None, _o=orig):
        out = _o(treatment, treatment_proxy, backdoor)
        ae = np.clip(treatment.view(-1).detach().cpu().numpy(), lo, hi)
        m = np.clip(np.polyval(coef, ae), P.norm_floor, None)
        return out / torch.as_tensor(m, dtype=out.dtype, device=out.device).reshape(out.shape)

    model.predict_treatment_bridge_function = normalized
    return model


def fit_treatment(d, l1, l2, lv, ds2, P):
    c = AlternativeTreatmentPCLConfig.abortion_and_crime(); c.device = DEVICE
    if P.tm_reg2 is not None:
        c.reg_second_stage = P.tm_reg2
    if P.tm_epochs is not None:
        c.n_epochs = P.tm_epochs
    build_fn = lambda device: build_nets_for_alternative_treatment_pcl_net_abortion_and_crime_experiment(
        device, width=P.tm_width, first_stage_a_out=P.tm_first_out, z_out=P.tm_z_out,
        w_out=P.tm_w_out, second_stage_a_out=P.tm_a2_out, dropout=P.tm_dropout)
    # kkt: a FACTORY (fresh constraint per member -- rbf/indicator bases cache from their first batch)
    constraint_factory = (lambda: ConditionalMeanConstraint(basis="poly", degree=P.norm_degree, mode="kkt")) \
        if P.normalization == "kkt" else None
    if P.tm_members > 1:
        m = train_alternative_treatment_pcl_net_ensemble(
            build_nets_fn=build_fn,
            first_stage_train_dataloader=l1, second_stage_train_dataloader=l2,
            validation_dataloader=lv, n_members=P.tm_members, device=c.device, base_seed=1000,
            lr_stage1=c.lr_first_stage, lr_stage2=c.lr_second_stage_ax,
            weight_decay=c.weight_decay, n_epochs=c.n_epochs,
            stage1_iter=c.stage1_iter, stage2_iter=c.stage2_iter,
            first_stage_final_layer_regularizer=c.reg_first_stage,
            second_stage_first_final_layer_regularizer=c.reg_second_stage_first,
            second_stage_final_layer_regularizer=c.reg_second_stage,
            consider_prev_weight=c.consider_prev_weight,
            loss_fn=get_loss_function(c.first_stage_loss_name, c.first_stage_loss_kwargs),
            head_mode="closed_form", normalization_constraint=constraint_factory)
    else:
        constraint = constraint_factory() if constraint_factory else None
        nets = build_fn(c.device)
        m = AlternativeTreatmentBridgePCLNET(nets[0], nets[1], nets[2], nets[3], device=c.device,
                                             normalization_constraint=constraint)
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
    if P.normalization == "posthoc":
        m = posthoc_normalize(m, ds2, P)
    return m


# ================================================================ estimation on ONE held-out fold
def estimate(d, om, tm, le, dse, lv, P):
    feat = torch.nn.Identity()
    om.eval()
    W_e = dse.W.to(DEVICE)
    grid = d["dev_tf"][0].transform(d["do_A"].to(DEVICE))
    with torch.no_grad():
        curve = [om.predict_outcome_bridge_function(
            a.reshape(1, -1).expand(W_e.shape[0], -1), W_e).mean(0) for a in grid]
    f_om = d["dev_tf"][1].inverse_transform(torch.stack(curve)).detach().cpu().numpy()

    hidden = [int(x) for x in P.third_hidden.split(",")]

    def head(ds, dsv, loss_fn, n_epochs=None):
        return train_third_stage_treatment_pcl_net_ensemble(
            model_class=ConditionalMeanMLP,
            dataloader=DataLoader(ds, batch_size=P.batch_size, shuffle=True),
            val_dataloader=DataLoader(dsv, batch_size=P.batch_size, shuffle=True),
            n_members=P.third_members, device=DEVICE, lr=P.third_lr, weight_decay=1e-6,
            n_epochs=(n_epochs if n_epochs else P.third_epochs), loss_fn=loss_fn, gap_penalty_weight=0.0,
            log_per_epoch=10 ** 9, input_dim=1, output_dim=1, hidden_dims=hidden,
            dropout_rate=P.third_dropout)

    def read(net, ds):
        x = ds.input_transformer.to(DEVICE).transform(feat(grid))
        return ds.outcome_transformer.to(DEVICE).inverse_transform(net(x)).detach().cpu().numpy()

    ds, dsv = create_third_stage_dataset_for_treatment_pcl_net_ate(
        tm, le, lv, TorchStandardScaler(), TorchStandardScaler(),
        treatment_featurizer=feat, device=DEVICE)
    _tm_ep = getattr(P, "tm_third_epochs", 0) or P.third_epochs
    _dr_ep = getattr(P, "dr_third_epochs", 0) or P.third_epochs
    f_tm = read(head(ds, dsv, nn.HuberLoss(delta=2.0), n_epochs=_tm_ep), ds)

    def slack(creator, allow_shrink=False):
        s, sv = creator(om, tm, le, lv, outcome_transformer=TorchStandardScaler(),
                        input_transformer=TorchStandardScaler(), input_type="features",
                        treatment_featurizer=feat, device=DEVICE)
        net = head(s, sv, nn.MSELoss(), n_epochs=_dr_ep)
        corr = read(net, s)
        if P.dr_val_shrink and allow_shrink and len(sv) > 0:
            # Validation-selected mean-preserving shrinkage of the correction's A-varying part
            # (V1 only): lam = <t_dev, p_dev>/||p_dev||^2 on the held-out DR pseudo-outcomes;
            # noise -> lam -> 0 (fall back toward OM up to the un-shrunk level term), signal ->
            # lam -> 1. lam is invariant to the affine outcome transform, so it is computed on the
            # transformed val tensors and applied to the natural-scale curve. The eval-mode forward
            # consumes no RNG, so turning the shrink on cannot perturb anything trained after it.
            # ASSUMES the net was NOT checkpoint-selected on sv: head passes log_per_epoch=1e9,
            # so the trainer keeps the last epoch. If periodic val checkpointing is ever restored
            # there, pv becomes optimistically correlated with tv and lam biases toward 1.
            net.eval()
            with torch.no_grad():
                pv = net(sv.tensors[0].to(DEVICE)).detach().cpu().numpy().reshape(-1)
            tv = sv.tensors[1].detach().cpu().numpy().reshape(-1)
            pv_c, tv_c = pv - pv.mean(), tv - tv.mean()
            lam = float(np.clip(float(np.dot(tv_c, pv_c)) / (float(np.dot(pv_c, pv_c)) + 1e-8),
                                0.0, 1.0))
            corr = corr.mean() + lam * (corr - corr.mean())
        return corr

    return {"OutcomeNet": f_om, "TreatmentNet": f_tm,
            "DRPCLNET-V1": f_om + slack(DR1, allow_shrink=True),
            "DRPCLNET-V2": f_om + f_tm - slack(DR2)}


# ================================================================ one full run
def run_once(data_seed, train_seed, P):
    d = build_data(data_seed, train_seed, P)
    lv = DataLoader(d["val"], batch_size=P.batch_size, shuffle=False)
    mse = lambda f: float(np.mean((np.asarray(f).reshape(-1, 1) - d["EY"].reshape(-1, 1)) ** 2))
    curves = []
    for s1, s2, est in d["assignment"]:
        torch.manual_seed(train_seed)                    # same init per loop, as in the notebook
        ds1, ds2, dse = d["mk"](s1), d["mk"](s2), d["mk"](est)
        l1 = DataLoader(ds1, batch_size=P.batch_size, shuffle=True)
        l2 = DataLoader(ds2, batch_size=P.batch_size, shuffle=True)
        le = DataLoader(dse, batch_size=P.batch_size, shuffle=True)
        om = fit_outcome(d, l1, l2, lv, P)
        tm = fit_treatment(d, l1, l2, lv, ds2, P)
        curves.append(estimate(d, om, tm, le, dse, lv, P))
    cf = {n: np.mean([c[n] for c in curves], axis=0) for n in ESTIMATORS}
    arm_name = "cross_fitted" if P.split_mode == "crossfit" else (
        "overlap%s_p%g_%s" % ("rot" if P.rotate_overlap else "",
                                              P.stage_perc, P.est_mode))
    out = {arm_name: {n: mse(cf[n]) for n in ESTIMATORS}}
    out["_split_diag"] = d["split_diag"]

    if P.include_single_split:
        # DISJOINT single-split arm: thirds of the training rows, one per role -- the no-cross-fitting
        # comparison that still satisfies the disjointness premise (NOT the legacy overlapping 75/75).
        torch.manual_seed(train_seed)
        t = d["train_idx"]; third = len(t) // 3
        ds1, ds2, dse = d["mk"](t[:third]), d["mk"](t[third:2 * third]), d["mk"](t[2 * third:])
        l1 = DataLoader(ds1, batch_size=P.batch_size, shuffle=True)
        l2 = DataLoader(ds2, batch_size=P.batch_size, shuffle=True)
        le = DataLoader(dse, batch_size=P.batch_size, shuffle=True)
        om = fit_outcome(d, l1, l2, lv, P)
        tm = fit_treatment(d, l1, l2, lv, ds2, P)
        ss = estimate(d, om, tm, le, dse, lv, P)
        out["single_split_disjoint"] = {n: mse(ss[n]) for n in ESTIMATORS}
    return out


# ================================================================ driver
def main():
    P = build_parser().parse_args()
    out_dir = os.path.join(HERE, "..", "Results", "Abortion_CrossFitting")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    seed_part = "s%s" % "-".join(str(s) for s in P.data_seeds)
    out = os.path.join(out_dir, "DRNMEPCL_DensRatioFree_AbortionCF_%s_%s_%s.csv"
                       % (P.tag, seed_part, stamp))
    fields = ["Algorithm", "Arm", "Data_Seed", "Repeat", "Train_Seed", "Causal_MSE",
              "stage1_rows", "stage2_rows", "est_rows", "overlap_rows", "overlap_of_pool",
              "est_reused_frac", "hostname"] + HYPER_COLUMNS
    with open(out, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    meta = {k: getattr(P, k) for k in HYPER_COLUMNS}
    meta["hostname"] = socket.gethostname()
    print("writing to %s" % out, flush=True)

    for data_seed in P.data_seeds:
        for repeat in range(P.repeat_offset, P.repeat_offset + P.n_repeats):
            train_seed = 1000 * data_seed + repeat
            t0 = time.time()
            res = run_once(data_seed, train_seed, P)
            diag = res.pop("_split_diag", {})
            with open(out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                for arm, vals in res.items():
                    for est in ESTIMATORS:
                        w.writerow(dict(Algorithm=est, Arm=arm, Data_Seed=data_seed, Repeat=repeat,
                                        Train_Seed=train_seed, Causal_MSE=vals[est],
                                        **diag, **meta))
            print("data_seed %d repeat %d (%ds) | CF: %s"
                  % (data_seed, repeat, time.time() - t0,
                     "  ".join("%s %.4f" % (e[:4], list(res.values())[0][e])
                               for e in ESTIMATORS)),
                  flush=True)


if __name__ == "__main__":
    main()
