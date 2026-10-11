"""Multi-seed simulation: density-ratio-FREE DR-PCL, STRICT three-fold cross-fitting on the Deaner
grade-retention benchmark (IM or IR) -- ALL THREE normalization arms (none / posthoc / kkt) run
PAIRED in every (seed, repeat).

Mirrors a tutorial notebook (not included in this package)
(the STRICTEST protocol in the repo, adopted from that notebook verbatim):
  * one DISJOINT fold per stage (stage-1 CME / stage-2 head / stage-3 estimation), cyclically rotated
    so every row serves every role exactly once; disjointness + coverage asserted per loop;
  * preprocessing (scalers) fitted on STAGE-1 ONLY, per loop -- not on the pooled training rows;
  * ONE full batch per fold (guarded), so the hard KKT constraint binds the WHOLE stage-2 fold;
  * the treatment bridge is an ensemble (default 3 members, per-member seeds);
  * fold curves combined by OBSERVATION-WEIGHTED averaging.
Config provenance: the notebook's own searches (poly-3 KKT, tm reg2 = 100, ens 3, third stage [32,32]).

PROTOCOL SEEDS. data_seed selects the shipped benchmark .npz (100, 200, ..., 1000);
train_seed = data_seed * 100 + repeat drives the val/fold permutation and, through the notebook's
explicit offsets (+101 nuisances, +202 + 10000*member bridge members, +303 third stages), every init.
Because every stage re-seeds EXPLICITLY, the arms are paired by construction: identical data, folds,
outcome bridge (shared -- it does not depend on the normalization) and identical member / third-stage
seeds; the normalization is the only difference.

NORMALIZATION ARMS. Default --normalization all = [none, posthoc, kkt]:
  none    = the raw ensemble, NO repair of E[varphi|A] -- the control arm that measures what the
            normalizations are actually buying (its flat_max / raw_moment_max columns quantify the
            raw conditional bias the other two arms remove);
  posthoc = the polynomial repair: divide out a degree-`norm_degree` polynomial fit of
            E[varphi|A] (fitted on THIS loop's stage-2 fold, floored, clipped), wrapping the ensemble;
  kkt     = ConditionalMeanConstraint(poly, degree=kkt_degree) inside every member's closed-form head.
The none and posthoc arms SHARE one trained ensemble (posthoc only wraps the predictor, so the bridge
is trained once and the wrapper is the arms' entire difference -- exact pairing by construction, and
one bridge fit saved per loop); the kkt arm trains its own ensemble from the same per-member seeds.
Deaner is the benchmark where posthoc catastrophically failed 2/10 seeds in the single-split study,
so this three-way pairing is the headline ablation here.

Usage:
  python Deaner_CrossFitting_Simulation.py --data_seeds 100 200 300 --n_repeats 3
  python Deaner_CrossFitting_Simulation.py --deaner_id IR --data_seeds 100 --normalization kkt
"""
import argparse, copy, csv, datetime, os, socket, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import math
import torch
from torch import nn
from torch.utils.data import DataLoader

from simulation_utils.data_generating_processes import read_deaner_dataset
from torch_utils.scalers import TorchStandardScaler, TorchIdentityTransformer
from torch_utils.dataset_classes import ProxyDataset
from torch_utils.networks import ConditionalMeanMLP
from torch_utils.torch_loss import get_loss_function
from python_utils.helpers import slice_tuple

from simulation_utils.outcome_pcl_net_configs import OutcomePCLConfig
from simulation_utils.outcome_pcl_net_nn_structures import (
    build_nets_for_outcome_pcl_net_deaner_experiment)
from simulation_utils.outcome_pcl_net_optimizers import build_adam_optimizers_for_outcome_pcl_net
from neural_causal_learning.proxy_outcome_neural_mean_embedding import (
    OutcomeBridgePCLNET, train_deep_feature_proxy_closed_form_ate_model)

from simulation_utils.alternative_treatment_pcl_net_configs import AlternativeTreatmentPCLConfig
from simulation_utils.treatment_pcl_net_nn_structures import (
    build_nets_for_alternative_treatment_pcl_net_deaner_experiment)
from simulation_utils.alternative_treatment_pcl_net_optimizers import (
    build_adam_optimizers_for_alternative_treatment_pcl_net)
from neural_causal_learning.alternative_treatment_neural_mean_embedding import (
    AlternativeTreatmentBridgePCLNET, EnsembleAlternativeTreatmentBridge,
    train_alternative_treatment_pcl_net_ate_model,
    create_third_stage_dataset_for_treatment_pcl_net_ate)
from neural_causal_learning.normalization_constraint import ConditionalMeanConstraint
from neural_causal_learning.proxy_treatment_neural_mean_embedding import (
    train_third_stage_treatment_pcl_net_ensemble)
from neural_causal_learning.alternative_doubly_robust_proxy_neural_mean_embedding import (
    create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_ate as DR1,
    create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_ate_v2 as DR2)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ESTIMATORS = ["OutcomeNet", "TreatmentNet", "DRPCLNET-V1", "DRPCLNET-V2"]

# The notebook pins cuDNN to deterministic kernels; without this the treatment-bridge training picks
# different kernels and the run is NOT reproducible against the notebook (measured: OM identical,
# TM/DR drifting).
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_seeds", type=int, nargs="+",
                   default=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                   help="benchmark .npz seeds shipped with the Deaner data")
    p.add_argument("--n_repeats", type=int, default=3)
    p.add_argument("--deaner_id", choices=["IM", "IR"], default="IM")
    # -- protocol (the notebook's strict scheme) --
    p.add_argument("--val_perc", type=float, default=0.10)
    p.add_argument("--n_folds", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2048,
                   help="must cover the largest fold: ONE batch per fold => fold-wide hard KKT")
    # -- normalization arms --
    p.add_argument("--normalization", choices=["all", "both", "posthoc", "kkt", "none"],
                   default="all",
                   help="all = [none, posthoc, kkt] paired; both = [posthoc, kkt]; or a single arm")
    p.add_argument("--kkt_degree", type=int, default=3,
                   help="KKT poly basis degree (the notebook's searched value)")
    p.add_argument("--norm_degree", type=int, default=5, help="posthoc polynomial degree")
    p.add_argument("--norm_floor", type=float, default=0.15, help="posthoc divisor floor")
    # -- bridges (notebook-searched values) --
    p.add_argument("--om_epochs", type=int, default=50)
    p.add_argument("--tm_epochs", type=int, default=100)
    p.add_argument("--tm_reg2", type=float, default=100.0)
    p.add_argument("--tm_members", type=int, default=3)
    # -- third stage --
    p.add_argument("--third_lr", type=float, default=1e-3)
    p.add_argument("--third_epochs", type=int, default=100)
    # --- third-stage options added; BOTH default to the shipped behaviour ---
    # CENTERED treatment third stage: under E[phi|A=a]=1 (kkt/posthoc), f(a)=E[phi*(Y-c)|A=a]+c for any
    # c. c=mean(Y) removes the phi-error x E[Y] cross term, >99% of the target's variance here
    # (mean(Y) ~ 4.8-5.0 vs signal sd ~0.05-0.06, a 77-95x amplification). Measured on 10 seeds,
    # repeat 5, kkt: TreatmentNet 2.52x -> 1.32x flat (IM, 8/10, p=0.010) and 7.25x -> 3.07x (IR, 9/10,
    # p=0.014); OutcomeNet and DRPCLNET-V1 are BIT-IDENTICAL (V1 is already centered by h).
    p.add_argument("--third_center", type=int, default=0, choices=[0, 1])
    p.add_argument("--third_center_c", type=float, default=float("nan"),
                   help="override c; 0.0 makes the centered path algebraically identical to the shipped "
                        "one -- the equivalence gate")
    # RFF third-stage INPUT featurization: z(a)=sqrt(2/D)cos(omega a + b), omega~N(0,bw^-2), FROZEN,
    # with the shipped ConditionalMeanMLP trained on top. Measured (IR, 5 seeds, repeat 5, kkt,
    # CENTERED): DRPCLNET-V1 0.000474 -> 0.000331 (-30%); TreatmentNet unchanged (3.14x vs 3.12x).
    # NOTE this is a FIXED RANDOM BASIS on the third-stage input -- disclose it if adopted.
    p.add_argument("--third_head", choices=["mlp", "rff_mlp"], default="mlp")
    p.add_argument("--rff_dim", type=int, default=128)
    p.add_argument("--rff_bandwidth", type=float, default=1.0)
    p.add_argument("--third_hidden", type=str, default="32,32")
    p.add_argument("--third_members", type=int, default=5)
    p.add_argument("--third_log_per_epoch", type=int, default=10,
                   help="best-val checkpoint frequency of the third-stage trainer (notebook: 10)")
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
    p.add_argument("--tag", default="baseline")
    return p


HYPER_COLUMNS = ["third_center", "third_head", "rff_dim", "rff_bandwidth",
                 "deaner_id", "split_mode", "stage_perc", "est_mode", "rotate_overlap", "val_perc", "n_folds", "batch_size", "kkt_degree", "norm_degree",
                 "norm_floor", "om_epochs", "tm_epochs", "tm_reg2", "tm_members",
                 "third_lr", "third_epochs", "third_hidden", "third_members", "third_log_per_epoch", "tag"]


# ================================================================ data / folds (notebook cell 4)
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


def build_data(data_seed, train_seed, P):
    set_seed(train_seed)
    W, Z, A, Y, do_A, EY = read_deaner_dataset(os.path.join(REPO, "data", "deaner"),
                                               P.deaner_id, data_seed)
    t = lambda v: torch.tensor(v, dtype=torch.float32)
    tens = (t(A), t(Y), t(Z), t(W))
    n = tens[0].shape[0]
    dr = torch.ones(n, 1)
    perm = np.random.permutation(n)
    n_val = int(n * P.val_perc)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    folds = np.array_split(train_idx, P.n_folds)
    if P.split_mode == "crossfit":
        assignment = [(folds[(k + 1) % P.n_folds], folds[(k + 2) % P.n_folds], folds[k])
                      for k in range(P.n_folds)]
        split_diag = dict(stage1_rows=len(assignment[0][0]), stage2_rows=len(assignment[0][1]),
                          est_rows=len(assignment[0][2]), overlap_rows=0, overlap_of_pool=0.0,
                          est_reused_frac=0.0)
        for s1, s2, s3 in assignment:
            for a, b in ((s1, s2), (s1, s3), (s2, s3), (s1, val_idx), (s2, val_idx), (s3, val_idx)):
                assert len(np.intersect1d(a, b)) == 0
            assert np.array_equal(np.sort(np.concatenate([s1, s2, s3])), np.sort(train_idx))
            assert max(len(s1), len(s2), len(s3)) <= P.batch_size, \
                "batch_size must cover the largest fold (fold-wide hard KKT)"
    else:
        assignment, split_diag = overlap_assignment(train_idx, P)
        for s1, s2, s3 in assignment:                    # what still HAS to hold under reuse
            for a, b in ((s1, val_idx), (s2, val_idx), (s3, val_idx)):
                assert len(np.intersect1d(a, b)) == 0
            assert max(len(s1), len(s2), len(s3)) <= P.batch_size, \
                ("batch_size must cover the largest stage -- otherwise the arm differs in lam_eff = "
                 "reg2/T as well as in splitting, and the fold-wide hard KKT stops being fold-wide")
            if P.est_mode == "holdout":
                assert len(np.intersect1d(s3, np.union1d(s1, s2))) == 0

    def make_loop_data(s1, s2, s3):
        """Preprocessing fitted on STAGE 1 ONLY -- the notebook's strict discipline."""
        transformers = [TorchStandardScaler() for _ in range(4)]
        for tf, x in zip(transformers, tens):
            tf.fit(x[s1])
        drt = TorchIdentityTransformer(); drt.fit(dr[s1])
        mk = lambda ix: ProxyDataset(slice_tuple(tens, ix), dens_ratio=dr[ix],
                                     dens_ratio_transformer=copy.deepcopy(drt),
                                     transformers=copy.deepcopy(transformers), device="cpu")
        return dict(stage1=mk(s1), stage2=mk(s2), stage3=mk(s3), validation=mk(val_idx),
                    dev_tf=[copy.deepcopy(x).to(DEVICE) for x in transformers])

    return dict(assignment=assignment, make_loop_data=make_loop_data,
                split_diag=split_diag, do_A=t(do_A), EY=EY, Y=Y)


# ================================================================ bridges (notebook cells 6+8)
def fit_outcome(l1, l2, lv, P):
    c = OutcomePCLConfig.deaner(); c.device = DEVICE
    c.lr_first_stage_base = c.lr_second_stage_ax = c.lr_second_stage_w = 1e-4
    c.weight_decay = 1e-7; c.stage1_iter = 10; c.consider_prev_weight = True
    c.n_epochs = P.om_epochs; c.reg_second_stage = (1e-1, 1e-1)
    c.log_per_epoch, c.plot_loss = 10 ** 9, False
    nets = build_nets_for_outcome_pcl_net_deaner_experiment(c.device)
    m = OutcomeBridgePCLNET(*nets, device=c.device)
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
    """The polynomial repair, applied to the ENSEMBLE's predictor."""
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


def fit_treatment(l1, l2, lv, P, norm_mode, experiment_seed):
    """The notebook's member-ensemble fit; per-member seeds make arms paired by construction."""
    if len(l2) != 1:
        raise ValueError("hard KKT binds the last stage-2 minibatch; use one full stage-2 batch")
    c = AlternativeTreatmentPCLConfig.deaner(); c.device = DEVICE
    c.lr_first_stage = 1e-3; c.lr_second_stage_ax = c.lr_second_stage_z = 1e-4
    c.weight_decay = 1e-6; c.scheduler_gamma = 1.0; c.n_epochs = P.tm_epochs
    c.reg_first_stage = c.reg_second_stage_first = 1e-2; c.reg_second_stage = P.tm_reg2
    c.consider_prev_weight = True
    members = []
    for member in range(P.tm_members):
        set_seed(experiment_seed + 202 + member * 10_000)
        nets = build_nets_for_alternative_treatment_pcl_net_deaner_experiment(DEVICE)
        constraint = (ConditionalMeanConstraint(basis="poly", degree=P.kkt_degree, mode="kkt",
                                                orthonormalize=True, eval_mode=True, rank_tol=1e-6)
                      if norm_mode == "kkt" else None)
        m = AlternativeTreatmentBridgePCLNET(*nets, device=DEVICE,
                                             normalization_constraint=constraint)
        o1, o2, s1, s2 = build_adam_optimizers_for_alternative_treatment_pcl_net(
            m, lr_first_stage=c.lr_first_stage, lr_second_stage_ax=c.lr_second_stage_ax,
            lr_second_stage_z=c.lr_second_stage_z, weight_decay=c.weight_decay,
            gamma=c.scheduler_gamma)
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
        members.append(m.eval())
    # NOTE the posthoc wrapper is applied by the CALLER (run_once), so the none and posthoc arms can
    # share this one trained ensemble -- the wrapper is their entire difference.
    return EnsembleAlternativeTreatmentBridge(members, device=DEVICE).eval()


# ================================================================ estimation (notebook cell 10)
def _centered_third_stage(pcl_model, loader, val_loader, treatment_featurizer, c):
    """EXACT copy of create_third_stage_dataset_for_treatment_pcl_net_ate with target phi*(Y-c).

    Replicated rather than post-patched because the creator STANDARDISES the composite target before
    storing it (`outcome_transformer.fit_transform`), so subtracting c*phi from the stored tensor would
    be subtracting from a transformed quantity. Everything else -- the y/dens-ratio inverse transforms,
    the featurizer, the fresh scalers, the attached transformer attributes -- is identical.
    """
    from torch.utils.data import TensorDataset
    yt = loader.dataset.transformers[1].to(DEVICE)
    drt = copy.deepcopy(loader.dataset.dens_ratio_transformer).to(DEVICE)

    def go(ldr):
        A_, T_ = [], []
        with torch.no_grad():
            for batch in ldr:
                batch = [b.to(DEVICE) for b in batch]
                treatment, outcome_scaled, treatment_proxy = batch[0], batch[1], batch[2]
                backdoor = pcl_model._backdoor_from(batch)
                y_un = yt.inverse_transform(outcome_scaled)
                phi = drt.inverse_transform(
                    pcl_model.predict_treatment_bridge_function(treatment, treatment_proxy, backdoor))
                T_.append(((y_un - c) * phi).cpu())
                A_.append(treatment_featurizer(treatment).cpu())
        return (torch.cat(A_), torch.cat(T_)) if A_ else (None, None)

    it, ot = TorchStandardScaler(), TorchStandardScaler()
    a_tr, t_tr = go(loader)
    dtr = TensorDataset(it.fit_transform(a_tr), ot.fit_transform(t_tr))
    dtr.input_transformer, dtr.outcome_transformer = it, ot
    dval = None
    a_v, t_v = go(val_loader) if val_loader is not None else (None, None)
    if a_v is not None:
        dval = TensorDataset(it.transform(a_v), ot.transform(t_v))
    return dtr, dval


class _FrozenRFF(torch.nn.Module):
    """z(a) = sqrt(2/D) cos(a @ omega.T + b), omega and b FROZEN. Deterministic in `seed`."""
    def __init__(self, input_dim, n_features, bandwidth, seed, device):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        w = torch.randn(n_features, input_dim, generator=g) / float(bandwidth)
        b = torch.rand(n_features, generator=g) * 2.0 * math.pi
        self.register_buffer("omega", w.to(device)); self.register_buffer("phase", b.to(device))
        self.scale = math.sqrt(2.0 / n_features)
    def forward(self, x):
        return self.scale * torch.cos(x @ self.omega.T + self.phase)


def estimate(d, loop, om, tm, P, third_seed):
    dev_tf = loop["dev_tf"]
    feat = torch.nn.Identity()
    _in_dim = 1
    if getattr(P, "third_head", "mlp") == "rff_mlp":
        feat = _FrozenRFF(1, P.rff_dim, P.rff_bandwidth, third_seed, DEVICE).to(DEVICE)
        _in_dim = P.rff_dim
    hidden = [int(x) for x in P.third_hidden.split(",")]
    # log_per_epoch is NOT cosmetic here: the third-stage trainer keeps a BEST-VALIDATION checkpoint
    # evaluated only at log epochs, so this sets the model-selection frequency. The notebook uses 10.
    third = dict(lr=P.third_lr, weight_decay=1e-6, n_epochs=P.third_epochs, loss_fn=nn.MSELoss(),
                 gap_penalty_weight=0.0, log_per_epoch=P.third_log_per_epoch, input_dim=_in_dim,
                 output_dim=1, hidden_dims=hidden, dropout_rate=0.05)
    l3 = DataLoader(loop["stage3"], batch_size=P.batch_size, shuffle=False)
    lv = DataLoader(loop["validation"], batch_size=P.batch_size, shuffle=False)

    om.eval()
    W3 = loop["stage3"].W.to(DEVICE)
    grid = dev_tf[0].transform(d["do_A"].to(DEVICE))
    with torch.no_grad():
        curve = [om.predict_outcome_bridge_function(
            a.reshape(1, -1).expand(W3.shape[0], -1), W3).mean(0) for a in grid]
    f_om = dev_tf[1].inverse_transform(torch.stack(curve)).detach().cpu().numpy()

    def head(ds, dsv):
        return train_third_stage_treatment_pcl_net_ensemble(
            model_class=ConditionalMeanMLP,
            dataloader=DataLoader(ds, batch_size=P.batch_size, shuffle=True),
            val_dataloader=DataLoader(dsv, batch_size=P.batch_size, shuffle=False),
            n_members=P.third_members, device=DEVICE, **third)

    def read(net, ds):
        x = ds.input_transformer.to(DEVICE).transform(feat(grid))
        with torch.no_grad():
            return ds.outcome_transformer.to(DEVICE).inverse_transform(net(x)).cpu().numpy()

    set_seed(third_seed)
    _c = 0.0
    if getattr(P, "third_center", 0):
        with torch.no_grad():
            _yt = l3.dataset.transformers[1].to(DEVICE)
            _c = float(_yt.inverse_transform(l3.dataset.Y.to(DEVICE)).mean())
        if P.third_center_c == P.third_center_c:
            _c = float(P.third_center_c)
    _rng = (torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    ds, dsv = create_third_stage_dataset_for_treatment_pcl_net_ate(
        tm, l3, lv, TorchStandardScaler(), TorchStandardScaler(),
        treatment_featurizer=feat, device=DEVICE)
    set_seed(third_seed)
    if getattr(P, "third_center", 0):
        ds, dsv = _centered_third_stage(tm, l3, lv, feat, _c)
    torch.set_rng_state(_rng[0])
    if _rng[1] is not None:
        torch.cuda.set_rng_state_all(_rng[1])
    f_tm = read(head(ds, dsv), ds) + _c

    def slack(creator, offset):
        set_seed(third_seed + offset)
        s, sv = creator(om, tm, l3, lv, outcome_transformer=TorchStandardScaler(),
                        input_transformer=TorchStandardScaler(), input_type="features",
                        treatment_featurizer=feat, device=DEVICE)
        set_seed(third_seed + offset)
        return read(head(s, sv), s)

    return {"OutcomeNet": f_om, "TreatmentNet": f_tm,
            "DRPCLNET-V1": f_om + slack(DR1, 1_000),
            "DRPCLNET-V2": f_om + f_tm - slack(DR2, 2_000)}


def phi_diag(tm, ds):
    with torch.no_grad():
        phi = tm.predict_treatment_bridge_function(
            ds.A.to(DEVICE), ds.Z.to(DEVICE), None).reshape(-1).cpu().numpy()
    a = ds.A.reshape(-1).cpu().numpy()
    grid = np.linspace(a.min(), a.max(), 200)
    flat = float(np.abs(np.polyval(np.polyfit(a, phi, 5), grid) - 1).max())
    raw = max(abs(float(np.mean(a ** k * (phi - 1)))) for k in range(6))
    return flat, raw


# ================================================================ one full run
def run_once(data_seed, train_seed, P, norm_modes):
    d = build_data(data_seed, train_seed, P)
    mse = lambda f: float(np.mean((np.asarray(f).reshape(-1, 1) - d["EY"].reshape(-1, 1)) ** 2))
    curves = {m: [] for m in norm_modes}
    diags = {m: [] for m in norm_modes}
    sizes = []
    for s1, s2, s3 in d["assignment"]:
        loop = d["make_loop_data"](s1, s2, s3)
        l1 = DataLoader(loop["stage1"], batch_size=P.batch_size, shuffle=True)
        l2 = DataLoader(loop["stage2"], batch_size=P.batch_size, shuffle=True)
        lv = DataLoader(loop["validation"], batch_size=P.batch_size, shuffle=False)
        set_seed(train_seed + 101)
        om = fit_outcome(l1, l2, lv, P)                  # SHARED across arms
        base_tm = None                                   # unconstrained ensemble: SHARED none+posthoc
        for mode in norm_modes:                          # arms paired via the explicit per-stage seeds
            if mode in ("none", "posthoc") and base_tm is not None:
                tm = base_tm
            else:
                tm = fit_treatment(l1, l2, lv, P, mode, experiment_seed=train_seed)
                if mode in ("none", "posthoc"):
                    base_tm = tm
            if mode == "posthoc":
                # wraps the predictor IN PLACE -- safe because main orders none before posthoc,
                # so the raw arm has already been estimated when the shared ensemble is wrapped
                tm = posthoc_normalize(tm, loop["stage2"], P)
            diags[mode].append(phi_diag(tm, loop["stage3"]))
            curves[mode].append(estimate(d, loop, om, tm, P, third_seed=train_seed + 303))
        sizes.append(len(s3))
        del om
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    w = np.asarray(sizes, float); w /= w.sum()           # observation-weighted, as in the notebook
    out = {}
    for mode in norm_modes:
        cf = {n: sum(wk * c[n] for wk, c in zip(w, curves[mode])) for n in ESTIMATORS}
        out[mode] = {"mse": {n: mse(cf[n]) for n in ESTIMATORS},
                     "flat_max": float(np.mean([x[0] for x in diags[mode]])),
                     "raw_moment_max": float(np.mean([x[1] for x in diags[mode]]))}
    out["_split_diag"] = d["split_diag"]
    return out


def main():
    P = build_parser().parse_args()
    # canonical order guarantees none precedes posthoc (they share one ensemble; the posthoc wrapper
    # mutates its predictor, so the raw arm must be estimated first)
    requested = {"all": ["none", "posthoc", "kkt"], "both": ["posthoc", "kkt"]}.get(
        P.normalization, [P.normalization])
    norm_modes = [m for m in ("none", "posthoc", "kkt") if m in requested]
    out_dir = os.path.join(HERE, "..", "Results", "Deaner_CrossFitting")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    seed_part = "%s_s%s" % (P.deaner_id, "-".join(str(s) for s in P.data_seeds))
    out = os.path.join(out_dir, "DRNMEPCL_DensRatioFree_DeanerCF_%s_%s_%s.csv"
                       % (P.tag, seed_part, stamp))
    fields = ["Algorithm", "Arm", "normalization", "Data_Seed", "Repeat", "Train_Seed",
              "Causal_MSE", "flat_max", "raw_moment_max",
              "stage1_rows", "stage2_rows", "est_rows", "overlap_rows", "overlap_of_pool",
              "est_reused_frac", "hostname"] + HYPER_COLUMNS
    with open(out, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    meta = {k: getattr(P, k) for k in HYPER_COLUMNS}
    meta["hostname"] = socket.gethostname()
    print("writing to %s   (norm arms: %s)" % (out, norm_modes), flush=True)

    for data_seed in P.data_seeds:
        for repeat in range(P.n_repeats):
            train_seed = data_seed * 100 + repeat
            t0 = time.time()
            res = run_once(data_seed, train_seed, P, norm_modes)
            diag = res.pop("_split_diag", {})
            arm_name = ("cross_fitted" if P.split_mode == "crossfit"
                        else "overlap%s_p%g_%s" % ("rot" if P.rotate_overlap else "",
                                              P.stage_perc, P.est_mode))
            with open(out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                for mode in norm_modes:
                    for est in ESTIMATORS:
                        w.writerow(dict(Algorithm=est, Arm=arm_name, normalization=mode,
                                        Data_Seed=data_seed, Repeat=repeat, Train_Seed=train_seed,
                                        Causal_MSE=res[mode]["mse"][est],
                                        flat_max=res[mode]["flat_max"],
                                        raw_moment_max=res[mode]["raw_moment_max"],
                                        **diag, **meta))
            print("%s data_seed %d repeat %d (%ds) | %s"
                  % (P.deaner_id, data_seed, repeat, time.time() - t0,
                     " || ".join("%s: %s" % (m, "  ".join("%s %.5f" % (e[:4], res[m]["mse"][e])
                                                          for e in ESTIMATORS))
                                 for m in norm_modes)), flush=True)


if __name__ == "__main__":
    main()
