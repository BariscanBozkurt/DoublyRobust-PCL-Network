"""Multi-seed, multi-hyperparameter simulation: density-ratio-FREE JOINT-ANCHOR ATT with DISJOINT-fold
cross-fitting on the synthetic low-dimensional ATT benchmark -- data size as the sweep axis, ALL THREE
normalization arms (none / posthoc / kkt) run PAIRED inside every (data_seed, repeat).

THE TARGET.  f_ATT(a, a') = E[Y^(a) | A = a'].  The first argument is the INTERVENTION level, the
second indexes the FACTUAL reference population and is NOT a second intervention.  One fit of one
bridge returns the whole surface, so every run is scored at several anchors (--anchors) and writes one
CSV row per (anchor, estimator, normalization arm).

NOTE ON THE DIAGONAL, because it is a natural thing to get wrong: f_ATT(a,a) is NOT f_ATE(a).  By
consistency f_ATT(a,a) = E[Y^(a) | A=a] = E[Y | A=a], the OBSERVATIONAL regression -- the ATE curve
integrates E[Y(a)|u] against the prior p(u), the ATT diagonal against the posterior p(u | A=a).  They
agree only when A is unconfounded.  The companion notebook verifies this numerically.

Protocol:
10% validation held out first, the rest cut into three MUTUALLY DISJOINT folds (stage-1 CME /
stage-2 head + normalizer / stage-3 estimation, roles rotating cyclically; disjointness and coverage
asserted per loop), scalers fitted once on the whole training portion, cross-fitted surface = plain
average of the three fold surfaces.

WHERE THE NORMALIZER LIVES.  D_hat(a,a') is a NUISANCE OF THE BRIDGE, so it is fitted on the STAGE-2
fold, never on the estimation fold.  That is the cross-fitting fix for the single-split protocol's own
caveat (derivation sec 11): fitted on the rows the third stages consume, a flexible D_hat could divide
varphi by itself and silently reduce the eq.-35 surface to the confounded regression E[Y|A=a].

NORMALIZATION ARMS.  Default --normalization all = [none, posthoc, kkt]:
  none    = the raw bridge, NO repair of E[varphi | A_1, A_2] = 1 (derivation eq. 41) -- the control
            that measures what the repairs buy; its moment_max / phi_mean columns quantify the raw
            violation on HELD-OUT pairs;
  posthoc = the derivation sec-10 calibration: fit D_hat(a,a') with the same two-input conditional-mean
            machinery on the stage-2 fold and wrap the bridge predictor to return varphi / D_hat, so
            BOTH third stages consume the calibrated bridge (dividing a fitted DR surface instead
            would also rescale the anchor-population term h(A_i, X_j, W_j), which carries no bridge
            scale error);
  kkt     = ConditionalMeanConstraintATT: the moments E[(varphi - 1) g_k(A_1,A_2)] = 0 imposed EXACTLY
            inside the closed-form head solve at every training step, with the tensor-product
            polynomial basis in (a, a') orthonormalised against the empirical pair measure and built
            over the SAME ordered pairs the head is solved on.
The arms are paired by RNG-STATE SNAPSHOTS: the outcome bridge is fitted once per fold and SHARED (it
does not depend on the normalization), every bridge fit replays the post-outcome state, and every
arm's estimation replays the post-bridge state -- so data, folds, inits, dropout masks, loader
shuffles and third-stage members are identical across arms.  The none and posthoc arms SHARE one
trained bridge (post-hoc only wraps the predictor, and is applied after the raw arm has been read),
so a three-arm run trains only TWO bridges per fold.

THE PAIR OBJECTIVE AND THE SIZE AXIS.  The complete ordered-pair U-statistic (eq. 24) is O(M^2) in
the stage-2 fold size when psi_{AA'} is a joint MLP on concat(a, a'): 359k pairs at n = 2000 but 36M
at n = 20000, per epoch, which is not runnable across a size sweep.

The DEFAULT here is therefore `--anchor_mode tensor`: psi_{AA'} = psi_A(a) (x) psi_A'(a').  A
SEPARABLE anchor feature makes both the objective and the normal equations factorise into O(M) einsum
contractions of row moments (minus an O(M) diagonal correction for the excluded i == j terms), so the
COMPLETE U-statistic costs one pass over the rows at EVERY sample size.  It is exact -- verified
against brute-force enumeration to ~1e-16 relative error on the normal equations and ~1e-14 on the
loss -- so with this mode `--pair_mode` is IRRELEVANT: nothing is sampled and
there is no derangement approximation to confound the size axis with.

The price is the model class, not the estimator: a rank-truncated family of bivariate functions in
(a, a') rather than an arbitrary MLP on their concatenation, and a head of
source_out * anchor_out * z_out instead of anchor_out * z_out.  On this smooth low-dimensional DGP
that is not a real restriction.

FEATURIZER DIMENSIONS were selected by a search against the concat baseline at n = 2000 (grid
{4x4, 6x6, 8x8, 12x12, 8x8@width64}, 3 seeds x 4 anchors): psi_A and
psi_A' at 8 each, giving a head of 8 * 8 * 8 = 512.  8x8 was the only setting that beat concat on
every tuning seed for both estimators while keeping the bridge well conditioned (std(varphi) 0.80
against 1.39 for 12x12, which had a better DR mean but the wildest bridge).  6x6 is WORSE than
concat, so these dimensions are not a free parameter.

HONEST READING OF THAT SEARCH, from a confirmation on DISJOINT seeds 100-102
(on disjoint confirmation seeds): the TreatmentNet gain REPLICATES -- 0.176 -> 0.139, +0.037
(+2.9 sem, 3/3 seeds) -- but the DR-surface gain does NOT: +0.0008 (+0.1 sem, 2/3), against +0.011
on the tuning seeds.  So tensor mode buys (i) the EXACT complete U-statistic at every sample size,
which is the real reason it is the default, and (ii) a clearly better single treatment bridge; it is
DR-NEUTRAL, not DR-improving.  Do not report it as a DR improvement.

Use `--anchor_mode concat` to fall back to the joint MLP; there the enumerated pair path applies and
`--pair_mode` matters again (`full` = eq. 24, `derangement` = eq. 29).

Each invocation writes its OWN timestamped CSV (tag + size + seeds in the name) -- parallel
tasks never write to the same file.  Every hyperparameter is a CSV column.

Examples:
  python Synthetic_ATT_LowDim_CrossFitting_Simulation.py --n_sample 2000 --data_seeds 0 1 2 --n_repeats 3
  python Synthetic_ATT_LowDim_CrossFitting_Simulation.py --n_sample 20000 --data_seeds 4 --normalization kkt
  python Synthetic_ATT_LowDim_CrossFitting_Simulation.py --n_sample 2000 --data_seeds 11 --anchor_mode concat --pair_mode full
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

from simulation_utils.data_generating_processes import SyntheticLowDimATTBenchmark
from torch_utils.scalers import TorchStandardScaler, TorchIdentityTransformer
from torch_utils.dataset_classes import ProxyDataset
from torch_utils.networks import ConditionalMeanMLP
from python_utils.helpers import slice_tuple

from simulation_utils.outcome_pcl_net_configs import OutcomePCLConfig
from simulation_utils.outcome_pcl_net_nn_structures import (
    build_nets_for_outcome_pcl_net_synthetic_low_dim_experiment)
from simulation_utils.outcome_pcl_net_optimizers import build_adam_optimizers_for_outcome_pcl_net
from neural_causal_learning.proxy_outcome_neural_mean_embedding import (
    OutcomeBridgePCLNET, train_deep_feature_proxy_closed_form_ate_model,
    create_third_stage_dataset_for_outcome_pcl_net_att)

from simulation_utils.alternative_treatment_att_pcl_net_configs import AlternativeTreatmentATTPCLConfig
from simulation_utils.treatment_pcl_net_nn_structures import (
    build_nets_for_alternative_treatment_att_pcl_net_synthetic_low_dim_experiment,
    build_nets_for_alternative_treatment_att_pcl_net_synthetic_low_dim_tensor_experiment)
from simulation_utils.alternative_treatment_att_pcl_net_optimizers import (
    build_adam_optimizers_for_alternative_treatment_att_pcl_net)
from neural_causal_learning.alternative_treatment_att import (
    AlternativeTreatmentBridgePCLNETATT, train_alternative_treatment_pcl_net_att_model,
    create_third_stage_dataset_for_alternative_treatment_att,
    create_third_stage_dataset_for_dr_alternative_treatment_att,
    create_normalizer_dataset_for_alternative_treatment_att,
    apply_att_bridge_normalization, remove_att_bridge_normalization, predict_att_surface)
from neural_causal_learning.normalization_constraint import ConditionalMeanConstraintATT
from neural_causal_learning.proxy_treatment_neural_mean_embedding import (
    train_third_stage_treatment_pcl_net_ensemble)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ATT has ONE doubly robust estimator: the one-regression form of eq. 39, whose single regression IS
# the DR surface (there is no f_om + slack decomposition, hence no V1 / V2 split as in the ATE code).
ESTIMATORS = ["OutcomeNet", "TreatmentNet", "DRPCLNET"]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--n_repeats", type=int, default=3)
    # -- data --
    p.add_argument("--n_sample", type=int, default=2000,
                   help="dataset size; the size-scaling axis (sweep: 2000..20000)")
    p.add_argument("--anchors", type=float, nargs="+", default=[-1.0, -0.5, 0.25, 1.0],
                   help="reference populations a' the surface is scored at (one CSV row each)")
    p.add_argument("--do_A_size", type=int, default=100)
    # -- protocol --
    p.add_argument("--val_perc", type=float, default=0.1)
    p.add_argument("--n_folds", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=512)
    # -- outcome bridge --
    p.add_argument("--om_epochs", type=int, default=100)
    p.add_argument("--om_reg2_lo", type=float, default=1e-3)
    p.add_argument("--om_reg2_hi", type=float, default=10.0)
    # -- ATT treatment bridge --
    p.add_argument("--tm_epochs", type=int, default=100)
    p.add_argument("--tm_reg2", type=float, default=10.0, help="lambda of eq. 28")
    p.add_argument("--pair_mode", choices=["derangement", "full"], default="derangement",
                   help="second-stage objective: 'full' = complete ordered-pair U-statistic (eq. 24, "
                        "O(M^2) -- infeasible past n~5000), 'derangement' = eq. 29, O(M), same "
                        "conditional mean. See the module docstring before changing this.")
    p.add_argument("--n_derangements", type=int, default=10,
                   help="derangement samples per epoch in the SECOND STAGE (pair_mode=derangement)")
    p.add_argument("--pair_chunk_size", type=int, default=200000)
    # -- anchor featurizer: the O(M) einsum path --
    p.add_argument("--anchor_mode", choices=["concat", "tensor"], default="tensor",
                   help="'concat' = one MLP on (a, a'), objective costs enumerated ordered pairs. "
                        "'tensor' = psi_A(a) (x) psi_A'(a'), which makes the COMPLETE U-statistic of "
                        "eq. 24 factorise into O(M) einsum contractions -- exact, and it makes "
                        "--pair_mode irrelevant (no sampling is needed at any size).")
    p.add_argument("--source_out", type=int, default=8, help="tensor mode: dim of psi_A(a)")
    p.add_argument("--anchor_out", type=int, default=8, help="tensor mode: dim of psi_A'(a')")
    p.add_argument("--z_out", type=int, default=8, help="dim of psi_Z")
    p.add_argument("--w_out", type=int, default=8, help="dim of phi_W")
    p.add_argument("--first_stage_a_out", type=int, default=16, help="dim of phi_A1")
    p.add_argument("--net_width", type=int, default=32, help="featurizer hidden width")
    # -- normalization arms --
    p.add_argument("--normalization", choices=["all", "both", "none", "posthoc", "kkt"], default="all",
                   help="all = [none, posthoc, kkt] paired; both = [posthoc, kkt]; or a single arm")
    p.add_argument("--kkt_degree", type=int, default=2,
                   help="tensor-product polynomial degree in (a, a'): K = (degree+1)^2 constraints, "
                        "so keep it LOW")
    p.add_argument("--denominator_floor", type=float, default=0.1,
                   help="posthoc: |D_hat| below this (or non-positive) falls back to D_hat = 1")
    # -- third stages --
    p.add_argument("--third_lr", type=float, default=1e-3)
    p.add_argument("--third_epochs", type=int, default=100)
    p.add_argument("--third_members", type=int, default=5)
    p.add_argument("--third_hidden", type=str, default="32,64")
    p.add_argument("--third_dropout", type=float, default=0.01)
    p.add_argument("--third_pair_mode", choices=["derangement", "full"], default="derangement",
                   help="pairing for the THIRD-STAGE pseudo-outcome rows (eq. 37 vs 36)")
    p.add_argument("--third_n_derangements", type=int, default=10)
    p.add_argument("--tag", type=str, default="baseline", help="free-text label written to the CSV")
    p.add_argument("--out_dir", type=str, default="",
                   help="results directory; '' = ../Results/Synthetic_ATT_LowDim_CrossFitting")
    return p


HYPER_COLUMNS = ["n_sample", "val_perc", "n_folds", "batch_size", "om_epochs", "om_reg2_lo",
                 "om_reg2_hi", "tm_epochs", "tm_reg2", "pair_mode", "n_derangements",
                 "anchor_mode", "source_out", "anchor_out", "z_out", "w_out", "first_stage_a_out",
                 "net_width",
                 "kkt_degree", "denominator_floor", "third_lr", "third_epochs", "third_members",
                 "third_hidden", "third_dropout", "third_pair_mode", "third_n_derangements", "tag"]


# ================================================================ RNG snapshot helpers
def rng_snapshot():
    return (torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def rng_restore(state):
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


# ================================================================ data + folds
def build_data(data_seed, P):
    """Dataset, scalers, ground-truth ATT curves per anchor, and the disjoint fold assignment."""
    benchmark = SyntheticLowDimATTBenchmark()
    d = benchmark.generate_dataset(n=P.n_sample, seed=data_seed, anchor_a=P.anchors[0],
                                   do_A_range=(-1.0, 2.0), do_A_size=P.do_A_size,
                                   include_ate_curve=True)
    t = lambda x: torch.tensor(x, dtype=torch.float32)
    tensors = (t(d["A"]), t(d["Y"]), t(d["Z"]), t(d["W"]))
    do_A_t = torch.tensor(d["do_A"], dtype=torch.float32)
    truth = {a: benchmark.generate_att_curve(anchor_a=a, do_A_range=(-1.0, 2.0),
                                             do_A_size=P.do_A_size)[1] for a in P.anchors}

    n = P.n_sample
    dens_ratio = torch.ones(n, 1, dtype=torch.float32)       # placeholder: NOT a density ratio
    drt = TorchIdentityTransformer()
    transformers = [TorchStandardScaler() for _ in range(4)]

    np.random.seed(data_seed)                                # fold permutation is data-seed determined
    idx = np.random.permutation(n)
    n_val = int(n * P.val_perc)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    for tr, x in zip(transformers, tensors):
        tr.fit(x[train_idx])
    drt.fit(dens_ratio[train_idx])

    folds = np.array_split(train_idx, P.n_folds)
    if P.n_folds < 3:
        raise ValueError("one disjoint fold per stage needs at least 3 folds")
    assignment = []
    for k in range(P.n_folds):
        others = [folds[(k + off) % P.n_folds] for off in range(1, P.n_folds)]
        half = max(len(others) // 2, 1)
        s1, s2, est = np.concatenate(others[:half]), np.concatenate(others[half:]), folds[k]
        assert len(np.intersect1d(s1, s2)) == 0
        assert len(np.intersect1d(est, s1)) == 0 and len(np.intersect1d(est, s2)) == 0
        assert len(np.intersect1d(est, val_idx)) == 0
        assert len(np.union1d(np.union1d(s1, s2), est)) == len(train_idx)
        assignment.append((s1, s2, est))

    def make_ds(sub):
        return ProxyDataset(slice_tuple(tensors, sub), dens_ratio=dens_ratio[sub],
                            dens_ratio_transformer=copy.deepcopy(drt),
                            transformers=copy.deepcopy(transformers), device="cpu")

    return {"make_ds": make_ds, "assignment": assignment, "val_idx": val_idx,
            "do_A": do_A_t, "truth": truth}


# ================================================================ bridges
def fit_outcome(l1, l2, lv, P):
    cfg = OutcomePCLConfig.low_dim()
    cfg.device = DEVICE
    cfg.n_epochs = P.om_epochs
    cfg.lr_first_stage_base = 1e-3
    cfg.reg_second_stage = (P.om_reg2_lo, P.om_reg2_hi)
    nets = build_nets_for_outcome_pcl_net_synthetic_low_dim_experiment(cfg.device)
    m = OutcomeBridgePCLNET(nets[0], nets[1], nets[2], None, device=cfg.device)
    o1, o2, s1, s2 = build_adam_optimizers_for_outcome_pcl_net(
        m, cfg.lr_first_stage, cfg.lr_second_stage_ax, cfg.lr_second_stage_w,
        cfg.weight_decay, gamma=cfg.scheduler_gamma)
    reg1, reg2_inner, reg2_outer = cfg.regularizers
    return train_deep_feature_proxy_closed_form_ate_model(
        pcl_model=m, first_stage_train_dataloader=l1, second_stage_train_dataloader=l2,
        stage1_optimizers=o1, stage2_optimizers=o2, stage1_schedulers=s1, stage2_schedulers=s2,
        n_epochs=cfg.n_epochs, stage1_iter=cfg.stage1_iter, stage2_iter=cfg.stage2_iter,
        first_stage_final_layer_regularizer=reg1,
        second_stage_final_layer_regularizer=reg2_outer,
        second_stage_first_final_layer_regularizer=reg2_inner,
        regularizer_annealing_method=cfg.reg_annealing_method,
        consider_prev_weight=cfg.consider_prev_weight,
        log_per_epoch=10 ** 9, plot_loss=False, validation_dataloader=lv)


def fit_treatment(l1, l2, lv, P, constraint=None):
    """The joint-anchor ATT bridge varphi(a, a', Z), optionally under the KKT normalization."""
    cfg = AlternativeTreatmentATTPCLConfig.synthetic_low_dim()
    cfg.device = DEVICE
    cfg.n_epochs = P.tm_epochs
    cfg.reg_second_stage = P.tm_reg2
    if P.anchor_mode == "tensor":
        nets = build_nets_for_alternative_treatment_att_pcl_net_synthetic_low_dim_tensor_experiment(
            cfg.device, source_out=P.source_out, anchor_out=P.anchor_out, z_out=P.z_out,
            first_stage_a_out=P.first_stage_a_out, w_out=P.w_out, width=P.net_width)
    else:
        nets = build_nets_for_alternative_treatment_att_pcl_net_synthetic_low_dim_experiment(cfg.device)
    m = AlternativeTreatmentBridgePCLNETATT(
        nets[0], nets[1], nets[2], nets[3],
        pair_mode=P.pair_mode, n_derangements=P.n_derangements,
        pair_chunk_size=P.pair_chunk_size, normalization_constraint=constraint,
        anchor_mode=P.anchor_mode, device=cfg.device)
    o1, o2, s1, s2 = build_adam_optimizers_for_alternative_treatment_att_pcl_net(
        m, lr_first_stage=cfg.lr_first_stage, lr_second_stage_aa=cfg.lr_second_stage_aa,
        lr_second_stage_z=cfg.lr_second_stage_z, weight_decay=cfg.weight_decay,
        gamma=cfg.scheduler_gamma)
    m, _, _ = train_alternative_treatment_pcl_net_att_model(
        model=m, first_stage_train_dataloader=l1, second_stage_train_dataloader=l2,
        stage1_optimizers=o1, stage2_optimizers=o2, stage1_schedulers=s1, stage2_schedulers=s2,
        n_epochs=cfg.n_epochs, stage1_iter=cfg.stage1_iter, stage2_iter=cfg.stage2_iter,
        first_stage_final_layer_regularizer=cfg.reg_first_stage,
        second_stage_final_layer_regularizer=cfg.reg_second_stage,
        second_stage_first_final_layer_regularizer=cfg.reg_second_stage_first,
        validation_dataloader=lv, log_per_epoch=10 ** 9, plot_loss=False)
    return m


def fit_normalizer(tm, l2, lv, P, third_cfg):
    """D_hat(a,a') on the STAGE-2 fold, wrapped onto the bridge (never the estimation fold)."""
    ds, dsv = create_normalizer_dataset_for_alternative_treatment_att(
        tm, l2, lv, outcome_transformer=TorchStandardScaler(),
        input_transformer=TorchStandardScaler(), pair_mode=P.third_pair_mode,
        n_derangements=P.third_n_derangements, device=DEVICE)
    net = train_third_stage_treatment_pcl_net_ensemble(
        model_class=ConditionalMeanMLP,
        dataloader=DataLoader(ds, batch_size=P.batch_size, shuffle=True),
        val_dataloader=DataLoader(dsv, batch_size=P.batch_size, shuffle=True),
        n_members=P.third_members, device=DEVICE, **third_cfg)
    return apply_att_bridge_normalization(tm, net, ds, denominator_floor=P.denominator_floor,
                                          device=DEVICE)


# ================================================================ estimation on the held-out fold
def estimate(d, om, tm, le, dse, lv, P):
    """OutcomeNet / TreatmentNet / DR ATT surfaces, read on the do_A grid at every anchor."""
    hidden = [int(x) for x in P.third_hidden.split(",")]
    third_cfg = dict(lr=P.third_lr, weight_decay=1e-6, n_epochs=P.third_epochs, loss_fn=nn.MSELoss(),
                     gap_penalty_weight=0.0, log_per_epoch=10 ** 9, input_dim=2, output_dim=1,
                     hidden_dims=hidden, dropout_rate=P.third_dropout)
    pair_kw = dict(outcome_transformer=TorchStandardScaler(), input_transformer=TorchStandardScaler(),
                   pair_mode=P.third_pair_mode, n_derangements=P.third_n_derangements, device=DEVICE)

    def surface_head(ds, dsv):
        return train_third_stage_treatment_pcl_net_ensemble(
            model_class=ConditionalMeanMLP,
            dataloader=DataLoader(ds, batch_size=P.batch_size, shuffle=True),
            val_dataloader=DataLoader(dsv, batch_size=P.batch_size, shuffle=True),
            n_members=P.third_members, device=DEVICE, **third_cfg)

    om_ds, om_dsv = create_third_stage_dataset_for_outcome_pcl_net_att(om, le, lv, device=DEVICE)
    om_net = train_third_stage_treatment_pcl_net_ensemble(
        model_class=ConditionalMeanMLP,
        dataloader=DataLoader(om_ds, batch_size=P.batch_size, shuffle=True),
        val_dataloader=DataLoader(om_dsv, batch_size=P.batch_size, shuffle=True),
        n_members=P.third_members, device=DEVICE,
        **{**third_cfg, "input_dim": om_ds.tensors[0].shape[1],
           "output_dim": om_ds.tensors[1].shape[1]})

    tm_ds, tm_dsv = create_third_stage_dataset_for_alternative_treatment_att(tm, le, lv, **pair_kw)
    tm_net = surface_head(tm_ds, tm_dsv)
    dr_ds, dr_dsv = create_third_stage_dataset_for_dr_alternative_treatment_att(
        om, tm, le, lv, **pair_kw)
    dr_net = surface_head(dr_ds, dr_dsv)

    out = {}
    for a in P.anchors:
        anchor = torch.tensor([[a]], dtype=torch.float32)
        out[a] = {
            "OutcomeNet": om.pred_conditional_structural_function(
                d["do_A"], anchor, om_net, dse.transformers[0], dse.transformers[1]
            ).detach().cpu().numpy(),
            "TreatmentNet": predict_att_surface(tm_net, tm_ds, d["do_A"], anchor,
                                                treatment_transformer=dse.transformers[0],
                                                device=DEVICE),
            "DRPCLNET": predict_att_surface(dr_net, dr_ds, d["do_A"], anchor,
                                            treatment_transformer=dse.transformers[0],
                                            device=DEVICE)}
    return out


def phi_diag(tm, ds, degree=3):
    """Held-out check of eq. 41: max_k |E[(varphi - 1) g_k(a, a')]| over independent pairs."""
    probe = ConditionalMeanConstraintATT(basis="poly", degree=degree)
    A_, Z_ = ds.A.to(DEVICE), ds.Z.to(DEVICE)
    n = A_.shape[0]
    torch.manual_seed(0)
    shift = int(torch.randint(1, n, size=(1,)).item())
    src = torch.arange(n, device=DEVICE)
    anc = (src + shift) % n
    tm.eval()
    with torch.no_grad():
        varphi = tm.predict_treatment_bridge_function(A_[src], A_[anc], Z_[src]).reshape(-1)
    g = probe._raw_basis_pairs(A_[src].reshape(-1), A_[anc].reshape(-1))
    return (float(((varphi - 1.0)[None, :] * g).mean(dim=1).abs().max()),
            float(varphi.mean()), float(varphi.std()))


# ================================================================ one full run
def run_once(data_seed, train_seed, P, norm_modes):
    d = build_data(data_seed, P)
    lv = DataLoader(d["make_ds"](d["val_idx"]), batch_size=P.batch_size, shuffle=False)
    hidden = [int(x) for x in P.third_hidden.split(",")]
    norm_third_cfg = dict(lr=P.third_lr, weight_decay=1e-6, n_epochs=P.third_epochs,
                          loss_fn=nn.MSELoss(), gap_penalty_weight=0.0, log_per_epoch=10 ** 9,
                          input_dim=2, output_dim=1, hidden_dims=hidden,
                          dropout_rate=P.third_dropout)
    mse = lambda f, t: float(np.mean((np.asarray(f).reshape(-1, 1) - np.asarray(t).reshape(-1, 1)) ** 2))

    curves = {m: [] for m in norm_modes}
    diags = {m: [] for m in norm_modes}
    for s1, s2, est in d["assignment"]:
        ds1, ds2, dse = d["make_ds"](s1), d["make_ds"](s2), d["make_ds"](est)
        l1 = DataLoader(ds1, batch_size=P.batch_size, shuffle=True)
        l2 = DataLoader(ds2, batch_size=P.batch_size, shuffle=True)
        le = DataLoader(dse, batch_size=P.batch_size, shuffle=True)

        torch.manual_seed(train_seed)
        om = fit_outcome(l1, l2, lv, P)                  # SHARED across arms
        rng_after_om = rng_snapshot()

        base_tm, rng_after_bridge = None, None
        for mode in norm_modes:
            if mode in ("none", "posthoc"):
                if base_tm is None:
                    rng_restore(rng_after_om)
                    base_tm = fit_treatment(l1, l2, lv, P)      # unconstrained, shared by both arms
                    rng_after_bridge = rng_snapshot()
                tm = base_tm
                remove_att_bridge_normalization(tm)
                if mode == "posthoc":
                    fit_normalizer(tm, l2, lv, P, norm_third_cfg)   # D_hat on the STAGE-2 fold
            else:
                rng_restore(rng_after_om)
                tm = fit_treatment(l1, l2, lv, P, constraint=ConditionalMeanConstraintATT(
                    basis="poly", degree=P.kkt_degree, mode="kkt"))
                if rng_after_bridge is None:
                    rng_after_bridge = rng_snapshot()
            diags[mode].append(phi_diag(tm, dse))
            rng_restore(rng_after_bridge)
            curves[mode].append(estimate(d, om, tm, le, dse, lv, P))
        if base_tm is not None:
            remove_att_bridge_normalization(base_tm)
        del om, base_tm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = {}
    for mode in norm_modes:
        per_anchor = {}
        for a in P.anchors:
            cf = {e: np.mean([c[a][e] for c in curves[mode]], axis=0) for e in ESTIMATORS}
            per_anchor[a] = {e: mse(cf[e], d["truth"][a]) for e in ESTIMATORS}
        dg = np.mean(diags[mode], axis=0)
        out[mode] = {"mse": per_anchor, "moment_max": float(dg[0]),
                     "phi_mean": float(dg[1]), "phi_std": float(dg[2])}
    return out


def main():
    P = build_parser().parse_args()
    # canonical order guarantees none precedes posthoc (they share one bridge; the posthoc wrapper
    # mutates its predictor, so the raw arm must be read first)
    requested = {"all": ["none", "posthoc", "kkt"], "both": ["posthoc", "kkt"]}.get(
        P.normalization, [P.normalization])
    norm_modes = [m for m in ("none", "posthoc", "kkt") if m in requested]
    out_dir = P.out_dir or os.path.join(HERE, "..", "Results", "Synthetic_ATT_LowDim_CrossFitting")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    seed_part = "n%d_s%s" % (P.n_sample, "-".join(str(s) for s in P.data_seeds))
    out = os.path.join(out_dir, "DRNMEPCL_DensRatioFree_SynthATTLowDimCF_%s_%s_%s.csv"
                       % (P.tag, seed_part, stamp))
    fields = ["Algorithm", "Arm", "normalization", "Anchor", "Data_Seed", "Repeat", "Train_Seed",
              "Causal_MSE", "moment_max", "phi_mean", "phi_std", "hostname"] + HYPER_COLUMNS
    with open(out, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    meta = {k: getattr(P, k) for k in HYPER_COLUMNS}
    meta["hostname"] = socket.gethostname()
    print("writing to %s   (norm arms: %s, anchors: %s, pair_mode: %s)"
          % (out, norm_modes, P.anchors, P.pair_mode), flush=True)

    for data_seed in P.data_seeds:
        for repeat in range(P.n_repeats):
            train_seed = 1000 * data_seed + repeat
            t0 = time.time()
            res = run_once(data_seed, train_seed, P, norm_modes)
            with open(out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                for mode in norm_modes:
                    for a in P.anchors:
                        for e in ESTIMATORS:
                            w.writerow(dict(Algorithm=e, Arm="cross_fitted", normalization=mode,
                                            Anchor=a, Data_Seed=data_seed, Repeat=repeat,
                                            Train_Seed=train_seed,
                                            Causal_MSE=res[mode]["mse"][a][e],
                                            moment_max=res[mode]["moment_max"],
                                            phi_mean=res[mode]["phi_mean"],
                                            phi_std=res[mode]["phi_std"], **meta))
            print("n=%d data_seed %d repeat %d (%ds) | %s"
                  % (P.n_sample, data_seed, repeat, time.time() - t0,
                     " || ".join("%s: %s [phi %.3f]"
                                 % (m, "  ".join("%s %.4f" % (e[:4], res[m]["mse"][P.anchors[0]][e])
                                                 for e in ESTIMATORS), res[m]["phi_mean"])
                                 for m in norm_modes)), flush=True)


if __name__ == "__main__":
    main()
