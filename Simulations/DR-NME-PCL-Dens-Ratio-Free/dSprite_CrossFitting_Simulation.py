"""dSprite (DFPV-family, the DFPV authors' DGP) ATE with DISJOINT-fold cross-fitting, density-ratio-free.

Cross-fitted dSprite experiment on the V2 benchmark. The DGP is
dSprite_ProxyVariable_DatasetV2 (liyuan9988/DeepFeatureProxyVariable "Update on dSprite experiment"):
A and W are 4096-D images, Z is 3-D, structural function ((A.w)^2 - 3000)/500, 588-point test grid.

BENCHMARK CAVEAT (read before making any
claim off this sim): the dSprite PCL benchmark is structurally DEGENERATE for proximal-learning claims
(Z is an instrument, not a confounding proxy; positivity fails; the treatment bridge's population
optimum is exactly phi==1). Results here are OPTIMIZATION / MECHANISM results — during- vs
after-training normalization, DR robustness to a diverging bridge — never evidence of proximal
information extraction. That is why the phi1 arm is a PERMANENT, REQUIRED control.

ARMS (all paired inside every fold: ONE shared OutcomeNet per fold, RNG state replayed before each
arm's bridge + third stages). The posthoc polynomial arm of the other sims DOES NOT EXIST here — A is
a 4096-D image, a polynomial in "the treatment" has nothing to act on; the KKT constraint's
`projection` hook supplies the missing scalar instead. Any function of A alone yields a valid moment
condition (E[phi|A]=1 implies E[(phi-1)g(A)]=0 for every g), so the arms form a ladder of assumptions:
  none      unconstrained bridge (the notebook's arm; the control the comparison rests on)
  kkt-mean  degree-0 basis: enforces only E[phi]=1 DURING training. Assumes nothing about A.
  kkt-pca   degree-`pca_degree` basis on the 1st principal component of the stage-2 fold's images
            (unsupervised, never sees Y).
  phi1      phi forced to 1: NO proximal reweighting. Trains the unconstrained bridge anyway so its
            third stages are RNG-paired with the `none` arm, then overrides the bridge predictors.
            DR-V1 collapses to f_om + E[Y-h|A] residual regression — what "DR improved" must beat
            before it can be attributed to the proximal machinery.
  global    optional failed-control (--arms ... global): scalar posthoc phi/mean(phi). Documented to
            make TM ~20% WORSE (same constraint as kkt-mean, applied after training) — keep it out of
            default runs.

HEAD-SOLVE / BATCH RULES (the dSprite trap): the closed-form treatment head is a PROXIMAL iteration whose effective ridge is
reg2 / T with T = batches_per_epoch * bridge_epochs, and its effective sample size is the BATCH. With
batch < stage-2 fold both vary with n unless the batch scales. Default --batch 0 therefore AUTO-SCALES
batch = int(1024 * n_sample / 5000) (the corrected E1 design: 1024/2048/4096 at n=5000/10000/20000,
always 2 batches/epoch, T = 2 * bridge_epochs constant in n). --stage2_batch 0 copies --batch; never
set it to the full fold while --batch is smaller (the trainer zips the two loaders, so epochs collapse
to one step).

VERIFIED: run_once(data_seed=42, train_seed=42, n_sample=5000) with defaults gives, in the `none` arm:
OutcomeNet 28.1950, TreatmentNet 32.6260, DRPCLNET-V1 25.8568, DRPCLNET-V2 25.8920.

TUNED (screen seed 100 + confirmation seeds 101/102, firewalled from the reporting seeds):
launch reporting runs with
`--third_loss huber` (delta 2) — DR-V2 better 3/3 seeds in BOTH arms (kkt -1.2..-3.4%), TM better
5/6 cells, DR-V1/OM never hurt. Defaults stay notebook-faithful (mse) for the validation contract.
Do NOT combine --tm_reg2 300 with kkt arms (helps the none arm 3/3 but blew up kkt TM +44% on one
tuning seed); OM knobs are locally optimal as shipped (both ridge directions and 2x epochs hurt).

  python dSprite_CrossFitting_Simulation.py --data_seeds 42 --n_repeats 1                    # smoke
  python dSprite_CrossFitting_Simulation.py --data_seeds 0 7 13 21 42 --n_repeats 3
"""
import argparse, copy, csv, datetime, glob, os, socket, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from simulation_utils.data_generating_processes import dSprite_ProxyVariable_DatasetV2
from torch_utils.scalers import TorchStandardScaler, TorchIdentityTransformer
from torch_utils.dataset_classes import ProxyDataset
from torch_utils.networks import ConditionalMeanMLP
from torch_utils.torch_loss import get_loss_function
from python_utils.helpers import slice_tuple

from simulation_utils.outcome_pcl_net_configs import OutcomePCLdSpriteConfig
from simulation_utils.outcome_pcl_net_nn_structures import (
    build_nets_for_outcome_pcl_net_dsprite_experiment,
    build_nets_for_outcome_pcl_net_dsprite_experiment_sn,
    build_nets_for_outcome_pcl_net_dsprite_experiment_configurable)
from simulation_utils.outcome_pcl_net_optimizers import build_adam_optimizers_for_outcome_pcl_net
from neural_causal_learning import proxy_outcome_neural_mean_embedding_w_SGD as om_sgd
from neural_causal_learning import proxy_outcome_neural_mean_embedding_V2 as om_v2
from neural_causal_learning import alternative_treatment_neural_mean_embedding_V2 as tm_v2
from neural_causal_learning.proxy_outcome_neural_mean_embedding import (
    OutcomeBridgePCLNET, train_deep_feature_proxy_closed_form_ate_model,
)
from simulation_utils.treatment_pcl_net_nn_structures import (
    build_nets_for_alternative_treatment_pcl_net_dsprite_experiment_dfpv,
    build_nets_for_alternative_treatment_pcl_net_dsprite_experiment,
)
from neural_causal_learning.alternative_treatment_neural_mean_embedding import (
    train_alternative_treatment_pcl_net_ensemble,
    create_third_stage_dataset_for_treatment_pcl_net_ate,
)
from neural_causal_learning.normalization_constraint import ConditionalMeanConstraint
from neural_causal_learning.proxy_treatment_neural_mean_embedding import (
    train_third_stage_treatment_pcl_net_ensemble,
)
from neural_causal_learning.alternative_doubly_robust_proxy_neural_mean_embedding import (
    create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_ate,
    create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_ate_v2,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ESTIMATOR_NAMES = ["OutcomeNet", "TreatmentNet", "DRPCLNET-V1", "DRPCLNET-V2"]
ORIG_SUFFIX = " (orig)"                          # --om_both 1: rows of the shipped closed-form OutcomeNet family
OM_ROWS = ["OutcomeNet", "DRPCLNET-V1", "DRPCLNET-V2"]   # the rows that depend on the outcome bridge
ARM_CHOICES = ["none", "kkt-mean", "kkt-pca", "phi1", "global"]


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_seeds", type=int, nargs="+", default=[0, 7, 13, 21, 42],
                   help="reporting seeds of the dsprite_kkt programme; tuning seeds 100-102 are "
                        "firewalled — never mix the two sets in one CSV")
    p.add_argument("--n_repeats", type=int, default=3)
    p.add_argument("--repeat_offset", type=int, default=0)
    p.add_argument("--n_sample", type=int, default=5000)
    # split
    p.add_argument("--val_perc", type=float, default=0.1)
    p.add_argument("--n_folds", type=int, default=3)
    # PROTOCOL ( overlap decision). disjoint = the paper-premise 3-fold cross-fitting.
    # overlap = the an earlier internal version dSprite split (single overlapping split: stage1 = first 75%, stage2 =
    # last 75% of the train rows -- ~50% shared -- estimation = the stage-2 rows, mean-W feature from
    # stage-2). Admissible as a DISCLOSED Donsker-class variant; keep a disjoint arm paired. NOTE it
    # estimates the DR residual on rows the outcome bridge trained on, which optimistically shrinks
    # the correction toward OM -- report that as a protocol effect, not a method gain.
    p.add_argument("--protocol", choices=["disjoint", "overlap"], default="disjoint")
    # OUTCOME-BRIDGE ARCHITECTURE search. default = the shipped builder (numerically identical).
    # configurable = capacity multiplier + optional residual blocks on all THREE outcome featurizers.
    p.add_argument("--om_arch", choices=["default", "configurable", "dfpv", "sn"], default="default")
    # sn = OUR OutcomeBridgePCLNET + the DFPV-paper ingredients (spectral-norm nets, tensor-product first
    #      stage, 32-d features) -- the RECOMMENDED dSprite config; see the builder docstring. Pair with
    #      --y_scale raw --z_scale raw --om_epochs 20 --om_stage1_iter 20 --om_weight_decay 0.1 --batch <fold>.
    p.add_argument("--om_stage1_iter", type=int, default=None, help="None = config default (10)")
    # OM-ONLY loader batch (0 = share --batch, numerically identical). Lets the outcome bridge train
    # full-batch (the DFPV recipe) WITHOUT touching the treatment bridge's loaders -- its proximal
    # head's lam_eff = reg2/T depends on batches-per-epoch, so a shared full batch would silently
    # 5x its regularisation (measured: TM 31 -> 47 at n=2000).
    p.add_argument("--om_batch", type=int, default=0)
    p.add_argument("--om_sn_dim", type=int, default=32, help="sn builder feature dim (first stage = (d+1)^2-1)")
    # Head solve: our default is the PROXIMAL iterate (consider_prev_weight=True: lam cancels at the
    # fixed point, regularisation = early stopping, lam_eff = reg/T). 0 = plain closed-form ridge with
    # the annealed reg tuple -- the same closed-form 2SLS head DFPV uses. An implementation choice of
    # the closed-form head, not a method change. None = config default (True).
    p.add_argument("--om_consider_prev_weight", type=int, default=None)
    # OUTCOME TRAINER: closed_form = shipped (alternating, proximal head). v2 = OUR class + OUR
    # ProxyDataset batch loop with the PROFILED objective (head solved inside the loss, gradient
    # through the solve; fixed per-sample ridges --om_reg1_lo / --om_reg21_lo / --om_reg2_lo; exact
    # batch-accumulated full-data refit at the end). Full batch = --om_batch <fold size>.
    # Recommended dSprite config: --om_trainer v2 --om_arch sn --y_scale raw --z_scale raw
    #   --om_batch 2000 --om_epochs 20 --om_stage1_iter 20 --om_lr 1e-3 --om_weight_decay 0.1
    #   --om_reg1_lo 0.1 --om_reg21_lo 0.1 --om_reg2_lo 10
    p.add_argument("--om_trainer", choices=["closed_form", "v2"], default="closed_form")
    p.add_argument("--om_v2_bn_train_pred", type=int, default=0, help="v2 only: predict with BatchNorm in train mode (DFPV convention)")
    p.add_argument("--y_scale", choices=["std", "raw"], default="std")
    p.add_argument("--z_scale", choices=["std", "raw"], default="std")
    # OFFICIAL DFPV OUTCOMENET (Xu et al. 2021) as a drop-in `om`. Their unmodified code
    # scores 15.7-16.6 at n=5000 on this exact benchmark/seeds vs our 24.2.
    # Trained on RAW arrays of stage1 U stage2 (their own 50/50 internal split, seeded here for
    # pairing), mean-W re-computed on OUR estimation fold; h is returned in OUR scaled-Y space
    # because the DR creators inverse-transform it. Defaults = their published dfpv_dsprite.json.
    p.add_argument("--dfpv_lam1", type=float, default=0.1)
    p.add_argument("--dfpv_lam2", type=float, default=0.1)      # 10.0 scored 12.30 in their harness
    p.add_argument("--dfpv_n_epoch", type=int, default=20)     # 50 scored 21.17: do NOT over-train
    p.add_argument("--dfpv_stage1_iter", type=int, default=20)
    p.add_argument("--dfpv_wd", type=float, default=0.1)
    p.add_argument("--dfpv_lr", type=float, default=1e-3)       # torch Adam default = theirs
    p.add_argument("--dfpv_split", type=float, default=0.5)
    p.add_argument("--om_width_mult", type=float, default=1.0, help="scales [1024,512,128] widths")
    p.add_argument("--om_residual", type=int, default=0, help="1 = residual-block featurizers")
    p.add_argument("--om_n_blocks", type=int, default=2, help="residual depth (om_residual only)")
    p.add_argument("--overlap_frac", type=float, default=0.75,
                   help="stage-1 = first frac, stage-2 = last frac of the train rows (overlap only)")
    p.add_argument("--batch", type=int, default=0,
                   help="loader batch for stage 1, estimation, validation and third stage. "
                        "0 = AUTO int(1024 * n_sample / 5000), the E1 scaling that keeps both the "
                        "head's T and its per-batch sample fraction constant across sizes")
    p.add_argument("--stage2_batch", type=int, default=0,
                   help="stage-2 loader batch; 0 = same as --batch (keep them equal)")
    # normalization arms
    p.add_argument("--arms", nargs="+", choices=ARM_CHOICES,
                   default=["none", "kkt-mean", "kkt-pca", "phi1"])
    p.add_argument("--pca_degree", type=int, default=1, help="basis degree for the kkt-pca arm")
    # outcome bridge
    p.add_argument("--om_epochs", type=int, default=50)
    # only fully portable lever: post-bugfix, same protocol, log_cosh beat the hardcoded-MSE trainer
    # by ~3.2 MSE at n=5000 (18.01 vs 21.18). The closed-form trainer HARDCODES nn.MSELoss, so a
    # non-mse choice routes through the _w_SGD subclass with the old script's exact head settings
    # (lr 1e-2, 5 steps). Default mse = the shipped numerically identical path.
    p.add_argument("--om_loss", choices=["mse", "log_cosh", "huber"], default="mse")
    p.add_argument("--om_huber_delta", type=float, default=1.0)
    # Old-config ridge knobs silently changed by the OutcomePCLConfig.dsprite -> dSpriteConfig
    # swap (old reg_first_stage (10,100) vs current (5e-3,1e-3); reg_second_stage_first anneal
    # reversed). None = keep the current dataclass defaults.
    p.add_argument("--om_reg1_lo", type=float, default=None)
    p.add_argument("--om_reg1_hi", type=float, default=None)
    p.add_argument("--om_reg21_lo", type=float, default=None)
    p.add_argument("--om_reg21_hi", type=float, default=None)
    p.add_argument("--om_lr", type=float, default=1e-4)
    p.add_argument("--om_weight_decay", type=float, default=1e-7)
    p.add_argument("--om_reg2_lo", type=float, default=10.0, help="head ridge annealing start")
    p.add_argument("--om_reg2_hi", type=float, default=100.0, help="head ridge annealing end")
    # treatment bridge (density-ratio-free ensemble)
    p.add_argument("--tm_reg2", type=float, default=1000.0,
                   help="KAP head regularizer — an INVERSE STEP SIZE, effective ridge tm_reg2/T")
    p.add_argument("--n_bridges", type=int, default=5, help="bridge ensemble size (K)")
    p.add_argument("--bridge_epochs", type=int, default=50)
    p.add_argument("--bridge_base_seed", type=int, default=1000)
    # DFPV-STYLE TREATMENT BRIDGE: the nets/optimiser recipe that fixed the outcome
    # bridge. default = shipped builder + trainer defaults (numerically identical). dfpv = spectral-norm
    # ReLU nets, Adam lr 1e-3 both stages, weight_decay 0.1, stage1_iter 20; pair with
    # --bridge_epochs 20 and full-batch loaders (--batch/--stage2_batch >= fold size).
    p.add_argument("--tm_arch", choices=["default", "dfpv"], default="default")
    p.add_argument("--tm_lr1", type=float, default=None)
    p.add_argument("--tm_lr2", type=float, default=None)
    p.add_argument("--tm_wd", type=float, default=None)
    p.add_argument("--tm_stage1_iter", type=int, default=None)
    # TREATMENT TRAINER V2: our AlternativeTreatmentBridgePCLNET + our batch loop with the
    # profiled objective (stage-1 ridge AND the KAP head solved inside the losses; plain ridge lam, not
    # the proximal inverse-step-size), intercept cross terms, BN-train prediction; exact accumulated refit.
    # See src/neural_causal_learning/alternative_treatment_neural_mean_embedding_V2.py. Default = shipped.
    p.add_argument("--om_both", type=int, default=0,
                   help="1 = ALSO train the shipped closed-form OutcomeNet (sim-DEFAULT om_* config, standardised Y/Z) "
                        "next to the configured OutcomeNet, sharing the treatment bridge, folds and RNG; its rows are "
                        "written as '<name> (orig)' (OutcomeNet / DRPCLNET-V1 / DRPCLNET-V2). 0 = single OutcomeNet.")
    p.add_argument("--tm_trainer", choices=["closed_form", "v2"], default="closed_form")
    p.add_argument("--tm_stage2_iter", type=int, default=1, help="TM-V2 only: stage-2 featurizer updates per epoch (DFPV default 1)")
    p.add_argument("--tm_batch", type=int, default=0,
                   help="TM-only loader batch (0 = shipped --batch/--stage2_batch loaders); e.g. fold size for full batch")
    p.add_argument("--tm_reg1", type=float, default=0.1, help="v2: stage-1 ridge (per-sample)")
    p.add_argument("--tm_reg21", type=float, default=0.1, help="v2: stage-1 ridge re-solved inside stage 2")
    p.add_argument("--tm_cross_terms", type=int, default=1)
    p.add_argument("--tm_v2_bn_train_pred", type=int, default=0)
    # third stage / DR (raw 4096-D A regression)
    p.add_argument("--third_lr", type=float, default=1e-5)
    p.add_argument("--third_epochs", type=int, default=50)
    p.add_argument("--dr_third_lr", type=float, default=0.0,
                   help="third-stage lr for the two DR-correction regressions only (0 = --third_lr)")
    p.add_argument("--dr_third_epochs", type=int, default=0,
                   help="third-stage epochs for the two DR-correction regressions only (0 = --third_epochs)")
    p.add_argument("--third_members", type=int, default=5)
    p.add_argument("--third_hidden", type=str, default="1024,512")
    p.add_argument("--third_dropout", type=float, default=0.05)
    p.add_argument("--third_loss", choices=["mse", "huber", "log_cosh"], default="mse",
                   help="loss for ALL third-stage heads (f_tm and both DR slacks). huber with "
                        "delta 2 is the dsprite_kkt programme's DR-tuning winner at n=5000")
    p.add_argument("--huber_delta", type=float, default=2.0)
    p.add_argument("--dgp", choices=["v2", "org"], default="v2",
                   help="v2 = the README-updated DFPV dSprite (= ours, numerically identical); org = the DFPV "
                        "PROCEEDINGS DGP (||BA||^2 structural, Var(EY) 24.68). Rows carry the `dgp` column; "
                        "org results go to Results/dSprite_CrossFitting_org/.")
    p.add_argument("--tag", type=str, default="baseline")
    p.add_argument("--out_dir", type=str, default="")
    P = p.parse_args(argv)
    P.om_defaults = {a.dest: a.default for a in p._actions if a.dest.startswith(("om_", "dfpv_")) and a.dest != "om_both"}
    return P


HYPER_COLUMNS = ["n_sample", "val_perc", "n_folds", "protocol", "overlap_frac", "batch", "stage2_batch", "pca_degree",
                 "om_epochs", "om_lr", "om_weight_decay", "om_reg2_lo", "om_reg2_hi",
                 "tm_reg2", "n_bridges", "bridge_epochs", "bridge_base_seed",
                 "third_lr", "third_epochs", "dr_third_lr", "dr_third_epochs", "third_members", "third_hidden", "third_dropout",
                 "third_loss", "huber_delta", "repeat_offset", "om_arch", "om_stage1_iter", "om_batch", "om_sn_dim", "om_consider_prev_weight", "om_trainer", "om_v2_bn_train_pred", "y_scale", "z_scale", "dfpv_lam1", "dfpv_lam2", "dfpv_n_epoch", "dfpv_stage1_iter", "dfpv_wd", "dfpv_lr", "dfpv_split", "om_width_mult", "om_residual", "om_n_blocks", "om_loss", "om_huber_delta", "om_reg1_lo", "om_reg1_hi", "om_reg21_lo", "om_reg21_hi", "tm_arch", "tm_lr1", "tm_lr2", "tm_wd", "tm_stage1_iter", "tm_trainer", "tm_batch", "tm_stage2_iter", "tm_reg1", "tm_reg21", "tm_cross_terms", "tm_v2_bn_train_pred", "dgp", "om_both", "tag"]


def _third_loss_fn(P):
    if P.third_loss == "huber":
        return nn.HuberLoss(delta=P.huber_delta)
    if P.third_loss == "log_cosh":
        return get_loss_function("log_cosh")
    return nn.MSELoss()


DFPV_ROOT = os.path.join(REPO, "..", "external", "DeepFeatureProxyVariable")


class _DFPVAsOutcomeBridge:
    """Wrap a trained official DFPVModel so our DR creators / estimate() can consume it unchanged.
    Contract (the three methods the sim touches): predict_outcome_bridge_function(A,W,X) -> h in OUR
    scaled-Y space; compute_mean_outcome_proxy_feature(loader) -> sets the mean W-feature over the
    ESTIMATION fold; pred_structural_function(do_A, tA, tY) -> raw-unit curve. NOTE: faithful to their
    code, the nets are NOT switched to eval() (their predict_t runs BatchNorm in train mode)."""

    def __init__(self, model, y_transformer, device):
        self.model, self.y_transformer, self._device = model, y_transformer, device
        self.mean_outcome_proxy_feature = model.mean_outcome_proxy_feature

    def eval(self):
        return self

    def to(self, device):
        return self

    def compute_mean_outcome_proxy_feature(self, loader):
        # ONE full-fold pass, exactly as their fit_t does over the whole stage-2 set. Their nets stay
        # in train mode (BatchNorm uses batch statistics), so a per-minibatch pass over a SHUFFLED
        # loader made the mean-W feature -- and hence f_om -- differ across arms (17.10/17.63/17.93
        # on seed 100). BN batch stats over the whole fold are permutation-invariant => deterministic.
        with torch.no_grad():
            W_all = torch.cat([batch[3] for batch in loader]).to(self._device)
            self.model.mean_outcome_proxy_feature = self.model.outcome_proxy_net(W_all).mean(0, keepdim=True)
        self.mean_outcome_proxy_feature = self.model.mean_outcome_proxy_feature

    def predict_outcome_bridge_function(self, A, W, X=None):
        with torch.no_grad():
            h_raw = self.model.predict_bridge_t(A.to(self._device), W.to(self._device)).reshape(-1, 1)
        return self.y_transformer.transform(h_raw.to(A.device))

    def pred_structural_function(self, do_A, treatment_transformer=None, outcome_transformer=None):
        a = do_A.to(self._device)
        if treatment_transformer is not None:
            a = treatment_transformer.to(self._device).transform(a)
        with torch.no_grad():
            return self.model.predict_t(a).reshape(-1, 1)      # trained on raw Y -> raw units


def _sn_first_stage_head(nets, device):
    """The class sizes its first-stage head from the featurizer's LAST nn.Linear (32), but the SN
    builder's tensor-product first stage outputs 33*33=1089 features -> pass the head explicitly,
    mirroring the class's own init convention (zeros, last column = 1/in_dim)."""
    d = nets[0].output_dim
    w = torch.zeros(nets[2].output_dim, d + 1, device=device)
    w[:, -1] = 1.0 / d
    return w


# ------------------------------------------------------------------ one full experiment
def run_once(data_seed, train_seed, P):
    """Cross-fitted causal MSE for the four estimators, per arm, on one DGP draw.

    The DGP draws from its OWN default_rng(data_seed) and never touches the global numpy stream, so
    np.random.seed(train_seed) AFTER it keys the split/init noise — and (data_seed=42,
    train_seed=42) replays the notebook's exact permutation.
    """
    t0 = time.time()
    batch = P.batch if P.batch > 0 else int(1024 * P.n_sample / 5000)
    stage2_batch = P.stage2_batch if P.stage2_batch > 0 else batch
    third_stage_kwargs = dict(
        lr=P.third_lr, weight_decay=1e-6, n_epochs=P.third_epochs, loss_fn=_third_loss_fn(P),
        gap_penalty_weight=0.0, log_per_epoch=10 ** 9, input_dim=4096, output_dim=1,
        hidden_dims=[int(h) for h in P.third_hidden.split(",")], dropout_rate=P.third_dropout)
    treatment_featurizer = torch.nn.Identity()

    # ---- data ----
    gen = dSprite_ProxyVariable_DatasetV2(variant=P.dgp)
    A, Y, Z, W, do_A, EY_do_A = gen.generate_dsprite_pv(
        os.path.join(REPO, "data", "dsprite"), n_sample=P.n_sample, generate_test=True,
        rand_seed=data_seed)
    causal_mse = lambda f: float(np.mean((np.asarray(f).reshape(-1, 1) - EY_do_A.reshape(-1, 1)) ** 2))
    tt = lambda x: torch.tensor(x, dtype=torch.float32)
    tensors = (tt(A), tt(Y), tt(Z), tt(W))
    do_A_t = tt(do_A)
    # DIAGNOSTIC AXIS ONLY (the DGP's own dose-response summary; nothing fitted may touch it)
    s_all = gen.structural_func(np.asarray(A, dtype=np.float64), gen.cal_weight())

    np.random.seed(train_seed)
    _sc = lambda w: TorchStandardScaler() if w == "std" else TorchIdentityTransformer()
    transformers = [TorchIdentityTransformer(), _sc(P.y_scale), _sc(P.z_scale),
                    TorchIdentityTransformer()]                       # A, Y, Z, W (images stay raw)
    drt = TorchIdentityTransformer(); dr = torch.ones(P.n_sample, 1, dtype=torch.float32)
    idx = np.random.permutation(P.n_sample)
    n_val = int(P.n_sample * P.val_perc)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    for i, ten in enumerate(tensors):
        transformers[i].fit(ten[train_idx])
    drt.fit(dr[train_idx])
    if P.om_both:   # OutcomeNet (orig): shipped trainer on STANDARDISED Y/Z; its estimation rows keep the TM's Z scaling
        tf_old = [TorchIdentityTransformer(), TorchStandardScaler(), TorchStandardScaler(), TorchIdentityTransformer()]
        tf_mix = [TorchIdentityTransformer(), TorchStandardScaler(), _sc(P.z_scale), TorchIdentityTransformer()]
        for i, ten in enumerate(tensors):
            tf_old[i].fit(ten[train_idx]); tf_mix[i].fit(ten[train_idx])
        dev_tf_old = [copy.deepcopy(tr).to(DEVICE) for tr in tf_old]
        P_old = copy.copy(P)
        for k, v in P.om_defaults.items():
            setattr(P_old, k, v)
    folds = np.array_split(train_idx, P.n_folds)

    def stage_indices(k):
        est = folds[k]
        other = [folds[(k + o) % P.n_folds] for o in range(1, P.n_folds)]
        half = max(len(other) // 2, 1)
        return np.concatenate(other[:half]), np.concatenate(other[half:]), est

    device_transformers = [copy.deepcopy(t).to(DEVICE) for t in transformers]

    def make_ds(ix, tf=None):
        return ProxyDataset(slice_tuple(tensors, ix), dens_ratio=dr[ix],
                            dens_ratio_transformer=copy.deepcopy(drt),
                            transformers=copy.deepcopy(tf if tf is not None else transformers), device="cpu")

    val_loader = DataLoader(make_ds(val_idx), batch_size=batch, shuffle=False)
    if P.om_both:
        val_old = DataLoader(make_ds(val_idx, tf_old), batch_size=batch, shuffle=False)   # OM (orig) training monitor
        val_mix = DataLoader(make_ds(val_idx, tf_mix), batch_size=batch, shuffle=False)   # OM (orig) third stages

    # ---- official DFPV outcome bridge (their code, unmodified; raw arrays) ----
    def fit_outcome_dfpv(train_rows):
        sys.path.insert(0, DFPV_ROOT)
        from src.models.DFPV.trainer import DFPVTrainer          # noqa: E402  (their `src` package)
        from src.data.ate.data_class import PVTrainDataSet       # noqa: E402
        tr = PVTrainDataSet(treatment=np.asarray(A, dtype=np.float32)[train_rows],
                            treatment_proxy=np.asarray(Z, dtype=np.float32)[train_rows],
                            outcome_proxy=np.asarray(W, dtype=np.float32)[train_rows],
                            outcome=np.asarray(Y, dtype=np.float32).reshape(-1, 1)[train_rows],
                            backdoor=None)
        cfg = {"lam1": P.dfpv_lam1, "lam2": P.dfpv_lam2, "stage1_iter": P.dfpv_stage1_iter,
               "stage2_iter": 1, "n_epoch": P.dfpv_n_epoch, "split_ratio": P.dfpv_split,
               "treatment_weight_decay": P.dfpv_wd, "treatment_proxy_weight_decay": P.dfpv_wd,
               "outcome_proxy_weight_decay": P.dfpv_wd, "backdoor_weight_decay": 0.0}
        np.random.seed(train_seed)            # their sklearn split is unseeded: seed it for pairing
        trainer = DFPVTrainer({"name": "dsprite"}, cfg, gpu_flg=(DEVICE == "cuda"), dump_folder=None)
        for opt in (trainer.treatment_1st_opt, trainer.treatment_2nd_opt,
                    trainer.treatment_proxy_opt, trainer.outcome_proxy_opt):
            for g in opt.param_groups:
                g["lr"] = P.dfpv_lr
        model = trainer.train(tr, verbose=0)
        dev = next(model.treatment_2nd_net.parameters()).device
        return _DFPVAsOutcomeBridge(model, device_transformers[1], dev)

    # ---- bridge fitters ----
    def fit_outcome(s1_loader, s2_loader, P=P, val_loader=val_loader):
        cfg_kw = dict(
            lr_first_stage_base=P.om_lr, lr_second_stage_ax=P.om_lr, lr_second_stage_w=P.om_lr,
            weight_decay=P.om_weight_decay, scheduler_gamma=0.99,
            reg_second_stage=(P.om_reg2_lo, P.om_reg2_hi), device=DEVICE)
        if P.om_reg1_lo is not None:
            cfg_kw["reg_first_stage"] = (P.om_reg1_lo, P.om_reg1_hi)
        if P.om_reg21_lo is not None:
            cfg_kw["reg_second_stage_first"] = (P.om_reg21_lo, P.om_reg21_hi)
        cfg = OutcomePCLdSpriteConfig(**cfg_kw)
        if P.om_arch == "sn":
            nets = build_nets_for_outcome_pcl_net_dsprite_experiment_sn(device=DEVICE, feat_dim=P.om_sn_dim, z_dim=3)
        elif P.om_arch == "configurable":
            nets = build_nets_for_outcome_pcl_net_dsprite_experiment_configurable(
                device=DEVICE, z_dim=3, width_mult=P.om_width_mult,
                residual=bool(P.om_residual), n_blocks=P.om_n_blocks)
        else:
            nets = build_nets_for_outcome_pcl_net_dsprite_experiment(device=DEVICE)
        use_sgd = P.om_loss != "mse"
        om_cls = (om_v2.OutcomeBridgePCLNET if P.om_trainer == "v2"
                  else om_sgd.OutcomeBridgePCLNET if use_sgd else OutcomeBridgePCLNET)
        om_kw = {"final_layer_first_stage_weight": _sn_first_stage_head(nets, DEVICE)} if P.om_arch == "sn" else {}
        if P.om_trainer == "v2":
            om_kw["predict_bn_train_mode"] = bool(P.om_v2_bn_train_pred)
        m = om_cls(nets[0], nets[1], nets[2], device=DEVICE, **om_kw)
        o1, o2, s1, s2 = build_adam_optimizers_for_outcome_pcl_net(
            m, cfg.lr_first_stage, cfg.lr_second_stage_ax, cfg.lr_second_stage_w,
            cfg.weight_decay, gamma=cfg.scheduler_gamma)
        r1, r2i, r2o = cfg.regularizers
        common = dict(
            pcl_model=m, first_stage_train_dataloader=s1_loader,
            second_stage_train_dataloader=s2_loader,
            stage1_optimizers=o1, stage2_optimizers=o2, stage1_schedulers=s1, stage2_schedulers=s2,
            n_epochs=P.om_epochs, stage1_iter=(P.om_stage1_iter if P.om_stage1_iter is not None else cfg.stage1_iter), stage2_iter=cfg.stage2_iter,
            first_stage_final_layer_regularizer=r1, second_stage_final_layer_regularizer=r2o,
            second_stage_first_final_layer_regularizer=r2i,
            regularizer_annealing_method=cfg.reg_annealing_method,
            consider_prev_weight=(bool(P.om_consider_prev_weight) if P.om_consider_prev_weight is not None else cfg.consider_prev_weight), log_per_epoch=10 ** 9, plot_loss=False,
            validation_dataloader=val_loader)
        if P.om_trainer == "v2":
            o1, o2 = om_v2.build_coupled_adam_optimizers(m, lr=P.om_lr, weight_decay=P.om_weight_decay)  # DFPV: Adam + coupled L2
            m = om_v2.train_deep_feature_proxy_profiled_ate_model(
                pcl_model=m, first_stage_train_dataloader=s1_loader, second_stage_train_dataloader=s2_loader,
                stage1_optimizers=o1, stage2_optimizers=o2, stage1_schedulers=None, stage2_schedulers=None,
                first_stage_final_layer_regularizer=(P.om_reg1_lo if P.om_reg1_lo is not None else 0.1),
                second_stage_first_final_layer_regularizer=(P.om_reg21_lo if P.om_reg21_lo is not None else 0.1),
                second_stage_final_layer_regularizer=P.om_reg2_lo,
                n_epochs=P.om_epochs, stage1_iter=(P.om_stage1_iter if P.om_stage1_iter is not None else 20),
                stage2_iter=1, validation_dataloader=val_loader)
        elif use_sgd:
            # the _w_SGD trainer (superset signature): closed-form ridge head + 5 SGD refinement
            # steps @ lr 1e-2 under the robust loss -- the old dSprite scripts' exact settings.
            loss_kw = {"delta": P.om_huber_delta} if P.om_loss == "huber" else {}
            m = om_sgd.train_deep_feature_proxy_closed_form_ate_model(
                first_stage_loss_fn=nn.MSELoss(),
                second_stage_loss_fn=get_loss_function(P.om_loss, loss_kw),
                second_stage_head_lr=1e-2, second_stage_head_steps=5, **common)
        else:
            m = train_deep_feature_proxy_closed_form_ate_model(**common)
        m.mean_outcome_proxy_feature = None      # prevent stage-2-fold leakage into f_OM
        return m

    def make_projection(arm, s2_ds):
        """The scalar summary the KKT constraint's basis acts on — a function of the OBSERVED A only."""
        if arm == "kkt-pca":
            A2 = s2_ds.A.to(DEVICE)
            centre = A2.mean(0, keepdim=True)
            v = torch.pca_lowrank(A2 - centre, q=1, center=False)[2][:, 0].detach()
            return (lambda A_t, _c=centre, _v=v: (A_t - _c) @ _v), P.pca_degree
        # kkt-mean: degree 0 — the basis is the constant function, projection values are irrelevant
        return (lambda A_t: A_t.mean(dim=1)), 0

    def fit_treatment(arm, s1_loader, s2_loader, s2_ds):
        constraint = None
        if arm.startswith("kkt"):
            proj, deg = make_projection(arm, s2_ds)
            constraint = (lambda _p=proj, _d=deg: ConditionalMeanConstraint(
                basis="poly", degree=_d, mode="kkt", orthonormalize=True, eval_mode=True,
                projection=_p))
        tm_kw = {}
        if P.tm_arch == "dfpv":
            tm_kw.update(lr_stage1=1e-3, lr_stage2=1e-3, weight_decay=0.1, stage1_iter=20)
        for k, v in (("lr_stage1", P.tm_lr1), ("lr_stage2", P.tm_lr2),
                     ("weight_decay", P.tm_wd), ("stage1_iter", P.tm_stage1_iter)):
            if v is not None:
                tm_kw[k] = v
        tm_builder = (build_nets_for_alternative_treatment_pcl_net_dsprite_experiment_dfpv
                      if P.tm_arch == "dfpv" else build_nets_for_alternative_treatment_pcl_net_dsprite_experiment)
        if P.tm_trainer == "v2":
            ens = tm_v2.train_alternative_treatment_pcl_net_ensemble_v2(
                build_nets_fn=tm_builder,
                first_stage_train_dataloader=s1_loader, second_stage_train_dataloader=s2_loader,
                validation_dataloader=val_loader, n_members=P.n_bridges, device=DEVICE,
                base_seed=P.bridge_base_seed,
                lr_stage1=tm_kw.get("lr_stage1", 1e-3), lr_stage2=tm_kw.get("lr_stage2", 1e-3),
                weight_decay=tm_kw.get("weight_decay", 0.1), n_epochs=P.bridge_epochs,
                stage1_iter=tm_kw.get("stage1_iter", 20), stage2_iter=P.tm_stage2_iter,
                first_stage_final_layer_regularizer=P.tm_reg1,
                second_stage_first_final_layer_regularizer=P.tm_reg21,
                second_stage_final_layer_regularizer=P.tm_reg2,
                normalization_constraint=constraint, cross_terms=bool(P.tm_cross_terms),
                predict_bn_train_mode=bool(P.tm_v2_bn_train_pred))
        else:
          ens = train_alternative_treatment_pcl_net_ensemble(
            build_nets_fn=tm_builder,
            first_stage_train_dataloader=s1_loader, second_stage_train_dataloader=s2_loader,
            validation_dataloader=val_loader, n_members=P.n_bridges, device=DEVICE,
            base_seed=P.bridge_base_seed, second_stage_final_layer_regularizer=P.tm_reg2,
            n_epochs=P.bridge_epochs, head_mode="closed_form",
            second_stage_head_lr=1.0, second_stage_head_steps=10,
            normalization_constraint=constraint, **tm_kw)
        if arm == "global":       # failed scalar posthoc control — kept only for mechanism runs
            with torch.no_grad():
                m_hat = float(ens.predict_treatment_bridge_function(
                    s2_ds.A.to(DEVICE), s2_ds.Z.to(DEVICE), None).mean())
            orig = ens.predict_treatment_bridge_function
            ens.predict_treatment_bridge_function = (
                lambda a, z, x=None, _o=orig, _m=m_hat: _o(a, z, x) / (_m if abs(_m) > 1e-6 else 1e-6))
        if arm == "phi1":         # attribution control: bridge trained (RNG pairing), then ignored
            ones_fn = lambda a, z=None, x=None: torch.ones(a.shape[0], 1, device=a.device)
            ens.predict_treatment_bridge_function = ones_fn
            ens.predict_dens_ratio = ones_fn
        return ens

    # ---- estimators on the held-out fold ----
    def fit_third(ds, ds_val, dr=False):
        kw = dict(third_stage_kwargs)
        if dr and P.dr_third_lr > 0:
            kw["lr"] = P.dr_third_lr
        if dr and P.dr_third_epochs > 0:
            kw["n_epochs"] = P.dr_third_epochs
        return train_third_stage_treatment_pcl_net_ensemble(
            model_class=ConditionalMeanMLP,
            dataloader=DataLoader(ds, batch_size=batch, shuffle=True),
            val_dataloader=DataLoader(ds_val, batch_size=batch, shuffle=True),
            n_members=P.third_members, device=DEVICE, **kw)

    def eval_third(net, ds):
        a_grid = device_transformers[0].transform(do_A_t.to(DEVICE))
        x = ds.input_transformer.to(DEVICE).transform(treatment_featurizer(a_grid))
        return ds.outcome_transformer.to(DEVICE).inverse_transform(net(x)).detach().cpu().numpy()

    def estimate(om, tm, est_loader, val_loader=val_loader, dev_tf=None):
        om.compute_mean_outcome_proxy_feature(est_loader)   # estimation-fold mean feature only
        f_om = om.pred_structural_function(
            do_A_t, device_transformers[0], (dev_tf or device_transformers)[1]).detach().cpu().numpy()
        tds, tds_val = create_third_stage_dataset_for_treatment_pcl_net_ate(
            tm, est_loader, val_loader, TorchStandardScaler(), TorchStandardScaler(),
            treatment_featurizer=treatment_featurizer, device=DEVICE)
        f_tm = eval_third(fit_third(tds, tds_val), tds)

        def slack(creator):
            ds, ds_val = creator(om, tm, est_loader, val_loader,
                                 outcome_transformer=TorchStandardScaler(),
                                 input_transformer=TorchStandardScaler(), input_type="features",
                                 treatment_featurizer=treatment_featurizer, device=DEVICE)
            return eval_third(fit_third(ds, ds_val, dr=True), ds)

        s1 = slack(create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_ate)
        s2 = slack(create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_ate_v2)
        return {"OutcomeNet": f_om, "TreatmentNet": f_tm,
                "DRPCLNET-V1": f_om + s1, "DRPCLNET-V2": f_om + f_tm - s2}

    def identity_diag(tm, est_ds, est_idx, n_bins=10):
        """Held-out E[phi | s(A)] = 1 check, binned on the DGP summary — same yardstick for every arm."""
        with torch.no_grad():
            phi = tm.predict_treatment_bridge_function(
                est_ds.A.to(DEVICE), est_ds.Z.to(DEVICE), None).reshape(-1).cpu().numpy()
        s = s_all[est_idx]
        edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
        b = np.clip(np.digitize(s, edges[1:-1]), 0, n_bins - 1)
        m = np.array([phi[b == k].mean() for k in range(n_bins)])
        return float(phi.mean()), float(phi.std()), float(np.abs(m - 1).max())

    # ---- cross-fitting loop: OM shared per fold, RNG replayed before each arm ----
    if P.protocol == "overlap":
        # an earlier internal version single overlapping split; estimation == stage-2 (in-sample DR residual)
        L = len(train_idx); c = int(round(P.overlap_frac * L))
        s1i = train_idx[:c]; s2i = train_idx[L - c:]
        fold_assign = [(s1i, s2i, s2i)]
        shared = len(np.intersect1d(s1i, s2i))
        print("  [overlap] stage1=%d stage2=%d shared=%d (%.0f%%) est=stage2"
              % (len(s1i), len(s2i), shared, 100.0 * shared / max(len(s2i), 1)), flush=True)
    else:
        fold_assign = [stage_indices(k) for k in range(P.n_folds)]
    acc = {a: {"folds": [], "diag": []} for a in P.arms}
    for (s1i, s2i, ei) in fold_assign:
        s1d, s2d, ed = make_ds(s1i), make_ds(s2i), make_ds(ei)
        s1l = DataLoader(s1d, batch_size=batch, shuffle=True)
        s2l = DataLoader(s2d, batch_size=stage2_batch, shuffle=True)
        el = DataLoader(ed, batch_size=batch, shuffle=True)
        tm_s1l, tm_s2l = s1l, s2l
        if P.tm_batch > 0:                            # TM-only full/large batch (OM + third-stage loaders untouched)
            tm_s1l = DataLoader(s1d, batch_size=P.tm_batch, shuffle=True)
            tm_s2l = DataLoader(s2d, batch_size=P.tm_batch, shuffle=True)
        torch.manual_seed(train_seed)                 # notebook: same init in every fold
        if P.om_batch > 0:                            # OM-only full/large batch (TM loaders untouched)
            om_s1l = DataLoader(s1d, batch_size=P.om_batch, shuffle=True)
            om_s2l = DataLoader(s2d, batch_size=P.om_batch, shuffle=True)
        else:
            om_s1l, om_s2l = s1l, s2l
        om = (fit_outcome_dfpv(np.concatenate([s1i, s2i])) if P.om_arch == "dfpv"
              else fit_outcome(om_s1l, om_s2l))            # shared by all arms -> exact pairing
        rng_cpu = torch.get_rng_state()
        rng_gpu = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        om_old = None
        if P.om_both:                                 # OutcomeNet (orig): its own seed, shipped loaders, std-scaled data
            el_mix = DataLoader(make_ds(ei, tf_mix), batch_size=batch, shuffle=True)
            torch.manual_seed(train_seed)
            om_old = fit_outcome(DataLoader(make_ds(s1i, tf_old), batch_size=batch, shuffle=True),
                                 DataLoader(make_ds(s2i, tf_old), batch_size=stage2_batch, shuffle=True),
                                 P=P_old, val_loader=val_old)
        for arm in P.arms:
            torch.set_rng_state(rng_cpu)
            if rng_gpu is not None:
                torch.cuda.set_rng_state_all(rng_gpu)
            tm = fit_treatment(arm, tm_s1l, tm_s2l, s2d)
            acc[arm]["diag"].append(identity_diag(tm, ed, ei))
            rng2_cpu = torch.get_rng_state()
            rng2_gpu = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            fam = estimate(om, tm, el)
            if P.om_both:                             # same post-TM RNG state -> same third-stage draws as a single-OM run
                torch.set_rng_state(rng2_cpu)
                if rng2_gpu is not None:
                    torch.cuda.set_rng_state_all(rng2_gpu)
                fam.update({k + ORIG_SUFFIX: v for k, v in
                            estimate(om_old, tm, el_mix, val_loader=val_mix, dev_tf=dev_tf_old).items() if k in OM_ROWS})
            acc[arm]["folds"].append(fam)

    results = {}
    for arm in P.arms:
        names = list(acc[arm]["folds"][0].keys())
        cf = {n: np.mean([f[n] for f in acc[arm]["folds"]], axis=0) for n in names}
        dg = np.mean(np.array(acc[arm]["diag"]), axis=0)
        results[arm] = dict(
            mse={n: causal_mse(cf[n]) for n in names},
            mean_fold_mse={n: float(np.mean([causal_mse(f[n]) for f in acc[arm]["folds"]]))
                           for n in names},
            phi_mean=float(dg[0]), phi_std=float(dg[1]), moment_max=float(dg[2]))
    return results, time.time() - t0


def main():
    P = parse_args()
    out_dir = P.out_dir or os.path.join(HERE, "..", "Results", "dSprite_CrossFitting" + ("_org" if P.dgp == "org" else ""))
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    seed_part = "n%d_s%s" % (P.n_sample, "-".join(str(s) for s in P.data_seeds))
    out = os.path.join(out_dir, "DRNMEPCL_DensRatioFree_dSpriteCF_%s_%s_%s.csv"
                       % (P.tag, seed_part, stamp))
    fields = ["Algorithm", "Arm", "normalization", "Data_Seed", "Repeat", "Train_Seed",
              "Causal_MSE", "MeanFold_MSE", "phi_mean", "phi_std", "moment_max",
              "hostname"] + HYPER_COLUMNS
    # out_dir with the same header, APPEND to the newest one and skip every (data_seed, repeat) that already holds
    # rows for ALL requested arms -- no duplicate files, no redone repeats. A fresh task never matches: unchanged.
    done = set()
    prev = sorted(glob.glob(os.path.join(out_dir, "DRNMEPCL_DensRatioFree_dSpriteCF_%s_%s_*.csv" % (P.tag, seed_part))))
    if prev:
        with open(prev[-1], newline="") as f:
            rd = csv.DictReader(f); rows = list(rd) if rd.fieldnames == fields else None
        if rows is not None:
            arms_seen = {}
            for r in rows:
                arms_seen.setdefault((int(r["Data_Seed"]), int(r["Repeat"])), set()).add(r["normalization"])
            done = {k for k, v in arms_seen.items() if set(P.arms) <= v}
            out = prev[-1]
            print("RESUME: appending to %s; finished (data_seed, repeat) cells skipped: %s"
                  % (os.path.basename(out), sorted(done)), flush=True)
    if not done and not (prev and rows is not None):
        with open(out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
    meta = {k: getattr(P, k) for k in HYPER_COLUMNS}
    host = socket.gethostname()

    for data_seed in P.data_seeds:
        for repeat in range(P.repeat_offset, P.repeat_offset + P.n_repeats):
            train_seed = 1000 * data_seed + repeat
            if (data_seed, repeat) in done:
                print("skip data_seed %d rep %d (already in the CSV)" % (data_seed, repeat), flush=True)
                continue
            results, secs = run_once(data_seed, train_seed, P)
            with open(out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                for arm, r in results.items():
                    for algo in r["mse"]:
                        w.writerow(dict(Algorithm=algo, Arm="cross_fitted", normalization=arm,
                                        Data_Seed=data_seed, Repeat=repeat, Train_Seed=train_seed,
                                        Causal_MSE=r["mse"][algo],
                                        MeanFold_MSE=r["mean_fold_mse"][algo],
                                        phi_mean=r["phi_mean"], phi_std=r["phi_std"],
                                        moment_max=r["moment_max"], hostname=host, **meta))
            summary = "  ".join(
                "%s[%s]" % (arm, " ".join("%s %.2f" % (a, results[arm]["mse"][a])
                                          for a in results[arm]["mse"])) for arm in results)
            print("data_seed %d rep %d (%.0fs): %s" % (data_seed, repeat, secs, summary), flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
