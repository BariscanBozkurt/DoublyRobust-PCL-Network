"""Multi-seed, multi-hyperparameter simulation: density-ratio-FREE DR-PCL with DISJOINT-fold
cross-fitting on the Demand benchmark -- data size as a first-class hyperparameter, and ALL THREE
normalization arms (none / posthoc / kkt) run PAIRED inside every (data_seed, repeat).

NORMALIZATION ARMS. Default --normalization all = [none, posthoc, kkt]:
  none    = the raw density-ratio-free bridge, NO repair of E[varphi|A]=1 -- the CONTROL that
            measures what the repairs actually buy on this benchmark;
  posthoc = divide out a degree-`norm_degree` polynomial fit of E[varphi|A] taken on THIS loop's
            stage-2 fold (clipped to the fit range, floored), wrapping the bridge predictor so the
            SAME normalized varphi feeds the third stage and both DR slacks;
  kkt     = ConditionalMeanConstraint(poly, degree=norm_degree) inside the closed-form head, so the
            moments E[(varphi-1)A^k]=0 hold exactly at every head update, during training.
The arms are paired by RNG-STATE SNAPSHOTS (see _fit_and_estimate_all_arms): one shared outcome
bridge, every bridge fit replaying the post-outcome state and every estimation replaying the
post-bridge state. The none and posthoc arms SHARE one trained bridge -- post-hoc is a predictor
wrapper, so it is applied after the raw arm has been read -- meaning a three-arm run trains only TWO
bridges per fold rather than three.

PROTOCOL. The demand DGP is synthetic, so data_seed draws the DATASET (the DGP's own RNG) and
train_seed = 1000 * data_seed + repeat drives the val/fold permutation and every network init.
Per cross-fitting loop the three roles use three MUTUALLY DISJOINT folds (asserted at runtime),
rotating so every fold serves every role once. val_perc is held out before the folds are cut and is
used only for validation monitoring -- never estimation.

DEMAND-SPECIFIC WRINKLES:
  * the third stage regresses on the fold's OWN outcome bridge's 16-D A-featurizer, not the raw
    treatment (input_dim is inferred at runtime, so featurizer-width sweeps stay consistent);
  * the third stage trains under MSE for ALL consumers (abortion uses Huber for f_tm);
  * batch_size defaults to n_sample so every arm gets ONE optimizer step per epoch at every size --
    reg_second_stage is an inverse step size, so a size sweep with a fixed small batch would change
    the effective regularisation alongside the data size and confound the comparison.

Every CLI knob is written into every CSV row; each invocation writes its OWN CSV (tag + size + seed
range + timestamp in the name), so parallel tasks never contend for a file.

Examples:
  python Demand_CrossFitting_Simulation.py --n_sample 10000 --data_seeds 0 1 2 --n_repeats 3
  python Demand_CrossFitting_Simulation.py --n_sample 20000 --data_seeds 4 --normalization kkt
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

from simulation_utils.data_generating_processes import generate_demand_experiment_data
from torch_utils.scalers import TorchStandardScaler, TorchIdentityTransformer
from torch_utils.dataset_classes import ProxyDataset
from torch_utils.networks import ConditionalMeanMLP
from torch_utils.torch_loss import get_loss_function
from python_utils.helpers import slice_tuple

from simulation_utils.outcome_pcl_net_configs import OutcomePCLDemandConfig
from simulation_utils.outcome_pcl_net_nn_structures import (
    build_nets_for_outcome_pcl_net_demand_experiment)
from simulation_utils.outcome_pcl_net_optimizers import build_adam_optimizers_for_outcome_pcl_net
from neural_causal_learning.proxy_outcome_neural_mean_embedding import (
    OutcomeBridgePCLNET, train_deep_feature_proxy_closed_form_ate_model)
from neural_causal_learning import proxy_outcome_neural_mean_embedding_V2 as om_v2   # opt-in profiled trainer (--om_trainer v2)

from simulation_utils.alternative_treatment_pcl_net_configs import AlternativeTreatmentPCLConfig
from simulation_utils.treatment_pcl_net_nn_structures import (
    build_nets_for_alternative_treatment_pcl_net_demand_experiment)
from simulation_utils.alternative_treatment_pcl_net_optimizers import (
    build_adam_optimizers_for_alternative_treatment_pcl_net)
from neural_causal_learning.alternative_treatment_neural_mean_embedding import (
    AlternativeTreatmentBridgePCLNET, train_alternative_treatment_pcl_net_ate_model,
    train_alternative_treatment_pcl_net_ensemble,
    create_third_stage_dataset_for_treatment_pcl_net_ate)
from neural_causal_learning.normalization_constraint import ConditionalMeanConstraint
from neural_causal_learning.proxy_treatment_neural_mean_embedding import (
    train_third_stage_treatment_pcl_net_ensemble)
from neural_causal_learning.alternative_doubly_robust_proxy_neural_mean_embedding import (
    create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_ate as DR1,
    create_third_stage_dataset_for_dr_proxy_neural_mean_embedding_ate_v2 as DR2)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ESTIMATORS = ["OutcomeNet", "TreatmentNet", "DRPCLNET-V1", "DRPCLNET-V2"]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_seeds", type=int, nargs="+", default=list(range(10)),
                   help="DGP seeds: each draws a fresh dataset of size n_sample")
    p.add_argument("--n_repeats", type=int, default=3)
    # -- data --
    p.add_argument("--n_sample", type=int, default=10000,
                   help="dataset size; the size-scaling axis of this simulation")
    p.add_argument("--epsilon_std", type=float, default=1.0)
    p.add_argument("--z_noise", type=float, default=0.0,
                   help="per-column Gaussian noise on the treatment proxy Z, in units of its own std")
    p.add_argument("--w_noise", type=float, default=0.0,
                   help="per-column Gaussian noise on the outcome proxy W, in units of its own std")
    # -- protocol --
    p.add_argument("--val_perc", type=float, default=0.1)
    p.add_argument("--n_folds", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=None,
                   help="None = n_sample (ONE step per epoch at every size; see module docstring)")
    # -- outcome bridge --
    p.add_argument("--om_epochs", type=int, default=None, help="None = preset")
    p.add_argument("--om_reg2_lo", type=float, default=1e-1,
                   help="outcome head ridge annealing START (the demand config anneals)")
    p.add_argument("--om_reg2_hi", type=float, default=10.0, help="annealing END")
    # OUTCOME-BRIDGE TRAINER / ARCHITECTURE (opt-in; every default = the shipped numerically identical path).
    # v2 = the profiled ("deep feature", DFPV-style) trainer accepted on dSprite: ridge heads solved INSIDE the losses,
    # coupled-L2 Adam, intercept cross terms (proxy_outcome_neural_mean_embedding_V2.py); its ridge is --om_reg2_lo (fixed,
    # no annealing). dfpv = the OFFICIAL DFPV outcome bridge (Xu et al. 2021, the DFPV authors' code, unmodified) trained on the raw
    # stage-1 U stage-2 rows with its own 50/50 split, as a REFERENCE arm (defaults = their published demand json).
    p.add_argument("--om_trainer", choices=["closed_form", "v2"], default="closed_form")
    p.add_argument("--om_lr", type=float, default=None, help="None = preset (1e-4 closed_form; 1e-3 v2)")
    p.add_argument("--om_weight_decay", type=float, default=None, help="None = preset (closed_form) / 0.1 coupled L2 (v2)")
    p.add_argument("--om_stage1_iter", type=int, default=None, help="None = preset (closed_form) / 20 (v2)")
    p.add_argument("--om_reg1", type=float, default=None, help="v2 only: first-stage ridge (None = 0.1)")
    p.add_argument("--om_reg21", type=float, default=None, help="v2 only: stage-2 first-stage ridge (None = 0.1)")
    p.add_argument("--om_batch", type=int, default=0, help="OM-ONLY loader batch size (0 = the shared --batch_size)")
    p.add_argument("--om_v2_bn_train_pred", type=int, default=0)
    p.add_argument("--om_arch", choices=["default", "dfpv"], default="default")
    p.add_argument("--dfpv_lam1", type=float, default=0.1)
    p.add_argument("--dfpv_lam2", type=float, default=0.1)
    p.add_argument("--dfpv_n_epoch", type=int, default=100)
    p.add_argument("--dfpv_stage1_iter", type=int, default=20)
    p.add_argument("--dfpv_wd", type=float, default=0.1)
    p.add_argument("--dfpv_lr", type=float, default=1e-3)
    p.add_argument("--dfpv_split", type=float, default=0.5)
    p.add_argument("--dfpv_feat", choices=["own", "ours"], default="own",
                   help="dfpv arm only: third-stage input featurizer = DFPV's own stage-2 treatment net ('own') or the treatment "
                        "featurizer of our shipped OutcomeNet fitted alongside ('ours', isolates the bridge quality from the featurizer)")
    # -- treatment bridge --
    p.add_argument("--tm_epochs", type=int, default=None, help="None = preset (150)")
    # TREATMENT-BRIDGE ENSEMBLE. This sim fitted a SINGLE bridge (K=1). The abortion CF sim had the
    # same gap and its round-2 tuning found ensembling was HALF the total DR gain there; the original
    # demand result reported earlier was itself run with a K=5 bridge ensemble, so K=1 here is a
    # regression rather than a considered choice. Default stays 1 so an unflagged run is unchanged.
    p.add_argument("--tm_members", type=int, default=1,
                   help="treatment-BRIDGE ensemble size (not the third-stage head, which is "
                        "--third_members). 1 = single bridge (shipped). >1 averages members via "
                        "train_alternative_treatment_pcl_net_ensemble.")
    p.add_argument("--tm_reg2", type=float, default=30.0,
                   help="KAP head ridge; an INVERSE STEP SIZE (U-shape, interior optimum ~30)")
    # -- E[phi | A = a] = 1 normalization --
    p.add_argument("--normalization", choices=["all", "both", "posthoc", "kkt", "none"],
                   default="all",
                   help="all = [none, posthoc, kkt] run PAIRED inside every (data_seed, repeat); "
                        "both = [posthoc, kkt]; or a single arm. The none arm is the control that "
                        "measures what the repairs actually buy.")
    p.add_argument("--norm_degree", type=int, default=5)
    p.add_argument("--norm_floor", type=float, default=0.15)
    # -- third stage --
    p.add_argument("--third_lr", type=float, default=1e-4)
    p.add_argument("--third_epochs", type=int, default=100)
    p.add_argument("--third_hidden", type=str, default="32,64")
    p.add_argument("--third_dropout", type=float, default=0.05)
    p.add_argument("--third_members", type=int, default=5)
    # PER-ROUTE THIRD-STAGE UPDATE BUDGETS. The estimation fold is ~600 rows but the
    # third-stage loaders inherit batch = n_sample, so ONE epoch is ONE full-batch optimizer update
    # and the shipped --third_epochs 100 is 100 Adam updates at lr 1e-4 -- the heads sit near their
    # initialisation (measured mean parameter drift 0.059-0.063 at step 100, fitted prediction std
    # only 5.8% / 7.4% of the TM / DR pseudo-target std). Held-out seeds 110-112 over n=2k..20k give
    # TreatmentNet 154.569 -> 146.002 (-5.68%, 12/15) and DR-V1 22.625 -> 22.271 (-1.50%, 13/15) at
    # 200 updates; 500 is no better. The two routes need SEPARATE controls because TreatmentNet
    # destabilises under an aggressive budget (+121.6% at lr 1e-3 / 1000 updates on held-out seeds)
    # while DR-V1 keeps improving. 0 = inherit --third_epochs, so the legacy flag keeps working and
    # the default path is numerically identical.
    p.add_argument("--tm_third_epochs", type=int, default=0,
                   help="TreatmentNet third-stage budget in full-fold updates; 0 = inherit --third_epochs")
    p.add_argument("--dr_third_epochs", type=int, default=0,
                   help="budget for BOTH DR slack heads (V1 correction, V2 slack); 0 = inherit --third_epochs")
    p.add_argument("--third_rng_fence", type=int, choices=[0, 1], default=0,
                   help="1 = fit each of the three third-stage heads off its own deterministic RNG "
                        "stream (train_seed + a fixed per-route offset), so changing one route's "
                        "budget cannot shift the others' inits/shuffles. 0 = shipped shared stream.")
    # THIRD-STAGE LOSSES. Shipped behaviour is MSE for BOTH the TreatmentNet third stage and the two
    # DR slack heads (one shared head built them all). They are exposed SEPARATELY because on the
    # synthetic low-dim benchmark the two heads respond in OPPOSITE directions to a robust loss, so a
    # single flag would hide which one is responsible. Defaults = mse = numerically identical to the
    # pre-flag script (regression-checked). OutcomeNet never passes through head, so it is an
    # invariant pairing control.
    p.add_argument("--out_dir", type=str, default="",
                   help="results directory; '' = ../Results/Demand_CrossFitting. Point "
                        "hyperparameter searches elsewhere so their CSVs never enter the reported "
                        "glob (which is *sizesweep*.csv).")
    p.add_argument("--tm_third_loss", choices=["mse", "huber", "log_cosh"], default="mse",
                   help="loss for the TreatmentNet third stage (f_tm). Default mse = shipped.")
    p.add_argument("--tm_huber_delta", type=float, default=2.0)
    p.add_argument("--dr_third_loss", choices=["mse", "huber", "log_cosh"], default="mse",
                   help="loss for BOTH DR slack heads (V1 correction, V2 slack). Default mse = shipped.")
    p.add_argument("--dr_huber_delta", type=float, default=2.0)
    # -- bookkeeping --
    p.add_argument("--include_single_split", type=int, default=0,
                   help="1 = also run a DISJOINT single-split arm (thirds of the training rows)")
    p.add_argument("--tag", default="baseline")
    return p


def parse_args(argv=None):
    """Parse, then RESOLVE the per-route third-stage budgets so every CSV row records the EFFECTIVE
    value rather than the 0 = 'inherit --third_epochs' sentinel."""
    P = build_parser().parse_args(argv)
    if P.tm_third_epochs == 0:
        P.tm_third_epochs = P.third_epochs
    if P.dr_third_epochs == 0:
        P.dr_third_epochs = P.third_epochs
    return P


# ---------------------------------------------------------------- proxy informativeness (opt-in)
def apply_proxy_noise(W, Z, z_noise, w_noise, data_seed):
    """Per-column Gaussian measurement noise, scaled by each proxy's OWN std, to reduce a proxy's
    informativeness about the latent confounder. 0.0 = untouched (the default path).

    The noise is drawn from a DEDICATED RandomState keyed on the data seed, so it never advances the
    global numpy stream that the DGP, the fold split, the initialisation and the estimation consume:
    every noise level is then a within-seed controlled ablation whose ONLY difference from s=0 is the
    proxy values themselves. Convention (scale, RNG isolation) copied from the earlier abortion
    proxy-noise ablation so the numbers are comparable.
    """
    if z_noise <= 0.0 and w_noise <= 0.0:
        return W, Z
    rng = np.random.RandomState(int(data_seed) + 100003)
    W, Z = np.asarray(W, dtype=float).copy(), np.asarray(Z, dtype=float).copy()
    if w_noise > 0.0:
        W += w_noise * W.std(axis=0, keepdims=True) * rng.randn(*W.shape)
    if z_noise > 0.0:
        Z += z_noise * Z.std(axis=0, keepdims=True) * rng.randn(*Z.shape)
    return W, Z


HYPER_COLUMNS = ["n_sample", "epsilon_std", "z_noise", "w_noise", "val_perc", "n_folds", "batch_size",
                 "om_epochs", "om_reg2_lo", "om_reg2_hi", "tm_epochs", "tm_reg2",
                 "norm_degree", "norm_floor", "tm_members",
                 "third_lr", "third_epochs", "third_hidden", "third_dropout", "third_members",
                 "tm_third_epochs", "dr_third_epochs", "third_rng_fence",
                 "tm_third_loss", "tm_huber_delta", "dr_third_loss", "dr_huber_delta",
                 "om_trainer", "om_lr", "om_weight_decay", "om_stage1_iter", "om_reg1", "om_reg21", "om_batch",
                 "om_v2_bn_train_pred", "om_arch", "dfpv_lam1", "dfpv_lam2", "dfpv_n_epoch", "dfpv_stage1_iter",
                 "dfpv_wd", "dfpv_lr", "dfpv_split", "dfpv_feat", "tag"]
DFPV_ROOT = os.path.join(REPO, "..", "external", "DeepFeatureProxyVariable")


# ================================================================ data and folds
def build_data(data_seed, train_seed, P):
    np.random.seed(train_seed); torch.manual_seed(train_seed)
    U, W, Z, A, Y, do_A, EY = generate_demand_experiment_data(
        size=P.n_sample, epsilon_std=P.epsilon_std, do_A_range=(10, 30), do_A_size=10,
        seed=data_seed)
    W, Z = apply_proxy_noise(W, Z, P.z_noise, P.w_noise, data_seed)
    t = lambda v: torch.tensor(v, dtype=torch.float32)
    A_t, Y_t, Z_t, W_t = t(A), t(Y), t(Z), t(W)
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
        others = [folds[(k + o) % P.n_folds] for o in range(1, P.n_folds)]
        half = max(len(others) // 2, 1)
        return np.concatenate(others[:half]), np.concatenate(others[half:]), folds[k]

    assignment = [stage_indices(k) for k in range(P.n_folds)]
    for s1, s2, est in assignment:                       # the disjointness premise, enforced
        assert len(np.intersect1d(s1, s2)) == 0
        assert len(np.intersect1d(est, s1)) == 0 and len(np.intersect1d(est, s2)) == 0
        assert len(np.intersect1d(est, val_idx)) == 0
        assert len(np.union1d(np.union1d(s1, s2), est)) == len(train_idx)

    return dict(mk=mk, assignment=assignment, val=mk(val_idx), train_idx=train_idx,
                do_A=t(do_A), EY=EY, dev_tf=[copy.deepcopy(x).to(DEVICE) for x in transformers],
                raw=dict(A=A, Y=Y, Z=Z, W=W))


# ================================================================ bridges
class _DFPVAsOutcomeBridge(nn.Module):
    """Official DFPVModel (their code) behind the three things estimate() touches: .treatment_featurizer (their stage-2
    treatment net on RAW A -- our grids/batches are standardised, so the wrapper inverse-transforms first), .eval(), and
    predict_outcome_bridge_function(A_scaled, W_scaled) -> h in OUR scaled-Y space (estimate() inverse-transforms it)."""

    class _Feat(nn.Module):
        def __init__(self, net, a_tf):
            super().__init__(); self.net, self.a_tf = net, a_tf
        def forward(self, A_scaled):
            return self.net(self.a_tf.inverse_transform(A_scaled))

    def __init__(self, model, tf_dev):
        super().__init__()
        self.model, self.a_tf, self.y_tf, self.w_tf = model, tf_dev[0], tf_dev[1], tf_dev[3]
        self.treatment_featurizer = self._Feat(model.treatment_2nd_net, tf_dev[0])
        self.mean_outcome_proxy_feature = None

    def predict_outcome_bridge_function(self, A, W, X=None):
        with torch.no_grad():
            h_raw = self.model.predict_bridge_t(self.a_tf.inverse_transform(A), self.w_tf.inverse_transform(W)).reshape(-1, 1)
        return self.y_tf.transform(h_raw)


def fit_outcome_dfpv(d, rows, P, train_seed):
    """The official DFPV outcome bridge (unmodified), trained on the RAW stage-1 U stage-2 rows with its own internal split."""
    sys.path.insert(0, DFPV_ROOT)
    from src.models.DFPV.trainer import DFPVTrainer          # noqa: E402  (their `src` package)
    from src.data.ate.data_class import PVTrainDataSet       # noqa: E402
    r = d["raw"]
    tr = PVTrainDataSet(treatment=r["A"][rows].astype(np.float32), treatment_proxy=r["Z"][rows].astype(np.float32),
                        outcome_proxy=r["W"][rows].astype(np.float32), outcome=r["Y"][rows].reshape(-1, 1).astype(np.float32), backdoor=None)
    cfg = {"lam1": P.dfpv_lam1, "lam2": P.dfpv_lam2, "stage1_iter": P.dfpv_stage1_iter, "stage2_iter": 1, "n_epoch": P.dfpv_n_epoch,
           "split_ratio": P.dfpv_split, "treatment_weight_decay": P.dfpv_wd, "treatment_proxy_weight_decay": P.dfpv_wd,
           "outcome_proxy_weight_decay": P.dfpv_wd, "backdoor_weight_decay": 0.0}
    np.random.seed(train_seed)                                # their sklearn split is unseeded: seed it for pairing
    trainer = DFPVTrainer({"name": "demand"}, cfg, gpu_flg=(DEVICE.type == "cuda"), dump_folder=None)
    for opt in (trainer.treatment_1st_opt, trainer.treatment_2nd_opt, trainer.treatment_proxy_opt, trainer.outcome_proxy_opt):
        for g in opt.param_groups:
            g["lr"] = P.dfpv_lr
    model = trainer.train(tr, verbose=0)
    return _DFPVAsOutcomeBridge(model, d["dev_tf"]).to(DEVICE)


def fit_outcome(d, l1, l2, lv, P, rows=None, train_seed=None):
    if P.om_arch == "dfpv":
        m = fit_outcome_dfpv(d, rows, P, train_seed)
        if P.dfpv_feat == "ours":       # our shipped OutcomeNet, fitted after DFPV, lends its treatment featurizer to the third stage
            Q = copy.copy(P); Q.om_arch = "default"
            ours = fit_outcome(d, l1, l2, lv, Q, rows=rows, train_seed=train_seed)
            m.treatment_featurizer = ours.treatment_featurizer; m._ours = ours
        return m
    c = OutcomePCLDemandConfig(
        lr_first_stage_base=1e-4, lr_second_stage_ax=1e-4, lr_second_stage_w=1e-4,
        reg_first_stage=(5e-2, 5e-2), reg_second_stage_first=(5e-2, 5e-2),
        reg_second_stage=(P.om_reg2_lo, P.om_reg2_hi), device=DEVICE)
    if P.om_epochs is not None:
        c.n_epochs = P.om_epochs
    if P.om_lr is not None:
        c.lr_first_stage_base = c.lr_second_stage_ax = c.lr_second_stage_w = P.om_lr
    if P.om_weight_decay is not None:
        c.weight_decay = P.om_weight_decay
    if P.om_stage1_iter is not None:
        c.stage1_iter = P.om_stage1_iter
    c.log_per_epoch, c.plot_loss = 10 ** 9, False
    nets = build_nets_for_outcome_pcl_net_demand_experiment(c.device)
    if P.om_trainer == "v2":
        m = om_v2.OutcomeBridgePCLNET(nets[0], nets[1], nets[2], device=c.device, cross_terms=True,
                                      predict_bn_train_mode=bool(P.om_v2_bn_train_pred))
        o1, o2 = om_v2.build_coupled_adam_optimizers(m, lr=(P.om_lr if P.om_lr is not None else 1e-3),
                                                     weight_decay=(P.om_weight_decay if P.om_weight_decay is not None else 0.1))
        m = om_v2.train_deep_feature_proxy_profiled_ate_model(
            pcl_model=m, first_stage_train_dataloader=l1, second_stage_train_dataloader=l2,
            stage1_optimizers=o1, stage2_optimizers=o2, stage1_schedulers=None, stage2_schedulers=None,
            first_stage_final_layer_regularizer=(P.om_reg1 if P.om_reg1 is not None else 0.1),
            second_stage_first_final_layer_regularizer=(P.om_reg21 if P.om_reg21 is not None else 0.1),
            second_stage_final_layer_regularizer=P.om_reg2_lo,
            n_epochs=c.n_epochs, stage1_iter=(P.om_stage1_iter if P.om_stage1_iter is not None else 20),
            stage2_iter=1, validation_dataloader=lv, log_per_epoch=10 ** 9)
        m.mean_outcome_proxy_feature = None
        return m
    m = OutcomeBridgePCLNET(nets[0], nets[1], nets[2], device=c.device)
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


def fit_treatment(d, l1, l2, lv, P, norm_mode):
    c = AlternativeTreatmentPCLConfig.demand(); c.device = DEVICE
    c.reg_second_stage = P.tm_reg2
    if P.tm_epochs is not None:
        c.n_epochs = P.tm_epochs
    # The ensemble trainer builds each member itself, so the constraint must be a FACTORY (one fresh
    # constraint object per member); the single-bridge path takes an instance, as before.
    constraint_factory = (lambda: ConditionalMeanConstraint(basis="poly", degree=P.norm_degree,
                                                           mode="kkt")) if norm_mode == "kkt" else None
    build_fn = build_nets_for_alternative_treatment_pcl_net_demand_experiment
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
        return m
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
    # NOTE the posthoc wrapper is applied by the CALLER, so the none and posthoc arms can SHARE this
    # one trained bridge -- the wrapper is their entire difference.
    return m


def rng_snapshot():
    return (torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def rng_restore(state):
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


# ================================================================ estimation on ONE held-out fold
def estimate(d, om, tm, le, dse, lv, P, train_seed=None):
    # DEMAND WRINKLE: the fold's OWN outcome bridge supplies the third-stage input featurizer.
    feat = om.treatment_featurizer
    om.eval()
    W_e = dse.W.to(DEVICE)
    grid = d["dev_tf"][0].transform(d["do_A"].to(DEVICE))
    with torch.no_grad():
        curve = [om.predict_outcome_bridge_function(
            a.reshape(1, -1).expand(W_e.shape[0], -1), W_e).mean(0) for a in grid]
        in_dim = feat(grid).shape[-1]                    # 16 with the default builder; inferred
    f_om = d["dev_tf"][1].inverse_transform(torch.stack(curve)).detach().cpu().numpy()

    hidden = [int(x) for x in P.third_hidden.split(",")]

    def _loss(name, delta):
        if name == "huber":
            return nn.HuberLoss(delta=delta)
        if name == "log_cosh":
            return get_loss_function("log_cosh")
        return nn.MSELoss()

    # PER-ROUTE BUDGETS. 0 = inherit --third_epochs; main resolves this so the CSV records the
    # effective value, and the `or` here also guards callers that predate the flags entirely.
    tm_third_epochs = getattr(P, "tm_third_epochs", 0) or P.third_epochs
    dr_third_epochs = getattr(P, "dr_third_epochs", 0) or P.third_epochs
    # RNG FENCE. The three heads below are fitted SEQUENTIALLY off one shared global stream and each
    # reshuffles its loader every epoch, so changing one route's budget shifts how much RNG the LATER
    # heads consume and silently changes their inits and shuffles -- which would make a TM-vs-DR
    # attribution meaningless. With --third_rng_fence 1 each fit is snapshot -> deterministic re-seed
    # (train_seed + a fixed per-route offset) -> fit -> restore, so every head is reproducible
    # independently of the others' budgets and the stream OUTSIDE the fits is unchanged.
    fence_requested = bool(getattr(P, "third_rng_fence", 0))
    fence = fence_requested and train_seed is not None
    if fence_requested and train_seed is None:
        # An external harness that calls estimate positionally
        # gets no train_seed, so the fence cannot be applied -- say so loudly rather than writing
        # third_rng_fence=1 into a row that was in fact produced on the shared stream.
        print("WARNING: --third_rng_fence 1 ignored (estimate() called without train_seed); "
              "the three heads ran on the shared RNG stream.", flush=True)
    # OFFSETS SPACED BY 100000, NOT BY 1. train_seed = 1000*data_seed + repeat is consecutive in the
    # repeat index, so offsets {90001, 90002, 90003} would ALIAS -- (ds, rep 0, dr2), (ds, rep 1, dr1)
    # and (ds, rep 2, tm) would all draw head seed 90003 and share an initialisation. Spacing beyond
    # the maximum repeat count removes the collision. Only reached when --third_rng_fence 1.
    FENCE_OFFSET = {"tm": 90001, "dr1": 190001, "dr2": 290001}

    def head(ds, dsv, n_epochs, route, loss_fn=None):
        def _fit():
            return train_third_stage_treatment_pcl_net_ensemble(
                model_class=ConditionalMeanMLP,
                dataloader=DataLoader(ds, batch_size=P.batch, shuffle=True),
                val_dataloader=DataLoader(dsv, batch_size=P.batch, shuffle=True),
                n_members=P.third_members, device=DEVICE, lr=P.third_lr, weight_decay=1e-6,
                n_epochs=n_epochs, loss_fn=loss_fn if loss_fn is not None else nn.MSELoss(),
                gap_penalty_weight=0.0,
                log_per_epoch=10 ** 9, input_dim=in_dim, output_dim=1, hidden_dims=hidden,
                dropout_rate=P.third_dropout)

        if not fence:
            return _fit()
        state = rng_snapshot()
        try:
            torch.manual_seed(train_seed + FENCE_OFFSET[route])
            return _fit()
        finally:
            rng_restore(state)

    def read(net, ds):
        with torch.no_grad():
            x = ds.input_transformer.to(DEVICE).transform(feat(grid))
        return ds.outcome_transformer.to(DEVICE).inverse_transform(net(x)).detach().cpu().numpy()

    ds, dsv = create_third_stage_dataset_for_treatment_pcl_net_ate(
        tm, le, lv, TorchStandardScaler(), TorchStandardScaler(),
        treatment_featurizer=feat, device=DEVICE)
    f_tm = read(head(ds, dsv, tm_third_epochs, "tm", _loss(P.tm_third_loss, P.tm_huber_delta)), ds)

    def slack(creator, route):
        s, sv = creator(om, tm, le, lv, outcome_transformer=TorchStandardScaler(),
                        input_transformer=TorchStandardScaler(), input_type="features",
                        treatment_featurizer=feat, device=DEVICE)
        return read(head(s, sv, dr_third_epochs, route, _loss(P.dr_third_loss, P.dr_huber_delta)), s)

    return {"OutcomeNet": f_om, "TreatmentNet": f_tm,
            "DRPCLNET-V1": f_om + slack(DR1, "dr1"), "DRPCLNET-V2": f_om + f_tm - slack(DR2, "dr2")}


# ================================================================ one full run
def _fit_and_estimate_all_arms(d, ds2, l1, l2, le, dse, lv, P, norm_modes, train_seed, om_loaders=None, rows=None):
    """One (stage1, stage2, estimation) triple -> {arm: {estimator: curve}}, the arms PAIRED.

    The outcome bridge is fitted once and SHARED (it does not depend on the normalization); the RNG
    state right after it is replayed before every bridge fit, and the state right after the first
    bridge fit is replayed before every arm's estimation -- so data, folds, inits, dropout masks,
    loader shuffles and third-stage members are identical across arms and the normalization is the
    only moving part. The none and posthoc arms additionally SHARE one trained bridge, because
    post-hoc is a predictor wrapper: train once, read the raw arm, then wrap.
    """
    torch.manual_seed(train_seed)
    l1_om, l2_om = om_loaders if om_loaders is not None else (l1, l2)
    om = fit_outcome(d, l1_om, l2_om, lv, P, rows=rows, train_seed=train_seed)
    rng_after_om = rng_snapshot()

    per_arm, base_tm, rng_after_bridge = {}, None, None
    for mode in norm_modes:
        if mode in ("none", "posthoc"):
            if base_tm is None:
                rng_restore(rng_after_om)
                base_tm = fit_treatment(d, l1, l2, lv, P, mode)     # unconstrained; shared
                rng_after_bridge = rng_snapshot()
            tm = base_tm
            if mode == "posthoc":
                tm = posthoc_normalize(tm, ds2, P)                  # wraps IN PLACE; none ran first
        else:
            rng_restore(rng_after_om)
            tm = fit_treatment(d, l1, l2, lv, P, mode)
            if rng_after_bridge is None:
                rng_after_bridge = rng_snapshot()
        rng_restore(rng_after_bridge)
        per_arm[mode] = estimate(d, om, tm, le, dse, lv, P, train_seed)
    return per_arm


def run_once(data_seed, train_seed, P, norm_modes):
    P.batch = P.batch_size if P.batch_size is not None else P.n_sample
    d = build_data(data_seed, train_seed, P)
    lv = DataLoader(d["val"], batch_size=P.batch, shuffle=False)
    mse = lambda f: float(np.mean((np.asarray(f).reshape(-1, 1) - d["EY"].reshape(-1, 1)) ** 2))

    curves = {m: [] for m in norm_modes}
    for s1, s2, est in d["assignment"]:
        ds1, ds2, dse = d["mk"](s1), d["mk"](s2), d["mk"](est)
        l1 = DataLoader(ds1, batch_size=P.batch, shuffle=True)
        l2 = DataLoader(ds2, batch_size=P.batch, shuffle=True)
        le = DataLoader(dse, batch_size=P.batch, shuffle=True)
        om_loaders = ((DataLoader(ds1, batch_size=P.om_batch, shuffle=True), DataLoader(ds2, batch_size=P.om_batch, shuffle=True))
                      if P.om_batch > 0 else None)
        per_arm = _fit_and_estimate_all_arms(d, ds2, l1, l2, le, dse, lv, P, norm_modes, train_seed,
                                             om_loaders=om_loaders, rows=np.concatenate([s1, s2]))
        for mode in norm_modes:
            curves[mode].append(per_arm[mode])

    out = {}
    for mode in norm_modes:
        cf = {n: np.mean([c[n] for c in curves[mode]], axis=0) for n in ESTIMATORS}
        out[(mode, "cross_fitted")] = {n: mse(cf[n]) for n in ESTIMATORS}

    if P.include_single_split:
        t = d["train_idx"]; third = len(t) // 3
        ds1, ds2, dse = d["mk"](t[:third]), d["mk"](t[third:2 * third]), d["mk"](t[2 * third:])
        l1 = DataLoader(ds1, batch_size=P.batch, shuffle=True)
        l2 = DataLoader(ds2, batch_size=P.batch, shuffle=True)
        le = DataLoader(dse, batch_size=P.batch, shuffle=True)
        per_arm = _fit_and_estimate_all_arms(d, ds2, l1, l2, le, dse, lv, P, norm_modes, train_seed)
        for mode in norm_modes:
            out[(mode, "single_split_disjoint")] = {n: mse(per_arm[mode][n]) for n in ESTIMATORS}
    return out


def main():
    P = parse_args()
    # canonical order guarantees none precedes posthoc (they share one bridge; the posthoc wrapper
    # mutates its predictor, so the raw arm must be read first)
    requested = {"all": ["none", "posthoc", "kkt"], "both": ["posthoc", "kkt"]}.get(
        P.normalization, [P.normalization])
    norm_modes = [m for m in ("none", "posthoc", "kkt") if m in requested]
    out_dir = P.out_dir or os.path.join(HERE, "..", "Results", "Demand_CrossFitting")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    seed_part = "n%d_s%s" % (P.n_sample, "-".join(str(s) for s in P.data_seeds))
    out = os.path.join(out_dir, "DRNMEPCL_DensRatioFree_DemandCF_%s_%s_%s.csv"
                       % (P.tag, seed_part, stamp))
    fields = ["Algorithm", "Arm", "normalization", "Data_Seed", "Repeat", "Train_Seed",
              "Causal_MSE", "hostname"] + HYPER_COLUMNS
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
                for (mode, arm), vals in res.items():
                    for est in ESTIMATORS:
                        w.writerow(dict(Algorithm=est, Arm=arm, normalization=mode,
                                        Data_Seed=data_seed, Repeat=repeat, Train_Seed=train_seed,
                                        Causal_MSE=vals[est], **meta))
            print("n=%d data_seed %d repeat %d (%ds) | %s"
                  % (P.n_sample, data_seed, repeat, time.time() - t0,
                     " || ".join("%s: %s" % (m, "  ".join(
                         "%s %.2f" % (e[:4], res[(m, "cross_fitted")][e]) for e in ESTIMATORS))
                         for m in norm_modes)), flush=True)


if __name__ == "__main__":
    main()
