"""Outcome bridge, trainer V2: a PROFILED (differentiate-through-the-head) training objective.

Same estimator, same class structure and same batch-wise ProxyDataset/DataLoader training loop as
`proxy_outcome_neural_mean_embedding.py`; what changes is HOW the featurizers receive their gradient.

    shipped trainer  : featurizer step with the closed-form head held as a constant buffer, THEN the
                       head is re-solved (alternating / block-coordinate; proximal iterate).
    this trainer     : the closed-form ridge head is solved INSIDE the loss, so the featurizer gradient
                       flows THROUGH the solve -- the loss is the ridge residual at the head that is
                       optimal for the current features (the "deep feature" objective of Xu et al. 2021,
                       DFPV). Ridge convention mirrors theirs: (F'F + lam * n * I)^-1 F'Y.

Why it exists: the official DFPV outcome bridge scores 15.7 on dSprite V2 at n=5000
where ours scored 24.2, and every attempt to transplant its ingredients (spectral-norm nets,
tensor-product first stage, raw inputs, full batch) into the ALTERNATING trainer failed -- those nets
do not train under alternation. Under the profiled objective they do. Full batch is not hard-coded:
set the DataLoader batch size to the fold size to reproduce DFPV exactly; smaller batches keep the
usual batch-wise logic (per-batch solves during training; an EXACT full-data refit of both heads,
accumulated batch-wise, at the end).

Interface parity: `OutcomeBridgePCLNET` (shadowing subclass) + `train_deep_feature_proxy_profiled_ate_model`
with the same argument family as the shipped trainer, so a simulation can swap it in behind a flag.
Everything downstream (predict_outcome_bridge_function, pred_structural_function,
compute_mean_outcome_proxy_feature, the DR third stage) is inherited unchanged.
"""
from itertools import cycle
from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_causal_learning.proxy_outcome_neural_mean_embedding import OutcomeBridgePCLNET as _BaseOutcomeBridgePCLNET
from torch_utils.linear_algebra import add_const_col, outer_prod
from torch_utils.scalers import TorchIdentityTransformer


def ridge_solve(target: torch.Tensor, feature: torch.Tensor, reg: float, jitter: float = 1e-6) -> torch.Tensor:
    """(F'F + reg * n * I)^-1 F'Y  -- DFPV's `fit_linear` convention (reg is per-sample), DIFFERENTIABLE.
    Returns weight of shape (feature_dim, target_dim)."""
    n, d = feature.shape
    gram = feature.T @ feature + (reg * n + jitter) * torch.eye(d, device=feature.device, dtype=feature.dtype)
    return torch.linalg.solve(gram, feature.T @ target)


def ridge_solve_accumulated(gram: torch.Tensor, xty: torch.Tensor, n_total: int, reg: float,
                            jitter: float = 1e-6) -> torch.Tensor:
    """Same solve from batch-accumulated sufficient statistics (sum F'F, sum F'Y over ALL batches):
    the exact full-data ridge, computed batch-wise."""
    d = gram.shape[0]
    return torch.linalg.solve(gram + (reg * n_total + jitter) * torch.eye(d, device=gram.device, dtype=gram.dtype), xty)


class OutcomeBridgePCLNET(_BaseOutcomeBridgePCLNET):
    """Shadowing subclass (repo idiom, cf. `_w_SGD`): identical construction and DR-facing methods; the
    two training losses solve their heads inline, and the SECOND-STAGE FEATURE carries DFPV's
    intercept cross terms: [phi_A,1] (x) [phi_W,1] (the base class uses phi_A (x) phi_W plus ONE
    constant, i.e. interaction-only -- fine for centred LayerNorm/GELU features, but with
    non-negative ReLU/spectral-norm features the missing main-effect terms cost a lot: OM 32 vs the
    wrapper's 17 on dSprite V2 seed 100). The prediction paths are overridden to build the same
    augmented feature, so train/predict stay consistent."""

    def __init__(self, first_stage_featurizer, treatment_featurizer, outcome_proxy_featurizer,
                 backdoor_featurizer=None, final_layer_first_stage_weight=None,
                 final_layer_second_stage_weight=None, outcome_dim: int = 1, device: str = "cuda",
                 cross_terms: bool = True, predict_bn_train_mode: bool = False):
        super().__init__(first_stage_featurizer, treatment_featurizer, outcome_proxy_featurizer,
                         backdoor_featurizer=backdoor_featurizer,
                         final_layer_first_stage_weight=final_layer_first_stage_weight,
                         final_layer_second_stage_weight=final_layer_second_stage_weight,
                         outcome_dim=outcome_dim, device=device)
        self.cross_terms = cross_terms
        # DFPV never calls eval: its predict_t / fit_t run the stage-2 nets in TRAIN mode, i.e. BatchNorm
        # uses the statistics of the batch being predicted (the 588-grid, the full W set). True mirrors that.
        self.predict_bn_train_mode = predict_bn_train_mode
        if cross_terms and backdoor_featurizer is not None:
            raise NotImplementedError("cross_terms second stage has no backdoor path (dSprite has no X)")

    def _pred_mode(self):
        """Featurizer mode for prediction: eval (ours) or DFPV's BatchNorm-batch-statistics convention.
        BN-train mode is applied to the BatchNorm modules ONLY. Putting whole featurizers in train mode
        let spectral_norm advance its power iteration on every forward (measured: two identical calls
        differ by up to 0.8), so predictions depended on how many calls preceded them -- the shared
        OutcomeNet drifted 13.16 -> 14.23 -> 14.88 across the paired arms. With spectral
        norm (and any dropout) frozen, a prediction is a pure function of its inputs while BatchNorm
        still normalises by the batch statistics."""
        self.eval()
        if self.predict_bn_train_mode:
            for feat in (self.treatment_featurizer, self.outcome_proxy_featurizer):
                for m in feat.modules():
                    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                        m.train(True)
    # ---- augmented second-stage feature: [phi_A,1] (x) [phi_W,1] (includes the constant) ----
    def _aug2(self, treatment_feature, outcome_proxy_feature):
        if not self.cross_terms:
            return add_const_col(outer_prod(treatment_feature, outcome_proxy_feature).flatten(start_dim=1))
        return outer_prod(add_const_col(treatment_feature), add_const_col(outcome_proxy_feature)).flatten(start_dim=1)

    def _compute_second_stage_features(self, treatment, treatment_proxy, backdoor, w1_weight_matrix):
        if not self.cross_terms:
            return super()._compute_second_stage_features(treatment, treatment_proxy, backdoor, w1_weight_matrix)
        with torch.no_grad():
            feat_stage1 = self._compute_first_stage_features(treatment, treatment_proxy, backdoor)
        pred_w = feat_stage1 @ w1_weight_matrix
        return self._aug2(self.treatment_featurizer(treatment), pred_w)

    def compute_mean_outcome_proxy_feature(self, train_dataloader):
        training_state = self.training
        self._pred_mode()
        with torch.no_grad():
            W_all = torch.cat([batch[3] for batch in train_dataloader]).to(self.device)
            self.mean_outcome_proxy_feature = self.outcome_proxy_featurizer(W_all).mean(0, keepdim=True)
        if training_state:
            self.train()

    def predict_outcome_bridge_function(self, treatment, outcome_proxy, backdoor=None):
        if not self.cross_terms:
            return super().predict_outcome_bridge_function(treatment, outcome_proxy, backdoor)
        training_state = self.training
        self._pred_mode()
        with torch.no_grad():
            feature = self._aug2(self.treatment_featurizer(treatment.to(self.device)),
                                 self.outcome_proxy_featurizer(outcome_proxy.to(self.device)))
        pred = feature @ self.final_layer_second_stage_weight.T
        if training_state:
            self.train()
        return pred

    def pred_structural_function(self, treatment, treatment_transformer=None, outcome_transformer=None):
        if not self.cross_terms:
            return super().pred_structural_function(treatment, treatment_transformer or TorchIdentityTransformer(),
                                                    outcome_transformer or TorchIdentityTransformer())
        treatment_transformer = (treatment_transformer or TorchIdentityTransformer()).to(self.device)
        outcome_transformer = (outcome_transformer or TorchIdentityTransformer()).to(self.device)
        training_state = self.training
        self._pred_mode()
        a = treatment_transformer.transform(treatment.to(self.device))
        with torch.no_grad():
            treatment_feature = self.treatment_featurizer(a)
        mean_w = self.mean_outcome_proxy_feature.expand(treatment_feature.shape[0], -1)
        feature = self._aug2(treatment_feature, mean_w)
        pred = outcome_transformer.inverse_transform(feature @ self.final_layer_second_stage_weight.T)
        if training_state:
            self.train()
        return pred

    def first_stage_loss(self, first_stage_data_tuple, loss_fn=None, reg: float = 0.1):
        """Ridge residual of phi_W on the first-stage features at the head optimal for the CURRENT
        features (gradient flows through the solve). Stores the solved head in the buffer so
        `first_stage_forward` / the DR creators see the same head the loss used."""
        treatment, _, treatment_proxy, outcome_proxy, backdoor = self._unpack_batch(first_stage_data_tuple)
        with torch.no_grad():                                   # W-featurizer is a stage-2 parameter
            outcome_proxy_feature = self.outcome_proxy_featurizer(outcome_proxy)
        feature = self._compute_first_stage_features(treatment, treatment_proxy, backdoor)   # (n, d1+1)
        w1 = ridge_solve(outcome_proxy_feature, feature, reg)   # (d1+1, out_W), differentiable
        pred = feature @ w1
        n = feature.shape[0]
        loss = torch.sum((outcome_proxy_feature - pred) ** 2) + reg * n * torch.sum(w1 ** 2)
        with torch.no_grad():
            self.final_layer_first_stage_weight = w1.T.detach().clone()          # buffer convention (out, in)
        return loss

    def second_stage_loss(self, first_stage_tuple, second_stage_tuple, loss_fn=None,
                          second_stage_first_final_layer_regularizer: float = 0.1,
                          second_stage_final_layer_regularizer: float = 0.1):
        """DFPV's fit_2sls as a loss: stage-1 head re-solved on the first-stage batch (differentiable
        w.r.t. the W-featurizer), predicted phi_W on the second-stage batch, stage-2 head solved inline
        (differentiable w.r.t. the treatment featurizer), ridge residual on Y. First-stage featurizer is
        frozen here (as in the shipped trainer and in DFPV)."""
        treatment, _, treatment_proxy, outcome_proxy, backdoor = self._unpack_batch(first_stage_tuple)
        with torch.no_grad():
            feature_1st = self._compute_first_stage_features(treatment, treatment_proxy, backdoor)
        outcome_proxy_feature = self.outcome_proxy_featurizer(outcome_proxy)                 # with grad
        w1 = ridge_solve(outcome_proxy_feature, feature_1st, second_stage_first_final_layer_regularizer)

        treatment, Y, treatment_proxy, outcome_proxy, backdoor = self._unpack_batch(second_stage_tuple)
        feature_2nd = self._compute_second_stage_features(treatment, treatment_proxy, backdoor, w1)  # (n, d2+1)
        w2 = ridge_solve(Y, feature_2nd, second_stage_final_layer_regularizer)                  # (d2+1, outcome_dim)
        pred = feature_2nd @ w2
        n = feature_2nd.shape[0]
        # DFPV's stage-2 loss penalises lam2 * ||w2||^2 WITHOUT the n factor (their fit_2sls); only the
        # stage-1 `linear_reg_loss` carries reg * n. Mirrored exactly -- with the n factor the
        # featurizers were pushed to shrink w2 and the bridge under-fit (OM 41 at lam2=10).
        loss = torch.sum((Y - pred) ** 2) + second_stage_final_layer_regularizer * torch.sum(w2 ** 2)
        with torch.no_grad():
            self.final_layer_first_stage_inside_second_stage_weight = w1.T.detach().clone()
            self.final_layer_second_stage_weight = w2.T.detach().clone()
        return loss

    @torch.no_grad()
    def refit_heads_full_data(self, first_stage_train_dataloader: DataLoader,
                              second_stage_train_dataloader: DataLoader,
                              reg1: float, reg2: float):
        """Exact full-data ridge refit of BOTH heads from batch-accumulated sufficient statistics
        (DFPV's final `fit_t` on the whole stage sets, kept batch-wise)."""
        self._pred_mode()
        gram1 = xty1 = None; n1 = 0
        for batch in first_stage_train_dataloader:
            treatment, _, treatment_proxy, outcome_proxy, backdoor = self._unpack_batch(batch)
            f = self._compute_first_stage_features(treatment, treatment_proxy, backdoor)
            t = self.outcome_proxy_featurizer(outcome_proxy)
            gram1 = f.T @ f if gram1 is None else gram1 + f.T @ f
            xty1 = f.T @ t if xty1 is None else xty1 + f.T @ t
            n1 += f.shape[0]
        w1 = ridge_solve_accumulated(gram1, xty1, n1, reg1)
        self.final_layer_first_stage_weight = w1.T.clone()
        self.final_layer_first_stage_inside_second_stage_weight = w1.T.clone()

        gram2 = xty2 = None; n2 = 0
        for batch in second_stage_train_dataloader:
            treatment, Y, treatment_proxy, outcome_proxy, backdoor = self._unpack_batch(batch)
            f = self._compute_second_stage_features(treatment, treatment_proxy, backdoor, w1)
            gram2 = f.T @ f if gram2 is None else gram2 + f.T @ f
            xty2 = f.T @ Y if xty2 is None else xty2 + f.T @ Y
            n2 += f.shape[0]
        w2 = ridge_solve_accumulated(gram2, xty2, n2, reg2)
        self.final_layer_second_stage_weight = w2.T.clone()


def build_coupled_adam_optimizers(pcl_model, lr: float = 1e-3, weight_decay: float = 0.1):
    """DFPV-style optimizers: torch.optim.Adam with COUPLED L2 weight decay (gradient += wd * w), one
    per featurizer, same stage grouping as the shipped builder. The shipped builder uses AdamW
    (decoupled decay), which at wd=0.1, lr=1e-3 shrinks weights by only 1e-4 per step -- effectively
    unregularised -- whereas DFPV's coupled 0.1*w is a strong shrinkage that keeps the spectral-norm
    ReLU features small and smooth (and is why a heavier stage-2 ridge helps in their loop).
    Returns (stage1_optimizers, stage2_optimizers)."""
    o1 = [torch.optim.Adam(pcl_model.first_stage_featurizer.parameters(), lr=lr, weight_decay=weight_decay)]
    o2 = [torch.optim.Adam(pcl_model.treatment_featurizer.parameters(), lr=lr, weight_decay=weight_decay),
          torch.optim.Adam(pcl_model.outcome_proxy_featurizer.parameters(), lr=lr, weight_decay=weight_decay)]
    return o1, o2


def train_deep_feature_proxy_profiled_ate_model(
        pcl_model: OutcomeBridgePCLNET,
        first_stage_train_dataloader: DataLoader,
        second_stage_train_dataloader: DataLoader,
        stage1_optimizers, stage2_optimizers,
        stage1_schedulers=None, stage2_schedulers=None,
        first_stage_final_layer_regularizer: float = 0.1,
        second_stage_final_layer_regularizer: float = 0.1,
        second_stage_first_final_layer_regularizer: Optional[float] = None,
        n_epochs: int = 20, stage1_iter: int = 20, stage2_iter: int = 1,
        validation_dataloader: Optional[DataLoader] = None,
        log_per_epoch: int = 10 ** 9,
        **kwargs):
    """Batch-wise two-stage training with the profiled losses. Mirrors the shipped trainer's loop
    (stage-1 sweep over the first-stage loader with the second-stage featurizers frozen; stage-2 sweep
    pairing first- and second-stage batches, cycling the shorter loader), then an exact full-data
    refit of both heads. Regularizers are FIXED per-sample ridges (no annealing) as in DFPV; scalars
    only -- pass (lo, hi) tuples to the shipped trainer if you want annealing. Schedulers are applied
    per epoch if given (DFPV uses none). Extra kwargs are accepted and ignored for interface parity."""
    if isinstance(first_stage_final_layer_regularizer, (tuple, list)):
        first_stage_final_layer_regularizer = float(first_stage_final_layer_regularizer[0])
    if isinstance(second_stage_final_layer_regularizer, (tuple, list)):
        second_stage_final_layer_regularizer = float(second_stage_final_layer_regularizer[0])
    reg1 = float(first_stage_final_layer_regularizer)
    reg2 = float(second_stage_final_layer_regularizer)
    reg12 = reg1 if second_stage_first_final_layer_regularizer is None else float(second_stage_first_final_layer_regularizer)
    if isinstance(reg12, (tuple, list)):
        reg12 = float(reg12[0])

    len_1, len_2 = len(first_stage_train_dataloader), len(second_stage_train_dataloader)
    batches_per_epoch = max(len_1, len_2)

    for epoch in tqdm(range(n_epochs), disable=(log_per_epoch >= 10 ** 9)):
        pcl_model.train()
        # ---- stage 1: first-stage featurizer (second-stage featurizers frozen) ----
        pcl_model.freeze_featurizer_params("second_stage")
        pcl_model.unfreeze_featurizer_params("first_stage")
        for _ in range(stage1_iter):
            for batch in first_stage_train_dataloader:
                for opt in stage1_optimizers:
                    opt.zero_grad()
                loss1 = pcl_model.first_stage_loss(batch, reg=reg1)
                loss1.backward()
                for opt in stage1_optimizers:
                    opt.step()
        # ---- stage 2: treatment + outcome-proxy featurizers (first-stage featurizer frozen) ----
        pcl_model.freeze_featurizer_params("first_stage")
        pcl_model.unfreeze_featurizer_params("second_stage")
        for _ in range(stage2_iter):
            if len_1 >= len_2:
                iter_1, iter_2 = iter(first_stage_train_dataloader), cycle(second_stage_train_dataloader)
            else:
                iter_1, iter_2 = cycle(first_stage_train_dataloader), iter(second_stage_train_dataloader)
            for _ in range(batches_per_epoch):
                b1, b2 = next(iter_1), next(iter_2)
                for opt in stage2_optimizers:
                    opt.zero_grad()
                loss2 = pcl_model.second_stage_loss(
                    b1, b2, second_stage_first_final_layer_regularizer=reg12,
                    second_stage_final_layer_regularizer=reg2)
                loss2.backward()
                for opt in stage2_optimizers:
                    opt.step()
        for sch in (stage1_schedulers or []):
            sch.step()
        for sch in (stage2_schedulers or []):
            sch.step()
        if log_per_epoch < 10 ** 9 and (epoch + 1) % log_per_epoch == 0:
            print("epoch %d  stage1 %.4g  stage2 %.4g" % (epoch + 1, loss1.item(), loss2.item()))

    # ---- exact full-data refit of both heads (batch-accumulated) ----
    pcl_model.refit_heads_full_data(first_stage_train_dataloader, second_stage_train_dataloader, reg12, reg2)
    pcl_model.eval()
    return pcl_model
