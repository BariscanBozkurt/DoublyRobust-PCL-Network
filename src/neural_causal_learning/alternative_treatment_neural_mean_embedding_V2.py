"""Density-ratio-free treatment bridge, trainer V2: a PROFILED (differentiate-through-the-head) objective.

Same estimator, same class structure and same batch-wise ProxyDataset/DataLoader loop as
`alternative_treatment_neural_mean_embedding.py`; what changes is HOW the featurizers get their gradient
-- the outcome-bridge V2 recipe (`proxy_outcome_neural_mean_embedding_V2.py`) applied to the KAP bridge:

  stage 1  E[phi_Z(Z) | A, W] : ridge head solved INSIDE the loss (gradient through the solve),
           DFPV convention (F'F + lam * n * I)^-1 F'Y, loss = SSE + lam * n * ||W1||^2.
  stage 2  KAP head Theta      : the KAP objective is a quadratic in Theta,
               L(Theta) = (1/N) sum_i <Theta, U_i>^2  -  2 <Theta, b_U>  (+ lam ||Theta||^2),
           U_i = mu_{Z|A_i,W_i} (x) phi_A2(A_i) (matched), b_U = the complete off-diagonal U-statistic of
           the mismatched cross term (exact O(N) mean-W reduction, unchanged from the base class). Its
           minimiser Theta* = (U'U/N + lam I)^-1 b_U is solved INSIDE the loss and the loss is evaluated
           AT Theta*, so phi_Z / phi_A2 (and W1 through mu) receive the profiled gradient. This is a plain
           ridge (no proximal previous-weight term), so lam IS the regulariser -- not an inverse step
           size as in the shipped head (see the base class comment). The KKT normalisation constraint,
           if set, replaces that solve by its block system exactly as in the base class.
  extras   intercept cross terms in the bridge feature ([phi_Z,1] (x) [phi_A2,1], `cross_terms`), and
           BatchNorm batch statistics at prediction (`predict_bn_train_mode`) -- the two ingredients the
           outcome-bridge V2 needed to reproduce the DFPV bridge with spectral-norm ReLU featurizers.
  end      an EXACT full-data refit of both heads from batch-ACCUMULATED sufficient statistics.

Full batch is a batch-size setting, not hard-coded. No backdoor X path (dSprite has none): raises.
Interface parity: `AlternativeTreatmentBridgePCLNET` (shadowing subclass) +
`train_alternative_treatment_pcl_net_ate_model_v2` + `train_alternative_treatment_pcl_net_ensemble_v2`
(same signature family as the shipped ensemble trainer), so a simulation swaps it in behind a flag.
"""
from typing import Callable, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_causal_learning.alternative_treatment_neural_mean_embedding import (
    AlternativeTreatmentBridgePCLNET as _Base, EnsembleAlternativeTreatmentBridge)
from torch_utils.linear_algebra import add_const_col, outer_prod, outer_prod_cross_batch


def ridge_solve(target, feature, reg: float, jitter: float = 1e-6):
    """(F'F + reg*n*I)^-1 F'Y, differentiable. Returns (feature_dim, target_dim)."""
    n, d = feature.shape
    gram = feature.T @ feature + (reg * n + jitter) * torch.eye(d, device=feature.device, dtype=feature.dtype)
    return torch.linalg.solve(gram, feature.T @ target)


class AlternativeTreatmentBridgePCLNET(_Base):
    """Shadowing subclass: identical construction and third-stage-facing contract; profiled losses,
    optional intercept cross terms in the bridge feature, optional BN-train-mode prediction."""

    def __init__(self, *args, cross_terms: bool = True, predict_bn_train_mode: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if (self.backdoor_featurizer_first_stage is not None) or (self.backdoor_featurizer_second_stage is not None):
            raise NotImplementedError("trainer V2 has no backdoor path (dSprite has no X)")
        self.cross_terms = cross_terms
        self.predict_bn_train_mode = predict_bn_train_mode

    # ------------------------------------------------------------------ features
    def _aug(self, z_feat, a2_feat):
        """Bridge feature: phi_Z (x) phi_A2, or [phi_Z,1] (x) [phi_A2,1] with cross terms."""
        if self.cross_terms:
            return outer_prod(add_const_col(z_feat), add_const_col(a2_feat)).flatten(start_dim=1)
        return outer_prod(z_feat, a2_feat).flatten(start_dim=1)

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
            for feat in (self.treatment_featurizer_second_stage, self.treatment_proxy_featurizer, self.treatment_featurizer_first_stage, self.outcome_proxy_featurizer):
                for m in feat.modules():
                    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                        m.train(True)
    def _first_stage_design_nb(self, treatment, outcome_proxy):
        """phi_A1(A) (x) phi_W(W), flattened (no backdoor)."""
        a1 = self.treatment_featurizer_first_stage(treatment)
        w = self.outcome_proxy_featurizer(outcome_proxy)
        return a1, w, outer_prod(a1, w).flatten(start_dim=1)

    # ------------------------------------------------------------------ losses
    def first_stage_loss(self, first_stage_data_tuple, loss_fn=None, reg: float = 0.1):
        treatment, _, treatment_proxy, outcome_proxy = first_stage_data_tuple[0], first_stage_data_tuple[1], first_stage_data_tuple[2], first_stage_data_tuple[3]
        _, _, F = self._first_stage_design_nb(treatment, outcome_proxy)          # (n, d1), with grad
        with torch.no_grad():                                                     # Z-featurizer is stage-2
            phi_Z = self.treatment_proxy_featurizer(treatment_proxy)
        W1 = ridge_solve(phi_Z, F, reg)                                           # (d1, D_Z)
        n = F.shape[0]
        loss = torch.sum((phi_Z - F @ W1) ** 2) + reg * n * torch.sum(W1 ** 2)
        with torch.no_grad():
            self.final_layer_first_stage_weight = W1.T.detach().clone()           # buffer convention (D_Z, d1)
        return loss

    def _kap_terms(self, first_stage_tuple, second_stage_tuple, reg12: float):
        """Matched design U (N, P) and the complete-U-statistic cross RHS b_U (1, P), with the stage-1
        head re-solved on the first-stage batch (differentiable w.r.t. phi_Z). First-stage nets frozen."""
        # ---- stage-1 head on the first-stage batch (grad flows to phi_Z through the target) ----
        with torch.no_grad():
            _, _, F1 = self._first_stage_design_nb(first_stage_tuple[0], first_stage_tuple[3])
        phi_Z_1 = self.treatment_proxy_featurizer(first_stage_tuple[2])
        W1 = ridge_solve(phi_Z_1, F1, reg12)                                       # (d1, D_Z)
        # ---- second-stage batch ----
        with torch.no_grad():
            a1, w, F2 = self._first_stage_design_nb(second_stage_tuple[0], second_stage_tuple[3])
        a2 = self.treatment_featurizer_second_stage(second_stage_tuple[0])
        mu = F2 @ W1                                                               # mu_{Z|A_i,W_i}
        U = self._aug(mu, a2)                                                      # matched
        # mismatched cross term: E_W[mu] = FirstStageMap(phi_A1(A_i) (x) mean_W phi_W)
        w_mean = w.mean(0, keepdim=True)
        mu_cross = outer_prod_cross_batch(a1, w_mean).flatten(start_dim=1) @ W1    # (N, D_Z)
        b = self._aug(mu_cross, a2).mean(0, keepdim=True)                           # (1, P), V-statistic
        b = self._complete_u_stat_cross_rhs(b, U)                                  # complete U-statistic
        return U, b, W1

    def second_stage_loss(self, first_stage_tuple, second_stage_tuple,
                          second_stage_first_final_layer_regularizer: float = 0.1,
                          second_stage_final_layer_regularizer: float = 0.1,
                          first_final_layer_update_type: str = "ridge"):
        U, b, W1 = self._kap_terms(first_stage_tuple, second_stage_tuple, second_stage_first_final_layer_regularizer)
        n, P = U.shape
        lam = second_stage_final_layer_regularizer
        A = U.T @ U + (n * lam + 1e-6) * torch.eye(P, device=U.device, dtype=U.dtype)
        RHS = b.T * n
        if self.normalization_constraint is None:
            theta = torch.linalg.solve(A, RHS)                                     # (P, 1), differentiable
        else:
            data_tuple = [x.to(self.device) for x in second_stage_tuple]
            theta = self.normalization_constraint.solve(self, data_tuple, A, RHS)  # KKT block solve, differentiable in A/RHS
        # KAP objective evaluated at the profiled head
        loss = torch.mean((U @ theta) ** 2) - 2.0 * (b @ theta).mean() + lam * torch.sum(theta ** 2)
        with torch.no_grad():
            self.final_layer_first_stage_weight = W1.T.detach().clone()
            self.final_layer_second_stage_weight = theta.T.detach().clone()
        return loss

    def second_stage_loss_val(self, second_stage_tuple):
        with torch.no_grad():
            U, b, _ = self._kap_terms(second_stage_tuple, second_stage_tuple, 1e-1)
            theta = self.final_layer_second_stage_weight.T
            return torch.mean((U @ theta) ** 2) - 2.0 * (b @ theta).mean()

    @torch.no_grad()
    def refit_heads_full_data(self, first_stage_train_dataloader, second_stage_train_dataloader,
                              reg1: float, reg2: float):
        """Exact full-data refit of both heads from batch-accumulated statistics (the U-statistic cross
        RHS is accumulated batch-wise as sum_b N_b * b_b, exact when one batch covers the fold)."""
        self._pred_mode()
        gram = xty = None; n1 = 0
        for batch in first_stage_train_dataloader:
            batch = [x.to(self.device) for x in batch]
            _, _, F = self._first_stage_design_nb(batch[0], batch[3]); t = self.treatment_proxy_featurizer(batch[2])
            gram = F.T @ F if gram is None else gram + F.T @ F
            xty = F.T @ t if xty is None else xty + F.T @ t
            n1 += F.shape[0]
        d = gram.shape[0]
        W1 = torch.linalg.solve(gram + (reg1 * n1 + 1e-6) * torch.eye(d, device=gram.device, dtype=gram.dtype), xty)
        self.final_layer_first_stage_weight = W1.T.clone()
        UtU = bsum = None; n2 = 0
        for batch in second_stage_train_dataloader:
            batch = [x.to(self.device) for x in batch]
            a1, w, F2 = self._first_stage_design_nb(batch[0], batch[3]); a2 = self.treatment_featurizer_second_stage(batch[0])
            U = self._aug(F2 @ W1, a2)
            mu_cross = outer_prod_cross_batch(a1, w.mean(0, keepdim=True)).flatten(start_dim=1) @ W1
            b = self._complete_u_stat_cross_rhs(self._aug(mu_cross, a2).mean(0, keepdim=True), U)
            UtU = U.T @ U if UtU is None else UtU + U.T @ U
            bsum = b.T * U.shape[0] if bsum is None else bsum + b.T * U.shape[0]
            n2 += U.shape[0]
        P = UtU.shape[0]
        A = UtU + (n2 * reg2 + 1e-6) * torch.eye(P, device=UtU.device, dtype=UtU.dtype)
        if self.normalization_constraint is None:
            theta = torch.linalg.solve(A, bsum)
        else:
            last = [x.to(self.device) for x in batch]
            theta = self.normalization_constraint.solve(self, last, A, bsum)
        self.final_layer_second_stage_weight = theta.T.clone()

    # ------------------------------------------------------------------ prediction (third-stage contract)
    def predict_treatment_bridge_function(self, treatment, treatment_proxy, backdoor=None):
        training_state = self.training
        self._pred_mode()
        with torch.no_grad():
            f = self._aug(self.treatment_proxy_featurizer(treatment_proxy.to(self.device)),
                          self.treatment_featurizer_second_stage(treatment.to(self.device)))
            out = f @ self.final_layer_second_stage_weight.T
        if training_state:
            self.train()
        return out

    def predict_dens_ratio(self, treatment, outcome_proxy, backdoor=None):
        training_state = self.training
        self._pred_mode()
        with torch.no_grad():
            _, _, F = self._first_stage_design_nb(treatment.to(self.device), outcome_proxy.to(self.device))
            mu = F @ self.final_layer_first_stage_weight.T
            f = self._aug(mu, self.treatment_featurizer_second_stage(treatment.to(self.device)))
            out = f @ self.final_layer_second_stage_weight.T
        if training_state:
            self.train()
        return out


def train_alternative_treatment_pcl_net_ate_model_v2(
        model, first_stage_train_dataloader, second_stage_train_dataloader,
        stage1_optimizers, stage2_optimizers, n_epochs: int = 20, stage1_iter: int = 20, stage2_iter: int = 1,
        first_stage_final_layer_regularizer: float = 0.1, second_stage_final_layer_regularizer: float = 0.1,
        second_stage_first_final_layer_regularizer: Optional[float] = None,
        stage1_schedulers=None, stage2_schedulers=None, log_per_epoch: int = 10 ** 9, **kwargs):
    """Batch-wise loop mirroring the shipped trainer's stage toggles (stage 1 trains {A1, W} with {A2, Z}
    frozen; stage 2 the reverse), losses profiled (heads set inside), then an exact accumulated refit."""
    reg1 = float(first_stage_final_layer_regularizer); reg2 = float(second_stage_final_layer_regularizer)
    reg12 = reg1 if second_stage_first_final_layer_regularizer is None else float(second_stage_first_final_layer_regularizer)
    device = model.device
    for epoch in tqdm(range(n_epochs), disable=(log_per_epoch >= 10 ** 9)):
        model.train()
        # ---- stage 1 ----
        model.treatment_featurizer_first_stage.train(True); model.outcome_proxy_featurizer.train(True)
        model.treatment_featurizer_second_stage.train(False); model.treatment_proxy_featurizer.train(False)
        for _ in range(stage1_iter):
            for batch in first_stage_train_dataloader:
                batch = [x.to(device) for x in batch]
                for opt in stage1_optimizers: opt.zero_grad()
                l1 = model.first_stage_loss(batch, reg=reg1); l1.backward()
                for opt in stage1_optimizers: opt.step()
        # ---- stage 2 ----
        model.treatment_featurizer_first_stage.train(False); model.outcome_proxy_featurizer.train(False)
        model.treatment_featurizer_second_stage.train(True); model.treatment_proxy_featurizer.train(True)
        for _ in range(stage2_iter):
            for b1, b2 in zip(first_stage_train_dataloader, second_stage_train_dataloader):
                b1 = [x.to(device) for x in b1]; b2 = [x.to(device) for x in b2]
                for opt in stage2_optimizers: opt.zero_grad()
                l2 = model.second_stage_loss(b1, b2, second_stage_first_final_layer_regularizer=reg12,
                                             second_stage_final_layer_regularizer=reg2)
                l2.backward()
                for opt in stage2_optimizers: opt.step()
        for sch in (stage1_schedulers or []): sch.step()
        for sch in (stage2_schedulers or []): sch.step()
        if log_per_epoch < 10 ** 9 and (epoch + 1) % log_per_epoch == 0:
            print("epoch %d  stage1 %.4g  stage2 %.4g" % (epoch + 1, l1.item(), l2.item()))
    model.refit_heads_full_data(first_stage_train_dataloader, second_stage_train_dataloader, reg12, reg2)
    model.eval()
    return model, [], []


def train_alternative_treatment_pcl_net_ensemble_v2(
        build_nets_fn: Callable, first_stage_train_dataloader, second_stage_train_dataloader,
        validation_dataloader=None, n_members: int = 5, device: str = "cuda", base_seed: int = 0,
        lr_stage1: float = 1e-3, lr_stage2: float = 1e-3, weight_decay: float = 0.1,
        n_epochs: int = 20, stage1_iter: int = 20, stage2_iter: int = 1,
        first_stage_final_layer_regularizer: float = 0.1, second_stage_final_layer_regularizer: float = 0.1,
        second_stage_first_final_layer_regularizer: Optional[float] = None,
        normalization_constraint=None, cross_terms: bool = True, predict_bn_train_mode: bool = False,
        **kwargs) -> EnsembleAlternativeTreatmentBridge:
    """Same contract as the shipped ensemble trainer (members seeded base_seed + k, coupled-L2 Adam per
    stage -- the shipped TM trainer already uses torch.optim.Adam -- constraint factory per member),
    with the V2 class and trainer. Returns an EnsembleAlternativeTreatmentBridge, so the third-stage
    creators, the phi1/global arm overrides and the identity diagnostics all work unchanged."""
    members = []
    for k in range(n_members):
        torch.manual_seed(base_seed + k)
        nets = build_nets_fn(device)
        t_first, t_second, t_proxy, out_proxy = nets[0], nets[1], nets[2], nets[3]
        if len(nets) == 6 and (nets[4] is not None or nets[5] is not None):
            raise NotImplementedError("trainer V2 has no backdoor path")
        member_constraint = (normalization_constraint() if callable(normalization_constraint)
                             and not hasattr(normalization_constraint, "system") else normalization_constraint)
        model = AlternativeTreatmentBridgePCLNET(
            treatment_featurizer_first_stage=t_first, treatment_featurizer_second_stage=t_second,
            treatment_proxy_featurizer=t_proxy, outcome_proxy_featurizer=out_proxy, device=device,
            normalization_constraint=member_constraint, cross_terms=cross_terms,
            predict_bn_train_mode=predict_bn_train_mode)
        o1 = [torch.optim.Adam(list(t_first.parameters()) + list(out_proxy.parameters()), lr=lr_stage1, weight_decay=weight_decay)]
        o2 = [torch.optim.Adam(list(t_second.parameters()) + list(t_proxy.parameters()), lr=lr_stage2, weight_decay=weight_decay)]
        model, _, _ = train_alternative_treatment_pcl_net_ate_model_v2(
            model, first_stage_train_dataloader, second_stage_train_dataloader, o1, o2,
            n_epochs=n_epochs, stage1_iter=stage1_iter, stage2_iter=stage2_iter,
            first_stage_final_layer_regularizer=first_stage_final_layer_regularizer,
            second_stage_final_layer_regularizer=second_stage_final_layer_regularizer,
            second_stage_first_final_layer_regularizer=second_stage_first_final_layer_regularizer)
        members.append(model)
    return EnsembleAlternativeTreatmentBridge(members, device=device).eval()
