import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Optional, Tuple
from torch.optim.lr_scheduler import _LRScheduler as Scheduler # Use abstract base class for type hint

import numpy as np
from tqdm import tqdm
import pylab as pl
from IPython.display import clear_output, display

import copy

import sys
sys.path.append("..")
from torch_utils.scalers import TorchIdentityTransformer
from torch_utils.linear_algebra import fit_linear_proximal, outer_prod, outer_prod_batch
from torch_utils.model_helpers import get_last_linear_out_features


def make_ordered_pair_indices(n_data: int,
                              pair_mode: str = "full",
                              n_derangements: int = 1,
                              device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    """Ordered (source i, anchor j) index pairs with i != j, for the joint-anchor ATT objective.

    Two pairings, both from the ATT derivation:
      - pair_mode="full"        : the COMPLETE ordered-pair U-statistic (derivation eq. 24), all M(M-1)
                                  ordered pairs. The diagonal i = j is excluded because (O_i, O_i) is
                                  NOT a draw from the product law P_{A,X,W}^{(x)2} -- keeping it would
                                  put matched (source, anchor) tuples into the cross term and bias it.
      - pair_mode="derangement" : the O(M) Monte-Carlo objective (derivation eq. 29), using
                                  `n_derangements` independent random NONZERO cyclic shifts
                                  (derivation eq. 32):  S ~ Unif{1, ..., M-1},  pi(i) = (i + S) mod M.
                                  A nonzero cyclic shift has no fixed point, so it is a derangement,
                                  and it selects each j != i with probability 1/(M-1) -- exactly the
                                  marginal that makes eq. 29 conditionally unbiased for eq. 24
                                  (derivation Prop. 7.1).

    Returns (source_idx, anchor_idx), each of length M(M-1) or n_derangements * M.
    """
    if n_data < 2:
        raise ValueError("Ordered-pair ATT objectives need at least two observations in the batch.")

    if pair_mode == "full":
        source_idx = torch.arange(n_data, device=device).repeat_interleave(n_data)
        anchor_idx = torch.arange(n_data, device=device).repeat(n_data)
        keep = source_idx != anchor_idx
        return source_idx[keep], anchor_idx[keep]

    elif pair_mode == "derangement":
        source_list, anchor_list = [], []
        base_idx = torch.arange(n_data, device=device)
        for _ in range(n_derangements):
            shift = int(torch.randint(1, n_data, size=(1,)).item())
            source_list.append(base_idx)
            anchor_list.append((base_idx + shift) % n_data)
        return torch.cat(source_list), torch.cat(anchor_list)

    raise ValueError(f"Unknown pair_mode {pair_mode!r}; use 'full' or 'derangement'.")


class AlternativeTreatmentBridgePCLNETATT(nn.Module):
    """Density-ratio-FREE JOINT-ANCHOR ATT treatment bridge (neural mean embeddings).

    Sibling of AlternativeTreatmentBridgePCLNET: same density-ratio-free idea, but the target is the
    conditional response surface
        f_ATT(a, a') = E[ Y^(a) | A = a' ]                                        (derivation eq. 1)
    rather than the marginal dose response. The first argument a is the INTERVENTION level; the second
    argument a' indexes the FACTUAL reference population and is NOT a second intervention.

    The bridge must satisfy the ATT moment equation
        E[ varphi(a, a', X, Z) | A = a, X = x, W = w ]  =  r0_ATT(a, a', x, w)
                                                        =  p_{X,W|A}(x,w | a') / p_{X,W|A}(x,w | a),
    i.e. T_ATT varphi = r0_ATT (derivation eq. 2, 4, 5). The ratio never has to be estimated: writing the
    least-squares objective under the two laws
        P_ATT : (A_1, A_2, X_1, W_1)     Q_ATT : (A_1, A_2, X_2, W_2)             (derivation eq. 6-8)
    -- two INDEPENDENT observational units, each internally intact -- the ratio cancels analytically and
    what is left is the ordered-pair objective (derivation eq. 12, 24)
        L(g) = E[ g(A_1, A_2, X_1, W_1)^2  -  2 g(A_1, A_2, X_2, W_2) ].
    Both terms use the SAME source treatment A_1; only the (X, W) block moves from the source unit to
    the anchor unit. That asymmetry is the whole method -- see `_pair_features`.

    Neural realization (derivation eq. 18-22):
        raw bridge          varphi(a, a', x, z) = <Theta, psi_{AA'X}(a, a', x) (x) psi_Z(z)>
        first-stage CME     mu_Z(a, x, w)       = E[ psi_Z(Z) | A = a, X = x, W = w ]
        projected bridge    g(a, a', x, w)      = <Theta, psi_{AA'X}(a, a', x) (x) mu_Z(a, x, w)>
    psi_{AA'X} is ONE network over (a, a', x) -- not one network per anchor -- so a single fit gives the
    whole surface. The LOSS is a loss for the PROJECTED bridge g; the RAW bridge varphi is what the third
    stage evaluates on (a, a', x, z).

    The first stage is identical to the ATE sibling's:
        mu_Z(a, x, w) = FirstStageMap( phi_A1(a) (x) phi_W(w) [(x) phi_X1(x)] ),
    a linear map of a tensor product, refit in closed form each iteration.

    COST. Unlike the ATE bridge -- whose mismatched cross term collapses to O(N) because only the mean
    of phi_W(W) enters -- the ATT objective genuinely couples i and j inside the JOINT featurizer
    psi_{AA'X}(a_i, a_j, .), which cannot be factorized across the two indices. So the complete
    U-statistic really is O(M^2) pair evaluations. Two mitigations are implemented, both from the
    derivation: the closed-form head accumulates G_U and b_U in CHUNKS (never materializing the
    M(M-1) x d pair-feature matrix), and pair_mode="derangement" gives the O(M) unbiased alternative.
    """

    def __init__(self,
                 treatment_featurizer_first_stage: nn.Module,
                 anchor_treatment_featurizer: nn.Module,
                 treatment_proxy_featurizer: nn.Module,
                 outcome_proxy_featurizer: nn.Module,
                 backdoor_featurizer_first_stage: Optional[nn.Module] = None,
                 final_layer_first_stage_weight: Optional[torch.Tensor] = None,
                 final_layer_second_stage_weight: Optional[torch.Tensor] = None,
                 pair_mode: str = "full",
                 n_derangements: int = 1,
                 pair_chunk_size: int = 200000,
                 normalization_constraint: Optional[object] = None,
                 anchor_mode: str = "concat",
                 device: str = "cuda",
                 **kwargs,):
        super().__init__()
        self.treatment_featurizer_first_stage = treatment_featurizer_first_stage.to(device)
        # psi_{AA'X}: ONE joint network over concat(A, A', X). It absorbs the backdoor on the second
        # stage, which is why -- unlike the ATE sibling -- there is no separate second-stage X featurizer.
        # anchor_mode = "concat" (default): psi_{AA'X} is ONE joint MLP on concat(a, a', x).
        # anchor_mode = "tensor" : psi_{AA'} = psi_A(a) (x) psi_A'(a'), supplied as an nn.ModuleList
        #   [psi_A, psi_A'].  A SEPARABLE anchor feature is what makes the ordered-pair normal
        #   equations and the objective factorize to O(M) einsum contractions (see
        #   _pair_normal_equations_einsum) instead of costing O(M^2) enumerated pairs.  It is a
        #   different model class -- a finite-rank truncation of the bivariate functions the concat
        #   MLP can represent -- not a free optimisation, so it is opt-in and the default is unchanged.
        if anchor_mode not in ("concat", "tensor"):
            raise ValueError("anchor_mode must be 'concat' or 'tensor', got %r" % anchor_mode)
        self.anchor_mode = anchor_mode
        if anchor_mode == "tensor":
            if not isinstance(anchor_treatment_featurizer, (nn.ModuleList, list, tuple)) or \
                    len(anchor_treatment_featurizer) != 2:
                raise ValueError("anchor_mode='tensor' expects anchor_treatment_featurizer to be an "
                                 "nn.ModuleList([psi_A, psi_A_prime]); got %r"
                                 % type(anchor_treatment_featurizer))
            anchor_treatment_featurizer = nn.ModuleList(list(anchor_treatment_featurizer))
        self.anchor_treatment_featurizer = anchor_treatment_featurizer.to(device)
        self.treatment_proxy_featurizer = treatment_proxy_featurizer.to(device)
        self.outcome_proxy_featurizer = outcome_proxy_featurizer.to(device)
        # Optional training-time enforcement of the ATT normalisation identity E[varphi | A_1, A_2] = 1
        # (neural_causal_learning/normalization_constraint.ConditionalMeanConstraintATT). When None --
        # the default -- the second-stage head solve is the ordinary ridge solve, byte for byte.
        self.normalization_constraint = normalization_constraint
        self._last_pair_indices = None
        self.backdoor_featurizer_first_stage = backdoor_featurizer_first_stage
        if self.backdoor_featurizer_first_stage is not None:
            self.backdoor_featurizer_first_stage = self.backdoor_featurizer_first_stage.to(device)

        treatment_proxy_feature_dim = get_last_linear_out_features(treatment_proxy_featurizer)
        if anchor_mode == "tensor":
            # psi_{AA'} = psi_A (x) psi_A', so the anchor feature dimension is the PRODUCT of the two
            # halves' output dimensions -- that product is exactly what _anchor_feature flattens.
            anchor_treatment_feature_dim = (
                get_last_linear_out_features(anchor_treatment_featurizer[0])
                * get_last_linear_out_features(anchor_treatment_featurizer[1]))
        else:
            anchor_treatment_feature_dim = get_last_linear_out_features(anchor_treatment_featurizer)
        if final_layer_first_stage_weight is None:
            treatment_feature_first_stage_dim = get_last_linear_out_features(treatment_featurizer_first_stage)
            outcome_proxy_feature_dim = get_last_linear_out_features(outcome_proxy_featurizer)
            if backdoor_featurizer_first_stage is not None:
                backdoor_feature_first_stage_dim = get_last_linear_out_features(backdoor_featurizer_first_stage)
            else:
                backdoor_feature_first_stage_dim = 1
            final_layer_first_stage_weight = torch.zeros(treatment_proxy_feature_dim,
                                                         treatment_feature_first_stage_dim * outcome_proxy_feature_dim * backdoor_feature_first_stage_dim,
                                                         device=device)
        if final_layer_second_stage_weight is None:
            # The ATT bridge is scalar valued, so Theta is a single row: (1, D_psi * D_Z). The flat
            # feature order is (psi, Z), matching outer_prod(psi, mu).flatten -- keep it consistent
            # in the loss, the head update and both predictors.
            final_layer_second_stage_weight = torch.zeros(1,
                                                          anchor_treatment_feature_dim * treatment_proxy_feature_dim,
                                                          device=device)

        self.final_layer_first_stage_weight = final_layer_first_stage_weight
        self.final_layer_first_stage_weight.requires_grad = False
        self.final_layer_second_stage_weight = final_layer_second_stage_weight
        self.final_layer_second_stage_weight.requires_grad = False

        self.anchor_treatment_feature_dim = anchor_treatment_feature_dim
        self.treatment_proxy_feature_dim = treatment_proxy_feature_dim
        self.pair_mode = pair_mode
        self.n_derangements = n_derangements
        self.pair_chunk_size = pair_chunk_size
        self.device = device

    def forward(self, treatment, anchor_treatment, treatment_proxy, backdoor: Optional = None):
        pass

    # =========================================================================
    # First stage:  mu_Z(a, x, w) = E[ psi_Z(Z) | A = a, X = x, W = w ]
    # =========================================================================

    def _backdoor_from(self, data_tuple):
        """X out of a (A, Y, Z, W[, X], dens_ratio) batch, or None if this model has no backdoor slot.

        Length alone is NOT a safe test: a no-backdoor batch is still length 5, with dens_ratio -- not X
        -- sitting in slot 4. Gate on the model actually having a backdoor featurizer.
        """
        has_backdoor_slot = (self.backdoor_featurizer_first_stage is not None)
        return data_tuple[4] if (has_backdoor_slot and len(data_tuple) > 4) else None

    def _first_stage_features(self, treatment, outcome_proxy, backdoor: Optional = None):
        """The three PER-ROW first-stage featurizer outputs (phi_A1, phi_W, phi_X1 or None).

        They are returned separately, rather than already tensor-multiplied, because the ATT objective
        needs to recombine them ACROSS rows: the transported term pairs phi_A1 of the source unit with
        phi_W / phi_X1 of the anchor unit (derivation eq. 22).
        """
        treatment_feature_first_stage = self.treatment_featurizer_first_stage(treatment)
        outcome_proxy_feature = self.outcome_proxy_featurizer(outcome_proxy)
        backdoor_feature_first_stage = None
        if (backdoor is not None) and (self.backdoor_featurizer_first_stage is not None):
            backdoor_feature_first_stage = self.backdoor_featurizer_first_stage(backdoor)
        return treatment_feature_first_stage, outcome_proxy_feature, backdoor_feature_first_stage

    @staticmethod
    def _first_stage_design(treatment_feature_first_stage, outcome_proxy_feature,
                            backdoor_feature_first_stage=None):
        """phi_A1(a) (x) phi_W(w) [(x) phi_X1(x)], flattened -- the first-stage regression design."""
        if backdoor_feature_first_stage is None:
            return outer_prod(treatment_feature_first_stage, outcome_proxy_feature).flatten(start_dim=1)
        return outer_prod_batch(treatment_feature_first_stage, outcome_proxy_feature,
                                backdoor_feature_first_stage).flatten(start_dim=1)

    def _predict_cme(self, treatment_feature_first_stage, outcome_proxy_feature,
                     backdoor_feature_first_stage=None, first_stage_weight=None):
        """mu_Z from already-computed features, so callers can mix rows from different observations."""
        design = self._first_stage_design(treatment_feature_first_stage, outcome_proxy_feature,
                                          backdoor_feature_first_stage)
        weight = self.final_layer_first_stage_weight if first_stage_weight is None else first_stage_weight
        return torch.matmul(design, weight.T)

    def first_stage_forward(self, treatment, outcome_proxy, backdoor: Optional = None):
        return self._predict_cme(*self._first_stage_features(treatment, outcome_proxy, backdoor))

    def first_stage_loss(self, first_stage_data_tuple, loss_fn=nn.MSELoss()):
        backdoor = self._backdoor_from(first_stage_data_tuple)
        predicted_conditional_treatment_proxy_feature = self.first_stage_forward(
            first_stage_data_tuple[0], first_stage_data_tuple[3], backdoor)
        with torch.no_grad():
            treatment_proxy_feature = self.treatment_proxy_featurizer(first_stage_data_tuple[2])
        loss = loss_fn(treatment_proxy_feature, predicted_conditional_treatment_proxy_feature)
        return loss

    # =========================================================================
    # Second stage:  the ordered-pair (joint-anchor) objective
    # =========================================================================

    def _anchor_feature(self, treatment, anchor_treatment, backdoor: Optional = None):
        """psi_{AA'X}(a, a', x) -- ONE joint network on concat(a, a', x) (derivation eq. 18).

        `backdoor` is the covariate block of whichever unit owns it in the term being built: the SOURCE
        unit in the matched term u (eq. 21), the ANCHOR unit in the transported term v (eq. 22).
        """
        if self.anchor_mode == "tensor":
            if backdoor is not None:
                raise NotImplementedError(
                    "anchor_mode='tensor' is implemented for the no-backdoor case. With X the feature "
                    "is still separable (psi_A(a) (x) psi_A'(a') (x) psi_X(x), the source X sitting in "
                    "the matched term and the anchor X in the transported one), but the einsum "
                    "contractions below have not been written for it -- use anchor_mode='concat'.")
            psi_a, psi_a_prime = self.anchor_treatment_featurizer
            return outer_prod(psi_a(treatment), psi_a_prime(anchor_treatment)).flatten(start_dim=1)
        if backdoor is None:
            return self.anchor_treatment_featurizer(torch.hstack([treatment, anchor_treatment]))
        return self.anchor_treatment_featurizer(torch.hstack([treatment, anchor_treatment, backdoor]))

    def _uses_einsum_pairs(self, data_tuple) -> bool:
        """True when the O(M) factorised path applies: separable anchor feature and no backdoor."""
        return self.anchor_mode == "tensor" and self._backdoor_from(data_tuple) is None

    # ------------------------------------------------------------------ the O(M) factorised path
    def _einsum_pair_terms(self, data_tuple, first_stage_weight=None, need_gram=True):
        """Complete ordered-pair moments (i != j) WITHOUT enumerating pairs.

        With psi separable the two pair features of eq. 21/22 become, writing f_i = psi_A(a_i),
        g_j = psi_A'(a_j), mu_i = mu_Z(a_i, w_i) and mu_ij = mu_Z(a_i, w_j),

            u_ij = f_i (x) g_j (x) mu_i          [slots i and j FACTORISE]
            v_ij = f_i (x) g_j (x) mu_ij ,       mu_ij = W1 (phi_A1(a_i) (x) phi_W(w_j))

        so every sum over ordered pairs splits into a product of two O(M) moments, minus an O(M)
        diagonal correction for the excluded i == j terms.  The identity is exact (verified against
        brute-force enumeration to ~1e-16 relative error), which is what makes
        the COMPLETE U-statistic of eq. 24 affordable at every sample size -- no derangement sampling.

        Returns a dict with the pieces the head solve and the loss need.
        """
        treatment, outcome_proxy = data_tuple[0], data_tuple[3]
        n = treatment.shape[0]
        psi_a, psi_a_prime = self.anchor_treatment_featurizer
        f = psi_a(treatment)                                            # (M, F)
        g = psi_a_prime(treatment)                                      # (M, G)  same rows: a_j ranges
                                                                        #          over the SAME batch
        with torch.no_grad():
            (treatment_feature_first_stage, outcome_proxy_feature, _) = self._first_stage_features(
                treatment, outcome_proxy, None)
        W1 = (self.final_layer_first_stage_weight if first_stage_weight is None
              else first_stage_weight)                                  # (Z, S*T)
        Z_dim = W1.shape[0]
        S, T = treatment_feature_first_stage.shape[1], outcome_proxy_feature.shape[1]
        W1 = W1.reshape(Z_dim, S, T)
        phi1, phiW = treatment_feature_first_stage, outcome_proxy_feature
        mu_matched = self._predict_cme(phi1, outcome_proxy_feature, None, first_stage_weight)  # (M, Z)

        out = {"n_pairs": n * (n - 1), "f": f, "g": g, "mu": mu_matched, "W1": W1,
               "phi1": phi1, "phiW": phiW}
        if need_gram:
            # G_U = [sum_i (f_i mu_i)(f_i mu_i)^T] x [sum_j g_j g_j^T]  -  diagonal(i == j)
            src = torch.einsum("ip,ir,iP,iR->prPR", f, mu_matched, f, mu_matched)
            anc = torch.einsum("jq,jQ->qQ", g, g)
            gram = (torch.einsum("prPR,qQ->pqrPQR", src, anc)
                    - torch.einsum("ip,iq,ir,iP,iQ,iR->pqrPQR", f, g, mu_matched, f, g, mu_matched))
            P = f.shape[1] * g.shape[1] * mu_matched.shape[1]
            out["gram"] = gram.reshape(P, P)
            # b_U = sum_{i != j} v_ij, the transported CME's bilinearity absorbed into the contraction
            A_src = torch.einsum("ip,is->ps", f, phi1)
            B_anc = torch.einsum("jq,jt->qt", g, phiW)
            b_full = torch.einsum("rst,ps,qt->pqr", W1, A_src, B_anc)
            b_diag = torch.einsum("rst,pqst->pqr", W1,
                                  torch.einsum("ip,iq,is,it->pqst", f, g, phi1, phiW))
            out["cross"] = (b_full - b_diag).reshape(P, 1)
        return out

    def _einsum_pair_loss(self, data_tuple, first_stage_weight=None):
        """(square term, cross term) of eq. 24 averaged over ALL ordered pairs, in O(M).

        Theta . u_ij = <S_i, g_j> with S_iq = sum_{p,r} Theta[p,q,r] f_ip mu_ir, so the square term is
        a contraction of sum_i S_i S_i^T against sum_j g_j g_j^T (minus the diagonal) -- again no pair
        tensor is ever built, and the whole expression stays differentiable in the featurizers.
        """
        t = self._einsum_pair_terms(data_tuple, first_stage_weight, need_gram=False)
        f, g, mu, W1, phi1, phiW = t["f"], t["g"], t["mu"], t["W1"], t["phi1"], t["phiW"]
        F, G, Z = f.shape[1], g.shape[1], mu.shape[1]
        theta = self.final_layer_second_stage_weight.reshape(F, G, Z)
        n_pairs = t["n_pairs"]

        S = torch.einsum("pqr,ip,ir->iq", theta, f, mu)                 # (M, G)
        square_full = torch.einsum("iq,iQ,jq,jQ->", S, S, g, g)
        square_diag = (torch.einsum("iq,iq->i", S, g) ** 2).sum()
        square_term = (square_full - square_diag) / n_pairs

        A_src = torch.einsum("ip,is->ps", f, phi1)
        B_anc = torch.einsum("jq,jt->qt", g, phiW)
        cross_full = torch.einsum("pqr,rst,ps,qt->", theta, W1, A_src, B_anc)
        cross_diag = torch.einsum("pqr,rst,pqst->", theta, W1,
                                  torch.einsum("ip,iq,is,it->pqst", f, g, phi1, phiW))
        cross_term = -2.0 * (cross_full - cross_diag) / n_pairs
        return square_term, cross_term

    def make_pair_indices(self, n_data: int, pair_mode: Optional[str] = None,
                          n_derangements: Optional[int] = None):
        return make_ordered_pair_indices(
            n_data,
            pair_mode=self.pair_mode if pair_mode is None else pair_mode,
            n_derangements=self.n_derangements if n_derangements is None else n_derangements,
            device=self.device)

    def _pair_features(self, data_tuple, source_idx, anchor_idx, first_stage_weight=None):
        """The matched u_ij and transported v_ij pair features (derivation eq. 21, 22).

            u_ij = psi_{AA'X}(a_i, a_j, x_i) (x) mu_Z(a_i, x_i, w_i)
            v_ij = psi_{AA'X}(a_i, a_j, x_j) (x) mu_Z(a_i, x_j, w_j)

        Index roles -- this is the part that is easy to get silently wrong (derivation sec 4.1, Table 2):
          * i is the SOURCE unit and supplies the treatment a_i in BOTH terms;
          * j is the ANCHOR unit and supplies ONLY its treatment a_j in u, but its whole covariate /
            proxy block (x_j, w_j) in v;
          * so the transported CME is queried at (a_i, x_j, w_j) -- source treatment, anchor covariates.
            Querying it at (a_j, x_j, w_j) instead would transport the wrong law and silently estimate
            something that is not the ATT.
        Each unit stays internally intact: (a_j, x_j, w_j) always comes from the SAME row j.

        With no backdoor X, psi(a_i, a_j, x_i) and psi(a_i, a_j, x_j) coincide, so the joint featurizer
        is evaluated once instead of twice.
        """
        treatment, outcome_proxy = data_tuple[0], data_tuple[3]
        backdoor = self._backdoor_from(data_tuple)

        # As in the ATE sibling's second_stage_loss, the first-stage FEATURES are detached here: the
        # second-stage gradient is meant to reach psi_{AA'X} and psi_Z (the latter through the freshly
        # fit `first_stage_weight`), not the CME featurizers, which stage 1 owns.
        with torch.no_grad():
            (treatment_feature_first_stage,
             outcome_proxy_feature,
             backdoor_feature_first_stage) = self._first_stage_features(treatment, outcome_proxy, backdoor)

        treatment_source, treatment_anchor = treatment[source_idx], treatment[anchor_idx]
        backdoor_source = None if backdoor is None else backdoor[source_idx]
        backdoor_anchor = None if backdoor is None else backdoor[anchor_idx]

        anchor_feature_matched = self._anchor_feature(treatment_source, treatment_anchor, backdoor_source)
        if backdoor is None:
            anchor_feature_transported = anchor_feature_matched
        else:
            anchor_feature_transported = self._anchor_feature(treatment_source, treatment_anchor, backdoor_anchor)

        # The matched CME depends on the SOURCE row only, so it is computed once per row and gathered --
        # never at pair resolution. The transported one genuinely mixes the two rows, so its first-stage
        # design is the one pair-sized tensor here; `pair_chunk_size` is what bounds it.
        cme_matched = self._predict_cme(
            treatment_feature_first_stage, outcome_proxy_feature, backdoor_feature_first_stage,
            first_stage_weight)[source_idx]
        cme_transported = self._predict_cme(
            treatment_feature_first_stage[source_idx],                 # source treatment a_i
            outcome_proxy_feature[anchor_idx],                         # anchor proxy    w_j
            None if backdoor_feature_first_stage is None else backdoor_feature_first_stage[anchor_idx],
            first_stage_weight)

        matched_feature = outer_prod(anchor_feature_matched, cme_matched).flatten(start_dim=1)
        transported_feature = outer_prod(anchor_feature_transported, cme_transported).flatten(start_dim=1)
        return matched_feature, transported_feature

    def second_stage_loss(self, first_stage_tuple, second_stage_tuple,
                          second_stage_first_final_layer_regularizer=1e-1,
                          first_final_layer_update_type: str = "ridge"):
        """The joint-anchor ATT objective (derivation eq. 24 / eq. 30), as a function of the featurizers.

            L = mean_pairs [ (Theta . u_ij)^2 - 2 (Theta . v_ij) ]

        The proximal term lambda ||beta - beta_t||^2 of eq. 24 is dropped here for the same reason as in
        the ATE sibling: Theta is a buffer with requires_grad=False, so that term is constant for the
        featurizer gradient. It IS applied in the closed-form head update below.

        As in the ATE sibling, the first-stage head is refit on the first-stage batch inside this call
        (with the gradient flowing through psi_Z), and the SAME freshly-fit weight is used for both the
        matched and the transported CME -- otherwise the two halves of the objective would be evaluated
        under different first stages.
        """
        with torch.no_grad():
            backdoor_first = self._backdoor_from(first_stage_tuple)
            (treatment_feature_first_stage,
             outcome_proxy_feature,
             backdoor_feature_first_stage) = self._first_stage_features(
                first_stage_tuple[0], first_stage_tuple[3], backdoor_first)
            first_stage_feature = self._first_stage_design(
                treatment_feature_first_stage, outcome_proxy_feature, backdoor_feature_first_stage)
            weight_previous = self.final_layer_first_stage_weight.detach()

        treatment_proxy_feature = self.treatment_proxy_featurizer(first_stage_tuple[2])
        new_weight = fit_linear_proximal(treatment_proxy_feature, first_stage_feature, weight_previous.T,
                                         second_stage_first_final_layer_regularizer, first_final_layer_update_type)

        if self._uses_einsum_pairs(second_stage_tuple):
            square_term_loss, cross_term_loss = self._einsum_pair_loss(
                second_stage_tuple, first_stage_weight=new_weight.T)
        else:
            source_idx, anchor_idx = self.make_pair_indices(second_stage_tuple[0].shape[0])
            matched_feature, transported_feature = self._pair_features(
                second_stage_tuple, source_idx, anchor_idx, first_stage_weight=new_weight.T)
            square_term_loss = (torch.matmul(matched_feature, self.final_layer_second_stage_weight.T) ** 2).mean()
            cross_term_loss = - 2 * torch.matmul(transported_feature, self.final_layer_second_stage_weight.T).mean()
        loss = square_term_loss + cross_term_loss
        self.final_layer_first_stage_weight = new_weight.T.detach()
        return loss

    def second_stage_loss_val(self, second_stage_tuple):
        with torch.no_grad():
            if self._uses_einsum_pairs(second_stage_tuple):
                square_term_loss, cross_term_loss = self._einsum_pair_loss(second_stage_tuple)
            else:
                source_idx, anchor_idx = self.make_pair_indices(second_stage_tuple[0].shape[0])
                matched_feature, transported_feature = self._pair_features(
                    second_stage_tuple, source_idx, anchor_idx)
                square_term_loss = (torch.matmul(matched_feature, self.final_layer_second_stage_weight.T) ** 2).mean()
                cross_term_loss = - 2 * torch.matmul(transported_feature, self.final_layer_second_stage_weight.T).mean()
            loss = square_term_loss + cross_term_loss
        return loss

    def evaluate_second_stage_loss(self, data_loader: DataLoader, n_subset: Optional[int] = None):
        loss = 0.0
        counter = 0
        self.eval()
        with torch.no_grad():
            for data_tuple in data_loader:
                data_tuple = [data_tuple[i].to(self.device) for i in range(len(data_tuple))]
                if data_tuple[0].shape[0] < 2:      # a single-row tail batch has no ordered pairs
                    continue
                loss += self.second_stage_loss_val(data_tuple).item()
                counter += 1
        return loss / max(counter, 1)

    def update_final_layer_with_batch_regression(self, data_tuple, weight_regularizer: float, stage: str,
                                                 consider_prev_weight: bool = True, update_type: str = "ridge"):
        """Closed-form update of a final layer (first stage: ridge; second stage: derivation eq. 28 / 34)."""
        data_tuple = [x.to(self.device) for x in data_tuple]
        if stage == "first":
            backdoor = self._backdoor_from(data_tuple)
            first_stage_feature = self._first_stage_design(
                *self._first_stage_features(data_tuple[0], data_tuple[3], backdoor))
            treatment_proxy_feature = self.treatment_proxy_featurizer(data_tuple[2])
            weight_previous = self.final_layer_first_stage_weight.detach()
            new_weight = fit_linear_proximal(treatment_proxy_feature, first_stage_feature,
                                             int(consider_prev_weight) * (weight_previous.T),
                                             weight_regularizer, update_type)
            self.final_layer_first_stage_weight = new_weight.T

        elif stage == "second":
            if self._uses_einsum_pairs(data_tuple):
                # O(M) factorised path: the COMPLETE ordered-pair statistic of eq. 24, no sampling.
                terms = self._einsum_pair_terms(data_tuple, need_gram=True)
                gram_matrix, cross_rhs = terms["gram"], terms["cross"]
                n_pairs = terms["n_pairs"]
                # No pair indices exist in this path, so a normalisation constraint draws its own
                # derangement sample of the SAME pair law (an unbiased estimate of the same moments).
                self._last_pair_indices = None
            else:
                source_idx, anchor_idx = self.make_pair_indices(data_tuple[0].shape[0])
                # Stashed so a normalisation constraint binds the SAME empirical pair measure the head
                # is solved on, rather than an independently drawn one (the ATE sibling gets this for
                # free by building G from the very rows of the batch).
                self._last_pair_indices = (source_idx, anchor_idx)
                gram_matrix, cross_rhs = self._accumulate_pair_normal_equations(
                    data_tuple, source_idx, anchor_idx)
                n_pairs = source_idx.shape[0]
            phi_prev = self.final_layer_second_stage_weight.detach().T
            n_feature = gram_matrix.shape[0]
            # beta = (G + lambda I)^{-1} (b + lambda beta_t), with G and b the PAIR AVERAGES of eq. 26/27.
            # Both sides are scaled by n_pairs here so the expression matches the ATE sibling's style.
            A = gram_matrix + n_pairs * weight_regularizer * torch.eye(n_feature, device=self.device)
            RHS = cross_rhs + int(consider_prev_weight) * weight_regularizer * n_pairs * phi_prev
            # The solve is factored out so the normalisation constraint can replace it with the
            # corresponding KKT system WITHOUT duplicating any of the assembly above -- one code path,
            # so the two can never drift apart. With no constraint this is the ordinary ridge solve.
            if self.normalization_constraint is None:
                theta = torch.linalg.solve(A, RHS)
            else:
                theta = self.normalization_constraint.solve(self, data_tuple, A, RHS)
            self.final_layer_second_stage_weight = theta.T

    def _accumulate_pair_normal_equations(self, data_tuple, source_idx, anchor_idx):
        """G_U = sum_pairs u u^T and b_U = sum_pairs v (derivation eq. 26, 27), accumulated in CHUNKS.

        This is the "Computational profile" note of derivation sec 6.1: a naive materialization of the
        complete objective costs O(M^2 d) feature memory, but the normal equations only ever need one
        chunk of pairs at a time, so chunking removes the memory cost while leaving the result EXACT
        (it is the same sum, added in a different order).

        EXACT ONLY IN eval MODE, which is why this method forces it. The featurizers contain Dropout and
        BatchNorm1d, both of which make a row's output depend on which other rows share its forward call
        -- so in train mode the "same sum in a different order" claim is simply false, and the solved head
        would depend on pair_chunk_size (measured: ~25% relative change in G_U on a toy batch). Forcing
        eval also makes the head a deterministic ridge solve on clean features, which is what the
        closed-form update of eq. 28 is supposed to be. The previous train/eval flags are restored on the
        way out so the trainer's stage toggles keep working.
        """
        n_pairs = source_idx.shape[0]
        n_feature = self.final_layer_second_stage_weight.shape[1]
        gram_matrix = torch.zeros(n_feature, n_feature, device=self.device)
        cross_rhs = torch.zeros(n_feature, 1, device=self.device)

        featurizers = [self.treatment_featurizer_first_stage, self.anchor_treatment_featurizer,
                       self.treatment_proxy_featurizer, self.outcome_proxy_featurizer,
                       self.backdoor_featurizer_first_stage]
        training_states = [None if f is None else f.training for f in featurizers]
        for f in featurizers:
            if f is not None:
                f.train(False)

        with torch.no_grad():
            for start in range(0, n_pairs, self.pair_chunk_size):
                stop = min(start + self.pair_chunk_size, n_pairs)
                matched_feature, transported_feature = self._pair_features(
                    data_tuple, source_idx[start:stop], anchor_idx[start:stop])
                gram_matrix += torch.matmul(matched_feature.t(), matched_feature)
                cross_rhs += transported_feature.sum(dim=0, keepdim=True).T

        for f, state in zip(featurizers, training_states):
            if f is not None:
                f.train(state)
        return gram_matrix, cross_rhs

    # =========================================================================
    # Prediction
    # =========================================================================

    def predict_treatment_bridge_function(self, treatment, anchor_treatment, treatment_proxy, backdoor=None):
        """The RAW bridge varphi(a, a', x, z) = <Theta, psi_{AA'X}(a, a', x) (x) psi_Z(z)> (eq. 18).

        This -- not the projected bridge -- is what the third stage evaluates: the pseudo-outcome of
        derivation eq. 35 is Y_i varphi(A_i, A_j, X_i, Z_i), a function of the treatment PROXY Z.
        """
        anchor_feature = self._anchor_feature(treatment, anchor_treatment, backdoor)
        treatment_proxy_feature = self.treatment_proxy_featurizer(treatment_proxy)
        second_stage_feature = outer_prod(anchor_feature, treatment_proxy_feature).flatten(start_dim=1)
        return torch.matmul(second_stage_feature, self.final_layer_second_stage_weight.T)

    def predict_att_ratio(self, treatment, anchor_treatment, outcome_proxy, backdoor=None):
        """The PROJECTED bridge g = T_ATT varphi (eq. 20), whose target is r0_ATT (eq. 2, 5).

        The analogue of the ATE bridge's predict_dens_ratio: the density-ratio-free method never fits a
        ratio, but this is the quantity the second-stage objective actually regresses onto r0_ATT.
        """
        anchor_feature = self._anchor_feature(treatment, anchor_treatment, backdoor)
        cme_feature = self.first_stage_forward(treatment, outcome_proxy, backdoor)
        second_stage_feature = outer_prod(anchor_feature, cme_feature).flatten(start_dim=1)
        return torch.matmul(second_stage_feature, self.final_layer_second_stage_weight.T)

    # =========================================================================
    # Train / eval toggles (same contract as the ATE sibling)
    # =========================================================================

    def freeze_featurizer_params(self, stage: str):
        if stage == "first_stage":
            self.treatment_featurizer_first_stage.train(False)
            self.outcome_proxy_featurizer.train(False)
            if self.backdoor_featurizer_first_stage is not None:
                self.backdoor_featurizer_first_stage.train(False)
        elif stage == "second_stage":
            self.anchor_treatment_featurizer.train(False)
            self.treatment_proxy_featurizer.train(False)

    def unfreeze_featurizer_params(self, stage: str):
        if stage == "first_stage":
            self.treatment_featurizer_first_stage.train(True)
            self.outcome_proxy_featurizer.train(True)
            if self.backdoor_featurizer_first_stage is not None:
                self.backdoor_featurizer_first_stage.train(True)
        elif stage == "second_stage":
            self.anchor_treatment_featurizer.train(True)
            self.treatment_proxy_featurizer.train(True)

    def freeze_final_layer_params(self, stage: str):
        pass

    def unfreeze_final_layer_params(self, stage: str):
        pass


def train_alternative_treatment_pcl_net_att_model(
    model: nn.Module,
    first_stage_train_dataloader,
    second_stage_train_dataloader,
    stage1_optimizers: List[torch.optim.Optimizer],
    stage2_optimizers: List[torch.optim.Optimizer],
    n_epochs: int = 10,
    stage1_iter: int = 1,
    stage2_iter: int = 1,
    first_stage_final_layer_regularizer: float = 1e-2,
    second_stage_final_layer_regularizer: float = 1e-2,
    second_stage_first_final_layer_regularizer: float = 1e-2,
    consider_prev_weight: bool = True,
    loss_fn=nn.MSELoss(),
    log_per_epoch: int = 1,
    validation_dataloader=None,
    plot_loss: bool = True,
    stage1_schedulers: Optional[List[Scheduler]] = None,
    stage2_schedulers: Optional[List[Scheduler]] = None,
    **kwargs
):
    """Alternating two-stage trainer for the joint-anchor ATT bridge.

    Same loop as train_alternative_treatment_pcl_net_ate_model: stage 1 trains the CME featurizers
    {phi_A1, phi_W [, phi_X1]} and refits the first-stage head; stage 2 trains the bridge featurizers
    {psi_{AA'X}, psi_Z} against the ordered-pair objective and refits the second-stage head in closed
    form. Only the closed-form head is supported (the derivation's eq. 28 / 34 solve).
    """
    if validation_dataloader is None:
        validation_dataloader = first_stage_train_dataloader

    if plot_loss:
        pl.figure(figsize=(32, 12), dpi=80)
    device = model.device
    stage1_loss_hist = []
    stage2_loss_hist = []
    stage2_val_loss_hist = []

    for epoch in tqdm(range(n_epochs)):
        epoch_stage1_loss = 0
        counter = 0
        model.train()

        for first_stage_data_tuple, second_stage_data_tuple in zip(first_stage_train_dataloader,
                                                                   second_stage_train_dataloader):

            first_stage_data_tuple = [first_stage_data_tuple[i].to(device) for i in range(len(first_stage_data_tuple))]
            second_stage_data_tuple = [second_stage_data_tuple[i].to(device) for i in range(len(second_stage_data_tuple))]
            if second_stage_data_tuple[0].shape[0] < 2:
                continue                                  # no ordered pairs in a single-row tail batch

            model.treatment_featurizer_first_stage.train(True)
            model.outcome_proxy_featurizer.train(True)
            model.anchor_treatment_featurizer.train(False)
            model.treatment_proxy_featurizer.train(False)
            if model.backdoor_featurizer_first_stage is not None:
                model.backdoor_featurizer_first_stage.train(True)

            for ii in range(stage1_iter):
                for opt in stage1_optimizers:
                    opt.zero_grad()

                first_stage_loss = model.first_stage_loss(first_stage_data_tuple, loss_fn=loss_fn)
                first_stage_loss.backward()

                for opt in stage1_optimizers:
                    opt.step()

                with torch.no_grad():
                    model.update_final_layer_with_batch_regression(first_stage_data_tuple,
                                                                   first_stage_final_layer_regularizer, stage="first",
                                                                   consider_prev_weight=consider_prev_weight)

            epoch_stage1_loss += first_stage_loss.item()

            model.treatment_featurizer_first_stage.train(False)
            model.outcome_proxy_featurizer.train(False)
            model.anchor_treatment_featurizer.train(True)
            model.treatment_proxy_featurizer.train(True)
            if model.backdoor_featurizer_first_stage is not None:
                model.backdoor_featurizer_first_stage.train(False)

            for ii in range(stage2_iter):
                for opt in stage2_optimizers:
                    opt.zero_grad()

                second_stage_loss = model.second_stage_loss(
                    first_stage_data_tuple, second_stage_data_tuple,
                    second_stage_first_final_layer_regularizer=second_stage_first_final_layer_regularizer)
                second_stage_loss.backward()

                for opt in stage2_optimizers:
                    opt.step()

                with torch.no_grad():
                    model.update_final_layer_with_batch_regression(second_stage_data_tuple,
                                                                   second_stage_final_layer_regularizer,
                                                                   stage="second", consider_prev_weight=consider_prev_weight)
            counter += 1

        # ==========================
        # Scheduler Step (End of Epoch)
        # ==========================
        if stage1_schedulers:
            for sched in stage1_schedulers:
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if len(stage2_val_loss_hist) > 0: sched.step(stage2_val_loss_hist[-1])
                else:
                    sched.step()

        if stage2_schedulers:
            for sched in stage2_schedulers:
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if len(stage2_val_loss_hist) > 0: sched.step(stage2_val_loss_hist[-1])
                else:
                    sched.step()

        stage1_loss_hist.append(epoch_stage1_loss / max(counter, 1))

        model.eval()
        if (epoch % log_per_epoch == 0) | (epoch == n_epochs - 1):
            stage2_loss_hist.append(model.evaluate_second_stage_loss(second_stage_train_dataloader))
            stage2_val_loss_hist.append(model.evaluate_second_stage_loss(validation_dataloader))
            if plot_loss:
                pl.clf()
                pl.subplot(1, 2, 1)
                pl.plot(np.arange(epoch + 1), stage1_loss_hist, linewidth=5)
                pl.xlabel("Number of Epochs", fontsize=20)
                pl.ylabel("MSE", fontsize=20)
                pl.title("Averaged Stage 1 Loss: {}".format(stage1_loss_hist[-1]), fontsize=20)
                pl.grid()
                pl.xticks(fontsize=20)
                pl.yticks(fontsize=20)

                val_loss_x_axis = np.arange(len(stage2_val_loss_hist))
                pl.subplot(1, 2, 2)
                pl.plot(val_loss_x_axis, stage2_loss_hist, linewidth=5, label='Training Loss')
                pl.plot(val_loss_x_axis, stage2_val_loss_hist, linewidth=5, label='Validation Loss')
                pl.xlabel(f"Number of Epochs / {log_per_epoch}", fontsize=20)
                pl.ylabel(r"$\mathcal{L}_{ATT}$", fontsize=20)
                pl.title("Training Stage 2 Loss: {}\n Validation Stage 2 Loss: {}\n".format(stage2_loss_hist[-1],
                                                                                            stage2_val_loss_hist[-1]), fontsize=20)
                pl.grid()
                pl.xticks(fontsize=20)
                pl.yticks(fontsize=20)
                # Keep these INSIDE `if plot_loss:` -- outside it, a caller that asked for no plotting
                # still gets its cell wiped twice per fit, erasing anything it printed around the call.
                clear_output(wait=True)
                display(pl.gcf())

    return model.eval(), stage1_loss_hist, stage2_loss_hist


# =============================================================================================
# Third stage: the causal ATT surface  (derivation sec 9)
# =============================================================================================

def _gather_ordered_dataset_tensors(loader: DataLoader, device: str):
    """(A, Y, Z, W, X-or-None) read straight off the DATASET, in dataset order.

    The pairing is between two INDEPENDENT observational units, so it must be formed over a
    well-defined index set rather than over whatever partition a shuffled DataLoader happens to
    produce. Reading the dataset tensors directly also keeps each unit internally intact
    (derivation sec 4.1), which is the property the whole construction rests on.
    """
    dataset = loader.dataset
    backdoor = dataset.X.to(device) if getattr(dataset, "X", None) is not None else None
    return (dataset.A.to(device), dataset.Y.to(device),
            dataset.Z.to(device), dataset.W.to(device), backdoor)


def _build_att_pair_dataset(target_fn,
                            pcl_dataloader: DataLoader,
                            pcl_val_dataloader: Optional[DataLoader],
                            outcome_transformer: Optional[nn.Module],
                            input_transformer: Optional[nn.Module],
                            pair_mode: str,
                            n_derangements: int,
                            pair_chunk_size: int,
                            device: str,
                            desc_text: str) -> Tuple[TensorDataset, Optional[TensorDataset]]:
    """Shared body of the ATT third-stage creators.

    Both the treatment-only pseudo-outcome (eq. 35) and the doubly robust one (eq. 39) are regressed on
    the SAME two-dimensional input (A_i, A_j) -- the source treatment and the anchor treatment -- so the
    fitted network is the whole surface (a, a') -> f_ATT(a, a'), not a curve at one anchor.

    `target_fn(A_src, A_anc, Y_src, Z_src, W_src, W_anc, X_src, X_anc)` returns the pseudo-outcome.
    """
    if outcome_transformer is None:
        outcome_transformer = TorchIdentityTransformer()
    if input_transformer is None:
        input_transformer = TorchIdentityTransformer()

    def _process_single_loader(loader, text):
        A, Y, Z, W, X = _gather_ordered_dataset_tensors(loader, device)
        source_idx, anchor_idx = make_ordered_pair_indices(
            A.shape[0], pair_mode=pair_mode, n_derangements=n_derangements, device=device)

        input_list, target_list = [], []
        with torch.no_grad():
            for start in tqdm(range(0, source_idx.shape[0], pair_chunk_size), desc=text):
                stop = min(start + pair_chunk_size, source_idx.shape[0])
                i, j = source_idx[start:stop], anchor_idx[start:stop]
                target_list.append(target_fn(A[i], A[j], Y[i], Z[i], W[i], W[j],
                                             None if X is None else X[i],
                                             None if X is None else X[j]).cpu())
                input_list.append(torch.hstack([A[i], A[j]]).cpu())
        return torch.cat(input_list), torch.cat(target_list)

    input_train, target_train = _process_single_loader(pcl_dataloader, desc_text)
    input_train = input_transformer.fit_transform(input_train)
    target_train = outcome_transformer.fit_transform(target_train)

    third_stage_dataset = TensorDataset(input_train, target_train)
    third_stage_dataset.outcome_transformer = outcome_transformer
    third_stage_dataset.input_transformer = input_transformer

    third_stage_dataset_val = None
    if pcl_val_dataloader is not None:
        input_val, target_val = _process_single_loader(pcl_val_dataloader, desc_text + " (val)")
        third_stage_dataset_val = TensorDataset(input_transformer.transform(input_val),
                                                outcome_transformer.transform(target_val))
        # Carry the transformers on the val dataset too, so it is a drop-in for predict_att_surface /
        # apply_att_bridge_normalization rather than raising AttributeError on the fitted-only train one.
        third_stage_dataset_val.outcome_transformer = outcome_transformer
        third_stage_dataset_val.input_transformer = input_transformer
    return third_stage_dataset, third_stage_dataset_val


def create_third_stage_dataset_for_alternative_treatment_att(
    pcl_model: nn.Module,
    pcl_dataloader: DataLoader,
    pcl_val_dataloader: Optional[DataLoader] = None,
    outcome_transformer: Optional[nn.Module] = None,
    input_transformer: Optional[nn.Module] = None,
    pair_mode: str = "derangement",
    n_derangements: int = 10,
    pair_chunk_size: int = 200000,
    device: str = "cuda",
) -> Tuple[TensorDataset, Optional[TensorDataset]]:
    """Treatment-only ATT third stage: input (A_i, A_j), target Y_i varphi(A_i, A_j, X_i, Z_i).

    Addendum eq. 35-37. Its oracle conditional mean given (A_i = a, A_j = a') is f_ATT(a, a') by eq. 17,
    so a plain conditional-mean network on the two treatments recovers the surface.

    NOTE the default pairing here is the DERANGEMENT one (eq. 37), not the complete U-statistic (eq. 36),
    while the BRIDGE is fitted with the complete objective. Both have the same conditional mean
    (derivation sec 9, and Prop. 7.1 for the loss), but they differ enormously in size: the complete
    pairing would hand the final network M(M-1) rows -- ~1.8M for M = 1350 -- to estimate a smooth
    two-input surface, where n_derangements * M randomly paired rows already sample (a, a') from
    P_A (x) P_A. Pass pair_mode="full" to use eq. 36 instead.
    """
    pcl_model.eval().to(device)
    y_transformer = pcl_dataloader.dataset.transformers[1].to(device)

    def target_fn(A_src, A_anc, Y_src, Z_src, W_src, W_anc, X_src, X_anc):
        outcome_unscaled = y_transformer.inverse_transform(Y_src)
        varphi = pcl_model.predict_treatment_bridge_function(A_src, A_anc, Z_src, X_src)
        return outcome_unscaled * varphi

    return _build_att_pair_dataset(target_fn, pcl_dataloader, pcl_val_dataloader,
                                   outcome_transformer, input_transformer, pair_mode, n_derangements,
                                   pair_chunk_size, device, "Creating ATT Stage 3 Dataset")


def create_third_stage_dataset_for_dr_alternative_treatment_att(
    outcome_pcl_model: nn.Module,
    treatment_pcl_model: nn.Module,
    pcl_dataloader: DataLoader,
    pcl_val_dataloader: Optional[DataLoader] = None,
    outcome_transformer: Optional[nn.Module] = None,
    input_transformer: Optional[nn.Module] = None,
    pair_mode: str = "derangement",
    n_derangements: int = 10,
    pair_chunk_size: int = 200000,
    device: str = "cuda",
) -> Tuple[TensorDataset, Optional[TensorDataset]]:
    """One-regression doubly robust ATT third stage (derivation eq. 38, 39).

        Gamma_ij = varphi(A_i, A_j, X_i, Z_i) { Y_i - h(A_i, X_i, W_i) }  +  h(A_i, X_j, W_j)

    regressed on (A_i, A_j). Note what this is NOT: unlike the ATE code, this is not a slack added to the
    outcome-bridge estimate. The regression output IS the doubly robust ATT surface, because both halves
    of the influence function are carried in a single pseudo-outcome -- which, as the derivation notes, is
    what preserves the covariance between the residual correction and the anchor-population outcome term.

    Index roles, again the easy thing to get wrong: the residual Y - h is evaluated entirely on the
    SOURCE unit i, while the second term evaluates the outcome bridge at the SOURCE treatment A_i but the
    ANCHOR unit's covariates and proxy (X_j, W_j) -- it is the empirical version of E[h(a, X, W) | A = a'].
    """
    outcome_pcl_model.eval().to(device)
    treatment_pcl_model.eval().to(device)
    y_transformer = pcl_dataloader.dataset.transformers[1].to(device)

    def target_fn(A_src, A_anc, Y_src, Z_src, W_src, W_anc, X_src, X_anc):
        outcome_unscaled = y_transformer.inverse_transform(Y_src)
        outcome_bridge_source = y_transformer.inverse_transform(
            outcome_pcl_model.predict_outcome_bridge_function(A_src, W_src, X_src))
        outcome_bridge_anchor = y_transformer.inverse_transform(
            outcome_pcl_model.predict_outcome_bridge_function(A_src, W_anc, X_anc))
        varphi = treatment_pcl_model.predict_treatment_bridge_function(A_src, A_anc, Z_src, X_src)
        return varphi * (outcome_unscaled - outcome_bridge_source) + outcome_bridge_anchor

    return _build_att_pair_dataset(target_fn, pcl_dataloader, pcl_val_dataloader,
                                   outcome_transformer, input_transformer, pair_mode, n_derangements,
                                   pair_chunk_size, device, "Creating ATT Stage 3 DR Dataset")


def create_normalizer_dataset_for_alternative_treatment_att(
    pcl_model: nn.Module,
    pcl_dataloader: DataLoader,
    pcl_val_dataloader: Optional[DataLoader] = None,
    outcome_transformer: Optional[nn.Module] = None,
    input_transformer: Optional[nn.Module] = None,
    pair_mode: str = "derangement",
    n_derangements: int = 10,
    pair_chunk_size: int = 200000,
    device: str = "cuda",
) -> Tuple[TensorDataset, Optional[TensorDataset]]:
    """Dataset for the ATT bridge NORMALIZER D(a, a') (derivation sec 10, eq. 41).

        D(a, a') := E[ varphi(a, a', X, Z) | A = a ]        should equal 1 for every (a, a').

    This is exactly the eq.-35 pseudo-outcome with the Y_i factor dropped, regressed on the same input
    (A_i, A_j) -- "estimate D(a,a') by the same two-input final regression machinery", as the derivation
    puts it. Fit a conditional-mean network on it and hand the result to `apply_att_bridge_normalization`.
    """
    pcl_model.eval().to(device)

    def target_fn(A_src, A_anc, Y_src, Z_src, W_src, W_anc, X_src, X_anc):
        return pcl_model.predict_treatment_bridge_function(A_src, A_anc, Z_src, X_src)

    return _build_att_pair_dataset(target_fn, pcl_dataloader, pcl_val_dataloader,
                                   outcome_transformer, input_transformer, pair_mode, n_derangements,
                                   pair_chunk_size, device, "Creating ATT Normalizer Dataset")


def remove_att_bridge_normalization(pcl_model: nn.Module):
    """Undo apply_att_bridge_normalization, restoring the raw bridge. No-op if none was applied."""
    pcl_model.__dict__.pop("predict_treatment_bridge_function", None)
    pcl_model.__dict__.pop("predict_att_ratio", None)


def apply_att_bridge_normalization(pcl_model: nn.Module,
                                   normalizer_net: nn.Module,
                                   normalizer_dataset: TensorDataset,
                                   denominator_floor: float = 0.1,
                                   device: str = "cuda") -> dict:
    """Calibrate the bridge to its own normalizer: varphi^N(a,a',x,z) = varphi(a,a',x,z) / D_hat(a,a').

    Addendum sec 10: "If desired, estimate D(a,a') := E[varphi(a,a',X,Z) | A=a] by the same two-input
    final regression machinery and calibrate varphi^N = varphi / D_hat. This enforces a necessary scale
    restriction but does not repair arbitrary bridge-shape error." It is a POST-HOC calibration of a
    frozen bridge, explicitly optional, and it is a scale fix rather than a cure -- report it as such.

    WHY WRAP THE PREDICTOR rather than divide a third-stage output. NOTE this argument is an inference
    from the structure of eq. 39, not something the derivation states -- sec 10 never mentions the doubly
    robust estimator. D_hat is a function of the conditioning variables (a, a') alone, so for the
    TREATMENT-ONLY surface the two are equivalent in population:
        E[Y varphi / D_hat(a,a') | A=a, A'=a'] = E[Y varphi | A=a, A'=a'] / D_hat(a,a').
    For the DOUBLY ROBUST surface they are NOT. Eq. 39 is varphi (Y - h) + h, and only the varphi term
    may be rescaled -- dividing the fitted DR surface by D_hat would also scale the anchor-population
    outcome term h(A_i, X_j, W_j), which carries no bridge scale error and would then be biased by
    1/D_hat. Wrapping the bridge itself is what keeps every downstream consumer coherent (the same
    lesson as the Deaner `postnorm_bridge` fix for the density-ratio ATE bridge: normalizing inside one
    third stage fixes that estimator and silently breaks the others).

    Double robustness survives the calibration. If h is a valid outcome bridge then
    E[varphi(Y - h) | A_i, A_j] = 0 for ANY rescaling of varphi, so that leg is untouched; if varphi is
    a valid ATT bridge then D = 1 by eq. 41 and the calibration is a no-op. What the calibration DOES
    cost the DR estimator is variance: it injects D_hat's estimation error into a correction term whose
    population value the rescaling cannot improve.

    GUARDING D_hat. Eq. 41 says the population value is 1, and neither the bridge head nor a
    conditional-mean MLP is sign constrained, so an estimated D_hat <= 0 is evidence of estimation
    failure rather than of a real negative scale. A one-sided clamp_min would turn D_hat = -0.4 into
    +0.1, i.e. flip the bridge's sign AND amplify it 10x exactly where it is least trustworthy. Instead,
    any D_hat that is non-positive or smaller in magnitude than `denominator_floor` falls back to
    D_hat = 1 (no calibration on that cell). The clip is NOT prescribed by the derivation; it is this
    implementation's reading of the eq.-3 support condition, which sec 10 does flag as substantive.

    Idempotent: calling this twice re-wraps the RAW bridge rather than composing two calibrations.

    Returns a diagnostics dict -- Table 2 asks for the deviation of E[varphi|A=a] from 1 to be reported,
    not merely checked -- measured on the normalizer's own training pairs (so it understates the rate at
    evaluation anchors outside their support).
    """
    remove_att_bridge_normalization(pcl_model)              # idempotence: always wrap the RAW bridge
    normalizer_net = normalizer_net.to(device)
    original_bridge = pcl_model.predict_treatment_bridge_function
    original_ratio = pcl_model.predict_att_ratio
    input_transformer = copy.deepcopy(normalizer_dataset.input_transformer).to(device)
    outcome_transformer = copy.deepcopy(normalizer_dataset.outcome_transformer).to(device)

    def denominator_at(treatment, anchor_treatment):
        """D_hat(a, a'), guarded. The bridge is always called with dataset-SCALED treatments, which is
        the scale the normalizer was fitted on, so the same input transformer applies unchanged."""
        training_state = normalizer_net.training
        normalizer_net.eval()                               # the head carries dropout; keep D_hat deterministic
        with torch.no_grad():
            normalizer_input = input_transformer.transform(torch.hstack([treatment, anchor_treatment]))
            denominator = outcome_transformer.inverse_transform(normalizer_net(normalizer_input))
        normalizer_net.train(training_state)
        unusable = (denominator <= 0) | (denominator.abs() < denominator_floor)
        return torch.where(unusable, torch.ones_like(denominator), denominator), unusable

    def normalized_bridge(treatment, anchor_treatment, treatment_proxy, backdoor=None):
        varphi = original_bridge(treatment, anchor_treatment, treatment_proxy, backdoor)
        return varphi / denominator_at(treatment, anchor_treatment)[0]

    def normalized_ratio(treatment, anchor_treatment, outcome_proxy, backdoor=None):
        # The projected bridge g = T_ATT varphi inherits the same (a, a')-only rescaling, so the
        # diagnostic and the estimator keep reporting the same bridge.
        g = original_ratio(treatment, anchor_treatment, outcome_proxy, backdoor)
        return g / denominator_at(treatment, anchor_treatment)[0]

    pcl_model.predict_treatment_bridge_function = normalized_bridge
    pcl_model.predict_att_ratio = normalized_ratio

    # normalizer_dataset.tensors[0] is ALREADY input-transformed (fit_transform built it), so feed the
    # net directly here rather than going through denominator_at, which transforms its arguments.
    training_state = normalizer_net.training
    normalizer_net.eval()
    with torch.no_grad():
        raw_denominator = outcome_transformer.inverse_transform(
            normalizer_net(normalizer_dataset.tensors[0].to(device)))
    normalizer_net.train(training_state)
    unusable = (raw_denominator <= 0) | (raw_denominator.abs() < denominator_floor)
    return {"fell_back_fraction": float(unusable.float().mean()),
            "negative_fraction": float((raw_denominator <= 0).float().mean()),
            "mean_abs_deviation_from_one": float((raw_denominator - 1.0).abs().mean()),
            "denominator_min": float(raw_denominator.min()),
            "denominator_max": float(raw_denominator.max())}


def predict_att_surface(third_stage_net: nn.Module,
                        third_stage_dataset: TensorDataset,
                        treatment: torch.Tensor,
                        anchor_treatment: torch.Tensor,
                        treatment_transformer: nn.Module = TorchIdentityTransformer(),
                        device: str = "cuda") -> np.ndarray:
    """Evaluate a fitted ATT surface m(a, a') on a grid of interventions at one or more anchors.

    `treatment` is the (n_a, a_dim) grid of intervention levels a; `anchor_treatment` is either a single
    anchor (1, a_dim) -- broadcast over the grid -- or one anchor per grid point. Both are given on the
    RAW data scale and are transformed here with the same treatment transformer the bridge was trained
    with, then mapped through the third stage's own input transformer.
    """
    third_stage_net.eval()
    treatment_transformer = treatment_transformer.to(device)
    with torch.no_grad():
        treatment_transformed = treatment_transformer.transform(treatment.to(device))
        anchor_transformed = treatment_transformer.transform(anchor_treatment.to(device))
        if anchor_transformed.shape[0] == 1 and treatment_transformed.shape[0] > 1:
            anchor_transformed = anchor_transformed.expand(treatment_transformed.shape[0], -1)
        third_stage_input = third_stage_dataset.input_transformer.to(device).transform(
            torch.hstack([treatment_transformed, anchor_transformed]))
        prediction = third_stage_dataset.outcome_transformer.to(device).inverse_transform(
            third_stage_net(third_stage_input))
    return prediction.detach().cpu().numpy()
