import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Optional, Tuple, Callable
from torch.optim.lr_scheduler import _LRScheduler as Scheduler # Use abstract base class for type hint

import numpy as np
from tqdm import tqdm
import pylab as pl
from IPython.display import clear_output, display

import copy

import sys
sys.path.append("..")
from torch_utils.scalers import TorchIdentityTransformer
from torch_utils.linear_algebra import fit_linear_proximal, outer_prod, outer_prod_cross_batch, outer_prod_batch
from torch_utils.model_helpers import get_last_linear_out_features


class AlternativeTreatmentBridgePCLNET(nn.Module):
    """Density-ratio-FREE neural treatment bridge (neural analogue of the kernel KAP method,
    the DRKPV paper (doubly robust kernel proxy causal learning) Eq. 25-27).

    Solves the treatment-bridge moment equation  E[varphi(A,Z) | A, W] = r0(A,W) = p(W)p(A)/p(W,A)
    WITHOUT ever estimating the density ratio r0. The KAP least-squares loss expands to
        L(varphi) = E_{p(W,A)}[ E[varphi|W,A]^2 ]           (square, MATCHED (W_i,A_i))
                  - 2 E_W E_A[ E[varphi|W,A] ]              (cross, MISMATCHED / product p(W)p(A))
                  + lambda ||varphi||^2 ,
    in which the ratio cancels analytically. The first stage learns the conditional mean embedding
    mu_{Z|A,W} = E[phi_Z(Z)|A,W] (linear map of outer_prod(phi_A_first(A), phi_W(W))); the bridge is
    varphi(A,Z) = <Theta, phi_Z(Z) (x) phi_A2(A)>.

    Because mu_{Z|A,W} is LINEAR in phi_W(W), the all-pairs product moment first reduces EXACTLY
    to O(N): E_W[mu_{Z|A,W}] = FirstStageMap(outer_prod(phi_A_first(A), mean_W phi_W(W))). The
    complete ordered-pair U-statistic is then also exact in O(N): subtract the matched diagonal
    features from that all-pairs sum and divide by N(N-1). We use this off-diagonal reduction in
    the loss, the closed-form second-stage head, and the validation loss; no O(N^2) pair tensor is
    materialized. The bridge inherits the identity E[varphi(Z,a)|A=a]=1, so the same conditional-
    normalization (E[varphi|A]=1) trick used for the density-ratio bridge applies here.

    Drop-in surface for the shared third-stage / doubly-robust machinery:
    predict_treatment_bridge_function(A, Z, X=None), predict_dens_ratio(A, W, X=None), and the
    non-None buffer final_layer_second_stage_weight.
    """

    def __init__(self,
                 treatment_featurizer_first_stage: nn.Module,
                 treatment_featurizer_second_stage: nn.Module,
                 treatment_proxy_featurizer: nn.Module,
                 outcome_proxy_featurizer: nn.Module,
                 backdoor_featurizer_first_stage: Optional[nn.Module] = None,
                 backdoor_featurizer_second_stage: Optional[nn.Module] = None,
                 final_layer_first_stage_weight: Optional[torch.Tensor] = None,
                 final_layer_second_stage_weight: Optional[torch.Tensor] = None,
                 outcome_dim: int = 1,
                 device: str = "cuda",
                 nonnegative_final_layer: bool = False,
                 joint_first_stage_backdoor: Optional[bool] = None,
                 normalization_constraint: Optional[object] = None,
                 **kwargs,):
        super().__init__()
        self.treatment_featurizer_first_stage = treatment_featurizer_first_stage.to(device)
        self.treatment_featurizer_second_stage = treatment_featurizer_second_stage.to(device)
        self.backdoor_featurizer_first_stage = backdoor_featurizer_first_stage
        self.backdoor_featurizer_second_stage = backdoor_featurizer_second_stage
        if self.backdoor_featurizer_first_stage is not None:
            self.backdoor_featurizer_first_stage = self.backdoor_featurizer_first_stage.to(device)
        if self.backdoor_featurizer_second_stage is not None:
            self.backdoor_featurizer_second_stage = self.backdoor_featurizer_second_stage.to(device)
        self.treatment_proxy_featurizer = treatment_proxy_featurizer.to(device)
        self.outcome_proxy_featurizer = outcome_proxy_featurizer.to(device)

        treatment_proxy_feature_dim = get_last_linear_out_features(treatment_proxy_featurizer)
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
            if backdoor_featurizer_second_stage is not None:
                backdoor_feature_second_stage_dim = get_last_linear_out_features(backdoor_featurizer_second_stage)
            else:
                backdoor_feature_second_stage_dim = 1
            treatment_feature_second_stage_dim = get_last_linear_out_features(treatment_featurizer_second_stage)
            final_layer_second_stage_weight = torch.zeros(outcome_dim,
                                                          treatment_proxy_feature_dim * treatment_feature_second_stage_dim * backdoor_feature_second_stage_dim,
                                                          device=device)


        self.final_layer_first_stage_weight = final_layer_first_stage_weight
        self.final_layer_first_stage_weight.requires_grad = False
        self.final_layer_second_stage_weight = final_layer_second_stage_weight
        self.final_layer_second_stage_weight.requires_grad = False
        self.device = device
        self.nonnegative_final_layer = nonnegative_final_layer
        # JOINT first-stage backdoor mode: a single featurizer consumes concat(W, X) instead of the
        # separate phi_W (x) phi_X1 tensor slots. Inferred when the caller supplies a second-stage
        # backdoor featurizer but no first-stage one (the only sensible reading of that combination).
        if joint_first_stage_backdoor is None:
            joint_first_stage_backdoor = ((backdoor_featurizer_first_stage is None)
                                          and (backdoor_featurizer_second_stage is not None))
        self.joint_first_stage_backdoor = joint_first_stage_backdoor
        # OPTIONAL training-time enforcement of E[varphi | A = a] = 1 (see
        # neural_causal_learning/normalization_constraint.ConditionalMeanConstraint). When None -- the
        # default -- the second-stage head uses the ordinary unconstrained solve. When set, that solve is
        # replaced by its KKT system.
        if (normalization_constraint is not None) and nonnegative_final_layer:
            raise ValueError(
                "normalization_constraint and nonnegative_final_layer are incompatible: clipping Theta "
                "after a constrained solve would silently break the constraint.")
        self.normalization_constraint = normalization_constraint

    def forward(self, treatment, treatment_proxy, outcome_proxy, backdoor: Optional = None):
        pass

    def _backdoor_from(self, data_tuple):
        """X out of a (A, Y, Z, W[, X], dens_ratio) batch, or None if this model has no backdoor slot.

        Length alone is NOT a safe test: a no-backdoor batch is still length 5, with dens_ratio -- not X
        -- sitting in slot 4. Gate on the model actually having a backdoor featurizer.
        """
        has_backdoor_slot = ((self.backdoor_featurizer_first_stage is not None)
                             or (self.backdoor_featurizer_second_stage is not None))
        return data_tuple[4] if (has_backdoor_slot and len(data_tuple) > 4) else None

    def _first_stage_w_feature(self, outcome_proxy, backdoor: Optional = None):
        """Outcome-proxy first-stage feature: phi_W(W), or phi_WX(concat(W, X)) in JOINT mode."""
        if self.joint_first_stage_backdoor and (backdoor is not None):
            return self.outcome_proxy_featurizer(torch.hstack([outcome_proxy, backdoor]))
        return self.outcome_proxy_featurizer(outcome_proxy)

    def _first_stage_x_feature(self, backdoor: Optional = None):
        """Separate first-stage backdoor feature phi_X1(X), or None when there is no such slot
        (no backdoor at all, or JOINT mode where X already lives inside phi_WX)."""
        if ((backdoor is not None) and (self.backdoor_featurizer_first_stage is not None)
                and (not self.joint_first_stage_backdoor)):
            return self.backdoor_featurizer_first_stage(backdoor)
        return None

    @staticmethod
    def _j_features(w_feature, x_feature=None):
        """The j-side first-stage slots, in tensor-slot order. Three shapes, all of which keep W and X
        on the SAME side of the KAP mismatch (A is index i, (W, X) is index j), so the O(N) cross-term
        reduction survives in every one of them:
          - no backdoor             -> (phi_W(W),)
          - tensor-product backdoor -> (phi_W(W), phi_X1(X))
          - JOINT backdoor          -> (phi_WX(concat(W, X)),)
        The joint shape is the density-ratio bridge's first-stage style (one net over the conditioning
        variables). Joining A into that net instead WOULD break the reduction: A and (W, X) sit on
        opposite sides of the mismatch, so the N^2 double sum would no longer factorize.

        NOTE the featurizers are CALLED by _first_stage_w_feature / _first_stage_x_feature, not here, so
        each call site can invoke them in its own order. That ordering is load-bearing: the featurizers
        contain Dropout, so the sequence of forward calls determines the RNG draws during training.
        """
        return (w_feature,) if x_feature is None else (w_feature, x_feature)

    def _first_stage_design(self, treatment_feature_first_stage, j_features):
        """phi_A1(A) (x) [j-side slots], flattened -- the first-stage regression design matrix."""
        return outer_prod_batch(treatment_feature_first_stage, *j_features).flatten(start_dim=1)

    def _first_stage_feature(self, treatment, outcome_proxy, backdoor: Optional = None):
        """Convenience wrapper for the call sites whose featurizer order is A -> W -> X."""
        treatment_feature_first_stage = self.treatment_featurizer_first_stage(treatment)
        w_feature = self._first_stage_w_feature(outcome_proxy, backdoor)
        x_feature = self._first_stage_x_feature(backdoor)
        return self._first_stage_design(treatment_feature_first_stage,
                                        self._j_features(w_feature, x_feature))

    def first_stage_forward(self, treatment, outcome_proxy, backdoor: Optional = None):
        first_stage_feature = self._first_stage_feature(treatment, outcome_proxy, backdoor)
        predicted_conditional_treatment_proxy_feature = torch.matmul(first_stage_feature, self.final_layer_first_stage_weight.T)
        return predicted_conditional_treatment_proxy_feature

    def predict_dens_ratio(self, treatment, outcome_proxy, backdoor=None):
        """Reconstruct the density ratio r0(A,W) = E[varphi(A,Z)|A,W] = <Theta, mu_{Z|A,W} (x) phi_A2(A)>.

        The density-ratio-free bridge never *fits* a ratio, but this quantity (its moment-equation
        target) is the natural analogue and is what the shared third-stage / DR inlier filter reads.
        `backdoor` accepted for interface parity with TreatmentBridgePCLNET (unused when None).
        """
        predicted_conditional_treatment_proxy_feature = self.first_stage_forward(treatment, outcome_proxy, backdoor)
        treatment_feature_second_stage = self.treatment_featurizer_second_stage(treatment)
        if (backdoor is not None) and (self.backdoor_featurizer_second_stage is not None):
            # Backdoor X: bridge feature is the 3-way tensor product mu (x) phi_A2 (x) phi_X2 (order Z,A2,X2).
            backdoor_feature_second_stage = self.backdoor_featurizer_second_stage(backdoor)
            second_stage_feature = outer_prod_batch(
                predicted_conditional_treatment_proxy_feature,
                treatment_feature_second_stage,
                backdoor_feature_second_stage
            ).flatten(start_dim=1)
        else:
            second_stage_feature = outer_prod(
                predicted_conditional_treatment_proxy_feature,
                treatment_feature_second_stage
            ).flatten(start_dim = 1)
        pred = torch.matmul(second_stage_feature, self.final_layer_second_stage_weight.T)
        return pred

    def predict_treatment_bridge_function(self, treatment, treatment_proxy, backdoor=None):
        """Evaluate the treatment bridge varphi(A,Z) = <Theta, phi_Z(Z) (x) phi_A2(A)>.

        NOTE (interface parity with TreatmentBridgePCLNET): the second argument is the TREATMENT
        PROXY Z (not W). The feature ordering (phi_Z first, phi_A2 second) matches the second-stage
        loss / closed-form head so Theta = final_layer_second_stage_weight is consistent everywhere.
        """
        phi_Z = self.treatment_proxy_featurizer(treatment_proxy)
        phi_A2 = self.treatment_featurizer_second_stage(treatment)
        if (backdoor is not None) and (self.backdoor_featurizer_second_stage is not None):
            # Backdoor X: varphi(A,Z,X) = <Theta, phi_Z (x) phi_A2 (x) phi_X2> (order Z,A2,X2).
            phi_X2 = self.backdoor_featurizer_second_stage(backdoor)
            second_stage_feature = outer_prod_batch(phi_Z, phi_A2, phi_X2).flatten(start_dim=1)
        else:
            second_stage_feature = outer_prod(phi_Z, phi_A2).flatten(start_dim=1)
        return torch.matmul(second_stage_feature, self.final_layer_second_stage_weight.T)

    def _cross_rhs_backdoor(self, phi_A1, j_features, phi_A2, phi_X2, first_stage_weight_ZxFirst):
        """Exact O(N) mismatched-cross RHS for the backdoor-X case.

        With backdoor X the density ratio is r0 = p(A)/p(A|W,X), so the KAP cross term is
            E_A E_{W,X}[ mu_{Z|A,W,X} (x) phi_A2(A) (x) phi_X2(X) ]
        with A drawn from its MARGINAL (index i) and (W,X) drawn TOGETHER from their JOINT (index j).
        Naively O(N^2) over mismatched (A_i,(W,X)_j) pairs, but because mu is MULTILINEAR (the
        first-stage map W1 is linear in phi_A1 and in the j-side features), the double sum FACTORIZES
        into a contraction of two O(N) moments -- A_i and (W,X)_j sit in separate tensor slots:
            AM[p,c] = (1/N) sum_i phi_A1(A_i)[p] phi_A2(A_i)[c]     # (D_A1, D_A2), over i
        and, depending on the first-stage mode (see _j_features):
          - tensor-product backdoor, j_features = (phi_W, phi_X1):
                b[z,c,d] = sum_{p,r,s} W1[z,p,r,s] * AM[p,c] * BM[r,s,d]
                BM[r,s,d] = (1/N) sum_j phi_W[r] phi_X1[s] phi_X2[d]        # (D_W, D_X1, D_X2)
                W1 reshaped (D_Z, D_A1, D_W, D_X1)
          - JOINT backdoor, j_features = (phi_WX,):
                b[z,c,d] = sum_{p,q} W1[z,p,q] * AM[p,c] * BM[q,d]
                BM[q,d]  = (1/N) sum_j phi_WX[q] phi_X2[d]                  # (D_WX, D_X2)
                W1 reshaped (D_Z, D_A1, D_WX)
        Both are EXACT: joining W and X keeps them on the same side of the mismatch, so only the rank
        of the j-side moment changes (3 -> 2). The flat order of b is (Z,A2,X2), matching
        outer_prod_batch(mu, phi_A2, phi_X2).flatten and the second-stage weight Theta. Returns a
        (1, D_Z*D_A2*D_X2) row (the E_A E_{W,X} expectation).
        """
        N = phi_A1.shape[0]
        D_Z = first_stage_weight_ZxFirst.shape[0]
        D_A1 = phi_A1.shape[1]
        AM = torch.einsum('np,nc->pc', phi_A1, phi_A2) / N
        if len(j_features) == 1:
            phi_WX = j_features[0]
            W1_3d = first_stage_weight_ZxFirst.reshape(D_Z, D_A1, phi_WX.shape[1])
            BM = torch.einsum('nq,nd->qd', phi_WX, phi_X2) / N
            cross = torch.einsum('zpq,pc,qd->zcd', W1_3d, AM, BM)     # (D_Z, D_A2, D_X2)
        else:
            phi_W, phi_X1 = j_features
            W1_4d = first_stage_weight_ZxFirst.reshape(D_Z, D_A1, phi_W.shape[1], phi_X1.shape[1])
            BM = torch.einsum('nr,ns,nd->rsd', phi_W, phi_X1, phi_X2) / N
            cross = torch.einsum('zprs,pc,rsd->zcd', W1_4d, AM, BM)   # (D_Z, D_A2, D_X2)
        return cross.reshape(1, -1)

    @staticmethod
    def _complete_u_stat_cross_rhs(product_cross_rhs, matched_feature):
        """Convert the factorized all-pairs product moment to the complete off-diagonal U-statistic.

        Let v_ij be the transported second-stage feature with the treatment-side slots from row i and
        the proxy/backdoor-side slots from row j. ``product_cross_rhs`` is the existing exact O(N)
        factorization of N^{-2} sum_{i,j} v_ij, while ``matched_feature[i]`` is exactly v_ii. Hence

            [N(N-1)]^{-1} sum_{i != j} v_ij
              = product_cross_rhs
                + (product_cross_rhs - mean_i matched_feature[i]) / (N - 1).

        This identity is shared by the no-backdoor, tensor-product-backdoor, and joint-WX paths. The
        correction must remain in the autograd graph when called from ``second_stage_loss``. This is
        the complete U-statistic on the current batch; averaging several batch losses is a stochastic
        mini-batch objective, not the complete U-statistic over the loader's entire dataset.
        """
        n_data = matched_feature.shape[0]
        if n_data < 2:
            raise ValueError(
                "The complete off-diagonal U-statistic requires a second-stage batch with at least "
                "two observations. Merge/resize a singleton final batch, or use drop_last=True for "
                "training loaders."
            )
        diagonal_mean = matched_feature.mean(dim=0, keepdim=True)
        return product_cross_rhs + (product_cross_rhs - diagonal_mean) / (n_data - 1)

    def update_final_layer_with_batch_regression(self, data_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                                                 weight_regularizer: float, stage: str, consider_prev_weight: bool = True, update_type: str = "ridge",
                                                 ):
        """
        Update final layer weights using linear regression on batch data.
        """
        data_tuple = [x.to(self.device) for x in data_tuple]
        if stage == "first":
            backdoor = self._backdoor_from(data_tuple)
            # Featurizer call order W -> A -> X is deliberate (dropout RNG order; see _j_features).
            w_feature = self._first_stage_w_feature(data_tuple[3], backdoor)
            treatment_feature_first_stage = self.treatment_featurizer_first_stage(data_tuple[0])
            x_feature = self._first_stage_x_feature(backdoor)
            first_stage_feature = self._first_stage_design(
                treatment_feature_first_stage, self._j_features(w_feature, x_feature))
            treatment_proxy_feature = self.treatment_proxy_featurizer(data_tuple[2])
            weight_previous = self.final_layer_first_stage_weight.detach()
            new_weight = fit_linear_proximal(treatment_proxy_feature, first_stage_feature, int(consider_prev_weight) * (weight_previous.T), weight_regularizer, update_type)
            self.final_layer_first_stage_weight = new_weight.T

        elif stage == "second":
            # ProxyDataset batches are (A, Y, Z, W[, X], dens_ratio); index rather than unpack.
            treatment, outcome_proxy = data_tuple[0], data_tuple[3]
            has_bd = (self.backdoor_featurizer_second_stage is not None) and (len(data_tuple) > 4)
            backdoor = self._backdoor_from(data_tuple)
            with torch.no_grad():
                treatment_feature_first_stage = self.treatment_featurizer_first_stage(treatment)
                j_features = self._j_features(self._first_stage_w_feature(outcome_proxy, backdoor),
                                              self._first_stage_x_feature(backdoor))
                first_stage_feature = self._first_stage_design(treatment_feature_first_stage, j_features)
            treatment_feature_second_stage = self.treatment_featurizer_second_stage(treatment)
            predicted_conditional_treatment_proxy_feature = torch.matmul(first_stage_feature, self.final_layer_first_stage_weight.T)

            if has_bd:
                # MATCHED (square) design with X: U_i = mu_{Z|A_i,W_i,X_i} (x) phi_A2(A_i) (x) phi_X2(X_i)
                backdoor_feature_second_stage = self.backdoor_featurizer_second_stage(backdoor)
                matched_feature = outer_prod_batch(
                    predicted_conditional_treatment_proxy_feature,
                    treatment_feature_second_stage,
                    backdoor_feature_second_stage
                ).flatten(start_dim=1)
                # MISMATCHED cross RHS via the exact O(N) tensor-moment contraction (see _cross_rhs_backdoor).
                cross_rhs = self._cross_rhs_backdoor(
                    treatment_feature_first_stage, j_features,
                    treatment_feature_second_stage, backdoor_feature_second_stage,
                    self.final_layer_first_stage_weight)  # (1, D_Z*D_A2*D_X2)
            else:
                # MATCHED (square) design: U_i = mu_{Z|A_i,W_i} (x) phi_A2(A_i)
                matched_feature = outer_prod(
                    predicted_conditional_treatment_proxy_feature,
                    treatment_feature_second_stage
                ).flatten(start_dim=1)
                # MISMATCHED (cross) RHS via the exact O(N) mean-W reduction (see second_stage_loss):
                # v_bar_i = mu_{Z|A_i, mean_W W} (x) phi_A2(A_i).
                outcome_proxy_feature_mean = j_features[0].mean(0, keepdim=True)
                outer_prod_cross_terms = outer_prod_cross_batch(
                    treatment_feature_first_stage, outcome_proxy_feature_mean).flatten(start_dim=1)
                predicted_conditional_treatment_proxy_feature_cross = torch.matmul(
                    outer_prod_cross_terms, self.final_layer_first_stage_weight.T)
                second_stage_feature_cross = outer_prod(
                    predicted_conditional_treatment_proxy_feature_cross,
                    treatment_feature_second_stage
                ).flatten(start_dim=1)
                cross_rhs = second_stage_feature_cross.mean(axis=0, keepdim=True)  # (1, D_Z*D_A2)

            # The factorized contractions above are all-pairs empirical-product (V-statistic) moments.
            # Remove their matched diagonal exactly, retaining O(N) computation and sample-axis memory,
            # so the head solves the complete ordered-pair U-statistic objective on this batch.
            cross_rhs = self._complete_u_stat_cross_rhs(cross_rhs, matched_feature)

            phi_prev = self.final_layer_second_stage_weight.detach().T
            nData, nFeat = matched_feature.size()
            # A = sum_i U_i U_i^T + N*lambda*I; RHS = N*b_U + proximal; theta = A^{-1} RHS.
            A = torch.matmul(matched_feature.t(), matched_feature) + nData * weight_regularizer * torch.eye(
                nFeat, device=matched_feature.device, dtype=matched_feature.dtype)
            RHS = cross_rhs.T * nData + int(consider_prev_weight) * weight_regularizer * nData * phi_prev
            # The solve is factored out so the normalisation constraint can replace it with the
            # corresponding KKT system WITHOUT duplicating any of the assembly above -- one code path,
            # so the two can never drift apart.
            #
            # WHAT THIS SOLVE ACTUALLY IS (verified; it is NOT an ordinary ridge solve).
            # nData cancels out of both sides, so with consider_prev_weight=True this is
            #     theta_t = (Ghat + lam I)^{-1} (b + lam theta_{t-1}),   Ghat = (1/N) sum_i U_i U_i^T,
            # a PROXIMAL-POINT iteration whose fixed point satisfies Ghat theta* = b -- the
            # UNREGULARISED normal equations. lam therefore does not regularise the solution at all; it
            # sets the contraction rate lam/(g+lam) per eigendirection. The estimator's real regulariser
            # is EARLY STOPPING, with an effective ridge of roughly lam/T after T solves (the filter
            # factor is 1-(lam/(g+lam))^T). CONSEQUENCE: `reg2` and `n_epochs` are the same knob, and
            # since the head is re-solved once per stage-2 MINI-BATCH, T grows with the sample size at a
            # fixed batch size -- so raising n silently REMOVES regularisation. Experiments whose batch
            # covers the whole stage-2 fold (batch = n) have T = n_epochs and are unaffected; dSprite,
            # which uses batch 1024 over folds of 1500/3000/6000, is not. Setting consider_prev_weight
            # False makes it a genuine ridge solve.
            if self.normalization_constraint is None:
                theta = torch.linalg.solve(A, RHS)
            else:
                theta = self.normalization_constraint.solve(self, data_tuple, A, RHS)
            if self.nonnegative_final_layer:
                self.final_layer_second_stage_weight = torch.clip(theta.T, min = 0.0, max = None)
            else:
                self.final_layer_second_stage_weight = theta.T

    def first_stage_loss(self, first_stage_data_tuple, loss_fn = nn.MSELoss()):
        backdoor = self._backdoor_from(first_stage_data_tuple)
        first_stage_feature = self._first_stage_feature(
            first_stage_data_tuple[0], first_stage_data_tuple[3], backdoor)
        predicted_conditional_treatment_proxy_feature = torch.matmul(first_stage_feature, self.final_layer_first_stage_weight.T)
        with torch.no_grad():
            treatment_proxy_feature = self.treatment_proxy_featurizer(first_stage_data_tuple[2])
        loss = loss_fn(treatment_proxy_feature, predicted_conditional_treatment_proxy_feature)
        return loss

    def second_stage_loss(self, first_stage_tuple, second_stage_tuple,
                          second_stage_first_final_layer_regularizer = 1e-1, first_final_layer_update_type: str = "ridge"):
        ## First stage final layer weight update:
        with torch.no_grad():
            backdoor_first = self._backdoor_from(first_stage_tuple)
            # Featurizer call order W -> A -> X is deliberate (dropout RNG order; see _j_features).
            w_feature = self._first_stage_w_feature(first_stage_tuple[3], backdoor_first)
            treatment_feature_first_stage = self.treatment_featurizer_first_stage(first_stage_tuple[0])
            x_feature = self._first_stage_x_feature(backdoor_first)
            first_stage_feature = self._first_stage_design(
                treatment_feature_first_stage, self._j_features(w_feature, x_feature))

            weight_previous = self.final_layer_first_stage_weight.detach()

        treatment_proxy_feature = self.treatment_proxy_featurizer(first_stage_tuple[2])
        new_weight = fit_linear_proximal(treatment_proxy_feature, first_stage_feature, weight_previous.T,
                                         second_stage_first_final_layer_regularizer, first_final_layer_update_type)

        has_bd = (self.backdoor_featurizer_second_stage is not None) and (len(second_stage_tuple) > 4)
        backdoor_second = self._backdoor_from(second_stage_tuple)
        with torch.no_grad():
            treatment_feature_first_stage = self.treatment_featurizer_first_stage(second_stage_tuple[0])
            j_features = self._j_features(
                self._first_stage_w_feature(second_stage_tuple[3], backdoor_second),
                self._first_stage_x_feature(backdoor_second))
            first_stage_feature = self._first_stage_design(treatment_feature_first_stage, j_features)
        treatment_feature_second_stage = self.treatment_featurizer_second_stage(second_stage_tuple[0])
        predicted_conditional_treatment_proxy_feature = torch.matmul(first_stage_feature, new_weight)

        if has_bd:
            # MATCHED (square) with X: U_i = mu_{Z|A_i,W_i,X_i} (x) phi_A2(A_i) (x) phi_X2(X_i)
            backdoor_feature_second_stage_2 = self.backdoor_featurizer_second_stage(backdoor_second)
            matched_feature = outer_prod_batch(
                predicted_conditional_treatment_proxy_feature,
                treatment_feature_second_stage,
                backdoor_feature_second_stage_2
            ).flatten(start_dim=1)
            square_term_loss = (torch.matmul(matched_feature, self.final_layer_second_stage_weight.T) ** 2).mean()
            # MISMATCHED cross RHS via the exact O(N) tensor-moment contraction (uses the SAME freshly-fit
            # first-stage weight new_weight.T = (D_Z, D_first) as the square term). See _cross_rhs_backdoor.
            cross_rhs = self._cross_rhs_backdoor(
                treatment_feature_first_stage, j_features,
                treatment_feature_second_stage, backdoor_feature_second_stage_2, new_weight.T)  # (1, D)
            cross_rhs = self._complete_u_stat_cross_rhs(cross_rhs, matched_feature)
            second_stage_result_cross = torch.matmul(cross_rhs, self.final_layer_second_stage_weight.T)  # (1, outcome_dim)
            cross_term_loss = - 2 * second_stage_result_cross.mean()
        else:
            matched_feature = outer_prod(
                predicted_conditional_treatment_proxy_feature,
                treatment_feature_second_stage
            ).flatten(start_dim=1)
            square_term_loss = (
                torch.matmul(matched_feature, self.final_layer_second_stage_weight.T) ** 2
            ).mean()

            # Cross term over the PRODUCT p(W)p(A) -- the density-ratio-free content (KAP Eq. 25/26). Naively
            # an O(N^2) double sum over MISMATCHED (A_i, W_j) pairs, but the first-stage CME is LINEAR in
            # phi_W(W), so E_W[mu_{Z|A,W}] = FirstStageMap(outer_prod(phi_A_first(A), mean_W phi_W(W))). Hence
            # the cross term reduces EXACTLY to O(N) (mean over the W batch), using the SAME freshly-fit
            # first-stage weight (new_weight) as the square term.
            outcome_proxy_feature_mean = j_features[0].mean(0, keepdim=True)  # (1, D_W)
            outer_prod_cross_terms = outer_prod_cross_batch(
                treatment_feature_first_stage, outcome_proxy_feature_mean).flatten(start_dim=1)  # (N, D_Afirst*D_W)
            predicted_conditional_treatment_proxy_feature_cross = torch.matmul(outer_prod_cross_terms, new_weight)  # (N, D_Z)
            second_stage_feature_cross = outer_prod(
                predicted_conditional_treatment_proxy_feature_cross, treatment_feature_second_stage).flatten(start_dim=1)
            cross_rhs = second_stage_feature_cross.mean(dim=0, keepdim=True)
            cross_rhs = self._complete_u_stat_cross_rhs(cross_rhs, matched_feature)
            second_stage_result_cross = torch.matmul(cross_rhs, self.final_layer_second_stage_weight.T)
            cross_term_loss = - 2 * second_stage_result_cross.mean()
        loss = square_term_loss + cross_term_loss
        self.final_layer_first_stage_weight = new_weight.T.detach()
        return loss

    def second_stage_loss_val(self, second_stage_tuple,):
        has_bd = (self.backdoor_featurizer_second_stage is not None) and (len(second_stage_tuple) > 4)
        backdoor = self._backdoor_from(second_stage_tuple)
        with torch.no_grad():
            # Featurizer call order A1 -> W -> A2 -> X1 -> X2 is deliberate (dropout RNG order).
            treatment_feature_first_stage = self.treatment_featurizer_first_stage(second_stage_tuple[0])
            w_feature = self._first_stage_w_feature(second_stage_tuple[3], backdoor)
            treatment_feature_second_stage = self.treatment_featurizer_second_stage(second_stage_tuple[0])
            x_feature = self._first_stage_x_feature(backdoor)
            j_features = self._j_features(w_feature, x_feature)
            first_stage_feature = self._first_stage_design(treatment_feature_first_stage, j_features)
            predicted_conditional_treatment_proxy_feature = torch.matmul(first_stage_feature, self.final_layer_first_stage_weight.T)
            if has_bd:
                backdoor_feature_second_stage = self.backdoor_featurizer_second_stage(backdoor)
                matched_feature = outer_prod_batch(
                    predicted_conditional_treatment_proxy_feature,
                    treatment_feature_second_stage,
                    backdoor_feature_second_stage
                ).flatten(start_dim=1)
                square_term_loss = (torch.matmul(matched_feature, self.final_layer_second_stage_weight.T) ** 2).mean()
                cross_rhs = self._cross_rhs_backdoor(
                    treatment_feature_first_stage, j_features,
                    treatment_feature_second_stage, backdoor_feature_second_stage,
                    self.final_layer_first_stage_weight)  # (1, D)
                cross_rhs = self._complete_u_stat_cross_rhs(cross_rhs, matched_feature)
                second_stage_result_cross = torch.matmul(cross_rhs, self.final_layer_second_stage_weight.T)
                cross_term_loss = - 2 * second_stage_result_cross.mean()
            else:
                matched_feature = outer_prod(
                    predicted_conditional_treatment_proxy_feature,
                    treatment_feature_second_stage
                ).flatten(start_dim=1)
                square_term_loss = (
                    torch.matmul(matched_feature, self.final_layer_second_stage_weight.T) ** 2
                ).mean()

                # Cross term via the exact O(N) mean-W reduction (see second_stage_loss).
                outcome_proxy_feature_mean = j_features[0].mean(0, keepdim=True)
                outer_prod_cross_terms = outer_prod_cross_batch(
                    treatment_feature_first_stage, outcome_proxy_feature_mean).flatten(start_dim=1)
                predicted_conditional_treatment_proxy_feature_cross = torch.matmul(
                    outer_prod_cross_terms, self.final_layer_first_stage_weight.T)
                second_stage_feature_cross = outer_prod(
                    predicted_conditional_treatment_proxy_feature_cross,
                    treatment_feature_second_stage
                ).flatten(start_dim=1)
                cross_rhs = second_stage_feature_cross.mean(dim=0, keepdim=True)
                cross_rhs = self._complete_u_stat_cross_rhs(cross_rhs, matched_feature)
                second_stage_result_cross = torch.matmul(cross_rhs, self.final_layer_second_stage_weight.T)
                cross_term_loss = - 2 * second_stage_result_cross.mean()
            loss = square_term_loss + cross_term_loss
        return loss

    def evaluate_second_stage_loss(self, data_loader: DataLoader, n_subset: Optional[int] = None):
        """Average complete within-batch U-statistic losses over a loader.

        For a multi-batch loader this is the stochastic mini-batch objective, not one complete
        off-diagonal U-statistic formed from every pair in the full dataset.
        """
        loss = 0.0
        self.eval()
        with torch.no_grad():
            for data_tuple in data_loader:
                data_tuple = [data_tuple[i].to(self.device) for i in range(len(data_tuple))]
                batch_size = data_tuple[0].shape[0]
                batch_loss = self.second_stage_loss_val(data_tuple )
                loss += batch_loss.item() * batch_size
        loss /= len(data_loader.dataset)
        return loss

    def freeze_featurizer_params(self, stage: str):
        if stage == "first_stage":
            self.treatment_featurizer_first_stage.train(False)
            self.outcome_proxy_featurizer.train(False)
            if (self.backdoor_featurizer_first_stage is not None):
                self.backdoor_featurizer_first_stage.train(False)

        elif stage == "second_stage":
            self.treatment_featurizer_second_stage.train(False)
            self.treatment_proxy_featurizer.train(False)
            if (self.backdoor_featurizer_second_stage is not None):
                self.backdoor_featurizer_second_stage.train(False)

    def unfreeze_featurizer_params(self, stage: str):
        if stage == "first_stage":
            self.treatment_featurizer_first_stage.train(True)
            self.outcome_proxy_featurizer.train(True)
            if (self.backdoor_featurizer_first_stage is not None):
                self.backdoor_featurizer_first_stage.train(True)

        elif stage == "second_stage":
            self.treatment_featurizer_second_stage.train(True)
            self.treatment_proxy_featurizer.train(True)
            if (self.backdoor_featurizer_second_stage is not None):
                self.backdoor_featurizer_second_stage.train(True)

    def freeze_final_layer_params(self, stage: str):
        pass

    def unfreeze_final_layer_params(self, stage: str):
        pass


def train_alternative_treatment_pcl_net_ate_model(
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
    loss_fn = nn.MSELoss(),
    log_per_epoch: int = 1,
    validation_dataloader = None,
    plot_loss: bool = True,
    stage1_schedulers: Optional[List[Scheduler]] = None,
    stage2_schedulers: Optional[List[Scheduler]] = None,
    head_mode: str = "closed_form",     # "closed_form" (ridge solve) or "sgd" (L-BFGS steps on the head)
    second_stage_head_lr: float = 1.0,
    second_stage_head_steps: int = 10,
    **kwargs
):
    # Mirror the sibling trainers: fall back to the training loader so the end-of-run evaluation
    # (which fires on the last epoch regardless of log_per_epoch) does not crash on None.
    if validation_dataloader is None:
        validation_dataloader = first_stage_train_dataloader

    if head_mode != "closed_form":
        raise NotImplementedError(
            f"head_mode={head_mode!r} is no longer supported: the density-ratio-free bridge uses the "
            "closed-form ridge solve for its second-stage head. The L-BFGS variant is kept, commented "
            "out, at the bottom of this module (it did not work well); revive it there if needed.")

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

            model.treatment_featurizer_first_stage.train(True)
            model.outcome_proxy_featurizer.train(True)
            model.treatment_featurizer_second_stage.train(False)
            model.treatment_proxy_featurizer.train(False)
            if model.backdoor_featurizer_first_stage is not None:
                model.backdoor_featurizer_first_stage.train(True)
            if model.backdoor_featurizer_second_stage is not None:
                model.backdoor_featurizer_second_stage.train(False)

            for ii in range(stage1_iter):
                for opt in stage1_optimizers:
                    opt.zero_grad()

                ## First stage
                for opt in stage1_optimizers:
                    opt.zero_grad()

                first_stage_loss = model.first_stage_loss(first_stage_data_tuple, loss_fn = loss_fn)
                first_stage_loss.backward()

                for opt in stage1_optimizers:
                    opt.step()

                with torch.no_grad():
                    model.update_final_layer_with_batch_regression(first_stage_data_tuple,
                                                                first_stage_final_layer_regularizer, stage = "first",
                                                                consider_prev_weight = consider_prev_weight)

            epoch_stage1_loss += first_stage_loss.item()

            model.treatment_featurizer_first_stage.train(False)
            model.outcome_proxy_featurizer.train(False)
            model.treatment_featurizer_second_stage.train(True)
            model.treatment_proxy_featurizer.train(True)
            if model.backdoor_featurizer_first_stage is not None:
                model.backdoor_featurizer_first_stage.train(False)
            if model.backdoor_featurizer_second_stage is not None:
                model.backdoor_featurizer_second_stage.train(True)

            ## Second stage
            for ii in range(stage2_iter):
                for opt in stage2_optimizers:
                    opt.zero_grad()

                second_stage_loss = model.second_stage_loss(first_stage_data_tuple, second_stage_data_tuple,
                                                            second_stage_first_final_layer_regularizer = second_stage_first_final_layer_regularizer)
                second_stage_loss.backward()

                for opt in stage2_optimizers:
                    opt.step()

                with torch.no_grad():
                    model.update_final_layer_with_batch_regression(second_stage_data_tuple,
                                                                   second_stage_final_layer_regularizer,
                                                                   stage = "second", consider_prev_weight = consider_prev_weight)
            counter += 1
        # ==========================
        # Scheduler Step (End of Epoch)
        # ==========================
        if stage1_schedulers:
            for sched in stage1_schedulers:
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Step with validation metric if available
                    if len(stage2_val_loss_hist) > 0: sched.step(stage2_val_loss_hist[-1])
                else:
                    sched.step()

        if stage2_schedulers:
            for sched in stage2_schedulers:
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if len(stage2_val_loss_hist) > 0: sched.step(stage2_val_loss_hist[-1])
                else:
                    sched.step()

        stage1_loss_hist.append(epoch_stage1_loss / counter)

        model.eval()
        if (epoch % log_per_epoch == 0) | (epoch == n_epochs - 1):
            stage2_loss_hist.append(model.evaluate_second_stage_loss(second_stage_train_dataloader, ))
            stage2_val_loss_hist.append(model.evaluate_second_stage_loss(validation_dataloader,))
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
                pl.plot(val_loss_x_axis, stage2_loss_hist, linewidth=5, label = 'Training Loss')
                pl.plot(val_loss_x_axis, stage2_val_loss_hist, linewidth=5, label = 'Validation Loss')
                pl.xlabel(f"Number of Epochs / {log_per_epoch}", fontsize=20)
                pl.ylabel(r"$\mathcal{L}_2$", fontsize=20)
                pl.title("Training Stage 2 Loss: {}\n Validation Stage 2 Loss: {}\n".format(stage2_loss_hist[-1],
                                                                                            stage2_val_loss_hist[-1]), fontsize=20)
                pl.grid()
                pl.xticks(fontsize=20)
                pl.yticks(fontsize=20)
                # Inside `if plot_loss:` -- matching the outcome trainer. Outside it, a caller that
                # asked for NO plotting still gets its cell wiped twice per fit (the enclosing guard
                # fires on epoch 0 and on the last epoch whatever log_per_epoch is), which silently
                # erases anything the caller printed around the call -- e.g. a cross-fitting loop's
                # own per-fold log.
                clear_output(wait=True)
                display(pl.gcf())

    return model.eval(), stage1_loss_hist, stage2_loss_hist


class EnsembleAlternativeTreatmentBridge(nn.Module):
    """Average K independently-trained AlternativeTreatmentBridgePCLNET bridges.

    The density-ratio-free treatment bridge is ill-posed (components of phi_Z(Z) not predictable from
    (A,W) are only weakly constrained), so a single bridge's dose-response is high-variance. Averaging
    the bridge PREDICTIONS over K members reduces that variance. By linearity of the third-stage
    conditional mean, E[Y * mean_k varphi_k | A] = mean_k E[Y * varphi_k | A], so ensembling the
    predictions is consistent with ensembling the dose-response curves; ensembling first also yields a
    single drop-in "model" for the shared third-stage / doubly-robust machinery.

    Exposes the minimal treatment-bridge contract: predict_treatment_bridge_function(A,Z,X),
    predict_dens_ratio(A,W,X) (both = mean over members), and a non-None presence-gate buffer
    final_layer_second_stage_weight.
    """

    def __init__(self, members: List[nn.Module], device: str = "cuda"):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.device = device
        # The shared third-stage/DR code only checks this attribute is present and non-None.
        self.register_buffer("final_layer_second_stage_weight", torch.zeros(1, 1, device=device))
        self.final_layer_second_stage_weight.requires_grad = False

    def _backdoor_from(self, data_tuple):
        """Delegate to a member: all members share the same featurizer layout, so they agree here.

        Part of the single-bridge contract the third-stage creators rely on -- without it an ensemble
        cannot be swapped in for an AlternativeTreatmentBridgePCLNET.
        """
        return self.members[0]._backdoor_from(data_tuple)

    def predict_treatment_bridge_function(self, treatment, treatment_proxy, backdoor=None):
        preds = [m.predict_treatment_bridge_function(treatment, treatment_proxy, backdoor) for m in self.members]
        return torch.stack(preds, dim=0).mean(dim=0)

    def predict_dens_ratio(self, treatment, outcome_proxy, backdoor=None):
        preds = [m.predict_dens_ratio(treatment, outcome_proxy, backdoor) for m in self.members]
        return torch.stack(preds, dim=0).mean(dim=0)

    def eval(self):
        for m in self.members:
            m.eval()
        return super().eval()

    def to(self, *args, **kwargs):
        for m in self.members:
            m.to(*args, **kwargs)
        return super().to(*args, **kwargs)


def train_alternative_treatment_pcl_net_ensemble(
    build_nets_fn: Callable,
    first_stage_train_dataloader,
    second_stage_train_dataloader,
    validation_dataloader=None,
    n_members: int = 5,
    device: str = "cuda",
    base_seed: int = 0,
    lr_stage1: float = 1e-3,
    lr_stage2: float = 1e-4,
    weight_decay: float = 1e-6,
    n_epochs: int = 100,
    stage1_iter: int = 1,
    stage2_iter: int = 1,
    first_stage_final_layer_regularizer: float = 1e-2,
    second_stage_final_layer_regularizer: float = 30.0,
    second_stage_first_final_layer_regularizer: float = 1e-2,
    consider_prev_weight: bool = True,
    loss_fn=None,
    head_mode: str = "closed_form",
    second_stage_head_lr: float = 1.0,
    second_stage_head_steps: int = 10,
    normalization_constraint=None,
    **kwargs,
) -> EnsembleAlternativeTreatmentBridge:
    """Train `n_members` density-ratio-free treatment bridges (different inits) and wrap them in an
    EnsembleAlternativeTreatmentBridge. `build_nets_fn(device)` must return the four featurizers in
    AlternativeTreatmentBridgePCLNET.__init__ order:
        (treatment_featurizer_first_stage, treatment_featurizer_second_stage,
         treatment_proxy_featurizer, outcome_proxy_featurizer).
    Stage-1 optimizers train {first-stage A featurizer, outcome-proxy W featurizer}; stage-2 optimizers
    train {second-stage A featurizer, treatment-proxy Z featurizer} -- matching the trainer's toggles.

    `normalization_constraint` (None = numerically identical old behaviour) is a ConditionalMeanConstraint, or
    a zero-arg factory returning one, enforcing E[varphi | A] = 1 inside every member's closed-form head.
    """
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    members = []
    for k in range(n_members):
        torch.manual_seed(base_seed + k)
        nets = build_nets_fn(device)
        # build_nets_fn returns 4 featurizers (no backdoor) or 6 (with backdoor X: + bd_first, bd_second).
        if len(nets) == 6:
            (t_first, t_second, t_proxy, out_proxy, bd_first, bd_second) = nets
        else:
            (t_first, t_second, t_proxy, out_proxy) = nets
            bd_first, bd_second = None, None
        # One constraint object per member: pass a zero-arg FACTORY when the constraint caches
        # anything from its first batch (rbf centres, indicator levels) so members cannot share state.
        member_constraint = (normalization_constraint() if callable(normalization_constraint)
                             and not hasattr(normalization_constraint, "system")
                             else normalization_constraint)
        model = AlternativeTreatmentBridgePCLNET(
            treatment_featurizer_first_stage=t_first, treatment_featurizer_second_stage=t_second,
            treatment_proxy_featurizer=t_proxy, outcome_proxy_featurizer=out_proxy,
            backdoor_featurizer_first_stage=bd_first, backdoor_featurizer_second_stage=bd_second,
            device=device, normalization_constraint=member_constraint)
        # Stage-1 trains {first-stage A featurizer, W featurizer, first-stage X featurizer}; stage-2 trains
        # {second-stage A featurizer, Z featurizer, second-stage X featurizer} -- matching the trainer toggles.
        stage1_params = list(t_first.parameters()) + list(out_proxy.parameters())
        if bd_first is not None:
            stage1_params = stage1_params + list(bd_first.parameters())
        stage2_params = list(t_second.parameters()) + list(t_proxy.parameters())
        if bd_second is not None:
            stage2_params = stage2_params + list(bd_second.parameters())
        stage1_optimizers = [torch.optim.Adam(stage1_params, lr=lr_stage1, weight_decay=weight_decay)]
        stage2_optimizers = [torch.optim.Adam(stage2_params, lr=lr_stage2, weight_decay=weight_decay)]
        model, _, _ = train_alternative_treatment_pcl_net_ate_model(
            model=model, first_stage_train_dataloader=first_stage_train_dataloader,
            second_stage_train_dataloader=second_stage_train_dataloader,
            stage1_optimizers=stage1_optimizers, stage2_optimizers=stage2_optimizers,
            n_epochs=n_epochs, stage1_iter=stage1_iter, stage2_iter=stage2_iter,
            first_stage_final_layer_regularizer=first_stage_final_layer_regularizer,
            second_stage_final_layer_regularizer=second_stage_final_layer_regularizer,
            second_stage_first_final_layer_regularizer=second_stage_first_final_layer_regularizer,
            consider_prev_weight=consider_prev_weight, loss_fn=loss_fn,
            log_per_epoch=10 ** 9, plot_loss=False, validation_dataloader=validation_dataloader)
        members.append(model)
    return EnsembleAlternativeTreatmentBridge(members, device=device).eval()



def create_third_stage_dataset_for_treatment_pcl_net_ate(
    pcl_model: nn.Module,
    pcl_dataloader: DataLoader,
    pcl_val_dataloader: Optional[DataLoader] = None,
    outcome_transformer: Optional[nn.Module] = TorchIdentityTransformer(),
    input_transformer: Optional[nn.Module] = TorchIdentityTransformer(),
    treatment_featurizer = torch.nn.Identity(),
    device: str = "cuda",
) -> Tuple[TensorDataset, Optional[TensorDataset]]:
    """Third-stage dataset for the DENSITY-RATIO-FREE bridge: input A, target Y * varphi(A, Z, X).

    Same regression as the density-ratio version in proxy_treatment_neural_mean_embedding.py --
    f_tm(a) = E[Y varphi | A = a] -- with ONE difference: there is NO INLIER SELECTION. Every row is
    kept.

    Why the filter does not belong here. The shared creator drops rows on
        |predict_dens_ratio - stored_ratio| < tol   AND   predict_dens_ratio > 0,
    which is meaningful for the density-ratio method, where `stored_ratio` is a real KDE/RuLSIF estimate
    and the comparison is a genuine calibration check. In the density-ratio-free setting the stored
    ratio is a `ones` PLACEHOLDER and every notebook passes tol = 100.5, so the first test is vacuous
    and the only live one is the sign test on the bridge's own implied ratio E[varphi | A, W, X] -- i.e.
    the model silently censoring its own output on a criterion that has nothing to do with the data.
    That is a self-referential filter, it is a function of (A, W, X) so the retained sample is no longer
    a clean draw, and it makes the estimand depend on a tolerance nobody tunes.

    The knob that actually controls this bridge's outliers is the second-stage ridge
    `reg_second_stage`, which is set explicitly in every notebook.

    NOTE the backdoor is taken from `pcl_model._backdoor_from(batch)`, so the third stage conditions the
    bridge on exactly the covariate block training used, rather than re-deriving it from the batch length.
    """
    pcl_model.eval().to(device)

    if not hasattr(pcl_model, "final_layer_second_stage_weight") or pcl_model.final_layer_second_stage_weight is None:
        return None, None

    y_transformer = pcl_dataloader.dataset.transformers[1].to(device)
    dens_ratio_transformer = copy.deepcopy(pcl_dataloader.dataset.dens_ratio_transformer).to(device)

    def _process_single_loader(loader, desc_text):
        treatment_list = []
        composite_target_list = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc_text):
                batch = [b.to(device) for b in batch]
                treatment, outcome_scaled, treatment_proxy = batch[0], batch[1], batch[2]
                backdoor = pcl_model._backdoor_from(batch)

                outcome_unscaled = y_transformer.inverse_transform(outcome_scaled)
                varphi = pcl_model.predict_treatment_bridge_function(treatment, treatment_proxy, backdoor)
                varphi_unscaled = dens_ratio_transformer.inverse_transform(varphi)

                composite_target_list.append((outcome_unscaled * varphi_unscaled).cpu())
                treatment_list.append(treatment_featurizer(treatment).cpu())

        if len(treatment_list) == 0:
            return None, None
        return torch.cat(treatment_list), torch.cat(composite_target_list)

    treatment_all, composite_target_all = _process_single_loader(
        pcl_dataloader, "Creating Stage 3 Dataset")
    if treatment_all is None:
        raise ValueError("Empty training set.")

    treatment_all = input_transformer.fit_transform(treatment_all)
    composite_target_all = outcome_transformer.fit_transform(composite_target_all)

    third_stage_dataset = TensorDataset(treatment_all, composite_target_all)
    third_stage_dataset.outcome_transformer = outcome_transformer
    third_stage_dataset.input_transformer = input_transformer

    third_stage_dataset_val = None
    if pcl_val_dataloader is not None:
        treatment_val, composite_target_val = _process_single_loader(
            pcl_val_dataloader, "Creating Stage 3 Val Dataset")
        if treatment_val is not None:
            third_stage_dataset_val = TensorDataset(
                input_transformer.transform(treatment_val),
                outcome_transformer.transform(composite_target_val))

    return third_stage_dataset, third_stage_dataset_val


# =============================================================================================
# PARKED: L-BFGS ("sgd") second-stage head.
#
# It is not working well but we keep it here in case we need it in the future. The density-ratio-free
# bridge now always uses the closed-form ridge solve in
# update_final_layer_with_batch_regression(stage="second"); train_alternative_treatment_pcl_net_ate_model
# raises NotImplementedError for any head_mode other than "closed_form".
#
# To revive: un-comment, re-indent into AlternativeTreatmentBridgePCLNET, and restore the head_mode
# branch in the trainer's second-stage block.
# =============================================================================================
#     def update_second_stage_head_with_sgd(self, data_tuple, lr: float = 1.0, n_steps: int = 10,
#                                           weight_decay: float = 0.0, reg: float = 0.0):
#         """SGD/L-BFGS update of the second-stage head Theta (density-ratio-FREE KAP objective).
#
#         Alternative to the closed-form ridge solve in update_final_layer_with_batch_regression(stage="second").
#         Runs `n_steps` L-BFGS iterations minimizing the SAME KAP objective in Theta:
#             L(Theta) = mean_i (U_i Theta)^2  - 2 mean_i (V_i Theta)  + reg * ||Theta||^2,
#         where U_i are the MATCHED second-stage features (mu_{Z|A_i,W_i} (x) phi_A2(A_i)) and V_i the
#         MISMATCHED ones (mu_{Z|A_i, mean_W} (x) phi_A2(A_i), the exact O(N) cross reduction). Since L is
#         quadratic in Theta, L-BFGS converges to the closed-form minimizer; this is the density-ratio-free
#         analogue of the density-ratio bridge's _w_SGD head update. L-BFGS runs on CPU for stability
#         (mirrors proxy_treatment_neural_mean_embedding_w_SGD).
#         """
#         data_tuple = [x.to(self.device) for x in data_tuple]
#         treatment, outcome_proxy = data_tuple[0], data_tuple[3]
#         has_bd = (self.backdoor_featurizer_second_stage is not None) and (len(data_tuple) > 4)
#         backdoor = self._backdoor_from(data_tuple)
#         with torch.no_grad:
#             # Featurizer call order A1 -> W -> A2 -> X1 -> X2 is deliberate (dropout RNG order).
#             treatment_feature_first_stage = self.treatment_featurizer_first_stage(treatment)
#             w_feature = self._first_stage_w_feature(outcome_proxy, backdoor)
#             treatment_feature_second_stage = self.treatment_featurizer_second_stage(treatment)
#             x_feature = self._first_stage_x_feature(backdoor)
#             j_features = self._j_features(w_feature, x_feature)
#             first_stage_feature = self._first_stage_design(treatment_feature_first_stage, j_features)
#             mu = torch.matmul(first_stage_feature, self.final_layer_first_stage_weight.T)
#             if has_bd:
#                 # Backdoor X: matched U_i is the 3-way tensor product; the mismatched cross collapses to
#                 # the SINGLE O(N) expectation row cross_rhs (the L-BFGS closure's mean over its one row = the
#                 # value), so no per-sample V is needed.
#                 backdoor_feature_second_stage = self.backdoor_featurizer_second_stage(backdoor)
#                 U = outer_prod_batch(mu, treatment_feature_second_stage, backdoor_feature_second_stage).flatten(start_dim=1)  # matched
#                 V = self._cross_rhs_backdoor(
#                     treatment_feature_first_stage, j_features,
#                     treatment_feature_second_stage, backdoor_feature_second_stage,
#                     self.final_layer_first_stage_weight)  # (1, D) mismatched
#             else:
#                 U = outer_prod(mu, treatment_feature_second_stage).flatten(start_dim=1)                 # matched
#                 outcome_proxy_feature_mean = j_features[0].mean(0, keepdim=True)
#                 cross = outer_prod_cross_batch(treatment_feature_first_stage, outcome_proxy_feature_mean).flatten(start_dim=1)
#                 mu_cross = torch.matmul(cross, self.final_layer_first_stage_weight.T)
#                 V = outer_prod(mu_cross, treatment_feature_second_stage).flatten(start_dim=1)           # mismatched (O(N))
#             U_cpu = U.detach.cpu.float; V_cpu = V.detach.cpu.float
#             if torch.isnan(U_cpu).any or torch.isinf(U_cpu).any:
#                 return
#         w_prev = self.final_layer_second_stage_weight.detach.cpu.float.clone
#         w_cpu = w_prev.clone.requires_grad_(True)
#         optimizer = torch.optim.LBFGS([w_cpu], lr=lr, max_iter=20, history_size=10,
#                                       line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-9)
#
#         def closure:
#             optimizer.zero_grad
#             pred_sq = torch.matmul(U_cpu, w_cpu.T)   # (N, outcome_dim)
#             pred_cr = torch.matmul(V_cpu, w_cpu.T)   # (N, outcome_dim)
#             loss = (pred_sq ** 2).mean - 2.0 * pred_cr.mean + reg * (w_cpu ** 2).sum
#             if weight_decay > 0:
#                 loss = loss + 0.5 * weight_decay * torch.sum((w_cpu - w_prev) ** 2)
#             loss.backward
#             return loss
#
#         try:
#             for _ in range(n_steps):
#                 optimizer.step(closure)
#         except RuntimeError as e:
#             print(f"L-BFGS head warning: {e}")
#         new_w = torch.clip(w_cpu.detach, min=0.0) if self.nonnegative_final_layer else w_cpu.detach
#         self.final_layer_second_stage_weight = new_w.to(self.device)
#
