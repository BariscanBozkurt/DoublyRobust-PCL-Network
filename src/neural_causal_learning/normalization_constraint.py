"""Training-time enforcement of the treatment-bridge normalisation identity E[varphi | A = a] = 1.

THE IDENTITY. The treatment-bridge moment equation E[varphi(A,Z) | A, W] = r0 = p(A)p(W)/p(A,W)
implies, on averaging over W given A = a,

    E[varphi | A = a] = 1   for every a,

so a correct bridge is FLAT in A. The fitted bridge satisfies this only approximately, and the cost of
violating it is not subtle: writing varphi_hat = m(A) * varphi_tilde with E[varphi_tilde | A] = 1, the
third stage returns

    f_TM(a) = E[Y varphi_hat | A = a] = m(a) * E[Y varphi_tilde | A = a],

i.e. the dose-response is multiplied POINTWISE by m(a). A mildly tilted bridge becomes a badly wrong
causal curve -- on the Deaner benchmark, causal MSE ~4.3 against a ground truth whose entire range
is 0.17.

WHY IT CAN BE IMPOSED EXACTLY. The bridge is bilinear in learned features and LINEAR in the
second-stage head Theta,

    varphi(a, z) = <Theta, phi_Z(z) (x) phi_A2(a)> = Theta^T u(a, z),

and E[varphi | A = a] = 1 for all a is equivalent to E[(varphi - 1) g(A)] = 0 for every g. Truncating
to a finite basis {g_k}_{k<K} therefore gives K LINEAR equations in Theta,

    G Theta = h,   G[k,:] = (1/n) sum_i g_k(a_i) u_i^T,   h[k] = (1/n) sum_i g_k(a_i).

The second-stage head is already obtained by a closed-form regularised quadratic solve
A_mat Theta = RHS, so the constraint is imposed by replacing that solve with its KKT system

    [ A_mat  G^T ] [Theta ]   [RHS]
    [ G      0   ] [lambda] = [ h ] ,

adding no hyperparameter. Every head iterate then satisfies the constraint, and the featurisers are
trained by SGD against a head already on the constraint manifold.

BACKDOOR X -- READ THIS BEFORE EXTENDING THE BASIS. With a backdoor the bridge gains a slot,
varphi(a,z,x) = <Theta, phi_Z (x) phi_A2 (x) phi_X2>, and

    E[varphi | A = a]        = 1                    (still true, marginalising over W and X)
    E[varphi | A = a, X = x] = p(a) / p(a | x)      (NOT 1)

so the basis must remain a function of A ALONE. Enriching it to g_k(A, X) would push E[varphi | A, X]
towards 1 when the correct value is p(a)/p(a|x): unless X is independent of A that constraint is FALSE
at the population level, makes the true bridge infeasible, and biases rather than regularises. This
module refuses bases that depend on X. Only u_i changes in the backdoor case; G, h and the KKT system
are structurally identical, and since K is unchanged while P grows the RELATIVE cost falls.

SCOPE. The head update is per mini-batch, so the constraint binds the batch it was last solved on. When
the stage-2 fold fits in one batch the two coincide; otherwise this is a stochastic approximation of the
fold-level condition. Use a batch that covers the stage-2 fold if exact fold-level enforcement matters.
"""
from typing import Optional

import numpy as np
import torch

import sys
sys.path.append("..")
from torch_utils.linear_algebra import outer_prod, outer_prod_batch


class ConditionalMeanConstraint:
    """E[varphi | A = a] = 1, imposed on the closed-form second-stage head.

    Parameters
    ----------
    basis : {"poly", "rbf", "indicator"}
        Functions g_k of the treatment. "indicator" (one column per distinct treatment value) makes the
        constraint state that the WITHIN-ARM mean of varphi is one -- the binary-treatment per-arm
        normalisation is exactly this constraint, not a separate method.
    degree : int
        Polynomial degree for basis="poly" (K = degree + 1).
    n_rbf : int
        Number of Gaussian bumps for basis="rbf" (K = n_rbf + 1, including a constant).
    mode : {"kkt", "penalty"}
        Hard equality constraint, or a quadratic penalty of weight `rho` (rho -> inf recovers "kkt").
    rho : float
        Penalty weight, used only when mode="penalty".
    orthonormalize : bool
        Whiten the constraint rows against the empirical distribution of A so that
        (1/n) sum_i g_j(a_i) g_k(a_i) = delta_jk, discarding rank-deficient directions.
        Replacing (G, h) by (Q G, Q h) leaves the constraint SET unchanged, so this is free -- but it is
        not cosmetic: the raw monomial basis is a Vandermonde system with nearly collinear rows, and a
        hard equality through an ill-conditioned G produces huge multipliers and hence a huge Theta.
        Measured on Deaner: without it the constraint was met by inflating std(varphi) from 1.0 to 9.8
        and the treatment bridge degraded 10x. Leave this on.
    eval_mode : bool
        Assemble the constraint with the second-stage featurisers in eval mode (dropout off). REQUIRED
        for the constraint to bind the deterministic bridge that the third stage and the DR slacks
        evaluate: with dropout active the in-sample violation is ~1e-1 instead of ~1e-7.
    rank_tol : float
        Relative eigenvalue threshold below which a constraint direction is discarded as redundant.
    """

    def __init__(self, basis: str = "poly", degree: int = 5, n_rbf: int = 8,
                 mode: str = "kkt", rho: float = 10.0, orthonormalize: bool = True,
                 eval_mode: bool = True, rank_tol: float = 1e-6, projection=None):
        if basis not in ("poly", "rbf", "indicator"):
            raise ValueError("basis must be 'poly', 'rbf' or 'indicator', got %r" % basis)
        if mode not in ("kkt", "penalty"):
            raise ValueError("mode must be 'kkt' or 'penalty', got %r" % mode)
        self.basis, self.degree, self.n_rbf = basis, degree, n_rbf
        self.mode, self.rho = mode, rho
        self.orthonormalize, self.eval_mode, self.rank_tol = orthonormalize, eval_mode, rank_tol
        # OPTIONAL PROJECTION for a NON-SCALAR treatment. The basis functions g_k need a scalar to
        # act on; with an image / functional / vector A there is none. `projection` maps the treatment
        # to a scalar summary, and the basis is built on THAT: g_k(pi(A)). This is legitimate exactly
        # when pi is a function of A ALONE -- the module's standing requirement -- since then g_k. pi
        # is itself a function of A, and E[varphi|A]=1 implies E[(varphi-1) g_k(pi(A))]=0 by the tower
        # rule. It is a NECESSARY condition, not a sufficient one: it constrains the identity along
        # pi's direction only, so a violation it catches is real, but flatness in pi does not prove
        # flatness in every direction of a high-dimensional treatment. Say so when reporting.
        # None (the default) = the treatment is used directly, i.e. the original scalar-A behaviour.
        self.projection = projection
        self._rbf_centers = None          # cached from the first batch, then held FIXED
        self._rbf_sigma = None
        self._levels = None               # cached treatment levels for basis="indicator"

    # ------------------------------------------------------------------ basis
    def _raw_basis(self, a_flat: torch.Tensor) -> torch.Tensor:
        """(K, N) matrix of g_k(a_i). A function of the TREATMENT ONLY -- see the module docstring."""
        if self.basis == "poly":
            return torch.stack([a_flat ** k for k in range(self.degree + 1)], dim=0)
        if self.basis == "indicator":
            if self._levels is None:
                self._levels = torch.unique(a_flat.detach())
            return (a_flat[None, :] == self._levels[:, None]).to(a_flat.dtype)
        if self._rbf_centers is None:
            qs = torch.linspace(0.05, 0.95, self.n_rbf, device=a_flat.device)
            self._rbf_centers = torch.quantile(a_flat.detach(), qs)
            self._rbf_sigma = float(
                (self._rbf_centers.max() - self._rbf_centers.min()) / max(self.n_rbf - 1, 1)) + 1e-6
        bumps = torch.exp(-0.5 * ((a_flat[None, :] - self._rbf_centers[:, None]) / self._rbf_sigma) ** 2)
        return torch.cat([torch.ones(1, a_flat.shape[0], device=a_flat.device), bumps], dim=0)

    def _whitener(self, gram: torch.Tensor) -> torch.Tensor:
        """(K', K) left-multiplier M orthonormalising a basis with the given empirical Gram.

        Whiten by eigendecomposition rather than Cholesky, DISCARDING near-null directions. Cholesky
        assumes positive definiteness, which fails whenever the basis is rank-deficient on the
        observed treatments (overlapping RBF bumps do exactly that). Dropping degenerate directions is
        also structurally right: a redundant constraint row carries no information but makes the KKT
        block singular.

        Factored out of _basis so the pair-law ATT subclass can whiten from a CHUNK-ACCUMULATED Gram
        without ever materialising the full basis matrix. Because whitening is a left-multiplication,
        M @ (g u / n) equals ((M g) u / n) exactly -- the two orders agree.
        """
        w, V = torch.linalg.eigh(gram)
        keep = w > (w.max().clamp_min(1e-30) * self.rank_tol)
        w, V = w[keep], V[:, keep]
        return (V * w.rsqrt()).T

    def _basis_argument(self, treatment: torch.Tensor) -> torch.Tensor:
        """The scalar the basis functions act on: the treatment itself, or its projection."""
        if self.projection is None:
            if treatment.dim() > 1 and treatment.shape[1] > 1:
                raise ValueError(
                    "the treatment has %d columns, so there is no scalar for the basis to act on; "
                    "pass projection=<callable mapping A to one scalar per row> (e.g. a known "
                    "sufficient summary s(A), or an unsupervised summary such as a principal "
                    "component). Any function of A alone yields a valid moment condition."
                    % treatment.shape[1])
            return treatment.reshape(-1)
        projected = self.projection(treatment).reshape(-1)
        if projected.shape[0] != treatment.shape[0]:
            raise ValueError("projection must return one scalar per row: got %d for %d rows"
                             % (projected.shape[0], treatment.shape[0]))
        # standardise so a polynomial basis is not a wildly scaled Vandermonde (the whitening below
        # would otherwise have to undo it, and rank truncation can bite first)
        return (projected - projected.mean()) / projected.std().clamp_min(1e-12)

    def _basis(self, a_flat: torch.Tensor) -> torch.Tensor:
        g = self._raw_basis(a_flat)
        if not self.orthonormalize:
            return g / g.norm(dim=1, keepdim=True).clamp_min(1e-12) * np.sqrt(g.shape[1])
        n = g.shape[1]
        return self._whitener((g @ g.T) / n) @ g

    # ------------------------------------------------------------------ the linear system
    def system(self, model, data_tuple):
        """(G, h) such that G Theta = h encodes the K moment conditions on THIS batch.

        Built from the bridge's OWN features u = phi_Z (x) phi_A2 [(x) phi_X2] -- exactly what
        predict_treatment_bridge_function evaluates -- and NOT from the matched first-stage design
        mu_{Z|A,W} (x) phi_A2 appearing in the KAP square term. The constraint has to bind the object
        the third stage and the DR slacks actually call.
        """
        treatment, treatment_proxy = data_tuple[0], data_tuple[2]
        backdoor = model._backdoor_from(data_tuple)
        featurisers = [model.treatment_proxy_featurizer, model.treatment_featurizer_second_stage]
        if model.backdoor_featurizer_second_stage is not None:
            featurisers.append(model.backdoor_featurizer_second_stage)

        was_training = [f.training for f in featurisers]
        if self.eval_mode:
            for f in featurisers:
                f.train(False)
        try:
            with torch.no_grad():
                phi_Z = model.treatment_proxy_featurizer(treatment_proxy)
                phi_A2 = model.treatment_featurizer_second_stage(treatment)
                if (backdoor is not None) and (model.backdoor_featurizer_second_stage is not None):
                    phi_X2 = model.backdoor_featurizer_second_stage(backdoor)
                    u = outer_prod_batch(phi_Z, phi_A2, phi_X2).flatten(start_dim=1)
                elif hasattr(model, "_aug"):
                    # trainer-V2 bridge (alternative_treatment_neural_mean_embedding_V2): its bridge
                    # feature may carry intercept cross terms; the constraint must bind THAT feature.
                    # Every existing model lacks `_aug`, so this branch is never taken for them.
                    u = model._aug(phi_Z, phi_A2)
                else:
                    u = outer_prod(phi_Z, phi_A2).flatten(start_dim=1)
                g = self._basis(self._basis_argument(treatment))
        finally:
            for f, state in zip(featurisers, was_training):
                f.train(state)

        n = u.shape[0]
        return g @ u / n, g.mean(dim=1, keepdim=True)

    # ------------------------------------------------------------------ the constrained solve
    def solve(self, model, data_tuple, A_mat, RHS):
        """Return Theta solving the head system subject to the moment conditions.

        A_mat (P, P) and RHS (P, outcome_dim) are the UNCONSTRAINED normal equations assembled by the
        model; this replaces `torch.linalg.solve(A_mat, RHS)`.
        """
        if RHS.shape[1] != 1:
            raise NotImplementedError(
                "the normalisation constraint assumes a scalar bridge (outcome_dim = 1); got %d"
                % RHS.shape[1])
        G, h = self.system(model, data_tuple)
        n_feat = A_mat.shape[0]

        if self.mode == "penalty":
            n = data_tuple[0].shape[0]
            return torch.linalg.solve(A_mat + self.rho * n * (G.T @ G),
                                      RHS + self.rho * n * (G.T @ h))
        n_con = G.shape[0]
        block = torch.cat([torch.cat([A_mat, G.T], dim=1),
                           torch.cat([G, torch.zeros(n_con, n_con, device=A_mat.device,
                                                     dtype=A_mat.dtype)], dim=1)], dim=0)
        solution = torch.linalg.solve(block, torch.cat([RHS, h], dim=0))
        return solution[:n_feat]

    # ------------------------------------------------------------------ diagnostics
    def violation(self, model, data_tuple):
        """max_k |(1/n) sum_i g_k(a_i)(varphi_i - 1)| -- 0 means the constraint binds exactly here."""
        G, h = self.system(model, data_tuple)
        theta = model.final_layer_second_stage_weight.detach().T
        return float((G @ theta - h).abs().max())

    def __repr__(self):
        detail = ("degree=%d" % self.degree if self.basis == "poly"
                  else "n_rbf=%d" % self.n_rbf if self.basis == "rbf" else "levels")
        return ("ConditionalMeanConstraint(basis=%r, %s, mode=%r%s, orthonormalize=%r)"
                % (self.basis, detail, self.mode,
                   ", rho=%g" % self.rho if self.mode == "penalty" else "", self.orthonormalize))


class ConditionalMeanConstraintATT(ConditionalMeanConstraint):
    """E[varphi^ATT(a, a', X, Z) | A = a] = 1, imposed on the ATT closed-form second-stage head.

    THE IDENTITY (derivation eq. 41). The ATT bridge solves E[varphi(a,a',X,Z) | A=a, X=x, W=w] =
    r0_ATT(a,a',x,w) = p_{X,W|A}(x,w|a') / p_{X,W|A}(x,w|a); averaging over (X,W) given A=a makes the
    right-hand side integrate to one, so a correct ATT bridge obeys

        E[varphi(A_1, A_2, X_1, Z_1) | A_1, A_2] = 1

    under the independent-pair law -- the two-argument analogue of the ATE identity, and violating it
    distorts the fitted surface pointwise in (a, a') exactly as m(a) does for the dose response.

    WHY THE SAME MACHINERY WORKS. The ATT bridge is varphi = <Theta, psi_{AA'X}(a,a',x) (x) psi_Z(z)>
    (eq. 18) -- bilinear in learned features and LINEAR in the head Theta, precisely like its ATE
    sibling. The moments E[(varphi - 1) g_k(A_1, A_2)] = 0 are therefore linear constraints G Theta = h,
    and the constrained head is the same KKT block system. Everything in the base class except the
    construction of (G, h) is inherited unchanged: the block solve, the penalty mode, the eigenvalue
    whitening with rank truncation, and the eval-mode discipline.

    THE THREE DIFFERENCES, all of them in `system`:

    1. THE BASIS IS TWO-INPUT. The ATT identity conditions on the PAIR (A_1, A_2), so the basis is
       g_k(a, a'), built here as the column-wise outer product of the base class's one-dimensional
       basis evaluated at the source and at the anchor treatment. That squares the constraint count --
       poly degree d gives K = (d+1)^2, binary A gives K = 4 -- so keep the degree LOW; a degree-3
       tensor basis is already 16 constraints. Orthonormalisation matters MORE here than in the ATE
       case, a two-dimensional Vandermonde being worse conditioned than the one-dimensional one that
       already inflated ||Theta|| on Deaner.

    2. THE EXPECTATION IS OVER PAIRS, NOT ROWS. G is accumulated over the ordered pairs of the
       independent-pair law, in the same CHUNKS the head's normal equations use, so the O(M^2 d) pair
       feature matrix is never materialised. The Gram of the basis is accumulated alongside, and the
       whitener is applied to the finished (G, h) -- identical to whitening the basis first, because
       whitening is a left-multiplication.

    3. THE PAIRS ARE THE HEAD'S OWN PAIRS. `system` reads the (source, anchor) indices the model
       stashed while assembling the normal equations, so the constraint binds the SAME empirical pair
       measure the head was solved on. This mirrors the ATE case, where G is built from the very rows
       of the batch whose normal equations are being solved. If no indices are available (a direct
       call outside the head update) a fresh set is drawn from the model's pair mode.

    THE BACKDOOR TRAP CARRIES OVER UNCHANGED. E[varphi | A=a, A'=a', X=x] is p(x,w|a')-transported and
    is NOT 1, so the basis must remain a function of (a, a') ALONE. An X-dependent basis would impose a
    population-FALSE restriction, exactly as g_k(A, X) does in the ATE case; X still enters the
    constraint correctly through the bridge features u.
    """

    def __init__(self, basis: str = "poly", degree: int = 3, n_rbf: int = 4,
                 mode: str = "kkt", rho: float = 10.0, orthonormalize: bool = True,
                 eval_mode: bool = True, rank_tol: float = 1e-6):
        # NOTE the lower default degree / n_rbf: the tensor-product basis squares the constraint count.
        super().__init__(basis=basis, degree=degree, n_rbf=n_rbf, mode=mode, rho=rho,
                         orthonormalize=orthonormalize, eval_mode=eval_mode, rank_tol=rank_tol)

    # ------------------------------------------------------------------ basis over pairs
    def _raw_basis_pairs(self, a_source: torch.Tensor, a_anchor: torch.Tensor) -> torch.Tensor:
        """(K, n_pairs) matrix of g_k(a_i, a_j) = g_p(a_i) g_q(a_j), the tensor-product basis.

        Both arguments go through the base class's one-dimensional `_raw_basis`, so the RBF centres and
        the indicator levels are shared between the two slots -- correct, since the source and the
        anchor treatment are two draws of the SAME marginal, and it keeps the cached basis stable
        across calls.
        """
        g_src = self._raw_basis(a_source)                      # (K1, n_pairs)
        g_anc = self._raw_basis(a_anchor)                      # (K2, n_pairs)
        return (g_src[:, None, :] * g_anc[None, :, :]).reshape(-1, g_src.shape[1])

    # ------------------------------------------------------------------ the linear system
    def system(self, model, data_tuple):
        """(G, h) with G Theta = h encoding the K moment conditions over THIS batch's ordered pairs."""
        treatment, treatment_proxy = data_tuple[0], data_tuple[2]
        backdoor = model._backdoor_from(data_tuple) if hasattr(model, "_backdoor_from") else None

        pair_indices = getattr(model, "_last_pair_indices", None)
        if pair_indices is None:
            pair_indices = model.make_pair_indices(treatment.shape[0])
        source_idx, anchor_idx = pair_indices
        n_pairs = int(source_idx.shape[0])
        chunk = int(getattr(model, "pair_chunk_size", 200000))

        featurisers = [model.anchor_treatment_featurizer, model.treatment_proxy_featurizer]
        was_training = [f.training for f in featurisers]
        if self.eval_mode:
            for f in featurisers:
                f.train(False)
        try:
            with torch.no_grad():
                G_raw, h_raw, gram = None, None, None
                for start in range(0, n_pairs, chunk):
                    s_idx = source_idx[start:start + chunk]
                    a_idx = anchor_idx[start:start + chunk]
                    a_src, a_anc = treatment[s_idx], treatment[a_idx]
                    # the bridge feature of the SOURCE unit, paired with the anchor treatment --
                    # exactly what predict_treatment_bridge_function evaluates (same factor order)
                    anchor_feature = model._anchor_feature(
                        a_src, a_anc, None if backdoor is None else backdoor[s_idx])
                    psi_z = model.treatment_proxy_featurizer(treatment_proxy[s_idx])
                    u = outer_prod(anchor_feature, psi_z).flatten(start_dim=1)      # (m, P)
                    g = self._raw_basis_pairs(a_src.reshape(-1), a_anc.reshape(-1))  # (K, m)
                    G_chunk, h_chunk = g @ u, g.sum(dim=1, keepdim=True)
                    gram_chunk = g @ g.T
                    if G_raw is None:
                        G_raw, h_raw, gram = G_chunk, h_chunk, gram_chunk
                    else:
                        G_raw += G_chunk
                        h_raw += h_chunk
                        gram += gram_chunk
        finally:
            for f, state in zip(featurisers, was_training):
                f.train(state)

        G_raw, h_raw, gram = G_raw / n_pairs, h_raw / n_pairs, gram / n_pairs
        if not self.orthonormalize:
            scale = gram.diagonal().clamp_min(1e-24).rsqrt().reshape(-1, 1)
            return G_raw * scale, h_raw * scale
        whitener = self._whitener(gram)
        return whitener @ G_raw, whitener @ h_raw

    def __repr__(self):
        detail = ("degree=%d" % self.degree if self.basis == "poly"
                  else "n_rbf=%d" % self.n_rbf if self.basis == "rbf" else "levels")
        return ("ConditionalMeanConstraintATT(basis=%r, %s [tensor-product in (a, a')], mode=%r, "
                "orthonormalize=%r)" % (self.basis, detail, self.mode, self.orthonormalize))
