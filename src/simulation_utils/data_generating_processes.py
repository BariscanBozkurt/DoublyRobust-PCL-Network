import os
import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.sparse import diags
from scipy.stats import norm
from scipy.spatial.distance import cdist
from itertools import product
import torch.utils.data as data
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple
#################################################################################################
############## PROXY CAUSAL LEARNING DATA GENERATION FUNCTIONS ##################################
#################################################################################################


class PCL_Synthetic_High_Dim:
    """
    This data generation process is taken from the official code of the following paper (from Openreview):
    Doubly Robust Proximal Causal Learning for Continuous Treatments, Yong Wu, Yanwei Fu, Shouyan Wang, Xinwei Sun
    """
    def __init__(self ,seeds = 43, size = 1000,dim_z = 1, dim_w = 3, dim_x = 10, type_ = 'quadratic'):
        self.seeds = seeds
        self.size = size
        self.dim_z = dim_z
        self.dim_w = dim_w
        self.dim_x = dim_x
        self.type_ = type_

    def Lambda(self, t):
        return (0.9 - 0.1) * np.exp(t) / (1 + np.exp(t)) + 0.1

    def generatate_high(self, totensor: bool = False,):
        np.random.seed(self.seeds)

        e1 = np.random.normal(0,1,self.size)
        e2 = np.random.normal(0,1,self.size)
        e3 = np.random.normal(0,1,self.size)
        vz = np.random.uniform(-1, 1, size= (self.size, self.dim_z))
        vw = np.random.uniform(-1, 1, size= (self.size, self.dim_w))



        Uz = e1 + e3
        Uw = e2 + e3

        Z = vz + 0.25* np.repeat(Uz.reshape(-1, 1), self.dim_z, axis=1)
        W = vw + 0.25* np.repeat(Uw.reshape(-1, 1), self.dim_w, axis=1)


        rho = 0.5
        k = [rho*np.ones(self.dim_x-1),np.ones(self.dim_x),rho*np.ones(self.dim_x-1)]
        offset = [-1,0,1]
        sigma = diags(k,offset).toarray()
        X = np.random.multivariate_normal(np.zeros(self.dim_x),sigma,size=[self.size,])

        theta_x = np.array([(1/(l**2)) for l in list(range(1,(self.dim_x+1)))])
        theta_w = np.array([(1/(l**2)) for l in list(range(1,(self.dim_w+1)))])
        theta_z = np.array([(1/(l**2)) for l in list(range(1,(self.dim_z+1)))])

        A = self.Lambda(3*X@theta_x + 3*Z@theta_z) + 0.25*Uw

        if self.type_ == 'quadratic':
            structure = 1.2*A + (A**2)
            Y = structure + 1.2*(X@theta_x + W@theta_w) + A*X[:,0] + 0.25*Uz
        elif self.type_ == 'peaked':
            structure = 2*(A**4/600 + np.exp(-4*A**2) + A/10 -2) + 1.2*A
            Y = structure + 1.2*(X@theta_x + W@theta_w) + A*X[:,0] + 0.25*Uz
        elif self.type_ == 'sigmoid':
            structure = np.log(abs(16*A-8)+1)*np.sign(A-0.5) + 1.2*A
            Y = structure + 1.2*(X@theta_x + W@theta_w) + A*X[:,0] + 0.25*Uz

        return A[:, np.newaxis], Z, W, Y[:, np.newaxis], X

    def generate_test(self,size,seed=43,totensor=False) -> None:
        np.random.seed(seed)
        e1 = np.random.normal(0,1,size)
        e2 = np.random.normal(0,1,size)
        e3 = np.random.normal(0,1,size)
        vz = np.random.uniform(-1, 1, size= (size, self.dim_z))
        vw = np.random.uniform(-1, 1, size= (size, self.dim_w))

        Uz = e1 + e3
        Uw = e2 + e3

        Z = vz + 0.25* np.repeat(Uz.reshape(-1, 1), self.dim_z, axis=1)
        W = vw + 0.25* np.repeat(Uw.reshape(-1, 1), self.dim_w, axis=1)

        rho =0.5
        k = [rho*np.ones(self.dim_x-1),np.ones(self.dim_x),rho*np.ones(self.dim_x-1)]
        offset = [-1,0,1]
        sigma = diags(k,offset).toarray()
        X = np.random.multivariate_normal(np.zeros(self.dim_x),sigma,size=[size,])

        theta_x = np.array([(1/(l**2)) for l in list(range(1,(self.dim_x+1)))])
        theta_w = np.array([(1/(l**2)) for l in list(range(1,(self.dim_w+1)))])
        theta_z = np.array([(1/(l**2)) for l in list(range(1,(self.dim_z+1)))])

        A = self.Lambda(3*X@theta_x + 3*Z@theta_z) + 0.25*Uw

        if self.type_ == 'quadratic':
            structure = 1.2*A + (A**2)

            Y = structure + 1.2*(X@theta_x + W@theta_w) + A*X[:,0] + 0.25*Uz
        elif self.type_ == 'peaked':
            structure = 2*(A**4/600 + np.exp(-4*A**2) + A/10 -2) + 1.2*A
            Y = structure + 1.2*(X@theta_x + W@theta_w) + A*X[:,0] + 0.25*Uz
        elif self.type_ == 'sigmoid':
            structure = np.log(abs(16*A-8)+1)*np.sign(A-0.5) + 1.2*A
            Y = structure + 1.2*(X@theta_x + W@theta_w) + A*X[:,0] + 0.25*Uz

        return A[:, np.newaxis], Z, W, Y[:, np.newaxis], X


    @staticmethod
    def generate_test_effect(a,b,c,type_,dim_z,dim_w,dim_x):
        A = np.linspace(a, b, c)
        e1 = np.random.normal(0,1,10000)
        e2 = np.random.normal(0,1,10000)
        e3 = np.random.normal(0,1,10000)
        vz = np.random.uniform(-1, 1, size= (10000,dim_z))
        vw = np.random.uniform(-1, 1, size= (10000,dim_w) )

        Uz = e1 + e3
        Uw = e2 + e3

        Z = vz + 0.25* np.repeat(Uz.reshape(-1, 1), dim_z, axis=1)
        W = vw + 0.25* np.repeat(Uw.reshape(-1, 1), dim_w, axis=1)

        rho = 0.5,
        k = [rho*np.ones(dim_x-1),np.ones(dim_x),rho*np.ones(dim_x-1)]
        offset = [-1,0,1]
        sigma = diags(k,offset).toarray()
        X = np.random.multivariate_normal(np.zeros(dim_x),sigma,size=[10000,])

        theta_x = np.array([(1/(l**2)) for l in list(range(1,(dim_x+1)))])
        theta_w = np.array([(1/(l**2)) for l in list(range(1,(dim_w+1)))])
        theta_z = np.array([(1/(l**2)) for l in list(range(1,(dim_z+1)))])

        if type_ == 'quadratic':
            treatment = np.array([np.mean(1.2*a + (a**2) + 1.2*(X@theta_x + W@theta_w) + a*X[:,0] + 0.25*Uz) for a in A])
        elif type_ == 'peaked':
            treatment = np.array([np.mean(2*(a**4/600 + np.exp(-4*a**2) + a/10 -2) + 1.2*a + 1.2*(X@theta_x + W@theta_w) + a*X[:,0] + 0.25*Uz) for a in A])
        elif type_ == 'sigmoid':
            treatment = np.array([np.mean(np.log(abs(16*a-8)+1)*np.sign(a-0.5) + 1.2*a+ 1.2*(X@theta_x + W@theta_w) + a*X[:,0] + 0.25*Uz) for a in A])
        A = A.reshape(-1, 1)
        treatment = treatment.reshape(-1, 1)
        return A,treatment


class PCL_Synthetic_High_DimNew:
    """
    Improved Synthetic Data Generator.

    Changes from original:
    1. Replaced Quadratic Decay (1/j^2) with Dense Normalized Weights (1/sqrt(d)).
       - Old: Only first ~5 features mattered.
       - New: All 100 features matter equally.
    2. Fixed method name typos.
    """
    def __init__(self, seeds=43, size=1000, dim_z=10, dim_w=10, dim_x=100, type_='quadratic'):
        self.seeds = seeds
        self.size = size
        self.dim_z = dim_z
        self.dim_w = dim_w
        self.dim_x = dim_x
        self.type_ = type_

    def Lambda(self, t):
        return (0.9 - 0.1) * np.exp(t) / (1 + np.exp(t)) + 0.1

    def _get_dense_weights(self, dim):
        """
        Returns weights where every dimension matters equally.
        Scaled by 1/sqrt(dim) to keep dot-product variance = 1.0.
        """
        return np.ones(dim) / np.sqrt(dim)

    def generatate_high(self, totensor: bool = False):
        np.random.seed(self.seeds)

        # 1. Latent Variables
        e1 = np.random.normal(0, 1, self.size)
        e2 = np.random.normal(0, 1, self.size)
        e3 = np.random.normal(0, 1, self.size)

        # Unobserved Confounders
        Uz = e1 + e3
        Uw = e2 + e3

        # 2. Proxies (Z, W)
        # Noise
        vz = np.random.uniform(-1, 1, size=(self.size, self.dim_z))
        vw = np.random.uniform(-1, 1, size=(self.size, self.dim_w))

        # Signal injection (Broadcasting Uz across all Z dimensions)
        Z = vz + 0.25 * np.repeat(Uz.reshape(-1, 1), self.dim_z, axis=1)
        W = vw + 0.25 * np.repeat(Uw.reshape(-1, 1), self.dim_w, axis=1)

        # 3. High-Dim Covariates (X) with Correlation
        rho = 0.5
        k = [rho * np.ones(self.dim_x - 1), np.ones(self.dim_x), rho * np.ones(self.dim_x - 1)]
        offset = [-1, 0, 1]
        sigma = diags(k, offset).toarray()
        X = np.random.multivariate_normal(np.zeros(self.dim_x), sigma, size=[self.size, ])

        # 4. Coefficients (The Fix!)
        # Use dense weights instead of decaying ones
        theta_x = self._get_dense_weights(self.dim_x)
        theta_w = self._get_dense_weights(self.dim_w)
        theta_z = self._get_dense_weights(self.dim_z)

        # 5. Treatment Assignment
        # Now depends on ALL dimensions of X and Z
        A = self.Lambda(3 * X @ theta_x + 3 * Z @ theta_z) + 0.25 * Uw

        # 6. Outcome Generation
        if self.type_ == 'quadratic':
            structure = 1.2 * A + (A**2)
            # Bias depends on ALL X and W
            Y = structure + 1.2 * (X @ theta_x + W @ theta_w) + A * X[:, 0] + 0.25 * Uz

        elif self.type_ == 'peaked':
            structure = 2 * (A**4 / 600 + np.exp(-4 * A**2) + A / 10 - 2) + 1.2 * A
            Y = structure + 1.2 * (X @ theta_x + W @ theta_w) + A * X[:, 0] + 0.25 * Uz

        elif self.type_ == 'sigmoid':
            structure = np.log(np.abs(16 * A - 8) + 1) * np.sign(A - 0.5) + 1.2 * A
            Y = structure + 1.2 * (X @ theta_x + W @ theta_w) + A * X[:, 0] + 0.25 * Uz

        return A[:, np.newaxis], Z, W, Y[:, np.newaxis], X

    @staticmethod
    def generate_test_effect(a, b, c, type_, dim_z, dim_w, dim_x):
        """
        Generates the Ground Truth causal curve.
        Must match the logic of generate_data EXACTLY.
        """
        A_grid = np.linspace(a, b, c)

        # Monte Carlo Integration for Ground Truth
        n_mc = 20000
        e1 = np.random.normal(0, 1, n_mc)
        e2 = np.random.normal(0, 1, n_mc)
        e3 = np.random.normal(0, 1, n_mc)

        Uz = e1 + e3

        # X generation
        rho = 0.5
        k = [rho * np.ones(dim_x - 1), np.ones(dim_x), rho * np.ones(dim_x - 1)]
        offset = [-1, 0, 1]
        sigma = diags(k, offset).toarray()
        X = np.random.multivariate_normal(np.zeros(dim_x), sigma, size=[n_mc, ])

        # W noise for confounding term in Y
        # Note: In Y equation, W is involved: 1.2 * (W @ theta_w)
        # We need to generate W to compute the expectation properly
        vw = np.random.uniform(-1, 1, size=(n_mc, dim_w))
        Uw = e2 + e3
        W = vw + 0.25 * np.repeat(Uw.reshape(-1, 1), dim_w, axis=1)

        # Dense Weights (Matching the instance method)
        theta_x = np.ones(dim_x) / np.sqrt(dim_x)
        theta_w = np.ones(dim_w) / np.sqrt(dim_w)

        # Precompute Confounding Baseline
        # E[Y|do(a)] = structure(a) + E[1.2(X theta_x + W theta_w) + a*X0 + 0.25Uz]
        # Since X, W, Uz are centered at 0, the linear terms average to 0.
        # However, let's compute it explicitly to be safe against random seed variance.
        confounding_baseline = 1.2 * (X @ theta_x + W @ theta_w) + 0.25 * Uz

        treatment_curve = []
        for a_val in A_grid:
            # The structural part depends only on a_val
            if type_ == 'quadratic':
                struct = 1.2 * a_val + (a_val**2)
            elif type_ == 'peaked':
                struct = 2 * (a_val**4 / 600 + np.exp(-4 * a_val**2) + a_val / 10 - 2) + 1.2 * a_val
            elif type_ == 'sigmoid':
                struct = np.log(np.abs(16 * a_val - 8) + 1) * np.sign(a_val - 0.5) + 1.2 * a_val

            # The interaction term a * X[:,0]
            interaction = a_val * X[:, 0]

            # E[Y_a]
            y_a = struct + interaction + confounding_baseline
            treatment_curve.append(np.mean(y_a))

        return A_grid.reshape(-1, 1), np.array(treatment_curve).reshape(-1, 1)


class PCL_Synthetic_High_DimV2:
    """Repaired high-dim PCL benchmark.  [lowdim-template candidate, verbatim]"""

    RHO_X = 0.5     # X tridiagonal correlation
    SIGMA_Z = 0.35  # per-coordinate proxy noise (Z)  -> R2(s|Z) ~ 0.89
    SIGMA_W = 0.75  # per-coordinate proxy noise (W)  -> R2(s|W) ~ 0.60
    A_X = 0.3       # backdoor weight in the treatment index
    A_U = 0.4       # confounder weight in the treatment index
    SIGMA_A = 1.0   # independent treatment noise INSIDE the link (positivity)
    B_X = 1.2       # backdoor weight in the outcome
    B_U = 1.8       # confounder weight in the outcome
    SIGMA_Y = 0.5   # independent outcome noise
    PARAM_SEED = 7  # fixed seed for the loading matrices (NOT the data seed)

    def __init__(self, seeds=43, size=1000, dim_z=10, dim_w=10, dim_x=100,
                 type_='quadratic'):
        self.seeds = seeds
        self.size = size
        self.dim_z = dim_z
        self.dim_w = dim_w
        self.dim_x = dim_x
        self.type_ = type_
        self._build_fixed_params()

    def _build_fixed_params(self):
        """Deterministic structural parameters, identical for every data seed."""
        rs = np.random.RandomState(self.PARAM_SEED)
        Mz = rs.normal(size=(self.dim_z, 2))
        Mw = rs.normal(size=(self.dim_w, 2))
        # unit-norm columns: each factor carries total loading 1 into each proxy block
        self.Mz = Mz / np.linalg.norm(Mz, axis=0, keepdims=True)
        self.Mw = Mw / np.linalg.norm(Mw, axis=0, keepdims=True)
        k = [self.RHO_X * np.ones(self.dim_x - 1), np.ones(self.dim_x),
             self.RHO_X * np.ones(self.dim_x - 1)]
        self.sigma_x = diags(k, [-1, 0, 1]).toarray()
        th = np.array([1.0 / (l ** 2) for l in range(1, self.dim_x + 1)])
        self.theta_x = th / np.sqrt(th @ self.sigma_x @ th)   # Var(X@theta_x)=1
        self.g = np.array([1.0, 1.0]) / np.sqrt(2.0)          # s = g^T U

    @staticmethod
    def Lambda(t):
        return 0.1 + 1.8 / (1.0 + np.exp(-t))

    @staticmethod
    def Lambda_inv(a):
        a = np.clip(a, 0.1 + 1e-12, 1.9 - 1e-12)
        return np.log((a - 0.1) / (1.9 - a))

    @staticmethod
    def struct(a, type_='quadratic'):
        a = np.asarray(a, dtype=float)
        if type_ == 'quadratic':
            return 1.2 * a + a ** 2
        elif type_ == 'peaked':
            return 2 * (a ** 4 / 600 + np.exp(-4 * a ** 2) + a / 10 - 2) + 1.2 * a
        elif type_ == 'sigmoid':
            return np.log(np.abs(16 * a - 8) + 1) * np.sign(a - 0.5) + 1.2 * a
        raise ValueError(type_)

    def generate_high(self, totensor: bool = False):
        np.random.seed(self.seeds)   # global-stream convention kept for consumers
        n = self.size
        U = np.random.normal(0, 1, size=(n, 2))
        s = U @ self.g
        Z = U @ self.Mz.T + self.SIGMA_Z * np.random.normal(0, 1, size=(n, self.dim_z))
        W = U @ self.Mw.T + self.SIGMA_W * np.random.normal(0, 1, size=(n, self.dim_w))
        X = np.random.multivariate_normal(np.zeros(self.dim_x), self.sigma_x, size=[n, ])
        xt = X @ self.theta_x
        eps_a = np.random.normal(0, 1, n)
        t = self.A_X * xt + self.A_U * s + self.SIGMA_A * eps_a
        A = self.Lambda(t)
        eps_y = np.random.normal(0, 1, n)
        Y = (self.struct(A, self.type_) + self.B_X * xt + A * X[:, 0]
             + self.B_U * s + self.SIGMA_Y * eps_y)
        self._last_latents = (U, s)   # diagnostics only; not part of the interface
        return A[:, np.newaxis], Z, W, Y[:, np.newaxis], X

    # the typo IS the API -- every existing consumer calls generatate_high
    def generatate_high(self, totensor: bool = False):
        return self.generate_high(totensor=totensor)

    @staticmethod
    def generate_test_effect(a, b, c, type_='quadratic', dim_z=10, dim_w=10,
                             dim_x=100, seed=0):
        """EY_do_A(a) = struct(a) EXACTLY."""
        do_A = np.linspace(a, b, c)
        ey = PCL_Synthetic_High_DimV2.struct(do_A, type_)
        return do_A.reshape(-1, 1), ey.reshape(-1, 1)

    # ---- oracle verification helpers (estimators must NOT use them) ----
    def oracle_outcome_bridge_coef(self):
        """c with E[c^T W | U] = B_U * s: exact closed-form outcome bridge."""
        Mw = self.Mw
        return Mw @ np.linalg.solve(Mw.T @ Mw, self.B_U * self.g)

    def oracle_outcome_bridge(self, A, W, X):
        A = np.asarray(A).reshape(-1)
        c = self.oracle_outcome_bridge_coef()
        xt = X @ self.theta_x
        return self.struct(A, self.type_) + self.B_X * xt + A * X[:, 0] + W @ c

    def _posterior_s_given(self, M, sig):
        """k, v with E[s|V]=k^T V, Var(s|V)=v for V = U M^T + sig*eps."""
        C = M @ M.T + (sig ** 2) * np.eye(M.shape[0])
        k = np.linalg.solve(C, M @ self.g)
        v = 1.0 - (M @ self.g) @ k
        return k, v

    def oracle_r0(self, A, W, X):
        """Exact r0 = p(A)/p(A|W,X) (MARGINAL numerator)."""
        A = np.asarray(A).reshape(-1)
        kw, vw = self._posterior_s_given(self.Mw, self.SIGMA_W)
        t = self.Lambda_inv(A)
        t_tilde = t - self.A_X * (X @ self.theta_x)
        mu = self.A_U * (W @ kw)
        Vt = self.A_X ** 2 + self.A_U ** 2 + self.SIGMA_A ** 2
        Vd = (self.A_U ** 2) * vw + self.SIGMA_A ** 2
        log_r = (-0.5 * t ** 2 / Vt + 0.5 * (t_tilde - mu) ** 2 / Vd
                 + 0.5 * np.log(Vd / Vt))
        return np.exp(log_r)

class dSprite_ProxyVariable_DatasetV2():
    ### This is based on "Update on dSprite experiment", see https://github.com/liyuan9988/DeepFeatureProxyVariable/tree/master
    def __init__(self, variant: str = "v2"):
        """variant='v2' (default, numerically identical): structural = ((a.w)^2 - 3000)/500 with column weights
        w_ij=|32-j|/32 -- the README-updated DFPV setting, numerically identical to their `dsprite`/ver2.
        variant='org': the DFPV PROCEEDINGS setting (`dsprite_org`): B in R^{4096x10} ~ Unif(0,1) from
        data/dsprite/dsprite_mat.npy, structural = (mean_k (a.B_k)^2 - 5000)/1000; Var(EY_do_A) 24.68
        (vs 53.27). Their README: on this f the sprite position is hard to recover ('effectively a
        constant function with noise'); DFPV scores ~6.2 at n=5000 there vs 15.7 on v2."""
        assert variant in ("v2", "org")
        self.variant = variant
        self._B = None

    def cal_weight(self,):
        weights = np.empty((64, 64))
        for i in range(64):
            for j in range(64):
                weights[i, j] = (np.abs(32 - j))
        return weights.reshape(64*64, 1) / 32


    def image_id(self, latent_bases: np.ndarray, posX_id_arr: np.ndarray, posY_id_arr: np.ndarray,
                 orientation_id_arr: np.ndarray,
                 scale_id_arr: np.ndarray):
        data_size = posX_id_arr.shape[0]
        color_id_arr = np.array([0] * data_size, dtype=int)
        shape_id_arr = np.array([2] * data_size, dtype=int)
        idx = np.c_[color_id_arr, shape_id_arr, scale_id_arr, orientation_id_arr, posX_id_arr, posY_id_arr]
        return idx.dot(latent_bases)


    def structural_func(self, image, weights):
        if self.variant == "org":
            return (np.mean(image.dot(self._B) ** 2, axis=1) - 5000) / 1000
        return (image.dot(weights)[:, 0] ** 2 - 3000) / 500


    def generate_test_dsprite(self, data_path: str):
        dataset_zip = np.load(os.path.join(data_path, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
                              allow_pickle=True, encoding="bytes")
        if self.variant == "org" and self._B is None:
            self._B = np.load(os.path.join(data_path, "dsprite_mat.npy"))

        weights = self.cal_weight()

        imgs = dataset_zip['imgs']
        latents_values = dataset_zip['latents_values']
        metadata = dataset_zip['metadata'][()]

        latents_sizes = metadata[b'latents_sizes']
        latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                        np.array([1, ])))

        posX_id_arr = [0, 5, 10, 15, 20, 25, 30]
        posY_id_arr = [0, 5, 10, 15, 20, 25, 30]
        scale_id_arr = [0, 3, 5]
        orientation_arr = [0, 10, 20, 30]
        latent_idx_arr = []
        for posX, posY, scale, orientation in product(posX_id_arr, posY_id_arr, scale_id_arr, orientation_arr):
            latent_idx_arr.append([0, 2, scale, orientation, posX, posY])

        image_idx_arr = np.array(latent_idx_arr).dot(latents_bases)
        data_size = 7 * 7 * 3 * 4
        treatment = imgs[image_idx_arr].reshape((data_size, 64 * 64))
        structural = self.structural_func(treatment, weights)
        structural = structural[:, np.newaxis]
        return treatment, structural


    def generate_dsprite_pv(self, data_path: str,
                            n_sample: int,
                            generate_test: bool = False,
                            rand_seed: int = 42, **kwargs):
        """
        Parameters
        ----------
        n_sample : int
            size of data
        rand_seed : int
            random seed

        Returns
        -------
        train_data : TrainDataSet
        """
        dataset_zip = np.load(os.path.join(data_path, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
                              allow_pickle=True, encoding="bytes")
        if self.variant == "org" and self._B is None:
            self._B = np.load(os.path.join(data_path, "dsprite_mat.npy"))
        # weights = np.load(os.path.join(data_path, "dsprite_mat.npy"))
        weights = self.cal_weight()
        imgs = dataset_zip['imgs']
        latents_values = dataset_zip['latents_values']
        metadata = dataset_zip['metadata'][()]

        latents_sizes = metadata[b'latents_sizes']
        latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                        np.array([1, ])))

        rng = default_rng(seed=rand_seed)
        posX_id_arr = rng.integers(32, size=n_sample)
        posY_id_arr = rng.integers(32, size=n_sample)
        scale_id_arr = rng.integers(6, size=n_sample)
        orientation_arr = rng.integers(40, size=n_sample)
        image_idx_arr = self.image_id(latents_bases, posX_id_arr, posY_id_arr, orientation_arr, scale_id_arr)
        treatment = imgs[image_idx_arr].reshape((n_sample, 64 * 64)).astype(np.float64)
        treatment += rng.normal(0.0, 0.1, treatment.shape)
        latent_feature = latents_values[image_idx_arr]  # (color, shape, scale, orientation, posX, posY)
        treatment_proxy = latent_feature[:, 2:5]  # (scale, orientation, posX)

        posX_id_proxy = np.array([16] * n_sample)
        scale_id_proxy = np.array([3] * n_sample)
        orientation_proxy = np.array([0] * n_sample)
        proxy_image_idx_arr = self.image_id(latents_bases, posX_id_proxy, posY_id_arr, orientation_proxy, scale_id_proxy)
        outcome_proxy = imgs[proxy_image_idx_arr].reshape((n_sample, 64 * 64)).astype(np.float64)
        outcome_proxy += rng.normal(0.0, 0.1, outcome_proxy.shape)

        structural = self.structural_func(treatment, weights)
        outcome = structural * (posY_id_arr - 15.5) ** 2 / 85.25 + rng.normal(0.0, 0.5, n_sample)
        outcome = outcome[:, np.newaxis]
        if generate_test:
            do_A, EY_do_A = self.generate_test_dsprite(data_path)
        else:
            do_A, EY_do_A = None, None
        return treatment, outcome, treatment_proxy, outcome_proxy, do_A, EY_do_A


class dSprite_ProxyVariable_DatasetV2_ATT:
    """
    Corrected Deep PCL dSprite benchmark, extended with an ATT evaluation target.
    Training DGP is unchanged relative to the corrected ATE setup.
    """

    def __init__(self):
        pass

    def cal_weight(self):
        weights = np.empty((64, 64))
        for i in range(64):
            for j in range(64):
                weights[i, j] = np.abs(32 - j)
        return weights.reshape(64 * 64, 1) / 32.0

    def image_id(
        self,
        latent_bases: np.ndarray,
        posX_id_arr: np.ndarray,
        posY_id_arr: np.ndarray,
        orientation_id_arr: np.ndarray,
        scale_id_arr: np.ndarray,
    ):
        data_size = posX_id_arr.shape[0]
        color_id_arr = np.zeros(data_size, dtype=int)
        shape_id_arr = np.full(data_size, 2, dtype=int)  # heart
        idx = np.c_[
            color_id_arr,
            shape_id_arr,
            scale_id_arr,
            orientation_id_arr,
            posX_id_arr,
            posY_id_arr,
        ]
        return idx.dot(latent_bases)

    def structural_func(self, image, weights):
        return (image.dot(weights)[:, 0] ** 2 - 3000.0) / 500.0

    def nearest_grid_id(self, x: float, n_grid: int = 32) -> int:
        return int(np.clip(np.round(x * (n_grid - 1)), 0, n_grid - 1))

    def get_anchor_ids(
        self,
        anchor_posX: float = 0.6,
        anchor_posY: float = 0.6,
        anchor_scale_id: int = 3,      # scale = 0.8
        anchor_orientation_id: int = 0 # rotation = 0
    ):
        return {
            "posX_id": self.nearest_grid_id(anchor_posX),
            "posY_id": self.nearest_grid_id(anchor_posY),
            "scale_id": anchor_scale_id,
            "orientation_id": anchor_orientation_id,
        }

    def get_clean_image(
        self,
        imgs: np.ndarray,
        latents_bases: np.ndarray,
        posX_id: int,
        posY_id: int,
        orientation_id: int,
        scale_id: int,
    ):
        idx = self.image_id(
            latents_bases,
            np.array([posX_id]),
            np.array([posY_id]),
            np.array([orientation_id]),
            np.array([scale_id]),
        )
        return imgs[idx].reshape(1, 64 * 64).astype(np.float64)

    def att_scaling_from_anchor(self, anchor_posY_id: int) -> float:
        return ((anchor_posY_id - 15.5) ** 2) / 85.25

    def generate_test_dsprite_ate(self, data_path: str):
        dataset_zip = np.load(
            os.path.join(data_path, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
            allow_pickle=True,
            encoding="bytes",
        )

        weights = self.cal_weight()
        imgs = dataset_zip["imgs"]
        metadata = dataset_zip["metadata"][()]

        latents_sizes = metadata[b"latents_sizes"]
        latents_bases = np.concatenate(
            (latents_sizes[::-1].cumprod()[::-1][1:], np.array([1]))
        )

        posX_id_arr = [0, 5, 10, 15, 20, 25, 30]
        posY_id_arr = [0, 5, 10, 15, 20, 25, 30]
        scale_id_arr = [0, 3, 5]
        orientation_arr = [0, 10, 20, 30]

        latent_idx_arr = []
        for posX, posY, scale, orientation in product(
            posX_id_arr, posY_id_arr, scale_id_arr, orientation_arr
        ):
            latent_idx_arr.append([0, 2, scale, orientation, posX, posY])

        image_idx_arr = np.array(latent_idx_arr).dot(latents_bases)
        data_size = 7 * 7 * 3 * 4

        treatment = imgs[image_idx_arr].reshape((data_size, 64 * 64)).astype(np.float64)
        structural = self.structural_func(treatment, weights)[:, np.newaxis]
        return treatment, structural

    def generate_test_dsprite_att(
        self,
        data_path: str,
        anchor_posX: float = 0.6,
        anchor_posY: float = 0.6,
        anchor_scale_id: int = 3,
        anchor_orientation_id: int = 0,
    ):
        """
        Returns:
            do_A: test treatments a on the standard 588-point grid
            EY_att: oracle ATT curve f_ATT(a; a')
            A_anchor: fixed anchor treatment a'
            anchor_info: dict with the anchor latent IDs
        """
        dataset_zip = np.load(
            os.path.join(data_path, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
            allow_pickle=True,
            encoding="bytes",
        )

        imgs = dataset_zip["imgs"]
        metadata = dataset_zip["metadata"][()]
        latents_sizes = metadata[b"latents_sizes"]
        latents_bases = np.concatenate(
            (latents_sizes[::-1].cumprod()[::-1][1:], np.array([1]))
        )

        weights = self.cal_weight()

        # Standard test grid for a
        do_A, structural = self.generate_test_dsprite_ate(data_path)

        # Fixed anchor a'
        anchor = self.get_anchor_ids(
            anchor_posX=anchor_posX,
            anchor_posY=anchor_posY,
            anchor_scale_id=anchor_scale_id,
            anchor_orientation_id=anchor_orientation_id,
        )

        A_anchor = self.get_clean_image(
            imgs,
            latents_bases,
            posX_id=anchor["posX_id"],
            posY_id=anchor["posY_id"],
            orientation_id=anchor["orientation_id"],
            scale_id=anchor["scale_id"],
        )

        # Oracle ATT curve: scaling fixed by anchor posY
        c_anchor = self.att_scaling_from_anchor(anchor["posY_id"])
        EY_att = c_anchor * structural

        return do_A, EY_att, A_anchor, anchor

    def generate_dsprite_pv(
        self,
        data_path: str,
        n_sample: int,
        generate_test: bool = False,
        rand_seed: int = 42,
        **kwargs,
    ):
        dataset_zip = np.load(
            os.path.join(data_path, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
            allow_pickle=True,
            encoding="bytes",
        )
        weights = self.cal_weight()
        imgs = dataset_zip["imgs"]
        latents_values = dataset_zip["latents_values"]
        metadata = dataset_zip["metadata"][()]

        latents_sizes = metadata[b"latents_sizes"]
        latents_bases = np.concatenate(
            (latents_sizes[::-1].cumprod()[::-1][1:], np.array([1]))
        )

        rng = default_rng(seed=rand_seed)
        posX_id_arr = rng.integers(32, size=n_sample)
        posY_id_arr = rng.integers(32, size=n_sample)
        scale_id_arr = rng.integers(6, size=n_sample)
        orientation_arr = rng.integers(40, size=n_sample)

        image_idx_arr = self.image_id(
            latents_bases, posX_id_arr, posY_id_arr, orientation_arr, scale_id_arr
        )

        treatment = imgs[image_idx_arr].reshape((n_sample, 64 * 64)).astype(np.float64)
        treatment += rng.normal(0.0, 0.1, treatment.shape)

        latent_feature = latents_values[image_idx_arr]  # (color, shape, scale, rotation, posX, posY)
        treatment_proxy = latent_feature[:, 2:5]        # (scale, rotation, posX)

        posX_id_proxy = np.full(n_sample, 16, dtype=int)
        scale_id_proxy = np.full(n_sample, 3, dtype=int)
        orientation_proxy = np.zeros(n_sample, dtype=int)

        proxy_image_idx_arr = self.image_id(
            latents_bases,
            posX_id_proxy,
            posY_id_arr,
            orientation_proxy,
            scale_id_proxy,
        )

        outcome_proxy = imgs[proxy_image_idx_arr].reshape((n_sample, 64 * 64)).astype(np.float64)
        outcome_proxy += rng.normal(0.0, 0.1, outcome_proxy.shape)

        structural = self.structural_func(treatment, weights)
        outcome = structural * ((posY_id_arr - 15.5) ** 2) / 85.25 + rng.normal(0.0, 0.5, n_sample)
        outcome = outcome[:, np.newaxis]

        if generate_test:
            do_A, EY_do_A = self.generate_test_dsprite_ate(data_path)
        else:
            do_A, EY_do_A = None, None

        return treatment, outcome, treatment_proxy, outcome_proxy, do_A, EY_do_A

    def generate_dsprite_pv_att(
        self,
        data_path: str,
        n_sample: int,
        generate_test: bool = False,
        rand_seed: int = 42,
        anchor_posX: float = 0.6,
        anchor_posY: float = 0.6,
        anchor_scale_id: int = 3,
        anchor_orientation_id: int = 0,
        **kwargs,
    ):
        """
        Same training DGP as generate_dsprite_pv, but returns ATT test objects.
        """
        A, Y, Z, W, _, _ = self.generate_dsprite_pv(
            data_path=data_path,
            n_sample=n_sample,
            generate_test=False,
            rand_seed=rand_seed,
            **kwargs,
        )

        if generate_test:
            do_A, EY_do_A, A_anchor, anchor_info = self.generate_test_dsprite_att(
                data_path=data_path,
                anchor_posX=anchor_posX,
                anchor_posY=anchor_posY,
                anchor_scale_id=anchor_scale_id,
                anchor_orientation_id=anchor_orientation_id,
            )
        else:
            do_A, EY_do_A, A_anchor, anchor_info = None, None, None, None

        return A, Y, Z, W, do_A, EY_do_A, A_anchor, anchor_info


def generate_synthetic_PCL_ATE_data(size = 1000, beta = 1, sigma = 1, do_A_range = (-1, 2), do_A_size = 100, seed = 10):
    """
    This data generation process is taken from Appendix H of the following paper:
    Doubly Robust Proximal Causal Learning for Continuous Treatments, Yong Wu, Yanwei Fu, Shouyan Wang, Xinwei Sun
    """
    np.random.seed(seed)

    U2 = np.random.uniform(-1, 2, size = size)
    U1 = np.random.uniform(0, 1, size = size) - ((U2 >= 0) & (U2 <= 1)).astype(int)
    U = np.c_[U1, U2]

    Z2 = U2 + np.random.uniform(-1, 1, size = size)
    Z1 = U1 + np.random.normal(0, sigma, size = size)
    Z = np.c_[Z1, Z2]

    W1 = U1 + np.random.uniform(-1, 1, size = size)
    W2 = U2 + np.random.normal(0, sigma, size = size)
    W = np.c_[W1, W2]

    A = U2 + np.random.normal(0, beta, size = size)
    Y = 3 * np.cos(2 * (0.3 * U1 + 0.3 * U2 + 0.2) + 1.5 * A) + np.random.normal(0, 1, size = size)

    A, Y = A.reshape(size, -1), Y.reshape(size, -1)
    do_A = np.linspace(do_A_range[0], do_A_range[1], do_A_size)

    EY_do_A = []
    for a_ in do_A:
        U2 = np.random.uniform(-1, 2, size = 10000)
        U1 = np.random.uniform(0, 1, size = 10000) - ((U2 >= 0) & (U2 <= 1)).astype(int)
        EY_do_A.append(np.mean(3 * np.cos(2 * ( 0.3 * U1 + 0.3 * U2 + 0.2) + 1.5 * a_)))
    EY_do_A = np.array(EY_do_A)

    do_A, EY_do_A = do_A.reshape(do_A_size, -1), EY_do_A.reshape(do_A_size, -1)
    return U, W, Z, A, Y, do_A, EY_do_A


def gaussian_pdf(x: np.ndarray, mean: np.ndarray, std: float) -> np.ndarray:
    z = (x - mean) / std
    return np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * std)


@dataclass
class SyntheticLowDimATTBenchmark:
    """
    Synthetic low-dimensional proxy-causal ATT benchmark.

    Observational DGP:
        U1 ~ Unif[-1, 2]
        V  ~ Unif[0, 1]
        U2 = V - 1{0 <= U1 <= 1}

        W1 = U2 + Unif[-1, 1]
        W2 = U1 + N(0, 1)

        Z1 = U2 + N(0, 1)
        Z2 = U1 + Unif[-1, 1)

        A  = U1 + N(0, sigma_A^2)
        Y  = 3 cos( 2(0.3 U2 + 0.3 U1 + 0.2) + 1.5 A ) + N(0, sigma_Y^2)

    Target:
        f_ATT(a; a') = E[Y(a) | A = a']

    We compute the ground-truth ATT curve by:
        1) integrating out U2 analytically given U1;
        2) integrating over the posterior density p(U1 | A=a') numerically.
    """

    u1_low: float = -1.0
    u1_high: float = 2.0
    a_noise_std: float = 1.0
    y_noise_std: float = 1.0
    posterior_grid_size: int = 4001

    # ------------------------------------------------------------------
    # Latent / structural parts
    # ------------------------------------------------------------------
    def _sample_latents(self, n: int, rng: np.random.Generator):
        u1 = rng.uniform(self.u1_low, self.u1_high, size=n)
        v = rng.uniform(0.0, 1.0, size=n)
        u2 = v - ((u1 >= 0.0) & (u1 <= 1.0)).astype(np.float64)
        return u1, u2

    def structural_mean(self, a: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
        """
        Noise-free conditional mean Y(a) given U1,U2.
        """
        return 3.0 * np.cos(2.0 * (0.3 * u2 + 0.3 * u1 + 0.2) + 1.5 * a)

    def _mean_over_u2_given_u1(self, a: np.ndarray, u1: np.ndarray) -> np.ndarray:
        """
        Computes E[Y(a) | U1=u1] exactly by integrating out U2 analytically.

        Since U2 = V - 1{0 <= U1 <= 1}, with V ~ Unif[0,1], we have
            Y(a) = 3 cos(0.6 V + c(u1,a)) + noise,
        so
            E[Y(a) | U1=u1] = 3 * [sin(c+0.6) - sin(c)] / 0.6
        """
        a = np.asarray(a, dtype=np.float64).reshape(-1, 1)      # (n_a, 1)
        u1 = np.asarray(u1, dtype=np.float64).reshape(1, -1)    # (1, n_u)

        indicator = ((u1 >= 0.0) & (u1 <= 1.0)).astype(np.float64)
        c = 0.6 * (u1 - indicator) + 0.4 + 1.5 * a              # broadcasted
        return 3.0 * (np.sin(c + 0.6) - np.sin(c)) / 0.6        # (n_a, n_u)

    def posterior_u1_given_anchor(self, anchor_a: float):
        """
        Posterior density p(U1 | A = anchor_a) on a dense grid.

        Since U1 ~ Unif[-1,2] and A|U1=u1 ~ N(u1, sigma_A^2),
        the posterior on [-1,2] is proportional to the Gaussian likelihood.
        """
        u1_grid = np.linspace(self.u1_low, self.u1_high, self.posterior_grid_size)
        post_unnorm = gaussian_pdf(anchor_a, mean=u1_grid, std=self.a_noise_std)
        post = post_unnorm / np.trapezoid(post_unnorm, u1_grid)
        return u1_grid, post

    # ------------------------------------------------------------------
    # Observational data
    # ------------------------------------------------------------------
    def generate_observational_data(self, n: int, seed: int = 0):
        rng = default_rng(seed)
        u1, u2 = self._sample_latents(n, rng)

        # outcome proxy W
        w1 = u2 + rng.uniform(-1.0, 1.0, size=n)
        w2 = u1 + rng.normal(0.0, 1.0, size=n)

        # treatment proxy Z
        z1 = u2 + rng.normal(0.0, 1.0, size=n)
        z2 = u1 + rng.uniform(-1.0, 1.0, size=n)

        # observed treatment and outcome
        a = u1 + rng.normal(0.0, self.a_noise_std, size=n)
        y = self.structural_mean(a, u1, u2) + rng.normal(0.0, self.y_noise_std, size=n)

        W = np.column_stack([w1, w2]).astype(np.float64)
        Z = np.column_stack([z1, z2]).astype(np.float64)
        A = a.reshape(-1, 1).astype(np.float64)
        Y = y.reshape(-1, 1).astype(np.float64)

        return {
            "U1": u1.reshape(-1, 1).astype(np.float64),
            "U2": u2.reshape(-1, 1).astype(np.float64),
            "W": W,
            "Z": Z,
            "A": A,
            "Y": Y,
        }

    # ------------------------------------------------------------------
    # Ground-truth curves
    # ------------------------------------------------------------------
    def generate_ate_curve(self, do_A_range=(-1.0, 2.0), do_A_size: int = 100):
        """
        Optional: population-level dose-response for comparison.
        """
        do_A = np.linspace(do_A_range[0], do_A_range[1], do_A_size, dtype=np.float64)
        u1_grid = np.linspace(self.u1_low, self.u1_high, self.posterior_grid_size)
        prior_u1 = np.ones_like(u1_grid) / (self.u1_high - self.u1_low)

        mean_surface = self._mean_over_u2_given_u1(do_A, u1_grid)   # (n_a, n_u)
        ey_ate = np.trapezoid(mean_surface * prior_u1.reshape(1, -1), u1_grid, axis=1)

        return do_A.reshape(-1, 1), ey_ate.reshape(-1, 1)

    def generate_att_curve(
        self,
        anchor_a: float,
        do_A_range=(-1.0, 2.0),
        do_A_size: int = 100,
    ):
        """
        Ground-truth ATT curve:
            f_ATT(a; a') = E[Y(a) | A = a']
        """
        do_A = np.linspace(do_A_range[0], do_A_range[1], do_A_size, dtype=np.float64)
        u1_grid, post_u1 = self.posterior_u1_given_anchor(anchor_a)

        mean_surface = self._mean_over_u2_given_u1(do_A, u1_grid)   # (n_a, n_u)
        ey_att = np.trapezoid(mean_surface * post_u1.reshape(1, -1), u1_grid, axis=1)

        return do_A.reshape(-1, 1), ey_att.reshape(-1, 1)

    # ------------------------------------------------------------------
    # Full benchmark object
    # ------------------------------------------------------------------
    def generate_dataset(
        self,
        n: int,
        seed: int = 0,
        anchor_a: float = 0.25,
        do_A_range=(-1.0, 2.0),
        do_A_size: int = 100,
        include_ate_curve: bool = False,
    ):
        obs = self.generate_observational_data(n=n, seed=seed)
        do_A, EY_att = self.generate_att_curve(
            anchor_a=anchor_a,
            do_A_range=do_A_range,
            do_A_size=do_A_size,
        )

        out = {
            "A": obs["A"],
            "Y": obs["Y"],
            "Z": obs["Z"],
            "W": obs["W"],
            "A_anchor": np.array([[anchor_a]], dtype=np.float64),
            "do_A": do_A,
            "EY_att": EY_att,
        }

        if include_ate_curve:
            _, EY_ate = self.generate_ate_curve(
                do_A_range=do_A_range,
                do_A_size=do_A_size,
            )
            out["EY_ate"] = EY_ate

        return out


def read_legalized_abortion_and_crime_dataset(data_path: str,
                                              return_test: bool = False,
                                              seed: int = 0):
    seed_str = str(seed)
    folder_path_train = data_path + '/train'
    folder_path_effect = data_path + '/test'

    train_name = f'main_ab_seed{seed_str}.npz'
    train_path = f'{folder_path_train}/{train_name}'
    train_data = np.load(train_path)
    W, Z, A, Y = train_data['train_w'], train_data['train_z'], train_data['train_a'], train_data['train_y']

    effect_name = f'do_A_ab_seed{seed_str}.npz'
    effect_path = f'{folder_path_effect}/{effect_name}'
    effect_data = np.load(effect_path)
    do_A, EY_do_A = effect_data['do_A'], effect_data['gt_EY_do_A']
    if return_test:
        W_test, Z_test, A_test, Y_test = train_data['test_w'], train_data['test_z'], train_data['test_a'], train_data['test_y']
        return W, Z, A, Y, W_test, Z_test, A_test, Y_test, do_A, EY_do_A
    else:
        return W, Z, A, Y, do_A, EY_do_A


def read_deaner_dataset(data_path: str,
                        id_: str,
                        seed: int,
                        return_test: bool = False):
    id_path = id_ + "_80_N"
    data_path = os.path.join(data_path, id_path)
    npz_train_file = f"main_edu_{id_}_80_seed{seed}.npz"
    npz_effect_file = f"do_A_edu_{id_}_80_seed{seed}.npz"
    train_data = np.load(os.path.join(data_path, npz_train_file))
    effect_data = np.load(os.path.join(data_path, npz_effect_file))
    W, Z, A, Y = train_data['train_w'], train_data['train_z'], train_data['train_a'], train_data['train_y']
    do_A, EY_do_A = effect_data['do_A'], effect_data['gt_EY_do_A'].reshape(-1, 1)
    filtered_indices = ((A > -0.1) & (A < 2.0)).reshape(-1)
    doA_filtered_indices = ((do_A > -0.1) & (do_A < 2.0)).reshape(-1)
    W, Z, A, Y, do_A, EY_do_A = W[filtered_indices], Z[filtered_indices], A[filtered_indices], Y[filtered_indices], do_A[doA_filtered_indices], EY_do_A[doA_filtered_indices]

    if return_test:
        W_test, Z_test, A_test, Y_test = train_data['test_w'], train_data['test_z'], train_data['test_a'], train_data['test_y']
        return W, Z, A, Y, W_test, Z_test, A_test, Y_test, do_A, EY_do_A
    else:
        return W, Z, A, Y, do_A, EY_do_A


def read_rhc_dataset(data_path: str, split: str = "train", use_all_X: bool = False):
    """Right Heart Catheterization (RHC) proximal-causal dataset.

    Real clinical benchmark from the SUPPORT study, used for proximal causal inference by
    Miao et al., Cui et al., and Zhang et al. (2025, DeepMMR). The CSVs are the preprocessed
    70/15/15 train/val/test splits shipped with the DeepMMR repo (continuous covariates already
    standardised/log-transformed, binary/categorical label-encoded).

    Variable roles (proximal setup):
      - A : ``swang1``  -- BINARY treatment (whether right heart catheterisation was performed).
      - Y : ``t3d30``   -- outcome, survival time in days over the first 30 days in the ICU.
      - Z : ``(pafi1, paco21)``  -- 2-D treatment proxy (physiological measurements).
      - W : ``(ph1, hema1)``     -- 2-D outcome proxy (physiological measurements).
      - X : backdoor covariates -- 20 significant features (``use_all_X=False``) or ~52 (``True``).

    Because the treatment is binary the estimand is the ATE ``E[Y(1)] - E[Y(0)]`` (there is no
    dose-response curve), so ``do_A = [0, 1]`` and there is **no ground-truth** ``EY_do_A`` (real
    data). The proximal ATE reported in the literature is ~ -2.2 to -2.6 days (RHC lowers 30-day
    survival), more negative than the confounded naive difference-in-means (~ -1.5).

    Parameters
    ----------
    data_path : str
        Directory containing ``rhc_{split}.csv`` and the ``RHC_X_*features_list.csv`` files.
    split : {"train", "val", "test"}
        Which preprocessed split to load.
    use_all_X : bool
        If True use all ~52 covariates as the backdoor X; else the 20 significant features
        (the choice used in the DeepMMR paper).

    Returns
    -------
    W, Z, A, Y, X : np.ndarray
        Arrays of shape (n, 2), (n, 2), (n, 1), (n, 1), (n, d_X).
    do_A : np.ndarray
        The two treatment levels ``[[0.], [1.]]`` at which the structural function is evaluated;
        the ATE is ``f(1) - f(0)``.
    """
    df = pd.read_csv(os.path.join(data_path, f"rhc_{split}.csv"))
    x_file = "RHC_X_allfeatures_list.csv" if use_all_X else "RHC_X_significantfeatures_list.csv"
    X_names = pd.read_csv(os.path.join(data_path, x_file)).variable.tolist()

    A = df["swang1"].to_numpy().reshape(-1, 1).astype(np.float64)
    Y = df["t3d30"].to_numpy().reshape(-1, 1).astype(np.float64)
    Z = df[["pafi1", "paco21"]].to_numpy().astype(np.float64)
    W = df[["ph1", "hema1"]].to_numpy().astype(np.float64)
    X = df[X_names].to_numpy().astype(np.float64)
    do_A = np.array([[0.0], [1.0]])   # binary treatment: ATE = f(1) - f(0)
    return W, Z, A, Y, X, do_A


def generate_demand_experiment_data(size = 1000, epsilon_std = 1.0, do_A_range = (10, 30), do_A_size = 10, seed = 10):

    def psi(t: np.ndarray) -> np.ndarray:
        return 2 * ((t - 5) ** 4 / 600 + np.exp(-4 * (t - 5) ** 2) + t / 10 - 2)

    def cal_outcome(price, views, demand):
        return np.clip(np.exp((views - price) / 10.0), None, 5.0) * price - 5 * psi(demand)

    np.random.seed(seed)

    demand = np.random.uniform(0, 10, size)
    cost1 = 2 * np.sin(demand * np.pi * 2 / 10) + np.random.normal(0, epsilon_std, size)
    cost2 = 2 * np.cos(demand * np.pi * 2 / 10) + np.random.normal(0, epsilon_std, size)
    price = 35 + (cost1 + 3) * psi(demand) + cost2 + np.random.normal(0, epsilon_std, size)
    views = 7 * psi(demand) + 45 + np.random.normal(0, epsilon_std, size)
    outcome = cal_outcome(price, views, demand) + np.random.normal(0, epsilon_std, size)

    A, Z, W, Y, U = price[:, np.newaxis], np.c_[cost1, cost2], views[:, np.newaxis], outcome[:, np.newaxis], demand[:, np.newaxis]

    do_A = np.linspace(do_A_range[0], do_A_range[1], do_A_size)
    EY_do_A = []
    for a_ in do_A:
        demand = np.random.uniform(0, 10, 10000)
        views = 7 * psi(demand) + 45 + np.random.normal(0, epsilon_std, 10000)
        EY_do_A.append(np.mean(cal_outcome(a_, views, demand)))

    EY_do_A = np.array(EY_do_A)

    do_A, EY_do_A = do_A.reshape(do_A_size, -1), EY_do_A.reshape(do_A_size, -1)

    return U, W, Z, A, Y, do_A, EY_do_A


def generate_synthetic_CATE_data(data_size: int,
                                 sigma: float = 0.5,
                                 uniform_noise_upper_bound: float = 0.5,
                                 uniform_noise_lower_bound: float = -0.5,
                                 covariate_v_range = (-0.5, 0.5), covariate_v_size: int = 100,
                                 seed: int = 42):
    """
    NOTE: this construction follows the DFPV paper (Xu et al., 2021), which itself adapts it from earlier work.
    """
    rng = default_rng(seed=seed)
    V = rng.uniform(low=-0.5, high=0.5, size=(data_size,))
    U1 = 1 + 2 * V + rng.uniform(low=-0.5, high=0.5, size=(data_size,))
    U2 = 1 + 2 * V + rng.uniform(low=-0.5, high=0.5, size=(data_size,))
    U3 = (V - 1) ** 2 + rng.uniform(low=-0.5, high=0.5, size=(data_size,))
    U = np.c_[U1, U2, U3]
    prob = 1.0 / (1.0 + np.exp(-0.5 * (V + U1 + U2 + U3)))
    A = (rng.random(data_size) < prob).astype(float)
    Y = V * U1 * U2 * U3 + rng.normal(0.0, 0.25, size=(data_size, ))
    Y *= A

    V = V.reshape(-1, 1)
    A, Y = A.reshape(-1, 1), Y.reshape(-1, 1)

    Z2 = U2 + rng.uniform(uniform_noise_lower_bound, uniform_noise_upper_bound, size = data_size)
    Z1 = U1 + rng.normal(0, sigma, size = data_size)
    Z3 = U3 + rng.uniform(uniform_noise_lower_bound, uniform_noise_upper_bound, size = data_size)
    Z = np.c_[Z1, Z2, Z3]

    W1 = U1 + rng.uniform(uniform_noise_lower_bound, uniform_noise_upper_bound, size = data_size)
    W2 = U2 + rng.normal(0, sigma, size = data_size)
    W3 = U3 + rng.normal(0, sigma / 2, size = data_size)
    W = np.c_[W1, W2, W3]

    # covariate_v_test = np.array([-0.4, -0.2, 0.0, 0.2, 0.4]).reshape(-1, 1)
    covariate_v_test = np.linspace(covariate_v_range[0], covariate_v_range[1], covariate_v_size).reshape(-1, 1)
    # do_A = np.array([1, 1, 1, 1, 1]).reshape(-1, 1)  # only test A = 1
    do_A = np.ones(shape = covariate_v_size).reshape(-1, 1)
    EY_do_A_CATE = covariate_v_test * ((1 + 2 * covariate_v_test) ** 2) * ((covariate_v_test - 1) ** 2)

    return U, W, Z, V, A, Y, covariate_v_test, do_A, EY_do_A_CATE


class ColorMNIST_ProxyVariable_Dataset:
    """
    Proximal ColorMNIST data-generating process with a STRUCTURED (image) outcome.

    This is the proxy-variable ("proximal causal inference") extension of the ColorMNIST design
    of Dance et al. (arXiv:2605.07665, Appendix B.2.4). A single scalar confounder ``X`` sets
    both the foreground colour of an MNIST digit and the treatment, so colour and treatment are
    entangled in the observational law. The treatment ``A`` selects the digit class, and the
    outcome ``Y`` is the resulting coloured digit image -- i.e. the outcome is structured.

    Causal graph (``X`` is UNOBSERVED; a proxy-causal-learning method uses only ``A, Y, Z, W``):

        X ~ Uniform[0, 1]                               latent colour confounder
        A | X ~ Bernoulli(sigmoid(w_conf (X - 1/2)))    binary treatment (0 -> digit d0, 1 -> digit d1)
        Z = X + sigma_z * eps_z,   eps_z ~ N(0, 1)      treatment-inducing proxy (noisy view of X)
        W = X + sigma_w * eps_w,   eps_w ~ N(0, 1)      outcome-inducing proxy  (noisy view of X)
        Y = recolour(S_{d_A}, X)                        structured outcome (a coloured digit image)

    The recolouring paints a grayscale digit ``S`` with a foreground colour that slides along the
    red-blue axis with ``X`` and keeps a black background:

        M(S)    = sigmoid(k_mask (S - tau))             soft foreground mask
        c_fg(X) = (X, alpha, 1 - X)                     red (X ~ 1) to blue (X ~ 0), fixed green
        Y       = M(S) * (c_fg(X) * S) + (1 - M(S)) * c_bg,   c_bg = (0, 0, 0)  (black).

    Because a large ``X`` makes BOTH the treated class ``d1`` and a red foreground more likely, the
    treated outcomes look redder than the controls in the logged data -- the treatment is confounded
    with the outcome's colour. Intervening on ``A`` re-selects the digit class while leaving
    ``X ~ Uniform[0, 1]``, which balances the colour: that is the bias a proximal method removes with
    the two proxies. ``Z`` and ``W`` are conditionally independent given ``X`` (each carries its own
    measurement noise) -- exactly the two-proxy structure that proximal identification needs
    (Miao, Geng & Tchetgen Tchetgen, 2018).

    The red-blue pixel ratio ``R / (R + B)`` recovers ``X`` almost exactly (black background), so it
    is a convenient scalar summary of an outcome's colour -- see ``color_summary``.
    """

    def __init__(self, digits=(1, 6), w_conf=5.0, tau=0.5, k_mask=20.0, alpha=0.2,
                 sigma_z=0.1, sigma_w=0.1):
        """
        Parameters
        ----------
        digits : tuple of two int
            MNIST digit classes selected by A = 0 and A = 1 (default: digit 1 vs digit 6).
        w_conf : float
            Confounding strength in the propensity sigmoid(w_conf (X - 1/2)).
        tau, k_mask : float
            Threshold and sharpness of the soft foreground mask M(S) = sigmoid(k_mask (S - tau)).
        alpha : float
            The fixed green channel of the foreground colour c_fg(X) = (X, alpha, 1 - X).
        sigma_z, sigma_w : float
            Gaussian measurement-noise scales of the treatment proxy Z and the outcome proxy W.
        """
        self.digits = digits
        self.w_conf = w_conf
        self.tau = tau
        self.k_mask = k_mask
        self.alpha = alpha
        self.sigma_z = sigma_z
        self.sigma_w = sigma_w
        self.image_shape = (3, 28, 28)         # channels-first; a flattened outcome is 3 * 28 * 28 = 2352 long

    def load_digit_pools(self, data_path):
        """
        Load grayscale MNIST digits, grouped into one pool per treated class.

        Parameters
        ----------
        data_path : str
            torchvision MNIST root (downloads on first use).

        Returns
        -------
        list of np.ndarray
            One (n_d, 28, 28) array in [0, 1] per digit class in ``self.digits``.
        """
        from torchvision import datasets       # imported lazily: only this DGP needs torchvision
        mnist = datasets.MNIST(root=data_path, train=True, download=True)
        images = mnist.data.numpy().astype(np.float64) / 255.0
        targets = mnist.targets.numpy()
        return [images[targets == digit] for digit in self.digits]

    def recolor(self, S, x):
        """
        Paint grayscale digits with the confounder-dependent foreground colour c_fg(x).

        Parameters
        ----------
        S : np.ndarray of shape (n, 28, 28), values in [0, 1]
            Grayscale digits.
        x : np.ndarray of shape (n,)
            Confounder values, one per digit.

        Returns
        -------
        np.ndarray of shape (n, 3, 28, 28) in [0, 1].
        """
        mask = 1.0 / (1.0 + np.exp(-self.k_mask * (S - self.tau)))       # (n, 28, 28) soft foreground
        mask = mask[:, None, :, :]                                       # (n, 1, 28, 28)

        foreground_color = np.stack([x, np.full_like(x, self.alpha), 1.0 - x], axis=1)   # (n, 3)
        foreground = foreground_color[:, :, None, None] * S[:, None, :, :]               # (n, 3, 28, 28)

        background_color = 0.0                                           # black background, c_bg = (0, 0, 0)
        return mask * foreground + (1.0 - mask) * background_color

    def sample_outcomes(self, digit_pools, x, a, rng):
        """
        Draw the structured outcome Y | X = x, A = a: a random digit of class d_a, recoloured by x.

        Parameters
        ----------
        digit_pools : list of np.ndarray
            Output of ``load_digit_pools``.
        x : np.ndarray of shape (n,)
        a : np.ndarray of shape (n,), integer treatments in {0, 1}
        rng : numpy Generator

        Returns
        -------
        np.ndarray of shape (n, 3, 28, 28).
        """
        n_sample = x.shape[0]
        S = np.empty((n_sample, 28, 28), dtype=np.float64)
        for class_index in range(len(digit_pools)):
            in_this_class = (a == class_index)
            n_in_class = int(in_this_class.sum())
            if n_in_class > 0:
                pool = digit_pools[class_index]
                chosen = rng.integers(pool.shape[0], size=n_in_class)
                S[in_this_class] = pool[chosen]
        return self.recolor(S, x)

    def interventional_mean_image(self, digit_pools, a, n_mc, rng):
        """
        The true structural mean E[Y(a)] = E_{X ~ U[0, 1], S ~ pool_a}[ recolour(S, X) ].

        Estimated by Monte Carlo: draw ``n_mc`` fresh (X, digit) pairs of class d_a -- so the
        confounding is removed (X is resampled uniformly, independently of A) -- and average.

        Returns
        -------
        np.ndarray of shape (3, 28, 28), the mean interventional image.
        """
        x = rng.uniform(0.0, 1.0, size=n_mc)
        pool = digit_pools[a]
        chosen = rng.integers(pool.shape[0], size=n_mc)
        images = self.recolor(pool[chosen], x)                          # (n_mc, 3, 28, 28)
        return images.mean(axis=0)

    def color_summary(self, Y):
        """
        Red-blue pixel ratio R / (R + B) of each (flattened) outcome image -- recovers X.

        Parameters
        ----------
        Y : np.ndarray of shape (n, 3 * 28 * 28)
            Flattened outcome images.

        Returns
        -------
        np.ndarray of shape (n,), each entry in [0, 1].
        """
        images = Y.reshape(-1, *self.image_shape)
        red = images[:, 0].sum(axis=(1, 2))
        blue = images[:, 2].sum(axis=(1, 2))
        return red / (red + blue + 1e-6)

    def generate_colormnist_pv(self, data_path, n_sample, rand_seed=42,
                               generate_test=True, n_test_mc=5000):
        """
        Generate one proximal ColorMNIST dataset.

        Parameters
        ----------
        data_path : str
            torchvision MNIST root.
        n_sample : int
            Number of logged (observational) samples.
        rand_seed : int
            Random seed.
        generate_test : bool
            If True, also compute the two interventional mean images E[Y(0)], E[Y(1)] -- the
            structural "outcome curve" for a binary treatment.
        n_test_mc : int
            Monte-Carlo sample size for each interventional mean image.

        Returns
        -------
        A : np.ndarray (n_sample, 1)              binary treatment
        Y : np.ndarray (n_sample, 2352)           flattened coloured-digit outcome image (3 x 28 x 28)
        Z : np.ndarray (n_sample, 1)              treatment-inducing proxy
        W : np.ndarray (n_sample, 1)              outcome-inducing proxy
        X : np.ndarray (n_sample, 1)              UNOBSERVED confounder (returned only for diagnostics)
        do_A : np.ndarray (2, 1) or None          the two interventions [[0], [1]]
        EY_do_A : np.ndarray (2, 2352) or None    the true means [E[Y(0)], E[Y(1)]], flattened
        """
        rng = default_rng(seed=rand_seed)
        digit_pools = self.load_digit_pools(data_path)

        # 1. latent confounder
        X = rng.uniform(0.0, 1.0, size=n_sample)

        # 2. confounded binary treatment  A | X ~ Bernoulli(sigmoid(w_conf (X - 1/2)))
        propensity = 1.0 / (1.0 + np.exp(-self.w_conf * (X - 0.5)))
        A = (rng.uniform(size=n_sample) < propensity).astype(np.float64)

        # 3. two conditionally-independent-given-X proxies (each a noisy measurement of X)
        Z = X + self.sigma_z * rng.standard_normal(n_sample)
        W = X + self.sigma_w * rng.standard_normal(n_sample)

        # 4. structured outcome: a coloured digit of the treated class, recoloured by X
        Y = self.sample_outcomes(digit_pools, X, A.astype(int), rng)     # (n, 3, 28, 28)
        Y = Y.reshape(n_sample, -1)                                      # (n, 2352)

        # column vectors, matching the (n, dim) convention of the other proxy datasets
        A = A[:, None]
        Z = Z[:, None]
        W = W[:, None]
        X = X[:, None]

        # 5. ground-truth structural means E[Y(0)], E[Y(1)] (for evaluation / visualisation)
        if generate_test:
            do_A = np.array([[0.0], [1.0]])
            EY_do_A = np.stack([
                self.interventional_mean_image(digit_pools, 0, n_test_mc, rng).reshape(-1),
                self.interventional_mean_image(digit_pools, 1, n_test_mc, rng).reshape(-1),
            ], axis=0)                                                  # (2, 2352)
        else:
            do_A, EY_do_A = None, None

        return A, Y, Z, W, X, do_A, EY_do_A

class OCTMNIST_ProxyVariable_Dataset:
    """Proximal OCTMNIST DGP with a STRUCTURED (image) outcome and a NONZERO causal mean effect.

    Proxy-causal-learning adaptation of the OCTMNIST mean-matched image-location experiment of
    "Interpretable doubly robust mean-embedding testing" (arXiv:2605.08034, Appendix I.7). Two changes
    are made relative to that design:

    1. THE CONFOUNDER IS LATENT AND SEEN ONLY THROUGH PROXIES. The original assigns treatment from
       observed synthetic covariates X ~ N(0, I_6) -- a backdoor setting. Here the same 6-D variable is
       an UNOBSERVED confounder U, and the estimator sees only two noisy proxies
           Z = L_Z U + eta_Z,      W = L_W U + eta_W,
       conditionally independent given U -- the two-proxy structure proximal identification needs.
    2. THE CAUSAL MEAN EFFECT IS NONZERO. The original calibrates the residuals so that
       E[R(1) - R(0) | U] = 0: the effect is purely distributional and invisible to counterfactual means,
       which makes it unusable as a benchmark for counterfactual MEAN-image estimation. We keep that
       mean-matched fluid mechanism as a source of structured confounding, and add a second, spatially
       distinct template M_T that carries a known nonzero mean effect delta * M_T.

    Construction (Y is a 28x28 grayscale image, flattened to 784):

        U ~ N(0, I_dU)                                          latent confounder
        A | U ~ Bernoulli(clip(sigmoid(alpha' U), eps, 1-eps))  binary treatment
        Z = L_Z U + eta_Z,   W = L_W U + eta_W                  proxies
        B ~ uniform draw from the OCTMNIST *normal* class       baseline anatomy, independent of U
        q(U) = q_min + (q_max - q_min) sigmoid(beta' U)         subject-specific fluid probability
        S | U ~ Bernoulli(q(U))                                 severe-fluid indicator
        J_a  = exp(tau xi_a - tau^2 / 2),  xi_a ~ N(0,1)        mean-one amplitude jitter, E[J_a] = 1

        Y(0) = B + q(U) theta J_0 M_C            + eps_0
        Y(1) = B +   S   theta J_1 M_C + delta M_T + eps_1
        Y    = A Y(1) + (1 - A) Y(0)

    Because E[S | U] = q(U) and E[J_a] = 1, the M_C component has the SAME conditional mean in both arms,
    so it contributes nothing to the causal contrast while still differing in distribution:

        E[Y(0) | U] = E[B] + q(U) theta M_C
        E[Y(1) | U] = E[B] + q(U) theta M_C + delta M_T
        =>  CATE(U) = delta M_T,   and marginally   mu_1 - mu_0 = delta M_T   (EXACT, analytic)

    `alpha` and `beta` are positively aligned, so subjects with a larger fluid probability are also more
    likely to be treated. The naive observational contrast therefore picks up a spurious M_C component,

        E[Y | A=1] - E[Y | A=0] = delta M_T + theta {E[q(U)|A=1] - E[q(U)|A=0]} M_C,

    which is visually distinguishable from the true effect because M_C and M_T are near-orthogonal. A
    successful proximal estimator keeps the M_T component and removes the M_C one; the projections
    `effect_projections` returns make that diagnosis quantitative.

    NO PIXEL CLIPPING is applied to the numerical outcomes: clipping is nonlinear, so
    E[clip(Y(a))] != mu_a and the analytic ground truth would no longer hold. `theta`, `delta` and the
    noise scale are chosen to keep values in a sensible range; clipped copies are for display only
    (`to_display`).

    Two regimes, following the write-up:
      delta = 0  ("OCT-PCL-Null")   -- mu_1 - mu_0 = 0 while the observed contrast is strongly nonzero.
      delta > 0  ("OCT-PCL-Effect") -- the observed contrast mixes the true M_T and the spurious M_C.
    """

    OCTMNIST_URL = "https://zenodo.org/records/10519652/files/octmnist.npz?download=1"
    OCTMNIST_MD5 = "c68d92d5b585d8d81f7112f81e2d0842"
    LABEL_DME, LABEL_NORMAL = 1, 3

    def __init__(self, d_u: int = 6, d_z: int = 6, d_w: int = 6,
                 theta: float = 0.60, delta: float = 0.30,
                 q_min: float = 0.05, q_max: float = 0.45, tau: float = 0.20,
                 noise_std: float = 0.025, sigma_z: float = 0.5, sigma_w: float = 0.5,
                 propensity_clip: float = 0.07, confounding_strength: float = 1.2,
                 mask_percentile: float = 85.0, mask_sigma: float = 1.0,
                 effect_center=(20.0, 7.0), effect_sigma=(2.2, 2.6)):
        """
        Parameters
        ----------
        d_u, d_z, d_w : int
            Latent-confounder and proxy dimensions.
        theta : float
            Fluid (confounding) template strength.
        delta : float
            TRUE causal mean-effect strength: mu_1 - mu_0 = delta * M_T. Set 0 for the null regime.
        q_min, q_max : float
            Range of the subject-specific severe-fluid probability q(U).
        tau : float
            Log-scale of the mean-one amplitude jitter J_a.
        noise_std : float
            Pixel noise standard deviation.
        sigma_z, sigma_w : float
            Proxy measurement-noise standard deviations (smaller = more informative proxies).
        propensity_clip : float
            Propensity is clipped to [propensity_clip, 1 - propensity_clip] for overlap.
        confounding_strength : float
            Scale of the propensity logit; larger = stronger confounding.
        mask_percentile, mask_sigma : float
            Threshold percentile and smoothing for the DME-minus-normal fluid template M_C.
        effect_center, effect_sigma : tuple of float
            Centre (row, col) and widths of the synthetic lesion used as the causal template M_T. The
            default sits low-left, away from M_C's central support, to keep the two near-orthogonal.
        """
        self.d_u, self.d_z, self.d_w = d_u, d_z, d_w
        self.theta, self.delta = theta, delta
        self.q_min, self.q_max, self.tau = q_min, q_max, tau
        self.noise_std = noise_std
        self.sigma_z, self.sigma_w = sigma_z, sigma_w
        self.propensity_clip = propensity_clip
        self.confounding_strength = confounding_strength
        self.mask_percentile, self.mask_sigma = mask_percentile, mask_sigma
        self.effect_center, self.effect_sigma = effect_center, effect_sigma
        self.image_shape = (28, 28)
        self.M_C = None
        self.M_T = None

    # ------------------------------------------------------------------ data
    def download_octmnist(self, data_path):
        """Fetch octmnist.npz into `data_path` (verified by MD5); returns the file path."""
        import hashlib
        import urllib.request
        folder = os.path.join(data_path, "octmnist")
        os.makedirs(folder, exist_ok=True)
        target = os.path.join(folder, "octmnist.npz")

        def _md5_ok(path):
            digest = hashlib.md5()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            return digest.hexdigest() == self.OCTMNIST_MD5

        if os.path.exists(target) and _md5_ok(target):
            return target
        print(f"Downloading OCTMNIST to {target} ...")
        urllib.request.urlretrieve(self.OCTMNIST_URL, target)
        if not _md5_ok(target):
            raise ValueError("Downloaded OCTMNIST failed MD5 verification.")
        return target

    def load_octmnist(self, data_path):
        """Return (images in [0,1] of shape (n,28,28), integer labels)."""
        with np.load(self.download_octmnist(data_path), allow_pickle=True) as archive:
            images = np.concatenate([archive["train_images"], archive["val_images"],
                                     archive["test_images"]], axis=0)
            labels = np.concatenate([archive["train_labels"], archive["val_labels"],
                                     archive["test_labels"]], axis=0).reshape(-1)
        images = images.astype(np.float64)
        if images.max(initial=0.0) > 1.5:
            images = images / 255.0
        return np.clip(images, 0.0, 1.0), labels.astype(int)

    # ------------------------------------------------------------- templates
    def build_templates(self, normal_images, dme_images):
        """Build the confounding template M_C and the causal template M_T, both with ||.||_F = 1.

        M_C is the paper's fluid template: the positive part of the smoothed DME-minus-normal class-mean
        difference, tapered by a mild central prior and thresholded at `mask_percentile`. M_T is a
        synthetic localized lesion placed away from M_C's support, so the true causal pattern and the
        confounding pattern are visually and numerically separable (check `template_coherence`).
        """
        from scipy.ndimage import gaussian_filter
        height, width = self.image_shape
        rows, cols = np.mgrid[0:height, 0:width]

        difference = gaussian_filter(dme_images.mean(axis=0) - normal_images.mean(axis=0), sigma=1.0)
        positive = np.maximum(difference, 0.0)
        positive = positive * np.exp(-0.5 * ((rows - 11.5) / 5.0) ** 2
                                     - 0.5 * ((cols - 14.0) / 8.0) ** 2)
        nonzero = positive[positive > 0]
        threshold = float(np.percentile(nonzero, self.mask_percentile)) if nonzero.size else 0.0
        mask = gaussian_filter(np.where(positive >= threshold, positive, 0.0), sigma=self.mask_sigma)
        mask = np.maximum(mask, 0.0)

        centre_row, centre_col = self.effect_center
        sigma_row, sigma_col = self.effect_sigma
        lesion = np.exp(-0.5 * ((rows - centre_row) / sigma_row) ** 2
                        - 0.5 * ((cols - centre_col) / sigma_col) ** 2)

        self.M_C = mask / (np.linalg.norm(mask) + 1e-12)
        self.M_T = lesion / (np.linalg.norm(lesion) + 1e-12)
        return self.M_C, self.M_T

    @property
    def template_coherence(self):
        """|<M_C, M_T>| / (||M_C|| ||M_T||) -- small means the two patterns are separable."""
        return float(abs(np.sum(self.M_C * self.M_T)) /
                     (np.linalg.norm(self.M_C) * np.linalg.norm(self.M_T) + 1e-12))

    # ------------------------------------------------------------------- DGP
    def _q_of_u(self, U, beta):
        return self.q_min + (self.q_max - self.q_min) / (1.0 + np.exp(-(U @ beta)))

    def _sample_arms(self, rng, U, baseline_pool):
        """Both potential outcomes for the given latent draws (no clipping)."""
        n = U.shape[0]
        B = baseline_pool[rng.choice(len(baseline_pool), size=n, replace=True)]
        q = self._q_of_u(U, self._beta)
        S = rng.binomial(1, q).astype(np.float64)
        jitter0 = np.exp(self.tau * rng.standard_normal(n) - 0.5 * self.tau ** 2)
        jitter1 = np.exp(self.tau * rng.standard_normal(n) - 0.5 * self.tau ** 2)
        noise0 = rng.normal(0.0, self.noise_std, size=B.shape)
        noise1 = rng.normal(0.0, self.noise_std, size=B.shape)

        Y0 = B + (q * self.theta * jitter0)[:, None, None] * self.M_C[None] + noise0
        Y1 = (B + (S * self.theta * jitter1)[:, None, None] * self.M_C[None]
              + self.delta * self.M_T[None] + noise1)
        return Y0, Y1, q, S

    def generate_octmnist_pv(self, data_path, n_sample, rand_seed=42,
                             generate_test=True, n_test_mc=20000):
        """Generate one proximal OCTMNIST dataset.

        Returns
        -------
        A : (n, 1) binary treatment
        Y : (n, 784) observed outcome image, flattened, UNCLIPPED
        Z : (n, d_z) treatment-side proxy
        W : (n, d_w) outcome-side proxy
        U : (n, d_u) latent confounder -- ORACLE ONLY, never given to an estimator
        do_A : (2, 1) = [[0], [1]]
        EY_do_A : (2, 784) = [mu_0, mu_1], the analytic counterfactual mean images
        oracle : dict with the potential outcomes, q(U), S, the templates and the Monte-Carlo means,
                 for validation and visualisation.
        """
        rng = np.random.default_rng(rand_seed)
        images, labels = self.load_octmnist(data_path)
        normal_images = images[labels == self.LABEL_NORMAL]
        dme_images = images[labels == self.LABEL_DME]
        if len(normal_images) == 0 or len(dme_images) == 0:
            raise ValueError("OCTMNIST normal or DME class is empty.")
        self.build_templates(normal_images, dme_images)

        # alpha (propensity) and beta (fluid probability) are positively aligned -> confounding.
        base = rng.standard_normal(self.d_u)
        base = base / np.linalg.norm(base)
        tilt = rng.standard_normal(self.d_u) * 0.35
        self._alpha = self.confounding_strength * base
        self._beta = 2.5 * (base + tilt) / np.linalg.norm(base + tilt)
        self._L_z = rng.standard_normal((self.d_u, self.d_z)) / np.sqrt(self.d_u)
        self._L_w = rng.standard_normal((self.d_u, self.d_w)) / np.sqrt(self.d_u)

        U = rng.standard_normal((n_sample, self.d_u))
        propensity = np.clip(1.0 / (1.0 + np.exp(-(U @ self._alpha))),
                             self.propensity_clip, 1.0 - self.propensity_clip)
        A = rng.binomial(1, propensity).astype(np.float64).reshape(-1, 1)
        Z = U @ self._L_z + rng.normal(0.0, self.sigma_z, size=(n_sample, self.d_z))
        W = U @ self._L_w + rng.normal(0.0, self.sigma_w, size=(n_sample, self.d_w))

        Y0, Y1, q, S = self._sample_arms(rng, U, normal_images)
        Y = np.where(A.reshape(-1, 1, 1) == 1, Y1, Y0)

        # --- analytic ground truth: mu_a = E[B] + theta E[q(U)] M_C + delta 1{a=1} M_T ---
        mean_baseline = normal_images.mean(axis=0)
        u_big = rng.standard_normal((200000, self.d_u))
        expected_q = float(self._q_of_u(u_big, self._beta).mean())
        mu0 = mean_baseline + expected_q * self.theta * self.M_C
        mu1 = mu0 + self.delta * self.M_T
        EY_do_A = np.stack([mu0.reshape(-1), mu1.reshape(-1)], axis=0)
        do_A = np.array([[0.0], [1.0]])

        oracle = {"Y0": Y0.reshape(n_sample, -1), "Y1": Y1.reshape(n_sample, -1),
                  "q": q, "S": S, "propensity": propensity,
                  "M_C": self.M_C, "M_T": self.M_T,
                  "template_coherence": self.template_coherence,
                  "expected_q": expected_q, "mean_baseline": mean_baseline,
                  "mu0_analytic": mu0, "mu1_analytic": mu1}

        if generate_test:
            # independent interventional Monte-Carlo check of the analytic means
            rng_mc = np.random.default_rng(rand_seed + 99991)
            U_mc = rng_mc.standard_normal((n_test_mc, self.d_u))
            Y0_mc, Y1_mc, _, _ = self._sample_arms(rng_mc, U_mc, normal_images)
            oracle["mu0_mc"] = Y0_mc.mean(axis=0)
            oracle["mu1_mc"] = Y1_mc.mean(axis=0)

        return A, Y.reshape(n_sample, -1), Z, W, U, do_A, EY_do_A, oracle

    # --------------------------------------------------------------- helpers
    def effect_projections(self, effect_image):
        """Project a (flattened or 2-D) mean-effect image onto the two templates.

        Returns (c_T, c_C) with c_T = <effect, M_T>/||M_T||^2 and c_C = <effect, M_C>/||M_C||^2.
        For the true causal effect these should be (delta, 0); for the naive observational contrast c_C
        is the spurious confounding component a proximal estimator has to remove.
        """
        effect = np.asarray(effect_image).reshape(self.image_shape)
        c_t = float(np.sum(effect * self.M_T) / (np.sum(self.M_T ** 2) + 1e-12))
        c_c = float(np.sum(effect * self.M_C) / (np.sum(self.M_C ** 2) + 1e-12))
        return c_t, c_c

    def to_display(self, flat_image):
        """Reshape a flattened outcome to 28x28 and clip to [0,1] FOR DISPLAY ONLY."""
        return np.clip(np.asarray(flat_image).reshape(self.image_shape), 0.0, 1.0)


#################################################################################################
############## ColorMNIST-PCL-CATE : structured (image-valued) heterogeneous response ###########
#################################################################################################
# Implements an internal design note
#
#     f_CATE(a, v) = E[Y^(a) | V = v],   Y an RGB image in [0,1]^{28x28x3}
#
# V is an OBSERVED subgroup label, U in {-1,+1} an UNOBSERVED binary colour confounder, A in {0,1} a
# binary treatment adding a subgroup-dependent GREEN component, Z and W noisy Gaussian proxies for U,
# and the grayscale MNIST digit G an exogenous baseline drawn from a FIXED per-class bank -- which is
# what makes the class-level expectations exact finite averages rather than Monte Carlo.
#
# Every population quantity is CLOSED FORM because the renderer is affine in the colour:
#     f_CATE(a,v) = Fbar_v (x) {c0 + a*delta(v)*cT} + Bbar_v (x) c_bg          (guide eq. 10)
#     tau(v)      = delta(v) * Fbar_v (x) cT                                    (eq. 11)
#     naive(v)    = tau(v) + eta*(m_1v - m_0v) * Fbar_v (x) cU                  (eq. 14)
# so the observed within-subgroup contrast is the true GREEN effect plus a spurious RED-BLUE
# confounding term -- a directly visual diagnostic of whether an estimator removed the confounding.
#
# VALIDATED against every oracle test in the guide's sec. 13 (counterfactual mean identity, naive
# contrast decomposition, both bridge identities, CATE self-normalisation, cell support); harness in
# a standalone validation script.
#
# NO PIXEL CLIPPING on the numerical outcomes (guide, "Important"): clipping is nonlinear and destroys
# the exact mean formulas above. The defaults keep unnoised colours inside [0,1]^3; clip for display
# only.

_SQRT_EPS = 1e-12


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class ColorMNIST_PCL_CATE_Dataset:
    """Guide Table 1 defaults. `digit_classes` indexes the subgroups V."""

    digit_classes: Sequence[int] = (0, 1, 3, 6, 8)
    group_scores: Sequence[float] = (-1.0, -0.5, 0.0, 0.5, 1.0)          # s_v
    delta: Sequence[float] = (0.05, 0.125, 0.20, 0.275, 0.35)            # delta(v), causal strength
    eps: float = 0.05                                                     # positivity floor
    gamma_u: float = 1.5
    gamma_v: float = 0.4
    gamma_uv: float = 0.5
    sigma_z: float = 0.5
    sigma_w: float = 0.5
    c0: Tuple[float, float, float] = (0.50, 0.25, 0.50)                   # baseline purple
    c_u: Tuple[float, float, float] = (1.0, 0.0, -1.0)                    # red-blue confounding dir
    c_t: Tuple[float, float, float] = (0.0, 1.0, 0.0)                     # green treatment dir
    c_bg: Tuple[float, float, float] = (0.05, 0.05, 0.05)
    eta: float = 0.25                                                     # confounding strength
    tau_m: float = 0.15                                                   # soft-mask threshold
    k_m: float = 20.0                                                      # soft-mask sharpness
    sigma_y: float = 0.01                                                  # pixel noise
    bank_size: int = 200                                                   # images per class bank
    data_path: str = "data"

    # filled by _build_banks
    banks: Dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    F_bar: Dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    B_bar: Dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ setup
    def _build_banks(self, rand_seed: int = 0) -> None:
        """Fixed per-class MNIST banks + the precomputed mean foreground/background (guide eq. 2).

        The bank is part of the POPULATION definition, so it is keyed on its own seed and must not be
        redrawn per sample size -- otherwise the oracle targets move.
        """
        if self.banks:
            return
        from torchvision import datasets          # lazy: only this DGP needs torchvision
        mnist = datasets.MNIST(root=self.data_path, train=True, download=True)
        imgs = mnist.data.numpy().astype(np.float64) / 255.0              # (N,28,28) in [0,1]
        labels = mnist.targets.numpy()
        rng = np.random.default_rng(rand_seed)
        for v in self.digit_classes:
            idx = np.flatnonzero(labels == v)
            pick = rng.choice(idx, size=min(self.bank_size, idx.size), replace=False)
            bank = imgs[pick]                                             # (N_v,28,28)
            self.banks[v] = bank
            M = _sigmoid(self.k_m * (bank - self.tau_m))                  # soft mask, guide sec 3.2
            F = M * bank                                                  # foreground intensity
            B = 1.0 - M                                                   # background mask
            self.F_bar[v] = F.mean(axis=0)                                # (28,28)
            self.B_bar[v] = B.mean(axis=0)

    def _mask(self, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        M = _sigmoid(self.k_m * (g - self.tau_m))
        return M * g, 1.0 - M

    def _render(self, F: np.ndarray, B: np.ndarray, colour: np.ndarray) -> np.ndarray:
        """R(G,c) = F (x) c + B (x) c_bg, broadcasting a 3-vector over the pixel grid (guide eq. 1)."""
        return F[..., None] * colour + B[..., None] * np.asarray(self.c_bg)

    # ------------------------------------------------------------------ population quantities
    def propensity(self, u: float, v: int) -> float:
        """e_v(u) = eps + (1-2eps) sigmoid(gamma_u u + gamma_v s_v + gamma_uv u s_v)  (guide eq. 3)."""
        s_v = self.group_scores[list(self.digit_classes).index(v)]
        return self.eps + (1.0 - 2.0 * self.eps) * _sigmoid(
            self.gamma_u * u + self.gamma_v * s_v + self.gamma_uv * u * s_v)

    def pi(self, a: int, u: float, v: int) -> float:
        e = self.propensity(u, v)
        return e if a == 1 else 1.0 - e

    def p_a_given_v(self, a: int, v: int) -> float:
        """p_a(v) = P(A=a|V=v) = 0.5{pi_a(+1,v) + pi_a(-1,v)}   (guide eq. 4)."""
        return 0.5 * (self.pi(a, +1.0, v) + self.pi(a, -1.0, v))

    def p_A(self, a: int) -> float:
        """p_A(a) = (1/K) sum_v p_a(v)   (guide eq. 5)."""
        return float(np.mean([self.p_a_given_v(a, v) for v in self.digit_classes]))

    def m_av(self, a: int, v: int) -> float:
        """m_{a,v} = E[U | A=a, V=v]  (guide eq. 12)."""
        p, m = self.pi(a, +1.0, v), self.pi(a, -1.0, v)
        return (p - m) / (p + m)

    def f_cate(self, a: int, v: int) -> np.ndarray:
        """EXACT f_CATE(a,v) = Fbar_v (x) {c0 + a delta(v) cT} + Bbar_v (x) c_bg  (guide eq. 10)."""
        self._build_banks()
        d = self.delta[list(self.digit_classes).index(v)]
        colour = np.asarray(self.c0) + a * d * np.asarray(self.c_t)
        return self._render(self.F_bar[v], self.B_bar[v], colour)

    def tau_true(self, v: int) -> np.ndarray:
        """EXACT causal contrast tau(v) = delta(v) Fbar_v (x) cT  (guide eq. 11)."""
        self._build_banks()
        d = self.delta[list(self.digit_classes).index(v)]
        return d * self.F_bar[v][..., None] * np.asarray(self.c_t)

    def naive_contrast_true(self, v: int) -> np.ndarray:
        """EXACT observed within-subgroup contrast (guide eq. 14) = causal + spurious red-blue."""
        self._build_banks()
        spurious = (self.eta * (self.m_av(1, v) - self.m_av(0, v))
                    * self.F_bar[v][..., None] * np.asarray(self.c_u))
        return self.tau_true(v) + spurious

    def h0(self, a: int, v: int, w: float) -> np.ndarray:
        """EXACT image-valued outcome bridge (guide eq. 15).

        h0(a,v,w) = Fbar_v (x) {c0 + eta w cU + a delta(v) cT} + Bbar_v (x) c_bg.
        """
        self._build_banks()
        d = self.delta[list(self.digit_classes).index(v)]
        colour = (np.asarray(self.c0) + self.eta * w * np.asarray(self.c_u)
                  + a * d * np.asarray(self.c_t))
        return self._render(self.F_bar[v], self.B_bar[v], colour)

    def phi0_ate(self, a: int, v: int, z: np.ndarray) -> np.ndarray:
        """EXACT ATE-scale treatment bridge phi_0^ATE(a,v,z) = alpha + beta z  (guide eqs. 16-18)."""
        s_plus = self.p_A(a) / self.pi(a, +1.0, v)
        s_minus = self.p_A(a) / self.pi(a, -1.0, v)
        alpha, beta = 0.5 * (s_plus + s_minus), 0.5 * (s_plus - s_minus)
        return alpha + beta * np.asarray(z)

    def D0(self, a: int, v: int) -> float:
        """Population normalizer D_0(a,v) = p_A(a)/p_a(v)  (guide eqs. 20-21)."""
        return self.p_A(a) / self.p_a_given_v(a, v)

    def phi0_cate(self, a: int, v: int, z: np.ndarray) -> np.ndarray:
        """EXACT CATE bridge = phi^ATE / D_0, so that E[phi^CATE|A=a,V=v] = 1  (guide eqs. 22-24)."""
        return self.phi0_ate(a, v, z) / self.D0(a, v)

    # ------------------------------------------------------------------ sampling
    def generate(self, n_sample: int, rand_seed: int = 0, bank_seed: int = 0,
                 common_noise: bool = True):
        """Draw the observational sample (guide sec. 12.1).

        Returns a dict with observed (Y, A, V, Z, W) plus oracle-only (U, Y0, Y1). `common_noise`
        reuses ONE pixel-noise tensor across both potential outcomes so oracle contrasts are
        variance-reduced; only Y^(A) is observed either way.
        """
        self._build_banks(bank_seed)
        rng = np.random.default_rng(rand_seed)
        K = len(self.digit_classes)
        classes = list(self.digit_classes)

        v_idx = rng.integers(K, size=n_sample)
        V = np.array([classes[i] for i in v_idx])
        U = rng.choice(np.array([-1.0, 1.0]), size=n_sample)

        e = np.array([self.propensity(U[i], V[i]) for i in range(n_sample)])
        A = rng.binomial(1, e).astype(np.int64)
        Z = U + self.sigma_z * rng.normal(size=n_sample)
        W = U + self.sigma_w * rng.normal(size=n_sample)

        Y0 = np.empty((n_sample, 28, 28, 3))
        Y1 = np.empty((n_sample, 28, 28, 3))
        for i in range(n_sample):
            bank = self.banks[V[i]]
            g = bank[rng.integers(len(bank))]
            F, B = self._mask(g)
            d = self.delta[classes.index(V[i])]
            base = np.asarray(self.c0) + self.eta * U[i] * np.asarray(self.c_u)
            noise = self.sigma_y * rng.normal(size=(28, 28, 3))
            Y0[i] = self._render(F, B, base) + noise
            Y1[i] = self._render(F, B, base + d * np.asarray(self.c_t)) + \
                (noise if common_noise else self.sigma_y * rng.normal(size=(28, 28, 3)))
        Y = np.where(A[:, None, None, None] == 1, Y1, Y0)
        return dict(Y=Y, A=A, V=V, Z=Z, W=W, U=U, Y0=Y0, Y1=Y1)

    # ------------------------------------------------------------------ evaluation (guide sec. 14)
    def mse_cate(self, f_hat: Dict[Tuple[int, int], np.ndarray]) -> float:
        """MSE_CATE over the 2K arm-subgroup cells, per pixel-channel (guide eq. 28)."""
        d_Y = 28 * 28 * 3
        tot = sum(float(np.sum((f_hat[(a, v)] - self.f_cate(a, v)) ** 2))
                  for a in (0, 1) for v in self.digit_classes)
        return tot / (2 * len(self.digit_classes) * d_Y)

    def mse_tau(self, f_hat: Dict[Tuple[int, int], np.ndarray]) -> float:
        """Effect-image error (guide eq. 29)."""
        d_Y = 28 * 28 * 3
        tot = sum(float(np.sum((f_hat[(1, v)] - f_hat[(0, v)] - self.tau_true(v)) ** 2))
                  for v in self.digit_classes)
        return tot / (len(self.digit_classes) * d_Y)

    def recovery_coefficients(self, f_hat: Dict[Tuple[int, int], np.ndarray]):
        """delta_hat(v) (should be ~ delta(v)) and c_U_hat(v) (should be ~ 0)  (guide eqs. 30-31)."""
        self._build_banks()
        out = {}
        for v in self.digit_classes:
            contrast = f_hat[(1, v)] - f_hat[(0, v)]
            tmpl_t = self.F_bar[v][..., None] * np.asarray(self.c_t)
            tmpl_u = self.F_bar[v][..., None] * np.asarray(self.c_u)
            out[v] = (float(np.sum(contrast * tmpl_t) / max(np.sum(tmpl_t ** 2), _SQRT_EPS)),
                      float(np.sum(contrast * tmpl_u) / max(np.sum(tmpl_u ** 2), _SQRT_EPS)))
        return out


# ============================================================================
# dSprite V3 -- the REPAIRED proximal benchmark (posX-channel latent confounder).
# This is the candidate design
# that SURVIVED the adversarial attack ladder; the two rejected candidates were broken by
# a column-sum shortcut. V2 (above) is structurally DEGENERATE for
# proximal claims, and V3 exists to replace it.
#
# OPEN ITEM, disclosed: the MIRROR-SYMMETRY HARDENING of the V3 design is NOT applied here. The reference attack ladder
# (naive Y~A 31.2 | adjust-Z 15.8 | adjust-(Z,W) 12.9 | oracle-U 4.95, Var(EY_do_A)=53.27)
# was measured on THIS exact parameterization, so the class is kept as measured rather
# than silently hardened. Apply the hardening and re-run the ladder before treating V3 as
# a settled benchmark.
# ============================================================================
class dSprite_ProxyVariable_DatasetV3():
    """
    REPLACEMENT dSprite proximal benchmark with a GENUINE two-proxy structure.

    The V2 benchmark (liyuan9988 "Update on dSprite experiment") is degenerate:
    Z is the exact latent code of A (so sigma(A,Z)=sigma(A)), and U=posY is a
    rendered latent of A (so p(U|A) is a point mass and W|A is degenerate).
    Both completeness conditions of proximal identification fail per treatment
    level, and a plain regression of Y on Z is consistent for the target.

    V3 fixes this with the textbook proximal DAG  Z <- U -> W,  U -> A,  U -> Y:

      U ~ Unif(0,1)                                   latent confounder, NEVER rendered
      scale ~ Unif{0..5}, orientation ~ Unif{0..39},
      posY_A ~ Unif{0..31}                            exogenous treatment latents
      posX_A = discretize_32( clip(U + 0.25 eps, 0, 1) )   U shifts A's horizontal
                                                            position only STOCHASTICALLY
      A = image(shape=heart, scale, orientation, posX_A, posY_A) + N(0, 0.1)   4096-D
      Z = ( U + 0.40 N(0,1),  U + Unif(-0.60, 0.60) )       2-D noisy view of U
      W = image(scale=3, orient=0, posX=16, posY_W) + N(0, 0.1),               4096-D
          posY_W = discretize_32( clip(U + 0.30 eps', 0, 1) )  LATENT-noisy view of U
      q(u) = exp(c (u - 1/2)) / M(c),  c = 3,  M(c) = (2/c) sinh(c/2)
             => E[q(U)] = 1 EXACTLY;  q ranges over [0.157, 3.157]
      Y = structural(A) * q(U) + N(0, 0.5)
      structural(a) = ((a . w)^2 - 3000)/500,  w_ij = |32-j|/32  (column weights,
          so structural depends on posX -> the confounding channel is real)

      EY_do_A(a) = structural(a) * E[q(U)] = structural(a)     (closed form)

    Why each defect is repaired:
      R1: p(U|A) has real spread (posX_A = U + noise, sd 0.25 before clipping);
          U is NOT recoverable from A (R^2 ~ 0.5), so W|A mixes over many posY_W
          tubes and the outcome-bridge equation constrains h off the diagonal.
      R2: Z is a noisy measurement OF U (not of A's latents), with its own
          independent noise -> Z is a genuine treatment-side proxy, and Z is
          not a function of A.
      R3: W = g(U, fresh noise)  =>  W _||_ (A,Z) | U.
          Y = f(A) q(U) + fresh noise  =>  Y _||_ Z | (A, U).
      R4: naive A / Z / (A,Z) / (A,Z,W) regressions are all biased (measured in
          the validation script); an oracle adjusting for U recovers the truth.
      R5: A stays a real 64x64 dSprite image; EY_do_A = structural(a) in closed
          form on the same 588-point grid as V2 (Var over the grid = 53.27).
    """

    # latent-noise scales (the identification-critical knobs; calibrated so that
    # population-level proxy adjustment fails materially while both proxies stay
    # informative -- see mine_calibrate2.py)
    SIGMA_POSX = 0.25   # sd of U -> posX_A channel noise (R1: keeps p(U|A) spread)
    SIGMA_Z = 0.40      # Gaussian noise of Z's first coordinate
    HALF_WIDTH_Z2 = 0.60  # half-width of Z's second (uniform-noise) coordinate
    SIGMA_W = 0.30      # sd of U -> posY_W channel noise (W is latent-noisy, not
                        # just pixel-noisy)
    C_CONF = 3.0        # confounder tilt strength in q(u)
    PIXEL_NOISE = 0.1
    OUTCOME_NOISE = 0.5

    def __init__(self):
        pass

    def cal_weight(self):
        weights = np.empty((64, 64))
        for i in range(64):
            for j in range(64):
                weights[i, j] = np.abs(32 - j)
        return weights.reshape(64 * 64, 1) / 32

    def image_id(self, latent_bases, posX_id_arr, posY_id_arr,
                 orientation_id_arr, scale_id_arr):
        data_size = posX_id_arr.shape[0]
        color_id_arr = np.array([0] * data_size, dtype=int)
        shape_id_arr = np.array([2] * data_size, dtype=int)
        idx = np.c_[color_id_arr, shape_id_arr, scale_id_arr,
                    orientation_id_arr, posX_id_arr, posY_id_arr]
        return idx.dot(latent_bases)

    def structural_func(self, image, weights):
        return (image.dot(weights)[:, 0] ** 2 - 3000) / 500

    @classmethod
    def confounder_factor(cls, u):
        """q(u) = exp(c(u-1/2)) / M(c), M(c) = (2/c) sinh(c/2).
        E[q(U)] = int_0^1 exp(c(u-1/2)) du / M(c) = 1 exactly for U ~ Unif(0,1)."""
        c = cls.C_CONF
        M = (2.0 / c) * np.sinh(c / 2.0)
        return np.exp(c * (u - 0.5)) / M

    @staticmethod
    def _discretize32(x):
        """Map [0,1] -> {0,...,31}."""
        return np.clip(np.round(x * 31), 0, 31).astype(int)

    def _load(self, data_path):
        dataset_zip = np.load(
            os.path.join(data_path, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
            allow_pickle=True, encoding="bytes")
        imgs = dataset_zip['imgs']
        metadata = dataset_zip['metadata'][()]
        latents_sizes = metadata[b'latents_sizes']
        latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                        np.array([1, ])))
        return imgs, latents_bases

    def generate_test_dsprite(self, data_path):
        """Same 588-point do(A) grid as V2; EY_do_A = structural(a) exactly."""
        imgs, latents_bases = self._load(data_path)
        weights = self.cal_weight()
        posX_id_arr = [0, 5, 10, 15, 20, 25, 30]
        posY_id_arr = [0, 5, 10, 15, 20, 25, 30]
        scale_id_arr = [0, 3, 5]
        orientation_arr = [0, 10, 20, 30]
        latent_idx_arr = []
        for posX, posY, scale, orientation in product(
                posX_id_arr, posY_id_arr, scale_id_arr, orientation_arr):
            latent_idx_arr.append([0, 2, scale, orientation, posX, posY])
        image_idx_arr = np.array(latent_idx_arr).dot(latents_bases)
        data_size = 7 * 7 * 3 * 4
        treatment = imgs[image_idx_arr].reshape((data_size, 64 * 64))
        structural = self.structural_func(treatment, weights)[:, np.newaxis]
        # grid latents (scale, orientation, posX, posY) for diagnostics/baselines
        grid_latents = np.array(latent_idx_arr)[:, 2:]
        return treatment, structural, grid_latents

    def generate_dsprite_pv(self, data_path, n_sample,
                            generate_test=False, rand_seed=42,
                            return_latents=False, **kwargs):
        """
        Returns
        -------
        treatment (n, 4096), outcome (n, 1), treatment_proxy Z (n, 2),
        outcome_proxy W (n, 4096), do_A, EY_do_A
        [ + dict of latents if return_latents ]
        """
        imgs, latents_bases = self._load(data_path)
        weights = self.cal_weight()
        rng = default_rng(seed=rand_seed)

        # --- latent confounder, never rendered into A -----------------------
        U = rng.uniform(0.0, 1.0, n_sample)

        # --- treatment image: U shifts posX only stochastically -------------
        posX_id_arr = self._discretize32(
            np.clip(U + self.SIGMA_POSX * rng.normal(size=n_sample), 0.0, 1.0))
        posY_id_arr = rng.integers(32, size=n_sample)
        scale_id_arr = rng.integers(6, size=n_sample)
        orientation_arr = rng.integers(40, size=n_sample)
        image_idx_arr = self.image_id(latents_bases, posX_id_arr, posY_id_arr,
                                      orientation_arr, scale_id_arr)
        treatment = imgs[image_idx_arr].reshape((n_sample, 64 * 64)).astype(np.float64)
        treatment += rng.normal(0.0, self.PIXEL_NOISE, treatment.shape)

        # --- treatment proxy Z: 2-D noisy view of U (independent noises) ----
        treatment_proxy = np.c_[
            U + self.SIGMA_Z * rng.normal(size=n_sample),
            U + rng.uniform(-self.HALF_WIDTH_Z2, self.HALF_WIDTH_Z2, n_sample)]

        # --- outcome proxy W: image whose posY is a NOISY function of U -----
        posY_W_id = self._discretize32(
            np.clip(U + self.SIGMA_W * rng.normal(size=n_sample), 0.0, 1.0))
        posX_id_proxy = np.array([16] * n_sample)
        scale_id_proxy = np.array([3] * n_sample)
        orientation_proxy = np.array([0] * n_sample)
        proxy_image_idx_arr = self.image_id(latents_bases, posX_id_proxy,
                                            posY_W_id, orientation_proxy,
                                            scale_id_proxy)
        outcome_proxy = imgs[proxy_image_idx_arr].reshape((n_sample, 64 * 64)).astype(np.float64)
        outcome_proxy += rng.normal(0.0, self.PIXEL_NOISE, outcome_proxy.shape)

        # --- outcome: multiplicative mean-one confounding --------------------
        structural = self.structural_func(treatment, weights)
        outcome = structural * self.confounder_factor(U) \
            + rng.normal(0.0, self.OUTCOME_NOISE, n_sample)
        outcome = outcome[:, np.newaxis]

        if generate_test:
            do_A, EY_do_A, _ = self.generate_test_dsprite(data_path)
        else:
            do_A, EY_do_A = None, None

        if return_latents:
            latents = dict(U=U, posX_id=posX_id_arr, posY_id=posY_id_arr,
                           scale_id=scale_id_arr, orientation_id=orientation_arr,
                           posY_W_id=posY_W_id, structural=structural)
            return treatment, outcome, treatment_proxy, outcome_proxy, do_A, EY_do_A, latents
        return treatment, outcome, treatment_proxy, outcome_proxy, do_A, EY_do_A


#################################################################################################
############## PROXY POLICY EVALUATION: BINARY LATENT-CONFOUNDING BENCHMARK #####################
#################################################################################################

@dataclass
class ProxyPolicyEvaluationBinaryBenchmark:
    """Closed-form binary benchmark for one-step proxy POLICY evaluation (Technical guide "One-Step Proxy Policy
    Evaluation", Sec. 8-9). All variables except Y take values in {-1,+1}:

        U ~ Rad(0),   W = U * xi_W  (E xi_W = lam_W),   Z = U * xi_Z  (E xi_Z = lam_Z),
        P(A = a | U = u) = (1 + a kappa u)/2,          Y(a) = mu + (beta + tau a) U + sigma_Y eps_a.

    Target policies depend on the pre-treatment OUTCOME proxy:  pi_gamma(a | w) = (1 + a (gamma0 + gamma1 w))/2.
    Everything the estimators target is closed form (guide eqs. 29-36):
        value                      V(pi_gamma)     = mu + tau gamma1 lam_W
        normalised treatment bridge phi0(a, z)      = (1 - a kappa z / lam_Z) / (1 - kappa^2)          E[phi0 | A=a, W] = p(a)/p(a|W)
        ordinary outcome bridge    h_ord(a, w)      = mu + (beta + tau a) w / lam_W                    (static interventions only)
        policy-specific bridge     h_pi(a, w)       = c0_a + c1_a w                                     (solves E[h | A=a, Z] = E[pi(a|W) Y | A=a, Z])
        invalid plug-in            E[sum_a pi(a|W) h_ord(a,W)] = mu + tau gamma1 / lam_W                (the negative control)
    The ordinary bridge multiplied by the policy is NOT a policy bridge: its bias tau gamma1 (1/lam_W - lam_W) vanishes only for a
    non-adaptive policy (gamma1 = 0), a homogeneous effect (tau = 0) or a perfect proxy (|lam_W| = 1).
    """
    mu: float = 1.0
    beta: float = 0.5
    tau: float = 1.0
    kappa: float = 0.4
    lam_W: float = 0.6
    lam_Z: float = 0.8
    sigma_Y: float = 0.5

    def __post_init__(self):
        assert 0 < abs(self.kappa) < 1 and 0 < abs(self.lam_W) < 1 and 0 < abs(self.lam_Z) < 1, "need 0<|kappa|,|lam_W|,|lam_Z|<1"

    @staticmethod
    def _rad(rng, n, mean=0.0):
        return np.where(rng.random(n) < (1.0 + mean) / 2.0, 1.0, -1.0)

    def generate(self, n_sample: int, seed: int, return_latents: bool = False):
        """Observational sample. Returns (A, Y, Z, W) as float arrays of shape (n, 1) [+ U when return_latents]."""
        rng = default_rng(seed)
        U = self._rad(rng, n_sample)
        W = U * self._rad(rng, n_sample, self.lam_W)
        Z = U * self._rad(rng, n_sample, self.lam_Z)
        A = np.where(rng.random(n_sample) < (1.0 + self.kappa * U) / 2.0, 1.0, -1.0)
        Y = self.mu + (self.beta + self.tau * A) * U + self.sigma_Y * rng.standard_normal(n_sample)
        out = (A[:, None], Y[:, None], Z[:, None], W[:, None])
        return out + (U[:, None],) if return_latents else out

    # ---- policy class and targets --------------------------------------------------------------------------------
    @staticmethod
    def policy_prob(a, w, gamma0: float, gamma1: float):
        """pi_gamma(a | w) for a, w in {-1,+1} (broadcasting)."""
        return (1.0 + a * (gamma0 + gamma1 * w)) / 2.0

    def true_value(self, gamma0: float, gamma1: float) -> float:
        return self.mu + self.tau * gamma1 * self.lam_W

    def ordinary_plugin_value(self, gamma0: float, gamma1: float) -> float:
        """Limit of the INVALID 'ordinary bridge x policy' plug-in (guide eq. 35)."""
        return self.mu + self.tau * gamma1 / self.lam_W

    # ---- closed-form bridges (RAW units) -------------------------------------------------------------------------
    def phi0(self, a, z):
        return (1.0 - a * self.kappa * z / self.lam_Z) / (1.0 - self.kappa ** 2)

    def h_ord(self, a, w):
        return self.mu + (self.beta + self.tau * a) * w / self.lam_W

    def h_pi_coefficients(self, a, gamma0: float, gamma1: float):
        d = self.beta + self.tau * a
        c0 = 0.5 * (self.mu * (1.0 + a * gamma0) + a * gamma1 * self.lam_W * d)
        c1 = (d * (1.0 + a * gamma0) + a * self.mu * gamma1 * self.lam_W) / (2.0 * self.lam_W)
        return c0, c1

    def h_pi(self, a, w, gamma0: float, gamma1: float):
        c0, c1 = self.h_pi_coefficients(a, gamma0, gamma1)
        return c0 + c1 * w


@dataclass
class ProxyPolicyEvaluationContinuousBenchmark:
    """Continuous-treatment proxy policy-evaluation benchmark with a CLOSED-FORM off-policy value (our construction):

        U ~ Rad(0);  W = U xi_W (E xi_W = lam_W);  Z = U xi_Z (E xi_Z = lam_Z);        A | U=u ~ N(kappa u, sigma_A^2);
        Y(a) = f(a) + (beta + tau a) U + sigma_Y eps,   f(a) = mu + a + c cos(omega a);
        target policies  pi_gamma(a | w) = N(a; gamma0 + gamma1 w, s^2)   (Gaussian density, W-dependent mean).

    Closed forms (Monte Carlo verified at N=1e6):
        V(pi_gamma)     = mu + gamma0 + c exp(-omega^2 s^2/2) cos(omega gamma0) cos(omega gamma1) + tau gamma1 lam_W
        p(a)/p(a|u)     = (1 + exp(-2 a kappa u/sigma_A^2))/2           (p(a) = 0.5 N(a;kappa,sigma_A^2) + 0.5 N(a;-kappa,sigma_A^2))
        phi0(a, z)      = 1/2 + cosh(2 a kappa/sigma_A^2)/2 - sinh(2 a kappa/sigma_A^2) z/(2 lam_Z)     (normalised, signed, heavy-tailed)
        h_ord(a, w)     = f(a) + (beta + tau a) w/lam_W                  (ordinary bridge: static interventions only)
        h_pi(a, w)      = c0(a) + c1(a) w                                (policy-specific bridge; see h_pi_coefficients)
        ordinary plug-in int E_W[pi(a|W) h_ord(a,W)] da = E f(A^pi) + tau gamma1/lam_W   (bias tau gamma1 (1/lam_W - lam_W))
    The value is estimated WITHOUT p(A) through the conditional-integration form V = int E[Y phi0(Z,a) pi(a|W) | A=a] da
    (a regression on A followed by quadrature); the single-expectation form E[Y phi0 pi(A|W)/p(A)] needs p(A) and is used only as
    an oracle check.
    """
    mu: float = 0.0
    beta: float = 0.5
    tau: float = 1.0
    kappa: float = 0.5
    sigma_A: float = 1.0
    lam_W: float = 0.6
    lam_Z: float = 0.8
    sigma_Y: float = 0.5
    c: float = 1.0
    omega: float = 1.5
    s: float = 0.5

    @staticmethod
    def _rad(rng, n, mean=0.0):
        return np.where(rng.random(n) < (1.0 + mean) / 2.0, 1.0, -1.0)

    def f(self, a):
        return self.mu + a + self.c * np.cos(self.omega * a)

    def generate(self, n_sample: int, seed: int, return_latents: bool = False):
        rng = default_rng(seed)
        U = self._rad(rng, n_sample); W = U * self._rad(rng, n_sample, self.lam_W); Z = U * self._rad(rng, n_sample, self.lam_Z)
        A = self.kappa * U + self.sigma_A * rng.standard_normal(n_sample)
        Y = self.f(A) + (self.beta + self.tau * A) * U + self.sigma_Y * rng.standard_normal(n_sample)
        out = (A[:, None], Y[:, None], Z[:, None], W[:, None])
        return out + (U[:, None],) if return_latents else out

    def policy_density(self, a, w, gamma0: float, gamma1: float):
        return norm.pdf(a, loc=gamma0 + gamma1 * w, scale=self.s)

    def sample_policy_action(self, rng, w, gamma0: float, gamma1: float):
        return gamma0 + gamma1 * w + self.s * rng.standard_normal(np.shape(w))

    def marginal_density(self, a):
        return 0.5 * norm.pdf(a, self.kappa, self.sigma_A) + 0.5 * norm.pdf(a, -self.kappa, self.sigma_A)

    def _E_f_policy(self, gamma0, gamma1):
        return self.mu + gamma0 + self.c * np.exp(-self.omega ** 2 * self.s ** 2 / 2) * np.cos(self.omega * gamma0) * np.cos(self.omega * gamma1)

    def true_value(self, gamma0: float, gamma1: float) -> float:
        return self._E_f_policy(gamma0, gamma1) + self.tau * gamma1 * self.lam_W

    def ordinary_plugin_value(self, gamma0: float, gamma1: float) -> float:
        return self._E_f_policy(gamma0, gamma1) + self.tau * gamma1 / self.lam_W

    def phi0(self, a, z):
        t = 2.0 * a * self.kappa / self.sigma_A ** 2
        return 0.5 + 0.5 * np.cosh(t) - np.sinh(t) / (2.0 * self.lam_Z) * z

    def h_ord(self, a, w):
        return self.f(a) + (self.beta + self.tau * a) * w / self.lam_W

    def h_pi_coefficients(self, a, gamma0: float, gamma1: float):
        Np, Nm = norm.pdf(a, gamma0 + gamma1, self.s), norm.pdf(a, gamma0 - gamma1, self.s); f = self.f(a); d = self.beta + self.tau * a
        c0 = 0.5 * (Np + Nm) * f + 0.5 * self.lam_W * (Np - Nm) * d
        c1 = (0.5 * (Np + Nm) * d + 0.5 * self.lam_W * (Np - Nm) * f) / self.lam_W
        return c0, c1

    def h_pi(self, a, w, gamma0: float, gamma1: float):
        c0, c1 = self.h_pi_coefficients(a, gamma0, gamma1); return c0 + c1 * w
