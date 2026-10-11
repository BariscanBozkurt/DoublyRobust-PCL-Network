import torch
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any


@dataclass
class AlternativeTreatmentATTPCLConfig:
    """Master configuration class for the DENSITY-RATIO-FREE JOINT-ANCHOR ATT Bridge PCL experiments."""

    # Counterpart of AlternativeTreatmentPCLConfig for AlternativeTreatmentBridgePCLNETATT. It keeps that
    # class's deliberate omissions (no second_stage_loss_*, no negative_penalty, no reg_annealing_method,
    # no disable_first_stage -- see the AlternativeTreatmentPCLConfig docstring, the same reasons apply)
    # and adds the three knobs the ordered-pair ATT objective needs:
    #   - pair_mode / n_derangements : which pairing estimates the objective (derivation sec 6 vs sec 7);
    #   - pair_chunk_size            : how many ordered pairs are held in memory at once.
    # The second-stage learning rate is named lr_second_stage_aa here rather than lr_second_stage_ax,
    # because the second-stage featurizer is psi_{AA'X}(a, a', x) -- it takes the ANCHOR treatment too.

    # =========================================================================
    # 1. Experiment Settings
    # =========================================================================
    experiment_name: str = "alternative_treatment_att_bridge_default"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # 2. Optimization
    # =========================================================================
    # Stage 1 trains the CME featurizers {A (first stage), W}; stage 2 trains the bridge featurizers
    # {psi_{AA'X}, Z}. A first-stage backdoor X featurizer rides along on lr_first_stage.
    lr_first_stage: float = 1e-3
    lr_second_stage_aa: float = 1e-4
    lr_second_stage_z: float = 1e-4

    weight_decay: float = 1e-6
    first_stage_momentum: float = 0.9
    second_stage_momentum: float = 0.9
    scheduler_gamma: float = 1.0

    # =========================================================================
    # 3. Training Loop Settings
    # =========================================================================
    n_epochs: int = 100
    stage1_iter: int = 1
    stage2_iter: int = 1
    log_per_epoch: int = 10 ** 9
    plot_loss: bool = False

    # =========================================================================
    # 4. Ordered-Pair Objective (the ATT-specific part)
    # =========================================================================
    # "full"        -> the complete ordered-pair U-statistic over all M(M-1) pairs (derivation eq. 24).
    #                  Exact empirical objective, no pair-subsampling noise, O(M^2) pair evaluations.
    # "derangement" -> n_derangements random nonzero cyclic shifts (derivation eq. 29, 32), O(M) per shift
    #                  and conditionally unbiased for the complete objective (derivation Prop. 7.1).
    # The stage-2 BATCH SIZE is what sets M, so it -- not the dataset size -- controls the cost of "full".
    pair_mode: str = "full"
    n_derangements: int = 1
    # Ordered pairs are consumed in chunks of this size when accumulating the closed-form normal
    # equations and when building the third-stage datasets, so the M(M-1) x d pair-feature matrix is
    # never materialized (derivation sec 6.1, "Computational profile").
    pair_chunk_size: int = 200000

    # The third stage regresses the pseudo-outcome on (A_i, A_j). It usually wants FEWER pairs than the
    # bridge: n_derangements_third_stage * M randomly paired rows already sample (a, a') from P_A (x) P_A,
    # whereas the complete pairing would hand a smooth two-input surface M(M-1) rows.
    third_stage_pair_mode: str = "derangement"
    n_derangements_third_stage: int = 10

    # Calibrate the fitted bridge to its own normalizer, varphi -> varphi / D_hat(a, a'), where
    # D_hat estimates E[varphi(a,a',X,Z) | A = a] -- which a valid normalized bridge makes identically 1
    # (derivation eq. 41, sec 10). Applied by WRAPPING predict_treatment_bridge_function, so both the
    # treatment-only and the doubly robust third stages see the calibrated bridge. `denominator_floor`
    # clips D_hat from below; the clipped fraction is the overlap diagnostic and should be reported.
    postnorm_bridge: bool = False
    denominator_floor: float = 0.1

    # =========================================================================
    # 5. Inner-Loop Ridge Regularization
    # =========================================================================
    # Scalars, NOT (start, end) tuples -- this trainer does not anneal.
    reg_first_stage: float = 1e-2
    reg_second_stage_first: float = 1e-2
    reg_second_stage: float = 10.0

    consider_prev_weight: bool = True

    # =========================================================================
    # 6. Loss Function & Second-Stage Head
    # =========================================================================
    # Only the FIRST stage has a swappable loss; the ordered-pair ATT objective is a fixed quadratic.
    first_stage_loss_name: str = "mse"
    first_stage_loss_kwargs: Dict[str, Any] = field(default_factory=dict)

    # The second-stage head is always the closed-form solve of derivation eq. 28 (or eq. 34 for the
    # derangement pairing); there is no SGD variant.
    head_mode: str = "closed_form"

    # =========================================================================
    # 7. Dynamic Properties
    # =========================================================================
    @property
    def regularizers(self) -> Tuple[float, float, float]:
        return (self.reg_first_stage, self.reg_second_stage_first, self.reg_second_stage)

    # =========================================================================
    # 8. Experimental Presets (Factory Methods)
    # =========================================================================
    @classmethod
    def synthetic_low_dim(cls):
        # Synthetic low-dim ATT benchmark (Z 2-D, W 2-D, A 1-D, no backdoor X). n = 2000, so the stage-2
        # split is ~1350 rows; a stage-2 batch of 512 gives ~261k ordered pairs per step, which the
        # complete U-statistic handles comfortably.
        return cls(
            experiment_name="alternative_treatment_att_bridge_synthetic_low_dim_v1",
            lr_first_stage = 1e-3,
            lr_second_stage_aa = 1e-4,
            lr_second_stage_z = 1e-4,
            n_epochs = 100,
            reg_second_stage = 10.0,
            log_per_epoch = 10,
            plot_loss = True,
        )

    @classmethod
    def dsprite(cls):
        # dSprite ATT (A and W 4096-D, Z 3-D). As in the ATE case the bridge is badly ill-posed at the
        # usual ridge, so the second-stage regularizer is much larger; the pair objective also has to run
        # on smaller stage-2 batches because psi_{AA'X} is evaluated on M(M-1) rows.
        return cls(
            experiment_name="alternative_treatment_att_bridge_dSprite_v1",
            n_epochs = 50,
            reg_second_stage = 1000.0,
            pair_mode = "derangement",
            n_derangements = 5,
        )
