import torch
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any


@dataclass
class AlternativeTreatmentPCLConfig:
    """Master configuration class for the DENSITY-RATIO-FREE Treatment Bridge PCL experiments."""

    # Counterpart of TreatmentPCLConfig for AlternativeTreatmentBridgePCLNET (the neural mean-embedding
    # analogue of the kernel KAP method). Four fields of TreatmentPCLConfig are deliberately absent:
    #   - second_stage_loss_name / second_stage_loss_kwargs: the density-ratio bridge REGRESSES onto an
    #     externally estimated ratio, so its second-stage loss is a free choice. The KAP second stage is
    #     the FIXED quadratic E[(E[varphi|W,A])^2] - 2 E_W E_A[E[varphi|W,A]], in which the ratio cancels
    #     analytically -- there is no pointwise loss to swap. Only the first stage takes a loss function.
    #   - negative_penalty: there is no density-ratio target to keep positive. (The model does expose an
    #     optional nonnegative_final_layer clip on Theta, but it is off by default and unused here.)
    #   - reg_annealing_method: the regularizers below are PLAIN FLOATS, not (start, end) tuples --
    #     train_alternative_treatment_pcl_net_ate_model has no annealing schedule, and a tuple would
    #     reach `nData * weight_regularizer * torch.eye(...)` and raise.
    #   - disable_first_stage: the KAP cross-term reduction is built ON the first-stage CME weight
    #     (E_W[mu] = FirstStageMap(outer_prod(phi_A1, mean_W phi_W))), so switching the first stage off
    #     would zero mu and make varphi degenerate -- not a meaningful mode for this bridge.

    # =========================================================================
    # 1. Experiment Settings
    # =========================================================================
    experiment_name: str = "alternative_treatment_bridge_default"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # 2. Optimization
    # =========================================================================
    # Stage 1 trains the CME featurizers {A (first stage), W}; stage 2 trains the bridge featurizers
    # {A (second stage), Z}. Backdoor X featurizers (high-dim experiment) ride along on
    # lr_first_stage / lr_second_stage_ax respectively.
    lr_first_stage: float = 1e-3
    lr_second_stage_ax: float = 1e-4
    lr_second_stage_z: float = 1e-4
    second_stage_head_lr: float = 1.0

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
    second_stage_head_steps: int = 10
    log_per_epoch: int = 10 ** 9
    plot_loss: bool = False

    # =========================================================================
    # 4. Inner-Loop Ridge Regularization
    # =========================================================================
    # Scalars, NOT (start, end) tuples -- the density-ratio-free trainer does not anneal (see docstring).
    reg_first_stage: float = 1e-2
    reg_second_stage_first: float = 1e-2
    reg_second_stage: float = 10.0

    consider_prev_weight: bool = True

    # =========================================================================
    # 5. Loss Function & Second-Stage Head
    # =========================================================================
    # Only the FIRST stage has a swappable loss; the KAP second stage is a fixed quadratic.
    first_stage_loss_name: str = "mse"
    first_stage_loss_kwargs: Dict[str, Any] = field(default_factory=dict)

    # "closed_form" (ridge solve) or "sgd" (L-BFGS steps on the head). This flag is what separates the
    # *_w_ClosedForm notebooks from the *_w_SGD ones. The KAP head objective is quadratic in Theta, so
    # L-BFGS reaches the ridge solution of the SAME objective -- but note the two are only exactly equal
    # when consider_prev_weight is False: the closed-form solve adds a proximal term in the previous
    # weight (reg * nData * Theta_prev) that the L-BFGS closure does not.
    head_mode: str = "closed_form"

    # =========================================================================
    # 6. Dynamic Properties
    # =========================================================================
    @property
    def regularizers(self) -> Tuple[float, float, float]:
        return (self.reg_first_stage, self.reg_second_stage_first, self.reg_second_stage)

    # =========================================================================
    # 7. Experimental Presets (Factory Methods)
    # =========================================================================
    @classmethod
    def demand(cls):
        # Demand benchmark (Z 2-D, W 1-D, A 1-D). scheduler_gamma = 1.0 (no LR decay) reproduces the
        # hand-rolled optimizers this preset replaces; reg_second_stage is the KAP head ridge for the
        # ill-posed bridge (fixed hyperparameter, NOT tuned to the ground truth).
        return cls(
            experiment_name="alternative_treatment_bridge_demand_v1",
            lr_first_stage = 1e-4,
            lr_second_stage_ax = 1e-4,
            lr_second_stage_z = 1e-4,
            weight_decay = 1e-4,
            n_epochs = 150,
            reg_second_stage = 100.0,
            log_per_epoch = 10,
            plot_loss = True,
        )

    @classmethod
    def low_dim(cls):
        return cls(
            experiment_name="alternative_treatment_bridge_low_dim_v1",
            reg_second_stage = 10.0,
        )

    @classmethod
    def high_dim(cls):
        # Synthetic high-dim (Z 10-D, W 10-D, A 1-D, backdoor X 100-D). The backdoor featurizers come
        # from the net builder (6-tuple); the tensor-product bridge keeps the KAP cross term O(N).
        return cls(
            experiment_name="alternative_treatment_bridge_high_dim_v1",
            reg_second_stage = 100.0,
        )

    @classmethod
    def abortion_and_crime(cls):
        # Legalized Abortion & Crime benchmark (Z 1-D, W 3-D, A 1-D).
        return cls(
            experiment_name="alternative_treatment_bridge_abortion_and_crime_v1",
            reg_second_stage = 10.0,
        )

    @classmethod
    def deaner(cls):
        # Deaner grade-retention benchmark (Z 4-D, W 3-D, A/Y 1-D, N ~ 2983). Same modest head ridge as
        # abortion: both are small real datasets where the ill-posed bridge needs damping but not much.
        return cls(
            experiment_name="alternative_treatment_bridge_deaner_v1",
            reg_second_stage = 10.0,
        )

    @classmethod
    def octmnist(cls):
        # Proximal OCTMNIST (BINARY A, 6-D proxies Z and W, 28x28 image outcome). The latent confounder
        # is 6-D and the proxies are noisy (R^2(U|Z) ~ 0.57), so the bridge is a harder inverse problem
        # than ColorMNIST's scalar-proxy version -- hence the larger second-stage ridge.
        return cls(
            experiment_name="alternative_treatment_bridge_octmnist_v1",
            reg_second_stage = 30.0,
        )

    @classmethod
    def colormnist(cls):
        # Proximal ColorMNIST (BINARY A = digit class, scalar proxies Z and W, image outcome Y).
        # The bridge sees only 1-D inputs, so this is a small, well-conditioned problem; the structured
        # outcome is handled entirely by the third stage (output_dim = 3*28*28).
        return cls(
            experiment_name="alternative_treatment_bridge_colormnist_v1",
            reg_second_stage = 10.0,
        )

    @classmethod
    def synthetic_cate(cls):
        # Synthetic CATE benchmark (Z 3-D, W 3-D, A 1-D BINARY, effect modifier V 1-D as the backdoor).
        # The bridge itself is the ordinary density-ratio-free ATE bridge -- chapter 21 recovers the
        # heterogeneous response by self-normalizing it in the third stage, so nothing here is
        # CATE-specific except the dimensions it is pointed at.
        return cls(
            experiment_name="alternative_treatment_bridge_synthetic_cate_v1",
            reg_second_stage = 10.0,
        )

    @classmethod
    def rhc(cls):
        # Right Heart Catheterization (BINARY A = swang1, Z = (pafi1, paco21), W = (ph1, hema1),
        # backdoor X = the 20 significant covariates of Zhang et al. 2025, Appendix F). N = 4014, a
        # small real clinical dataset, so the nets are modest and the head ridge follows the abortion
        # preset rather than the high-dim one.
        #
        # NOTE on reg_second_stage: despite the name it is NOT a ridge -- update_final_layer_with_batch_
        # regression solves A = U^T U + N*lambda*I with RHS = N*b + N*lambda*Theta_prev, whose fixed
        # point is the UNREGULARISED normal equations. lambda is an inverse step size on a proximal
        # iteration, so what matters is lambda / T with T = epochs * batches-per-epoch. Scale it with the
        # update budget if you change n_epochs or the batch size.
        return cls(
            experiment_name="alternative_treatment_bridge_rhc_v1",
            n_epochs = 100,
            reg_second_stage = 10.0,
        )

    @classmethod
    def dsprite(cls):
        # dSprite (A and W 4096-D, Z 3-D). The ill-posed bridge blows up at the usual reg; the fix here
        # is REGULARIZATION, not normalization (reg_second_stage 1000.0 tames it: TM 950 -> 35).
        return cls(
            experiment_name="alternative_treatment_bridge_dSprite_v1",
            n_epochs = 50,
            reg_second_stage = 1000.0,
        )
