import torch
from dataclasses import dataclass, field, replace
from typing import Tuple, Dict, Any

@dataclass
class OutcomePCLConfig:
    """Master configuration class for Outcome Bridge experiments."""

    # =========================================================================
    # 1. Experiment Settings
    # =========================================================================
    experiment_name: str = "outcome_bridge_default"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    disable_first_stage: bool = False

    # =========================================================================
    # 2. Optimization (Base Defaults)
    # =========================================================================
    lr_first_stage_base: float = 1e-4
    lr_second_stage_ax: float = 1e-4
    lr_second_stage_w: float = 1e-4
    second_stage_head_lr: float = 1e-2

    weight_decay: float = 1e-5
    first_stage_momentum: float = 0.9
    second_stage_momentum: float = 0.9
    scheduler_gamma: float = 0.99

    # =========================================================================
    # 3. Training Loop Settings
    # =========================================================================
    n_epochs: int = 50
    stage1_iter: int = 10
    stage2_iter: int = 1
    second_stage_head_steps: int = 5
    log_per_epoch: int = 10
    plot_loss: bool = True

    # =========================================================================
    # 4. Inner-Loop Ridge Regularization Schedules
    # =========================================================================
    reg_first_stage: Tuple[float, float] = (5e-3, 1e-3)
    reg_second_stage_first: Tuple[float, float] = (1e-3, 5e-3)
    reg_second_stage: Tuple[float, float] = (1e-2, 1.0)

    reg_annealing_method: str = 'exponential'
    consider_prev_weight: bool = True

    # =========================================================================
    # 5. Loss Functions
    # =========================================================================
    first_stage_loss_name: str = "mse"
    first_stage_loss_kwargs: Dict[str, Any] = field(default_factory=dict)

    second_stage_loss_name: str = "huber"
    second_stage_loss_kwargs: Dict[str, Any] = field(default_factory=lambda: {"delta": 1.0})

    # =========================================================================
    # 6. Dynamic Properties
    # =========================================================================
    @property
    def lr_first_stage(self) -> float:
        """Returns 0.0 if first stage is disabled, else base LR."""
        return 0.0 if self.disable_first_stage else self.lr_first_stage_base

    @property
    def regularizers(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Returns (reg_first, reg_second_inner, reg_second_outer)."""
        if self.disable_first_stage:
            infinite_reg = (1e8, 1e8)
            return (infinite_reg, infinite_reg, self.reg_second_stage)
        return (self.reg_first_stage, self.reg_second_stage_first, self.reg_second_stage)

    # =========================================================================
    # 7. Experimental Presets (Factory Methods)
    # =========================================================================
    @classmethod
    def low_dim(cls):
        return cls(
            experiment_name="outcome_bridge_low_dim_v1",
            lr_first_stage_base = 1e-4,
            lr_second_stage_ax = 1e-3,
            lr_second_stage_w = 1e-3,
            reg_first_stage = (1e-4, 1e-2),
            reg_second_stage_first = (1e-5, 1e-3),
            reg_second_stage=(1., 10.0),
            second_stage_loss_name = "huber",
            second_stage_loss_kwargs = {"delta": 0.5},
        )

    @classmethod
    def high_dim(cls):
        return cls(
            experiment_name="outcome_bridge_high_dim_v1",
            lr_first_stage_base = 1e-3,
            lr_second_stage_ax = 1e-3,
            lr_second_stage_w = 1e-3,
            reg_first_stage=(1e-4, 1e-3),
            reg_second_stage_first=(5e-5, 1e-2),
            reg_second_stage=(1., 100.0),
            scheduler_gamma = 0.99,
            reg_annealing_method = "linear",
            weight_decay = 1e-7,
            n_epochs = 50,
        )

    @classmethod
    def octmnist(cls):
        # Proximal OCTMNIST (BINARY A, 6-D proxies, 28x28 image outcome kept on its natural [0,1] scale
        # via an identity outcome transformer, so the second-stage ridge is set on that scale).
        return cls(
            experiment_name="outcome_bridge_octmnist_v1",
            reg_first_stage = (1e-3, 1e-2),
            reg_second_stage_first = (1e-3, 1e-2),
            reg_second_stage = (1e-4, 1e-2),
            second_stage_loss_name = "mse",
            second_stage_loss_kwargs = {},
            n_epochs = 50,
        )

    @classmethod
    def colormnist(cls):
        # Proximal ColorMNIST (BINARY A = digit class, scalar proxies Z and W, STRUCTURED image outcome).
        # The second-stage target is a 2352-dim image kept on its natural [0, 1] scale (identity outcome
        # transformer), so the second-stage ridge is set on that scale rather than a standardised one.
        return cls(
            experiment_name="outcome_bridge_colormnist_v1",
            reg_first_stage = (1e-3, 1e-2),
            reg_second_stage_first = (1e-3, 1e-2),
            reg_second_stage = (1e-4, 1e-2),
            second_stage_loss_name = "mse",
            second_stage_loss_kwargs = {},
            n_epochs = 50,
        )

    @classmethod
    def abortion_and_crime(cls):
        return cls(
            experiment_name="outcome_bridge_abortion_and_crime_v1",
            weight_decay=1e-7,
            reg_second_stage=(5e-3, 5e-3)
        )

    @classmethod
    def deaner(cls):
        # Deaner grade-retention benchmark (W 3-D, Z 4-D, A/Y 1-D). Same outcome-bridge
        # config as abortion_and_crime; the notebook overrides lr/reg/epochs as needed.
        return cls(
            experiment_name="outcome_bridge_deaner_v1",
            weight_decay=1e-7,
            reg_second_stage=(5e-3, 5e-3)
        )

    @classmethod
    def rhc(cls):
        # Right Heart Catheterization (BINARY A, W 2-D, Z 2-D, backdoor X 20-D, survival-time Y).
        # Same outcome-bridge base config as abortion/Deaner; the backdoor X is handled by the RHC
        # net builder (build_nets_for_outcome_pcl_net_rhc_experiment), not the config. The notebook
        # overrides lr/reg/epochs as needed.
        return cls(
            experiment_name="outcome_bridge_rhc_v1",
            weight_decay=1e-7,
            reg_second_stage=(5e-3, 5e-3)
        )

    @classmethod
    def dsprite(cls):
        return cls(
            experiment_name="outcome_bridge_dSprite_v1",
            lr_first_stage_base = 1e-4,
            lr_second_stage_ax= 1e-4,
            lr_second_stage_w= 1e-4,
            reg_first_stage = (10., 100.),
            second_stage_loss_name = "log_cosh",
            second_stage_loss_kwargs = {},
            weight_decay=1e-6
        )

    @classmethod
    def demand(cls):
        return cls(
            experiment_name="outcome_bridge_demand_v1",
            lr_first_stage_base=1e-3,
            lr_second_stage_ax=1e-3,
            lr_second_stage_w=1e-3,
            reg_second_stage=(1.0, 1.0)
        )

    @classmethod
    def synthetic_cate(cls):
        return cls(
            experiment_name="outcome_bridge_synthetic_cate_v1",
            second_stage_loss_name = "log_cosh",
            second_stage_loss_kwargs = {},
            n_epochs = 100,
        )

@dataclass
class OutcomePCLdSpriteConfig:
    # =========================================================================
    # 1. Experiment Settings
    # =========================================================================
    experiment_name: str = "outcome_bridge_dSprite_v1"
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    disable_first_stage: bool = False

    # =========================================================================
    # 3. Optimization
    # =========================================================================
    # Base Learning Rates
    lr_first_stage_base: float = 1e-4
    lr_second_stage_ax: float = 1e-4
    lr_second_stage_w: float = 1e-4
    second_stage_head_lr:float = 1e-2

    weight_decay: float = 1e-6  # Stronger decay to prevent norm explosion
    first_stage_momentum: float = 0.9
    second_stage_momentum: float = 0.9
    scheduler_gamma: float = 0.99

    # =========================================================================
    # 4. Training Loop Settings
    # =========================================================================
    n_epochs: int = 50
    stage1_iter: int = 10
    stage2_iter: int = 1
    second_stage_head_steps: int = 5
    log_per_epoch: int = 10
    plot_loss: bool = True

    # =========================================================================
    # 5. Inner-Loop Ridge Regularization Schedules
    # =========================================================================
    # Stage 1: Solving Z ~ beta * (A, W, X)
    reg_first_stage: Tuple[float, float] = (5e-3, 1e-3)

    # Stage 2 (Inner): Solving dens_ratio ~ gamma * (A, W, X)
    reg_second_stage_first: Tuple[float, float] = (5e-3, 1e-3)

    # Stage 2 (Outer): Solving dens_ratio ~ psi * SecondStageFeatures
    reg_second_stage: Tuple[float, float] = (1e-2, 1.)

    reg_annealing_method: str = 'exponential'
    consider_prev_weight: bool = True

    # =========================================================================
    # 6. Loss Functions
    # =========================================================================
    first_stage_loss_name: str = "mse"
    first_stage_loss_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Treatment Bridge predicts density ratios (can be large), so Huber is safer than MSE
    second_stage_loss_name: str = "huber"
    second_stage_loss_kwargs: Dict[str, Any] = field(default_factory=lambda: {"delta": 1.0})

    # =========================================================================
    # 7. Dynamic Properties
    # =========================================================================

    @property
    def lr_first_stage(self) -> float:
        """Returns 0.0 if first stage is disabled, else base LR."""
        return 0.0 if self.disable_first_stage else self.lr_first_stage_base

    @property
    def regularizers(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """
        Returns (reg_first, reg_second_inner, reg_second_outer).
        If disabled, returns effective infinity (1e8) for Stage 1.
        """
        if self.disable_first_stage:
            infinite_reg = (1e8, 1e8)
            return (infinite_reg, infinite_reg, self.reg_second_stage)

        return (self.reg_first_stage, self.reg_second_stage_first, self.reg_second_stage)


@dataclass
class OutcomePCLDemandConfig:
    # =========================================================================
    # 1. Experiment Settings
    # =========================================================================
    experiment_name: str = "outcome_bridge_demand_v1"
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    disable_first_stage: bool = False

    # =========================================================================
    # 3. Optimization
    # =========================================================================
    # Base Learning Rates
    lr_first_stage_base: float = 1e-3
    lr_second_stage_ax: float = 1e-3
    lr_second_stage_w: float = 1e-3
    second_stage_head_lr:float = 1e-2

    weight_decay: float = 1e-5  # Stronger decay to prevent norm explosion
    first_stage_momentum: float = 0.9
    second_stage_momentum: float = 0.9
    scheduler_gamma: float = 0.99

    # =========================================================================
    # 4. Training Loop Settings
    # =========================================================================
    n_epochs: int = 50
    stage1_iter: int = 10
    stage2_iter: int = 1
    second_stage_head_steps: int = 5
    log_per_epoch: int = 10
    plot_loss: bool = True

    # =========================================================================
    # 5. Inner-Loop Ridge Regularization Schedules
    # =========================================================================
    # Stage 1: Solving Z ~ beta * (A, W, X)
    reg_first_stage: Tuple[float, float] = (5e-3, 1e-3)

    # Stage 2 (Inner): Solving dens_ratio ~ gamma * (A, W, X)
    reg_second_stage_first: Tuple[float, float] = (5e-3, 1e-3)

    # Stage 2 (Outer): Solving dens_ratio ~ psi * SecondStageFeatures
    reg_second_stage: Tuple[float, float] = (1., 1.)

    reg_annealing_method: str = 'exponential'
    consider_prev_weight: bool = True

    # =========================================================================
    # 6. Loss Functions
    # =========================================================================
    first_stage_loss_name: str = "mse"
    first_stage_loss_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Treatment Bridge predicts density ratios (can be large), so Huber is safer than MSE
    second_stage_loss_name: str = "huber"
    second_stage_loss_kwargs: Dict[str, Any] = field(default_factory=lambda: {"delta": 1.0})

    # =========================================================================
    # 7. Dynamic Properties
    # =========================================================================

    @property
    def lr_first_stage(self) -> float:
        """Returns 0.0 if first stage is disabled, else base LR."""
        return 0.0 if self.disable_first_stage else self.lr_first_stage_base

    @property
    def regularizers(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """
        Returns (reg_first, reg_second_inner, reg_second_outer).
        If disabled, returns effective infinity (1e8) for Stage 1.
        """
        if self.disable_first_stage:
            infinite_reg = (1e8, 1e8)
            return (infinite_reg, infinite_reg, self.reg_second_stage)

        return (self.reg_first_stage, self.reg_second_stage_first, self.reg_second_stage)

