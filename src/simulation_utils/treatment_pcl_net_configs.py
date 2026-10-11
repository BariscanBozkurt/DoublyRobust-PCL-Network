import torch
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any


@dataclass
class TreatmentPCLConfig:
    """Master configuration class for Treatment Bridge PCL experiments."""

    # =========================================================================
    # 1. Experiment Settings
    # =========================================================================
    experiment_name: str = "treatment_bridge_default"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    disable_first_stage: bool = False

    # =========================================================================
    # 2. Optimization
    # =========================================================================
    lr_first_stage_base: float = 1e-3
    lr_second_stage_ax: float = 1e-3
    lr_second_stage_z: float = 1e-3
    second_stage_head_lr: float = 1e-2

    weight_decay: float = 1e-6
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
    reg_first_stage: Tuple[float, float] = (5e-3, 1e-1)
    reg_second_stage_first: Tuple[float, float] = (5e-5, 1e-3)
    reg_second_stage: Tuple[float, float] = (1e-3, 1e-3)

    reg_annealing_method: str = 'exponential'
    consider_prev_weight: bool = True

    # =========================================================================
    # 5. Loss Functions & Penalties
    # =========================================================================
    first_stage_loss_name: str = "mse"
    first_stage_loss_kwargs: Dict[str, Any] = field(default_factory=dict)

    second_stage_loss_name: str = "huber"
    second_stage_loss_kwargs: Dict[str, Any] = field(default_factory=lambda: {"delta": 1.0})

    # Specific to Treatment Bridge (Penalty for negative density ratio estimates)
    negative_penalty: float = 10.0

    # =========================================================================
    # 6. Dynamic Properties
    # =========================================================================
    @property
    def lr_first_stage(self) -> float:
        return 0.0 if self.disable_first_stage else self.lr_first_stage_base

    @property
    def regularizers(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
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
            experiment_name="treatment_bridge_low_dim_v1",
            lr_first_stage_base = 1e-4,
            lr_second_stage_ax = 1e-3,
            lr_second_stage_z = 1e-3,
            reg_first_stage = (5e-3, 1e-2),
            reg_second_stage = (1e-1, 5.),
            second_stage_loss_name = "huber",
            second_stage_loss_kwargs = {"delta": 0.5},
            )

    @classmethod
    def high_dim(cls):
        return cls(
            experiment_name="treatment_bridge_high_dim_v1",
            lr_first_stage_base=1e-4,
            lr_second_stage_ax=1e-4,
            lr_second_stage_z=1e-4,
            reg_first_stage = (5e-3, 1e-2),
            reg_second_stage = (1e-1, 5.),
            second_stage_loss_name = "huber",
            second_stage_loss_kwargs = {"delta": 0.5},
            scheduler_gamma = 0.95,
            reg_annealing_method = "linear",
            weight_decay = 1e-6
        )

    @classmethod
    def abortion_and_crime(cls):
        return cls(
            experiment_name="treatment_bridge_abortion_and_crime_v1",
            negative_penalty=0.0
        )

    @classmethod
    def deaner(cls):
        # Deaner grade-retention benchmark (W 3-D, Z 4-D, A/Y 1-D). Same treatment-bridge
        # config as abortion_and_crime; the notebook overrides lr/reg/epochs as needed.
        return cls(
            experiment_name="treatment_bridge_deaner_v1",
            negative_penalty=0.0
        )

    @classmethod
    def rhc(cls):
        # Right Heart Catheterization (BINARY A, W 2-D, Z 2-D, backdoor X 20-D). Same treatment-bridge
        # base config as abortion/Deaner; backdoor X is folded into the RHC net builder
        # (build_nets_for_treatment_pcl_net_rhc_experiment). The notebook overrides lr/reg/epochs.
        return cls(
            experiment_name="treatment_bridge_rhc_v1",
            negative_penalty=0.0
        )

    @classmethod
    def dsprite(cls):
        return cls(
            experiment_name="treatment_bridge_dSprite_v1",
            second_stage_loss_name = "log_cosh",
            second_stage_loss_kwargs = {},
            reg_second_stage=(1e-2, 1.0)
        )

    @classmethod
    def demand(cls):
        # Inherits standard defaults but keeps explicit naming
        return cls(experiment_name="treatment_bridge_demand_v1")

    @classmethod
    def synthetic_cate(cls):
        return cls(
            experiment_name="treatment_bridge_synthetic_cate_v1",
            lr_first_stage_base=1e-3,
            lr_second_stage_ax=1e-3,
            lr_second_stage_z=1e-3,
            # reg_first_stage = (5e-3, 1e-2),
            reg_second_stage = (1e-3, 1.),
            second_stage_loss_name = "log_cosh",
            second_stage_loss_kwargs = {},
            n_epochs = 100,
            # reg_annealing_method = "linear",
            # weight_decay = 1e-6

        )

@dataclass
class TreatmentPCLdSpriteConfig:

    # =========================================================================
    # 1. Experiment Settings
    # =========================================================================
    experiment_name: str = "treatment_bridge_dSprite_v1"
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    disable_first_stage: bool = False

    # =========================================================================
    # 3. Optimization
    # =========================================================================
    # Base Learning Rates
    lr_first_stage_base: float = 1e-3
    lr_second_stage_ax: float = 1e-3
    lr_second_stage_z: float = 1e-3
    second_stage_head_lr:float = 1e-2

    weight_decay: float = 1e-6
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
    reg_first_stage: Tuple[float, float] = (5e-3, 1e-1)

    # Stage 2 (Inner): Solving dens_ratio ~ gamma * (A, W, X)
    reg_second_stage_first: Tuple[float, float] = (5e-5, 1e-3)

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

    negative_penalty: float = 10.0

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
class TreatmentPCLDemandConfig:
    """
    Configuration for Treatment Bridge PCL (Synthetic High-Dim Experiment).
    Focus: Estimating the bridge function h(A, X) that approximates density ratios.
    """
    # =========================================================================
    # 1. Experiment Settings
    # =========================================================================
    experiment_name: str = "treatment_bridge_low_dim_v1"
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    disable_first_stage: bool = False

    # =========================================================================
    # 3. Optimization
    # =========================================================================
    # Base Learning Rates
    lr_first_stage_base: float = 1e-3
    lr_second_stage_ax: float = 1e-3
    lr_second_stage_z: float = 1e-3
    second_stage_head_lr:float = 1e-2

    weight_decay: float = 1e-6
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
    reg_first_stage: Tuple[float, float] = (5e-3, 1e-1)

    # Stage 2 (Inner): Solving dens_ratio ~ gamma * (A, W, X)
    reg_second_stage_first: Tuple[float, float] = (5e-5, 1e-3)

    # Stage 2 (Outer): Solving dens_ratio ~ psi * SecondStageFeatures
    reg_second_stage: Tuple[float, float] = (1e-3, 1e-3)

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

    negative_penalty: float = 10.0

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