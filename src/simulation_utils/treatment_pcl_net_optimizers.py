import torch
from torch.nn import Module
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ExponentialLR
from typing import Tuple, List

def build_sgd_optimizers_for_treatment_pcl_net(
    treatment_pcl_model: Module,
    lr_first_stage: float = 5e-4, # Default slower rate for stability
    lr_second_stage_ax: float = 5e-4, # Default slower rate for stability
    lr_second_stage_z: float = 5e-4, # Default slower rate for stability
    weight_decay: float = 1e-3,
    first_stage_momentum: float = 0.0,
    second_stage_momentum: float = 0.0,
    gamma: float = 0.999 # Decay factor (e.g., 0.999 reduces LR by 0.1% each epoch)
) -> Tuple[List[Adam], List[Adam]]:
    """
    Build Adam optimizers with two distinct learning rates:
    1. A slower rate for the featurizer (deep theta networks).
    2. A faster rate for the final linear layers (regression weights).
    """

    # --- Stage 1 Optimizers ---

    # 1. Featurizer (Slow Rate)
    optimizer_first_stage = SGD(
        treatment_pcl_model.first_stage_featurizer.parameters(),
        lr=lr_first_stage,
        weight_decay=weight_decay,
        nesterov = (first_stage_momentum > 0.0),
        momentum = first_stage_momentum,
    )

    optimizers_stage1 = [
        optimizer_first_stage
    ]

    # --- Stage 2 Optimizers ---

    # 1. Featurizer (Slow Rate)
    optimizer_second_stage_ax = SGD(
        treatment_pcl_model.treatment_backdoor_featurizer.parameters(),
        lr=lr_second_stage_ax,
        weight_decay=weight_decay,
        nesterov = (second_stage_momentum > 0.0),
        momentum = second_stage_momentum,
    )

    optimizer_second_stage_z = SGD(
        treatment_pcl_model.treatment_proxy_featurizer.parameters(),
        lr=lr_second_stage_z,
        weight_decay=weight_decay,
        nesterov = (second_stage_momentum > 0.0),
        momentum = second_stage_momentum,
    )


    optimizers_stage2 = [
        optimizer_second_stage_ax,
        optimizer_second_stage_z,
    ]

    # Schedulers for Stage 1 (applied to ALL stage 1 optimizers)
    schedulers_stage1 = [
        ExponentialLR(opt, gamma=gamma)
        for opt in optimizers_stage1
    ]

    # Schedulers for Stage 2 (applied to ALL stage 2 optimizers)
    schedulers_stage2 = [
        ExponentialLR(opt, gamma=gamma)
        for opt in optimizers_stage2
    ]

    return optimizers_stage1, optimizers_stage2, schedulers_stage1, schedulers_stage2

def build_adam_optimizers_for_treatment_pcl_net(
    treatment_pcl_model: Module,
    lr_first_stage: float = 5e-4, # Default slower rate for stability
    lr_second_stage_ax: float = 5e-4, # Default slower rate for stability
    lr_second_stage_z: float = 5e-4, # Default slower rate for stability
    weight_decay: float = 1e-3,
    gamma: float = 0.999 # Decay factor (e.g., 0.999 reduces LR by 0.1% each epoch)
) -> Tuple[List[Adam], List[Adam]]:
    """
    Build Adam optimizers with two distinct learning rates:
    1. A slower rate for the featurizer (deep theta networks).
    2. A faster rate for the final linear layers (regression weights).
    """

    # --- Stage 1 Optimizers ---

    # 1. Featurizer (Slow Rate)
    optimizer_first_stage = AdamW(
        treatment_pcl_model.first_stage_featurizer.parameters(),
        lr=lr_first_stage,
        weight_decay=weight_decay,
    )

    optimizers_stage1 = [
        optimizer_first_stage
    ]

    # --- Stage 2 Optimizers ---

    # 1. Featurizer (Slow Rate)
    optimizer_second_stage_ax = AdamW(
        treatment_pcl_model.treatment_backdoor_featurizer.parameters(),
        lr=lr_second_stage_ax,
        weight_decay=weight_decay,
    )

    optimizer_second_stage_z = AdamW(
        treatment_pcl_model.treatment_proxy_featurizer.parameters(),
        lr=lr_second_stage_z,
        weight_decay=weight_decay,
    )


    optimizers_stage2 = [
        optimizer_second_stage_ax,
        optimizer_second_stage_z,
    ]

    # Schedulers for Stage 1 (applied to ALL stage 1 optimizers)
    schedulers_stage1 = [
        ExponentialLR(opt, gamma=gamma)
        for opt in optimizers_stage1
    ]

    # Schedulers for Stage 2 (applied to ALL stage 2 optimizers)
    schedulers_stage2 = [
        ExponentialLR(opt, gamma=gamma)
        for opt in optimizers_stage2
    ]

    return optimizers_stage1, optimizers_stage2, schedulers_stage1, schedulers_stage2

