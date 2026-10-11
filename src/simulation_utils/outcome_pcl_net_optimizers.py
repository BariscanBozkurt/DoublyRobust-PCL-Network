import torch
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import Module
from typing import List, Tuple


def build_sgd_optimizers_for_outcome_pcl_net(
    outcome_pcl_model: Module,
    lr_first_stage: float = 5e-4, # Default slower rate for stability
    lr_second_stage_ax: float = 5e-4, # Default slower rate for stability
    lr_second_stage_w: float = 5e-4, # Default slower rate for stability
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
        outcome_pcl_model.first_stage_featurizer.parameters(),
        lr=lr_first_stage,
        weight_decay=weight_decay,
        nesterov = (first_stage_momentum > 0),
        momentum = first_stage_momentum,
    )

    optimizers_stage1 = [
        optimizer_first_stage
    ]

    # --- Stage 2 Optimizers ---

    # 1. Featurizer (Slow Rate)
    optimizer_treatment = SGD(
        outcome_pcl_model.treatment_featurizer.parameters(),
        lr=lr_second_stage_ax,
        weight_decay=weight_decay,
        nesterov = (second_stage_momentum > 0),
        momentum = second_stage_momentum,
    )

    optimizer_outcome_proxy = SGD(
        outcome_pcl_model.outcome_proxy_featurizer.parameters(),
        lr=lr_second_stage_w,
        weight_decay=weight_decay,
        nesterov = (second_stage_momentum > 0),
        momentum = second_stage_momentum,
    )

    optimizers_stage2 = [
        optimizer_treatment,
        optimizer_outcome_proxy,
    ]

    if hasattr(outcome_pcl_model, 'covariate_featurizer'):
        if outcome_pcl_model.covariate_featurizer is not None:
            optimizer_backdoor = AdamW(
                outcome_pcl_model.covariate_featurizer.parameters(),
                lr=lr_second_stage_ax,
                weight_decay=weight_decay,
            )
            optimizers_stage2.append(optimizer_backdoor)

    if outcome_pcl_model.backdoor_featurizer is not None:
        optimizer_backdoor = SGD(
            outcome_pcl_model.backdoor_featurizer.parameters(),
            lr=lr_second_stage_ax,
            weight_decay=weight_decay,
            nesterov = (second_stage_momentum > 0.0),
            momentum = second_stage_momentum,
        )
        optimizers_stage2.append(optimizer_backdoor)
    # --- Create Schedulers ---

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


def build_adam_optimizers_for_outcome_pcl_net(
    outcome_pcl_model: Module,
    lr_first_stage: float = 5e-4, # Default slower rate for stability
    lr_second_stage_ax: float = 5e-4, # Default slower rate for stability
    lr_second_stage_w: float = 5e-4, # Default slower rate for stability
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
        outcome_pcl_model.first_stage_featurizer.parameters(),
        lr=lr_first_stage,
        weight_decay=weight_decay,
    )

    optimizers_stage1 = [
        optimizer_first_stage
    ]

    # --- Stage 2 Optimizers ---

    # 1. Featurizer (Slow Rate)
    optimizer_treatment = AdamW(
        outcome_pcl_model.treatment_featurizer.parameters(),
        lr=lr_second_stage_ax,
        weight_decay=weight_decay,
    )

    optimizer_outcome_proxy = AdamW(
        outcome_pcl_model.outcome_proxy_featurizer.parameters(),
        lr=lr_second_stage_w,
        weight_decay=weight_decay,
    )

    optimizers_stage2 = [
        optimizer_treatment,
        optimizer_outcome_proxy,
    ]

    if hasattr(outcome_pcl_model, 'covariate_featurizer'):
        if outcome_pcl_model.covariate_featurizer is not None:
            optimizer_backdoor = AdamW(
                outcome_pcl_model.covariate_featurizer.parameters(),
                lr=lr_second_stage_ax,
                weight_decay=weight_decay,
            )
            optimizers_stage2.append(optimizer_backdoor)

    if outcome_pcl_model.backdoor_featurizer is not None:
        optimizer_backdoor = AdamW(
            outcome_pcl_model.backdoor_featurizer.parameters(),
            lr=lr_second_stage_ax,
            weight_decay=weight_decay,
        )
        optimizers_stage2.append(optimizer_backdoor)
    # --- Create Schedulers ---

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
