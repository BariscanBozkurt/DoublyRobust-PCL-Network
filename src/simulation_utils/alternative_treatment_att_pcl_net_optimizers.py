import torch
from torch.nn import Module
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ExponentialLR
from typing import Tuple, List

# Optimizer builders for the DENSITY-RATIO-FREE JOINT-ANCHOR ATT bridge
# (AlternativeTreatmentBridgePCLNETATT). The builders in alternative_treatment_pcl_net_optimizers.py
# cannot be reused: that bridge's second stage carries a treatment-only featurizer
# (`treatment_featurizer_second_stage`) plus a separate second-stage backdoor featurizer, whereas the
# ATT bridge has ONE joint featurizer `anchor_treatment_featurizer` = psi_{AA'X}(A, A', X), which
# absorbs the backdoor on the second stage. The first stage is the same in both.

def build_sgd_optimizers_for_alternative_treatment_att_pcl_net(
    treatment_pcl_model: Module,
    lr_first_stage: float = 5e-4, # Default slower rate for stability
    lr_second_stage_aa: float = 5e-4, # Default slower rate for stability
    lr_second_stage_z: float = 5e-4, # Default slower rate for stability
    weight_decay: float = 1e-3,
    first_stage_momentum: float = 0.0,
    second_stage_momentum: float = 0.0,
    gamma: float = 0.999 # Decay factor (e.g., 0.999 reduces LR by 0.1% each epoch)
) -> Tuple[List[SGD], List[SGD], List[ExponentialLR], List[ExponentialLR]]:
    """
    Build SGD optimizers with two distinct learning rates:
    1. Stage 1: the conditional mean embedding featurizers (A and W).
    2. Stage 2: the bridge featurizers (the joint (A, A', X) featurizer and the treatment proxy Z).
    """

    # --- Stage 1 Optimizers ---

    # 1. Featurizer (Slow Rate)
    optimizer_first_stage_a = SGD(
        treatment_pcl_model.treatment_featurizer_first_stage.parameters(),
        lr=lr_first_stage,
        weight_decay=weight_decay,
        nesterov = (first_stage_momentum > 0.0),
        momentum = first_stage_momentum,
    )

    optimizer_first_stage_w = SGD(
        treatment_pcl_model.outcome_proxy_featurizer.parameters(),
        lr=lr_first_stage,
        weight_decay=weight_decay,
        nesterov = (first_stage_momentum > 0.0),
        momentum = first_stage_momentum,
    )

    optimizers_stage1 = [
        optimizer_first_stage_a,
        optimizer_first_stage_w,
    ]

    if treatment_pcl_model.backdoor_featurizer_first_stage is not None:
        optimizer_first_stage_x = SGD(
            treatment_pcl_model.backdoor_featurizer_first_stage.parameters(),
            lr=lr_first_stage,
            weight_decay=weight_decay,
            nesterov = (first_stage_momentum > 0.0),
            momentum = first_stage_momentum,
        )
        optimizers_stage1.append(optimizer_first_stage_x)

    # --- Stage 2 Optimizers ---

    # 1. Featurizer (Slow Rate)
    optimizer_second_stage_aa = SGD(
        treatment_pcl_model.anchor_treatment_featurizer.parameters(),
        lr=lr_second_stage_aa,
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
        optimizer_second_stage_aa,
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


def build_adam_optimizers_for_alternative_treatment_att_pcl_net(
    treatment_pcl_model: Module,
    lr_first_stage: float = 5e-4, # Default slower rate for stability
    lr_second_stage_aa: float = 5e-4, # Default slower rate for stability
    lr_second_stage_z: float = 5e-4, # Default slower rate for stability
    weight_decay: float = 1e-3,
    gamma: float = 0.999 # Decay factor (e.g., 0.999 reduces LR by 0.1% each epoch)
) -> Tuple[List[AdamW], List[AdamW], List[ExponentialLR], List[ExponentialLR]]:
    """
    Build Adam optimizers with two distinct learning rates:
    1. Stage 1: the conditional mean embedding featurizers (A and W).
    2. Stage 2: the bridge featurizers (the joint (A, A', X) featurizer and the treatment proxy Z).
    """

    # --- Stage 1 Optimizers ---

    # 1. Featurizer (Slow Rate)
    optimizer_first_stage_a = AdamW(
        treatment_pcl_model.treatment_featurizer_first_stage.parameters(),
        lr=lr_first_stage,
        weight_decay=weight_decay,
    )

    optimizer_first_stage_w = AdamW(
        treatment_pcl_model.outcome_proxy_featurizer.parameters(),
        lr=lr_first_stage,
        weight_decay=weight_decay,
    )

    optimizers_stage1 = [
        optimizer_first_stage_a,
        optimizer_first_stage_w,
    ]

    if treatment_pcl_model.backdoor_featurizer_first_stage is not None:
        optimizer_first_stage_x = AdamW(
            treatment_pcl_model.backdoor_featurizer_first_stage.parameters(),
            lr=lr_first_stage,
            weight_decay=weight_decay,
        )
        optimizers_stage1.append(optimizer_first_stage_x)

    # --- Stage 2 Optimizers ---

    # 1. Featurizer (Slow Rate)
    optimizer_second_stage_aa = AdamW(
        treatment_pcl_model.anchor_treatment_featurizer.parameters(),
        lr=lr_second_stage_aa,
        weight_decay=weight_decay,
    )

    optimizer_second_stage_z = AdamW(
        treatment_pcl_model.treatment_proxy_featurizer.parameters(),
        lr=lr_second_stage_z,
        weight_decay=weight_decay,
    )

    optimizers_stage2 = [
        optimizer_second_stage_aa,
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
