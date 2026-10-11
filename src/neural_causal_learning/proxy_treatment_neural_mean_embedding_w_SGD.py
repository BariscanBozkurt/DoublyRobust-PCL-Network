import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Optional, Tuple, Callable, Union
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler # Use abstract base class for type hint
from torch.optim.lr_scheduler import ExponentialLR

import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

import sys
sys.path.append("..")
from torch_utils.scalers import TorchIdentityTransformer
from torch_utils.linear_algebra import add_const_col, fit_linear_proximal, outer_prod, outer_prod_batch
from torch_utils.model_helpers import get_last_linear_out_features
from torch_utils.torch_eval import evaluate_loss_on_dataloader
from neural_causal_learning.proxy_treatment_neural_mean_embedding import TreatmentBridgePCLNET, HeterogeneousTreatmentBridgePCLNET, get_annealed_value_for_reg


class TreatmentBridgePCLNET(TreatmentBridgePCLNET):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict_dens_ratio(self, treatment, outcome_proxy, backdoor=None):
        """
        Predicts r(A, X, W) (density ratios) using Stage 2 weights.
        """
        # We use the 'inside_second_stage' weights logic usually, or the main weights?
        # Typically for inference we use the standard flow.
        feat_stage2 = self._compute_second_stage_features(treatment, outcome_proxy, backdoor, self.final_layer_first_stage_weight.T)
        pred_raw = torch.matmul(feat_stage2, self.final_layer_second_stage_weight.T)
        dens_ratio_pred = self.dens_ratio_transformer.inverse_transform(pred_raw)
        return nn.ReLU()(dens_ratio_pred)

    def second_stage_loss(self, first_stage_tuple, second_stage_tuple,
                          loss_fn = None,
                          second_stage_first_final_layer_regularizer = 1e-2,
                          negative_penalty: float = 0.0):
        if loss_fn is None: loss_fn = nn.MSELoss()
        ##############################################################################
        # Extract first stage data from the input tuple.
        ##############################################################################
        treatment, treatment_proxy, outcome_proxy, backdoor, _ = self._unpack_batch(first_stage_tuple)

        ######################################################################################################
        # Construct First Stage Feature and first stage final layer weight for inside second stage.
        ######################################################################################################
        with torch.no_grad():
            first_stage_feature = self._compute_first_stage_features(treatment, outcome_proxy, backdoor)
            weight_previous = self.final_layer_first_stage_inside_second_stage_weight.detach()

        treatment_proxy_feature = self.treatment_proxy_featurizer(treatment_proxy)
        new_weight = fit_linear_proximal(treatment_proxy_feature, first_stage_feature, weight_previous.T,
                                         second_stage_first_final_layer_regularizer)

        ##############################################################################
        # Extract second stage data from the input tuple.
        ##############################################################################
        treatment, treatment_proxy, outcome_proxy, backdoor, dens_ratios = self._unpack_batch(second_stage_tuple)

        # ##############################################################################
        # # Compute SECOND stage features
        # ##############################################################################
        second_stage_feature = self._compute_second_stage_features(treatment, outcome_proxy, backdoor, new_weight)

        dens_ratio_preds = (torch.matmul(second_stage_feature, self.final_layer_second_stage_weight.T))
        loss = loss_fn(dens_ratio_preds, dens_ratios)
        loss += (nn.ReLU()(-self.dens_ratio_transformer.inverse_transform(dens_ratio_preds)) ** 2).mean() * negative_penalty
        with torch.no_grad():
            self.final_layer_first_stage_inside_second_stage_weight = new_weight.T
        return loss

    def update_second_stage_final_layer_with_sgd(
            self,
            data_tuple: Tuple[torch.Tensor, ...],
            loss_fn: Callable,
            lr: float = 1.0,
            n_steps: int = 10,
            weight_decay: float = 0.0,
            negative_penalty: float = 0.0,
        ):
        """
        Updates the final layer using L-BFGS on the CPU for maximum stability.
        """
        # 1. Unpack Data
        treatment, _, outcome_proxy, backdoor, dens_ratio = self._unpack_batch(data_tuple)

        # 2. Compute Features on GPU (Fast)
        # We keep the heavy lifting (feature extraction) on the GPU
        with torch.no_grad():
            w1_inner = self.final_layer_first_stage_inside_second_stage_weight.T

            # Get features on GPU
            features_gpu = self._compute_second_stage_features(
                treatment, outcome_proxy, backdoor, w1_inner
            )

            # 3. Move to CPU for Optimization (Stable)
            # Casting to float ensures standard precision for LBFGS
            X_cpu = features_gpu.detach().cpu().float()
            Y_cpu = dens_ratio.detach().cpu().float()

            # Safety: Check for NaNs before they crash the CPU optimizer
            if torch.isnan(X_cpu).any() or torch.isinf(X_cpu).any():
                return

        # 4. Snapshot Weight to CPU
        # We work on a detached CPU copy of the weights.
        # This prevents disrupting the main model's computation graph.
        w_cpu = self.final_layer_second_stage_weight.detach().cpu().float().clone()
        w_cpu.requires_grad_(True)
        self.dens_ratio_transformer.to("cpu") # Ensure transformer is on CPU for penalty computation in closure
        # 5. L-BFGS Optimizer (CPU)
        # Strong Wolfe is safe and effective on CPU
        optimizer = torch.optim.LBFGS(
            [w_cpu],
            lr=lr,
            max_iter=20,
            history_size=10,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
            tolerance_change=1e-9
        )

        # 6. Optimization Closure
        def closure():
            optimizer.zero_grad()

            # Forward pass on CPU
            pred = (torch.matmul(X_cpu, w_cpu.T))

            loss = loss_fn(pred, Y_cpu)

            # Regularization
            if weight_decay > 0:
                # We use the initial snapshot as the anchor (w_prev)
                # w_cpu starts as w_prev, but updates in place, so we need a frozen copy
                # if we wanted proximal. For simple L2 decay, we decay towards 0.
                # Assuming you want Proximal (stay close to previous GPU weight):
                w_prev_cpu = self.final_layer_second_stage_weight.detach().cpu().float()
                diff = w_cpu - w_prev_cpu
                loss = loss + 0.5 * weight_decay * torch.sum(diff ** 2)

            # Negative Penalty
            if negative_penalty > 0:
                loss += (nn.ReLU()(-self.dens_ratio_transformer.inverse_transform(pred)) ** 2).mean() * negative_penalty
            loss.backward()
            return loss

        # 7. Run Optimization
        try:
            for _ in range(n_steps):
                optimizer.step(closure)
        except RuntimeError as e:
            # L-BFGS on CPU is rare to crash, but good to handle
            print(f"L-BFGS CPU Warning: {e}")

        # 8. Copy Updated Weights Back to Device
        # This updates the main model's weights with the optimized result
        self.final_layer_second_stage_weight.data.copy_(w_cpu.data.to(self.device))
        self.dens_ratio_transformer.to(self.device)


    # # ## The following LFGS-based update in GPU led to the following error: CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling `cublasSdot(handle, n, x, incx, y, incy, result)`
    # # ## This optimisation step is placed in CPU with L-BFGS, which is more stable for small linear layers (see update_second_stage_final_layer_with_sgd below).
    # # ## The code below is the original GPU-based SGD update for the second stage final layer, which is more efficient per step but less stable.
    # # ## You can experiment with it if you want to see if it works in your environment, but the CPU L-BFGS version is recommended for maximum stability.
    # def update_second_stage_final_layer_with_sgd(
    #         self,
    #         data_tuple: Tuple[torch.Tensor,...],
    #         loss_fn: Callable,
    #         lr: float = 1.0,
    #         n_steps: int = 10,
    #         weight_decay: float = 0.0,
    #         negative_penalty: float = 0.0,
    #     ):
    #     """
    #     Updates the final layer using L-BFGS directly on the GPU for maximum speed.
    #     """
    #     # 1. Unpack Data (Already on self.device via _unpack_batch)
    #     treatment, _, outcome_proxy, backdoor, dens_ratio = self._unpack_batch(data_tuple)

    #     # 2. Compute Features on GPU
    #     with torch.no_grad:
    #         w1_inner = self.final_layer_first_stage_inside_second_stage_weight.T

    #         # Get features and detach to form the constants for our local optimization
    #         X_gpu = self._compute_second_stage_features(
    #             treatment, outcome_proxy, backdoor, w1_inner
    #         ).detach
    #         Y_gpu = dens_ratio.detach

    #         # Safety: Check for NaNs before they crash the optimizer
    #         if torch.isnan(X_gpu).any or torch.isinf(X_gpu).any:
    #             return

    #     # 3. Snapshot Weight on GPU
    #     # Clone creates a new leaf tensor on the same device, then we track gradients
    #     w_gpu = self.final_layer_second_stage_weight.detach.clone
    #     w_gpu.requires_grad_(True)

    #     # 4. L-BFGS Optimizer (GPU)
    #     optimizer = torch.optim.LBFGS(
    #         [w_gpu],
    #         lr=lr,
    #         max_iter=20,
    #         history_size=10,
    #         line_search_fn="strong_wolfe",
    #         tolerance_grad=1e-7,
    #         tolerance_change=1e-9
    #     )

    #     # Pre-anchor the previous weights for proximal regularization
    #     w_prev_gpu = self.final_layer_second_stage_weight.detach

    #     # 5. Optimization Closure
    #     def closure:
    #         optimizer.zero_grad

    #         # Forward pass entirely on GPU
    #         pred = torch.matmul(X_gpu, w_gpu.T)

    #         loss = loss_fn(pred, Y_gpu)

    #         # Regularization (Proximal penalty towards previous weights)
    #         if weight_decay > 0:
    #             diff = w_gpu - w_prev_gpu
    #             loss = loss + 0.5 * weight_decay * torch.sum(diff ** 2)

    #         # Negative Penalty
    #         if negative_penalty > 0:
    #             # dens_ratio_transformer is already on self.device, no need to move it
    #             unscaled_pred = self.dens_ratio_transformer.inverse_transform(pred)
    #             loss = loss + (nn.ReLU(-unscaled_pred) ** 2).mean * negative_penalty

    #         loss.backward
    #         return loss

    #     # 6. Run Optimization
    #     try:
    #         for _ in range(n_steps):
    #             optimizer.step(closure)
    #     except RuntimeError as e:
    #         print(f"L-BFGS GPU Warning: {e}")

    #     # 7. Copy Updated Weights Back to the Model
    #     # Since both are on the same device, this is a highly optimized memory copy
    #     self.final_layer_second_stage_weight.data.copy_(w_gpu.data)


class HeterogeneousTreatmentBridgePCLNET(HeterogeneousTreatmentBridgePCLNET):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict_dens_ratio(self, treatment, covariate, outcome_proxy, backdoor=None):
        """
        Predicts r(A, X, W) (density ratios) using Stage 2 weights.
        """
        # We use the 'inside_second_stage' weights logic usually, or the main weights?
        # Typically for inference we use the standard flow.
        feat_stage2 = self._compute_second_stage_features(treatment, covariate, outcome_proxy, backdoor, self.final_layer_first_stage_weight.T)
        pred_raw = torch.matmul(feat_stage2, self.final_layer_second_stage_weight.T)
        dens_ratio_pred = self.dens_ratio_transformer.inverse_transform(pred_raw)
        return nn.ReLU()(dens_ratio_pred)

    def second_stage_loss(self, first_stage_tuple, second_stage_tuple,
                          loss_fn = None,
                          second_stage_first_final_layer_regularizer = 1e-2,
                          negative_penalty: float = 0.0):
        if loss_fn is None: loss_fn = nn.MSELoss()
        ##############################################################################
        # Extract first stage data from the input tuple.
        ##############################################################################
        treatment, treatment_proxy, outcome_proxy, covariate, backdoor, _ = self._unpack_batch(first_stage_tuple)

        ######################################################################################################
        # Construct First Stage Feature and first stage final layer weight for inside second stage.
        ######################################################################################################
        with torch.no_grad():
            first_stage_feature = self._compute_first_stage_features(treatment, covariate, outcome_proxy, backdoor)
            weight_previous = self.final_layer_first_stage_inside_second_stage_weight.detach()

        treatment_proxy_feature = self.treatment_proxy_featurizer(treatment_proxy)
        new_weight = fit_linear_proximal(treatment_proxy_feature, first_stage_feature, weight_previous.T,
                                         second_stage_first_final_layer_regularizer)

        ##############################################################################
        # Extract second stage data from the input tuple.
        ##############################################################################
        treatment, treatment_proxy, outcome_proxy, covariate, backdoor, dens_ratios = self._unpack_batch(second_stage_tuple)

        # ##############################################################################
        # # Compute SECOND stage features
        # ##############################################################################
        second_stage_feature = self._compute_second_stage_features(treatment, covariate, outcome_proxy, backdoor, new_weight)

        dens_ratio_preds = (torch.matmul(second_stage_feature, self.final_layer_second_stage_weight.T))
        loss = loss_fn(dens_ratio_preds, dens_ratios)
        loss += (nn.ReLU()(-self.dens_ratio_transformer.inverse_transform(dens_ratio_preds)) ** 2).mean() * negative_penalty
        with torch.no_grad():
            self.final_layer_first_stage_inside_second_stage_weight = new_weight.T
        return loss

    def update_second_stage_final_layer_with_sgd(
            self,
            data_tuple: Tuple[torch.Tensor, ...],
            loss_fn: Callable,
            lr: float = 1.0,
            n_steps: int = 10,
            weight_decay: float = 0.0,
            negative_penalty: float = 0.0,
        ):
        """
        Updates the final layer using L-BFGS on the CPU for maximum stability.
        """
        # 1. Unpack Data
        treatment, _, outcome_proxy, covariate, backdoor, dens_ratio = self._unpack_batch(data_tuple)

        # 2. Compute Features on GPU (Fast)
        # We keep the heavy lifting (feature extraction) on the GPU
        with torch.no_grad():
            w1_inner = self.final_layer_first_stage_inside_second_stage_weight.T

            # Get features on GPU
            features_gpu = self._compute_second_stage_features(
                treatment, covariate, outcome_proxy, backdoor, w1_inner
            )

            # 3. Move to CPU for Optimization (Stable)
            # Casting to float ensures standard precision for LBFGS
            X_cpu = features_gpu.detach().cpu().float()
            Y_cpu = dens_ratio.detach().cpu().float()

            # Safety: Check for NaNs before they crash the CPU optimizer
            if torch.isnan(X_cpu).any() or torch.isinf(X_cpu).any():
                return

        # 4. Snapshot Weight to CPU
        # We work on a detached CPU copy of the weights.
        # This prevents disrupting the main model's computation graph.
        w_cpu = self.final_layer_second_stage_weight.detach().cpu().float().clone()
        w_cpu.requires_grad_(True)
        self.dens_ratio_transformer.to("cpu") # Ensure transformer is on CPU for penalty computation in closure
        # 5. L-BFGS Optimizer (CPU)
        # Strong Wolfe is safe and effective on CPU
        optimizer = torch.optim.LBFGS(
            [w_cpu],
            lr=lr,
            max_iter=20,
            history_size=10,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
            tolerance_change=1e-9
        )

        # 6. Optimization Closure
        def closure():
            optimizer.zero_grad()

            # Forward pass on CPU
            pred = (torch.matmul(X_cpu, w_cpu.T))

            loss = loss_fn(pred, Y_cpu)

            # Regularization
            if weight_decay > 0:
                # We use the initial snapshot as the anchor (w_prev)
                # w_cpu starts as w_prev, but updates in place, so we need a frozen copy
                # if we wanted proximal. For simple L2 decay, we decay towards 0.
                # Assuming you want Proximal (stay close to previous GPU weight):
                w_prev_cpu = self.final_layer_second_stage_weight.detach().cpu().float()
                diff = w_cpu - w_prev_cpu
                loss = loss + 0.5 * weight_decay * torch.sum(diff ** 2)

            # Negative Penalty
            if negative_penalty > 0:
                loss += (nn.ReLU()(-self.dens_ratio_transformer.inverse_transform(pred)) ** 2).mean() * negative_penalty
            loss.backward()
            return loss

        # 7. Run Optimization
        try:
            for _ in range(n_steps):
                optimizer.step(closure)
        except RuntimeError as e:
            # L-BFGS on CPU is rare to crash, but good to handle
            print(f"L-BFGS CPU Warning: {e}")

        # 8. Copy Updated Weights Back to Device
        # This updates the main model's weights with the optimized result
        self.final_layer_second_stage_weight.data.copy_(w_cpu.data.to(self.device))
        self.dens_ratio_transformer.to(self.device)


def train_treatment_pcl_net_ate_model(
    model: nn.Module,
    first_stage_train_dataloader,
    second_stage_train_dataloader,
    stage1_optimizers: List[torch.optim.Optimizer],
    stage2_optimizers: List[torch.optim.Optimizer],
    n_epochs: int = 10,
    stage1_iter: int = 1,
    stage2_iter: int = 1,
    first_stage_final_layer_regularizer: Union[float, Tuple[float, float]] = 1e-2,
    second_stage_final_layer_regularizer: Union[float, Tuple[float, float]] = 1e-2,
    second_stage_first_final_layer_regularizer: Union[float, Tuple[float, float]] = 1e-2,
    regularizer_annealing_method: str = "linear", # "linear" or "cosine"
    consider_prev_weight: bool = True,
    first_stage_loss_fn = None,
    second_stage_loss_fn = None,
    ############################### parameters for the SGD head update in the second stage ###############################
    second_stage_head_lr: float = 1e-2,
    second_stage_head_steps: int = 5,
    ###########################################
    negative_penalty: float = 0.0,
    log_per_epoch: int = 1,
    validation_dataloader = None,
    plot_loss: bool = True,
    stage1_schedulers: Optional[List[Scheduler]] = None,
    stage2_schedulers: Optional[List[Scheduler]] = None,
    **kwargs
):
    if first_stage_loss_fn is None: first_stage_loss_fn = nn.MSELoss()
    if second_stage_loss_fn is None: second_stage_loss_fn = nn.MSELoss()
    if validation_dataloader is None: validation_dataloader = first_stage_train_dataloader
    if plot_loss: plt.figure(figsize=(32, 12), dpi=80)

    stage1_loss_hist, stage1_val_loss_hist, stage2_loss_hist, stage2_val_loss_hist = [], [], [], []
    # Initial evaluation before training
    model.eval()
    stage2_loss_hist.append(model.evaluate_loss(second_stage_train_dataloader, "second"))
    stage2_val_loss_hist.append(model.evaluate_loss(validation_dataloader, "second"))
    # --- 1. Parse Regularization Schedules ---
    def parse_reg_schedule(reg_input):
        if isinstance(reg_input, (tuple, list)):
            return reg_input[0], reg_input[1]
        return reg_input, reg_input # Constant if single float provided

    reg1_start, reg1_end = parse_reg_schedule(first_stage_final_layer_regularizer)
    reg2_start, reg2_end = parse_reg_schedule(second_stage_final_layer_regularizer)
    reg12_start, reg12_end = parse_reg_schedule(second_stage_first_final_layer_regularizer)

    # --- HELPER: Infinite Iterator for the smaller dataset ---
    # This prevents 'zip' from cutting off the larger dataset
    def cycle(iterable):
        while True:
            for x in iterable: yield x

    # Identify which loader is longer to define "One Epoch"
    len_1 = len(first_stage_train_dataloader)
    len_2 = len(second_stage_train_dataloader)
    batches_per_epoch = max(len_1, len_2)

    for epoch in tqdm(range(n_epochs)):
        epoch_stage1_loss = 0
        counter = 0
        model.train()

        # --- 2. CALL ANNEALING FUNCTION ---
        curr_reg_1 = get_annealed_value_for_reg(reg1_start, reg1_end, epoch, n_epochs, method=regularizer_annealing_method)
        curr_reg_2 = get_annealed_value_for_reg(reg2_start, reg2_end, epoch, n_epochs, method=regularizer_annealing_method)
        curr_reg_12 = get_annealed_value_for_reg(reg12_start, reg12_end, epoch, n_epochs, method=regularizer_annealing_method)

        # Create Iterators (Cycle the shorter one)
        if len_1 >= len_2:
            iter_1 = iter(first_stage_train_dataloader)
            iter_2 = cycle(second_stage_train_dataloader)
        else:
            iter_1 = cycle(first_stage_train_dataloader)
            iter_2 = iter(second_stage_train_dataloader)

        for _ in range(batches_per_epoch):
            # Fetch Batches
            first_stage_data_tuple = next(iter_1)
            second_stage_data_tuple = next(iter_2)

            model.first_stage_featurizer.train(True)
            model.treatment_backdoor_featurizer.train(False)
            model.treatment_proxy_featurizer.train(False)

            for ii in range(stage1_iter):
                for opt in stage1_optimizers:
                    opt.zero_grad()

                ## First stage
                for opt in stage1_optimizers:
                    opt.zero_grad()

                first_stage_loss = model.first_stage_loss(first_stage_data_tuple, loss_fn = first_stage_loss_fn)
                first_stage_loss.backward()

                for opt in stage1_optimizers:
                    opt.step()

                with torch.no_grad():
                    model.update_final_layer_with_batch_regression( first_stage_data_tuple,
                                                                    curr_reg_1, stage = "first",
                                                                    consider_prev_weight = consider_prev_weight)

            epoch_stage1_loss += first_stage_loss.item()

            model.first_stage_featurizer.train(False)
            model.treatment_backdoor_featurizer.train(True)
            model.treatment_proxy_featurizer.train(True)

            ## Second stage
            for ii in range(stage2_iter):
                for opt in stage2_optimizers:
                    opt.zero_grad()

                second_stage_loss = model.second_stage_loss(first_stage_data_tuple, second_stage_data_tuple,
                                                            second_stage_loss_fn,
                                                            second_stage_first_final_layer_regularizer = curr_reg_12,
                                                            negative_penalty = negative_penalty)
                second_stage_loss.backward()

                for opt in stage2_optimizers:
                    opt.step()

                model.update_second_stage_final_layer_with_sgd(
                    data_tuple=second_stage_data_tuple,
                    loss_fn=second_stage_loss_fn,
                    lr=second_stage_head_lr,
                    n_steps=second_stage_head_steps,
                    weight_decay=curr_reg_2, # Annealed L2 regularization
                    negative_penalty=negative_penalty,
                )
            counter += 1
        # ==========================
        # Scheduler Step (End of Epoch)
        # ==========================
        if stage1_schedulers:
            for sched in stage1_schedulers:
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Step with validation metric if available
                    if len(stage1_val_loss_hist) > 0: sched.step(stage1_val_loss_hist[-1])
                else:
                    sched.step()

        if stage2_schedulers:
            for sched in stage2_schedulers:
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if len(stage2_val_loss_hist) > 0: sched.step(stage2_val_loss_hist[-1])
                else:
                    sched.step()

        if (epoch % log_per_epoch == 0) | (epoch == n_epochs - 1):
            model.eval()
            stage1_loss_hist.append(epoch_stage1_loss / counter)
            stage1_val_loss_hist.append(model.evaluate_loss(first_stage_train_dataloader, "first"))
            stage2_loss_hist.append(model.evaluate_loss(second_stage_train_dataloader, "second"))
            stage2_val_loss_hist.append(model.evaluate_loss(validation_dataloader, "second"))
            if plot_loss:
                val_loss_x_axis = np.arange(1, len(stage1_val_loss_hist) + 1)

                plt.clf()
                plt.subplot(1, 2, 1)
                plt.plot(val_loss_x_axis, stage1_loss_hist, linewidth=5, label = 'Training Averaged Loss')
                plt.plot(val_loss_x_axis, stage1_val_loss_hist, linewidth=5, label = 'Validation Loss')
                plt.xlabel(f"Number of Epochs / {log_per_epoch}", fontsize=20)
                plt.ylabel("MSE", fontsize=20)
                plt.title("Averaged Stage 1 Loss: {}".format(stage1_loss_hist[-1]), fontsize=20)
                plt.grid()
                plt.legend(fontsize=15)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)

                val_loss_x_axis = np.arange(len(stage2_val_loss_hist))
                plt.subplot(1, 2, 2)
                plt.plot(val_loss_x_axis, stage2_loss_hist, linewidth=5, label = 'Training Loss')
                plt.plot(val_loss_x_axis, stage2_val_loss_hist, linewidth=5, label = 'Validation Loss')
                plt.xlabel(f"Number of Epochs / {log_per_epoch}", fontsize=20)
                plt.ylabel(r"$\mathcal{L}_2$", fontsize=20)
                plt.title("Training Stage 2 Loss: {}\n Validation Stage 2 Loss: {}\n".format(stage2_loss_hist[-1],
                                                                                            stage2_val_loss_hist[-1]), fontsize=20)
                plt.grid()
                plt.legend(fontsize=15)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                clear_output(wait=True)
                display(plt.gcf())

    return model.eval()