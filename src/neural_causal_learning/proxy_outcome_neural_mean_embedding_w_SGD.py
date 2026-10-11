import torch
import torch.nn as nn
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
from neural_causal_learning.proxy_outcome_neural_mean_embedding import OutcomeBridgePCLNET, HeterogeneousOutcomeBridgePCLNET, get_annealed_value_for_reg


class OutcomeBridgePCLNET(OutcomeBridgePCLNET):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_second_stage_final_layer_with_sgd(
            self,
            data_tuple: Tuple[torch.Tensor, ...],
            loss_fn: Callable,
            lr: float = 1.0,
            n_steps: int = 10,
            weight_decay: float = 0.0,
        ):
        """
        Updates the final layer using L-BFGS on the CPU for maximum stability.
        """
        # 1. Unpack Data
        treatment, Y, treatment_proxy, outcome_proxy, backdoor = self._unpack_batch(data_tuple)

        # 2. Compute Features on GPU (Fast)
        # We keep the heavy lifting (feature extraction) on the GPU
        with torch.no_grad():
            w1_inner = self.final_layer_first_stage_inside_second_stage_weight.T

            # Get features on GPU
            features_gpu = self._compute_second_stage_features(
                treatment, treatment_proxy, backdoor, w1_inner
            )

            # 3. Move to CPU for Optimization (Stable)
            # Casting to float ensures standard precision for LBFGS
            X_cpu = features_gpu.detach().cpu().float()
            Y_cpu = Y.detach().cpu().float()

            # Safety: Check for NaNs before they crash the CPU optimizer
            if torch.isnan(X_cpu).any() or torch.isinf(X_cpu).any():
                return

        # 4. Snapshot Weight to CPU
        # We work on a detached CPU copy of the weights.
        # This prevents disrupting the main model's computation graph.
        w_cpu = self.final_layer_second_stage_weight.detach().cpu().float().clone()
        w_cpu.requires_grad_(True)

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


class HeterogeneousOutcomeBridgePCLNET(HeterogeneousOutcomeBridgePCLNET):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_second_stage_final_layer_with_sgd(
            self,
            data_tuple: Tuple[torch.Tensor, ...],
            loss_fn: Callable,
            lr: float = 1.0,
            n_steps: int = 10,
            weight_decay: float = 0.0,
        ):
        """
        Updates the final layer using L-BFGS on the CPU for maximum stability.
        """
        # 1. Unpack Data
        treatment, Y, treatment_proxy, outcome_proxy, covariate, backdoor = self._unpack_batch(data_tuple)

        # 2. Compute Features on GPU (Fast)
        # We keep the heavy lifting (feature extraction) on the GPU
        with torch.no_grad():
            w1_inner = self.final_layer_first_stage_inside_second_stage_weight.T

            # Get features on GPU
            features_gpu = self._compute_second_stage_features(
                treatment, covariate, treatment_proxy, backdoor, w1_inner
            )

            # 3. Move to CPU for Optimization (Stable)
            # Casting to float ensures standard precision for LBFGS
            X_cpu = features_gpu.detach().cpu().float()
            Y_cpu = Y.detach().cpu().float()

            # Safety: Check for NaNs before they crash the CPU optimizer
            if torch.isnan(X_cpu).any() or torch.isinf(X_cpu).any():
                return

        # 4. Snapshot Weight to CPU
        # We work on a detached CPU copy of the weights.
        # This prevents disrupting the main model's computation graph.
        w_cpu = self.final_layer_second_stage_weight.detach().cpu().float().clone()
        w_cpu.requires_grad_(True)

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


def train_deep_feature_proxy_closed_form_ate_model(
        pcl_model: torch.nn.Module,
        first_stage_train_dataloader,
        second_stage_train_dataloader,
        stage1_optimizers,
        stage2_optimizers,
        stage1_schedulers: Optional[List[Scheduler]] = None,
        stage2_schedulers: Optional[List[Scheduler]] = None,
        first_stage_final_layer_regularizer: Union[float, Tuple[float, float]] = 1e-2,
        second_stage_final_layer_regularizer: Union[float, Tuple[float, float]] = 1e-2,
        second_stage_first_final_layer_regularizer: Union[float, Tuple[float, float]] = 1e-2,
        regularizer_annealing_method: str = "exponential", # "linear" or "cosine"
        consider_prev_weight: bool = True,
        first_stage_loss_fn = None,
        second_stage_loss_fn = None,
        ############################### parameters for the SGD head update in the second stage ###############################
        second_stage_head_lr: float = 1e-2,
        second_stage_head_steps: int = 5,
        ###########################################
        n_epochs = 1,
        stage1_iter: int = 1,
        stage2_iter: int = 1,
        validation_dataloader: Optional[DataLoader] = None,
        log_per_epoch: int = 25,
        **kwargs,
        ):

    if first_stage_loss_fn is None: first_stage_loss_fn = nn.MSELoss()
    if second_stage_loss_fn is None: second_stage_loss_fn = nn.MSELoss()
    if validation_dataloader is None: validation_dataloader = first_stage_train_dataloader

    plot_loss = kwargs.pop('plot_loss', False)
    do_A = kwargs.pop('do_A', None) # These are only for debugging purposes in the case that interventional curves are available!
    EY_do_A = kwargs.pop('EY_do_A', None)

    # Check if we have ground truth and if the model is the standard (non-heterogeneous) version
    is_standard_outcome_model = isinstance(pcl_model, OutcomeBridgePCLNET)
    if plot_loss:
        plt.figure(figsize=(18, 14) if ((do_A is not None) and (EY_do_A is not None) and is_standard_outcome_model) else (14, 8))

    # --- Helper: Infinite Iterator ---
    def cycle(iterable):
        while True:
            for x in iterable: yield x

    len_1 = len(first_stage_train_dataloader)
    len_2 = len(second_stage_train_dataloader)
    batches_per_epoch = max(len_1, len_2)

    device = pcl_model.device

    stage1_loss_hist = []
    stage2_loss_hist = []
    stage2_val_loss_hist = []

    # --- 1. Parse Regularization Schedules ---
    def parse_reg_schedule(reg_input):
        if isinstance(reg_input, (tuple, list)):
            return reg_input[0], reg_input[1]
        return reg_input, reg_input # Constant if single float provided

    reg1_start, reg1_end = parse_reg_schedule(first_stage_final_layer_regularizer)
    reg2_start, reg2_end = parse_reg_schedule(second_stage_final_layer_regularizer)
    reg12_start, reg12_end = parse_reg_schedule(second_stage_first_final_layer_regularizer)

    if (do_A is not None) and (EY_do_A is not None) and is_standard_outcome_model: causal_mse_list, mu_W_norm_hist = [], []

    for epoch in tqdm(range(n_epochs)):
        epoch_stage1_loss = 0
        counter = 0
        pcl_model.train()

        # --- 2. CALL ANNEALING FUNCTION ---
        curr_reg_1 = get_annealed_value_for_reg(reg1_start, reg1_end, epoch, n_epochs, method=regularizer_annealing_method)
        curr_reg_2 = get_annealed_value_for_reg(reg2_start, reg2_end, epoch, n_epochs, method=regularizer_annealing_method)
        curr_reg_12 = get_annealed_value_for_reg(reg12_start, reg12_end, epoch, n_epochs, method=regularizer_annealing_method)

        # Iterators
        if len_1 >= len_2:
            iter_1 = iter(first_stage_train_dataloader)
            iter_2 = cycle(second_stage_train_dataloader)
        else:
            iter_1 = cycle(first_stage_train_dataloader)
            iter_2 = iter(second_stage_train_dataloader)

        for _ in range(batches_per_epoch):
            first_stage_data_tuple = next(iter_1)
            second_stage_data_tuple = next(iter_2)

            pcl_model.freeze_featurizer_params("second_stage")
            pcl_model.unfreeze_featurizer_params("first_stage")

            for ii in range(stage1_iter):
                for opt in stage1_optimizers:
                    opt.zero_grad()
                loss_stage1 = pcl_model.first_stage_loss(first_stage_data_tuple,
                                                         loss_fn = first_stage_loss_fn)
                loss_stage1.backward()
                for opt in stage1_optimizers:
                    opt.step()
                with torch.no_grad():
                    pcl_model.update_final_layer_with_batch_regression(first_stage_data_tuple, curr_reg_1,
                                                                        "first", consider_prev_weight)
            epoch_stage1_loss += loss_stage1.item()
            # Stage 2 update
            pcl_model.freeze_featurizer_params("first_stage")
            pcl_model.unfreeze_featurizer_params("second_stage")

            for i in range(stage2_iter):
                for opt in stage2_optimizers:
                    opt.zero_grad()

                loss_stage2 = pcl_model.second_stage_loss(first_stage_data_tuple,
                                                          second_stage_data_tuple,
                                                          loss_fn = second_stage_loss_fn,
                                                          second_stage_first_final_layer_regularizer = curr_reg_12)
                loss_stage2.backward()
                for opt in stage2_optimizers:
                    opt.step()

                pcl_model.update_second_stage_final_layer_with_sgd(
                    data_tuple=second_stage_data_tuple,
                    loss_fn=second_stage_loss_fn,
                    lr=second_stage_head_lr,
                    n_steps=second_stage_head_steps,
                    weight_decay=curr_reg_2, # Annealed L2 regularization
                )
            counter += 1
        # Schedulers (Epoch Level)
        if stage1_schedulers:
            for s in stage1_schedulers: s.step()
        if stage2_schedulers:
            for s in stage2_schedulers: s.step()
        stage1_loss_hist.append(epoch_stage1_loss / counter)

        pcl_model.eval()
        if (epoch % log_per_epoch == 0) | (epoch == n_epochs - 1):
            stage2_loss_hist.append(pcl_model.evaluate_second_stage_loss(second_stage_train_dataloader, loss_fn=second_stage_loss_fn))
            stage2_val_loss_hist.append(pcl_model.evaluate_second_stage_loss(validation_dataloader, loss_fn=second_stage_loss_fn))
            if plot_loss:
                n_plots = 2 + 2 * int(((do_A is not None) and (EY_do_A is not None) and is_standard_outcome_model))
                plt.clf()
                plt.subplot(int(n_plots / 2), 2, 1)
                plt.plot(np.arange(epoch + 1), stage1_loss_hist, linewidth=5)
                plt.xlabel("Number of Epochs", fontsize=20)
                plt.ylabel("MSE", fontsize=20)
                plt.title("Averaged Stage 1 Loss: {}".format(stage1_loss_hist[-1]), fontsize=20)
                plt.grid()
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)

                val_loss_x_axis = np.arange(len(stage2_val_loss_hist)) * log_per_epoch
                plt.subplot(int(n_plots / 2), 2, 2)
                plt.plot(val_loss_x_axis, stage2_loss_hist, linewidth=5, label = 'Training Loss')
                plt.plot(val_loss_x_axis, stage2_val_loss_hist, linewidth=5, label = 'Validation Loss')
                plt.xlabel("Number of Epochs", fontsize=20)
                plt.ylabel("MSE", fontsize=20)
                plt.title("Training Stage 2 Loss: {}\n Validation Stage 2 Loss: {}\n".format(stage2_loss_hist[-1],
                                                                                            stage2_val_loss_hist[-1]), fontsize=20)
                plt.grid()
                plt.legend()
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)

                if (do_A is not None) and (EY_do_A is not None) and is_standard_outcome_model:
                    pcl_model.compute_mean_outcome_proxy_feature(second_stage_train_dataloader)
                    # 2. Get the mean feature vector
                    mean_proxy_feature = pcl_model.mean_outcome_proxy_feature.detach()

                    # 3. Calculate L2 Norm and store history
                    mu_W_norm = torch.linalg.norm(mean_proxy_feature).item()
                    mu_W_norm_hist.append(mu_W_norm)

                    f_struct_pred = pcl_model.pred_structural_function( do_A,
                                                                        second_stage_train_dataloader.dataset.transformers[0],
                                                                        second_stage_train_dataloader.dataset.transformers[1]
                                              ).detach().cpu().numpy()
                    structured_pred_mse = (np.mean((f_struct_pred.reshape(-1, 1) - EY_do_A.detach().cpu().numpy().reshape(-1, 1)) ** 2))
                    causal_mse_list.append(structured_pred_mse)

                    plt.subplot(int(n_plots / 2), 2, 3)
                    plt.plot(np.arange(len(causal_mse_list)), causal_mse_list, linewidth=5)
                    plt.xlabel("Number of Epochs / {}".format(log_per_epoch), fontsize=20)
                    plt.ylabel("Causal MSE", fontsize=20)
                    plt.title("Causal MSE: {}".format(structured_pred_mse), fontsize=20)
                    plt.grid()
                    plt.xticks(fontsize=20)
                    plt.yticks(fontsize=20)

                    # Plot 4: Feature Magnitude (Diagnostic)
                    plt.subplot(int(n_plots / 2), 2, 4)
                    plt.plot(np.arange(len(mu_W_norm_hist)), mu_W_norm_hist, linewidth=5)
                    plt.title(f"Mean Proxy Feature Norm", fontsize=20)
                    plt.xlabel("Number of Epochs / {}".format(log_per_epoch), fontsize=20)
                    plt.ylabel("L2 Norm", fontsize=20)
                    plt.grid()
                    plt.xticks(fontsize=20)
                    plt.yticks(fontsize=20)

                clear_output(wait=True)
                display(plt.gcf())

    pcl_model.compute_mean_outcome_proxy_feature(second_stage_train_dataloader)
    return pcl_model.eval()