import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional
from torch.nn.utils import spectral_norm
import torch.nn.utils.parametrizations as parametrizations

class GaussianFourierProjection(nn.Module):
    """
    Projects 1D input to high-dimensional Fourier features.
    x -> [sin(2*pi*x*B), cos(2*pi*x*B)]
    This guarantees inputs are mapped to a high-rank manifold.
    """
    def __init__(self, input_dim=1, embedding_size=64, scale=1.0):
        super().__init__()
        # B is a fixed, non-learnable matrix sampled from Normal(0, scale)
        self.B = nn.Parameter(torch.randn(input_dim, embedding_size) * scale, requires_grad=False)
        self.embedding_size = embedding_size

    def forward(self, x):
        # x: (N, 1) -> (N, embedding_size)
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1) / np.sqrt(self.embedding_size)


def build_nets_for_treatment_pcl_net_synthetic_low_dim_experiment(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Builds efficient featurizer networks for the low-dimensional synthetic benchmark.
    Significantly reduces final output dimensions (from 64/128 to 16/32) to speed up
    closed-form matrix operations (reducing the final layer size from 8192 to 512).
    """
    FIRST_STAGE_OUTPUT_DIM = 128
    Z_OUTPUT_DIM = 16
    SECOND_STAGE_AX_OUTPUT_DIM = 32
    INTERNAL_WIDTH = 512 # Keep internal layers reasonably wide
    DROPOUT_RATE = 0.05


    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "linear"):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, INTERNAL_WIDTH),
                nn.LayerNorm(INTERNAL_WIDTH),
                nn.GELU(),
                # nn.BatchNorm1d(INTERNAL_WIDTH),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH, INTERNAL_WIDTH * 2),
                nn.LayerNorm(INTERNAL_WIDTH * 2),
                nn.GELU(),
                # nn.BatchNorm1d(INTERNAL_WIDTH // 2),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH * 2, output_dim),
                nn.LayerNorm(output_dim),
            )
            self.output_dim = output_dim
            self.final_activation = final_activation
            self.final_bn = nn.BatchNorm1d(output_dim)
            self.tanh = nn.Tanh()
            self.gelu = nn.GELU()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            if self.final_activation == "tanh":
                return self.tanh(out)
            elif self.final_activation == "gelu":
                return self.gelu(out)
            return out

    # --- Treatment Featurizers (Input: 1D) ---
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 3):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM, final_activation="gelu")

    # --- Outcome Proxy Featurizer (Input: 2D) ---
    class SecondStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_AX_OUTPUT_DIM, final_activation="gelu")

    # --- Treatment Proxy Featurizer (Input: 2D) ---
    class TreatmentProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 2):
            super().__init__(input_dim=input_dim, output_dim=Z_OUTPUT_DIM, final_activation="gelu")

    def _small_init(m):
        if isinstance(m, nn.Linear):
            # We use a very small scale (0.01) to squash the initial output variance
            nn.init.kaiming_normal_(m.weight, a=0.01) # Use GELU/ReLU compatible init
            m.weight.data *= 0.05
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 0.01) # Scale down LayerNorm gains too

    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    second_stage_ax_featurizer = SecondStageFeaturizer().to(device)
    treatment_proxy_featurizer = TreatmentProxyFeaturizer().to(device)
    treatment_proxy_featurizer.apply(_small_init)
    return first_stage_featurizer, second_stage_ax_featurizer, treatment_proxy_featurizer


def build_nets_for_treatment_pcl_net_synthetic_high_dim_new_version_experimentV1(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    FIRST_STAGE_OUTPUT_DIM = 128
    Z_OUTPUT_DIM = 8
    SECOND_STAGE_AX_OUTPUT_DIM = 16
    INTERNAL_WIDTH = 512 # Keep internal layers reasonably wide
    DROPOUT_RATE = 0.1


    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, DROPOUT_RATE: float = 0.0):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, INTERNAL_WIDTH),
                nn.LayerNorm(INTERNAL_WIDTH),
                nn.GELU(),
                # nn.BatchNorm1d(INTERNAL_WIDTH),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH, INTERNAL_WIDTH * 2),
                nn.LayerNorm(INTERNAL_WIDTH * 2),
                nn.GELU(),
                # nn.BatchNorm1d(INTERNAL_WIDTH * 2),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH * 2, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                # nn.BatchNorm1d(output_dim)
            )
            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    # --- Treatment Featurizers (Input: 1D) ---
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 111):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM)

    # --- Outcome Proxy Featurizer (Input: 2D) ---
    class SecondStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 101):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_AX_OUTPUT_DIM)

    # --- Treatment Proxy Featurizer (Input: 2D) ---
    class TreatmentProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 10, DROPOUT_RATE: float = DROPOUT_RATE):
            super().__init__(input_dim=input_dim, output_dim=Z_OUTPUT_DIM)


    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    second_stage_ax_featurizer = SecondStageFeaturizer().to(device)
    treatment_proxy_featurizer = TreatmentProxyFeaturizer().to(device)

    return first_stage_featurizer, second_stage_ax_featurizer, treatment_proxy_featurizer

def build_nets_for_treatment_pcl_net_synthetic_high_dim_new_version_experiment(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Builds efficient featurizer networks for the low-dimensional synthetic benchmark.
    Significantly reduces final output dimensions (from 64/128 to 16/32) to speed up
    closed-form matrix operations (reducing the final layer size from 8192 to 512).
    """
    FIRST_STAGE_OUTPUT_DIM = 128
    Z_OUTPUT_DIM = 8
    SECOND_STAGE_AX_OUTPUT_DIM = 16
    INTERNAL_WIDTH = 512 # Keep internal layers reasonably wide
    DROPOUT_RATE = 0.1


    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, DROPOUT_RATE: float = 0.0):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, INTERNAL_WIDTH),
                nn.LayerNorm(INTERNAL_WIDTH),
                nn.GELU(),
                nn.BatchNorm1d(INTERNAL_WIDTH),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH, INTERNAL_WIDTH * 2),
                nn.LayerNorm(INTERNAL_WIDTH * 2),
                nn.GELU(),
                nn.BatchNorm1d(INTERNAL_WIDTH * 2),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH * 2, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.BatchNorm1d(output_dim)
            )
            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    # --- Treatment Featurizers (Input: 1D) ---
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 111):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM)

    # --- Outcome Proxy Featurizer (Input: 2D) ---
    class SecondStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 101):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_AX_OUTPUT_DIM)

    # --- Treatment Proxy Featurizer (Input: 2D) ---
    class TreatmentProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 10, DROPOUT_RATE: float = DROPOUT_RATE):
            super().__init__(input_dim=input_dim, output_dim=Z_OUTPUT_DIM)

    def _small_init(m):
        if isinstance(m, nn.Linear):
            # We use a very small scale (0.01) to squash the initial output variance
            nn.init.kaiming_normal_(m.weight, a=0.05) # Use GELU/ReLU compatible init
            m.weight.data *= 0.05
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 0.05) # Scale down LayerNorm gains too

    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    second_stage_ax_featurizer = SecondStageFeaturizer().to(device)
    treatment_proxy_featurizer = TreatmentProxyFeaturizer().to(device)
    treatment_proxy_featurizer.apply(_small_init)

    return first_stage_featurizer, second_stage_ax_featurizer, treatment_proxy_featurizer
    # FIRST_STAGE_OUTPUT_DIM = 128
    # Z_OUTPUT_DIM = 8
    # SECOND_STAGE_AX_OUTPUT_DIM = 16
    # INTERNAL_WIDTH = 512 # Keep internal layers reasonably wide
    # DROPOUT_RATE = 0.1


    # class FeaturizerBase(nn.Module):
    #     def __init__(self, input_dim: int, output_dim: int, DROPOUT_RATE: float = 0.0):
    #         super.__init__
    #         self.net = nn.Sequential(
    #             nn.Linear(input_dim, INTERNAL_WIDTH),
    #             nn.LayerNorm(INTERNAL_WIDTH),
    #             nn.GELU,
    #             # nn.BatchNorm1d(INTERNAL_WIDTH),
    #             nn.Dropout(DROPOUT_RATE),
    #             nn.Linear(INTERNAL_WIDTH, INTERNAL_WIDTH * 2),
    #             nn.LayerNorm(INTERNAL_WIDTH * 2),
    #             nn.GELU,
    #             # nn.BatchNorm1d(INTERNAL_WIDTH * 2),
    #             nn.Dropout(DROPOUT_RATE),
    #             nn.Linear(INTERNAL_WIDTH * 2, output_dim),
    #             nn.LayerNorm(output_dim),
    #             nn.GELU,
    #             # nn.BatchNorm1d(output_dim)
    #         )
    #         self.output_dim = output_dim

    #     def forward(self, x: torch.Tensor) -> torch.Tensor:
    #         return self.net(x)

    # # --- Treatment Featurizers (Input: 1D) ---
    # class FirstStageFeaturizer(FeaturizerBase):
    #     def __init__(self, input_dim: int = 111):
    #         super.__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM)

    # # --- Outcome Proxy Featurizer (Input: 2D) ---
    # class SecondStageFeaturizer(FeaturizerBase):
    #     def __init__(self, input_dim: int = 101):
    #         super.__init__(input_dim=input_dim, output_dim=SECOND_STAGE_AX_OUTPUT_DIM)

    # # --- Treatment Proxy Featurizer (Input: 2D) ---
    # class TreatmentProxyFeaturizer(FeaturizerBase):
    #     def __init__(self, input_dim: int = 10, DROPOUT_RATE: float = DROPOUT_RATE):
    #         super.__init__(input_dim=input_dim, output_dim=Z_OUTPUT_DIM)


    # # Instantiate the models
    # first_stage_featurizer = FirstStageFeaturizer.to(device)
    # second_stage_ax_featurizer = SecondStageFeaturizer.to(device)
    # treatment_proxy_featurizer = TreatmentProxyFeaturizer.to(device)

    # return first_stage_featurizer, second_stage_ax_featurizer, treatment_proxy_featurizer


def build_nets_for_treatment_pcl_net_synthetic_high_dim_experiment(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Builds efficient featurizer networks for the low-dimensional synthetic benchmark.
    Significantly reduces final output dimensions (from 64/128 to 16/32) to speed up
    closed-form matrix operations (reducing the final layer size from 8192 to 512).
    """
    FIRST_STAGE_OUTPUT_DIM = 128
    Z_OUTPUT_DIM = 8
    SECOND_STAGE_AX_OUTPUT_DIM = 16
    INTERNAL_WIDTH = 512 # Keep internal layers reasonably wide
    DROPOUT_RATE = 0.2


    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, DROPOUT_RATE: float = 0.0):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, INTERNAL_WIDTH),
                nn.LayerNorm(INTERNAL_WIDTH),
                nn.GELU(),
                nn.BatchNorm1d(INTERNAL_WIDTH),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH, INTERNAL_WIDTH * 2),
                nn.LayerNorm(INTERNAL_WIDTH * 2),
                nn.GELU(),
                nn.BatchNorm1d(INTERNAL_WIDTH * 2),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH * 2, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.BatchNorm1d(output_dim)
            )
            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    # --- Treatment Featurizers (Input: 1D) ---
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 111):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM)

    # --- Outcome Proxy Featurizer (Input: 2D) ---
    class SecondStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 101):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_AX_OUTPUT_DIM)

    # --- Treatment Proxy Featurizer (Input: 2D) ---
    class TreatmentProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 10, DROPOUT_RATE: float = DROPOUT_RATE):
            super().__init__(input_dim=input_dim, output_dim=Z_OUTPUT_DIM)

    def _small_init(m):
        if isinstance(m, nn.Linear):
            # We use a very small scale (0.01) to squash the initial output variance
            nn.init.kaiming_normal_(m.weight, a=0.05) # Use GELU/ReLU compatible init
            m.weight.data *= 0.05
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 0.05) # Scale down LayerNorm gains too

    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    second_stage_ax_featurizer = SecondStageFeaturizer().to(device)
    treatment_proxy_featurizer = TreatmentProxyFeaturizer().to(device)
    treatment_proxy_featurizer.apply(_small_init)

    return first_stage_featurizer, second_stage_ax_featurizer, treatment_proxy_featurizer


def build_nets_for_treatment_pcl_net_abortion_and_crime_experiment(device: str = "cuda", w_dim: int = 3) -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    # w_dim = number of outcome-proxy (W) columns kept; the treatment first stage regresses on concat(A, W),
    # so its input dim is 1 + w_dim (default 3 = full).
    """
    Builds efficient featurizer networks for the low-dimensional synthetic benchmark.
    Significantly reduces final output dimensions (from 64/128 to 16/32) to speed up
    closed-form matrix operations (reducing the final layer size from 8192 to 512).
    """
    FIRST_STAGE_OUTPUT_DIM = 16
    Z_OUTPUT_DIM = 8
    SECOND_STAGE_AX_OUTPUT_DIM = 8
    INTERNAL_WIDTH = 32
    DROPOUT_RATE = 0.01

    class GaussianNoise(nn.Module):
        def __init__(self, std=0.05):
            super().__init__()
            self.std = std
        def forward(self, x):
            if self.training and self.std > 0:
                return x + torch.randn_like(x) * self.std
            return x


    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "linear"):
            super().__init__()
            self.net = nn.Sequential(
                # GaussianNoise(std=0.01),
                nn.Linear(input_dim, INTERNAL_WIDTH),
                nn.LayerNorm(INTERNAL_WIDTH),
                nn.GELU(),
                # nn.BatchNorm1d(INTERNAL_WIDTH),
                nn.Dropout(DROPOUT_RATE),
                # nn.Linear(INTERNAL_WIDTH, INTERNAL_WIDTH // 2),
                # nn.LayerNorm(INTERNAL_WIDTH // 2),
                # nn.GELU,
                # nn.BatchNorm1d(INTERNAL_WIDTH // 2),
                # nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH, output_dim),
                # nn.LayerNorm(output_dim),
            )
            self.output_dim = output_dim
            self.final_activation = final_activation
            self.tanh = nn.Tanh()
            self.gelu = nn.GELU()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            if self.final_activation == "tanh":
                return self.tanh(out)
            elif self.final_activation == "gelu":
                return self.gelu(out)
            return out

    # --- Treatment Featurizers (Input: 1D) ---
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1 + w_dim, final_activation="gelu"):   # concat(A, W): 1 + w_dim
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM)

    # --- Outcome Proxy Featurizer (Input: 2D) ---
    class SecondStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1, final_activation="gelu"):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_AX_OUTPUT_DIM)

    # --- Treatment Proxy Featurizer (Input: 2D) ---
    class TreatmentProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1, final_activation="gelu"):
            super().__init__(input_dim=input_dim, output_dim=Z_OUTPUT_DIM)


    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    second_stage_ax_featurizer = SecondStageFeaturizer().to(device)
    treatment_proxy_featurizer = TreatmentProxyFeaturizer().to(device)

    return first_stage_featurizer, second_stage_ax_featurizer, treatment_proxy_featurizer


def build_nets_for_treatment_pcl_net_deaner_experiment(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module]:
    """
    Treatment-bridge featurizers for the Deaner grade-retention benchmark (IM/IR).

    Identical to the abortion-and-crime treatment nets EXCEPT the treatment-proxy featurizer
    takes Z of dim 4 (Deaner Z is 4-D) instead of 1. The first-stage featurizer input stays
    A(1)+W(3)=4 and the A-featurizer stays 1, since W is 3-D in both benchmarks.
    """
    FIRST_STAGE_OUTPUT_DIM = 16
    Z_OUTPUT_DIM = 8
    SECOND_STAGE_AX_OUTPUT_DIM = 8
    INTERNAL_WIDTH = 32
    DROPOUT_RATE = 0.01

    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "linear"):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, INTERNAL_WIDTH),
                nn.LayerNorm(INTERNAL_WIDTH),
                nn.GELU(),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH, output_dim),
            )
            self.output_dim = output_dim
            self.final_activation = final_activation
            self.tanh = nn.Tanh()
            self.gelu = nn.GELU()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            if self.final_activation == "tanh":
                return self.tanh(out)
            elif self.final_activation == "gelu":
                return self.gelu(out)
            return out

    # First stage input = concat(A, W) = 1 + 3 = 4
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 4, final_activation="gelu"):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM)

    # A featurizer, input A = 1
    class SecondStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1, final_activation="gelu"):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_AX_OUTPUT_DIM)

    # Treatment-proxy featurizer, input Z = 4  (the only change from abortion)
    class TreatmentProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 4, final_activation="gelu"):
            super().__init__(input_dim=input_dim, output_dim=Z_OUTPUT_DIM)

    first_stage_featurizer = FirstStageFeaturizer().to(device)
    second_stage_ax_featurizer = SecondStageFeaturizer().to(device)
    treatment_proxy_featurizer = TreatmentProxyFeaturizer().to(device)

    return first_stage_featurizer, second_stage_ax_featurizer, treatment_proxy_featurizer


def build_nets_for_treatment_pcl_net_rhc_experiment(
        device: str = "cuda", x_dim: int = 20,
        width: int = 64, first_stage_out: int = 16, z_out: int = 8, ax_out: int = 8,
        dropout: float = 0.01) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """
    Treatment-bridge featurizers for the Right Heart Catheterization (RHC) benchmark.

    RHC folds the **backdoor X (20-D)** into the treatment-bridge inputs (as the high-dim synthetic
    treatment builder does -- there is no separate backdoor featurizer on the treatment side).
    Dimensions: A = 1 (binary), Z = 2, W = 2, X = 20. The bilinear bridge phi(A, Z, X) combines the
    (A, X) "anchor" features with the treatment-proxy features of Z; the first stage regresses the
    treatment-proxy features on concat(A, W, X). So:
        first stage        input = concat(A, W, X) = 1 + 2 + 20 = 23
        second-stage (ax)  input = concat(A, X)    = 1 + 20     = 21
        treatment-proxy    input = Z               = 2
    Width kept modest (64) -- RHC is small (N~4000); larger treatment nets overfit (abortion lesson).
    The width / output-dim / dropout knobs are exposed so a follow-up study can probe whether a more
    expressive bridge extracts more of Cui et al.'s A x Z / A x X proximal correction (defaults = the
    locked config; a sweep overrides them via run_once's tm_builder_kwargs).
    """
    FIRST_STAGE_OUTPUT_DIM = first_stage_out
    Z_OUTPUT_DIM = z_out
    SECOND_STAGE_AX_OUTPUT_DIM = ax_out
    INTERNAL_WIDTH = width
    DROPOUT_RATE = dropout

    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "linear"):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, INTERNAL_WIDTH),
                nn.LayerNorm(INTERNAL_WIDTH),
                nn.GELU(),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH, output_dim),
            )
            self.output_dim = output_dim
            self.final_activation = final_activation
            self.tanh = nn.Tanh()
            self.gelu = nn.GELU()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            if self.final_activation == "tanh":
                return self.tanh(out)
            elif self.final_activation == "gelu":
                return self.gelu(out)
            return out

    class FirstStageFeaturizer(FeaturizerBase):           # concat(A, W, X) = 1 + 2 + 20 = 23
        def __init__(self, input_dim: int = 23, final_activation="gelu"):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM)

    class SecondStageFeaturizer(FeaturizerBase):          # concat(A, X) = 1 + 20 = 21
        def __init__(self, input_dim: int = 21, final_activation="gelu"):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_AX_OUTPUT_DIM)

    class TreatmentProxyFeaturizer(FeaturizerBase):        # Z = 2
        def __init__(self, input_dim: int = 2, final_activation="gelu"):
            super().__init__(input_dim=input_dim, output_dim=Z_OUTPUT_DIM)

    first_stage_featurizer = FirstStageFeaturizer(input_dim=3 + x_dim).to(device)   # A + W + X = 1 + 2 + x_dim
    second_stage_ax_featurizer = SecondStageFeaturizer(input_dim=1 + x_dim).to(device)   # A + X
    treatment_proxy_featurizer = TreatmentProxyFeaturizer().to(device)

    return first_stage_featurizer, second_stage_ax_featurizer, treatment_proxy_featurizer


def build_nets_for_treatment_pcl_net_dsprite_experiment( device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Builds efficient featurizer networks for the low-dimensional synthetic benchmark.
    Significantly reduces final output dimensions (from 64/128 to 16/32) to speed up
    closed-form matrix operations (reducing the final layer size from 8192 to 512).
    """
    DIM_IM = 4096
    FIRST_STAGE_OUTPUT_DIM = 128
    Z_OUTPUT_DIM = 8
    SECOND_STAGE_AX_OUTPUT_DIM = 32
    INTERNAL_WIDTH = 1024 # Keep internal layers reasonably wide
    DROPOUT_RATE = 0.05


    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "linear"):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, INTERNAL_WIDTH),
                nn.LayerNorm(INTERNAL_WIDTH),
                nn.ReLU(),
                nn.BatchNorm1d(INTERNAL_WIDTH),
                nn.Dropout(DROPOUT_RATE),

                nn.Linear(INTERNAL_WIDTH, INTERNAL_WIDTH // 2),
                nn.LayerNorm(INTERNAL_WIDTH // 2),
                nn.ReLU(),
                nn.BatchNorm1d(INTERNAL_WIDTH // 2),
                nn.Dropout(DROPOUT_RATE),

                nn.Linear(INTERNAL_WIDTH // 2, INTERNAL_WIDTH // 4),
                nn.LayerNorm(INTERNAL_WIDTH // 4),
                nn.ReLU(),
                nn.BatchNorm1d(INTERNAL_WIDTH // 4),
                nn.Dropout(DROPOUT_RATE),

                nn.Linear(INTERNAL_WIDTH // 4, output_dim),
                nn.LayerNorm(output_dim),
            )
            self.output_dim = output_dim
            self.final_activation = final_activation
            self.tanh = nn.Tanh()
            self.gelu = nn.ReLU()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            if self.final_activation == "tanh":
                return self.tanh(out)
            elif self.final_activation == "gelu":
                return self.gelu(out)
            return out

    # --- Treatment Featurizers (Input: 1D) ---
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 2 * DIM_IM):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM, final_activation="gelu")

    # --- Outcome Proxy Featurizer (Input: 2D) ---
    class SecondStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = DIM_IM):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_AX_OUTPUT_DIM, final_activation="gelu")

    # --- Treatment Proxy Featurizer (Input: 3D) ---
    class TreatmentProxyFeaturizer(nn.Module):
        def __init__(self, input_dim: int = 3, output_dim: int = Z_OUTPUT_DIM, final_activation: str = "gelu"):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 8),
                nn.LayerNorm(8),
                nn.ReLU(),
                nn.BatchNorm1d(8),
                nn.Dropout(DROPOUT_RATE),

                nn.Linear(8, 4),
                nn.LayerNorm(4),
                nn.ReLU(),
                nn.BatchNorm1d(4),
                nn.Dropout(DROPOUT_RATE),

                nn.Linear(4, output_dim),
                nn.LayerNorm(output_dim),
            )
            self.output_dim = output_dim
            self.final_activation = final_activation
            self.tanh = nn.Tanh()
            self.gelu = nn.ReLU()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            if self.final_activation == "tanh":
                return self.tanh(out)
            elif self.final_activation == "gelu":
                return self.gelu(out)
            return out

    def _small_init(m):
        if isinstance(m, nn.Linear):
            # We use a very small scale (0.01) to squash the initial output variance
            nn.init.kaiming_normal_(m.weight, a=0.05) # Use GELU/ReLU compatible init
            m.weight.data *= 0.05
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 0.05) # Scale down LayerNorm gains too

    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    second_stage_ax_featurizer = SecondStageFeaturizer().to(device)
    treatment_proxy_featurizer = TreatmentProxyFeaturizer().to(device)
    # treatment_proxy_featurizer.apply(_small_init)

    return first_stage_featurizer, second_stage_ax_featurizer, treatment_proxy_featurizer


def build_nets_for_treatment_pcl_net_dsprite_experiment_with_compressed_images(latent_dim: int = 8, device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Builds efficient featurizer networks for the low-dimensional synthetic benchmark.
    Significantly reduces final output dimensions (from 64/128 to 16/32) to speed up
    closed-form matrix operations (reducing the final layer size from 8192 to 512).
    """
    DIM_IM = latent_dim
    FIRST_STAGE_OUTPUT_DIM = 32
    Z_OUTPUT_DIM = 16
    SECOND_STAGE_AX_OUTPUT_DIM = 16
    INTERNAL_WIDTH = 256 # Keep internal layers reasonably wide
    DROPOUT_RATE = 0.01


    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "linear"):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, INTERNAL_WIDTH),
                nn.LayerNorm(INTERNAL_WIDTH),
                nn.GELU(),
                nn.BatchNorm1d(INTERNAL_WIDTH),
                nn.Dropout(DROPOUT_RATE),

                nn.Linear(INTERNAL_WIDTH, INTERNAL_WIDTH // 2),
                nn.LayerNorm(INTERNAL_WIDTH // 2),
                nn.GELU(),
                nn.BatchNorm1d(INTERNAL_WIDTH // 2),
                nn.Dropout(DROPOUT_RATE),

                # nn.Linear(INTERNAL_WIDTH // 2, INTERNAL_WIDTH // 4),
                # nn.LayerNorm(INTERNAL_WIDTH // 4),
                # nn.GELU,
                # nn.BatchNorm1d(INTERNAL_WIDTH // 4),
                # nn.Dropout(DROPOUT_RATE),

                nn.Linear(INTERNAL_WIDTH // 2, output_dim),
                nn.LayerNorm(output_dim),
            )
            self.output_dim = output_dim
            self.final_activation = final_activation
            self.tanh = nn.Tanh()
            self.gelu = nn.GELU()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            if self.final_activation == "tanh":
                return self.tanh(out)
            elif self.final_activation == "gelu":
                return self.gelu(out)
            return out

    # --- Treatment Featurizers (Input: 1D) ---
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 2 * DIM_IM):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM, final_activation="gelu")

    # --- Outcome Proxy Featurizer (Input: 2D) ---
    class SecondStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = DIM_IM):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_AX_OUTPUT_DIM, final_activation="gelu")

    # --- Treatment Proxy Featurizer (Input: 2D) ---
    class TreatmentProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 3):
            super().__init__(input_dim=input_dim, output_dim=Z_OUTPUT_DIM, final_activation="gelu")


    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    second_stage_ax_featurizer = SecondStageFeaturizer().to(device)
    treatment_proxy_featurizer = TreatmentProxyFeaturizer().to(device)

    return first_stage_featurizer, second_stage_ax_featurizer, treatment_proxy_featurizer


def build_nets_for_treatment_pcl_net_demand_experiment(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Builds efficient featurizer networks for the low-dimensional synthetic benchmark.
    Significantly reduces final output dimensions (from 64/128 to 16/32) to speed up
    closed-form matrix operations (reducing the final layer size from 8192 to 512).
    """
    FIRST_STAGE_OUTPUT_DIM = 16
    Z_OUTPUT_DIM = 4
    SECOND_STAGE_AX_OUTPUT_DIM = 4
    INTERNAL_WIDTH = 32 # Keep internal layers reasonably wide
    DROPOUT_RATE = 0.01


    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "linear"):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, INTERNAL_WIDTH),
                nn.LayerNorm(INTERNAL_WIDTH),
                nn.GELU(),
                nn.BatchNorm1d(INTERNAL_WIDTH),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH, INTERNAL_WIDTH // 2),
                nn.LayerNorm(INTERNAL_WIDTH // 2),
                nn.GELU(),
                nn.BatchNorm1d(INTERNAL_WIDTH // 2),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH // 2, output_dim),
                nn.LayerNorm(output_dim),
            )
            self.output_dim = output_dim
            self.final_activation = final_activation
            self.tanh = nn.Tanh()
            self.relu = nn.GELU()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            if self.final_activation == "tanh":
                return self.tanh(out)
            elif self.final_activation == "relu":
                return self.relu(out)
            return out

    # --- Treatment Featurizers (Input: 1D) ---
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 2):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM, final_activation="relu")

    # --- Outcome Proxy Featurizer (Input: 2D) ---
    class SecondStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_AX_OUTPUT_DIM, final_activation="relu")

    # --- Treatment Proxy Featurizer (Input: 2D) ---
    class TreatmentProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 2):
            super().__init__(input_dim=input_dim, output_dim=Z_OUTPUT_DIM, final_activation="relu")


    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    second_stage_ax_featurizer = SecondStageFeaturizer().to(device)
    treatment_proxy_featurizer = TreatmentProxyFeaturizer().to(device)

    return first_stage_featurizer, second_stage_ax_featurizer, treatment_proxy_featurizer


def build_alternative_treatment_nets(
    device: str = "cuda", z_dim: int = 2, w_dim: int = 1, a_dim: int = 1,
    first_stage_a_out: int = 16, w_out: int = 8, z_out: int = 8, second_stage_a_out: int = 8,
    width: int = 32, dropout: float = 0.01,
    x_dim: Optional[int] = None, x_first_out: int = 4, x_second_out: int = 4,
    joint_backdoor_first_stage: bool = False, wx_out: Optional[int] = None):
    """Generic featurizer builder for the DENSITY-RATIO-FREE treatment bridge
    (AlternativeTreatmentBridgePCLNET, the neural mean-embedding analogue of the kernel KAP method).

    Unlike the density-ratio bridge (single joint featurizer over concat(A,W) + a treatment-only
    featurizer), the density-ratio-free bridge needs FOUR featurizers, because its first stage regresses
    phi_Z(Z) onto the tensor product outer_prod(phi_A_first(A), phi_W(W)):
      - treatment_featurizer_first_stage : A -> R^{first_stage_a_out}
      - outcome_proxy_featurizer         : W -> R^{w_out}   (first-stage feature = phi_A_first (x) phi_W)
      - treatment_proxy_featurizer       : Z -> R^{z_out}
      - treatment_featurizer_second_stage: A -> R^{second_stage_a_out}  (bridge feature = phi_Z (x) phi_A2)

    Returned in AlternativeTreatmentBridgePCLNET.__init__ order:
        (treatment_featurizer_first_stage, treatment_featurizer_second_stage,
         treatment_proxy_featurizer, outcome_proxy_featurizer).

    BACKDOOR X (high-dim experiment): if `x_dim` is given, TWO extra featurizers are appended so the bridge
    uses the TENSOR-PRODUCT structure that keeps the KAP cross term O(N) (A_i and (W,X)_j in separate slots):
      - backdoor_featurizer_first_stage  : X -> R^{x_first_out}   (first-stage feat = phi_A1 (x) phi_W (x) phi_X1)
      - backdoor_featurizer_second_stage : X -> R^{x_second_out}  (bridge feat = phi_Z (x) phi_A2 (x) phi_X2)
    and the return is the 6-tuple (t_first, t_second, t_proxy, out_proxy, bd_first, bd_second).

    JOINT BACKDOOR FIRST STAGE (`joint_backdoor_first_stage=True`, requires x_dim): the first stage instead
    uses ONE net over concat(W, X) -- the density-ratio bridge's style -- so the slots become
      - outcome_proxy_featurizer         : concat(W,X) -> R^{wx_out or w_out}  (feat = phi_A1 (x) phi_WX)
      - backdoor_featurizer_first_stage  : None        (X already lives inside phi_WX; x_first_out unused)
      - backdoor_featurizer_second_stage : X -> R^{x_second_out}   (bridge is unchanged, still 3-way)
    The return is still a 6-tuple, with None in the bd_first slot; AlternativeTreatmentBridgePCLNET infers
    joint mode from that (bd_first None, bd_second present). This is still EXACTLY O(N): W and X are both
    on the index-j side of the KAP mismatch, so only the j-side moment's rank drops (3 -> 2). Joining A
    into that net would NOT be safe -- A is the index-i side and the double sum would stop factorizing.
    """
    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "relu"):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.BatchNorm1d(width),
                nn.Dropout(dropout),
                nn.Linear(width, width // 2),
                nn.LayerNorm(width // 2),
                nn.GELU(),
                nn.BatchNorm1d(width // 2),
                nn.Dropout(dropout),
                nn.Linear(width // 2, output_dim),
                nn.LayerNorm(output_dim),
            )
            self.output_dim = output_dim
            self.final_activation = final_activation
            self.tanh = nn.Tanh()
            self.relu = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            if self.final_activation == "tanh":
                return self.tanh(out)
            elif self.final_activation == "relu":
                return self.relu(out)
            return out

    treatment_featurizer_first_stage = FeaturizerBase(input_dim=a_dim, output_dim=first_stage_a_out).to(device)
    # JOINT backdoor first stage: one net over concat(W, X) replaces the phi_W (x) phi_X1 slots. W and X
    # stay on the SAME side of the KAP mismatch (both index j), so the O(N) cross reduction is preserved
    # exactly -- only the j-side moment's rank drops from 3 to 2. Joining A in as well would NOT be safe.
    # NOTE the joint net's OUTPUT dim defaults to w_out, which also shrinks the first-stage tensor product
    # from D_A1*D_W*D_X1 to D_A1*D_WX. Pass wx_out to restore (or choose) that capacity independently.
    use_joint_backdoor = joint_backdoor_first_stage and (x_dim is not None)
    first_stage_w_input_dim = w_dim + x_dim if use_joint_backdoor else w_dim
    first_stage_w_output_dim = (wx_out if (use_joint_backdoor and wx_out is not None) else w_out)
    outcome_proxy_featurizer = FeaturizerBase(input_dim=first_stage_w_input_dim,
                                              output_dim=first_stage_w_output_dim).to(device)
    treatment_proxy_featurizer = FeaturizerBase(input_dim=z_dim, output_dim=z_out).to(device)
    treatment_featurizer_second_stage = FeaturizerBase(input_dim=a_dim, output_dim=second_stage_a_out).to(device)
    if x_dim is not None:
        if joint_backdoor_first_stage:
            # No separate phi_X1 -- AlternativeTreatmentBridgePCLNET infers joint mode from
            # (backdoor_featurizer_first_stage is None) and (backdoor_featurizer_second_stage is not None).
            backdoor_featurizer_second_stage = FeaturizerBase(input_dim=x_dim, output_dim=x_second_out).to(device)
            return (treatment_featurizer_first_stage, treatment_featurizer_second_stage,
                    treatment_proxy_featurizer, outcome_proxy_featurizer,
                    None, backdoor_featurizer_second_stage)
        backdoor_featurizer_first_stage = FeaturizerBase(input_dim=x_dim, output_dim=x_first_out).to(device)
        backdoor_featurizer_second_stage = FeaturizerBase(input_dim=x_dim, output_dim=x_second_out).to(device)
        return (treatment_featurizer_first_stage, treatment_featurizer_second_stage,
                treatment_proxy_featurizer, outcome_proxy_featurizer,
                backdoor_featurizer_first_stage, backdoor_featurizer_second_stage)
    return (treatment_featurizer_first_stage, treatment_featurizer_second_stage,
            treatment_proxy_featurizer, outcome_proxy_featurizer)


def build_nets_for_alternative_treatment_pcl_net_demand_experiment(device: str = "cuda"):
    """Density-ratio-free treatment-bridge featurizers for the demand experiment (Z 2-D, W 1-D, A 1-D)."""
    return build_alternative_treatment_nets(device, z_dim=2, w_dim=1, a_dim=1)


def build_nets_for_alternative_treatment_pcl_net_synthetic_low_dim_experiment(device: str = "cuda"):
    """Density-ratio-free treatment-bridge featurizers for the synthetic low-dim experiment (Z 2-D, W 2-D, A 1-D)."""
    return build_alternative_treatment_nets(device, z_dim=2, w_dim=2, a_dim=1)


def build_nets_for_alternative_treatment_pcl_net_octmnist_experiment(
        device: str = "cuda", first_stage_a_out: int = 16, w_out: int = 16,
        z_out: int = 16, second_stage_a_out: int = 8, width: int = 64, dropout: float = 0.01):
    """Density-ratio-free treatment-bridge featurizers for the proximal OCTMNIST experiment
    (BINARY A, 6-D proxies Z and W, STRUCTURED 28x28 image outcome Y).

    As in ColorMNIST, Y never enters the bridge: varphi(A, Z) and its first stage
    phi_Z(Z) ~ phi_A1(A) (x) phi_W(W) touch only A, Z and W. The 784-dimensional outcome shows up
    solely in the third stage's output_dim. Proxies here are 6-D (not scalar), so the Z and W
    featurizers are wider than the ColorMNIST ones.

    Binary A means phi_A2(A) takes two values, so the E[varphi|A=a]=1 normalization must be done PER
    GROUP rather than by a polynomial fit in A.
    """
    return build_alternative_treatment_nets(
        device, z_dim=6, w_dim=6, a_dim=1,
        first_stage_a_out=first_stage_a_out, w_out=w_out, z_out=z_out,
        second_stage_a_out=second_stage_a_out, width=width, dropout=dropout)


def build_nets_for_alternative_treatment_pcl_net_colormnist_experiment(
        device: str = "cuda", first_stage_a_out: int = 16, w_out: int = 8,
        z_out: int = 8, second_stage_a_out: int = 8, width: int = 32, dropout: float = 0.01):
    """Density-ratio-free treatment-bridge featurizers for the proximal ColorMNIST experiment
    (BINARY A in {0,1} selecting the digit class, scalar proxies Z and W, STRUCTURED image outcome Y).

    Note what is NOT here: the outcome. The treatment bridge is varphi(A, Z) and its first stage
    regresses phi_Z(Z) on phi_A1(A) (x) phi_W(W) -- Y never enters either stage. So a 3*28*28 image
    outcome needs no special bridge architecture at all; the structure only shows up downstream, where
    the third stage regresses the 2352-dimensional composite target Y * varphi on A. That is why these
    featurizers are small: every bridge input here is 1-D.

    Because A is binary, phi_A2(A) takes exactly two values, so the bridge is effectively two separate
    linear functions of phi_Z(Z) -- and the E[varphi|A=a]=1 normalization must be done PER GROUP rather
    than by the polynomial-in-A fit used for continuous treatments.
    """
    return build_alternative_treatment_nets(
        device, z_dim=1, w_dim=1, a_dim=1,
        first_stage_a_out=first_stage_a_out, w_out=w_out, z_out=z_out,
        second_stage_a_out=second_stage_a_out, width=width, dropout=dropout)


def build_nets_for_alternative_treatment_pcl_net_synthetic_cate_experiment(
        device: str = "cuda", joint_backdoor_first_stage: bool = False):
    """Density-ratio-free treatment-bridge featurizers for the synthetic CATE experiment
    (Z 3-D, W 3-D, A 1-D binary, effect modifier V 1-D).

    Per chapter 21 the CATE needs NO CATE-specific bridge: we fit the ordinary density-ratio-free ATE
    bridge over ALL covariates X = (S, V) and self-normalize it in the third stage. This benchmark has
    no S, so X = V and V enters here as the 1-D BACKDOOR -- which is what makes the bridge satisfy
    eq. (21.2) conditionally on V, hence D_0(a,v) = p_A(a)/p_{A|V}(a|v). Returns the usual SIX
    featurizers (phi_A1, phi_A2, phi_Z, phi_W, phi_V1, phi_V2).
    """
    return build_alternative_treatment_nets(
        device, z_dim=3, w_dim=3, a_dim=1, x_dim=1,
        first_stage_a_out=8, w_out=8, z_out=8, second_stage_a_out=8,
        x_first_out=4, x_second_out=4, width=32, dropout=0.01,
        joint_backdoor_first_stage=joint_backdoor_first_stage)


def build_nets_for_alternative_treatment_pcl_net_abortion_and_crime_experiment(
        device: str = "cuda", width: int = 32, first_stage_a_out: int = 16,
        z_out: int = 8, w_out: int = 8, second_stage_a_out: int = 8, dropout: float = 0.01):
    """Density-ratio-free treatment-bridge featurizers for the Abortion & Crime experiment (Z 1-D, W 3-D,
    A 1-D). The capacity kwargs default to the historical values (numerically identical behaviour when omitted);
    they exist so the tuning harness can simplify or enrich the bridge without touching this module."""
    return build_alternative_treatment_nets(
        device, z_dim=1, w_dim=3, a_dim=1, width=width, first_stage_a_out=first_stage_a_out,
        z_out=z_out, w_out=w_out, second_stage_a_out=second_stage_a_out, dropout=dropout)


def build_nets_for_alternative_treatment_pcl_net_deaner_experiment(device: str = "cuda"):
    """Density-ratio-free treatment-bridge featurizers for the Deaner grade-retention benchmark
    (Z 4-D, W 3-D, A 1-D) -- the density-ratio-free counterpart of
    build_nets_for_treatment_pcl_net_deaner_experiment.

    Same shape as the abortion wrapper; only the proxy dimensions differ (Deaner's treatment proxy Z is
    4-D and its outcome proxy W is 3-D, the reverse of abortion's weak-Z / rich-W asymmetry).
    """
    return build_alternative_treatment_nets(device, z_dim=4, w_dim=3, a_dim=1)


def build_nets_for_alternative_treatment_pcl_net_dsprite_experiment(device: str = "cuda", z_dim: int = 3):
    """Density-ratio-free treatment-bridge featurizers for the dSprite experiment: A and W are 4096-D
    (flattened 64x64 images), Z is z_dim-D (3 for the inherited V2 benchmark; 2 for the repaired V3 DGP,
    whose Z is a 2-D noisy view of the latent confounder). No backdoor X, so
    the O(N) cross-term reduction is preserved. The A/W featurizers are wide MLPs; the small Z featurizer
    shares the same width (harmlessly over-parameterized).
    """
    return build_alternative_treatment_nets(
        device, z_dim=z_dim, w_dim=4096, a_dim=4096,
        first_stage_a_out=16, w_out=8, z_out=8, second_stage_a_out=8,
        width=512, dropout=0.05)


def build_nets_for_alternative_treatment_pcl_net_dsprite_experiment_dfpv(device: str = "cuda", z_dim: int = 3):
    """DFPV-STYLE featurizers for the density-ratio-free dSprite TREATMENT bridge.

    Mirrors the official DFPV OutcomeNet nets (Xu et al. 2021, nn_structure_for_dsprite.py), which the
    ablation showed is what makes their bridge work on 4096-D images:
    spectral-norm Linear 4096-1024-512(BatchNorm)-128-32 with plain ReLU -- NO LayerNorm (LayerNorm
    nets scored 23.7 vs 15.7 in their harness), no dropout, no GELU. The Z net is their 3-128-64-32.
    Returns the 6-tuple (t_first, t_second, t_proxy, out_proxy, None, None) the ensemble trainer wants.
    """
    from torch.nn.utils import spectral_norm

    def img_net():
        return nn.Sequential(spectral_norm(nn.Linear(64 * 64, 1024)), nn.ReLU(),
                             spectral_norm(nn.Linear(1024, 512)), nn.ReLU(), nn.BatchNorm1d(512),
                             spectral_norm(nn.Linear(512, 128)), nn.ReLU(),
                             spectral_norm(nn.Linear(128, 32)), nn.ReLU())

    class _Wrap(nn.Module):
        def __init__(self, net, out):
            super().__init__(); self.net = net; self.output_dim = out
        def forward(self, x):
            return self.net(x)

    z_net = nn.Sequential(nn.Linear(z_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                          nn.Linear(64, 32), nn.ReLU())
    t_first = _Wrap(img_net(), 32).to(device)
    t_second = _Wrap(img_net(), 32).to(device)
    out_proxy = _Wrap(img_net(), 32).to(device)
    t_proxy = _Wrap(z_net, 32).to(device)
    return (t_first, t_second, t_proxy, out_proxy, None, None)


def build_nets_for_alternative_treatment_pcl_net_synthetic_high_dim_experiment(
        device: str = "cuda", joint_backdoor_first_stage: bool = False,
        wx_out: Optional[int] = None):
    """Density-ratio-free treatment-bridge featurizers for the synthetic HIGH-DIM experiment with a
    100-D backdoor X (A 1-D, Z 10-D, W 10-D, X 100-D). Returns SIX featurizers (adds phi_X1, phi_X2) so the
    bridge uses the tensor-product structure phi_Z (x) phi_A2 (x) phi_X2 that keeps the KAP mismatched-cross
    term EXACTLY O(N) (A_i and (W,X)_j occupy separate tensor slots -- see AlternativeTreatmentBridgePCLNET
    ._cross_rhs_backdoor). Feature dims are kept modest so the tensor products D_Z*D_A2*D_X2 and
    D_A1*D_W*D_X1 stay small (256 each): first stage 8*8*4, bridge 8*8*4.

    With joint_backdoor_first_stage=True the first stage instead uses ONE net over concat(W, X) (the
    density-ratio bridge's style), returning phi_X1 = None. This is still EXACTLY O(N): W and X are both
    on the index-j side of the mismatch, so only the j-side moment's rank changes. It also shrinks the
    first-stage tensor product from D_A1*D_W*D_X1 = 8*8*4 = 256 to D_A1*D_WX = 8*8 = 64; pass wx_out=32
    to keep the design at 256 while still using the joint net.
    """
    return build_alternative_treatment_nets(
        device, z_dim=10, w_dim=10, a_dim=1, x_dim=100,
        first_stage_a_out=8, w_out=8, z_out=8, second_stage_a_out=8,
        x_first_out=4, x_second_out=4, width=64, dropout=0.05,
        joint_backdoor_first_stage=joint_backdoor_first_stage, wx_out=wx_out)


def build_nets_for_treatment_pcl_net_synthetic_cate_experiment(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Builds efficient featurizer networks for the low-dimensional synthetic benchmark.
    Significantly reduces final output dimensions (from 64/128 to 16/32) to speed up
    closed-form matrix operations (reducing the final layer size from 8192 to 512).
    """
    FIRST_STAGE_OUTPUT_DIM = 64
    Z_OUTPUT_DIM = 16
    SECOND_STAGE_AX_OUTPUT_DIM = 4
    INTERNAL_WIDTH = 256 # Keep internal layers reasonably wide
    DROPOUT_RATE = 0.1


    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, DROPOUT_RATE: float = 0.0):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, INTERNAL_WIDTH),
                nn.LayerNorm(INTERNAL_WIDTH),
                nn.GELU(),
                nn.BatchNorm1d(INTERNAL_WIDTH),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH, INTERNAL_WIDTH * 2),
                nn.LayerNorm(INTERNAL_WIDTH * 2),
                nn.GELU(),
                nn.BatchNorm1d(INTERNAL_WIDTH * 2),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH * 2, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                # nn.BatchNorm1d(output_dim)
            )
            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    # --- Treatment Featurizers (Input: 1D) ---
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 5):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM)

    # --- Outcome Proxy Featurizer (Input: 2D) ---
    class SecondStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 2):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_AX_OUTPUT_DIM)

    # --- Treatment Proxy Featurizer (Input: 2D) ---
    class TreatmentProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 3, DROPOUT_RATE: float = DROPOUT_RATE):
            super().__init__(input_dim=input_dim, output_dim=Z_OUTPUT_DIM)

    def _small_init(m):
        if isinstance(m, nn.Linear):
            # We use a very small scale (0.01) to squash the initial output variance
            nn.init.kaiming_normal_(m.weight, a=0.05) # Use GELU/ReLU compatible init
            m.weight.data *= 0.05
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 0.05) # Scale down LayerNorm gains too
    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    second_stage_ax_featurizer = SecondStageFeaturizer().to(device)
    treatment_proxy_featurizer = TreatmentProxyFeaturizer().to(device)
    treatment_proxy_featurizer.apply(_small_init)

    return first_stage_featurizer, second_stage_ax_featurizer, treatment_proxy_featurizer

def build_alternative_treatment_att_nets(
    device: str = "cuda", z_dim: int = 2, w_dim: int = 2, a_dim: int = 1,
    first_stage_a_out: int = 16, w_out: int = 8, z_out: int = 8, anchor_out: int = 8,
    width: int = 32, dropout: float = 0.01,
    x_dim: Optional[int] = None, x_first_out: int = 4):
    """Generic featurizer builder for the DENSITY-RATIO-FREE JOINT-ANCHOR ATT bridge
    (AlternativeTreatmentBridgePCLNETATT).

    Four featurizers, in AlternativeTreatmentBridgePCLNETATT.__init__ order:
      - treatment_featurizer_first_stage : A -> R^{first_stage_a_out}
      - anchor_treatment_featurizer      : concat(A, A'[, X]) -> R^{anchor_out}   <- psi_{AA'X}
      - treatment_proxy_featurizer       : Z -> R^{z_out}                         <- psi_Z
      - outcome_proxy_featurizer         : W -> R^{w_out}

    The difference from build_alternative_treatment_nets is the SECOND-stage treatment featurizer. There
    it is phi_A2(A), a function of the intervention level alone; here it is psi_{AA'X}(A, A', X), ONE
    network that also takes the ANCHOR treatment A' (and the covariates, which is why the ATT bridge has
    no separate second-stage backdoor featurizer). That single network is what lets one fit give the whole
    surface (a, a') -> f_ATT(a, a') instead of one bridge per anchor.

    The bridge head is therefore (anchor_out * z_out)-dimensional, and the second-stage cost is the
    ordered-pair one: psi_{AA'X} is evaluated on M(M-1) pairs per stage-2 batch under the complete
    U-statistic objective. Keep `width` and `anchor_out` modest for that reason.

    BACKDOOR X: if `x_dim` is given, ONE extra featurizer is appended for the FIRST stage
    (phi_X1: X -> R^{x_first_out}, so the first-stage design is phi_A1 (x) phi_W (x) phi_X1) and the
    anchor featurizer's input grows to 2 * a_dim + x_dim. The return is the 5-tuple
    (t_first, anchor, t_proxy, out_proxy, bd_first).
    """
    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "relu"):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.BatchNorm1d(width),
                nn.Dropout(dropout),
                nn.Linear(width, width // 2),
                nn.LayerNorm(width // 2),
                nn.GELU(),
                nn.BatchNorm1d(width // 2),
                nn.Dropout(dropout),
                nn.Linear(width // 2, output_dim),
                nn.LayerNorm(output_dim),
            )
            self.output_dim = output_dim
            self.final_activation = final_activation
            self.tanh = nn.Tanh()
            self.relu = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            if self.final_activation == "tanh":
                return self.tanh(out)
            elif self.final_activation == "relu":
                return self.relu(out)
            return out

    anchor_input_dim = 2 * a_dim + (x_dim if x_dim is not None else 0)

    treatment_featurizer_first_stage = FeaturizerBase(input_dim=a_dim, output_dim=first_stage_a_out).to(device)
    anchor_treatment_featurizer = FeaturizerBase(input_dim=anchor_input_dim, output_dim=anchor_out).to(device)
    treatment_proxy_featurizer = FeaturizerBase(input_dim=z_dim, output_dim=z_out).to(device)
    outcome_proxy_featurizer = FeaturizerBase(input_dim=w_dim, output_dim=w_out).to(device)

    if x_dim is not None:
        backdoor_featurizer_first_stage = FeaturizerBase(input_dim=x_dim, output_dim=x_first_out).to(device)
        return (treatment_featurizer_first_stage, anchor_treatment_featurizer,
                treatment_proxy_featurizer, outcome_proxy_featurizer, backdoor_featurizer_first_stage)
    return (treatment_featurizer_first_stage, anchor_treatment_featurizer,
            treatment_proxy_featurizer, outcome_proxy_featurizer)


def build_nets_for_alternative_treatment_att_pcl_net_synthetic_low_dim_experiment(device: str = "cuda"):
    """Density-ratio-free joint-anchor ATT featurizers for the synthetic low-dim ATT benchmark
    (Z 2-D, W 2-D, A 1-D, no backdoor X). The anchor featurizer therefore sees concat(A, A') = 2-D."""
    return build_alternative_treatment_att_nets(device, z_dim=2, w_dim=2, a_dim=1)


def build_alternative_treatment_att_tensor_nets(
        device: str = "cuda", z_dim: int = 2, w_dim: int = 2, a_dim: int = 1,
        first_stage_a_out: int = 16, w_out: int = 8, z_out: int = 8,
        source_out: int = 6, anchor_out: int = 6, width: int = 32, dropout: float = 0.01):
    """SEPARABLE anchor featurizers for AlternativeTreatmentBridgePCLNETATT(anchor_mode="tensor").

    Same four slots as build_alternative_treatment_att_nets, except that the single joint network
    psi_{AA'}(concat(a, a')) is replaced by an nn.ModuleList([psi_A, psi_A']) whose OUTER PRODUCT is
    the anchor feature:

        psi_{AA'}(a, a') = psi_A(a) (x) psi_A'(a')     ->  R^{source_out * anchor_out}

    WHY. A separable anchor feature makes the ordered-pair objective and normal equations factorise
    into O(M) einsum contractions (see AlternativeTreatmentBridgePCLNETATT._einsum_pair_terms), so the
    COMPLETE U-statistic of eq. 24 -- 36M ordered pairs at n = 20000 under the concat model -- costs
    the same as a single pass over the rows. The price is the model class: a rank-truncated family of
    bivariate functions rather than an arbitrary MLP on concat(a, a'), and a head of
    source_out * anchor_out * z_out instead of anchor_out * z_out. Keep source_out and anchor_out
    modest for that reason -- 6 x 6 x 8 = 288 is already four times the concat head.

    Returns (t_first, nn.ModuleList([psi_A, psi_A']), t_proxy, out_proxy).
    """
    t_first, psi_source, t_proxy, out_proxy = build_alternative_treatment_att_nets(
        device, z_dim=z_dim, w_dim=w_dim, a_dim=a_dim, first_stage_a_out=first_stage_a_out,
        w_out=w_out, z_out=z_out, anchor_out=source_out, width=width, dropout=dropout)
    # psi_source above was built with input concat(a, a'); rebuild both halves on a_dim inputs.
    _, psi_a, _, _ = build_alternative_treatment_att_nets(
        device, z_dim=z_dim, w_dim=w_dim, a_dim=a_dim, first_stage_a_out=first_stage_a_out,
        w_out=w_out, z_out=z_out, anchor_out=source_out, width=width, dropout=dropout)
    _, psi_a_prime, _, _ = build_alternative_treatment_att_nets(
        device, z_dim=z_dim, w_dim=w_dim, a_dim=a_dim, first_stage_a_out=first_stage_a_out,
        w_out=w_out, z_out=z_out, anchor_out=anchor_out, width=width, dropout=dropout)
    # the anchor featurizers take ONE treatment each, so rebuild their first layer for a_dim inputs
    psi_a = _rebuild_first_layer(psi_a, a_dim, device)
    psi_a_prime = _rebuild_first_layer(psi_a_prime, a_dim, device)
    return t_first, nn.ModuleList([psi_a, psi_a_prime]).to(device), t_proxy, out_proxy


def _rebuild_first_layer(featurizer, input_dim: int, device: str):
    """Swap a featurizer's first Linear so it consumes `input_dim` inputs (a, not concat(a, a'))."""
    first = featurizer.net[0]
    if first.in_features != input_dim:
        featurizer.net[0] = nn.Linear(input_dim, first.out_features).to(device)
    return featurizer.to(device)


def build_nets_for_alternative_treatment_att_pcl_net_synthetic_low_dim_tensor_experiment(
        device: str = "cuda", source_out: int = 6, anchor_out: int = 6, z_out: int = 8,
        first_stage_a_out: int = 16, w_out: int = 8, width: int = 32, dropout: float = 0.01):
    """Separable (einsum-factorisable) ATT featurizers for the synthetic low-dim ATT benchmark."""
    return build_alternative_treatment_att_tensor_nets(
        device, z_dim=2, w_dim=2, a_dim=1, first_stage_a_out=first_stage_a_out, w_out=w_out,
        z_out=z_out, source_out=source_out, anchor_out=anchor_out, width=width, dropout=dropout)


def build_nets_for_alternative_treatment_pcl_net_rhc_experiment(
        device: str = "cuda", x_dim: int = 20, joint_backdoor_first_stage: bool = True,
        wx_out: int = 16):
    """Density-ratio-free treatment-bridge featurizers for the Right Heart Catheterization benchmark.

    Variable roles follow Zhang et al. (2025) exactly -- BINARY A = ``swang1``, Z = (``pafi1``,
    ``paco21``), W = (``ph1``, ``hema1``), backdoor X = the 20 significant covariates of their
    Appendix F (``x_dim=53`` for the all-features variant).

    Two RHC-specific choices:
      * ``second_stage_a_out`` is small (4). A is BINARY, so phi_A2(A) only ever takes two values and
        extra width there buys nothing -- it only inflates the bridge dimension z_out*a2_out*x2_out.
      * the first stage defaults to the JOINT phi_WX(concat(W, X)) variant. With W 2-D and X 20-D the
        tensor-product design would be D_A1 * D_W * D_X1 = 512 columns against ~2400 stage-1 rows;
        the joint one is 8 * 16 = 128. Both keep the KAP cross term exactly O(N) (W and X sit on the
        same side of the mismatch); this is purely about conditioning on a small real dataset.

    Returns the 6-tuple (t_first, t_second, t_proxy, out_proxy, bd_first, bd_second); in joint mode
    ``bd_first`` is None, which is how AlternativeTreatmentBridgePCLNET infers that mode.
    """
    return build_alternative_treatment_nets(
        device, z_dim=2, w_dim=2, a_dim=1, x_dim=x_dim,
        first_stage_a_out=8, w_out=8, z_out=8, second_stage_a_out=4,
        x_first_out=4, x_second_out=4,
        joint_backdoor_first_stage=joint_backdoor_first_stage, wx_out=wx_out,
        width=64, dropout=0.01)


def build_nets_for_alternative_treatment_pcl_net_colormnist_cate_experiment(
        device: str = "cuda", v_dim: int = 5, joint_backdoor_first_stage: bool = False):
    """Density-ratio-free treatment-bridge featurizers for ColorMNIST-PCL-CATE (A 1-D binary,
    Z 1-D, W 1-D, effect modifier V one-hot over the digit subgroups).

    Per chapter 21 there is NO CATE-specific bridge: fit the ordinary density-ratio-free ATE bridge
    over all covariates X = V and SELF-NORMALIZE it, which is exactly the guide's eq. 22-24
    (phi^CATE = phi^ATE / D_0(a,v), D_0 = p_A(a)/p_a(v)). V therefore enters here as the BACKDOOR.

    Returns the usual SIX featurizers:
      (treatment_first, treatment_second, treatment_proxy, outcome_proxy, backdoor_first, backdoor_second)
    """
    return build_alternative_treatment_nets(
        device=device, z_dim=1, w_dim=1, a_dim=1, x_dim=v_dim,
        first_stage_a_out=8, w_out=8, z_out=8, second_stage_a_out=4,
        x_first_out=8, x_second_out=4,
        width=64, dropout=0.01, joint_backdoor_first_stage=joint_backdoor_first_stage)
