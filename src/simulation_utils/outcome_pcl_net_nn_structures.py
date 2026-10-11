import torch
import torch.nn as nn
from typing import Tuple

def build_nets_for_outcome_pcl_net_synthetic_low_dim_experiment(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Builds efficient featurizer networks for the low-dimensional synthetic benchmark.
    Significantly reduces final output dimensions (from 64/128 to 16/32) to speed up
    closed-form matrix operations (reducing the final layer size from 8192 to 512).
    """
    FIRST_STAGE_OUTPUT_DIM = 128
    W_OUTPUT_DIM = 16
    SECOND_STAGE_A_OUTPUT_DIM = 8
    INTERNAL_WIDTH = 128 # Keep internal layers reasonably wide
    DROPOUT_RATE = 0.05
    class FeaturizerBase(nn.Module):
        """Standard featurizer structure with GELU activation."""
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "relu"):
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
            )
            # self.outbatchnorm = nn.BatchNorm1d(output_dim)
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
        def __init__(self, input_dim: int = 3, final_activation = "gelu"):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM, final_activation=final_activation)

    # --- Outcome Proxy Featurizer (Input: 2D) ---
    class OutcomeProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 2, final_activation = "gelu"):
            super().__init__(input_dim=input_dim, output_dim=W_OUTPUT_DIM, final_activation=final_activation)

    # --- Treatment Proxy Featurizer (Input: 2D) ---
    class TreatmentFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1, final_activation = "gelu"):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_A_OUTPUT_DIM, final_activation=final_activation)

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
    outcome_proxy_featurizer = OutcomeProxyFeaturizer().to(device)
    treatment_featurizer = TreatmentFeaturizer().to(device)
    outcome_proxy_featurizer.apply(_small_init)
    return first_stage_featurizer, treatment_featurizer, outcome_proxy_featurizer

def build_nets_for_outcome_pcl_net_synthetic_high_dim_new_version_experimentV1(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:

    class GaussianNoise(nn.Module):
        def __init__(self, std=0.05):
            super().__init__()
            self.std = std
        def forward(self, x):
            if self.training and self.std > 0:
                return x + torch.randn_like(x) * self.std
            return x

    # --- DIMENSIONS ---
    # Reduced slightly to force compression
    FIRST_STAGE_OUTPUT_DIM = 256
    W_OUTPUT_DIM = 8
    SECOND_STAGE_A_OUTPUT_DIM = 8
    SECOND_STAGE_X_OUTPUT_DIM = 32

    INTERNAL_WIDTH = 256

    DROPOUT_RATE = 0.05

    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "linear"):
            super().__init__()
            self.net = nn.Sequential(
                # Stronger Input Noise for small N
                # GaussianNoise(std=0.0001),

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

                # nn.Linear(INTERNAL_WIDTH // 2, INTERNAL_WIDTH // 2),
                # nn.LayerNorm(INTERNAL_WIDTH // 2),
                # nn.ReLU,
                # nn.Dropout(DROPOUT_RATE),

                nn.Linear(INTERNAL_WIDTH // 2, output_dim),
                nn.LayerNorm(output_dim),
                # Activation logic handled below
            )
            self.outbatchnorm = nn.BatchNorm1d(output_dim)
            self.final_activation = final_activation
            self.tanh = nn.Tanh()
            self.gelu = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            if self.final_activation == "tanh":
                return (self.tanh(out) )
            elif self.final_activation == "gelu":
                return (self.gelu(out))
            return (out)

    class OutcomeProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 10):
            super().__init__(input_dim=input_dim, output_dim=W_OUTPUT_DIM, final_activation="linear")

    # Keep others flexible but regularized
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 111):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM, final_activation="gelu")

    class TreatmentFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_A_OUTPUT_DIM, final_activation="gelu")

    class BackdoorFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 100):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_X_OUTPUT_DIM, final_activation="gelu")

    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    treatment_featurizer = TreatmentFeaturizer().to(device)
    backdoor_featurizer = BackdoorFeaturizer().to(device)
    outcome_proxy_featurizer = OutcomeProxyFeaturizer().to(device)

    return first_stage_featurizer, treatment_featurizer, backdoor_featurizer, outcome_proxy_featurizer


def build_nets_for_outcome_pcl_net_synthetic_high_dim_new_version_experiment(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:

    class GaussianNoise(nn.Module):
        def __init__(self, std=0.05):
            super().__init__()
            self.std = std
        def forward(self, x):
            if self.training and self.std > 0:
                return x + torch.randn_like(x) * self.std
            return x

    # --- DIMENSIONS ---
    # Reduced slightly to force compression
    FIRST_STAGE_OUTPUT_DIM = 256
    W_OUTPUT_DIM = 8
    SECOND_STAGE_A_OUTPUT_DIM = 8
    SECOND_STAGE_X_OUTPUT_DIM = 32

    INTERNAL_WIDTH = 256

    DROPOUT_RATE = 0.05

    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "linear"):
            super().__init__()
            self.net = nn.Sequential(
                # Stronger Input Noise for small N
                # GaussianNoise(std=0.0001),

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

                # nn.Linear(INTERNAL_WIDTH * 2, INTERNAL_WIDTH * 2),
                # nn.LayerNorm(INTERNAL_WIDTH * 2),
                # nn.ReLU,
                # nn.Dropout(DROPOUT_RATE),

                nn.Linear(INTERNAL_WIDTH * 2, output_dim),
                nn.LayerNorm(output_dim),
                # Activation logic handled below
            )
            self.outbatchnorm = nn.BatchNorm1d(output_dim)
            self.final_activation = final_activation
            self.tanh = nn.Tanh()
            self.gelu = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            if self.final_activation == "tanh":
                return (self.tanh(out) )
            elif self.final_activation == "gelu":
                # return self.outbatchnorm(self.gelu(out))
                return self.gelu(out)
            return (out)

    class OutcomeProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 10):
            super().__init__(input_dim=input_dim, output_dim=W_OUTPUT_DIM, final_activation="linear")

    # Keep others flexible but regularized
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 111):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM, final_activation="gelu")

    class TreatmentFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_A_OUTPUT_DIM, final_activation="gelu")

    class BackdoorFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 100):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_X_OUTPUT_DIM, final_activation="gelu")

    # 2. TARGETED SMALL INITIALIZATION
    # This ensures phi_W(w) starts very close to zero for stability
    def _small_init(m):
        if isinstance(m, nn.Linear):
            # We use a very small scale (0.01) to squash the initial output variance
            nn.init.kaiming_normal_(m.weight, a=0.05) # Use GELU/ReLU compatible init
            m.weight.data *= 0.1
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 0.05) # Scale down LayerNorm gains too

    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    treatment_featurizer = TreatmentFeaturizer().to(device)
    backdoor_featurizer = BackdoorFeaturizer().to(device)
    outcome_proxy_featurizer = OutcomeProxyFeaturizer().to(device)
    # outcome_proxy_featurizer.apply(_small_init)

    return first_stage_featurizer, treatment_featurizer, backdoor_featurizer, outcome_proxy_featurizer


def build_nets_for_outcome_pcl_net_synthetic_high_dim_experiment(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:

    class GaussianNoise(nn.Module):
        def __init__(self, std=0.05):
            super().__init__()
            self.std = std
        def forward(self, x):
            if self.training and self.std > 0:
                return x + torch.randn_like(x) * self.std
            return x

    # --- DIMENSIONS ---
    # Reduced slightly to force compression
    FIRST_STAGE_OUTPUT_DIM = 256
    W_OUTPUT_DIM = 8
    SECOND_STAGE_A_OUTPUT_DIM = 8
    SECOND_STAGE_X_OUTPUT_DIM = 32

    INTERNAL_WIDTH = 256

    DROPOUT_RATE = 0.1

    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "linear"):
            super().__init__()
            self.net = nn.Sequential(
                # Stronger Input Noise for small N
                # GaussianNoise(std=0.0001),

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

                # nn.Linear(INTERNAL_WIDTH * 2, INTERNAL_WIDTH * 2),
                # nn.LayerNorm(INTERNAL_WIDTH * 2),
                # nn.ReLU,
                # nn.Dropout(DROPOUT_RATE),

                nn.Linear(INTERNAL_WIDTH * 2, output_dim),
                nn.LayerNorm(output_dim),
                # Activation logic handled below
            )
            self.outbatchnorm = nn.BatchNorm1d(output_dim)
            self.final_activation = final_activation
            self.tanh = nn.Tanh()
            self.gelu = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            if self.final_activation == "tanh":
                return (self.tanh(out) )
            elif self.final_activation == "gelu":
                return self.outbatchnorm(self.gelu(out))
            return (out)

    class OutcomeProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 10):
            super().__init__(input_dim=input_dim, output_dim=W_OUTPUT_DIM, final_activation="linear")

    # Keep others flexible but regularized
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 111):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM, final_activation="gelu")

    class TreatmentFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_A_OUTPUT_DIM, final_activation="gelu")

    class BackdoorFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 100):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_X_OUTPUT_DIM, final_activation="gelu")

    # 2. TARGETED SMALL INITIALIZATION
    # This ensures phi_W(w) starts very close to zero for stability
    def _small_init(m):
        if isinstance(m, nn.Linear):
            # We use a very small scale (0.01) to squash the initial output variance
            nn.init.kaiming_normal_(m.weight, a=0.05) # Use GELU/ReLU compatible init
            m.weight.data *= 0.1
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 0.05) # Scale down LayerNorm gains too

    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    treatment_featurizer = TreatmentFeaturizer().to(device)
    backdoor_featurizer = BackdoorFeaturizer().to(device)
    outcome_proxy_featurizer = OutcomeProxyFeaturizer().to(device)
    # outcome_proxy_featurizer.apply(_small_init)

    return first_stage_featurizer, treatment_featurizer, backdoor_featurizer, outcome_proxy_featurizer


def build_nets_for_outcome_pcl_net_abortion_and_crime_experiment(device: str = "cuda", w_dim: int = 3) -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    # w_dim = number of outcome-proxy (W) columns kept (default 3 = full). Lowering it (proxy-dimension
    # ablation) reduces the outcome proxy's informativeness -> the outcome bridge deconfounds less well.

    class GaussianNoise(nn.Module):
        def __init__(self, std=0.05):
            super().__init__()
            self.std = std
        def forward(self, x):
            if self.training and self.std > 0:
                return x + torch.randn_like(x) * self.std
            return x

    FIRST_STAGE_OUTPUT_DIM = 64
    W_OUTPUT_DIM = 16
    SECOND_STAGE_A_OUTPUT_DIM = 8

    INTERNAL_WIDTH = 256

    DROPOUT_RATE = 0.01

    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "linear"):
            super().__init__()
            self.net = nn.Sequential(
                # Stronger Input Noise for small N
                GaussianNoise(std=0.01),

                nn.Linear(input_dim, INTERNAL_WIDTH),
                nn.LayerNorm(INTERNAL_WIDTH),
                nn.GELU(),
                # nn.BatchNorm1d(INTERNAL_WIDTH),
                nn.Dropout(DROPOUT_RATE),

                nn.Linear(INTERNAL_WIDTH, INTERNAL_WIDTH // 2),
                nn.LayerNorm(INTERNAL_WIDTH // 2),
                nn.GELU(),
                # nn.BatchNorm1d(INTERNAL_WIDTH // 2),
                nn.Dropout(DROPOUT_RATE),

                nn.Linear(INTERNAL_WIDTH // 2, output_dim),
                # Activation logic handled below
            )
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

    class OutcomeProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 3):
            super().__init__(input_dim=input_dim, output_dim=W_OUTPUT_DIM, final_activation="gelu")

    # Keep others flexible but regularized
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 2):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM, final_activation="gelu")

    class TreatmentFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_A_OUTPUT_DIM, final_activation="gelu")


    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    treatment_featurizer = TreatmentFeaturizer().to(device)
    outcome_proxy_featurizer = OutcomeProxyFeaturizer(input_dim=w_dim).to(device)

    return first_stage_featurizer, treatment_featurizer, outcome_proxy_featurizer


def build_nets_for_outcome_pcl_net_deaner_experiment(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module]:
    """
    Outcome-bridge featurizers for the Deaner grade-retention benchmark (IM/IR).

    Identical to the abortion-and-crime outcome nets EXCEPT the first-stage featurizer takes
    concat(A, Z) of dim 1 + 4 = 5 (Deaner Z is 4-D) instead of 1 + 1 = 2. The outcome-proxy
    featurizer stays W(3) and the A-featurizer stays 1, since W is 3-D in both benchmarks.
    """
    class GaussianNoise(nn.Module):
        def __init__(self, std=0.05):
            super().__init__()
            self.std = std
        def forward(self, x):
            if self.training and self.std > 0:
                return x + torch.randn_like(x) * self.std
            return x

    FIRST_STAGE_OUTPUT_DIM = 64
    W_OUTPUT_DIM = 16
    SECOND_STAGE_A_OUTPUT_DIM = 8
    INTERNAL_WIDTH = 256
    DROPOUT_RATE = 0.01

    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "linear"):
            super().__init__()
            self.net = nn.Sequential(
                GaussianNoise(std=0.01),
                nn.Linear(input_dim, INTERNAL_WIDTH),
                nn.LayerNorm(INTERNAL_WIDTH),
                nn.GELU(),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH, INTERNAL_WIDTH // 2),
                nn.LayerNorm(INTERNAL_WIDTH // 2),
                nn.GELU(),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH // 2, output_dim),
            )
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

    # Outcome-proxy featurizer, input W = 3
    class OutcomeProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 3):
            super().__init__(input_dim=input_dim, output_dim=W_OUTPUT_DIM, final_activation="gelu")

    # First stage input = concat(A, Z) = 1 + 4 = 5  (the only change from abortion)
    class FirstStageFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 5):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM, final_activation="gelu")

    # A featurizer, input A = 1
    class TreatmentFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_A_OUTPUT_DIM, final_activation="gelu")

    first_stage_featurizer = FirstStageFeaturizer().to(device)
    treatment_featurizer = TreatmentFeaturizer().to(device)
    outcome_proxy_featurizer = OutcomeProxyFeaturizer().to(device)

    return first_stage_featurizer, treatment_featurizer, outcome_proxy_featurizer


def build_nets_for_outcome_pcl_net_rhc_experiment(device: str = "cuda", x_dim: int = 20) -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Outcome-bridge featurizers for the Right Heart Catheterization (RHC) benchmark.

    RHC has a **backdoor covariate X (20-D)**, so this returns FOUR featurizers (with a backdoor
    featurizer) like the high-dim synthetic builder -- unlike abortion/Deaner which have no X.
    Dimensions: outcome proxy W = 2, treatment A = 1 (binary), backdoor X = 20, treatment proxy
    Z = 2. The first stage regresses the outcome-proxy features on concat(A, Z, X) = 1 + 2 + 20 = 23.
    Nets are kept modest (width 128) -- RHC is a small real dataset (N~4000) where large capacity
    overfits (the abortion lesson).
    """
    FIRST_STAGE_OUTPUT_DIM = 64
    W_OUTPUT_DIM = 16
    SECOND_STAGE_A_OUTPUT_DIM = 8
    SECOND_STAGE_X_OUTPUT_DIM = 32
    INTERNAL_WIDTH = 128
    DROPOUT_RATE = 0.05

    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "linear"):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, INTERNAL_WIDTH),
                nn.LayerNorm(INTERNAL_WIDTH),
                nn.GELU(),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH, INTERNAL_WIDTH),
                nn.LayerNorm(INTERNAL_WIDTH),
                nn.GELU(),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(INTERNAL_WIDTH, output_dim),
            )
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

    class OutcomeProxyFeaturizer(FeaturizerBase):        # W = 2
        def __init__(self, input_dim: int = 2):
            super().__init__(input_dim=input_dim, output_dim=W_OUTPUT_DIM, final_activation="linear")

    class FirstStageFeaturizer(FeaturizerBase):          # concat(A, Z, X) = 1 + 2 + 20 = 23
        def __init__(self, input_dim: int = 23):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM, final_activation="gelu")

    class TreatmentFeaturizer(FeaturizerBase):            # A = 1
        def __init__(self, input_dim: int = 1):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_A_OUTPUT_DIM, final_activation="gelu")

    class BackdoorFeaturizer(FeaturizerBase):             # X = 20
        def __init__(self, input_dim: int = 20):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_X_OUTPUT_DIM, final_activation="gelu")

    first_stage_featurizer = FirstStageFeaturizer(input_dim=3 + x_dim).to(device)   # A + Z + X = 1 + 2 + x_dim
    treatment_featurizer = TreatmentFeaturizer().to(device)
    backdoor_featurizer = BackdoorFeaturizer(input_dim=x_dim).to(device)
    outcome_proxy_featurizer = OutcomeProxyFeaturizer().to(device)

    return first_stage_featurizer, treatment_featurizer, backdoor_featurizer, outcome_proxy_featurizer


def build_nets_for_outcome_pcl_net_dsprite_experiment(device: str = "cuda", z_dim: int = 3) -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Builds the DFPCL featurizer networks based on Table 3 for the dSprite high-dimensional
    image experiment. These networks include Spectral Normalization (SN) and
    Batch Normalization (BN) for stability with high-dimensional input.

    Inputs: A (4096D), Z (3D), W (4096D).

    Returns: (phi_A1, psi_A2, psi_W, phi_Z)
    """

    # --- INPUT DIMENSIONS ---
    DIM_IMAGE = 4096
    DIM_Z = z_dim   # 3 = inherited V2 benchmark (numerically identical default); 2 = the repaired V3 DGP

    # --- Output/Internal Dimensions ---
    FIRST_STAGE_OUTPUT_DIM = 128
    SECOND_STAGE_A_OUTPUT_DIM = 16
    OUT_DIM_W = 16

    # 1. Stage 1 Treatment Feature (phi_A1)
    class FirstStageFeaturizer(nn.Module):
        def __init__(self, input_dim: int = DIM_IMAGE + DIM_Z, output_dim: int = FIRST_STAGE_OUTPUT_DIM):
            super().__init__()
            self.net = nn.Sequential(
                # FC(4096, 1024), SN, ReLU
                nn.Linear(input_dim, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                # FC(1024, 512), SN, ReLU, BN
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                # nn.BatchNorm1d(512),
                # FC(512, 128), SN, ReLU
                nn.Linear(512, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                # FC(128, 32), SN, ReLU
                nn.Linear(128, output_dim), # Output dim is 32
                nn.LayerNorm(output_dim),
                nn.ReLU(),
            )
            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    # 3. Outcome Proxy Feature (psi_W)
    class OutcomeProxyFeaturizer(nn.Module):
        def __init__(self, input_dim: int = DIM_IMAGE, output_dim: int = OUT_DIM_W):
            super().__init__()
            self.net = nn.Sequential(
                # FC(4096, 1024), SN, ReLU
                nn.Linear(input_dim, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                # FC(1024, 512), SN, ReLU, BN
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                # FC(512, 128), SN, ReLU
                nn.Linear(512, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                # FC(128, 32), SN, ReLU
                nn.Linear(128, output_dim), # Output dim is 32
                nn.LayerNorm(output_dim),
                nn.ReLU(),
            )
            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class TreatmentFeaturizer(nn.Module):
        def __init__(self, input_dim: int = DIM_IMAGE, output_dim: int = SECOND_STAGE_A_OUTPUT_DIM):
            super().__init__()
            self.net = nn.Sequential(
                # FC(4096, 1024), SN, ReLU
                nn.Linear(input_dim, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                # FC(1024, 512), SN, ReLU, BN
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                # FC(512, 128), SN, ReLU
                nn.Linear(512, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                # FC(128, 32), SN, ReLU
                nn.Linear(128, output_dim), # Output dim is 32
                nn.LayerNorm(output_dim),
                nn.ReLU(),
            )
            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    # 2. TARGETED SMALL INITIALIZATION
    # This ensures phi_W(w) starts very close to zero for stability
    def _small_init(m):
        if isinstance(m, nn.Linear):
            # We use a very small scale (0.01) to squash the initial output variance
            nn.init.kaiming_normal_(m.weight, a=0.2) # Use GELU/ReLU compatible init
            m.weight.data *= 0.2
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 0.2) # Scale down LayerNorm gains too


    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    outcome_proxy_featurizer = OutcomeProxyFeaturizer().to(device)
    treatment_featurizer = TreatmentFeaturizer().to(device)
    # outcome_proxy_featurizer.apply(_small_init)
    return first_stage_featurizer, treatment_featurizer, outcome_proxy_featurizer


def build_nets_for_outcome_pcl_net_dsprite_experiment_configurable(
        device: str = "cuda", z_dim: int = 3, width_mult: float = 1.0,
        residual: bool = False, n_blocks: int = 2, first_out: int = 128,
        w_out: int = 16, a2_out: int = 16):
    """Capacity/residual-configurable dSprite outcome featurizers (architecture-search variant).

    A SEPARATE builder from build_nets_for_outcome_pcl_net_dsprite_experiment so the default sim path
    stays numerically identical -- this is only reached under --om_arch configurable. Two modes:
      residual=False : the shipped 3-layer width ladder [1024,512,128] scaled by width_mult
                       (0.5 = smaller, 2.0 = larger), LN+ReLU each; BN after the 512-scaled layer on
                       the W and treatment featurizers only (matches the shipped asymmetry: the
                       first-stage featurizer has NO BN).
      residual=True  : Linear stem -> width w=round(512*width_mult); n_blocks pre-activation residual
                       blocks x = x + ReLU(LN(Linear(w,w))); Linear head -> out; same BN placement.
    Returns (first_stage, treatment, outcome_proxy) -- the order OutcomeBridgePCLNET wants.
    """
    DIM_IMAGE = 4096
    DIM_Z = z_dim
    m = float(width_mult)
    ladder = [max(int(round(h * m)), 8) for h in (1024, 512, 128)]
    w_res = max(int(round(512 * m)), 8)

    class _ResBlock(nn.Module):
        def __init__(self, w):
            super().__init__()
            self.lin = nn.Linear(w, w); self.ln = nn.LayerNorm(w); self.act = nn.ReLU()
        def forward(self, x):
            return x + self.act(self.ln(self.lin(x)))

    class Featurizer(nn.Module):
        def __init__(self, input_dim, output_dim, use_bn):
            super().__init__()
            layers = []
            if residual:
                layers += [nn.Linear(input_dim, w_res), nn.LayerNorm(w_res), nn.ReLU()]
                if use_bn:
                    layers += [nn.BatchNorm1d(w_res)]
                for _ in range(n_blocks):
                    layers += [_ResBlock(w_res)]
                layers += [nn.Linear(w_res, output_dim), nn.LayerNorm(output_dim), nn.ReLU()]
            else:
                d = input_dim
                for i, h in enumerate(ladder):
                    layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU()]
                    if use_bn and i == 1:                 # BN after the 2nd (512-scaled) layer
                        layers += [nn.BatchNorm1d(h)]
                    d = h
                layers += [nn.Linear(d, output_dim), nn.LayerNorm(output_dim), nn.ReLU()]
            self.net = nn.Sequential(*layers)
            self.output_dim = output_dim
        def forward(self, x):
            return self.net(x)

    first_stage = Featurizer(DIM_IMAGE + DIM_Z, first_out, use_bn=False).to(device)
    outcome_proxy = Featurizer(DIM_IMAGE, w_out, use_bn=True).to(device)
    treatment = Featurizer(DIM_IMAGE, a2_out, use_bn=True).to(device)
    return first_stage, treatment, outcome_proxy


def build_nets_for_outcome_pcl_net_dsprite_experiment_sn(device: str = "cuda", z_dim: int = 3, feat_dim: int = 32):
    """OUR OutcomeBridgePCLNET with the DFPV-paper INGREDIENTS (Xu et al. 2021) that the
    ablation showed carry their dSprite result -- the METHOD (our class, our proximal closed-form
    trainer, our third stage) is unchanged; only these featurizers differ from the shipped builder:
      * spectral-norm Linear + ReLU image nets 4096-1024-512(BatchNorm)-128-32 -- NO LayerNorm
        (LayerNorm nets scored 23.7 vs 15.7 in their harness), no dropout, no GELU, no small-init;
      * FIRST STAGE IS A TENSOR PRODUCT phi_A(A) (x) phi_Z(Z) with intercepts (33x33 = 1089 features),
        built INSIDE the first-stage featurizer from the (A||Z) concat the class hands it, so the
        class is untouched (it appends its own const column on top -- harmless collinearity);
      * output dims 32/32 (shipped: 128 first stage, 16/16).
    Pair with the dSprite sim knobs that transfer the rest of their recipe: --y_scale raw --z_scale raw
    --om_epochs 20 --om_stage1_iter 20 --om_weight_decay 0.1 and full-batch loaders.
    Returns (first_stage, treatment_second_stage, outcome_proxy) as OutcomeBridgePCLNET wants.
    """
    from torch.nn.utils import spectral_norm
    DIM_IMAGE = 4096

    def img_net(out=32):
        return nn.Sequential(spectral_norm(nn.Linear(DIM_IMAGE, 1024)), nn.ReLU(),
                             spectral_norm(nn.Linear(1024, 512)), nn.ReLU(), nn.BatchNorm1d(512),
                             spectral_norm(nn.Linear(512, 128)), nn.ReLU(),
                             spectral_norm(nn.Linear(128, out)), nn.ReLU())

    class TensorProductFirstStage(nn.Module):
        """input = hstack([A (4096), Z (z_dim)]) -> flatten( [phi_A(A),1] (x) [phi_Z(Z),1] )."""
        def __init__(self):
            super().__init__()
            self.a_net = img_net(feat_dim)
            self.z_net = nn.Sequential(nn.Linear(z_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                                       nn.Linear(64, feat_dim), nn.ReLU())
            self.output_dim = (feat_dim + 1) ** 2 - 1   # the (1 x 1) entry is dropped: see forward()
            # NOTE the first-stage fold at n=2000 has ~600 rows < 1088 features (feat_dim 32): the
            # stage-1 Gram is rank-deficient, so either a REAL first-stage ridge or a smaller feat_dim
            # is required (probe arm a exploded to 1e23 with reg_first_stage (5e-3,1e-3)).

        def forward(self, x):
            a, z = x[:, :DIM_IMAGE], x[:, DIM_IMAGE:DIM_IMAGE + z_dim]
            fa = self.a_net(a); fz = self.z_net(z)
            one = torch.ones(fa.shape[0], 1, device=fa.device, dtype=fa.dtype)
            fa = torch.cat([fa, one], 1); fz = torch.cat([fz, one], 1)
            # [phi_A,1] (x) [phi_Z,1] minus its constant (1x1) entry: OutcomeBridgePCLNET appends its
            # OWN const column, and an exact duplicate made the first-stage Gram singular
            # (torch.linalg.solve: "input matrix is singular" -> NaN -> OM 161 / crash).
            return (fa.unsqueeze(2) * fz.unsqueeze(1)).flatten(start_dim=1)[:, :-1]

    class Wrap(nn.Module):
        def __init__(self, net, out):
            super().__init__(); self.net = net; self.output_dim = out
        def forward(self, x):
            return self.net(x)

    first_stage = TensorProductFirstStage().to(device)
    treatment = Wrap(img_net(feat_dim), feat_dim).to(device)
    outcome_proxy = Wrap(img_net(feat_dim), feat_dim).to(device)
    return first_stage, treatment, outcome_proxy


def build_nets_for_outcome_pcl_net_demand_experiment(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:

    FIRST_STAGE_OUTPUT_DIM = 32
    W_OUTPUT_DIM = 16
    SECOND_STAGE_A_OUTPUT_DIM = 16
    INTERNAL_WIDTH = 32 # Keep internal layers reasonably wide
    DROPOUT_RATE = 0.01

    class FeaturizerBase(nn.Module):
        """Standard featurizer structure with GELU activation."""
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "relu"):
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
            )
            # self.outbatchnorm = nn.BatchNorm1d(output_dim)
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
        def __init__(self, input_dim: int = 3, final_activation = "gelu"):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM, final_activation=final_activation)

    # --- Outcome Proxy Featurizer (Input: 2D) ---
    class OutcomeProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1, final_activation = "gelu"):
            super().__init__(input_dim=input_dim, output_dim=W_OUTPUT_DIM, final_activation=final_activation)

    # --- Treatment Proxy Featurizer (Input: 2D) ---
    class TreatmentFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1, final_activation = "gelu"):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_A_OUTPUT_DIM, final_activation=final_activation)


    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    outcome_proxy_featurizer = OutcomeProxyFeaturizer().to(device)
    treatment_featurizer = TreatmentFeaturizer().to(device)

    return first_stage_featurizer, treatment_featurizer, outcome_proxy_featurizer


def build_nets_for_outcome_pcl_net_octmnist_experiment(
        device: str = "cuda", first_stage_out: int = 64, w_out: int = 16, a_out: int = 8,
        width: int = 64, dropout: float = 0.01) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """Outcome-bridge featurizers for the proximal OCTMNIST experiment (STRUCTURED image outcome).

    Inputs are low-dimensional -- first stage on concat(A, Z) = 7-D, treatment on A = 1-D, outcome proxy
    on W = 6-D -- while the bridge h(A, W) must output a 28x28 image. As in ColorMNIST, the image is
    produced by the SECOND-STAGE HEAD, not by the featurizers:

        h(a, w) = <Theta, phi_A(a) (x) phi_W(w)>,     Theta of shape (784, a_out * w_out + 1),

    so predictions live in a rank-(a_out*w_out+1) subspace of R^784. The conditional mean image is
    E[B] + g(w) theta M_C + delta 1{a=1} M_T, which spans only about three directions ({mean baseline,
    M_C, M_T}) -- but w_out must still be large enough to represent the scalar g(w) over a 6-D proxy,
    and THAT is what sets the rank the bridge needs. Rank sweep at the notebook operating point
    (n = 12000, sigma_z = sigma_w = 0.1, delta = 0.30; c_C = 0 is perfect deconfounding):

        (a_out, w_out) = (2, 4)   rank   9   c_T = +0.000   c_C = +0.000   <- collapses, no effect at all
        (a_out, w_out) = (4, 8)   rank  33   c_T = +0.276   c_C = +0.092   <- effect back, still confounded
        (a_out, w_out) = (8, 16)  rank 129   c_T = +0.231   c_C = -0.006   <- deconfounded  (DEFAULT)
        (a_out, w_out) = (16, 32) rank 513   c_T = +0.253   c_C = +0.006   <- saturated

    Defaults therefore give rank 129. Note the causal MSE over the full 784-D image is nearly flat
    across this sweep (~5.5e-5): it is dominated by reconstructing the shared baseline image E[B], not
    by the effect, which is why the template projections (c_T, c_C) are the informative diagnostics.

    Pass ``outcome_dim = 28 * 28`` to OutcomeBridgePCLNET so Theta is built with the right height.
    """

    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "gelu"):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width, width * 2),
                nn.LayerNorm(width * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width * 2, output_dim),
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

    first_stage_featurizer = FeaturizerBase(input_dim=7, output_dim=first_stage_out).to(device)  # (A, Z)
    treatment_featurizer = FeaturizerBase(input_dim=1, output_dim=a_out).to(device)              # A
    outcome_proxy_featurizer = FeaturizerBase(input_dim=6, output_dim=w_out).to(device)          # W

    return first_stage_featurizer, treatment_featurizer, outcome_proxy_featurizer


def build_nets_for_outcome_pcl_net_colormnist_experiment(
        device: str = "cuda", first_stage_out: int = 32, w_out: int = 4, a_out: int = 4,
        width: int = 32, dropout: float = 0.01) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """Outcome-bridge featurizers for the proximal ColorMNIST experiment (STRUCTURED image outcome).

    All THREE featurizer inputs are low-dimensional -- first stage on concat(A, Z) = 2-D, treatment on
    A = 1-D, outcome proxy on W = 1-D -- even though the bridge h(A, W) must output a 3*28*28 image.
    That is not an oversight: the image is produced by the SECOND-STAGE HEAD, not by the featurizers.
    The head is a linear map

        h(a, w) = <Theta, phi_A(a) (x) phi_W(w)> ,     Theta of shape (2352, a_out * w_out + 1),

    so predicted images live in a subspace of R^2352 of rank at most ``a_out * w_out + 1``. The target
    manifold here is far smaller than that: the conditional mean image is (mean digit-a shape) painted
    with a foreground colour c_fg(X) = (X, alpha, 1 - X) that is AFFINE in the confounder, so
    E[Y | a, x] ~ base_a + x * slope_a spans only about four directions. A convolutional or otherwise
    "advanced" featurizer would add capacity on the input side, where there is nothing to gain -- the
    binding constraint is the rank of Theta, which ``a_out`` and ``w_out`` control directly.

    MEASURED (n = 5000, seed 42). An SVD of the binned conditional-mean images E[Y | a, x] puts 99% of
    the energy in FOUR directions out of 20, and a capacity sweep over the rank of Theta saturates
    immediately:

        a_out x w_out    rank(Theta)    causal MSE      (naive observational baseline: 7.7e-4)
            2 x 2              5         1.8e-3   <- under-capacity, worse than doing nothing
            4 x 4             17         1.1e-4   <- saturated
            8 x 8             65         1.1e-4
           16 x 16           257         1.2e-4   <- mild over-fitting

    Hence the defaults ``a_out = w_out = 4``. Capacity is NOT the bottleneck past rank ~17; adding a
    convolutional featurizer would spend parameters where the sweep shows no headroom.

    Pass ``outcome_dim = 3 * 28 * 28`` to OutcomeBridgePCLNET so Theta is built with the right height.
    """

    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "gelu"):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width, width * 2),
                nn.LayerNorm(width * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width * 2, output_dim),
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

    first_stage_featurizer = FeaturizerBase(input_dim=2, output_dim=first_stage_out).to(device)   # (A, Z)
    treatment_featurizer = FeaturizerBase(input_dim=1, output_dim=a_out).to(device)               # A
    outcome_proxy_featurizer = FeaturizerBase(input_dim=1, output_dim=w_out).to(device)           # W

    return first_stage_featurizer, treatment_featurizer, outcome_proxy_featurizer


def build_nets_for_dfpcl_dsprite_experiment_with_compressed_images(latent_dim: int = 8, device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Builds the DFPCL featurizer networks based on Table 3 for the dSprite high-dimensional
    image experiment. These networks include Spectral Normalization (SN) and
    Batch Normalization (BN) for stability with high-dimensional input.

    Inputs: A (4096D), Z (3D), W (4096D).

    Returns: (phi_A1, psi_A2, psi_W, phi_Z)
    """

    # --- INPUT DIMENSIONS ---
    DIM_IMAGE = latent_dim
    DIM_Z = 3

    # --- Output/Internal Dimensions ---
    FIRST_STAGE_OUTPUT_DIM = 128
    SECOND_STAGE_A_OUTPUT_DIM = 16
    OUT_DIM_W = 16

    # 1. Stage 1 Treatment Feature (phi_A1)
    class FirstStageFeaturizer(nn.Module):
        def __init__(self, input_dim: int = DIM_IMAGE + DIM_Z, output_dim: int = FIRST_STAGE_OUTPUT_DIM):
            super().__init__()
            self.net = nn.Sequential(
                # FC(4096, 1024), SN, ReLU
                nn.Linear(input_dim, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                # FC(1024, 512), SN, ReLU, BN
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                # nn.BatchNorm1d(512),
                # FC(512, 128), SN, ReLU
                nn.Linear(512, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                # FC(128, 32), SN, ReLU
                nn.Linear(128, output_dim), # Output dim is 32
                nn.LayerNorm(output_dim),
                nn.ReLU(),
            )
            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    # 3. Outcome Proxy Feature (psi_W)
    class OutcomeProxyFeaturizer(nn.Module):
        def __init__(self, input_dim: int = DIM_IMAGE, output_dim: int = OUT_DIM_W):
            super().__init__()
            self.net = nn.Sequential(
                # FC(4096, 1024), SN, ReLU
                nn.Linear(input_dim, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                # FC(1024, 512), SN, ReLU, BN
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                # FC(512, 128), SN, ReLU
                nn.Linear(512, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                # FC(128, 32), SN, ReLU
                nn.Linear(128, output_dim), # Output dim is 32
                nn.LayerNorm(output_dim),
                nn.ReLU(),
            )
            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class TreatmentFeaturizer(nn.Module):
        def __init__(self, input_dim: int = DIM_IMAGE, output_dim: int = SECOND_STAGE_A_OUTPUT_DIM):
            super().__init__()
            self.net = nn.Sequential(
                # FC(4096, 1024), SN, ReLU
                nn.Linear(input_dim, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                # FC(1024, 512), SN, ReLU, BN
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                # FC(512, 128), SN, ReLU
                nn.Linear(512, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                # FC(128, 32), SN, ReLU
                nn.Linear(128, output_dim), # Output dim is 32
                nn.LayerNorm(output_dim),
                nn.ReLU(),
            )
            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    outcome_proxy_featurizer = OutcomeProxyFeaturizer().to(device)
    treatment_featurizer = TreatmentFeaturizer().to(device)

    return first_stage_featurizer, treatment_featurizer, outcome_proxy_featurizer


def build_nets_for_outcome_pcl_net_synthetic_cate_experiment(device: str = "cuda") -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module]:
    FIRST_STAGE_OUTPUT_DIM = 128
    W_OUTPUT_DIM = 16
    SECOND_STAGE_A_OUTPUT_DIM = 4
    SECOND_STAGE_V_OUTPUT_DIM = 8
    INTERNAL_WIDTH = 64 # Keep internal layers reasonably wide
    DROPOUT_RATE = 0.01
    class FeaturizerBase(nn.Module):
        """Standard featurizer structure with GELU activation."""
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "relu"):
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
            )
            # self.outbatchnorm = nn.BatchNorm1d(output_dim)
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
        def __init__(self, input_dim: int = 5, final_activation = "gelu"):
            super().__init__(input_dim=input_dim, output_dim=FIRST_STAGE_OUTPUT_DIM, final_activation=final_activation)

    # --- Outcome Proxy Featurizer (Input: 2D) ---
    class OutcomeProxyFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 3, final_activation = "gelu"):
            super().__init__(input_dim=input_dim, output_dim=W_OUTPUT_DIM, final_activation=final_activation)

    # --- Treatment Proxy Featurizer (Input: 2D) ---
    class TreatmentFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1, final_activation = "gelu"):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_A_OUTPUT_DIM, final_activation=final_activation)

    class CovariateFeaturizer(FeaturizerBase):
        def __init__(self, input_dim: int = 1, final_activation = "gelu"):
            super().__init__(input_dim=input_dim, output_dim=SECOND_STAGE_V_OUTPUT_DIM, final_activation=final_activation)

    def _small_init(m):
        if isinstance(m, nn.Linear):
            # We use a very small scale (0.01) to squash the initial output variance
            nn.init.kaiming_normal_(m.weight, a=0.2) # Use GELU/ReLU compatible init
            m.weight.data *= 0.2
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 0.2) # Scale down LayerNorm gains too

    # Instantiate the models
    first_stage_featurizer = FirstStageFeaturizer().to(device)
    outcome_proxy_featurizer = OutcomeProxyFeaturizer().to(device)
    treatment_featurizer = TreatmentFeaturizer().to(device)
    covariate_featurizer = CovariateFeaturizer().to(device)
    outcome_proxy_featurizer.apply(_small_init)
    return first_stage_featurizer, treatment_featurizer, covariate_featurizer, outcome_proxy_featurizer





def build_nets_for_outcome_pcl_net_colormnist_cate_experiment(
        device: str = "cuda", v_dim: int = 5, width: int = 64, dropout: float = 0.01) -> Tuple[
        nn.Module, nn.Module, nn.Module, nn.Module]:
    """Heterogeneous outcome-bridge featurizers for the ColorMNIST-PCL-CATE benchmark.

    Dims: A 1-D binary, V one-hot over the digit subgroups (v_dim, default 5), Z 1-D, W 1-D, and the
    OUTCOME IS AN IMAGE -- pass ``outcome_dim = 3 * 28 * 28`` to HeterogeneousOutcomeBridgePCLNET so
    Theta is built with the right height (same contract as the ColorMNIST ATE builder above).

    V is one-hot rather than a scalar because the subgroups are CATEGORICAL digit classes; feeding the
    raw label would impose a spurious ordering and let the featurizer interpolate between classes.

    Returns (first_stage, treatment, covariate, outcome_proxy) -- the positional order
    HeterogeneousOutcomeBridgePCLNET expects.
    """
    FIRST_STAGE_OUTPUT_DIM = 64
    W_OUTPUT_DIM = 8
    SECOND_STAGE_A_OUTPUT_DIM = 4
    SECOND_STAGE_V_OUTPUT_DIM = 8

    class FeaturizerBase(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, final_activation: str = "gelu"):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(width, width * 2), nn.LayerNorm(width * 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(width * 2, output_dim), nn.LayerNorm(output_dim),
            )
            self.output_dim = output_dim
            self.final_activation = final_activation
            self.tanh, self.gelu = nn.Tanh(), nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            if self.final_activation == "tanh":
                return self.tanh(out)
            if self.final_activation == "gelu":
                return self.gelu(out)
            return out

    # first stage regresses phi_W(W) on features of (A, V, Z)
    first_stage_featurizer = FeaturizerBase(1 + v_dim + 1, FIRST_STAGE_OUTPUT_DIM).to(device)
    treatment_featurizer = FeaturizerBase(1, SECOND_STAGE_A_OUTPUT_DIM).to(device)          # A
    covariate_featurizer = FeaturizerBase(v_dim, SECOND_STAGE_V_OUTPUT_DIM).to(device)      # V (one-hot)
    outcome_proxy_featurizer = FeaturizerBase(1, W_OUTPUT_DIM).to(device)                   # W
    # NOTE: no _small_init here. It helps some synthetic benchmarks but is recorded as
    # BACKFIRING on others (abortion; binary +-1 proxies, where it broke OutcomeNet outright). This
    # benchmark's W is a plain Gaussian proxy, so the default init is used unless a sweep says otherwise.
    return (first_stage_featurizer, treatment_featurizer, covariate_featurizer,
            outcome_proxy_featurizer)
