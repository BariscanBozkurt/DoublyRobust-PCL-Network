import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import List
import matplotlib.pyplot as plt

class ConditionalMeanMLP(nn.Module):
    """
    Multilayer Perceptron for Conditional Mean Estimation (CME).

    Used in the third stage of the Treatment Bridge PCL method to estimate
    the structural function Psi(A) = E[Y varphi_Z(A, Z) | A]
    by regressing composite targets onto treatment features.

    Parameters
    ----------
    input_dim : int
        Dimension of the input features (typically the dimension of Treatment A).
    output_dim : int
        Dimension of the composite target to predict.
    hidden_dims : List[int], default=[32, 64]
        Dimensions of the hidden layers.
    dropout_rate : float, default=0.0
        Dropout probability for regularization.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [32, 64],
        dropout_rate: float = 0.0
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers dynamically
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            # layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            # layers.append(nn.BatchNorm1d(h_dim))
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        # Final projection layer (No activation to allow regression over all Reals)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

        # # Apply stable weight initialization
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Kaiming Normal handles the variance shift caused by GELU activations
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, A_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Returns
        -------
        torch.Tensor
            Estimated conditional mean of shape (N, output_dim)
        """
        return self.net(A_input)


class EnsembleConditionalMeanMLP(nn.Module):
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        # ModuleList ensures all sub-models are properly registered (e.g. for.to(device))
        self.models = nn.ModuleList(models)

    def forward(self, x):
        # 1. Get predictions from all models: Shape [N_models, Batch_Size, 1]
        preds = torch.stack([model(x) for model in self.models])

        # 2. Average them: Shape [Batch_Size, 1]
        return torch.mean(preds, dim=0)

    def predict(self, x):
        """Helper for inference that handles no_grad automatically"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class ConvBetaVAE_dSprite(nn.Module):
    def __init__(self, latent_dim: int = 16, beta: float = 4.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # --- ENCODER ---
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Flatten()
        )

        self.fc_mu = nn.Linear(4096, latent_dim)
        self.fc_logvar = nn.Linear(4096, latent_dim)

        # --- DECODER ---
        self.decoder_input = nn.Linear(latent_dim, 4096)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),

            nn.Identity()
        )

    def encode(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.view(-1, 1, 64, 64)

        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        logvar = torch.clamp(logvar, min=-20.0, max=10.0)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor):
        h = self.decoder_input(z)
        h = h.view(-1, 256, 4, 4)
        recon_image = self.decoder(h)
        return recon_image.view(-1, 4096)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)

        # --- THE TRICK IS BACK! ---
        # Injecting the noise forces the latent space to become smooth and continuous.
        z = self.reparameterize(mu, logvar)

        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def get_latent_features(self, x: torch.Tensor):
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
            return mu

def beta_vae_loss(recon_x, x, mu, logvar, current_beta):
    recon_loss = F.mse_loss(recon_x, x, reduction='none').sum(dim=1).mean()
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    return recon_loss + current_beta * kl_divergence, recon_loss, kl_divergence

def train_beta_vae(
    A_tensor: torch.Tensor,
    latent_dim: int = 8,
    target_beta: float = 2.0,
    n_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cuda"
):
    print(f"Initializing ConvBeta-VAE (Latent Dim: {latent_dim}, Target Beta: {target_beta})")

    dataset = TensorDataset(A_tensor.to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vae = ConvBetaVAE_dSprite(latent_dim=latent_dim, beta=target_beta).to(device)

    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    loss_hist, recon_hist, kl_hist = [], [], []

    for epoch in tqdm(range(n_epochs), desc="Training VAE"):
        vae.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0

        anneal_ratio = min(1.0, epoch / (n_epochs * 0.5))
        current_beta = target_beta * anneal_ratio

        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()

            recon_x, mu, logvar = vae(x)

            total_loss, recon_loss, kl_loss = beta_vae_loss(recon_x, x, mu, logvar, current_beta)

            total_loss.backward()

            # Restored Gradient Clipping! Vital for stability.
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()

        scheduler.step()

        num_batches = len(dataloader)
        loss_hist.append(epoch_loss / num_batches)
        recon_hist.append(epoch_recon / num_batches)
        kl_hist.append(epoch_kl / num_batches)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(loss_hist)
    plt.title("Total Beta-VAE Loss")
    plt.subplot(1, 3, 2)
    plt.plot(recon_hist, color='orange')
    plt.title("Reconstruction Loss (MSE)")
    plt.subplot(1, 3, 3)
    plt.plot(kl_hist, color='green')
    plt.title("KL Divergence")
    plt.tight_layout()
    plt.show()
    print("Final reconstruction loss:", recon_hist[-1])
    print("Final KL divergence:", kl_hist[-1])
    return vae.eval()