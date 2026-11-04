"""MAE benchmark = masked denoising VAE.

Same network and masking as CMVAE but WITHOUT the projection head/contrastive term:
the only training signal is masked reconstruction + KL or MMD regularisation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from CMVAE.model import apply_mask_noise   # same masking function as CMVAE


class MAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 128, hidden1: int = 2048, hidden2: int = 1024):
        super().__init__()
        # encoder: 2 x (Linear - BatchNorm - ReLU), then mu/logvar heads
        self.enc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.enc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.mu = nn.Linear(hidden2, latent_dim)
        self.logvar = nn.Linear(hidden2, latent_dim)
        # decoder: mirror of the encoder, linear output (inputs are z-scored)
        self.dec1 = nn.Linear(latent_dim, hidden2)
        self.bn3 = nn.BatchNorm1d(hidden2)
        self.dec2 = nn.Linear(hidden2, hidden1)
        self.bn4 = nn.BatchNorm1d(hidden1)
        self.out = nn.Linear(hidden1, input_dim)

    def encode(self, x):
        h = F.relu(self.bn1(self.enc1(x)))
        h = F.relu(self.bn2(self.enc2(h)))
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar.clamp(-4, 4))
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = F.relu(self.bn3(self.dec1(z)))
        h = F.relu(self.bn4(self.dec2(h)))
        return self.out(h)

    def project(self, mu):
        """No projection head in MAE (kept so the training loop is shared with CMVAE)."""
        return None

    def forward(self, x, mask_ratio: float = 0.0):
        x_noisy = apply_mask_noise(x, mask_ratio=mask_ratio)    # reconstruction target stays the clean x
        mu, logvar = self.encode(x_noisy)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z, self.project(mu)
