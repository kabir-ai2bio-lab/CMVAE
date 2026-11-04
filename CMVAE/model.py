"""
CMVAE model classes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_mask_noise(x: torch.Tensor, mask_ratio: float = 0.15) -> torch.Tensor:
    """Element-wise random masking: each feature is zeroed with probability mask_ratio.

    The reconstruction target stays the clean input.
    """
    if mask_ratio <= 0:
        return x
    keep = (torch.rand_like(x) > mask_ratio).float()
    return x * keep


class MaskedDenoisingVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 128, hidden1: int = 2048, hidden2: int = 1024):
        super().__init__()
        # encoder
        self.enc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.enc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.mu = nn.Linear(hidden2, latent_dim)
        self.logvar = nn.Linear(hidden2, latent_dim)
        # decoder
        self.dec1 = nn.Linear(latent_dim, hidden2)
        self.bn3 = nn.BatchNorm1d(hidden2)
        self.dec2 = nn.Linear(hidden2, hidden1)
        self.bn4 = nn.BatchNorm1d(hidden1)
        self.out = nn.Linear(hidden1, input_dim)   # linear output: inputs are z-scored, continuous

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
        """No projection head in the plain masked VAE."""
        return None

    def forward(self, x, mask_ratio: float = 0.0):
        x_noisy = apply_mask_noise(x, mask_ratio=mask_ratio)
        mu, logvar = self.encode(x_noisy)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z, self.project(mu)


class CMVAE(MaskedDenoisingVAE):
    """Contrastive Masked denoising VAE = MaskedDenoisingVAE + projection head g(.)."""

    def __init__(self, input_dim: int, latent_dim: int = 128, proj_dim: int = 64, **kw):
        super().__init__(input_dim, latent_dim, **kw)
        self.proj_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, proj_dim),
        )

    def project(self, mu):
        return F.normalize(self.proj_head(mu), dim=1)  
