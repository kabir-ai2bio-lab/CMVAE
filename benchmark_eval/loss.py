"""Loss functions for the AE/VAE benchmarks (reconstruction x regulariser combinations).

reconstruction : 'mse'     - mean squared error
                 'logcosh' - log-cosh (smooth, outlier-robust), numerically stable form
regulariser    : 'kl'  - KL(q(z|x) || N(0, I)) with an optional per-dimension free-bits floor
                 'mmd' - RBF-kernel MMD between z and N(0, I)

The CMVAE/MAE models use their own (MSE + KL / MSE + MMD) losses in ``CMVAE/losses.py``.
"""
import math

import torch
import torch.nn.functional as F

_LOG2 = math.log(2.0)

# (display name, reconstruction, regulariser) - the four VAE benchmark variants.
VAE_LOSS_COMBOS = [
    ("MSE+KL", "mse", "kl"),
    ("Logcosh+KL", "logcosh", "kl"),
    ("MSE+MMD", "mse", "mmd"),
    ("Logcosh+MMD", "logcosh", "mmd"),
]


def recon_mse(recon, x):
    return F.mse_loss(recon, x, reduction="mean")


def recon_logcosh(recon, x):
    absd = (recon - x).abs()
    return (absd + F.softplus(-2.0 * absd) - _LOG2).mean()


def kl_divergence(mu, log_var, free_bits: float = 0.0):
    kl_per_dim = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp()).mean(dim=0)
    if free_bits > 0.0:
        kl_per_dim = kl_per_dim.clamp(min=free_bits)
    return kl_per_dim.mean()


def mmd_loss(z, sigma: float = 1.0):
    z_prior = torch.randn_like(z)

    def rbf(a, b):
        d2 = ((a.unsqueeze(1) - b.unsqueeze(0)) ** 2).sum(-1)
        return torch.exp(-d2 / (2.0 * sigma ** 2))

    return (rbf(z, z).mean() + rbf(z_prior, z_prior).mean() - 2.0 * rbf(z, z_prior).mean()).clamp(min=0.0)
