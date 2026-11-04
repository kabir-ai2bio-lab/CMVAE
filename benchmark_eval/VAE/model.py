"""Vanilla VAE benchmark: MLP encoder/decoder with dropout, mu/logvar heads."""
import torch
import torch.nn as nn


class VAE(nn.Module):
    """Encoder: input -> 512 -> 256 -> 128 -> (mu, log_var);  decoder mirrors it with a linear output.
    At inference the deterministic posterior mean ``mu`` is the embedding (no sampling)."""

    def __init__(self, input_dim: int, latent_dim: int = 32, dropout_rate: float = 0.3):
        super().__init__()
        self.encoder_hidden = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout_rate),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, input_dim),
        )

    def encode(self, x):
        h = self.encoder_hidden(x)
        return self.fc_mu(h), self.fc_log_var(h)

    @staticmethod
    def reparameterise(mu, log_var):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterise(mu, log_var)
        return self.decode(z), mu, log_var, z
