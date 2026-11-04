"""Plain autoencoder benchmark : symmetric MLP with dropout."""
import torch.nn as nn


class Autoencoder(nn.Module):
    """input -> 512 -> 256 -> 128 -> latent  |  latent -> 128 -> 256 -> 512 -> input (linear output)."""

    def __init__(self, input_dim: int, latent_dim: int = 32, dropout_rate: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent), latent
