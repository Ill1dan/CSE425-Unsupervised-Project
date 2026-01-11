from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CNNVAEConfig:
    latent_dim: int = 16
    base_channels: int = 32
    beta: float = 1.0  # KL weight


class CNNVAE(nn.Module):
    """
    CNN-VAE for mel-spectrograms shaped (B, 1, H, W).

    Encoder:
      Conv blocks (stride=2) -> flatten -> mu/logvar

    Decoder:
      linear -> reshape -> ConvTranspose2D blocks -> recon (B, 1, H, W)
    """

    def __init__(self, input_hw: tuple[int, int], cfg: CNNVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.input_h, self.input_w = input_hw

        c = cfg.base_channels

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, stride=2, padding=1),  # /2
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 2 * c, kernel_size=3, stride=2, padding=1),  # /4
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * c, 4 * c, kernel_size=3, stride=2, padding=1),  # /8
            nn.ReLU(inplace=True),
        )

        # compute feature map size after encoder
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.input_h, self.input_w)
            h = self.enc(dummy)
            self._enc_out_shape = h.shape  # (1, C, H', W')
            flat_dim = h.view(1, -1).shape[1]
            self._flat_dim = flat_dim

        self.fc_mu = nn.Linear(self._flat_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(self._flat_dim, cfg.latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(cfg.latent_dim, self._flat_dim)

        C_, H_, W_ = self._enc_out_shape[1], self._enc_out_shape[2], self._enc_out_shape[3]
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(C_, 2 * c, kernel_size=4, stride=2, padding=1),  # x2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2 * c, c, kernel_size=4, stride=2, padding=1),  # x4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, 1, kernel_size=4, stride=2, padding=1),  # x8
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        h = h.view(x.shape[0], -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # z = mu + std * eps
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z)
        # reshape back to conv feature map
        B = z.shape[0]
        C_, H_, W_ = self._enc_out_shape[1], self._enc_out_shape[2], self._enc_out_shape[3]
        h = h.view(B, C_, H_, W_)
        x_rec = self.dec(h)
        # crop/pad to exact input size (handles odd sizes)
        x_rec = self._match_size(x_rec, (self.input_h, self.input_w))
        return x_rec

    @staticmethod
    def _match_size(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        H, W = hw
        # crop if too large
        x = x[:, :, :H, :W]
        # pad if too small
        pad_h = max(0, H - x.shape[2])
        pad_w = max(0, W - x.shape[3])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decode(z)
        return x_rec, mu, logvar, z

    def loss(self, x: torch.Tensor, x_rec: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> dict:
        recon = F.mse_loss(x_rec, x, reduction="mean")
        # KL divergence
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon + self.cfg.beta * kl
        return {"total": total, "recon": recon, "kl": kl}
