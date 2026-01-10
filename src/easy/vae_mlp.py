import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16, hidden_dims=(512, 256), dropout=0.0):
        super().__init__()

        # Encoder
        enc = []
        prev = input_dim
        for h in hidden_dims:
            enc.append(nn.Linear(prev, h))
            enc.append(nn.ReLU())
            if dropout and dropout > 0:
                enc.append(nn.Dropout(dropout))
            prev = h
        self.encoder = nn.Sequential(*enc)

        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        # Decoder
        dec = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec.append(nn.Linear(prev, h))
            dec.append(nn.ReLU())
            if dropout and dropout > 0:
                dec.append(nn.Dropout(dropout))
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def vae_loss(x, recon, mu, logvar, beta=0.0001, recon_type="mse"):
    if recon_type == "l1":
        recon_loss = F.l1_loss(recon, x, reduction="mean")
    else:
        recon_loss = F.mse_loss(recon, x, reduction="mean")

    # Standard VAE KL divergence
    # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar), dim=1))

    total = recon_loss + beta * kl
    return total, recon_loss, kl
