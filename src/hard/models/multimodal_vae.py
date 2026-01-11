import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalVAE(nn.Module):
    """
    Multimodal VAE (optional stretch):
    - Audio branch: flattened audio features
    - Text branch: TF-IDF / bag-of-words lyrics vectors
    - Shared latent: z
    Loss: audio recon + text recon + beta*KL

    You can skip training this if you don't have text vectors ready.
    """

    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        latent_dim: int = 32,
        beta: float = 1.0,
        hidden_audio=(512, 256),
        hidden_text=(512, 256),
        hidden_joint=256,
    ):
        super().__init__()
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.latent_dim = latent_dim
        self.beta = float(beta)

        # Audio encoder
        self.enc_audio = nn.Sequential(
            nn.Linear(audio_dim, hidden_audio[0]),
            nn.ReLU(),
            nn.Linear(hidden_audio[0], hidden_audio[1]),
            nn.ReLU(),
        )

        # Text encoder
        self.enc_text = nn.Sequential(
            nn.Linear(text_dim, hidden_text[0]),
            nn.ReLU(),
            nn.Linear(hidden_text[0], hidden_text[1]),
            nn.ReLU(),
        )

        # Joint encoder head
        self.enc_joint = nn.Sequential(
            nn.Linear(hidden_audio[1] + hidden_text[1], hidden_joint),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_joint, latent_dim)
        self.fc_logvar = nn.Linear(hidden_joint, latent_dim)

        # Audio decoder
        self.dec_audio = nn.Sequential(
            nn.Linear(latent_dim, hidden_audio[1]),
            nn.ReLU(),
            nn.Linear(hidden_audio[1], hidden_audio[0]),
            nn.ReLU(),
            nn.Linear(hidden_audio[0], audio_dim),
        )

        # Text decoder
        self.dec_text = nn.Sequential(
            nn.Linear(latent_dim, hidden_text[1]),
            nn.ReLU(),
            nn.Linear(hidden_text[1], hidden_text[0]),
            nn.ReLU(),
            nn.Linear(hidden_text[0], text_dim),
        )

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_audio, x_text):
        ha = self.enc_audio(x_audio)
        ht = self.enc_text(x_text)
        h = self.enc_joint(torch.cat([ha, ht], dim=1))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        z = self.reparameterize(mu, logvar)

        recon_audio = self.dec_audio(z)
        recon_text = self.dec_text(z)
        return recon_audio, recon_text, mu, logvar

    def loss(self, recon_audio, x_audio, recon_text, x_text, mu, logvar, w_text=1.0):
        audio_loss = F.mse_loss(recon_audio, x_audio, reduction="mean")
        text_loss = F.mse_loss(recon_text, x_text, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = audio_loss + (w_text * text_loss) + self.beta * kl
        return total, audio_loss, text_loss, kl
