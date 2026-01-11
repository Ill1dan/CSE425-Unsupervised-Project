from __future__ import annotations

import time
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.medium.models.cnn_vae import CNNVAE, CNNVAEConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "features_mel"
RESULTS_DIR = PROJECT_ROOT / "results" / "medium"

MODELS_DIR = RESULTS_DIR / "models"
LOGS_DIR = RESULTS_DIR / "logs"


def ensure_dirs():
    for p in [RESULTS_DIR, MODELS_DIR, LOGS_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def load_features_pack(features_dir: Path):
    X = np.load(features_dir / "features.npy", allow_pickle=False)
    ids = np.load(features_dir / "ids.npy", allow_pickle=True)
    languages = np.load(features_dir / "languages.npy", allow_pickle=True)
    return X, ids, languages


def pick_device() -> torch.device:
    # CUDA (NVIDIA)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon (harmless on Windows)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_report(device: torch.device):
    print(f"[INFO] Device: {device}")
    if device.type == "cuda":
        print(f"[INFO] CUDA runtime: {torch.version.cuda}")
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        # small speed boost for conv-heavy models
        torch.backends.cudnn.benchmark = True


def infer_mel_reshape(X_flat: np.ndarray) -> tuple[int, int]:
    """
    If X is (N, D), try to infer (n_mels, time) using common n_mels values.
    """
    D = X_flat.shape[1]
    candidates = [128, 96, 80, 64, 60, 40]
    for n_mels in candidates:
        if D % n_mels == 0:
            T = D // n_mels
            if T >= 10:
                return n_mels, T

    # fallback: find a divisor near sqrt(D)
    n_mels = int(np.sqrt(D))
    while n_mels > 1 and D % n_mels != 0:
        n_mels -= 1
    T = D // n_mels
    return n_mels, T


def to_4d_mel(X: np.ndarray) -> np.ndarray:
    """
    Convert mel features into (N, 1, H, W) float32.
    Supports:
      (N, H, W)
      (N, 1, H, W)
      (N, D) flattened
    """
    if X.ndim == 4:
        # assume (N, C, H, W)
        if X.shape[1] != 1:
            raise ValueError(f"Expected channel=1, got shape {X.shape}")
        X4 = X
    elif X.ndim == 3:
        # (N, H, W) -> add channel
        X4 = X[:, None, :, :]
    elif X.ndim == 2:
        # (N, D) -> reshape into (N, 1, H, W)
        H, W = infer_mel_reshape(X)
        X4 = X.reshape(X.shape[0], 1, H, W)
    else:
        raise ValueError(f"Unsupported mel features shape: {X.shape}")

    return X4.astype(np.float32)


def normalize_global(X4: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Global mean/std normalization (simple + works well for CNN input).
    """
    mean = float(X4.mean())
    std = float(X4.std() + 1e-8)
    Xn = (X4 - mean) / std
    return Xn, mean, std


class MelDataset(Dataset):
    def __init__(self, X4: np.ndarray):
        self.X = torch.from_numpy(X4)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx]


def train():
    ensure_dirs()

    # -------------------------
    # Hyperparams (Medium)
    # -------------------------
    latent_dim = 32          # was 16 (better capacity for clustering)
    beta = 1.0               # target KL weight
    warmup_epochs = 10       # KL warmup: beta ramps 0 -> 1 during first N epochs
    base_channels = 32
    batch_size = 32
    lr = 1e-3
    epochs = 60              # was 40
    seed = 42

    np.random.seed(seed)
    torch.manual_seed(seed)

    X, ids, languages = load_features_pack(DATA_DIR)
    X4 = to_4d_mel(X)
    X4, mean, std = normalize_global(X4)

    N, C, H, W = X4.shape
    print(f"[INFO] Loaded mel features: {X.shape} -> {X4.shape} (normalized: mean={mean:.4f}, std={std:.4f})")

    device = pick_device()
    device_report(device)

    ds = MelDataset(X4)

    pin = (device.type == "cuda")
    # NOTE: If Windows gives DataLoader worker issues, set num_workers=0
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=2,
        pin_memory=pin,
    )

    cfg = CNNVAEConfig(latent_dim=latent_dim, base_channels=base_channels, beta=beta)
    model = CNNVAE(input_hw=(H, W), cfg=cfg).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # Log file
    log_path = LOGS_DIR / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "beta", "loss_total", "loss_recon", "loss_kl", "seconds"])

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        # KL warmup
        beta_now = beta * min(1.0, epoch / max(1, warmup_epochs))
        model.cfg.beta = beta_now

        tot_total = 0.0
        tot_recon = 0.0
        tot_kl = 0.0
        count = 0

        for xb in dl:
            xb = xb.to(device, non_blocking=pin)

            x_rec, mu, logvar, z = model(xb)
            losses = model.loss(xb, x_rec, mu, logvar)

            optim.zero_grad(set_to_none=True)
            losses["total"].backward()
            optim.step()

            bsz = xb.shape[0]
            tot_total += float(losses["total"].item()) * bsz
            tot_recon += float(losses["recon"].item()) * bsz
            tot_kl += float(losses["kl"].item()) * bsz
            count += bsz

        sec = time.time() - t0
        avg_total = tot_total / count
        avg_recon = tot_recon / count
        avg_kl = tot_kl / count

        print(
            f"[E{epoch:03d}] beta={beta_now:.2f} total={avg_total:.5f} recon={avg_recon:.5f} kl={avg_kl:.5f} ({sec:.1f}s)"
        )

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, beta_now, avg_total, avg_recon, avg_kl, sec])

    # Save model checkpoint
    ckpt = {
        "model_state": model.state_dict(),
        "config": cfg.__dict__,           # includes latent_dim/base_channels/beta
        "input_hw": (H, W),
        "norm_mean": mean,
        "norm_std": std,
    }
    out_path = MODELS_DIR / "cnn_vae.pt"
    torch.save(ckpt, out_path)
    print("[INFO] Saved model ->", out_path)


if __name__ == "__main__":
    train()