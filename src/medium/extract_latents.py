from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.medium.models.cnn_vae import CNNVAE, CNNVAEConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "features_mel"
RESULTS_DIR = PROJECT_ROOT / "results" / "medium"
MODELS_DIR = RESULTS_DIR / "models"
LATENTS_DIR = RESULTS_DIR / "latents"


def ensure_dirs():
    LATENTS_DIR.mkdir(parents=True, exist_ok=True)


def load_features_pack(features_dir: Path):
    X = np.load(features_dir / "features.npy", allow_pickle=False)
    ids = np.load(features_dir / "ids.npy", allow_pickle=True)
    languages = np.load(features_dir / "languages.npy", allow_pickle=True)
    return X, ids, languages


def infer_mel_reshape(X_flat: np.ndarray) -> tuple[int, int]:
    D = X_flat.shape[1]
    candidates = [128, 96, 80, 64, 60, 40]
    for n_mels in candidates:
        if D % n_mels == 0:
            T = D // n_mels
            if T >= 10:
                return n_mels, T
    n_mels = int(np.sqrt(D))
    while n_mels > 1 and D % n_mels != 0:
        n_mels -= 1
    T = D // n_mels
    return n_mels, T


def to_4d_mel(X: np.ndarray) -> np.ndarray:
    if X.ndim == 4:
        if X.shape[1] != 1:
            raise ValueError(f"Expected channel=1, got shape {X.shape}")
        X4 = X
    elif X.ndim == 3:
        X4 = X[:, None, :, :]
    elif X.ndim == 2:
        H, W = infer_mel_reshape(X)
        X4 = X.reshape(X.shape[0], 1, H, W)
    else:
        raise ValueError(f"Unsupported mel features shape: {X.shape}")
    return X4.astype(np.float32)


class MelDataset(Dataset):
    def __init__(self, X4: np.ndarray):
        self.X = torch.from_numpy(X4)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx]


def main():
    ensure_dirs()

    ckpt_path = MODELS_DIR / "cnn_vae.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}. Train first.")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = CNNVAEConfig(**ckpt["config"])
    input_hw = tuple(ckpt["input_hw"])
    mean = float(ckpt["norm_mean"])
    std = float(ckpt["norm_std"])

    X, ids, languages = load_features_pack(DATA_DIR)
    X4 = to_4d_mel(X)
    X4 = (X4 - mean) / (std + 1e-8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNVAE(input_hw=input_hw, cfg=cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dl = DataLoader(MelDataset(X4), batch_size=64, shuffle=False)

    mus = []
    logvars = []

    with torch.no_grad():
        for xb in dl:
            xb = xb.to(device)
            mu, logvar = model.encode(xb)
            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())

    mu_all = np.concatenate(mus, axis=0)
    logvar_all = np.concatenate(logvars, axis=0)

    np.save(LATENTS_DIR / "mu.npy", mu_all)
    np.save(LATENTS_DIR / "logvar.npy", logvar_all)
    np.save(LATENTS_DIR / "ids.npy", ids)
    np.save(LATENTS_DIR / "languages.npy", languages)

    print("[INFO] Saved latents ->", LATENTS_DIR)
    print("[INFO] mu shape:", mu_all.shape)


if __name__ == "__main__":
    main()
