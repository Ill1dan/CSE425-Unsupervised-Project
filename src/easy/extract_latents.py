import os
import json
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset

from .vae_mlp import MLPVAE


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_features_flat():
    root = project_root()
    feat_dir = os.path.join(root, "data", "features_mfcc_flat")

    X = np.load(os.path.join(feat_dir, "features.npy")).astype(np.float32)
    ids_path = os.path.join(feat_dir, "ids.npy")
    lang_path = os.path.join(feat_dir, "languages.npy")

    ids = np.load(ids_path, allow_pickle=True) if os.path.exists(ids_path) else None
    languages = np.load(lang_path, allow_pickle=True) if os.path.exists(lang_path) else None


    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)

    return X, ids, languages


def main():
    root = project_root()

    models_dir = os.path.join(root, "results", "easy", "models")
    latents_dir = os.path.join(root, "results", "easy", "latents")
    ensure_dir(latents_dir)

    ckpt_path = os.path.join(models_dir, "vae_mlp.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("Missing model checkpoint: " + ckpt_path + " (run train_vae first)")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]

    X, ids, languages = load_features_flat()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MLPVAE(
        input_dim=cfg["input_dim"],
        latent_dim=cfg["latent_dim"],
        hidden_dims=tuple(cfg["hidden_dims"]),
        dropout=cfg.get("dropout", 0.0)
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=256, shuffle=False)

    all_mu = []
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            _, mu, _ = model(xb)
            all_mu.append(mu.detach().cpu().numpy())

    mu_np = np.concatenate(all_mu, axis=0)

    mu_path = os.path.join(latents_dir, "latent_mu.npy")
    np.save(mu_path, mu_np)
    print("Saved latent mu ->", mu_path)

    # Save meta info aligned with rows
    meta = pd.DataFrame({
        "index": np.arange(len(mu_np)),
        "id": ids
    })
    if languages is not None:
        meta["language"] = languages

    meta_path = os.path.join(latents_dir, "meta.csv")
    meta.to_csv(meta_path, index=False)
    print("Saved meta ->", meta_path)


if __name__ == "__main__":
    main()
