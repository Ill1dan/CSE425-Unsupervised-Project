import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset

from .vae_mlp import MLPVAE, vae_loss


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def project_root():
    # This file is src/easy/train_vae.py -> root is 2 levels up
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_features_flat():
    root = project_root()
    feat_dir = os.path.join(root, "data", "features_mfcc_flat")

    X_path = os.path.join(feat_dir, "features.npy")
    ids_path = os.path.join(feat_dir, "ids.npy")
    lang_path = os.path.join(feat_dir, "languages.npy")

    if not os.path.exists(X_path):
        raise FileNotFoundError("Missing: " + X_path + " (expected data/features_flat/features.npy)")

    X = np.load(X_path)
    ids = np.load(ids_path, allow_pickle=True) if os.path.exists(ids_path) else None
    languages = np.load(lang_path, allow_pickle=True) if os.path.exists(lang_path) else None


    # Ensure float32
    X = X.astype(np.float32)

    return X, ids, languages


def main():
    root = project_root()

    # Output dirs (Phase 4 Easy)
    results_dir = os.path.join(root, "results", "easy")
    models_dir = os.path.join(results_dir, "models")
    logs_dir = os.path.join(results_dir, "logs")
    plots_dir = os.path.join(results_dir, "latent_visualization")

    ensure_dir(models_dir)
    ensure_dir(logs_dir)
    ensure_dir(plots_dir)

    # Load data
    X, ids, languages = load_features_flat()

    # Flatten if needed (safe)
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)

    input_dim = X.shape[1]

    # Hyperparams (easy baseline)
    cfg = {
        "input_dim": int(input_dim),
        "latent_dim": 16,
        "hidden_dims": [512, 256],
        "dropout": 0.0,
        "batch_size": 64,
        "epochs": 50,
        "lr": 1e-3,
        "beta": 0.0001,
        "recon_type": "mse",  # or "l1"
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42
    }

    # Reproducibility
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # Dataloader
    X_tensor = torch.from_numpy(X)
    ds = TensorDataset(X_tensor)
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=False)

    # Model
    model = MLPVAE(
        input_dim=cfg["input_dim"],
        latent_dim=cfg["latent_dim"],
        hidden_dims=tuple(cfg["hidden_dims"]),
        dropout=cfg["dropout"]
    ).to(cfg["device"])

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # Train loop
    history = []
    model.train()

    for epoch in range(1, cfg["epochs"] + 1):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0

        for (xb,) in dl:
            xb = xb.to(cfg["device"])

            recon, mu, logvar = model(xb)
            loss, recon_loss, kl = vae_loss(
                xb, recon, mu, logvar,
                beta=cfg["beta"],
                recon_type=cfg["recon_type"]
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            total_recon += float(recon_loss.item())
            total_kl += float(kl.item())
            n_batches += 1

        row = {
            "epoch": epoch,
            "loss": total_loss / max(1, n_batches),
            "recon": total_recon / max(1, n_batches),
            "kl": total_kl / max(1, n_batches)
        }
        history.append(row)

        if epoch == 1 or epoch % 5 == 0:
            print("Epoch %d/%d | loss=%.6f recon=%.6f kl=%.6f" %
                  (epoch, cfg["epochs"], row["loss"], row["recon"], row["kl"]))

    # Save logs
    log_path = os.path.join(logs_dir, "vae_train_log.csv")
    pd.DataFrame(history).to_csv(log_path, index=False)
    print("Saved training log ->", log_path)

    # Save config
    cfg_path = os.path.join(models_dir, "vae_config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("Saved config ->", cfg_path)

    # Save model
    ckpt_path = os.path.join(models_dir, "vae_mlp.pt")
    torch.save({"state_dict": model.state_dict(), "config": cfg}, ckpt_path)
    print("Saved model ->", ckpt_path)

    # Plot losses
    df = pd.DataFrame(history)

    plt.figure()
    plt.plot(df["epoch"], df["loss"], label="total")
    plt.plot(df["epoch"], df["recon"], label="recon")
    plt.plot(df["epoch"], df["kl"], label="kl")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("VAE training losses")
    plot_path = os.path.join(plots_dir, "vae_loss_curves.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved loss plot ->", plot_path)


if __name__ == "__main__":
    main()
