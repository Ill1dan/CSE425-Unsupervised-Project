import os
import json
import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import load_feature_pack
from .models.beta_vae import BetaVAE


def unpack_feature_pack(pack):
    if isinstance(pack, dict):
        X = pack["X"]
        y = pack.get("y", None)
        ids = pack.get("ids", None)
        return X, y, ids

    if isinstance(pack, (tuple, list)):
        if len(pack) == 1:
            return pack[0], None, None
        if len(pack) == 2:
            return pack[0], pack[1], None

        # len == 3: (X, a, b) where a/b could be ids or y
        X, a, b = pack[0], pack[1], pack[2]

        def looks_like_label(arr):
            try:
                arr = np.array(arr)
                uniq = np.unique(arr)
                # language labels usually have very small unique set (2 or a few)
                return len(uniq) <= 5
            except Exception:
                return False

        if looks_like_label(a) and not looks_like_label(b):
            y, ids = a, b
        elif looks_like_label(b) and not looks_like_label(a):
            y, ids = b, a
        else:
            # fallback: assume old order
            y, ids = a, b

        return X, y, ids

    raise TypeError(f"Unsupported feature pack type: {type(pack)}")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_beta(X: np.ndarray, out_dir: str, beta: float, args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = BetaVAE(
        input_dim=X.shape[1],
        latent_dim=args.latent_dim,
        beta=beta,
        hidden_dims=(args.h1, args.h2),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    X_t = torch.tensor(X, dtype=torch.float32)
    ds = TensorDataset(X_t)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    log_rows = []
    model.train()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0

        for (xb,) in dl:
            xb = xb.to(device)
            recon, mu, logvar = model(xb)
            loss, recon_loss, kl = model.loss(recon, xb, mu, logvar)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.detach().cpu().item())
            total_recon += float(recon_loss.detach().cpu().item())
            total_kl += float(kl.detach().cpu().item())
            n_batches += 1

        row = {
            "epoch": epoch,
            "beta": beta,
            "loss": total_loss / max(1, n_batches),
            "recon": total_recon / max(1, n_batches),
            "kl": total_kl / max(1, n_batches),
        }
        log_rows.append(row)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"[beta={beta}] epoch {epoch:03d} | loss={row['loss']:.6f} recon={row['recon']:.6f} kl={row['kl']:.6f}")

    # Save
    models_dir = os.path.join(out_dir, "models")
    logs_dir = os.path.join(out_dir, "logs")
    ensure_dir(models_dir)
    ensure_dir(logs_dir)

    model_path = os.path.join(models_dir, f"beta_{beta}.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": X.shape[1],
            "latent_dim": args.latent_dim,
            "beta": beta,
            "h1": args.h1,
            "h2": args.h2,
        },
        model_path,
    )

    log_path = os.path.join(logs_dir, f"train_log_beta_{beta}.csv")
    pd.DataFrame(log_rows).to_csv(log_path, index=False)

    # Save config snapshot
    cfg_path = os.path.join(logs_dir, f"config_beta_{beta}.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("[INFO] Saved model ->", model_path)
    print("[INFO] Saved log   ->", log_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_pack", type=str, default="data/features_mfcc_flat")
    ap.add_argument("--results_dir", type=str, default="results/hard")
    ap.add_argument("--betas", type=float, nargs="+", default=[1, 2, 4, 6])
    ap.add_argument("--latent_dim", type=int, default=32)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--h1", type=int, default=512)
    ap.add_argument("--h2", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.results_dir)

    pack = load_feature_pack(args.feature_pack)
    X, y, ids = unpack_feature_pack(pack)

    print("[INFO] Loaded feature pack:", args.feature_pack, "X:", X.shape)
    for beta in args.betas:
        train_one_beta(X, args.results_dir, beta, args)


if __name__ == "__main__":
    main()
