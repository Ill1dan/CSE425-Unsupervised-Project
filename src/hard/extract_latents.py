import os
import argparse
import numpy as np
import torch

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


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_pack", type=str, default="data/features_mfcc_flat")
    ap.add_argument("--results_dir", type=str, default="results/hard")
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    pack = load_feature_pack(args.feature_pack)
    X, y, ids = unpack_feature_pack(pack)

    model_path = os.path.join(args.results_dir, "models", f"beta_{args.beta}.pt")
    ckpt = torch.load(model_path, map_location=device)

    model = BetaVAE(
        input_dim=ckpt["input_dim"],
        latent_dim=ckpt["latent_dim"],
        beta=ckpt["beta"],
        hidden_dims=(ckpt["h1"], ckpt["h2"]),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    mu, logvar = model.encode(X_t)
    mu = mu.detach().cpu().numpy()
    logvar = logvar.detach().cpu().numpy()

    out_dir = os.path.join(args.results_dir, "latents")
    ensure_dir(out_dir)

    np.save(os.path.join(out_dir, f"mu_beta_{args.beta}.npy"), mu)
    np.save(os.path.join(out_dir, f"logvar_beta_{args.beta}.npy"), logvar)

    if y is not None:
        np.save(os.path.join(out_dir, f"y_beta_{args.beta}.npy"), np.array(y))
    if ids is not None:
        np.save(os.path.join(out_dir, f"ids_beta_{args.beta}.npy"), np.array(ids))

    print("[INFO] Saved latents ->", out_dir)
    print("[INFO] mu shape:", mu.shape)


if __name__ == "__main__":
    main()
