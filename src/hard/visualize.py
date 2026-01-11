import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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


def plot_scatter(Z2, y=None, title="", out_path="plot.png"):
    plt.figure()
    if y is None:
        plt.scatter(Z2[:, 0], Z2[:, 1], s=12)
    else:
        # y can be strings or ints; map to numeric groups for coloring
        y_arr = np.array(y)
        uniq = list(dict.fromkeys(list(y_arr)))
        for u in uniq:
            m = (y_arr == u)
            plt.scatter(Z2[m, 0], Z2[m, 1], s=12, label=str(u))
        plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_spec_shape(spec_shape_str):
    # "128,173" -> (128,173)
    if spec_shape_str is None:
        return None
    parts = [p.strip() for p in spec_shape_str.split(",")]
    if len(parts) != 2:
        return None
    return (int(parts[0]), int(parts[1]))


@torch.no_grad()
def recon_examples(feature_pack, results_dir, beta, n_samples=3, spec_shape=None, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    pack = load_feature_pack(feature_pack)
    X, y, ids = unpack_feature_pack(pack)

    ckpt_path = os.path.join(results_dir, "models", f"beta_{beta}.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    model = BetaVAE(
        input_dim=ckpt["input_dim"],
        latent_dim=ckpt["latent_dim"],
        beta=ckpt["beta"],
        hidden_dims=(ckpt["h1"], ckpt["h2"]),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    out_dir = os.path.join(results_dir, "reconstructions")
    ensure_dir(out_dir)

    idxs = np.linspace(0, len(X) - 1, n_samples, dtype=int)

    for i, idx in enumerate(idxs, start=1):
        x = torch.tensor(X[idx:idx+1], dtype=torch.float32).to(device)
        recon, mu, logvar = model(x)
        x_np = x.detach().cpu().numpy().squeeze()
        r_np = recon.detach().cpu().numpy().squeeze()

        out_path = os.path.join(out_dir, f"beta_{beta}_sample_{i:02d}.png")

        plt.figure(figsize=(10, 4))
        if spec_shape is not None and np.prod(spec_shape) == x_np.shape[0]:
            x_img = x_np.reshape(spec_shape)
            r_img = r_np.reshape(spec_shape)
            plt.subplot(1, 2, 1)
            plt.imshow(x_img, aspect="auto")
            plt.title("Original")
            plt.subplot(1, 2, 2)
            plt.imshow(r_img, aspect="auto")
            plt.title("Reconstructed")
        else:
            # fallback: plot as 1D curves
            plt.plot(x_np, label="Original", linewidth=1)
            plt.plot(r_np, label="Reconstructed", linewidth=1)
            plt.legend()
            plt.title("Original vs Reconstructed (1D view)")

        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="results/hard")
    ap.add_argument("--feature_pack", type=str, default="data/features_mfcc_flat")
    ap.add_argument("--beta", type=float, default=4.0)

    ap.add_argument("--do_pca", action="store_true")
    ap.add_argument("--do_tsne", action="store_true")
    ap.add_argument("--do_recon", action="store_true")

    ap.add_argument("--tsne_perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--spec_shape", type=str, default=None)  # e.g. "128,173"
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    lat_dir = os.path.join(args.results_dir, "latents")
    vis_dir = os.path.join(args.results_dir, "visualizations")
    ensure_dir(vis_dir)

    mu = np.load(os.path.join(lat_dir, f"mu_beta_{args.beta}.npy"))
    y_path = os.path.join(lat_dir, f"y_beta_{args.beta}.npy")
    y = np.load(y_path, allow_pickle=True) if os.path.exists(y_path) else None

    if args.do_pca:
        Z2 = PCA(n_components=2, random_state=args.seed).fit_transform(mu)
        out_path = os.path.join(vis_dir, f"latent_pca_beta_{args.beta}.png")
        plot_scatter(Z2, y=y, title=f"PCA latent (beta={args.beta})", out_path=out_path)
        print("[INFO] Saved:", out_path)

    if args.do_tsne:
        Z2 = TSNE(
            n_components=2,
            random_state=args.seed,
            perplexity=args.tsne_perplexity,
            init="pca",
            learning_rate="auto",
        ).fit_transform(mu)
        out_path = os.path.join(vis_dir, f"latent_tsne_beta_{args.beta}.png")
        plot_scatter(Z2, y=y, title=f"t-SNE latent (beta={args.beta})", out_path=out_path)
        print("[INFO] Saved:", out_path)

    if args.do_recon:
        spec_shape = parse_spec_shape(args.spec_shape)
        recon_examples(
            feature_pack=args.feature_pack,
            results_dir=args.results_dir,
            beta=args.beta,
            n_samples=3,
            spec_shape=spec_shape,
            device=args.device,
        )
        print("[INFO] Saved reconstructions ->", os.path.join(args.results_dir, "reconstructions"))


if __name__ == "__main__":
    main()
