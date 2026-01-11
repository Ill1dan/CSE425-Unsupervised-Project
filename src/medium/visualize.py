from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "medium"
LATENTS_DIR = RESULTS_DIR / "latents"
CLUSTER_DIR = RESULTS_DIR / "clustering"
VIZ_DIR = RESULTS_DIR / "visualizations"


def ensure_dirs():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)


def scatter_2d(Z: np.ndarray, labels: np.ndarray, title: str, save_path: Path):
    plt.figure(figsize=(7, 6))
    plt.scatter(Z[:, 0], Z[:, 1], c=labels, s=18, alpha=0.8)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def encode_languages(languages: np.ndarray) -> np.ndarray:
    # languages may be strings like 'en', 'bn' or ints
    if np.issubdtype(languages.dtype, np.number):
        return languages.astype(int)

    uniq = sorted(list(set(languages.tolist())))
    mapping = {u: i for i, u in enumerate(uniq)}
    return np.array([mapping[x] for x in languages], dtype=int)


def main():
    ensure_dirs()

    mu = np.load(LATENTS_DIR / "mu.npy")
    languages = np.load(LATENTS_DIR / "languages.npy", allow_pickle=True)
    lang_labels = encode_languages(languages)

    # choose a default clustering label to visualize
    cluster_path = CLUSTER_DIR / "kmeans_k2_labels.npy"
    if cluster_path.exists():
        cluster_labels = np.load(cluster_path)
    else:
        cluster_labels = None

    # PCA
    pca = PCA(n_components=2, random_state=42)
    Zp = pca.fit_transform(mu)

    scatter_2d(Zp, lang_labels, "PCA (colored by language)", VIZ_DIR / "pca_language.png")

    if cluster_labels is not None:
        scatter_2d(Zp, cluster_labels, "PCA (colored by clusters: kmeans k=2)", VIZ_DIR / "pca_kmeans_k2.png")

    # t-SNE (can be slow but ok for 200)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca", learning_rate="auto")
    Zt = tsne.fit_transform(mu)

    scatter_2d(Zt, lang_labels, "t-SNE (colored by language)", VIZ_DIR / "tsne_language.png")

    if cluster_labels is not None:
        scatter_2d(Zt, cluster_labels, "t-SNE (colored by clusters: kmeans k=2)", VIZ_DIR / "tsne_kmeans_k2.png")

    print("[INFO] Saved visualizations ->", VIZ_DIR)


if __name__ == "__main__":
    main()
