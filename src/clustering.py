import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from dataset import load_feature_pack


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def append_metrics(results_dir, row):
    path = os.path.join(results_dir, "clustering_metrics.csv")

    # If file doesn't exist OR is empty -> create fresh with header
    if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
        pd.DataFrame([row]).to_csv(path, index=False)
        print("Saved metrics ->", path)
        return

    # Otherwise append
    old = pd.read_csv(path)
    new = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    new.to_csv(path, index=False)
    print("Saved metrics ->", path)



def plot_2d(Z, labels, out_path, title):
    # Ensure the parent folder exists before saving
    out_dir = os.path.dirname(out_path)
    if out_dir and (not os.path.exists(out_dir)):
        os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(9, 7))
    plt.scatter(Z[:, 0], Z[:, 1], c=labels, s=18)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved plot ->", out_path)



def run_baseline(features_dir="data/features_mfcc_flat", results_dir="results/baseline", pca_dim=64, k=2, seed=42):
    ensure_dir(results_dir)
    vis_dir = os.path.join(results_dir, "latent_visualization")
    ensure_dir(vis_dir)

    # 1) Load final feature matrix
    X, ids, langs = load_feature_pack(features_dir)
    print("Loaded:", X.shape)

    # 2) PCA
    pca = PCA(n_components=pca_dim, random_state=seed)
    X_pca = pca.fit_transform(X)
    explained = float(np.sum(pca.explained_variance_ratio_))
    print("PCA:", X_pca.shape, "explained:", explained)

    # 3) KMeans
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(X_pca)
    print("KMeans done. k =", k)

    # 4) Metrics
    sil = silhouette_score(X_pca, labels)
    ch = calinski_harabasz_score(X_pca, labels)
    print("Silhouette:", sil)
    print("Calinski-Harabasz:", ch)

    append_metrics(results_dir, {
        "method": "baseline_kmeans",
        "features": os.path.basename(features_dir),
        "pca_dim": pca_dim,
        "k": k,
        "silhouette": sil,
        "calinski_harabasz": ch,
        "pca_explained_var_sum": explained
    })

    # Save per-song cluster assignments (helpful)
    pd.DataFrame({"id": ids, "language": langs, "cluster": labels}).to_csv(
        os.path.join(results_dir, "baseline_cluster_assignments.csv"),
        index=False
    )

    # 5) t-SNE plot
    N = X_pca.shape[0]
    perplexity = min(30, max(5, N // 4))
    perplexity = min(perplexity, N - 1)

    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca", learning_rate="auto")
    Z = tsne.fit_transform(X_pca)

    out_png = os.path.join(vis_dir, "baseline_tsne_pca{}_k{}.png".format(pca_dim, k))
    plot_2d(Z, labels, out_png, "Baseline t-SNE (PCA {} -> KMeans k={})".format(pca_dim, k))
    print("Saved plot:", out_png)


if __name__ == "__main__":
    run_baseline(
        features_dir="data/features_mfcc_flat",
        results_dir="results/baseline",
        pca_dim=64,
        k=2
    )
