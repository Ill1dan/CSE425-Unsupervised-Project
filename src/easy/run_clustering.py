import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    root = project_root()

    latents_dir = os.path.join(root, "results", "easy", "latents")
    clustering_dir = os.path.join(root, "results", "easy", "clustering")
    viz_dir = os.path.join(root, "results", "easy", "latent_visualization")

    ensure_dir(clustering_dir)
    ensure_dir(viz_dir)

    mu_path = os.path.join(latents_dir, "latent_mu.npy")
    meta_path = os.path.join(latents_dir, "meta.csv")

    if not os.path.exists(mu_path):
        raise FileNotFoundError("Missing: " + mu_path + " (run extract_latents first)")

    Z = np.load(mu_path)
    meta = pd.read_csv(meta_path) if os.path.exists(meta_path) else None

    n = Z.shape[0]
    d = Z.shape[1]
    print(f"[INFO] Loaded latent mu: shape={Z.shape}")

    # ---- KMeans ----
    k = 2
    print(f"[INFO] Running KMeans (k={k})...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Z)

    # ---- Metrics (required) ----
    print("[INFO] Computing Silhouette + CH...")
    sil = silhouette_score(Z, labels)
    ch = calinski_harabasz_score(Z, labels)

    metrics_row = {"k": k, "silhouette": float(sil), "calinski_harabasz": float(ch)}
    metrics_path = os.path.join(clustering_dir, "clustering_metrics_k2.csv")
    pd.DataFrame([metrics_row]).to_csv(metrics_path, index=False)
    print("Saved metrics ->", metrics_path)
    print("Silhouette:", sil)
    print("Calinski-Harabasz:", ch)

    # Save labels
    labels_path = os.path.join(clustering_dir, "kmeans_k2_labels.npy")
    np.save(labels_path, labels)
    print("Saved labels ->", labels_path)

    # Save assignments
    if meta is not None:
        out = meta.copy()
        out["cluster"] = labels
        assign_path = os.path.join(clustering_dir, "kmeans_k2_assignments.csv")
        out.to_csv(assign_path, index=False)
        print("Saved assignments ->", assign_path)

    # ---- Visualization: PCA first (fast) ----
    print("[INFO] Making PCA plot (fast sanity check)...")
    pca2 = PCA(n_components=2, random_state=42)
    Zp = pca2.fit_transform(Z)

    plt.figure()
    plt.scatter(Zp[:, 0], Zp[:, 1], c=labels, s=12)
    plt.title("PCA on VAE latent μ (KMeans k=2)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    pca_path = os.path.join(viz_dir, "pca_k2.png")
    plt.savefig(pca_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved PCA plot ->", pca_path)

    # ---- Visualization: t-SNE (can be slow) ----
    # Safe perplexity: must be < n
    # Rule of thumb: 5 to 50, and also < (n-1)/3 is often safer
    perp = 30
    perp = min(perp, max(5, (n - 1) // 3))
    perp = min(perp, n - 1)

    print(f"[INFO] Running t-SNE (perplexity={perp})... this may take a bit")
    t0 = time.time()

    # Speed trick: reduce dimension before TSNE
    Z50 = Z
    if d > 50:
        Z50 = PCA(n_components=50, random_state=42).fit_transform(Z)

    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=42,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        verbose=1
    )
    Zt = tsne.fit_transform(Z50)

    print(f"[INFO] t-SNE finished in {time.time() - t0:.1f}s")

    plt.figure()
    plt.scatter(Zt[:, 0], Zt[:, 1], c=labels, s=12)
    plt.title("t-SNE on VAE latent μ (KMeans k=2)")
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    tsne_path = os.path.join(viz_dir, "tsne_k2.png")
    plt.savefig(tsne_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved t-SNE plot ->", tsne_path)


if __name__ == "__main__":
    main()
