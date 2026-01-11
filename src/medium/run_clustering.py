from __future__ import annotations

from pathlib import Path
import csv
import numpy as np

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

from src.medium.evaluation import compute_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "medium"
LATENTS_DIR = RESULTS_DIR / "latents"
CLUSTER_DIR = RESULTS_DIR / "clustering"
METRICS_DIR = RESULTS_DIR / "metrics"


def ensure_dirs():
    for p in [CLUSTER_DIR, METRICS_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def append_metrics(csv_path: Path, row: dict):
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    ensure_dirs()

    mu = np.load(LATENTS_DIR / "mu.npy")
    languages = np.load(LATENTS_DIR / "languages.npy", allow_pickle=True)

    print("[INFO] Loaded latents mu:", mu.shape)

    out_csv = METRICS_DIR / "clustering_metrics.csv"

    feature_set = "mel"
    model_name = "cnn_vae"
    latent_dim = mu.shape[1]

    # 1) KMeans + Agglo
    ks = [2, 3, 4, 5]
    for k in ks:
        # KMeans
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(mu)
        np.save(CLUSTER_DIR / f"kmeans_k{k}_labels.npy", labels)

        m = compute_metrics(mu, labels, languages)
        row = {
            "feature_set": feature_set,
            "model": model_name,
            "latent_dim": latent_dim,
            "cluster_algo": "kmeans",
            "k": k,
            "eps": "",
            **m,
        }
        append_metrics(out_csv, row)
        print(f"[KMeans k={k}]", m)

        # Agglomerative
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = agg.fit_predict(mu)
        np.save(CLUSTER_DIR / f"agglo_k{k}_labels.npy", labels)

        m = compute_metrics(mu, labels, languages)
        row = {
            "feature_set": feature_set,
            "model": model_name,
            "latent_dim": latent_dim,
            "cluster_algo": "agglo",
            "k": k,
            "eps": "",
            **m,
        }
        append_metrics(out_csv, row)
        print(f"[Agglo k={k}]", m)

    # 2) DBSCAN sweep
    eps_list = [0.5, 1.0, 1.5, 2.0]
    min_samples = 5
    for eps in eps_list:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(mu)
        np.save(CLUSTER_DIR / f"dbscan_eps{eps}_labels.npy", labels)

        m = compute_metrics(mu, labels, languages)
        row = {
            "feature_set": feature_set,
            "model": model_name,
            "latent_dim": latent_dim,
            "cluster_algo": "dbscan",
            "k": "",
            "eps": eps,
            **m,
        }
        append_metrics(out_csv, row)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"[DBSCAN eps={eps}] clusters={n_clusters} metrics={m}")

    print("[INFO] Metrics saved ->", out_csv)
    print("[INFO] Labels saved ->", CLUSTER_DIR)


if __name__ == "__main__":
    main()
