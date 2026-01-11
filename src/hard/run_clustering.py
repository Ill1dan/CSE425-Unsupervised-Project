import os
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering

from .evaluation import clustering_metrics


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def append_csv(path, row: dict):
    if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
        pd.DataFrame([row]).to_csv(path, index=False)
    else:
        old = pd.read_csv(path)
        new = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
        new.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="results/hard")
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    lat_dir = os.path.join(args.results_dir, "latents")
    mu_path = os.path.join(lat_dir, f"mu_beta_{args.beta}.npy")
    y_path = os.path.join(lat_dir, f"y_beta_{args.beta}.npy")

    Z = np.load(mu_path)
    y_true = np.load(y_path, allow_pickle=True) if os.path.exists(y_path) else None


    out_dir = os.path.join(args.results_dir, "clustering")
    met_dir = os.path.join(args.results_dir, "metrics")
    ensure_dir(out_dir)
    ensure_dir(met_dir)

    # KMeans
    km = KMeans(n_clusters=args.k, random_state=args.seed, n_init="auto")
    y_km = km.fit_predict(Z)
    m_km = clustering_metrics(Z, y_km, y_true=y_true)
    row_km = {"beta": args.beta, "method": "kmeans", "k": args.k, **m_km}
    append_csv(os.path.join(met_dir, "clustering_metrics_beta.csv"), row_km)
    np.save(os.path.join(out_dir, f"labels_kmeans_beta_{args.beta}.npy"), y_km)
    print(f"[KMeans k={args.k}] {m_km}")

    # Agglomerative
    ag = AgglomerativeClustering(n_clusters=args.k)
    y_ag = ag.fit_predict(Z)
    m_ag = clustering_metrics(Z, y_ag, y_true=y_true)
    row_ag = {"beta": args.beta, "method": "agglomerative", "k": args.k, **m_ag}
    append_csv(os.path.join(met_dir, "clustering_metrics_beta.csv"), row_ag)
    np.save(os.path.join(out_dir, f"labels_agglomerative_beta_{args.beta}.npy"), y_ag)
    print(f"[Agglo k={args.k}] {m_ag}")

    print("[INFO] Saved clustering outputs ->", out_dir)
    print("[INFO] Saved metrics ->", os.path.join(met_dir, "clustering_metrics_beta.csv"))


if __name__ == "__main__":
    main()
