from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score


def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    # silhouette needs >= 2 clusters, and no cluster containing all points
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return float("nan")
    # also fails if a cluster has 1 sample in some implementations; but usually ok
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return float("nan")


def safe_ch(X: np.ndarray, labels: np.ndarray) -> float:
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return float("nan")
    try:
        return float(calinski_harabasz_score(X, labels))
    except Exception:
        return float("nan")


def safe_dbi(X: np.ndarray, labels: np.ndarray) -> float:
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return float("nan")
    try:
        return float(davies_bouldin_score(X, labels))
    except Exception:
        return float("nan")


def safe_ari(labels: np.ndarray, languages: np.ndarray) -> float:
    """
    ARI compares cluster labels with known categories (language).
    languages can be strings or ints.
    """
    try:
        return float(adjusted_rand_score(languages, labels))
    except Exception:
        return float("nan")


def compute_metrics(X: np.ndarray, labels: np.ndarray, languages: np.ndarray) -> dict:
    return {
        "silhouette": safe_silhouette(X, labels),
        "calinski_harabasz": safe_ch(X, labels),
        "davies_bouldin": safe_dbi(X, labels),  # lower better
        "ari": safe_ari(labels, languages),
    }
