import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


def purity_score(y_true, y_pred):
    """
    Purity = sum(max contingency column) / N
    Works for any discrete y_true.
    """
    ct = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(y_pred, name="pred"))
    return float(np.sum(np.max(ct.values, axis=0)) / np.sum(ct.values))


def clustering_metrics(Z, labels, y_true=None):
    """
    Z: (N, d) embeddings
    labels: predicted cluster labels
    y_true: optional ground truth labels (language)
    """
    out = {
        "silhouette": float(silhouette_score(Z, labels)) if len(set(labels)) > 1 else None,
        "calinski_harabasz": float(calinski_harabasz_score(Z, labels)) if len(set(labels)) > 1 else None,
        "davies_bouldin": float(davies_bouldin_score(Z, labels)) if len(set(labels)) > 1 else None,
    }

    if y_true is not None:
        out["ari"] = float(adjusted_rand_score(y_true, labels))
        out["nmi"] = float(normalized_mutual_info_score(y_true, labels))
        out["purity"] = float(purity_score(y_true, labels))
    else:
        out["ari"] = None
        out["nmi"] = None
        out["purity"] = None

    return out
