
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt

# ------------------------------
# Utilities
# ------------------------------

ID_COLS_DEFAULT = ["player_name", "club", "position"]

def is_goalkeeper_label(x: str) -> bool:
    if not isinstance(x, str):
        return False
    x_low = x.lower()
    return ("gk" in x_low) or ("keeper" in x_low) or ("goalkeeper" in x_low)

def split_goalkeepers(df: pd.DataFrame, position_col: str = "position") -> Tuple[pd.DataFrame, pd.DataFrame]:
    if position_col not in df.columns:
        # If position not present, assume no GK separation possible
        return df.copy(), df.iloc[0:0].copy()
    mask_gk = df[position_col].apply(is_goalkeeper_label)
    return df[~mask_gk].copy(), df[mask_gk].copy()

def keep_numeric_features(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> pd.DataFrame:
    exclude = exclude or []
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    for col in exclude:
        if col in numeric_df.columns:
            numeric_df.drop(columns=[col], inplace=True)
    # Remove columns entirely NA or constant
    numeric_df = numeric_df.loc[:, numeric_df.notna().any(axis=0)]
    if numeric_df.shape[1] == 0:
        return numeric_df
    vt = VarianceThreshold(threshold=0.0)
    try:
        _ = vt.fit(numeric_df.fillna(0))
        numeric_df = numeric_df.loc[:, vt.get_support()]
    except Exception:
        # Fallback: return as-is if VT fails
        pass
    return numeric_df

def drop_highly_correlated(X: pd.DataFrame, threshold: float = 0.90) -> Tuple[pd.DataFrame, List[str]]:
    """Greedy removal of features with abs(corr) > threshold."""
    if X.shape[1] <= 1:
        return X, list(X.columns)
    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    kept = [c for c in X.columns if c not in to_drop]
    return X[kept].copy(), kept

def scale_features(X: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.fillna(0))
    return Xs, scaler

def find_best_k_silhouette(X: np.ndarray, k_range: range) -> Tuple[int, Dict[int, float]]:
    scores = {}
    best_k = None
    best_score = -1.0
    for k in k_range:
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(X)
        # Silhouette needs at least 2 labels
        if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
            continue
        score = silhouette_score(X, labels)
        scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k
    if best_k is None:
        # Fallback to 3 clusters if everything fails
        best_k = 3
    return best_k, scores

def run_kmeans(X: np.ndarray, n_clusters: int) -> KMeans:
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    km.fit(X)
    return km

def centroid_top_features(km: KMeans, feature_names: List[str], topn: int = 8) -> Dict[int, List[Tuple[str, float]]]:
    """Return top features by absolute centroid value for each cluster."""
    centroids = km.cluster_centers_
    results = {}
    for i, c in enumerate(centroids):
        idx = np.argsort(np.abs(c))[::-1][:topn]
        results[i] = [(feature_names[j], float(c[j])) for j in idx]
    return results

def plot_metric_curve(metric_dict: Dict[int, float], title: str, xlabel: str = "k", ylabel: str = "score") -> None:
    if not metric_dict:
        return
    ks = sorted(metric_dict.keys())
    vals = [metric_dict[k] for k in ks]
    plt.figure()
    plt.plot(ks, vals, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def cluster_players_pipeline(
    df: pd.DataFrame,
    id_cols: Optional[List[str]] = None,
    position_col: str = "position",
    correlation_threshold: float = 0.90,
    k_min: int = 2,
    k_max: int = 10,
    drop_cols: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Full pipeline:
    - Split GK vs field players
    - Keep numeric features, drop IDs + custom drop_cols
    - Remove highly correlated features
    - Scale
    - Search best k by silhouette
    - Fit KMeans
    - Return artifacts and a labeled dataframe
    """
    id_cols = id_cols or ID_COLS_DEFAULT
    drop_cols = (drop_cols or []) + [c for c in ["serial"] if c in df.columns]

    # Separate GK
    field_df, gk_df = split_goalkeepers(df, position_col=position_col)

    # Preserve identifiers for output
    id_cols_present = [c for c in id_cols if c in field_df.columns]
    ids = field_df[id_cols_present].copy() if id_cols_present else pd.DataFrame(index=field_df.index)

    # Build feature matrix
    X_num = keep_numeric_features(field_df.drop(columns=[c for c in id_cols_present if c in field_df.columns], errors="ignore"),
                                  exclude=drop_cols)

    if X_num.shape[1] == 0:
        raise ValueError("No numeric features available after filtering. Check your columns.")

    # Drop highly correlated
    X_reduced, kept_cols = drop_highly_correlated(X_num, threshold=correlation_threshold)

    # Scale
    Xs, scaler = scale_features(X_reduced)

    # Best k (silhouette)
    best_k, sil_scores = find_best_k_silhouette(Xs, range(k_min, k_max + 1))

    # Fit
    km = run_kmeans(Xs, best_k)
    labels = km.predict(Xs)

    # Output dataframe
    out_df = ids.copy()
    out_df["cluster"] = labels
    out_df = pd.concat([out_df, X_reduced.reset_index(drop=True)], axis=1)

    # Centroid interpretation
    centroids = centroid_top_features(km, list(X_reduced.columns), topn=8)

    return dict(
        field_players_features=X_reduced,
        field_players_scaled=Xs,
        scaler=scaler,
        kept_features=list(X_reduced.columns),
        kmeans=km,
        best_k=best_k,
        silhouette_scores=sil_scores,
        clusters_df=out_df,
        centroid_top_features=centroids,
        goalkeepers_df=gk_df.reset_index(drop=True),
    )
