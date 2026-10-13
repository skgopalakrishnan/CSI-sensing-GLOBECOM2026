from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

############################################################################################################

DEFAULT_RANDOM_STATE = 42


def _ensure_numpy(array_like: Any) -> np.ndarray:
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def flatten_csi_samples(data_x: Any) -> np.ndarray:
    """Flatten (N, C, F, T) CSI tensors into (N, features)."""
    x = _ensure_numpy(data_x).astype(np.float32, copy=False)
    if x.ndim < 2:
        raise ValueError(f"Expected >=2D array with batch axis, got shape {x.shape}")
    batch = x.shape[0]
    return x.reshape(batch, -1)


def scale_features(features: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, scaler


def compute_pca_embedding(
    features: np.ndarray,
    n_components: int = 2,
    random_state: int = DEFAULT_RANDOM_STATE,
    whiten: bool = False,
) -> Tuple[np.ndarray, PCA]:
    pca = PCA(
        n_components=n_components,
        whiten=whiten,
        svd_solver="auto",
        random_state=random_state,
    )
    embedding = np.asarray(pca.fit_transform(features))
    return embedding, pca


def compute_umap_embedding(
    features: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[np.ndarray, umap.UMAP]:
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=max(2, n_neighbors),
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    embedding = np.asarray(reducer.fit_transform(features))
    return embedding, reducer


def _safe_metric(fn, *args, **kwargs) -> Optional[float]:
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return None


def summarize_embedding(
    embedding: np.ndarray,
    labels: Optional[Sequence[int]] = None,
    high_dim_features: Optional[np.ndarray] = None,
    trust_n_neighbors: int = 10,
) -> Dict[str, Optional[float]]:
    summary: Dict[str, Optional[float]] = {}
    if labels is not None:
        labels = np.asarray(labels)
        unique = np.unique(labels)
        if unique.size >= 2 and embedding.shape[0] > unique.size:
            summary["silhouette"] = _safe_metric(
                silhouette_score, embedding, labels, metric="euclidean"
            )
            summary["calinski_harabasz"] = _safe_metric(
                calinski_harabasz_score, embedding, labels
            )
            summary["davies_bouldin"] = _safe_metric(
                davies_bouldin_score, embedding, labels
            )
    if high_dim_features is not None and high_dim_features.shape[0] >= 3:
        k = min(trust_n_neighbors, high_dim_features.shape[0] - 2)
        if k >= 2:
            summary["trustworthiness"] = _safe_metric(
                trustworthiness, high_dim_features, embedding, n_neighbors=k
            )
    return summary


def _flatten_upper_tri(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square for upper-triangular flattening")
    idx = np.triu_indices_from(matrix, k=1)
    return matrix[idx]


def procrustes_rmse(reference: np.ndarray, candidate: np.ndarray) -> Optional[float]:
    """Orientation/scale-invariant RMSE via Procrustes alignment."""
    if reference.shape != candidate.shape or reference.shape[0] < 2:
        return None
    ref = reference - reference.mean(axis=0, keepdims=True)
    cand = candidate - candidate.mean(axis=0, keepdims=True)
    ref_norm = np.linalg.norm(ref)
    cand_norm = np.linalg.norm(cand)
    if ref_norm == 0 or cand_norm == 0:
        return None
    ref /= ref_norm
    cand /= cand_norm
    matrix = cand.T @ ref
    try:
        U, _, Vt = np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    rotation = U @ Vt
    aligned = cand @ rotation
    rmse = np.sqrt(np.mean((aligned - ref) ** 2))
    return float(rmse)


@dataclass
class UMAPResult:
    name: str
    config: Dict[str, Any]
    embedding: np.ndarray
    metrics: Dict[str, Optional[float]]
    structure_corr: Optional[float]
    procrustes_rmse: Optional[float]


def run_umap_sensitivity(
    features: np.ndarray,
    base_config: Optional[Mapping[str, Any]] = None,
    sweep: Optional[Iterable[Mapping[str, Any]]] = None,
    labels: Optional[Sequence[int]] = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> List[UMAPResult]:
    """Run a baseline UMAP embedding and sensitivity sweep over hyper-parameters."""
    base_cfg: Dict[str, Any] = {
        "n_components": 2,
        "n_neighbors": 15,
        "min_dist": 0.1,
        "metric": "euclidean",
    }
    if base_config:
        base_cfg.update(base_config)
    sweep = list(sweep or [])
    results: List[UMAPResult] = []

    base_embedding, _ = compute_umap_embedding(
        features,
        n_components=base_cfg["n_components"],
        n_neighbors=base_cfg["n_neighbors"],
        min_dist=base_cfg["min_dist"],
        metric=base_cfg.get("metric", "euclidean"),
        random_state=random_state,
    )
    base_metrics = summarize_embedding(base_embedding, labels, features)
    base_dists = pairwise_distances(base_embedding)
    base_flat = _flatten_upper_tri(base_dists) if base_embedding.shape[0] > 2 else None
    results.append(
        UMAPResult(
            name=f"UMAP base (n={base_cfg['n_neighbors']}, min_dist={base_cfg['min_dist']})",
            config=dict(base_cfg),
            embedding=base_embedding,
            metrics=base_metrics,
            structure_corr=1.0,
            procrustes_rmse=0.0,
        )
    )

    for cfg in sweep:
        cur_cfg = dict(base_cfg)
        cur_cfg.update(cfg)
        embedding, _ = compute_umap_embedding(
            features,
            n_components=cur_cfg["n_components"],
            n_neighbors=cur_cfg["n_neighbors"],
            min_dist=cur_cfg["min_dist"],
            metric=cur_cfg.get("metric", "euclidean"),
            random_state=random_state,
        )
        metrics = summarize_embedding(embedding, labels, features)
        structure_corr = None
        if base_flat is not None and embedding.shape[0] == base_embedding.shape[0]:
            cand_flat = _flatten_upper_tri(pairwise_distances(embedding))
            if cand_flat.size == base_flat.size and cand_flat.size > 1:
                structure_corr = float(np.corrcoef(base_flat, cand_flat)[0, 1])
        rmse = procrustes_rmse(base_embedding, embedding)
        label = f"UMAP n={cur_cfg['n_neighbors']}, min_dist={cur_cfg['min_dist']}"
        results.append(
            UMAPResult(
                name=label,
                config=cur_cfg,
                embedding=embedding,
                metrics=metrics,
                structure_corr=structure_corr,
                procrustes_rmse=rmse,
            )
        )
    return results


def reduce_dimensionality(data_x, new_chan=2, n_items=100) -> np.ndarray:
    """Backwards-compatible helper that now routes through the UMAP utility."""
    x = flatten_csi_samples(data_x)
    print(f"Pre reduction shape: {x.shape}")
    print(f"Data range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"Data mean: {x.mean():.4f}, std: {x.std():.4f}")

    if new_chan >= x.shape[1]:
        print("No reduction needed")
        return x

    x_scaled, _ = scale_features(x)
    print(f"After scaling - mean: {x_scaled.mean():.4f}, std: {x_scaled.std():.4f}")

    n_neighbors = min(max(5, n_items), max(5, x_scaled.shape[0] - 1))
    embedding, _ = compute_umap_embedding(
        x_scaled,
        n_components=new_chan,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=DEFAULT_RANDOM_STATE,
    )
    print(f"Post UMAP shape: {embedding.shape}")
    print(f"UMAP output range: [{embedding.min():.4f}, {embedding.max():.4f}]")
    return embedding

##########

def cluster_and_plot(feature_matrix, n_classes):
    """
    Perform K-means clustering and plot the results.
    
    Args:
        feature_matrix (np.ndarray): Feature matrix for clustering (should be 2D for plotting).
        n_classes (int): Number of clusters.
        
    Returns:
        kmeans: Fitted KMeans model
    """
    # Ensure proper data type
    feature_matrix = np.asarray(feature_matrix, dtype=np.float64)
    
    if feature_matrix.shape[1] != 2:
        print(f"Warning: Feature matrix has {feature_matrix.shape[1]} dimensions, plotting may not work correctly")
    
    # Initialize and fit KMeans
    kmeans = KMeans(init="k-means++", n_clusters=n_classes, n_init=4, random_state=42)
    kmeans.fit(feature_matrix)

    # Create mesh for decision boundary
    h = 0.02  # Step size of the mesh
    x_min, x_max = feature_matrix[:, 0].min() - 1, feature_matrix[:, 0].max() + 1
    y_min, y_max = feature_matrix[:, 1].min() - 1, feature_matrix[:, 1].max() + 1
    
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h, dtype=np.float64),
        np.arange(y_min, y_max, h, dtype=np.float64)
    )

    # Predict cluster labels for mesh points
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = kmeans.predict(mesh_points).reshape(xx.shape)

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.clf()
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired, # type: ignore
        aspect="auto",
        origin="lower",
    )
    
    # Plot data points
    plt.plot(feature_matrix[:, 0], feature_matrix[:, 1], "k.", markersize=2)
    
    # Plot centroids
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    
    plt.title(
        f"K-means clustering on CSI dataset (n_clusters={n_classes})\n"
        "Centroids are marked with white cross"
    )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig(f"clustering_results.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return kmeans

############################################################################################################
