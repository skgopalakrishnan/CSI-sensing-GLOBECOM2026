"""Train a 2-D surrogate that mimics a pretrained CNN in projection space."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Mapping

import pickle

import matplotlib

# Use a non-interactive backend for headless environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from csi.data import get_experiment_loaders
from csi.models.utils.datasets import DatasetSpec
from csi.models.utils.model_utils import load_arch_metadata

from csi.data.utils.common.eda_utils import (
    compute_pca_embedding,
    compute_umap_embedding,
    flatten_csi_samples,
    scale_features,
)
from csi.models.utils.experiment import (
    _load_config,
    _parse_args,
    _resolve_config_path_from_arg,
    normalize_dataset_name,
)
from ..common.optuna_clean_runtime import build_model_for_checkpoint, ensure_optuna_clean_checkpoint
from ..common.shared_utils import DEFAULT_CLEAN_WEIGHTS_DIR, parse_bool, print_training_start, save_training_metadata
from ..common.train_models_utils import infer_input_shape_from_loader, infer_ntu_shape, resolve_project_path, resolve_ntu_num_classes
from ..attacks_raw_csi.whitebox_attack_apgd import (
    _as_weight_dir_list,
    _load_checkpoint_into_model,
    _nested_checkpoint_override,
    _resolve_checkpoint,
)


@dataclass
class SplitCollection:
    split: str
    flat_features: np.ndarray
    teacher_probs: np.ndarray
    teacher_preds: np.ndarray
    true_labels: Optional[np.ndarray]
    embedding: Optional[np.ndarray] = None
    attack_targets: Optional[np.ndarray] = None


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_SPLITS = {"train", "val", "test"}
DATASET_CLASS_NAMES = {
    "ntu-fi_har": {
        0: "box",
        1: "circle",
        2: "clean",
        3: "fall",
        4: "run",
        5: "walk",
    },
}
DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent / "config" / "training" / "train_lenet_surrogate.yaml"
)


def _ensure_root(path: str) -> str:
    return path if path.endswith("/") else f"{path}/"


def _fit_surrogate_model(
    model,
    embeddings: np.ndarray,
    teacher_preds: np.ndarray,
    teacher_probs: Optional[np.ndarray],
):
    if getattr(model, "requires_soft_labels", False):
        if teacher_probs is None:
            raise ValueError("Soft-label surrogate requires teacher probabilities")
        model.fit(embeddings, teacher_probs)
    else:
        model.fit(embeddings, teacher_preds)


def _build_class_name_map(dataset, dataset_name: str) -> Dict[int, str]:
    key = dataset_name.strip().lower()
    if key in DATASET_CLASS_NAMES:
        return dict(DATASET_CLASS_NAMES[key])
    idx_to_name: Dict[int, str] = {}
    class_to_idx = getattr(dataset, "class_to_idx", None)
    if isinstance(class_to_idx, Mapping) and class_to_idx:
        for name, idx in class_to_idx.items():
            idx_to_name[int(idx)] = str(name)
        return idx_to_name
    classes = getattr(dataset, "classes", None)
    if isinstance(classes, (list, tuple)) and classes:
        for idx, name in enumerate(classes):
            idx_to_name[int(idx)] = str(name)
        return idx_to_name
    return idx_to_name


class DistillMLPSurrogate(nn.Module):
    requires_soft_labels = True

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_layers: Sequence[int],
        activation: str,
        learning_rate: float,
        weight_decay: float,
        epochs: int,
        batch_size: int,
        temperature: float,
        device: Optional[str],
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        act = nn.ReLU if activation.lower() == "relu" else nn.Tanh
        for size in hidden_layers:
            layers.append(nn.Linear(prev, size))
            layers.append(act())
            prev = size
        layers.append(nn.Linear(prev, num_classes))
        self.model = nn.Sequential(*layers)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.temperature = temperature
        self._device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self._device)

    def fit(self, X: np.ndarray, teacher_probs: np.ndarray):
        dataset = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(teacher_probs).float(),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        for _ in range(self.epochs):
            for batch_x, batch_p in loader:
                batch_x = batch_x.to(self._device)
                batch_p = batch_p.to(self._device)
                optimizer.zero_grad()
                logits = self.model(batch_x) / self.temperature
                log_probs = F.log_softmax(logits, dim=1)
                target = batch_p / batch_p.sum(dim=1, keepdim=True).clamp_min(1e-8)
                loss = F.kl_div(log_probs, target, reduction="batchmean") * (self.temperature ** 2)
                loss.backward()
                optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X).float().to(self._device))
            preds = logits.argmax(dim=1).cpu().numpy()
        return preds


def _select_loader(bundle, split: str):
    split = split.lower()
    if split == "train":
        loader = bundle.train_loader
    elif split == "val":
        loader = bundle.val_loader
    elif split == "test":
        loader = bundle.test_loader
    else:  # pragma: no cover - defensive programming
        raise ValueError(f"Unsupported split '{split}'.")
    if loader is None:
        raise RuntimeError(f"Split '{split}' not available for this dataset.")
    return loader


def _extract_surrogate_params(model, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if hasattr(model, "get_params"):
        try:
            return model.get_params()  # type: ignore[attr-defined]
        except Exception:
            pass
    return dict(fallback or {})


def _merge_arch_metadata(base: Optional[Mapping[str, Any]], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base or {})
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_arch_metadata(current, value)
        else:
            merged[key] = value
    return merged


def _collect_split_data(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    split: str,
    max_samples: Optional[int],
) -> SplitCollection:
    flat_list: List[np.ndarray] = []
    prob_list: List[np.ndarray] = []
    pred_list: List[np.ndarray] = []
    label_list: List[np.ndarray] = []

    samples_collected = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, (tuple, list)):
                raise ValueError("Expected data loader to yield tuple/list.")
            features = batch[0]
            labels = batch[1] if len(batch) > 1 else None

            batch_size = features.size(0)
            take = batch_size
            if max_samples is not None:
                remaining = max_samples - samples_collected
                if remaining <= 0:
                    break
                take = min(take, remaining)

            inputs = features[:take].to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

            flat = flatten_csi_samples(features[:take].detach().cpu())
            teacher_preds = np.asarray(np.argmax(probs, axis=1), dtype=np.int64)

            flat_list.append(flat)
            prob_list.append(probs.astype(np.float32))
            pred_list.append(teacher_preds)
            if labels is not None:
                label_list.append(labels[:take].detach().cpu().numpy())

            samples_collected += take
            if max_samples is not None and samples_collected >= max_samples:
                break

    if not flat_list:
        raise RuntimeError(f"No samples collected for split '{split}'.")

    flat_features = np.concatenate(flat_list, axis=0)
    teacher_probs = np.concatenate(prob_list, axis=0)
    teacher_preds = np.concatenate(pred_list, axis=0)
    true_labels = np.concatenate(label_list, axis=0) if label_list else None

    return SplitCollection(
        split=split,
        flat_features=flat_features,
        teacher_probs=teacher_probs,
        teacher_preds=teacher_preds,
        true_labels=true_labels,
        attack_targets=None,
    )


def _fit_projection(
    scaled_features: np.ndarray,
    projection_cfg: Dict[str, object],
    random_state: int,
) -> Tuple[np.ndarray, object, Dict[str, float], str]:
    method = str(projection_cfg.get("method", "pca")).lower()
    n_components = int(projection_cfg.get("n_components", 2))
    if n_components != 2:
        raise ValueError("Surrogate visualisation requires exactly 2 projection components.")

    meta: Dict[str, float] = {}
    if method == "pca":
        embedding, projector = compute_pca_embedding(
            scaled_features, n_components=n_components, random_state=random_state
        )
        if hasattr(projector, "explained_variance_ratio_"):
            meta["explained_variance_ratio"] = float(np.sum(projector.explained_variance_ratio_))
    elif method == "umap":
        n_neighbors = int(projection_cfg.get("n_neighbors", 15))
        min_dist = float(projection_cfg.get("min_dist", 0.1))
        metric = str(projection_cfg.get("metric", "euclidean"))
        embedding, projector = compute_umap_embedding(
            scaled_features,
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        meta.update({"n_neighbors": n_neighbors, "min_dist": min_dist})
    else:
        raise ValueError(f"Unsupported projection method '{method}'.")

    return embedding, projector, meta, method


def _transform_projection(projector: object, features: np.ndarray, method: str) -> np.ndarray:
    if method not in {"pca", "umap"}:
        raise ValueError(f"Unsupported projection method '{method}'.")
    return np.asarray(projector.transform(features))


def _build_surrogate(classifier_cfg: Dict[str, object], input_dim: int, num_classes: int, device: torch.device):
    model_type = str(classifier_cfg.get("type", "logistic")).lower()
    if model_type == "logistic":
        max_iter = int(classifier_cfg.get("max_iter", 400))
        C = float(classifier_cfg.get("C", 1.0))
        solver = str(classifier_cfg.get("solver", "lbfgs"))
        multi_class = str(classifier_cfg.get("multi_class", "multinomial"))
        verbose = int(classifier_cfg.get("verbose", 0))
        return LogisticRegression(max_iter=max_iter, C=C, solver=solver, multi_class=multi_class, verbose=verbose)
    if model_type == "svm":
        kernel = str(classifier_cfg.get("kernel", "rbf"))
        C = float(classifier_cfg.get("C", 1.0))
        gamma = classifier_cfg.get("gamma", "scale")
        probability = bool(classifier_cfg.get("probability", False))
        return SVC(kernel=kernel, C=C, gamma=gamma, probability=probability)
    if model_type == "mlp":
        hidden = classifier_cfg.get("hidden_layers", [64, 32])
        if isinstance(hidden, int):
            hidden = [hidden]
        activation = str(classifier_cfg.get("activation", "relu"))
        alpha = float(classifier_cfg.get("alpha", 1e-4))
        max_iter = int(classifier_cfg.get("max_iter", 500))
        return MLPClassifier(hidden_layer_sizes=tuple(hidden), activation=activation, alpha=alpha, max_iter=max_iter)
    if model_type == "distill_mlp":
        hidden = classifier_cfg.get("hidden_layers", [64])
        if isinstance(hidden, int):
            hidden = [hidden]
        lr = float(classifier_cfg.get("learning_rate", 1e-3))
        weight_decay = float(classifier_cfg.get("weight_decay", 0.0))
        epochs = int(classifier_cfg.get("epochs", 500))
        batch_size = int(classifier_cfg.get("batch_size", 128))
        temperature = float(classifier_cfg.get("temperature", 1.0))
        activation = str(classifier_cfg.get("activation", "relu"))
        return DistillMLPSurrogate(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_layers=hidden,
            activation=activation,
            learning_rate=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            temperature=temperature,
            device=str(classifier_cfg.get("device")) if classifier_cfg.get("device") else device.type,
        )
    raise ValueError(f"Unsupported surrogate classifier type '{model_type}'")


def _entry_matches_filters(entry: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
    if not filters:
        return True
    dataset = str(entry.get("dataset", "")).lower()
    model = str(entry.get("model", "")).lower()
    budget = str(entry.get("budget_label", "")).lower()
    source = entry.get("source_label")
    target = entry.get("target_label")

    def _match(value, allowed):
        if not allowed:
            return True
        allowed_norm = {str(v).lower() for v in allowed}
        return str(value).lower() in allowed_norm

    if not _match(dataset, filters.get("datasets")):
        return False
    if not _match(model, filters.get("models")):
        return False
    if not _match(budget, filters.get("budgets")):
        return False
    if filters.get("sources") is not None:
        allowed_sources = {int(v) for v in filters["sources"]}
        if source is None or int(source) not in allowed_sources:
            return False
    if filters.get("targets") is not None:
        allowed_targets = {int(v) for v in filters["targets"]}
        if target is None or int(target) not in allowed_targets:
            return False
    return True


def _iter_manifest_npzs(manifest_cfg: Dict[str, Any]) -> List[Tuple[Path, Dict[str, Any]]]:
    manifest_path = Path(manifest_cfg["path"]).expanduser().resolve()
    base_dir = manifest_path.parent
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    entries = manifest.get("entries", [])
    matched: List[Tuple[Path, Dict[str, Any]]] = []
    filters = manifest_cfg.get("filters")
    max_entries = manifest_cfg.get("max_entries")
    for entry in entries:
        if not _entry_matches_filters(entry, filters):
            continue
        storage = entry.get("storage") or {}
        data_rel = storage.get("data_path")
        if not data_rel:
            continue
        npz_path = (base_dir / data_rel).resolve()
        matched.append((npz_path, entry))
        if max_entries is not None and len(matched) >= int(max_entries):
            break
    return matched


def _collect_from_numpy_batches(
    model: torch.nn.Module,
    data_array: np.ndarray,
    true_labels: Optional[np.ndarray],
    target_labels: Optional[np.ndarray],
    split: str,
    device: torch.device,
    batch_size: int,
) -> SplitCollection:
    total = data_array.shape[0]
    flat_list: List[np.ndarray] = []
    prob_list: List[np.ndarray] = []
    pred_list: List[np.ndarray] = []
    label_array = None
    if true_labels is not None:
        label_array = np.asarray(true_labels)
        if label_array.shape[0] != total:
            raise ValueError("True labels must align with data samples")

    model.eval()
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        batch_np = data_array[start:end].astype(np.float32, copy=False)
        inputs = torch.from_numpy(batch_np).to(device)
        with torch.no_grad():
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        flat = flatten_csi_samples(batch_np)
        preds = np.argmax(probs, axis=1)
        flat_list.append(flat)
        prob_list.append(probs.astype(np.float32))
        pred_list.append(preds.astype(np.int64))

    flat_features = np.concatenate(flat_list, axis=0)
    teacher_probs = np.concatenate(prob_list, axis=0)
    teacher_preds = np.concatenate(pred_list, axis=0)
    labels_final = label_array.copy() if label_array is not None else None
    target_final = None
    if target_labels is not None:
        target_array = np.asarray(target_labels)
        if target_array.shape[0] != total:
            raise ValueError("Target labels must align with data samples")
        target_final = target_array.copy()
    return SplitCollection(
        split=split,
        flat_features=flat_features,
        teacher_probs=teacher_probs,
        teacher_preds=teacher_preds,
        true_labels=labels_final,
        attack_targets=target_final,
    )


def _load_adversarial_collections(
    adv_cfg: Dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
) -> Dict[str, SplitCollection]:
    manifests = adv_cfg.get("manifests") or []
    if not manifests:
        return {}
    batch_size = int(adv_cfg.get("batch_size", 256))
    results: Dict[str, SplitCollection] = {}
    for manifest_cfg in manifests:
        split_name = manifest_cfg.get("split_name") or manifest_cfg.get("name") or "adv"
        data_key = manifest_cfg.get("data_key", "adv")
        label_key = manifest_cfg.get("label_key", "source_labels")
        target_label_key = manifest_cfg.get("target_label_key")
        max_samples = manifest_cfg.get("max_samples")
        npz_entries = _iter_manifest_npzs(manifest_cfg)
        if not npz_entries:
            print(f"[WARN] No adversarial entries matched for manifest {manifest_cfg.get('path')}")
            continue
        arrays: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        collected = 0
        for npz_path, _ in npz_entries:
            with np.load(npz_path) as data:
                if data_key not in data:
                    raise KeyError(f"Array '{data_key}' missing in {npz_path}")
                arr = data[data_key]
                if arr.ndim < 2:
                    raise ValueError(f"Expected >=2D array for '{data_key}' in {npz_path}")
                take = arr
                if max_samples is not None:
                    remaining = int(max_samples) - collected
                    if remaining <= 0:
                        break
                    take = arr[:remaining]
                arrays.append(take)
                collected += take.shape[0]
                if label_key and label_key in data:
                    labels.append(data[label_key][: take.shape[0]])
                if target_label_key and target_label_key in data:
                    targets.append(data[target_label_key][: take.shape[0]])
            if max_samples is not None and collected >= int(max_samples):
                break
        if not arrays:
            print(f"[WARN] No adversarial tensors loaded for split '{split_name}'")
            continue
        stacked = np.concatenate(arrays, axis=0)
        label_array = np.concatenate(labels, axis=0) if labels else None
        target_array = np.concatenate(targets, axis=0) if targets else None
        results[split_name] = _collect_from_numpy_batches(
            model=model,
            data_array=stacked,
            true_labels=label_array,
            target_labels=target_array,
            split=split_name,
            device=device,
            batch_size=batch_size,
        )
        print(f"[INFO] Loaded adversarial split '{split_name}' with {stacked.shape[0]} samples")
    return results


def _plot_decision_surface(
    surrogate,
    embeddings: Dict[str, np.ndarray],
    teacher_preds: Dict[str, np.ndarray],
    true_labels_map: Dict[str, Optional[np.ndarray]],
    metrics: Dict[str, Dict[str, float]],
    num_classes: int,
    class_names: Dict[int, str],
    figure_path: Path,
    grid_resolution: int,
    padding: float,
    ordered_splits: Sequence[str],
    split_aliases: Optional[Dict[str, str]] = None,
) -> None:
    if not embeddings:
        raise ValueError("No embeddings supplied for plotting.")

    all_points = np.concatenate(list(embeddings.values()), axis=0)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    x_pad = (x_max - x_min) * padding if x_max > x_min else padding
    y_pad = (y_max - y_min) * padding if y_max > y_min else padding
    x_lin = np.linspace(x_min - x_pad, x_max + x_pad, grid_resolution)
    y_lin = np.linspace(y_min - y_pad, y_max + y_pad, grid_resolution)
    xx, yy = np.meshgrid(x_lin, y_lin)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_preds = surrogate.predict(grid_points).reshape(xx.shape)

    cmap = plt.get_cmap("tab20", num_classes)
    levels = np.arange(num_classes + 1) - 0.5
    norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N)

    n_cols = len(ordered_splits)
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(6 * n_cols, 5.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    split_aliases = split_aliases or {}
    for idx, split in enumerate(ordered_splits):
        ax = axes[idx]
        emb = embeddings[split]
        preds = teacher_preds[split]
        ax.contourf(xx, yy, grid_preds, levels=levels, cmap=cmap, alpha=0.35)
        true_labels = true_labels_map.get(split)
        if true_labels is not None:
            face_colors = cmap(np.asarray(true_labels, dtype=int))
        else:
            face_colors = cmap(preds)
        edge_colors = cmap(preds)
        ax.scatter(
            emb[:, 0],
            emb[:, 1],
            s=65,
            facecolors=face_colors,
            edgecolors=edge_colors,
            linewidths=1.2,
            alpha=0.95,
        )
        acc = metrics.get(split, {}).get("surrogate_vs_teacher_acc")
        subtitle = split_aliases.get(split, split)
        if acc is not None:
            subtitle += f" | mimic acc={acc:.3f}"
        ax.set_title(subtitle)
        ax.set_xlabel("Component 1")
        if idx == 0:
            ax.set_ylabel("Component 2")
        else:
            ax.set_ylabel("")
        ax.set_aspect("equal", adjustable="box")

    fig.subplots_adjust(bottom=0.18)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes,
        orientation="horizontal",
        pad=0.04,
        fraction=0.05,
    )
    tick_positions = list(range(num_classes))
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels([class_names.get(int(idx), str(idx)) for idx in tick_positions])
    cbar.ax.tick_params(labelrotation=0)
    cbar.set_label("Teacher (CNN) prediction")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=400)
    plt.close(fig)


def _unique_ordered(values: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def main(config_path: Optional[str] = None):
    cfg_path = config_path or str(DEFAULT_CONFIG)
    cfg = _load_config(cfg_path)

    common_cfg = cfg.get("common", {})
    datasets = common_cfg.get("datasets") or [common_cfg.get("dataset") or cfg.get("dataset")]
    dataset_name = normalize_dataset_name(datasets[0], "NTU-Fi_HAR")
    root = _ensure_root(common_cfg.get("root", str(REPO_ROOT / "datasets")))

    data_cfg = cfg.get("data", {})
    batch_sizes = data_cfg.get("batch_sizes") or {"train": 64, "val": 64, "test": 64}
    task = data_cfg.get("task", "classification")
    timesteps = int(data_cfg.get("timesteps", 250))
    base_normalisation = data_cfg.get("base_normalisation", "none")
    canon_normalisation = data_cfg.get("canon_normalisation", "per_subcarrier")
    variant = data_cfg.get("variant", "raw")
    smoothing = data_cfg.get("smoothing")
    augment_train = data_cfg.get("augment", {}) or {}
    augment_eval = data_cfg.get("augment_eval", augment_train) or augment_train
    data_seed = int(data_cfg.get("seed", 42))
    device_name = data_cfg.get("device")
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    elif device_name == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    model_cfg = cfg.get("model", {})
    architectures = model_cfg.get("architectures") or [model_cfg.get("architecture") or "LeNet"]
    architecture = str(architectures[0])
    weights_dir_name = model_cfg.get("weights_dir_name") or model_cfg.get(
        "new_weights_dir_name", "model_params"
    )
    weights_cfg = dict(cfg.get("weights", {}) or {})
    strict_load = parse_bool(weights_cfg.get("strict_load"), default=True)
    selection = str(weights_cfg.get("selection", "auto")).strip().lower()
    weights_ntu = dict(weights_cfg.get("ntu", {}) or {})
    weights_root = str(weights_ntu.get("datasets_root", root))
    weights_dir_names = _as_weight_dir_list(
        weights_ntu.get("weights_dir_names", weights_ntu.get("weights_dir_name")),
        default=str(weights_cfg.get("weights_dir_name") or weights_dir_name or DEFAULT_CLEAN_WEIGHTS_DIR),
    )
    resolved_weights_dir_name = str(weights_dir_names[0])
    checkpoint_glob = weights_ntu.get("checkpoint_glob")
    checkpoint_paths = weights_ntu.get("checkpoint_paths")

    surrogate_cfg = cfg.get("surrogate", {})
    fit_split = str(surrogate_cfg.get("fit_split", "train")).lower()
    eval_splits_raw = surrogate_cfg.get("eval_splits") or [fit_split]
    eval_splits = _unique_ordered([str(split).lower() for split in eval_splits_raw])
    requested_splits = _unique_ordered([fit_split] + eval_splits)
    max_samples_per_split = surrogate_cfg.get("max_samples_per_split")
    if max_samples_per_split is not None:
        max_samples_per_split = int(max_samples_per_split)
    plot_clean_splits = bool(surrogate_cfg.get("plot_clean_splits", False))
    classifier_cfg: Dict[str, Any] = dict(surrogate_cfg.get("classifier", {}))
    surrogate_params_meta: Dict[str, Any] = dict(classifier_cfg)

    projection_cfg = surrogate_cfg.get("projection", {})
    proj_seed = int(projection_cfg.get("seed", data_seed))
    load_artifacts_path = surrogate_cfg.get("load_artifacts")
    artifacts_loaded = False

    grid_cfg = surrogate_cfg.get("grid", {})
    grid_resolution = int(grid_cfg.get("resolution", 400))
    padding = float(grid_cfg.get("padding", 0.35))

    output_cfg = cfg.get("output", {})
    output_dir = Path(output_cfg.get("dir", REPO_ROOT / "experiments" / "results" / "surrogates"))
    figure_name = output_cfg.get("figure_name") or f"{dataset_name}_{architecture}_surrogate.png"
    npz_name = output_cfg.get("npz_name") or figure_name.replace(".png", "_embeddings.npz")
    metrics_name = output_cfg.get("metrics_name") or figure_name.replace(".png", "_metrics.json")
    artifacts_name = output_cfg.get("artifacts_name") or figure_name.replace(".png", "_surrogate.pkl")
    artifacts_path = output_dir / artifacts_name
    print_training_start(
        f"Surrogate training ({dataset_name}/{architecture})",
        {
            "output_dir": str(output_dir),
            "dataset": dataset_name,
            "architecture": architecture,
            "root": root,
            "device": str(device),
            "fit_split": fit_split,
            "eval_splits": eval_splits,
            "weights_dir_name": resolved_weights_dir_name,
            "batch_sizes": batch_sizes,
            "timesteps": timesteps,
        },
    )
    save_training_metadata(
        str(output_dir),
        {
            "stage": "surrogate_train",
            "dataset": dataset_name,
            "architecture": architecture,
            "root": root,
            "device": str(device),
            "task": task,
            "fit_split": fit_split,
            "eval_splits": eval_splits,
            "requested_splits": requested_splits,
            "max_samples_per_split": max_samples_per_split,
            "plot_clean_splits": plot_clean_splits,
            "weights_dir_name": resolved_weights_dir_name,
            "batch_sizes": batch_sizes,
            "timesteps": timesteps,
            "normalisation": {
                "base": base_normalisation,
                "canonical": canon_normalisation,
            },
            "variant": variant,
            "smoothing": smoothing,
            "projection": projection_cfg,
            "classifier": classifier_cfg,
            "output_files": {
                "figure": figure_name,
                "npz": npz_name,
                "metrics": metrics_name,
                "artifacts": artifacts_name,
            },
        },
    )

    print(f"[INFO] Loading dataset {dataset_name} from {root}")
    bundle = get_experiment_loaders(
        dataset=dataset_name,
        root=root,
        task=task,
        timesteps=timesteps,
        base_normalisation=base_normalisation,
        canon_normalisation=canon_normalisation,
        variant=variant,
        batch_sizes=batch_sizes,
        device=device,
        seed=data_seed,
        smoothing=smoothing,
        augment_train=augment_train,
        augment_eval=augment_eval,
    )

    if bundle.info is None or bundle.info.get("num_classes") is None:
        raise RuntimeError("Dataset bundle missing num_classes metadata.")
    num_classes = int(bundle.info["num_classes"])
    class_name_map: Dict[int, str] = dict(DATASET_CLASS_NAMES.get(dataset_name.lower(), {}))
    train_dataset = getattr(bundle, "train_loader", None)
    if train_dataset is not None and getattr(train_dataset, "dataset", None) is not None:
        derived = _build_class_name_map(train_dataset.dataset, dataset_name)
        if derived:
            class_name_map = derived

    print(f"[INFO] Building {architecture} and loading pretrained weights")
    train_loader = bundle.train_loader
    if train_loader is None:
        raise RuntimeError(f"Missing NTU train loader for {dataset_name}")
    shape = infer_ntu_shape(bundle.info or {}, train_loader)
    if shape is None:
        shape = infer_input_shape_from_loader(train_loader)
    if shape is None:
        raise RuntimeError(f"Unable to infer input shape for {dataset_name}/{architecture}.")
    channels, freq_bins, time_steps = shape
    base_arch_metadata: Dict[str, Any] = {
        "num_classes": int(num_classes),
        "channels": int(channels),
        "freq_bins": int(freq_bins),
        "time_steps": int(time_steps),
        "input_shape": [int(channels), int(freq_bins), int(time_steps)],
    }
    spec = DatasetSpec(
        dataset_name,
        channels=int(channels),
        freq_bins=int(freq_bins),
        time_steps=int(time_steps),
        num_classes=int(num_classes),
    )

    base_dir = resolve_project_path(weights_root) / dataset_name / resolved_weights_dir_name
    explicit_checkpoint = _nested_checkpoint_override(checkpoint_paths, dataset_name, architecture)
    explicit_checkpoint = ensure_optuna_clean_checkpoint(
        domain="ntu",
        model_name=architecture,
        weights_dir_name=resolved_weights_dir_name,
        weights_cfg=weights_cfg,
        canonical_base_dir=base_dir,
        alias_stems=[f"{dataset_name}_{architecture}"],
        dataset_name=dataset_name,
    ) or explicit_checkpoint
    weights_path = _resolve_checkpoint(
        base_dir=base_dir,
        model_name=architecture,
        selection=selection,
        alias_stems=[f"{dataset_name}_{architecture}"],
        checkpoint_glob=str(checkpoint_glob) if checkpoint_glob else None,
        explicit_checkpoint=explicit_checkpoint,
    )
    checkpoint_meta = load_arch_metadata(weights_path)
    resolved_arch_metadata = _merge_arch_metadata(checkpoint_meta, base_arch_metadata)
    model, _ = build_model_for_checkpoint(
        domain="ntu",
        model_name=architecture,
        weights_dir_name=resolved_weights_dir_name,
        weights_cfg=weights_cfg,
        spec=spec,
        dataset_key_for_model=dataset_name,
        num_classes=int(num_classes),
        arch_metadata=resolved_arch_metadata,
        dataset_name=dataset_name,
    )
    model = model.to(device)
    _load_checkpoint_into_model(
        model=model,
        checkpoint_path=str(weights_path),
        strict=strict_load,
        device=device,
    )
    model.eval()

    split_data: Dict[str, SplitCollection] = {}
    for split in requested_splits:
        if split.lower() not in CANONICAL_SPLITS:
            print(
                f"[INFO] Split '{split}' is not a canonical dataset split; expecting external loaders/payloads."
            )
            continue
        try:
            loader = _select_loader(bundle, split)
        except RuntimeError as exc:
            if split == fit_split:
                raise
            print(f"[WARN] {exc} Skipping split '{split}'.")
            continue
        print(f"[INFO] Collecting split '{split}' (max_samples={max_samples_per_split or 'all'})")
        split_data[split] = _collect_split_data(
            model, loader, device, split, max_samples=max_samples_per_split
        )
        print(f"        gathered {split_data[split].teacher_probs.shape[0]} samples")

    adversarial_cfg = cfg.get("adversarial")
    if adversarial_cfg:
        adv_splits = _load_adversarial_collections(adversarial_cfg, model, device)
        for name, payload in adv_splits.items():
            if name in split_data:
                print(f"[WARN] Overwriting existing split '{name}' with adversarial data")
            split_data[name] = payload

    if fit_split not in split_data:
        raise RuntimeError(f"Fit split '{fit_split}' could not be collected.")

    available_splits = set(split_data.keys())
    eval_splits = [split for split in eval_splits if split in available_splits]
    if not plot_clean_splits:
        non_clean = [split for split in eval_splits if split.lower() not in CANONICAL_SPLITS]
        if non_clean:
            eval_splits = non_clean
    if not eval_splits:
        eval_splits = [fit_split]

    fit_data = split_data[fit_split]
    projection_meta: Dict[str, float] = {}
    if load_artifacts_path:
        artifact_file = Path(load_artifacts_path).expanduser().resolve()
        if not artifact_file.exists():
            raise FileNotFoundError(f"Surrogate artifacts not found at {artifact_file}")
        with open(artifact_file, "rb") as fh:
            artifact_bundle = pickle.load(fh)
        scaler = artifact_bundle["scaler"]
        projector = artifact_bundle["projector"]
        projection_method = artifact_bundle["projection_method"]
        surrogate = artifact_bundle["surrogate"]
        projection_meta = artifact_bundle.get("projection_meta", {})
        surrogate_params_meta = artifact_bundle.get("classifier_cfg", surrogate_params_meta)
        scaled_fit = scaler.transform(fit_data.flat_features)
        fit_data.embedding = _transform_projection(projector, scaled_fit, projection_method)
        artifacts_loaded = True
        print(f"[INFO] Loaded surrogate artifacts from {artifact_file}")
    else:
        scaled_fit, scaler = scale_features(fit_data.flat_features)
        embedding_fit, projector, projection_meta, projection_method = _fit_projection(
            scaled_fit, projection_cfg, proj_seed
        )
        fit_data.embedding = embedding_fit

    for split, data in split_data.items():
        if split == fit_split:
            continue
        scaled = scaler.transform(data.flat_features)
        data.embedding = _transform_projection(projector, scaled, projection_method)

    if not artifacts_loaded:
        train_classifier_cfg = dict(classifier_cfg)
        train_classifier_cfg.setdefault("input_dim", fit_data.embedding.shape[1])
        train_classifier_cfg.setdefault("num_classes", num_classes)
        surrogate = _build_surrogate(
            train_classifier_cfg,
            input_dim=fit_data.embedding.shape[1],
            num_classes=num_classes,
            device=device,
        )
        surrogate_params_meta = dict(train_classifier_cfg)
        print(f"[INFO] Training surrogate classifier ({surrogate.__class__.__name__})")
        _fit_surrogate_model(
            surrogate,
            fit_data.embedding,
            fit_data.teacher_preds,
            fit_data.teacher_probs,
        )
        if output_cfg.get("save_artifacts", True):
            artifacts_payload = {
                "scaler": scaler,
                "projector": projector,
                "projection_method": projection_method,
                "projection_meta": projection_meta,
                "surrogate": surrogate,
                "classifier_cfg": surrogate_params_meta,
            }
            artifacts_path.parent.mkdir(parents=True, exist_ok=True)
            with open(artifacts_path, "wb") as fh:
                pickle.dump(artifacts_payload, fh)
            print(f"[INFO] Saved surrogate artifacts to {artifacts_path}")

    metrics: Dict[str, Dict[str, float]] = {}
    embeddings_map: Dict[str, np.ndarray] = {}
    teacher_preds_map: Dict[str, np.ndarray] = {}
    true_labels_map: Dict[str, Optional[np.ndarray]] = {}

    for split, data in split_data.items():
        if data.embedding is None:
            raise RuntimeError(f"Missing embedding for split '{split}'")
        embeddings_map[split] = data.embedding
        teacher_preds_map[split] = data.teacher_preds
        true_labels_map[split] = data.true_labels
        surrogate_preds = surrogate.predict(data.embedding)
        mimic_acc = float(accuracy_score(data.teacher_preds, surrogate_preds))
        metrics[split] = {"surrogate_vs_teacher_acc": mimic_acc}
        if data.true_labels is not None:
            teacher_vs_gt = float(accuracy_score(data.true_labels, data.teacher_preds))
            metrics[split]["teacher_vs_ground_truth_acc"] = teacher_vs_gt
        if data.attack_targets is not None:
            teacher_vs_target = float(accuracy_score(data.attack_targets, data.teacher_preds))
            surrogate_vs_target = float(accuracy_score(data.attack_targets, surrogate_preds))
            metrics[split]["teacher_vs_attack_target_acc"] = teacher_vs_target
            metrics[split]["surrogate_vs_attack_target_acc"] = surrogate_vs_target

    eval_embeddings = {split: embeddings_map[split] for split in eval_splits}
    eval_teacher_preds = {split: teacher_preds_map[split] for split in eval_splits}
    figure_path = output_dir / figure_name
    _plot_decision_surface(
        surrogate,
        eval_embeddings,
        eval_teacher_preds,
        {split: true_labels_map.get(split) for split in eval_splits},
        metrics,
        num_classes=num_classes,
        class_names=class_name_map,
        figure_path=figure_path,
        grid_resolution=grid_resolution,
        padding=padding,
        ordered_splits=eval_splits,
        split_aliases=surrogate_cfg.get("split_aliases"),
    )
    print(f"[INFO] Saved surrogate decision surface at {figure_path}")

    if output_cfg.get("save_npz", True):
        payload = {}
        for split, data in split_data.items():
            payload[f"{split}_embedding"] = data.embedding
            payload[f"{split}_teacher_preds"] = data.teacher_preds
            payload[f"{split}_teacher_probs"] = data.teacher_probs
            if data.true_labels is not None:
                payload[f"{split}_true_labels"] = data.true_labels
            if data.attack_targets is not None:
                payload[f"{split}_attack_targets"] = data.attack_targets
        npz_path = output_dir / npz_name
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, **payload)
        print(f"[INFO] Stored embeddings/probabilities at {npz_path}")

    metrics_payload = {
        "dataset": dataset_name,
        "architecture": architecture,
        "weights_path": str(weights_path),
        "projection": {"method": projection_method, **projection_meta},
        "surrogate": {
            "type": surrogate.__class__.__name__,
            "params": _extract_surrogate_params(surrogate, surrogate_params_meta),
        },
        "splits": metrics,
        "fit_split": fit_split,
        "eval_splits": eval_splits,
    }

    if output_cfg.get("save_metrics", True):
        metrics_path = output_dir / metrics_name
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(metrics_payload, fh, indent=2)
        print(f"[INFO] Wrote metrics summary to {metrics_path}")


if __name__ == "__main__":
    cfg_path = os.environ.get("EXP_TRAIN_CONFIG")
    if cfg_path is None and len(sys.argv) > 1:
        cfg_path = _resolve_config_path_from_arg(sys.argv[1])
    main(cfg_path)
