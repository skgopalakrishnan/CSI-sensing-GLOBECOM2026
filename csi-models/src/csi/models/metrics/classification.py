"""Classification metric helpers (F1 scores)."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np


def _to_numpy(array_like: Iterable[int]) -> np.ndarray:
    if isinstance(array_like, np.ndarray):
        return array_like
    return np.asarray(list(array_like))


def compute_f1_scores(
    y_true: Iterable[int],
    y_pred: Iterable[int],
) -> Dict[str, float]:
    """Compute macro/micro/weighted F1 scores for single-label classification."""

    true_arr = _to_numpy(y_true)
    pred_arr = _to_numpy(y_pred)

    if true_arr.size == 0 or pred_arr.size == 0:
        return {"macro": 0.0, "weighted": 0.0, "micro": 0.0}

    classes = np.union1d(np.unique(true_arr), np.unique(pred_arr))
    if classes.size == 0:
        return {"macro": 0.0, "weighted": 0.0, "micro": 0.0}

    per_class_f1: Sequence[float] = []
    supports: Sequence[int] = []
    tp_list = []
    fp_list = []
    fn_list = []

    for cls in classes:
        true_mask = true_arr == cls
        pred_mask = pred_arr == cls
        tp = int(np.sum(true_mask & pred_mask))
        fp = int(np.sum(~true_mask & pred_mask))
        fn = int(np.sum(true_mask & ~pred_mask))

        supports.append(tp + fn)
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)

        if tp == 0 and fp == 0 and fn == 0:
            per_class_f1.append(0.0)
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            per_class_f1.append(0.0)
        else:
            per_class_f1.append(2.0 * precision * recall / (precision + recall))

    supports_arr = np.asarray(supports, dtype=float)
    f1_arr = np.asarray(per_class_f1, dtype=float)
    valid_mask = supports_arr > 0

    macro = float(np.mean(f1_arr[valid_mask])) if np.any(valid_mask) else 0.0
    weighted = (
        float(np.average(f1_arr[valid_mask], weights=supports_arr[valid_mask]))
        if np.any(valid_mask)
        else 0.0
    )

    tp_sum = float(np.sum(tp_list))
    fp_sum = float(np.sum(fp_list))
    fn_sum = float(np.sum(fn_list))
    if tp_sum == 0 and (fp_sum == 0 or fn_sum == 0):
        micro = 0.0
    else:
        precision_micro = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
        recall_micro = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
        if precision_micro + recall_micro == 0:
            micro = 0.0
        else:
            micro = 2.0 * precision_micro * recall_micro / (precision_micro + recall_micro)

    return {
        "macro": float(macro),
        "weighted": float(weighted),
        "micro": float(micro),
    }

