import numpy as np


def _row_normalize(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        sums = cm.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        cm = cm / sums
    return np.round(cm, 2)


def compare_pairwise_adversarial(all_predictions, true_labels, adv_labels, num_classes):
    
    """
    Compare the pairwise adversarial performance of the model.
    Two scenarios, concealment and promotion, are considered.
    In concealment, predictions are aggregated such that any one label is changed to one of the other labels.
    In promotion, predictions are aggregated such that any all labels are changed to one of the labels.
    """
    
    preds = np.asarray(all_predictions, dtype=int)
    trues = np.asarray(true_labels, dtype=int)
    advs = np.asarray(adv_labels, dtype=int)
    
    # Build confusion matrices without sklearn
    cm = np.zeros((num_classes, num_classes), dtype=float)
    for t, p in zip(trues, preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1.0
    cm = _row_normalize(cm)
    cm2 = np.zeros((num_classes, num_classes), dtype=float)
    for a, p in zip(advs, preds):
        if 0 <= a < num_classes and 0 <= p < num_classes:
            cm2[a, p] += 1.0
    cm2 = _row_normalize(cm2)
    
    best_conceal = 0
    worst_conceal = 1.1
    best_label_conceal = None
    worst_label_conceal = None
    
    best_promote = 0
    worst_promote = 1.1
    best_label_promote = None
    worst_label_promote = None

    for i in range(num_classes):
        # Concealment for class i: among attempts with true=i, measure P(pred != i)
        mask_t = (trues == i)
        n_t = int(mask_t.sum())
        if n_t > 0:
            correct_i = int((preds[mask_t] == i).sum())
            concealment_perf = 1.0 - (correct_i / n_t)
            if concealment_perf > best_conceal:
                best_conceal = concealment_perf
                best_label_conceal = i
            if concealment_perf < worst_conceal:
                worst_conceal = concealment_perf
                worst_label_conceal = i

        # Promotion to class i: among attempts targeted to i, measure P(pred == i)
        mask_a = (advs == i)
        n_a = int(mask_a.sum())
        if n_a > 0:
            hits = int((preds[mask_a] == i).sum())
            promotion_perf = hits / n_a
            if promotion_perf > best_promote:
                best_promote = promotion_perf
                best_label_promote = i
            if promotion_perf < worst_promote:
                worst_promote = promotion_perf
                worst_label_promote = i

    res = {
        "best_conceal": (best_label_conceal, float(best_conceal) if best_label_conceal is not None else None),
        "worst_conceal": (worst_label_conceal, float(worst_conceal) if worst_label_conceal is not None else None),
        "best_promote": (best_label_promote, float(best_promote) if best_label_promote is not None else None),
        "worst_promote": (worst_label_promote, float(worst_promote) if worst_label_promote is not None else None),
        # Include the normalized confusion matrices to persist in results
        "cm_true_vs_pred": cm.tolist(),
        "cm_adv_vs_pred": cm2.tolist(),
    }

    return res
