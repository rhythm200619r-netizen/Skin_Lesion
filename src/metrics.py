"""
metrics.py — Pure numpy/sklearn metric helpers (no torch dependency).

Used by evaluate.py, calibrate.py, and calibration code.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def classification_metrics(labels, probs, threshold=0.5):
    """Compute a comprehensive dict of classification metrics at a given threshold."""
    labels, probs = np.asarray(labels), np.asarray(probs)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    n = len(labels)

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else float("nan")

    try:
        auc_roc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc_roc = float("nan")
    try:
        auc_pr = float(average_precision_score(labels, probs))
    except ValueError:
        auc_pr = float("nan")

    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / n),
        "precision": precision,
        "recall_sensitivity": sensitivity,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "npv": npv,
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def sensitivity_at_specificity(labels, probs, target_spec):
    """Return the sensitivity achievable at ≥ target_spec specificity."""
    labels, probs = np.asarray(labels), np.asarray(probs)
    fpr, tpr, _ = roc_curve(labels, probs)
    spec = 1 - fpr
    # Find points where spec >= target
    mask = spec >= target_spec
    if not mask.any():
        return 0.0
    return float(np.max(tpr[mask]))


def brier_score(labels, probs):
    """Brier score of P(malignant) vs. the true label (lower = better)."""
    return float(np.mean((np.asarray(probs) - np.asarray(labels)) ** 2))


def brier_baseline(labels):
    """Brier score of a constant-prevalence predictor (always predict prevalence)."""
    labels = np.asarray(labels, dtype=float)
    prev = labels.mean()
    return float(np.mean((prev - labels) ** 2))


def expected_calibration_error(labels, probs, n_bins=10):
    """
    Binary ECE on P(malignant): bin by predicted probability, compare mean
    prediction to observed malignant frequency, weight by bin size.
    Returns (ece, bins) where bins feeds a reliability diagram.
    """
    labels, probs = np.asarray(labels, float), np.asarray(probs, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    ece, bins = 0.0, []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc = probs[m].mean(), labels[m].mean()
        ece += m.mean() * abs(conf - acc)
        bins.append({"bin": b, "n": int(m.sum()),
                     "mean_pred": float(conf), "frac_positive": float(acc)})
    return float(ece), bins


def bootstrap_ci(labels, probs, fn, n_boot=1000, seed=0, alpha=0.05,
                 patient_ids=None):
    """
    Percentile bootstrap CI for any metric fn(labels, probs).

    If patient_ids is provided, resamples at the patient level (cluster bootstrap)
    to respect the grouped structure.  Otherwise, resamples at image level.
    """
    rng = np.random.default_rng(seed)
    labels, probs = np.asarray(labels), np.asarray(probs)

    if patient_ids is not None:
        patient_ids = np.asarray(patient_ids)
        unique_patients = np.unique(patient_ids)
        vals = []
        for _ in range(n_boot):
            sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
            idx = np.concatenate([np.where(patient_ids == p)[0] for p in sampled])
            if len(set(labels[idx])) < 2:
                continue
            vals.append(fn(labels[idx], probs[idx]))
    else:
        vals = []
        for _ in range(n_boot):
            i = rng.integers(0, len(labels), len(labels))
            if len(set(labels[i])) < 2:
                continue
            vals.append(fn(labels[i], probs[i]))

    if len(vals) < 10:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))
