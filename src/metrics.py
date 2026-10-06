"""metrics.py - pure numpy/sklearn metric helpers (no torch), reused by evaluate.py and calibration code."""

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def classification_metrics(labels, probs, threshold=0.5):
    labels, probs = np.asarray(labels), np.asarray(probs)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(labels)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall_sensitivity": float(recall_score(labels, preds, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auc_roc": float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else float("nan"),
        "auc_pr": float(average_precision_score(labels, probs)) if len(set(labels)) > 1 else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def brier_score(labels, probs):
    """Brier score of P(malignant) vs. the true label (lower = better)."""
    return float(np.mean((np.asarray(probs) - np.asarray(labels)) ** 2))


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
        bins.append({"bin": b, "n": int(m.sum()), "mean_pred": float(conf), "frac_positive": float(acc)})
    return float(ece), bins


def bootstrap_ci(labels, probs, fn, n_boot=1000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for any metric fn(labels, probs). Essential with ~90 positives."""
    rng = np.random.default_rng(seed)
    labels, probs = np.asarray(labels), np.asarray(probs)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(labels), len(labels))
        if len(set(labels[i])) < 2:
            continue
        vals.append(fn(labels[i], probs[i]))
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))
