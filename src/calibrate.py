"""
calibrate.py — Post-hoc probability calibration and operating-threshold selection.

Fits a calibrator (temperature scaling, Platt scaling, or isotonic regression) on
VALIDATION predictions only.  Also selects decision thresholds:
  - Target-sensitivity threshold: the lowest threshold achieving ≥ --target-sens
    (default 0.90) on val, with the resulting specificity.
  - Youden's J threshold: max(sensitivity + specificity − 1).

Outputs:
    models/calibrator.json  — calibrator parameters (reproducible, JSON-serialisable)
    models/threshold.json   — chosen thresholds + val operating-point metrics

Usage (standalone — needs val predictions CSV from evaluate.py or a quick run):
    python src/calibrate.py \
        --val-probs outputs/eval/val_probs.csv \
        --method temperature \
        --target-sens 0.90

Or as an importable module — see calibrate_probs() and select_threshold().
"""

import argparse
import json
import os

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import confusion_matrix, log_loss, roc_curve


# ──────────────────────────────────────────────────
# Calibrators
# ──────────────────────────────────────────────────

def _fit_temperature(probs, labels):
    """Temperature scaling: logit(p_cal) = logit(p_raw) / T.  T > 1 softens."""
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    raw_logits = logit(probs)

    def neg_ll(T):
        scaled = expit(raw_logits / T)
        return log_loss(labels, scaled)

    result = minimize_scalar(neg_ll, bounds=(0.1, 10.0), method="bounded")
    T = float(result.x)
    return {"method": "temperature", "temperature": T}


def _fit_platt(probs, labels, max_iter=200):
    """Platt scaling: logit(p_cal) = A * logit(p_raw) + B."""
    from scipy.optimize import minimize

    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    raw_logits = logit(probs)

    def neg_ll(params):
        A, B = params
        scaled = expit(A * raw_logits + B)
        return log_loss(labels, scaled)

    res = minimize(neg_ll, x0=[1.0, 0.0], method="Nelder-Mead",
                   options={"maxiter": max_iter})
    A, B = float(res.x[0]), float(res.x[1])
    return {"method": "platt", "A": A, "B": B}


def _fit_isotonic(probs, labels):
    """Isotonic regression (non-parametric, monotone)."""
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(probs, labels)
    # Store the piecewise-linear breakpoints so we can serialise to JSON
    return {
        "method": "isotonic",
        "X_thresholds": iso.X_thresholds_.tolist(),
        "y_thresholds": iso.y_thresholds_.tolist(),
    }


def fit_calibrator(probs, labels, method="temperature"):
    """Fit and return a calibrator dict (JSON-serialisable)."""
    probs, labels = np.asarray(probs, dtype=float), np.asarray(labels, dtype=int)
    if method == "temperature":
        return _fit_temperature(probs, labels)
    elif method == "platt":
        return _fit_platt(probs, labels)
    elif method == "isotonic":
        return _fit_isotonic(probs, labels)
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def calibrate_probs(probs, calibrator):
    """Apply a saved calibrator to raw probabilities."""
    probs = np.asarray(probs, dtype=float)
    method = calibrator["method"]

    if method == "temperature":
        clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        return expit(logit(clipped) / calibrator["temperature"])

    elif method == "platt":
        clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        return expit(calibrator["A"] * logit(clipped) + calibrator["B"])

    elif method == "isotonic":
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.X_thresholds_ = np.array(calibrator["X_thresholds"])
        iso.y_thresholds_ = np.array(calibrator["y_thresholds"])
        iso.X_min_ = iso.X_thresholds_[0]
        iso.X_max_ = iso.X_thresholds_[-1]
        iso.f_ = None  # force rebuild from thresholds
        # Manual piecewise-linear interpolation to avoid sklearn internals
        return np.interp(probs, iso.X_thresholds_, iso.y_thresholds_).clip(0, 1)

    elif method == "none":
        return probs.copy()

    else:
        raise ValueError(f"Unknown calibration method: {method}")


# ──────────────────────────────────────────────────
# Threshold selection (val-only)
# ──────────────────────────────────────────────────

def select_threshold(labels, probs, target_sens=0.90):
    """
    Returns dict with:
      - threshold_target_sens: lowest threshold giving ≥ target_sens on this data
      - specificity_at_target_sens
      - threshold_youden: threshold maximising Youden's J = sens + spec − 1
      - sensitivity_at_youden / specificity_at_youden
    """
    labels, probs = np.asarray(labels), np.asarray(probs)
    fpr, tpr, thresholds = roc_curve(labels, probs)
    sens = tpr
    spec = 1 - fpr

    # --- Target sensitivity threshold ---
    # Among all thresholds where sensitivity ≥ target, pick the highest threshold
    # (most specific) that still meets the target.
    mask = sens >= target_sens
    if mask.any():
        idx = np.where(mask)[0]
        # roc_curve returns thresholds in increasing order of threshold,
        # tpr/fpr are monotone.  We want the highest threshold with sens >= target.
        best_idx = idx[np.argmax(spec[idx])]
        thr_target = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.0
        sens_at_target = float(sens[best_idx])
        spec_at_target = float(spec[best_idx])
    else:
        # Cannot reach target sensitivity at any threshold
        thr_target = 0.0
        sens_at_target = float(sens[-1])
        spec_at_target = float(spec[-1])

    # --- Youden's J ---
    j = sens + spec - 1
    best_j_idx = int(np.argmax(j))
    thr_youden = float(thresholds[best_j_idx]) if best_j_idx < len(thresholds) else 0.5
    sens_youden = float(sens[best_j_idx])
    spec_youden = float(spec[best_j_idx])

    return {
        "target_sensitivity": target_sens,
        "threshold_target_sens": thr_target,
        "achieved_sensitivity": sens_at_target,
        "specificity_at_target_sens": spec_at_target,
        "threshold_youden": thr_youden,
        "sensitivity_at_youden": sens_youden,
        "specificity_at_youden": spec_youden,
    }


# ──────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────

from config import MODELS_DIR, OUTPUTS_DIR

def main(args):
    import pandas as pd

    df = pd.read_csv(args.val_probs)
    required = {"target", "prob_malignant"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Val probs CSV missing columns: {missing}")

    labels = df["target"].values
    probs = df["prob_malignant"].values

    print(f"Loaded {len(df)} val samples ({int(labels.sum())} malignant)")

    # --- Fit calibrator ---
    cal = fit_calibrator(probs, labels, method=args.method)
    cal_probs = calibrate_probs(probs, cal)

    os.makedirs(os.path.dirname(args.calibrator_out) or MODELS_DIR, exist_ok=True)
    with open(args.calibrator_out, "w") as f:
        json.dump(cal, f, indent=2)
    print(f"Saved calibrator ({cal['method']}) -> {args.calibrator_out}")

    raw_ll = log_loss(labels, np.clip(probs, 1e-7, 1 - 1e-7))
    cal_ll = log_loss(labels, np.clip(cal_probs, 1e-7, 1 - 1e-7))
    print(f"  Log-loss: raw={raw_ll:.4f}  calibrated={cal_ll:.4f}")

    # --- Threshold selection (on calibrated probs) ---
    thr_info = select_threshold(labels, cal_probs, target_sens=args.target_sens)
    thr_info["applies_to"] = "calibrated"
    thr_info["val_sensitivity"] = thr_info["achieved_sensitivity"]
    thr_info["val_specificity"] = thr_info["specificity_at_target_sens"]

    # Also add the raw-prob thresholds for reference
    thr_info_raw = select_threshold(labels, probs, target_sens=args.target_sens)
    thr_info["raw_threshold_target_sens"] = thr_info_raw["threshold_target_sens"]
    thr_info["raw_threshold_youden"] = thr_info_raw["threshold_youden"]
    thr_info["calibration_method"] = cal["method"]
    
    model_hash = "unknown"
    model_name = "unknown"
    if args.model and os.path.exists(args.model):
        import hashlib
        model_name = os.path.basename(args.model)
        if os.path.isdir(args.model):
            pth_files = sorted([f for f in os.listdir(args.model) if f.endswith(".pth")])
            hashes = []
            for f in pth_files:
                with open(os.path.join(args.model, f), "rb") as mf:
                    hashes.append(hashlib.md5(mf.read()).hexdigest())
            model_hash = hashlib.md5("".join(hashes).encode()).hexdigest()
        else:
            with open(args.model, "rb") as mf:
                model_hash = hashlib.md5(mf.read()).hexdigest()
            
    cal["checkpoint_name"] = model_name
    cal["weights_hash"] = model_hash
    thr_info["checkpoint_name"] = model_name
    thr_info["weights_hash"] = model_hash

    os.makedirs(os.path.dirname(args.threshold_out) or MODELS_DIR, exist_ok=True)
    with open(args.threshold_out, "w") as f:
        json.dump(thr_info, f, indent=2)
    print(f"Saved thresholds -> {args.threshold_out}")
    print(f"  Target-sens threshold (calibrated): {thr_info['threshold_target_sens']:.4f}  "
          f"(sens={thr_info['achieved_sensitivity']:.3f}, spec={thr_info['specificity_at_target_sens']:.3f})")
    print(f"  Youden's J threshold (calibrated):  {thr_info['threshold_youden']:.4f}  "
          f"(sens={thr_info['sensitivity_at_youden']:.3f}, spec={thr_info['specificity_at_youden']:.3f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Calibrate probabilities and select operating thresholds (val only)")
    p.add_argument("--val-probs", required=True,
                    help="CSV with columns: target, prob_malignant (from evaluate.py --split val)")
    p.add_argument("--method", choices=["temperature", "platt", "isotonic"], default="temperature",
                    help="Calibration method (default: temperature)")
    p.add_argument("--target-sens", type=float, default=0.90,
                    help="Target sensitivity for threshold selection (default: 0.90)")
    p.add_argument("--calibrator-out", default=os.path.join(MODELS_DIR, "calibrator.json"))
    p.add_argument("--threshold-out", default=os.path.join(MODELS_DIR, "threshold.json"))
    p.add_argument("--model", default=None, help="Path to the model to compute weights hash")
    main(p.parse_args())
