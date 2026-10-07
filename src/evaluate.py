"""
evaluate.py — Comprehensive evaluation of a trained checkpoint.

Loads a trained checkpoint, runs it on a held-out split, and saves:
  <out-dir>/<tag>_probs.csv        image_name, target, prob_malignant, prob_calibrated
  <out-dir>/<tag>_metrics.json     full metrics dict (with CIs)
  <out-dir>/<tag>_confusion.png    confusion matrix at chosen threshold
  <out-dir>/<tag>_roc.png          ROC curve
  <out-dir>/<tag>_pr.png           Precision-Recall curve
  <out-dir>/<tag>_reliability.png  reliability diagram (raw vs calibrated)

Usage (test split — run ONCE at the end):
    python src/evaluate.py \\
        --csv data/train.csv \\
        --img-dir data/jpeg/train \\
        --split test \\
        --tag final_test \\
        --calibrator models/calibrator.json \\
        --threshold-json models/threshold.json

Usage (val split — for calibrator fitting):
    python src/evaluate.py --csv data/train.csv --img-dir data/jpeg/train --split val --tag val_baseline
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

# Add src to path for sibling imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import (
    brier_baseline,
    brier_score,
    bootstrap_ci,
    classification_metrics,
    expected_calibration_error,
    sensitivity_at_specificity,
)


def get_probs(df, img_dir, model_path, batch_size, num_workers, is_ensemble=False, use_tta=False):
    """Run inference and return P(malignant) for every image in df."""
    import torch
    from torch.utils.data import DataLoader
    from dataset import ISICDataset
    from predict import load_model, load_ensemble_models

    if is_ensemble:
        models, device = load_ensemble_models(os.path.dirname(model_path))
    else:
        model, device = load_model(model_path)
        models = [model]

    ds = ISICDataset(df, img_dir, train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))
    
    def apply_tta(batch, idx):
        if idx == 0: return batch
        elif idx == 1: return torch.rot90(batch, 1, [2, 3])
        elif idx == 2: return torch.rot90(batch, 2, [2, 3])
        elif idx == 3: return torch.rot90(batch, 3, [2, 3])
        elif idx == 4: return torch.flip(batch, [3])
        elif idx == 5: return torch.rot90(torch.flip(batch, [3]), 1, [2, 3])
        elif idx == 6: return torch.rot90(torch.flip(batch, [3]), 2, [2, 3])
        elif idx == 7: return torch.rot90(torch.flip(batch, [3]), 3, [2, 3])
        return batch

    num_ttas = 8 if use_tta else 1
    probs = []
    
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            batch_probs = torch.zeros(imgs.shape[0], device=device)
            total_passes = len(models) * num_ttas
            
            for m in models:
                for tta_idx in range(num_ttas):
                    v = apply_tta(imgs, tta_idx)
                    out = m(v)
                    batch_probs += torch.softmax(out, dim=1)[:, 1]
            
            batch_probs /= total_passes
            probs.extend(batch_probs.cpu().numpy())
    return np.array(probs)


def plot_confusion(labels, probs, threshold, m, out_path):
    """Plot and save confusion matrix."""
    preds = (probs >= threshold).astype(int)
    from sklearn.metrics import confusion_matrix as cm_fn
    cm = cm_fn(labels, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (threshold={threshold:.3f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc(labels, probs, auc_val, out_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6)
    ax.set_xlabel("1 − Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr(labels, probs, ap_val, out_path):
    """Plot and save Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(labels, probs)
    prevalence = labels.mean()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"AP = {ap_val:.3f}", linewidth=2)
    ax.axhline(prevalence, linestyle="--", color="gray", alpha=0.6,
               label=f"Prevalence = {prevalence:.3f}")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reliability(labels, probs_raw, bins_raw, probs_cal, bins_cal,
                     ece_raw, ece_cal, out_path):
    """Reliability diagram: raw vs calibrated."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, bins, ece, title in [
        (axes[0], bins_raw, ece_raw, "Before Calibration"),
        (axes[1], bins_cal, ece_cal, "After Calibration"),
    ]:
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        if bins:
            ax.plot([b["mean_pred"] for b in bins],
                    [b["frac_positive"] for b in bins],
                    "o-", label="Model", linewidth=2)
        ax.set_xlabel("Predicted P(malignant)")
        ax.set_ylabel("Observed fraction malignant")
        ax.set_title(f"{title} (ECE = {ece:.4f})")
        ax.legend()
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_metrics(labels, probs_raw, probs_cal, threshold, n_boot,
                  patient_ids=None):
    """Build the full metrics dict for a split."""
    m_raw = classification_metrics(labels, probs_raw, threshold)
    m_cal = classification_metrics(labels, probs_cal, threshold)

    # ECE
    ece_raw, bins_raw = expected_calibration_error(labels, probs_raw)
    ece_cal, bins_cal = expected_calibration_error(labels, probs_cal)

    # Brier
    brier_raw = brier_score(labels, probs_raw)
    brier_cal = brier_score(labels, probs_cal)
    brier_const = brier_baseline(labels)

    # Sensitivity at fixed specificity
    sens_at_90spec = sensitivity_at_specificity(labels, probs_cal, 0.90)
    sens_at_95spec = sensitivity_at_specificity(labels, probs_cal, 0.95)

    bootstrap_note = "patient-level" if patient_ids is not None else "image-level"

    # Bootstrap CIs
    def _ci(fn):
        return bootstrap_ci(labels, probs_cal, fn, n_boot=n_boot,
                           patient_ids=patient_ids)

    auc_ci = _ci(lambda y, p: classification_metrics(y, p)["auc_roc"])
    pr_auc_ci = _ci(lambda y, p: classification_metrics(y, p)["auc_pr"])
    sens_ci = _ci(lambda y, p: classification_metrics(y, p, threshold)["sensitivity"])
    spec_ci = _ci(lambda y, p: classification_metrics(y, p, threshold)["specificity"])
    brier_cal_ci = _ci(lambda y, p: brier_score(y, p))

    result = {
        # Primary metrics (calibrated probs at chosen threshold)
        "threshold": float(threshold),
        "sensitivity": m_cal["sensitivity"],
        "specificity": m_cal["specificity"],
        "precision": m_cal["precision"],
        "npv": m_cal["npv"],
        "f1": m_cal["f1"],
        "accuracy": m_cal["accuracy"],  # secondary

        # Discrimination
        "auc_roc": m_cal["auc_roc"],
        "auc_pr": m_cal["auc_pr"],
        "sensitivity_at_90pct_specificity": sens_at_90spec,
        "sensitivity_at_95pct_specificity": sens_at_95spec,

        # Calibration
        "ece_raw": ece_raw,
        "ece_calibrated": ece_cal,
        "brier_raw": brier_raw,
        "brier_calibrated": brier_cal,
        "brier_constant_baseline": brier_const,

        # Confusion matrix entries
        "tn": m_cal["tn"], "fp": m_cal["fp"],
        "fn": m_cal["fn"], "tp": m_cal["tp"],

        # Sample info
        "n_images": int(len(labels)),
        "n_malignant": int(labels.sum()),
        "prevalence": float(labels.mean()),

        # CIs
        "auc_roc_95ci": list(auc_ci),
        "auc_pr_95ci": list(pr_auc_ci),
        "sensitivity_95ci": list(sens_ci),
        "specificity_95ci": list(spec_ci),
        "brier_calibrated_95ci": list(brier_cal_ci),
        "bootstrap_method": bootstrap_note,
        "n_bootstrap": n_boot,
    }
    return result, bins_raw, bins_cal


from config import DATA_ROOT, WORK_DIR, MODELS_DIR, OUTPUTS_DIR, get_data_paths

# ... later down in main() ...
def main(args):
    from dataset import load_metadata, three_way_split

    os.makedirs(args.out_dir, exist_ok=True)

    csv_path, img_dir = get_data_paths()
    if args.csv: csv_path = args.csv
    if args.img_dir: img_dir = args.img_dir
    
    if not csv_path or not img_dir:
        raise FileNotFoundError(f"Could not auto-detect data in {DATA_ROOT}. Specify --csv and --img-dir.")

    # Load and split
    df = load_metadata(csv_path)
    if args.split != "all":
        train_df, val_df, test_df = three_way_split(df, args.val_size,
                                                     args.test_size, args.seed)
        # Patient-leakage assertion
        if "patient_id" in df.columns:
            sets = [set(train_df.patient_id), set(val_df.patient_id),
                    set(test_df.patient_id)]
            for i, (a, b) in enumerate([(0,1),(0,2),(1,2)]):
                overlap = sets[a] & sets[b]
                assert len(overlap) == 0, (
                    f"Patient leakage detected between splits {a} and {b}: "
                    f"{len(overlap)} shared patients")
            print("[OK] Patient-leakage assertion passed (0 shared patients)")

        split_map = {"train": train_df, "val": val_df, "test": test_df}
        df = split_map[args.split]
    df = df.reset_index(drop=True)

    n_mal = int(df["target"].sum())
    print(f"Evaluating {len(df)} images ({n_mal} malignant) [{args.split}]")

    # Inference
    probs_raw = get_probs(df, img_dir, args.model, args.batch_size,
                          args.num_workers, args.ensemble, args.tta)
    labels = df["target"].values

    # Save raw probs
    prefix = os.path.join(args.out_dir, args.tag)
    probs_df = pd.DataFrame({
        "image_name": df["image_name"],
        "target": labels,
        "prob_malignant": probs_raw,
    })

    # Calibration
    probs_cal = probs_raw.copy()
    if args.calibrator and os.path.exists(args.calibrator):
        from calibrate import calibrate_probs
        with open(args.calibrator) as f:
            cal = json.load(f)
        probs_cal = calibrate_probs(probs_raw, cal)
        probs_df["prob_calibrated"] = probs_cal
        print(f"Applied calibrator: {cal['method']}")
    else:
        probs_df["prob_calibrated"] = probs_raw
        print("No calibrator applied (using raw probabilities)")

    probs_df.to_csv(prefix + "_probs.csv", index=False)

    # Threshold
    threshold = args.threshold
    if args.threshold_json and os.path.exists(args.threshold_json):
        with open(args.threshold_json) as f:
            thr = json.load(f)
        threshold = thr.get("threshold_target_sens", threshold)
        print(f"Using threshold from {args.threshold_json}: {threshold:.4f}")
    else:
        print(f"Using threshold: {threshold}")

    # Patient IDs for bootstrap
    patient_ids = df["patient_id"].values if "patient_id" in df.columns else None
    if patient_ids is not None:
        print(f"Using patient-level bootstrap ({df['patient_id'].nunique()} patients)")
    else:
        print("Using image-level bootstrap (no patient_id column)")

    # Build metrics
    metrics, bins_raw, bins_cal = build_metrics(
        labels, probs_raw, probs_cal, threshold, args.n_boot,
        patient_ids=patient_ids,
    )

    # Save metrics
    with open(prefix + "_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    plot_confusion(labels, probs_cal, threshold, metrics, prefix + "_confusion.png")
    plot_roc(labels, probs_cal, metrics["auc_roc"], prefix + "_roc.png")
    plot_pr(labels, probs_cal, metrics["auc_pr"], prefix + "_pr.png")

    ece_raw, _ = expected_calibration_error(labels, probs_raw)
    ece_cal, _ = expected_calibration_error(labels, probs_cal)
    plot_reliability(labels, probs_raw, bins_raw, probs_cal, bins_cal,
                     ece_raw, ece_cal, prefix + "_reliability.png")

    # Print summary
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS — {args.tag} ({args.split} split)")
    print("=" * 60)
    print(f"  N = {metrics['n_images']}  ({metrics['n_malignant']} malignant, "
          f"prevalence = {metrics['prevalence']:.3%})")
    print(f"  Threshold: {threshold:.4f}")
    print()
    print(f"  Sensitivity:    {metrics['sensitivity']:.3f}  "
          f"95% CI [{metrics['sensitivity_95ci'][0]:.3f}, {metrics['sensitivity_95ci'][1]:.3f}]")
    print(f"  Specificity:    {metrics['specificity']:.3f}  "
          f"95% CI [{metrics['specificity_95ci'][0]:.3f}, {metrics['specificity_95ci'][1]:.3f}]")
    print(f"  Precision:      {metrics['precision']:.3f}")
    print(f"  NPV:            {metrics['npv']:.3f}")
    print(f"  F1:             {metrics['f1']:.3f}")
    print(f"  Accuracy:       {metrics['accuracy']:.3f}  (secondary)")
    print()
    print(f"  AUC-ROC:        {metrics['auc_roc']:.3f}  "
          f"95% CI [{metrics['auc_roc_95ci'][0]:.3f}, {metrics['auc_roc_95ci'][1]:.3f}]")
    print(f"  PR-AUC (AP):    {metrics['auc_pr']:.3f}  "
          f"95% CI [{metrics['auc_pr_95ci'][0]:.3f}, {metrics['auc_pr_95ci'][1]:.3f}]")
    print(f"  Sens@90%Spec:   {metrics['sensitivity_at_90pct_specificity']:.3f}")
    print(f"  Sens@95%Spec:   {metrics['sensitivity_at_95pct_specificity']:.3f}")
    print()
    print(f"  ECE (raw):          {metrics['ece_raw']:.4f}")
    print(f"  ECE (calibrated):   {metrics['ece_calibrated']:.4f}")
    print(f"  Brier (raw):        {metrics['brier_raw']:.4f}")
    print(f"  Brier (calibrated): {metrics['brier_calibrated']:.4f}  "
          f"95% CI [{metrics['brier_calibrated_95ci'][0]:.4f}, {metrics['brier_calibrated_95ci'][1]:.4f}]")
    print(f"  Brier (constant):   {metrics['brier_constant_baseline']:.4f}")
    print()
    print(f"  Bootstrap: {metrics['bootstrap_method']} ({metrics['n_bootstrap']} resamples)")
    print(f"  Saved outputs with prefix: {prefix}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate a checkpoint on a held-out split with full metrics")
    p.add_argument("--csv", default=None)
    p.add_argument("--img-dir", default=None)
    p.add_argument("--model", default=os.path.join(MODELS_DIR, "best_model.pth"))
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    p.add_argument("--tag", default="baseline_test")
    p.add_argument("--out-dir", default=os.path.join(OUTPUTS_DIR, "eval"))
    p.add_argument("--threshold", type=float, default=0.5,
                    help="Decision threshold (overridden by --threshold-json if present)")
    p.add_argument("--threshold-json", default=None,
                    help="Path to threshold.json from calibrate.py (uses threshold_target_sens)")
    p.add_argument("--calibrator", default=None,
                    help="Path to calibrator.json from calibrate.py")
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers (default 0 for safety on Windows)")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--ensemble", action="store_true")
    p.add_argument("--tta", action="store_true")
    main(p.parse_args())
