"""
evaluate.py

Loads a trained checkpoint, runs it on a held-out split, and saves:
  <out-dir>/<tag>_probs.csv     image_name, target, prob_malignant   <- reused by calibration work
  <out-dir>/<tag>_metrics.json  accuracy/precision/recall/specificity/F1/AUC/ECE/Brier (+95% CI)
  <out-dir>/<tag>_confusion.png, <tag>_roc.png, <tag>_reliability.png

Usage (test split, same split train.py made):
    python src/evaluate.py --csv data/train.csv --img-dir /path/to/img224 --split test --tag baseline_test

Any CSV with image_name,target (e.g. a degraded-set manifest) can be scored with --split all.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

from metrics import (bootstrap_ci, brier_score, classification_metrics,
                     expected_calibration_error)


def get_probs(df, img_dir, model_path, batch_size, num_workers):
    import torch
    from torch.utils.data import DataLoader
    from dataset import ISICDataset
    from predict import load_model

    model, device = load_model(model_path)
    loader = DataLoader(ISICDataset(df, img_dir, train=False), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    probs = []
    with torch.no_grad():
        for imgs, _ in loader:
            probs.extend(torch.softmax(model(imgs.to(device)), dim=1)[:, 1].cpu().numpy())
    return np.array(probs)


def plot_all(labels, probs, bins, m, out_prefix):
    plt.figure(figsize=(5, 4))
    sns.heatmap([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Confusion matrix (threshold {m['threshold']})")
    plt.tight_layout(); plt.savefig(out_prefix + "_confusion.png", dpi=150); plt.close()

    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {m['auc_roc']:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("1 - Specificity (FPR)"); plt.ylabel("Sensitivity (TPR)"); plt.title("ROC curve")
    plt.legend(); plt.tight_layout(); plt.savefig(out_prefix + "_roc.png", dpi=150); plt.close()

    plt.figure(figsize=(5, 4.5))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    plt.plot([b["mean_pred"] for b in bins], [b["frac_positive"] for b in bins], "o-", label="model")
    plt.xlabel("Predicted P(malignant)"); plt.ylabel("Observed fraction malignant")
    plt.title(f"Reliability (ECE = {m['ece']:.3f})")
    plt.legend(); plt.tight_layout(); plt.savefig(out_prefix + "_reliability.png", dpi=150); plt.close()


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    from dataset import load_metadata, three_way_split

    df = load_metadata(args.csv)
    if args.split != "all":
        train_df, val_df, test_df = three_way_split(df, args.val_size, args.test_size, args.seed)
        df = {"train": train_df, "val": val_df, "test": test_df}[args.split]
    df = df.reset_index(drop=True)
    print(f"Evaluating {len(df)} images ({int(df['target'].sum())} malignant) [{args.split}]")

    probs = get_probs(df, args.img_dir, args.model, args.batch_size, args.num_workers)
    labels = df["target"].values

    prefix = os.path.join(args.out_dir, args.tag)
    pd.DataFrame({"image_name": df["image_name"], "target": labels,
                  "prob_malignant": probs}).to_csv(prefix + "_probs.csv", index=False)

    m = classification_metrics(labels, probs, args.threshold)
    m["ece"], bins = expected_calibration_error(labels, probs)
    m["brier"] = brier_score(labels, probs)
    m["n_images"], m["n_malignant"] = int(len(labels)), int(labels.sum())
    m["auc_roc_95ci"] = bootstrap_ci(labels, probs, lambda y, p: classification_metrics(y, p)["auc_roc"], args.n_boot)
    m["recall_95ci"] = bootstrap_ci(labels, probs, lambda y, p: classification_metrics(y, p, args.threshold)["recall_sensitivity"], args.n_boot)
    with open(prefix + "_metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    plot_all(labels, probs, bins, m, prefix)

    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, indent=2))
    print(f"Saved outputs with prefix {prefix}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on a held-out split")
    p.add_argument("--csv", required=True)
    p.add_argument("--img-dir", required=True)
    p.add_argument("--model", default="models/best_model.pth")
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    p.add_argument("--tag", default="baseline_test")
    p.add_argument("--out-dir", default="outputs/eval")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-boot", type=int, default=1000)
    main(p.parse_args())
