"""
train.py

Fine-tunes an ImageNet-pretrained EfficientNet-B0 (via timm) for binary
benign/malignant classification on ISIC images.

- Weighted CrossEntropyLoss to handle class imbalance (~1:8-1:50 depending on subset)
- Saves best checkpoint (by validation recall on malignant class) to models/best_model.pth
- Prints + saves accuracy, precision, recall, F1, confusion matrix, and a
  threshold sweep so the melanoma-recall-priority cutoff can be tuned later.

Usage:
    python src/train.py \
        --csv data/subset_train.csv \
        --img-dir data/jpeg/train \
        --epochs 10 \
        --batch-size 32
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import timm
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ISICDataset, compute_class_weights, load_metadata, stratified_split


def build_model(pretrained=True):
    model = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=2)
    return model


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for imgs, labels in tqdm(loader, desc="train" if train else "val", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # P(malignant)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(labels, preds, probs):
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),  # malignant recall
        "f1": f1_score(labels, preds, zero_division=0),
    }
    try:
        metrics["auc_roc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["auc_roc"] = float("nan")  # only one class present in this batch
    return metrics


def save_confusion_matrix(labels, preds, out_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def threshold_sweep(labels, probs, out_path):
    """
    Sweeps the malignant-probability cutoff and reports recall/precision at each.
    Useful because the default 0.5 threshold is a bad choice for imbalanced,
    high-stakes classes -- melanoma recall should be prioritized over accuracy.
    """
    rows = []
    for t in np.arange(0.1, 0.95, 0.05):
        preds = (probs >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 2),
            "recall": recall_score(labels, preds, zero_division=0),
            "precision": precision_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
        })
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    df = load_metadata(args.csv)
    train_df, val_df = stratified_split(df, val_size=args.val_size, seed=args.seed)
    print(f"Train: {len(train_df)} images | Val: {len(val_df)} images")
    print(f"Train malignant ratio: {train_df['target'].mean():.3%} | "
          f"Val malignant ratio: {val_df['target'].mean():.3%}")

    train_ds = ISICDataset(train_df, args.img_dir, train=True)
    val_ds = ISICDataset(val_df, args.img_dir, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = build_model(pretrained=True).to(device)

    class_weights = compute_class_weights(train_df).to(device)
    print(f"Class weights (benign, malignant): {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_recall = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, tr_labels, tr_preds, tr_probs = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_labels, val_preds, val_probs = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        val_metrics = compute_metrics(val_labels, val_preds, val_probs)
        scheduler.step(val_metrics["recall"])

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.3f} val_recall={val_metrics['recall']:.3f} "
            f"val_precision={val_metrics['precision']:.3f} val_f1={val_metrics['f1']:.3f} "
            f"val_auc={val_metrics['auc_roc']:.3f}"
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics})

        # Save best checkpoint by malignant recall -- the metric that matters most clinically
        if val_metrics["recall"] >= best_recall:
            best_recall = val_metrics["recall"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "class_weights": class_weights.cpu().tolist(),
            }, os.path.join("models", "best_model.pth"))
            print(f"  -> New best model saved (val_recall={best_recall:.3f})")

            save_confusion_matrix(val_labels, val_preds, "outputs/confusion_matrix.png")
            threshold_sweep(val_labels, val_probs, "outputs/threshold_sweep.json")

    with open("outputs/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete.")
    print(f"Best model: models/best_model.pth (val_recall={best_recall:.3f})")
    print("Confusion matrix: outputs/confusion_matrix.png")
    print("Threshold sweep: outputs/threshold_sweep.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on ISIC lesion data")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
