"""
train.py

Fine-tunes an ImageNet-pretrained EfficientNet-B0 (via timm) for binary
benign/malignant classification on ISIC images.

Changes vs. the Review-1 version:
  - patient-grouped train/val/test split (test is held out, used only by evaluate.py)
  - best checkpoint chosen by validation AUC-ROC (threshold-independent, less noisy)
  - optional mixed precision (--amp) for ~2x speed on NVIDIA GPUs
  - split CSVs saved to outputs/ so every later script uses the identical split

Usage:
    python src/train.py --csv data/train.csv --img-dir data/jpeg/train \
        --epochs 12 --batch-size 64 --num-workers 4 --amp
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

from dataset import ISICDataset, compute_class_weights, load_metadata, three_way_split


def build_model(pretrained=True):
    return timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=2)


def run_epoch(model, loader, criterion, optimizer, device, train=True, scaler=None, amp=False):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for imgs, labels in tqdm(loader, desc="train" if train else "val", leave=False):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=amp):
                outputs = model(imgs)
                loss = criterion(outputs.float(), labels)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            probs = torch.softmax(outputs.float(), dim=1)[:, 1]  # P(malignant)
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
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }
    try:
        metrics["auc_roc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["auc_roc"] = float("nan")
    return metrics


def save_confusion_matrix(labels, preds, out_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (validation)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def threshold_sweep(labels, probs, out_path):
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
    use_amp = args.amp and device.type == "cuda"
    print(f"Using device: {device} | AMP: {use_amp}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    df = load_metadata(args.csv)
    train_df, val_df, test_df = three_way_split(df, args.val_size, args.test_size, args.seed)
    train_df.to_csv("outputs/train_split.csv", index=False)
    val_df.to_csv("outputs/val_split.csv", index=False)
    test_df.to_csv("outputs/test_split.csv", index=False)

    for name, d in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"{name}: {len(d)} images | malignant {int(d['target'].sum())} ({d['target'].mean():.3%})")
    if "patient_id" in df.columns:
        overlap = set(train_df.patient_id) & (set(val_df.patient_id) | set(test_df.patient_id))
        overlap |= set(val_df.patient_id) & set(test_df.patient_id)
        print(f"Patients shared across splits: {len(overlap)} (should be 0)")

    train_ds = ISICDataset(train_df, args.img_dir, train=True)
    val_ds = ISICDataset(val_df, args.img_dir, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = build_model(pretrained=not args.no_pretrained).to(device)

    class_weights = compute_class_weights(train_df).to(device)
    print(f"Class weights (benign, malignant): {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_auc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _, _ = run_epoch(model, train_loader, criterion, optimizer, device,
                                        train=True, scaler=scaler, amp=use_amp)
        val_loss, val_labels, val_preds, val_probs = run_epoch(model, val_loader, criterion, optimizer,
                                                               device, train=False, amp=use_amp)

        m = compute_metrics(val_labels, val_preds, val_probs)
        scheduler.step(m["auc_roc"])

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
              f"acc={m['accuracy']:.3f} recall={m['recall']:.3f} prec={m['precision']:.3f} "
              f"f1={m['f1']:.3f} auc={m['auc_roc']:.3f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **m})

        if m["auc_roc"] >= best_auc:
            best_auc = m["auc_roc"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_metrics": m,
                "class_weights": class_weights.cpu().tolist(),
                "seed": args.seed,
                "val_size": args.val_size,
                "test_size": args.test_size,
            }, os.path.join("models", "best_model.pth"))
            print(f"  -> New best model saved (val_auc={best_auc:.3f})")
            save_confusion_matrix(val_labels, val_preds, "outputs/confusion_matrix.png")
            threshold_sweep(val_labels, val_probs, "outputs/threshold_sweep.json")

        with open("outputs/training_history.json", "w") as f:  # written every epoch (safe if session dies)
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val AUC = {best_auc:.3f}")
    print("Next: python src/evaluate.py --csv <train.csv> --img-dir <img dir> --split test")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train EfficientNet-B0 on ISIC lesion data")
    p.add_argument("--csv", required=True)
    p.add_argument("--img-dir", required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", help="mixed precision (NVIDIA GPU only)")
    p.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (for offline smoke tests)")
    main(p.parse_args())