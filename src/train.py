"""
train.py — Fine-tunes an ImageNet-pretrained CNN for binary classification.

Kaggle-hardened version:
  - Resumable via --resume (saves last.pth with RNG state)
  - Memory safe: OOM retry with gradient accumulation, gc.collect()
  - Time budget via --time-budget-hours
  - Uses config.py for auto-discovery of datasets and paths
  - Auto-falls back to offline weights if internet is disabled
"""

import argparse
import json
import math
import os
import sys
import time
import gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import (
    ISICDataset,
    assert_no_patient_leakage,
    compute_class_weights,
    detect_image_size,
    get_train_transforms,
    get_val_transforms,
    load_metadata,
    three_way_split,
)
from config import DATA_ROOT, WORK_DIR, MODELS_DIR, OUTPUTS_DIR, get_data_paths
from utils import seed_everything, worker_init_fn, create_backbone, TimeBudget

SUPPORTED_ARCHS = ["efficientnet_b0", "efficientnet_b3", "convnext_tiny", "resnet50"]

# ──────────────────────────────────────────────────
# Losses & Schedulers
# ──────────────────────────────────────────────────

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.05, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, logits, targets):
        n_classes = logits.size(-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        if self.weight is not None:
            w = self.weight[targets].unsqueeze(1)
            loss = -(w * smooth_targets * log_probs).sum(dim=-1)
        else:
            loss = -(smooth_targets * log_probs).sum(dim=-1)
        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=-1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - target_probs) ** self.gamma
        log_probs = torch.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        if self.alpha is not None:
            loss = self.alpha[targets] * focal_weight * nll
        else:
            loss = focal_weight * nll
        return loss.mean()

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.01):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def make_weighted_sampler(df, target_malignant_ratio=0.1, num_samples=None):
    targets = df["target"].values
    n_mal = (targets == 1).sum()
    n_ben = (targets == 0).sum()
    if n_mal == 0 or n_ben == 0:
        return None
    w_mal = target_malignant_ratio * n_ben / ((1 - target_malignant_ratio) * n_mal)
    w_ben = 1.0
    weights = np.where(targets == 1, w_mal, w_ben)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=num_samples if num_samples else len(df),
        replacement=True,
    )

# ──────────────────────────────────────────────────
# Training Logic
# ──────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True,
              scaler=None, amp=False, grad_clip=1.0, scheduler=None, accum_steps=1, **kwargs):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        if train:
            optimizer.zero_grad(set_to_none=True)
            
        pbar = tqdm(loader, desc="train" if train else "val  ", leave=False, mininterval=30.0)
        for i, batch in enumerate(pbar):
            imgs, labels = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=amp):
                outputs = model(imgs)
                loss = criterion(outputs.float(), labels)
                if train and accum_steps > 1:
                    loss = loss / accum_steps

            if train and kwargs.get("smoke", False) and i < 3:
                print(f"Batch {i} loss: {loss.item():.4f}")

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
                        if grad_clip > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        if scheduler is not None:
                            scheduler.step()
                else:
                    loss.backward()
                    if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
                        if grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        if scheduler is not None:
                            scheduler.step()

            # Fix total loss accumulation scale
            batch_loss = loss.item() * (accum_steps if train else 1.0)
            total_loss += batch_loss * imgs.size(0)
            
            probs = torch.softmax(outputs.float(), dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            
            if (i + 1) % 50 == 0:
                pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

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
    try:
        metrics["auc_pr"] = average_precision_score(labels, probs)
    except ValueError:
        metrics["auc_pr"] = float("nan")
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

def safe_log(msg, log_file=None):
    print(msg, flush=True)
    if log_file:
        with open(log_file, "a") as f:
            f.write(msg + "\n")

# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main(args):
    seed_everything(args.seed)
    time_budget = TimeBudget(args.time_budget_hours)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    channels_last = device.type == "cuda"

    if args.smoke:
        print("[SMOKE TEST MODE]")
        args.epochs = 2
        args.batch_size = min(args.batch_size, 8)
        args.num_workers = 0
        args.samples_per_epoch = 100

    print(f"Device: {device} | AMP: {use_amp} | Arch: {args.arch}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    log_file = os.path.join(OUTPUTS_DIR, "train_log.txt")

    # Resolve Data
    csv_path, img_dir = get_data_paths()
    if args.csv: csv_path = args.csv
    if args.img_dir: img_dir = args.img_dir
    
    if not csv_path or not img_dir:
        raise FileNotFoundError(f"Could not auto-detect data in {DATA_ROOT}. Specify --csv and --img-dir.")

    safe_log(f"Using CSV: {csv_path}", log_file)
    safe_log(f"Using Img Dir: {img_dir}", log_file)

    df = load_metadata(csv_path)
    train_df, val_df, test_df = three_way_split(df, args.val_size, args.test_size, args.seed)
    assert_no_patient_leakage(train_df, val_df, test_df)

    if args.smoke:
        def make_smoke_split(df_split, n_pos=5, n_neg=20):
            pos = df_split[df_split["target"] == 1].head(n_pos)
            neg = df_split[df_split["target"] == 0].head(n_neg)
            return pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)

        train_df = make_smoke_split(train_df, 5, 95)
        val_df = make_smoke_split(val_df, 5, 15)
        test_df = make_smoke_split(test_df, 5, 15)

    train_df.to_csv(os.path.join(OUTPUTS_DIR, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUTS_DIR, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUTS_DIR, "test_split.csv"), index=False)

    detected_size = detect_image_size(img_dir)
    img_size = args.img_size
    if detected_size is not None:
        src_size = min(detected_size)
        if img_size > src_size:
            safe_log(f"⚠ WARNING: --img-size {img_size} > source images {detected_size}.", log_file)

    cache_path = os.path.join(WORK_DIR, "train_cache.dat") if args.cache_resized else None
    train_ds = ISICDataset(train_df, img_dir, train=True, img_size=img_size, 
                           transform=get_train_transforms(img_size), cache_path=cache_path)
    val_ds = ISICDataset(val_df, img_dir, train=False, img_size=img_size, 
                         transform=get_val_transforms(img_size), cache_path=cache_path)

    sampler = None
    loss_weight = None
    if args.imbalance == "sampler":
        sampler = make_weighted_sampler(train_df, target_malignant_ratio=0.1, num_samples=args.samples_per_epoch)
        safe_log(f"Using WeightedRandomSampler (len={args.samples_per_epoch})", log_file)
    elif args.imbalance == "weighted_loss":
        loss_weight = compute_class_weights(train_df).to(device)
    elif args.imbalance == "focal":
        loss_weight = None

    # Safe worker wrapping
    num_workers = min(args.num_workers, os.cpu_count() or 1)
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": (device.type == "cuda"),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["worker_init_fn"] = worker_init_fn

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(sampler is None),
                              sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = create_backbone(args.arch, pretrained=not args.no_pretrained, 
                            drop_rate=args.drop_rate, drop_path_rate=args.drop_path_rate)
    model = model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.imbalance == "focal":
        criterion = FocalLoss(alpha=None, gamma=2.0)
    elif args.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing, weight=loss_weight)
    else:
        criterion = nn.CrossEntropyLoss(weight=loss_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(args.warmup_epochs * len(train_loader), total_steps // 4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Resumability
    start_epoch = 1
    best_auc = -1.0
    best_val_loss = float("inf")
    history = []
    accum_steps = 1
    
    if args.resume:
        last_path = os.path.join(MODELS_DIR, "last.pth")
        if os.path.exists(last_path):
            safe_log(f"Resuming from {last_path}", log_file)
            ckpt = torch.load(last_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler and "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            if scaler and "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt["epoch"] + 1
            best_auc = ckpt["best_auc"]
            if "history" in ckpt:
                history = ckpt["history"]
            if "rng" in ckpt:
                torch.set_rng_state(ckpt["rng"])
            if "cuda_rng" in ckpt and torch.cuda.is_available():
                torch.cuda.set_rng_state(ckpt["cuda_rng"])
        else:
            safe_log("No last.pth found to resume from.", log_file)

    config = vars(args).copy()
    config.update({"device": str(device), "use_amp": use_amp})
    with open(os.path.join(OUTPUTS_DIR, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    safe_log(f"\n{'='*60}\nTraining {args.arch} from epoch {start_epoch} to {args.epochs}\n{'='*60}", log_file)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        
        # OOM handling loop
        retry = True
        while retry:
            try:
                train_loss, _, _, _ = run_epoch(
                    model, train_loader, criterion, optimizer, device,
                    train=True, scaler=scaler, amp=use_amp,
                    grad_clip=args.grad_clip, scheduler=scheduler,
                    accum_steps=accum_steps, smoke=args.smoke
                )
                retry = False
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and accum_steps < 8:
                    safe_log("CUDA OOM detected! Halving batch size and doubling gradient accumulation...", log_file)
                    torch.cuda.empty_cache()
                    args.batch_size = max(1, args.batch_size // 2)
                    accum_steps *= 2
                    # Rebuild loaders
                    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(sampler is None),
                                              sampler=sampler, **loader_kwargs)
                    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
                else:
                    raise e
                    
        val_loss, val_labels, val_preds, val_probs = run_epoch(
            model, val_loader, criterion, optimizer, device,
            train=False, amp=use_amp,
        )

        m = compute_metrics(val_labels, val_preds, val_probs)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start
        
        safe_log(f"Epoch {epoch}/{args.epochs} [{epoch_time:.0f}s] | "
                 f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                 f"auc={m['auc_roc']:.3f} pr_auc={m['auc_pr']:.3f} | lr={current_lr:.2e}", log_file)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": current_lr, **m})

        if math.isnan(m["auc_roc"]):
            is_best = val_loss <= best_val_loss
            if is_best:
                best_val_loss = val_loss
                safe_log(f"  -> AUC is NaN, checkpointing based on best val_loss ({val_loss:.4f})", log_file)
        else:
            is_best = m["auc_roc"] >= best_auc

        if is_best:
            if not math.isnan(m["auc_roc"]):
                best_auc = m["auc_roc"]
            best_path = os.path.join(MODELS_DIR, "best_model.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "val_metrics": m,
                "class_weights": (loss_weight.cpu().tolist() if loss_weight is not None else None),
                "arch": args.arch, "img_size": img_size,
            }, best_path)
            save_confusion_matrix(val_labels, val_preds, os.path.join(OUTPUTS_DIR, "confusion_matrix.png"))
            threshold_sweep(val_labels, val_probs, os.path.join(OUTPUTS_DIR, "threshold_sweep.json"))

        # Save resumable state
        last_ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_auc": best_auc,
            "history": history,
            "rng": torch.get_rng_state(),
        }
        if scheduler: last_ckpt["scheduler"] = scheduler.state_dict()
        if scaler: last_ckpt["scaler"] = scaler.state_dict()
        if torch.cuda.is_available(): last_ckpt["cuda_rng"] = torch.cuda.get_rng_state()
        
        torch.save(last_ckpt, os.path.join(MODELS_DIR, "last.pth"))
        with open(os.path.join(OUTPUTS_DIR, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)
            
        # Free memory
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Time budget check
        if not time_budget.check_epoch(epoch_time):
            safe_log(f"Time budget of {args.time_budget_hours}h almost exhausted. Stopping cleanly at epoch {epoch}.", log_file)
            break

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None)
    p.add_argument("--img-dir", default=None)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--arch", choices=SUPPORTED_ARCHS, default="efficientnet_b0")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--drop-rate", type=float, default=0.3)
    p.add_argument("--drop-path-rate", type=float, default=0.2)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--imbalance", choices=["weighted_loss", "sampler", "focal"], default="sampler")
    p.add_argument("--samples-per-epoch", type=int, default=12000)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume from last.pth")
    p.add_argument("--cache-resized", action="store_true", help="Build memmap cache of resized images")
    p.add_argument("--time-budget-hours", type=float, default=8.5)
    main(p.parse_args())