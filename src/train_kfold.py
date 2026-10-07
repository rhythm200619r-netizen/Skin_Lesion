"""
train_kfold.py — 5-fold cross-validation ensemble training.

Merges train+val into a single CV set, holding out the test set untouched.
Supports resumability, memory constraints, and time budgeting per Kaggle limits.
Outputs Out-Of-Fold (OOF) predictions for global calibration.
"""

import argparse
import json
import os
import sys
import time
import gc

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import (
    ISICDataset,
    compute_class_weights,
    get_train_transforms,
    get_val_transforms,
    load_metadata,
)
from train import (
    SUPPORTED_ARCHS,
    FocalLoss,
    LabelSmoothingCrossEntropy,
    get_cosine_schedule_with_warmup,
    make_weighted_sampler,
    run_epoch,
    compute_metrics,
    save_confusion_matrix,
    safe_log,
)
from config import DATA_ROOT, WORK_DIR, MODELS_DIR, OUTPUTS_DIR, get_data_paths
from utils import seed_everything, worker_init_fn, create_backbone, TimeBudget


def main(args):
    seed_everything(args.seed)
    time_budget = TimeBudget(args.time_budget_hours)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    channels_last = device.type == "cuda"

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    log_file = os.path.join(OUTPUTS_DIR, "kfold_log.txt")

    # Load Progress
    progress_file = os.path.join(OUTPUTS_DIR, "kfold_progress.json")
    progress = {}
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            progress = json.load(f)

    # Data
    csv_path, img_dir = get_data_paths()
    if args.csv: csv_path = args.csv
    if args.img_dir: img_dir = args.img_dir
    
    df = load_metadata(csv_path)
    
    # Isolate TEST set completely (using the exact same logic as train.py)
    # three_way_split holds out test_size, then val_size. We reproduce the test split:
    from dataset import three_way_split
    _, _, test_df = three_way_split(df, args.val_size, args.test_size, args.seed)
    
    # The CV set is everything NOT in test_df
    test_names = set(test_df["image_name"])
    cv_df = df[~df["image_name"].isin(test_names)].reset_index(drop=True)
    
    safe_log(f"Total dataset: {len(df)} | Held-out Test: {len(test_df)} | CV Set: {len(cv_df)}", log_file)

    # Build 5 Folds
    if "patient_id" in cv_df.columns:
        splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        folds = list(splitter.split(cv_df, cv_df["target"], groups=cv_df["patient_id"]))
    else:
        splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        folds = list(splitter.split(cv_df, cv_df["target"]))

    cache_path = os.path.join(WORK_DIR, "train_cache.dat") if args.cache_resized else None
    
    # OOF tracking
    oof_preds = np.zeros(len(cv_df))
    oof_targets = cv_df["target"].values

    folds_to_run = args.folds if args.folds else list(range(args.n_splits))

    for fold_idx in folds_to_run:
        fold_name = f"fold_{fold_idx}"
        
        if fold_name in progress and progress[fold_name].get("completed"):
            safe_log(f"Skipping {fold_name} - already completed with AUC {progress[fold_name]['best_auc']:.4f}", log_file)
            # Load OOF if exists
            oof_csv = os.path.join(OUTPUTS_DIR, "oof_preds.csv")
            if os.path.exists(oof_csv):
                oof_df = pd.read_csv(oof_csv)
                # Map back to our array
                for _, row in oof_df.iterrows():
                    match = cv_df.index[cv_df["image_name"] == row["image_name"]]
                    if len(match):
                        oof_preds[match[0]] = row["prob_malignant"]
            continue
            
        safe_log(f"\n{'='*40}\nStarting {fold_name}\n{'='*40}", log_file)
        
        train_idx, val_idx = folds[fold_idx]
        train_df = cv_df.iloc[train_idx].reset_index(drop=True)
        val_df = cv_df.iloc[val_idx].reset_index(drop=True)
        
        train_ds = ISICDataset(train_df, img_dir, train=True, img_size=args.img_size, 
                               transform=get_train_transforms(args.img_size), cache_path=cache_path)
        val_ds = ISICDataset(val_df, img_dir, train=False, img_size=args.img_size, 
                             transform=get_val_transforms(args.img_size), cache_path=cache_path)

        sampler = None
        loss_weight = None
        if args.imbalance == "sampler":
            sampler = make_weighted_sampler(train_df, target_malignant_ratio=0.1, num_samples=args.samples_per_epoch)
        elif args.imbalance in ["weighted_loss", "focal"]:
            loss_weight = compute_class_weights(train_df).to(device)

        num_workers = min(args.num_workers, os.cpu_count() or 1)
        loader_kwargs = {"num_workers": num_workers, "pin_memory": (device.type == "cuda")}
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
            loader_kwargs["worker_init_fn"] = worker_init_fn

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(sampler is None),
                                  sampler=sampler, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

        model = create_backbone(args.arch, pretrained=not args.no_pretrained, 
                                drop_rate=args.drop_rate, drop_path_rate=args.drop_path_rate).to(device)
        if channels_last: model = model.to(memory_format=torch.channels_last)

        if args.imbalance == "focal":
            criterion = FocalLoss(alpha=loss_weight, gamma=2.0)
        elif args.label_smoothing > 0:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing, weight=loss_weight)
        else:
            criterion = nn.CrossEntropyLoss(weight=loss_weight)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        total_steps = args.epochs * len(train_loader)
        warmup_steps = min(args.warmup_epochs * len(train_loader), total_steps // 4)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        best_auc = -1.0
        best_epoch = -1
        history = []
        patience_counter = 0
        best_val_probs = None
        
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            
            try:
                train_loss, _, _, _ = run_epoch(
                    model, train_loader, criterion, optimizer, device,
                    train=True, scaler=scaler, amp=use_amp,
                    grad_clip=args.grad_clip, scheduler=scheduler,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    raise RuntimeError("OOM in kfold. Please run with a smaller batch size.")
                raise e
                
            val_loss, val_labels, val_preds, val_probs = run_epoch(
                model, val_loader, criterion, optimizer, device,
                train=False, amp=use_amp,
            )

            m = compute_metrics(val_labels, val_preds, val_probs)
            epoch_time = time.time() - epoch_start
            
            safe_log(f"Fold {fold_idx} Ep {epoch}/{args.epochs} [{epoch_time:.0f}s] | "
                     f"val_loss={val_loss:.4f} auc={m['auc_roc']:.3f} pr_auc={m['auc_pr']:.3f}", log_file)

            if m["auc_roc"] >= best_auc:
                best_auc = m["auc_roc"]
                best_epoch = epoch
                best_val_probs = val_probs
                patience_counter = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch, "val_metrics": m,
                    "arch": args.arch, "img_size": args.img_size,
                }, os.path.join(MODELS_DIR, f"fold{fold_idx}.pth"))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    safe_log(f"Early stopping at epoch {epoch}", log_file)
                    break

            if not time_budget.check_epoch(epoch_time):
                safe_log("Time budget exhausted during kfold! Exiting.", log_file)
                return

        safe_log(f"Fold {fold_idx} completed. Best AUC: {best_auc:.4f} at epoch {best_epoch}", log_file)
        
        # Update OOF
        oof_preds[val_idx] = best_val_probs
        
        # Update progress
        progress[fold_name] = {"completed": True, "best_auc": best_auc, "best_epoch": best_epoch}
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)
            
        # Free memory before next fold
        del model, optimizer, scheduler, scaler, train_loader, val_loader, train_ds, val_ds
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Save final OOF
    oof_df = pd.DataFrame({
        "image_name": cv_df["image_name"],
        "target": oof_targets,
        "prob_malignant": oof_preds
    })
    oof_df.to_csv(os.path.join(OUTPUTS_DIR, "oof_preds.csv"), index=False)
    
    # If all folds ran, evaluate overall OOF AUC
    if len(folds_to_run) == args.n_splits or all(f"fold_{i}" in progress for i in range(args.n_splits)):
        from metrics import classification_metrics
        m = classification_metrics(oof_targets, oof_preds)
        safe_log(f"\nFinal OOF AUC-ROC: {m['auc_roc']:.4f}", log_file)
        safe_log(f"Final OOF PR-AUC: {m['auc_pr']:.4f}", log_file)
        
        # We can now fit the global calibrator on OOF
        safe_log(f"\nFit global calibrator on OOF using: python src/calibrate.py --val-probs {os.path.join(OUTPUTS_DIR, 'oof_preds.csv')}", log_file)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None)
    p.add_argument("--img-dir", default=None)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--arch", choices=SUPPORTED_ARCHS, default="efficientnet_b0")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--drop-rate", type=float, default=0.3)
    p.add_argument("--drop-path-rate", type=float, default=0.2)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--imbalance", choices=["weighted_loss", "sampler", "focal"], default="sampler")
    p.add_argument("--samples-per-epoch", type=int, default=10000)
    
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--folds", type=int, nargs="+", default=None, help="Which folds to run (e.g., 0 1)")
    
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--cache-resized", action="store_true")
    p.add_argument("--time-budget-hours", type=float, default=8.5)
    main(p.parse_args())
