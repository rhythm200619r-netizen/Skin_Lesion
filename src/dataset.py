"""
dataset.py

PyTorch Dataset + helper functions for ISIC skin lesion images.

- Reads image_name / target columns from a metadata CSV
- Resizes to 224x224, normalizes with ImageNet stats
- Basic augmentation (flip/rotate/color jitter) for the training split only
- Patient-grouped, stratified TRAIN / VAL / TEST split (no patient in two splits)
    train -> fit weights
    val   -> pick checkpoint, fit calibrators
    test  -> final reported numbers (never touched during training)
"""

import os

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


def load_metadata(csv_path):
    df = pd.read_csv(csv_path)
    if "image_name" not in df.columns and "isic_id" in df.columns:
        df = df.rename(columns={"isic_id": "image_name"})  # ISIC-2024-style CSVs
    required = {"image_name", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")
    return df


def _one_fold(df, frac, seed):
    """Hold out ~`frac` of df, stratified by target and grouped by patient_id if present."""
    n_splits = max(2, round(1 / frac))
    if "patient_id" in df.columns:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        rest_idx, held_idx = next(splitter.split(df, df["target"], groups=df["patient_id"]))
    else:
        print("WARNING: no 'patient_id' column -> split is NOT patient-grouped (possible leakage)")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        rest_idx, held_idx = next(splitter.split(df, df["target"]))
    return df.iloc[rest_idx].reset_index(drop=True), df.iloc[held_idx].reset_index(drop=True)


def three_way_split(df, val_size=0.15, test_size=0.15, seed=42):
    """Deterministic train/val/test split. Same args + same CSV -> same split everywhere."""
    rest, test_df = _one_fold(df, test_size, seed)
    train_df, val_df = _one_fold(rest, val_size / (1 - test_size), seed)
    return train_df, val_df, test_df


def stratified_split(df, val_size=0.15, seed=42):
    """Backward-compatible 2-way split (patient-grouped if possible)."""
    train_df, val_df = _one_fold(df, val_size, seed)
    return train_df, val_df


def _resolve_image_path(img_dir, image_name):
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(img_dir, f"{image_name}{ext}")
        if os.path.exists(p):
            return p
    # Fall back: maybe image_name already has extension
    return os.path.join(img_dir, image_name)


class ISICDataset(Dataset):
    """
    Expects a dataframe with columns: image_name, target (0=benign, 1=malignant)
    and a directory containing the corresponding .jpg/.jpeg files.
    """

    def __init__(self, df, img_dir, train=True, img_size=IMG_SIZE):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.train = train
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def _augment(self, img):
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 0)
        k = np.random.randint(0, 4)
        if k:
            img = np.rot90(img, k).copy()
        if np.random.rand() < 0.5:
            alpha = 1.0 + (np.random.rand() - 0.5) * 0.3  # contrast 0.85-1.15
            beta = (np.random.rand() - 0.5) * 30  # brightness -15..15
            img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = _resolve_image_path(self.img_dir, row["image_name"])

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        if self.train:
            img = self._augment(img)

        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()

        label = torch.tensor(int(row["target"]), dtype=torch.long)
        return img, label


def compute_class_weights(df):
    """Returns per-class weights (inverse frequency), used for a weighted loss."""
    counts = df["target"].value_counts().sort_index()  # index 0, 1
    total = counts.sum()
    weights = total / (len(counts) * counts)
    return torch.tensor(weights.values, dtype=torch.float32)