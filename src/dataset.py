"""
dataset.py

PyTorch Dataset + helper functions for ISIC skin lesion images.

- Reads image_name / target columns from a metadata CSV
- Resizes to 224x224, normalizes with ImageNet stats
- Basic augmentation (flip/rotate/color jitter) for the training split only
- Stratified train/val split so both classes are represented in validation
"""

import os

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


def load_metadata(csv_path):
    df = pd.read_csv(csv_path)
    required = {"image_name", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")
    return df


def stratified_split(df, val_size=0.15, seed=42):
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=df["target"],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _resolve_image_path(img_dir, image_name):
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(img_dir, f"{image_name}{ext}")
        if os.path.exists(p):
            return p
    # Fall back: maybe image_name already has extension
    p = os.path.join(img_dir, image_name)
    return p


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
        # Random horizontal flip
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
        # Random vertical flip (dermoscopic images have no canonical orientation)
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 0)
        # Random rotation (0/90/180/270 -- cheap, no interpolation artifacts)
        k = np.random.randint(0, 4)
        if k:
            img = np.rot90(img, k).copy()
        # Mild brightness/contrast jitter
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