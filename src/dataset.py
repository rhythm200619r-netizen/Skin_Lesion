"""
dataset.py — PyTorch Dataset + helper functions for ISIC skin lesion images.

- Reads image_name / target columns from a metadata CSV (also handles isic_id)
- Augmentation via albumentations (configurable) for training, resize+normalize for val/test
- Patient-grouped, stratified TRAIN / VAL / TEST split (no patient in two splits)
    train -> fit weights
    val   -> pick checkpoint, fit calibrators
    test  -> final reported numbers (never touched during training)
- Metadata preprocessing for optional fusion (age, sex, site)
"""

import os
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import Dataset

import config

# Prevent CPU thread oversubscription on Kaggle
cv2.setNumThreads(0)

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _ALBU_AVAILABLE = True
    _ALBU_VERSION = tuple(int(x) for x in A.__version__.split(".")[:2])
except ImportError:
    _ALBU_AVAILABLE = False
    _ALBU_VERSION = (0, 0)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


# ──────────────────────────────────────────────────
# Metadata loading & splitting
# ──────────────────────────────────────────────────

def load_metadata(csv_path):
    """Load metadata CSV, normalising column names."""
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
    if "patient_id" not in df.columns:
        raise ValueError("Cannot perform patient-grouped split: 'patient_id' column is missing from data.")
    
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, held_idx = next(splitter.split(df, df["target"], groups=df["patient_id"]))
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


def assert_no_patient_leakage(train_df, val_df, test_df):
    """Hard assertion: no patient appears in more than one split."""
    if "patient_id" not in train_df.columns:
        raise ValueError("Cannot check patient leakage: no patient_id column")
    sets = [set(train_df["patient_id"]), set(val_df["patient_id"]),
            set(test_df["patient_id"])]
    names = ["train", "val", "test"]
    for i in range(3):
        for j in range(i + 1, 3):
            overlap = sets[i] & sets[j]
            assert len(overlap) == 0, (
                f"PATIENT LEAKAGE: {len(overlap)} patients shared between "
                f"{names[i]} and {names[j]}: {list(overlap)[:5]}...")
    print("[OK] Patient-leakage assertion passed (0 shared patients across all splits)")


# ──────────────────────────────────────────────────
# Image path resolution
# ──────────────────────────────────────────────────

def _resolve_image_path(img_dir, image_name):
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(img_dir, f"{image_name}{ext}")
        if os.path.exists(p):
            return p
    # Fall back: maybe image_name already has extension
    return os.path.join(img_dir, image_name)


def detect_image_size(img_dir, n_sample=5):
    """Sample a few images and return their (width, height). Warns if inconsistent."""
    import random
    files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        return None
    sample = random.sample(files, min(n_sample, len(files)))
    sizes = set()
    for f in sample:
        img = cv2.imread(os.path.join(img_dir, f))
        if img is not None:
            sizes.add((img.shape[1], img.shape[0]))  # (W, H)
    if len(sizes) > 1:
        warnings.warn(f"Inconsistent image sizes detected: {sizes}")
    return sizes.pop() if sizes else None


# ──────────────────────────────────────────────────
# Albumentations transforms
# ──────────────────────────────────────────────────

def get_train_transforms(img_size=IMG_SIZE):
    """Training augmentation pipeline via albumentations."""
    if not _ALBU_AVAILABLE:
        return None

    # Handle albumentations API differences across versions
    blur_transforms = []
    try:
        blur_transforms = [
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.GaussNoise(p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ]
    except TypeError:
        # Older albumentations versions may have different argument names
        blur_transforms = [
            A.GaussianBlur(p=1.0),
            A.GaussNoise(p=1.0),
            A.MotionBlur(p=1.0),
        ]

    # CoarseDropout — handle API differences
    try:
        dropout = A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(int(img_size * 0.05), int(img_size * 0.15)),
            hole_width_range=(int(img_size * 0.05), int(img_size * 0.15)),
            fill=0, p=0.3
        )
    except TypeError:
        try:
            dropout = A.CoarseDropout(
                max_holes=4, max_height=int(img_size * 0.15),
                max_width=int(img_size * 0.15),
                min_holes=1, min_height=int(img_size * 0.05),
                min_width=int(img_size * 0.05),
                fill_value=0, p=0.3
            )
        except TypeError:
            dropout = A.CoarseDropout(p=0.3)

    transforms = [
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), ratio=(0.9, 1.1), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.OneOf(blur_transforms, p=0.3),
        dropout,
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    return A.Compose(transforms)


def get_val_transforms(img_size=IMG_SIZE):
    """Validation/test transform: resize + normalize only."""
    if not _ALBU_AVAILABLE:
        return None
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size=IMG_SIZE):
    """Return 8 dihedral TTA transforms (identity + 7 variants)."""
    if not _ALBU_AVAILABLE:
        return None
    base = [A.Resize(img_size, img_size)]
    normalize = [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]

    variants = []
    for hflip in [False, True]:
        for rot in [0, 1, 2, 3]:
            ops = list(base)
            if hflip:
                ops.append(A.HorizontalFlip(p=1.0))
            if rot > 0:
                # RandomRotate90 always rotates; we use a deterministic approach
                pass  # handled below
            ops.extend(normalize)
            variants.append((A.Compose(ops), rot))
    return variants


# ──────────────────────────────────────────────────
# Metadata preprocessing
# ──────────────────────────────────────────────────

META_COLS = ["age_approx", "sex", "anatom_site_general_challenge"]


def has_metadata(df):
    """Check if metadata columns exist in the dataframe."""
    return all(c in df.columns for c in META_COLS)


def fit_metadata_preprocessor(train_df):
    """
    Fit metadata preprocessing on TRAIN split only.
    Returns a dict of statistics for reproducible inference.
    """
    stats = {}

    # Age: median imputation + standardize
    age = train_df["age_approx"].dropna()
    stats["age_median"] = float(age.median())
    stats["age_mean"] = float(age.mean())
    stats["age_std"] = float(age.std()) if len(age) > 1 else 1.0

    # Sex categories (with "unknown" for missing)
    sex_vals = train_df["sex"].dropna().unique().tolist()
    stats["sex_categories"] = sorted(sex_vals) + ["unknown"]

    # Anatomical site categories (with "unknown")
    site_vals = train_df["anatom_site_general_challenge"].dropna().unique().tolist()
    stats["site_categories"] = sorted(site_vals) + ["unknown"]

    # Total feature dimension
    stats["meta_dim"] = 1 + len(stats["sex_categories"]) + len(stats["site_categories"])

    return stats


def preprocess_metadata(row, stats):
    """Convert a single row's metadata to a feature vector using train-fitted stats."""
    features = []

    # Age (standardized)
    age = row.get("age_approx", None)
    if pd.isna(age) or age is None:
        age = stats["age_median"]
    age_z = (float(age) - stats["age_mean"]) / max(stats["age_std"], 1e-6)
    features.append(age_z)

    # Sex (one-hot)
    sex = row.get("sex", None)
    if pd.isna(sex) or sex is None:
        sex = "unknown"
    for cat in stats["sex_categories"]:
        features.append(1.0 if sex == cat else 0.0)

    # Anatomical site (one-hot)
    site = row.get("anatom_site_general_challenge", None)
    if pd.isna(site) or site is None:
        site = "unknown"
    for cat in stats["site_categories"]:
        features.append(1.0 if site == cat else 0.0)

    return np.array(features, dtype=np.float32)


# ──────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────

def build_cache(df, img_dir, img_size, cache_path):
    """Builds a uint8 memmap cache of all resized images to avoid JPEG decoding bottleneck."""
    import shutil
    n_imgs = len(df)
    shape = (n_imgs, img_size, img_size, 3)
    dtype = np.uint8
    bytes_per_img = img_size * img_size * 3
    total_bytes = n_imgs * bytes_per_img
    
    if total_bytes > 10 * 1024**3:
        print(f"Skipping cache: would require {total_bytes / 1024**3:.1f} GB (> 10GB limit)")
        return None

    if os.path.exists(cache_path):
        print(f"Using existing image cache at {cache_path}")
        return np.memmap(cache_path, dtype=dtype, mode='r', shape=shape)

    print(f"Building image cache ({total_bytes / 1024**3:.1f} GB) at {cache_path}...")
    # Write mode
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    temp_path = cache_path + ".tmp"
    memmap = np.memmap(temp_path, dtype=dtype, mode='w+', shape=shape)
    
    from tqdm.auto import tqdm
    for idx, row in tqdm(df.iterrows(), total=n_imgs, desc="Caching images"):
        img_path = _resolve_image_path(img_dir, row["image_name"])
        img = cv2.imread(img_path)
        if img is None:
            # Fallback to empty if missing
            memmap[idx] = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        memmap[idx] = img
        
    memmap.flush()
    shutil.move(temp_path, cache_path)
    print("Cache built successfully.")
    
    # Reload in read mode
    return np.memmap(cache_path, dtype=dtype, mode='r', shape=shape)


class ISICDataset(Dataset):
    """
    Expects a dataframe with columns: image_name, target (0=benign, 1=malignant)
    and a directory containing the corresponding .jpg/.jpeg files.

    Supports both legacy cv2-based augmentation and albumentations.
    Optionally returns metadata features.
    Supports memmap caching to bypass Kaggle CPU bottlenecks.
    """

    def __init__(self, df, img_dir, train=True, img_size=IMG_SIZE,
                 transform=None, meta_stats=None, cache_path=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.train = train
        self.img_size = img_size
        self.meta_stats = meta_stats
        
        self.cache = None
        if cache_path:
            self.cache = build_cache(self.df, self.img_dir, self.img_size, cache_path)

        # Use provided transform, or auto-build from albumentations
        if transform is not None:
            self.transform = transform
        elif _ALBU_AVAILABLE:
            self.transform = get_train_transforms(img_size) if train else get_val_transforms(img_size)
        else:
            self.transform = None

    def __len__(self):
        return len(self.df)

    def _legacy_augment(self, img):
        """Fallback augmentation when albumentations is not available."""
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
        
        if self.cache is not None:
            # Copy to avoid modifying the memmap
            img = np.array(self.cache[idx])
        else:
            img_path = _resolve_image_path(self.img_dir, row["image_name"])
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Could not read image at {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        if self.transform is not None:
            # albumentations path
            augmented = self.transform(image=img)
            img_tensor = augmented["image"]
        else:
            # Legacy path
            if self.train:
                img = self._legacy_augment(img)
            img = img.astype(np.float32) / 255.0
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()

        label = torch.tensor(int(row["target"]), dtype=torch.long)

        if self.meta_stats is not None and has_metadata(self.df):
            meta = preprocess_metadata(row, self.meta_stats)
            return img_tensor, label, torch.from_numpy(meta)

        return img_tensor, label


def compute_class_weights(df):
    """Returns per-class weights (inverse frequency), used for a weighted loss."""
    counts = df["target"].value_counts().sort_index()  # index 0, 1
    total = counts.sum()
    weights = total / (len(counts) * counts)
    return torch.tensor(weights.values, dtype=torch.float32)