"""
make_subset.py

The full ISIC 2020 training set has ~33,000 images and is heavily imbalanced
(malignant is roughly 1.7% of the data -> only ~580 malignant images total).

For a first working pipeline (Review 1), training on the full set is slow and
unnecessary. This script builds a smaller, stratified subset:
  - Keeps ALL malignant images (they're rare and precious)
  - Randomly samples a fixed number of benign images
  - Writes out a new CSV + optionally copies images into data/subset/

Usage:
    python src/make_subset.py \
        --csv data/train.csv \
        --img-dir data/jpeg/train \
        --out-csv data/subset_train.csv \
        --target-size 6000 \
        --benign-malignant-ratio 8

This does NOT copy image files by default (ISIC images are large) -- it just
writes a CSV with the subset's image names + labels. dataset.py reads images
directly from --img-dir using that CSV, so no duplication of image data is
needed. Pass --copy-images if you actually want a physically separate folder.
"""

import argparse
import os
import shutil

import pandas as pd


def build_subset(csv_path, img_dir, out_csv, target_size, ratio, seed, copy_images, out_img_dir):
    df = pd.read_csv(csv_path)

    if "target" not in df.columns:
        raise ValueError(
            f"Expected a 'target' column (0=benign, 1=malignant) in {csv_path}, "
            f"found columns: {list(df.columns)}"
        )
    if "image_name" not in df.columns:
        raise ValueError(
            f"Expected an 'image_name' column in {csv_path}, "
            f"found columns: {list(df.columns)}"
        )

    malignant = df[df["target"] == 1]
    benign = df[df["target"] == 0]

    n_malignant = len(malignant)
    n_benign_available = len(benign)

    # How many benign images to keep, based on desired ratio to malignant count
    n_benign_target = min(n_benign_available, int(n_malignant * ratio))

    # If that's still less than what's needed to hit target_size, top up with more benign
    n_benign_needed_for_size = max(0, target_size - n_malignant)
    n_benign = min(n_benign_available, max(n_benign_target, n_benign_needed_for_size))

    benign_sample = benign.sample(n=n_benign, random_state=seed)

    subset = pd.concat([malignant, benign_sample], axis=0).sample(frac=1, random_state=seed)
    subset = subset.reset_index(drop=True)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    subset.to_csv(out_csv, index=False)

    print(f"Full dataset:   {len(df)} images ({n_malignant} malignant, {n_benign_available} benign)")
    print(f"Subset written: {len(subset)} images ({n_malignant} malignant, {n_benign} benign)")
    print(f"Malignant ratio in subset: {n_malignant / len(subset):.3%}")
    print(f"Saved subset CSV to: {out_csv}")

    if copy_images:
        os.makedirs(out_img_dir, exist_ok=True)
        missing = 0
        for name in subset["image_name"]:
            src = os.path.join(img_dir, f"{name}.jpg")
            if not os.path.exists(src):
                src = os.path.join(img_dir, f"{name}.jpeg")
            if not os.path.exists(src):
                missing += 1
                continue
            dst = os.path.join(out_img_dir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        print(f"Copied images to: {out_img_dir} ({missing} files not found and skipped)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a stratified subset of ISIC 2020")
    parser.add_argument("--csv", required=True, help="Path to full train.csv metadata")
    parser.add_argument("--img-dir", required=True, help="Path to full-size jpeg image folder")
    parser.add_argument("--out-csv", default="data/subset_train.csv", help="Output subset CSV path")
    parser.add_argument("--out-img-dir", default="data/subset_images", help="Output image folder if --copy-images")
    parser.add_argument("--target-size", type=int, default=6000, help="Approx. total images in subset")
    parser.add_argument("--benign-malignant-ratio", type=float, default=8.0,
                         help="How many benign images to keep per malignant image")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-images", action="store_true",
                         help="Physically copy subset images into --out-img-dir (slower, uses disk)")
    args = parser.parse_args()

    build_subset(
        csv_path=args.csv,
        img_dir=args.img_dir,
        out_csv=args.out_csv,
        target_size=args.target_size,
        ratio=args.benign_malignant_ratio,
        seed=args.seed,
        copy_images=args.copy_images,
        out_img_dir=args.out_img_dir,
    )
