"""
calibrate.py

Fits temperature scaling on the validation set.

IMPORTANT:
Calibration is fitted on validation data, NOT the test set.

Usage:
    python src\calibrate.py

Output:
    outputs/calibration.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from predict import load_model, preprocess_image


def find_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not find any of {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


class CalibrationDataset(Dataset):

    def __init__(self, dataframe, image_dir):

        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir

        self.image_column = find_column(
            self.df,
            [
                "image_name",
                "isic_id",
                "image"
            ]
        )

        self.target_column = find_column(
            self.df,
            [
                "target",
                "label"
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        image_name = str(
            self.df.loc[index, self.image_column]
        )

        if not image_name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            image_name += ".jpg"

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        image_tensor = preprocess_image(
            image_path
        )[0]

        target = int(
            self.df.loc[index, self.target_column]
        )

        return image_tensor, target


def fit_temperature(logits, labels):

    """
    Learn a single positive temperature T.

    Calibrated logits = logits / T
    """

    log_temperature = torch.nn.Parameter(
        torch.zeros(1)
    )

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=50,
        line_search_fn="strong_wolfe"
    )

    def closure():

        optimizer.zero_grad()

        temperature = torch.exp(
            log_temperature
        )

        calibrated_logits = (
            logits / temperature
        )

        loss = criterion(
            calibrated_logits,
            labels
        )

        loss.backward()

        return loss

    optimizer.step(closure)

    temperature = torch.exp(
        log_temperature
    ).item()

    return float(temperature)


def main():

    parser = argparse.ArgumentParser(
        description="Fit temperature scaling on validation data"
    )

    parser.add_argument(
        "--csv",
        default="data/val_split.csv"
    )

    parser.add_argument(
        "--img-dir",
        default="data/jpeg/train"
    )

    parser.add_argument(
        "--model",
        default="models/best_model.pth"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    args = parser.parse_args()

    print()
    print("=" * 60)
    print("DERMAVISION CONFIDENCE CALIBRATION")
    print("=" * 60)

    print(f"Validation CSV: {args.csv}")
    print(f"Image directory: {args.img_dir}")
    print(f"Model: {args.model}")

    dataframe = pd.read_csv(
        args.csv
    )

    print(
        f"Validation samples: {len(dataframe)}"
    )

    model, device = load_model(
        args.model
    )

    print(
        f"Device: {device}"
    )

    dataset = CalibrationDataset(
        dataframe,
        args.img_dir
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    all_logits = []
    all_labels = []

    print()
    print("Generating validation predictions...")

    with torch.no_grad():

        for batch_index, (images, labels) in enumerate(loader):

            images = images.to(device)

            logits = model(images)

            all_logits.append(
                logits.cpu()
            )

            all_labels.append(
                labels
            )

            if (
                batch_index + 1
            ) % 20 == 0:

                print(
                    f"Processed "
                    f"{batch_index + 1}/"
                    f"{len(loader)} batches"
                )

    logits = torch.cat(
        all_logits
    )

    labels = torch.cat(
        all_labels
    ).long()

    # ---------------------------------------------------------
    # BEFORE CALIBRATION
    # ---------------------------------------------------------

    uncalibrated_nll = F.cross_entropy(
        logits,
        labels
    ).item()

    # ---------------------------------------------------------
    # FIT TEMPERATURE
    # ---------------------------------------------------------

    print()
    print("Fitting temperature...")

    temperature = fit_temperature(
        logits,
        labels
    )

    # ---------------------------------------------------------
    # AFTER CALIBRATION
    # ---------------------------------------------------------

    calibrated_logits = (
        logits / temperature
    )

    calibrated_nll = F.cross_entropy(
        calibrated_logits,
        labels
    ).item()

    calibrated_probabilities = torch.softmax(
        calibrated_logits,
        dim=1
    )[:, 1]

    predictions = (
        calibrated_probabilities >= 0.5
    ).long()

    accuracy = (
        predictions == labels
    ).float().mean().item()

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    output = {
        "method": "temperature_scaling",
        "temperature": temperature,
        "validation_samples": int(len(labels)),
        "uncalibrated_nll": float(
            uncalibrated_nll
        ),
        "calibrated_nll": float(
            calibrated_nll
        ),
        "calibrated_validation_accuracy": float(
            accuracy
        )
    }

    with open(
        "outputs/calibration.json",
        "w"
    ) as file:

        json.dump(
            output,
            file,
            indent=2
        )

    # ---------------------------------------------------------
    # REPORT
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)

    print(
        f"Temperature:       {temperature:.4f}"
    )

    print(
        f"Validation NLL:     "
        f"{uncalibrated_nll:.4f} -> "
        f"{calibrated_nll:.4f}"
    )

    print(
        f"Validation samples: {len(labels)}"
    )

    print(
        "Saved: outputs\\calibration.json"
    )

    print("=" * 60)
    print()


if __name__ == "__main__":
    main()