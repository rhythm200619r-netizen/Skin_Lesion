"""
predict.py

Loads the trained EfficientNet-B0 checkpoint and runs inference
on a single skin-lesion image.

Supports:
- Standard model inference
- Temperature-scalibrated probabilities
- Image quality assessment
- Configurable malignant threshold

Calibration is learned separately on the validation set and stored at:
    outputs/calibration.json

This is an AI-assisted screening prototype, not a medical diagnosis.
"""

import argparse
import json
import os

import cv2
import numpy as np
import timm
import torch

from quality import assess_image_quality


# ============================================================
# CONFIGURATION
# ============================================================

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)

IMG_SIZE = 224

CLASS_NAMES = [
    "Benign",
    "Malignant",
]

DEFAULT_THRESHOLD = 0.5


# ============================================================
# MODEL
# ============================================================

def load_model(checkpoint_path, device=None):
    """
    Load the trained EfficientNet-B0 checkpoint.
    """

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=2,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, device


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image_path,
    img_size=IMG_SIZE,
):
    """
    Apply the exact preprocessing used during inference/training.
    """

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(
            f"Could not read image at: {image_path}"
        )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB,
    )

    img = cv2.resize(
        img,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA,
    )

    img = img.astype(
        np.float32
    ) / 255.0

    img = (
        img - IMAGENET_MEAN
    ) / IMAGENET_STD

    tensor = torch.from_numpy(
        img.transpose(2, 0, 1)
    ).float().unsqueeze(0)

    return tensor


# ============================================================
# CALIBRATION
# ============================================================

def load_calibration(
    calibration_path=None,
):
    """
    Load the learned temperature scaling parameter.

    Returns:
        temperature, calibration_available
    """

    if calibration_path is None:

        base_dir = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        calibration_path = os.path.join(
            base_dir,
            "outputs",
            "calibration.json",
        )

    if not os.path.exists(
        calibration_path
    ):
        return 1.0, False

    try:

        with open(
            calibration_path,
            "r",
        ) as file:

            data = json.load(file)

        temperature = float(
            data["temperature"]
        )

        if temperature <= 0:
            return 1.0, False

        return temperature, True

    except (
        OSError,
        KeyError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ):

        return 1.0, False


# ============================================================
# PREDICTION
# ============================================================

def predict_image(
    image_path,
    model,
    device,
    threshold=DEFAULT_THRESHOLD,
    calibration_temperature=None,
):
    """
    Run inference on one image.

    Returns:
        Prediction
        Raw probability
        Calibrated probability
        Calibration information
        Image quality information
    """

    # --------------------------------------------------------
    # IMAGE QUALITY
    # --------------------------------------------------------

    quality = assess_image_quality(
        image_path
    )

    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    tensor = preprocess_image(
        image_path
    ).to(device)

    # --------------------------------------------------------
    # MODEL INFERENCE
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            tensor
        )

        # ----------------------------------------------------
        # RAW MODEL PROBABILITY
        # ----------------------------------------------------

        raw_probs = torch.softmax(
            outputs,
            dim=1,
        )[0]

        raw_malignant_probability = (
            raw_probs[1].item()
        )

        # ----------------------------------------------------
        # CALIBRATION
        # ----------------------------------------------------

        calibration_applied = (
            calibration_temperature is not None
            and calibration_temperature > 0
        )

        if calibration_applied:

            calibrated_logits = (
                outputs
                / calibration_temperature
            )

            probs = torch.softmax(
                calibrated_logits,
                dim=1,
            )[0]

        else:

            probs = raw_probs

        malignant_probability = (
            probs[1].item()
        )

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    if malignant_probability >= threshold:

        predicted_class = (
            CLASS_NAMES[1]
        )

        confidence = (
            malignant_probability
        )

    else:

        predicted_class = (
            CLASS_NAMES[0]
        )

        confidence = (
            1.0 - malignant_probability
        )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {
        # Prediction
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "malignant_probability": float(
            malignant_probability
        ),

        # Raw model output
        "raw_malignant_probability": float(
            raw_malignant_probability
        ),

        # Decision
        "threshold_used": float(
            threshold
        ),

        # Calibration
        "calibration_applied": bool(
            calibration_applied
        ),

        "calibration_temperature": float(
            calibration_temperature
            if calibration_applied
            else 1.0
        ),

        # Quality
        "quality_score": float(
            quality["quality_score"]
        ),

        "quality_status": (
            quality["quality_status"]
        ),

        "quality_warnings": (
            quality["warnings"]
        ),

        "quality_details": {
            "width": quality["width"],
            "height": quality["height"],
            "sharpness": quality["sharpness"],
            "brightness": quality["brightness"],
            "contrast": quality["contrast"],
            "saturation": quality["saturation"],
        },
    }


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Run calibrated inference and "
            "image-quality assessment"
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to lesion image",
    )

    parser.add_argument(
        "--model",
        default="models/best_model.pth",
        help="Path to trained model",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Malignant probability threshold",
    )

    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Disable temperature calibration",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model, device = load_model(
        args.model
    )

    # --------------------------------------------------------
    # LOAD CALIBRATION
    # --------------------------------------------------------

    if args.no_calibration:

        temperature = None

    else:

        temperature, available = (
            load_calibration()
        )

        if not available:
            temperature = None

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    result = predict_image(
        image_path=args.image,
        model=model,
        device=device,
        threshold=args.threshold,
        calibration_temperature=temperature,
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("DERMAVISION INFERENCE")
    print("=" * 60)

    print(
        f"Image:                    "
        f"{args.image}"
    )

    print(
        f"Prediction:               "
        f"{result['predicted_class']}"
    )

    print(
        f"Calibrated confidence:    "
        f"{result['confidence'] * 100:.2f}%"
    )

    print(
        f"Malignant probability:    "
        f"{result['malignant_probability'] * 100:.2f}%"
    )

    print(
        f"Raw malignant probability:"
        f" {result['raw_malignant_probability'] * 100:.2f}%"
    )

    print(
        f"Threshold:                "
        f"{result['threshold_used']}"
    )

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    if result["calibration_applied"]:

        print(
            f"Calibration:              "
            f"Temperature scaling "
            f"(T={result['calibration_temperature']:.4f})"
        )

    else:

        print(
            "Calibration:              "
            "Not applied"
        )

    # --------------------------------------------------------
    # IMAGE QUALITY
    # --------------------------------------------------------

    print()
    print("IMAGE QUALITY")
    print("-" * 60)

    print(
        f"Quality score:            "
        f"{result['quality_score'] * 100:.1f}%"
    )

    print(
        f"Quality status:            "
        f"{result['quality_status']}"
    )

    details = result["quality_details"]

    print(
        f"Resolution:               "
        f"{details['width']} x {details['height']}"
    )

    print(
        f"Sharpness:                "
        f"{details['sharpness']:.2f}"
    )

    print(
        f"Brightness:               "
        f"{details['brightness']:.2f}"
    )

    print(
        f"Contrast:                 "
        f"{details['contrast']:.2f}"
    )

    if result["quality_warnings"]:

        print()
        print("Warnings:")

        for warning in result["quality_warnings"]:

            print(
                f"  - {warning}"
            )

    else:

        print(
            "Warnings:                 None"
        )

    print("=" * 60)
    print()