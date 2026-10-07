"""
quality.py

Image quality assessment for Dermavision.

This module evaluates basic image characteristics that can affect
model reliability:

    - Resolution
    - Blur / sharpness
    - Brightness
    - Contrast
    - Saturation

The resulting score is a prototype quality indicator. It is NOT
a clinically validated image-quality assessment.

Usage:
    python src/quality.py --image test_images/ISIC_0052212.jpg
"""

import argparse

import cv2
import numpy as np


# ============================================================
# Thresholds
# ============================================================

MIN_WIDTH = 150
MIN_HEIGHT = 150

# Laplacian variance.
# Higher = sharper image.
BLUR_GOOD = 100.0
BLUR_ACCEPTABLE = 50.0

# Mean grayscale intensity.
BRIGHTNESS_LOW = 40.0
BRIGHTNESS_HIGH = 220.0

# Standard deviation of grayscale intensity.
# Very low contrast images tend to have a small value.
CONTRAST_LOW = 25.0

# Saturation percentage is only used as an additional signal.
SATURATION_LOW = 10.0


# ============================================================
# Individual metrics
# ============================================================

def calculate_sharpness(gray):
    """
    Estimate image sharpness using variance of Laplacian.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_brightness(gray):
    """
    Mean grayscale intensity.
    Range: approximately 0-255.
    """
    return float(np.mean(gray))


def calculate_contrast(gray):
    """
    Standard deviation of grayscale intensity.
    """
    return float(np.std(gray))


def calculate_saturation(image_rgb):
    """
    Percentage of pixels with noticeable HSV saturation.
    """
    image_hsv = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2HSV
    )

    saturation = image_hsv[:, :, 1]

    percentage = np.mean(saturation > 40) * 100.0

    return float(percentage)


# ============================================================
# Scoring
# ============================================================

def score_sharpness(value):
    if value >= BLUR_GOOD:
        return 1.0

    if value >= BLUR_ACCEPTABLE:
        return 0.7

    return 0.3


def score_brightness(value):
    if BRIGHTNESS_LOW <= value <= BRIGHTNESS_HIGH:
        return 1.0

    if 25 <= value < BRIGHTNESS_LOW:
        return 0.6

    if BRIGHTNESS_HIGH < value <= 235:
        return 0.6

    return 0.3


def score_contrast(value):
    if value >= CONTRAST_LOW:
        return 1.0

    if value >= 15:
        return 0.6

    return 0.3


def score_resolution(width, height):
    if width >= MIN_WIDTH and height >= MIN_HEIGHT:
        return 1.0

    return 0.3


def score_saturation(value):
    if value >= SATURATION_LOW:
        return 1.0

    return 0.8


# ============================================================
# Main quality assessment
# ============================================================

def assess_image_quality(image_path):
    """
    Assess basic image quality.

    Returns a dictionary containing individual metrics,
    quality score, and quality status.
    """

    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        raise FileNotFoundError(
            f"Could not read image at: {image_path}"
        )

    height, width = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    sharpness = calculate_sharpness(gray)
    brightness = calculate_brightness(gray)
    contrast = calculate_contrast(gray)
    saturation = calculate_saturation(image_rgb)

    # --------------------------------------------------------
    # Component scores
    # --------------------------------------------------------

    sharpness_score = score_sharpness(sharpness)
    brightness_score = score_brightness(brightness)
    contrast_score = score_contrast(contrast)
    resolution_score = score_resolution(
        width,
        height
    )
    saturation_score = score_saturation(
        saturation
    )

    # --------------------------------------------------------
    # Weighted quality score
    # --------------------------------------------------------

    quality_score = (
        0.35 * sharpness_score
        + 0.20 * brightness_score
        + 0.20 * contrast_score
        + 0.20 * resolution_score
        + 0.05 * saturation_score
    )

    quality_score = float(
        np.clip(
            quality_score,
            0.0,
            1.0
        )
    )

    # --------------------------------------------------------
    # Quality classification
    # --------------------------------------------------------

    if quality_score >= 0.80:
        quality_status = "Good"

    elif quality_score >= 0.60:
        quality_status = "Acceptable"

    else:
        quality_status = "Poor"

    # --------------------------------------------------------
    # Human-readable warnings
    # --------------------------------------------------------

    warnings = []

    if sharpness < BLUR_ACCEPTABLE:
        warnings.append(
            "Image may be blurry"
        )

    if brightness < BRIGHTNESS_LOW:
        warnings.append(
            "Image may be too dark"
        )

    elif brightness > BRIGHTNESS_HIGH:
        warnings.append(
            "Image may be too bright"
        )

    if contrast < CONTRAST_LOW:
        warnings.append(
            "Low image contrast"
        )

    if width < MIN_WIDTH or height < MIN_HEIGHT:
        warnings.append(
            "Low image resolution"
        )

    return {
        "width": width,
        "height": height,
        "sharpness": sharpness,
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,

        "sharpness_score": sharpness_score,
        "brightness_score": brightness_score,
        "contrast_score": contrast_score,
        "resolution_score": resolution_score,
        "saturation_score": saturation_score,

        "quality_score": quality_score,
        "quality_status": quality_status,
        "warnings": warnings,
    }


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Assess skin lesion image quality"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to lesion image"
    )

    args = parser.parse_args()

    result = assess_image_quality(
        args.image
    )

    print()
    print("=" * 60)
    print("DERMAVISION IMAGE QUALITY ASSESSMENT")
    print("=" * 60)

    print(
        f"Image:             {args.image}"
    )

    print(
        f"Resolution:        "
        f"{result['width']} x {result['height']}"
    )

    print(
        f"Sharpness:         "
        f"{result['sharpness']:.2f}"
    )

    print(
        f"Brightness:        "
        f"{result['brightness']:.2f}"
    )

    print(
        f"Contrast:          "
        f"{result['contrast']:.2f}"
    )

    print(
        f"Saturation:        "
        f"{result['saturation']:.2f}%"
    )

    print()
    print(
        f"Quality score:     "
        f"{result['quality_score'] * 100:.1f}%"
    )

    print(
        f"Quality status:    "
        f"{result['quality_status']}"
    )

    if result["warnings"]:

        print()
        print("Warnings:")

        for warning in result["warnings"]:
            print(f"  - {warning}")

    else:

        print()
        print("Warnings:          None")

    print("=" * 60)
    print()


if __name__ == "__main__":
    main()