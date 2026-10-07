"""
gradcam.py

Generate a Grad-CAM visualization for an EfficientNet-B0 skin lesion model.

Usage:
    python src/gradcam.py --image test_images/ISIC_0052212.jpg

Output:
    outputs/gradcam_ISIC_0052212.png
"""

import argparse
import os

import cv2
import numpy as np
import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from predict import (
    CLASS_NAMES,
    IMG_SIZE,
    load_model,
    preprocess_image,
)


def generate_gradcam(
    model,
    input_tensor,
    target_layer,
    target_class,
):
    """
    Generate a Grad-CAM visualization for the specified target layer.
    """

    targets = [
        ClassifierOutputTarget(target_class)
    ]

    with GradCAM(
        model=model,
        target_layers=[target_layer],
    ) as cam:

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets,
        )[0]

    # ---------------------------------------------------------
    # Reconstruct the original RGB image
    # ---------------------------------------------------------

    rgb_img = (
        input_tensor[0]
        .detach()
        .cpu()
        .numpy()
    )

    mean = np.array(
        [0.485, 0.456, 0.406]
    ).reshape(3, 1, 1)

    std = np.array(
        [0.229, 0.224, 0.225]
    ).reshape(3, 1, 1)

    # Undo ImageNet normalization
    rgb_img = (
        rgb_img * std
    ) + mean

    # CHW -> HWC
    rgb_img = np.transpose(
        rgb_img,
        (1, 2, 0),
    )

    # Keep values in valid image range
    rgb_img = np.clip(
        rgb_img,
        0,
        1,
    ).astype(np.float32)

    # ---------------------------------------------------------
    # Overlay Grad-CAM on original image
    # ---------------------------------------------------------

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True,
    )

    return visualization


def main():

    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM explanation for a skin lesion image"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the lesion image",
    )

    parser.add_argument(
        "--model",
        default="models/best_model.pth",
        help="Path to the trained model checkpoint",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Malignant probability threshold",
    )

    args = parser.parse_args()

    # =========================================================
    # HEADER
    # =========================================================

    print()
    print("=" * 55)
    print("GRAD-CAM ANALYSIS")
    print("=" * 55)

    # =========================================================
    # LOAD MODEL
    # =========================================================

    model, device = load_model(
        args.model
    )

    print(
        f"Device:                 {device}"
    )

    # =========================================================
    # LOAD IMAGE
    # =========================================================

    input_tensor = preprocess_image(
        args.image
    ).to(device)

    # =========================================================
    # MODEL PREDICTION
    # =========================================================

    with torch.no_grad():

        outputs = model(
            input_tensor
        )

        probs = torch.softmax(
            outputs,
            dim=1,
        )[0]

        malignant_probability = (
            probs[1].item()
        )

    # Determine predicted class
    if malignant_probability >= args.threshold:
        predicted_class = 1
    else:
        predicted_class = 0

    predicted_name = (
        CLASS_NAMES[predicted_class]
    )

    # Confidence corresponds to the predicted class
    if predicted_class == 1:
        confidence = malignant_probability
    else:
        confidence = 1 - malignant_probability

    print(
        f"Image:                  {args.image}"
    )

    print(
        f"Prediction:             {predicted_name}"
    )

    print(
        f"Confidence:             "
        f"{confidence * 100:.2f}%"
    )

    print(
        f"Raw malignant probability: "
        f"{malignant_probability * 100:.2f}%"
    )

    print(
        f"Threshold:              "
        f"{args.threshold}"
    )

    # =========================================================
    # GRAD-CAM TARGET LAYER
    # =========================================================

    # Selected after comparing:
    #   model.blocks[-2]
    #   model.blocks[-1]
    #   model.conv_head
    #
    # blocks[-1] provided the most useful deep feature
    # localization across the tested images.

    target_layer = model.blocks[-1]

    # =========================================================
    # GENERATE GRAD-CAM
    # =========================================================

    print()
    print("Generating Grad-CAM...")

    visualization = generate_gradcam(
        model=model,
        input_tensor=input_tensor,
        target_layer=target_layer,
        target_class=predicted_class,
    )

    # =========================================================
    # SAVE RESULT
    # =========================================================

    os.makedirs(
        "outputs",
        exist_ok=True,
    )

    base_name = os.path.splitext(
        os.path.basename(args.image)
    )[0]

    output_path = os.path.join(
        "outputs",
        f"gradcam_{base_name}.png",
    )

    cv2.imwrite(
        output_path,
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR,
        ),
    )

    print(
        f"Grad-CAM output:       {output_path}"
    )

    print()
    print("=" * 55)
    print("GRAD-CAM COMPLETE")
    print("=" * 55)
    print()


if __name__ == "__main__":
    main()