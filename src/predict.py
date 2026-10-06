"""
predict.py

Loads a trained checkpoint and runs inference on a single image.

Usage:
    python src/predict.py --image path/to/lesion.jpg --model models/best_model.pth

Output:
    image.jpg
        v
      model
        v
    Malignant
    Confidence: 78.4%

This also exposes `predict_image()` as an importable function so Person 2 can
call it directly from the integration/demo code without shelling out.
"""

import argparse

import cv2
import numpy as np
import timm
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
IMG_SIZE = 224
CLASS_NAMES = ["Benign", "Malignant"]


def load_model(checkpoint_path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device


def preprocess_image(image_path, img_size=IMG_SIZE):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor


def predict_image(image_path, model, device, threshold=0.5):
    tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        malignant_prob = probs[1].item()

    predicted_class = CLASS_NAMES[1] if malignant_prob >= threshold else CLASS_NAMES[0]
    confidence = malignant_prob if predicted_class == "Malignant" else 1 - malignant_prob

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "malignant_probability": malignant_prob,
        "threshold_used": threshold,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single lesion image")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="models/best_model.pth")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Malignant-probability cutoff (tune using outputs/threshold_sweep.json)")
    args = parser.parse_args()

    model, device = load_model(args.model)
    result = predict_image(args.image, model, device, threshold=args.threshold)

    print(f"\n{args.image}")
    print("  |")
    print("  v")
    print("model")
    print("  |")
    print("  v")
    print(f"{result['predicted_class']}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print(f"(raw malignant probability: {result['malignant_probability']*100:.1f}%, "
          f"threshold: {result['threshold_used']})")