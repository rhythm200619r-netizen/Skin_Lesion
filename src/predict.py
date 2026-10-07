"""
predict.py — Single-image inference (CLI + importable API).

Backward-compatible API:
    predict_image(image_path, model, device, threshold=0.5) -> dict
    load_model(checkpoint_path, device=None) -> (model, device)

Extended API (same return keys + extras):
    predict_image_ensemble(image_path, models, device, threshold, calibrator, tta) -> dict

Usage:
    python src/predict.py --image path/to/lesion.jpg --model models/best_model.pth
    python src/predict.py --image path/to/lesion.jpg --ensemble  # uses all fold models
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import timm
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
IMG_SIZE = 224
CLASS_NAMES = ["Benign", "Malignant"]


def load_model(checkpoint_path, device=None):
    """Load a trained checkpoint. Backward compatible."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Auto-detect architecture and img_size from checkpoint
    arch = checkpoint.get("arch", "efficientnet_b0")
    img_size = checkpoint.get("img_size", 224)
    num_classes = 2

    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device


def preprocess_image(image_path, img_size=IMG_SIZE):
    """Load and preprocess a single image for inference."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor


def predict_image(image_path, model, device, threshold=0.5, calibrator=None):
    """
    Run inference on a single image.

    Returns:
        dict with keys: predicted_class, confidence, malignant_probability, threshold_used, threshold_applies_to
    """
    tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        malignant_prob = probs[1].item()

    if calibrator is not None:
        from calibrate import calibrate_probs
        malignant_prob = float(calibrate_probs(np.array([malignant_prob]), calibrator)[0])

    predicted_class = CLASS_NAMES[1] if malignant_prob >= threshold else CLASS_NAMES[0]
    confidence = malignant_prob if predicted_class == "Malignant" else 1 - malignant_prob

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "malignant_probability": malignant_prob,
        "threshold_used": threshold,
        "threshold_applies_to": "calibrated" if calibrator is not None else "raw",
    }


# ──────────────────────────────────────────────────
# TTA (Test-Time Augmentation)
# ──────────────────────────────────────────────────

def _tta_variants(img):
    """Generate 8 dihedral variants of a numpy HWC image."""
    variants = []
    for hflip in [False, True]:
        for k in range(4):
            v = img.copy()
            if hflip:
                v = np.flip(v, axis=1).copy()
            if k > 0:
                v = np.rot90(v, k).copy()
            variants.append(v)
    return variants


def predict_image_single_tta(image_path, model, device, img_size=IMG_SIZE):
    """Run TTA on a single model, return mean P(malignant) over 8 dihedral variants."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

    variants = _tta_variants(img)
    probs_list = []

    model.eval()
    with torch.no_grad():
        for v in variants:
            v_norm = v.astype(np.float32) / 255.0
            v_norm = (v_norm - IMAGENET_MEAN) / IMAGENET_STD
            tensor = torch.from_numpy(v_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            out = model(tensor)
            p = torch.softmax(out, dim=1)[0, 1].item()
            probs_list.append(p)

    return float(np.mean(probs_list))


# ──────────────────────────────────────────────────
# Ensemble prediction
# ──────────────────────────────────────────────────

def load_ensemble_models(model_dir=None, pattern="fold"):
    """Load all fold models from model_dir."""
    from config import MODELS_DIR
    if model_dir is None:
        model_dir = MODELS_DIR
    models_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for f in sorted(os.listdir(model_dir)):
        if f.startswith(pattern) and f.endswith(".pth"):
            path = os.path.join(model_dir, f)
            model, _ = load_model(path, device)
            models_list.append(model)
    if not models_list:
        # Fall back to best_model.pth
        best_path = os.path.join(model_dir, "best_model.pth")
        if os.path.exists(best_path):
            model, _ = load_model(best_path, device)
            models_list.append(model)
    return models_list, device


def predict_image_ensemble(image_path, models, device, threshold=0.5,
                           calibrator=None, tta=False, img_size=IMG_SIZE):
    """
    Ensemble prediction over multiple models, with optional TTA and calibration.

    Returns same keys as predict_image plus:
        - model_agreement: std of member probabilities (uncertainty signal)
    """
    member_probs = []
    for model in models:
        if tta:
            p = predict_image_single_tta(image_path, model, device, img_size)
        else:
            tensor = preprocess_image(image_path, img_size).to(device)
            model.eval()
            with torch.no_grad():
                out = model(tensor)
                p = torch.softmax(out, dim=1)[0, 1].item()
        member_probs.append(p)

    # Average probabilities
    mean_prob = float(np.mean(member_probs))

    # Apply calibration if available
    if calibrator is not None:
        from calibrate import calibrate_probs
        mean_prob = float(calibrate_probs(np.array([mean_prob]), calibrator)[0])

    predicted_class = CLASS_NAMES[1] if mean_prob >= threshold else CLASS_NAMES[0]
    confidence = mean_prob if predicted_class == "Malignant" else 1 - mean_prob
    model_agreement = float(np.std(member_probs))

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "malignant_probability": mean_prob,
        "threshold_used": threshold,
        "threshold_applies_to": "calibrated" if calibrator is not None else "raw",
        "model_agreement": model_agreement,
    }


# ──────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────

from config import MODELS_DIR

# ... later in __main__ ...
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single lesion image")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default=os.path.join(MODELS_DIR, "best_model.pth"))
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold (auto-loaded from models/threshold.json if not set)")
    parser.add_argument("--calibrator", default=None,
                        help="Path to calibrator.json (auto-loaded from models/calibrator.json)")
    parser.add_argument("--ensemble", action="store_true",
                        help="Use all fold models for ensemble prediction")
    parser.add_argument("--tta", action="store_true",
                        help="Enable test-time augmentation (8 dihedral variants)")
    args = parser.parse_args()

    # Helper to compute current run's hash
    import hashlib
    def get_run_hash(is_ensemble, model_path):
        if not is_ensemble:
            with open(model_path, "rb") as mf:
                return hashlib.md5(mf.read()).hexdigest()
        else:
            model_dir = os.path.dirname(model_path)
            pth_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".pth")])
            hashes = []
            for f in pth_files:
                with open(os.path.join(model_dir, f), "rb") as mf:
                    hashes.append(hashlib.md5(mf.read()).hexdigest())
            return hashlib.md5("".join(hashes).encode()).hexdigest()
            
    current_hash = get_run_hash(args.ensemble, args.model)

    # Auto-load threshold
    threshold = args.threshold
    threshold_applies_to = "raw"
    if args.threshold is None:
        thr_path = os.path.join(os.path.dirname(args.model), "threshold.json")
        if os.path.exists(thr_path):
            with open(thr_path) as f:
                thr_data = json.load(f)
            threshold = thr_data.get("threshold_target_sens", 0.5)
            threshold_applies_to = thr_data.get("applies_to", "unknown")
            print(f"Auto-loaded threshold: {threshold:.4f} (applies to {threshold_applies_to}) from {thr_path}")
            
            expected_hash = thr_data.get("weights_hash", "")
            if expected_hash and expected_hash != "unknown":
                if current_hash != expected_hash:
                    raise ValueError(f"Model weights hash ({current_hash}) does not match threshold.json ({expected_hash})!")
        else:
            threshold = 0.5

    # Auto-load calibrator
    calibrator = None
    cal_path = args.calibrator or os.path.join(os.path.dirname(args.model), "calibrator.json")
    if os.path.exists(cal_path):
        with open(cal_path) as f:
            calibrator = json.load(f)
        print(f"Auto-loaded calibrator ({calibrator.get('method', 'unknown')}) from {cal_path}")
        
        cal_hash = calibrator.get("weights_hash", "")
        if cal_hash and cal_hash != "unknown":
            if current_hash != cal_hash:
                raise ValueError(f"Model weights hash does not match calibrator.json!")

    # Validate applies_to
    if threshold_applies_to != "unknown":
        expected = "calibrated" if calibrator is not None else "raw"
        if threshold_applies_to != expected:
            raise ValueError(f"threshold.json applies_to '{threshold_applies_to}' but the current run is using '{expected}' probabilities! (Missing calibrator?)")

    if args.ensemble:
        models, device = load_ensemble_models(os.path.dirname(args.model))
        print(f"Loaded {len(models)} ensemble models")
        result = predict_image_ensemble(
            args.image, models, device, threshold=threshold,
            calibrator=calibrator, tta=args.tta,
        )
    else:
        model, device = load_model(args.model)
        if args.tta:
            raw_prob = predict_image_single_tta(args.image, model, device)
            if calibrator is not None:
                from calibrate import calibrate_probs
                raw_prob = float(calibrate_probs(np.array([raw_prob]), calibrator)[0])
            predicted_class = CLASS_NAMES[1] if raw_prob >= threshold else CLASS_NAMES[0]
            confidence = raw_prob if predicted_class == "Malignant" else 1 - raw_prob
            result = {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "malignant_probability": raw_prob,
                "threshold_used": threshold,
            }
        else:
            result = predict_image(args.image, model, device, threshold=threshold, calibrator=calibrator)

    print(f"\n{args.image}")
    print("  |")
    print("  v")
    print("model")
    print("  |")
    print("  v")
    print(f"{result['predicted_class']}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print(f"(raw malignant probability: {result['malignant_probability']*100:.1f}%, "
          f"threshold: {result['threshold_used']:.4f})")
    if "model_agreement" in result:
        print(f"Model agreement (std): {result['model_agreement']:.4f}")