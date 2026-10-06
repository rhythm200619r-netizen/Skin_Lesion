"""
quality_stub.py  -- CONTRACT for Person 2 (GUI / retake guidance).

Returns hard-coded dummy values with the FINAL output shape, so the GUI can be built now.
Person 1 will replace this with the real src/quality.py + src/calibration.py (same function
names and keys), so the only change on Person 2's side is the import line.

Agreed `problems` strings (retake guidance maps from these):
    "blur", "too_dark", "too_bright", "low_contrast", "low_resolution", "poor_framing"
"""

PROBLEMS = ["blur", "too_dark", "too_bright", "low_contrast", "low_resolution", "poor_framing"]


def assess_quality(image_path):
    return {
        "score": 82,                      # 0-100 overall
        "label": "Good",                  # "Good" | "Moderate" | "Poor"
        "subscores": {"blur": 90, "brightness": 85, "contrast": 80, "resolution": 95, "framing": 82},
        "problems": [],                   # subset of PROBLEMS, e.g. ["blur", "too_dark"]
    }


def calibrate_confidence(malignant_prob, quality):
    p = float(malignant_prob)
    return {
        "raw_confidence": max(p, 1 - p),
        "adjusted_confidence": max(p, 1 - p),
        "adjusted_malignant_prob": p,
        "level": "High",                  # "High" | "Low"
        "retake": False,                  # True -> withhold prediction, show retake guidance
    }
