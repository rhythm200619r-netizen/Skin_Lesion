"""
config.py — Kaggle-aware configuration and path management.

Auto-detects if running on Kaggle (/kaggle/working) and sets paths accordingly.
Locates the dataset automatically in DATA_ROOT.
"""

import os
import glob

# Auto-detect Kaggle environment
IS_KAGGLE = os.path.exists("/kaggle")

if IS_KAGGLE:
    # On Kaggle, /kaggle/input is read-only.
    # /kaggle/working is writable and preserved.
    # Dataset will be mapped under /kaggle/input (exact name varies)
    DATA_ROOT = "/kaggle/input"
    WORK_DIR = "/kaggle/working/skin-lesion-risk" # Assume repo is copied here
    
    # Kaggle notebooks start in /kaggle/working
    if not os.path.exists(WORK_DIR):
        # Fallback if running directly in /kaggle/working without a subfolder
        WORK_DIR = "/kaggle/working"
else:
    # Local environment
    DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    WORK_DIR = os.path.dirname(os.path.dirname(__file__))

MODELS_DIR = os.path.join(WORK_DIR, "models")
OUTPUTS_DIR = os.path.join(WORK_DIR, "outputs")

def get_data_paths():
    """
    Scans DATA_ROOT to find the training CSV and the image folder.
    Returns (csv_path, img_dir).
    """
    csv_path = None
    img_dir = None
    
    if IS_KAGGLE:
        # Search for train.csv or similar in /kaggle/input
        # Prioritize files containing 'train' and '.csv'
        for root, dirs, files in os.walk(DATA_ROOT):
            for f in files:
                if f.endswith(".csv") and "train" in f.lower():
                    csv_path = os.path.join(root, f)
                    break
            if csv_path:
                break
        
        # Search for folder with images
        for root, dirs, files in os.walk(DATA_ROOT):
            jpgs = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(jpgs) > 100:
                img_dir = root
                break
    else:
        # Local defaults
        csv_path = os.path.join(DATA_ROOT, "train.csv")
        img_dir = os.path.join(DATA_ROOT, "jpeg", "train")
        if not os.path.exists(img_dir):
            img_dir = DATA_ROOT # fallback
            
    return csv_path, img_dir

def setup_dirs():
    """Ensure working directories exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
