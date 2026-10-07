"""
utils.py — Utilities for Kaggle execution, robust seeding, and offline weights.
"""

import os
import random
import numpy as np
import torch
import timm
import time
import json
from config import DATA_ROOT, IS_KAGGLE

def seed_everything(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True  # Can be slow, skip unless strictly needed
    # torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    """Seed worker processes."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def find_offline_weights(arch):
    """Search /kaggle/input for offline weights matching the arch name."""
    if not IS_KAGGLE:
        return None
    search_root = "/kaggle/input" if IS_KAGGLE else DATA_ROOT
    for root, _, files in os.walk(search_root):
        for f in files:
            if f.endswith(('.pth', '.safetensors', '.bin')) and arch in f:
                return os.path.join(root, f)
    return None

def create_backbone(arch="efficientnet_b0", pretrained=True, num_classes=2, **kwargs):
    """
    Robust timm model creation.
    1. Try downloading pretrained weights.
    2. If Internet is off, look for offline weights in DATA_ROOT.
    3. Fail loudly if pretrained=True but no weights found.
    """
    try:
        model = timm.create_model(arch, pretrained=pretrained, num_classes=num_classes, **kwargs)
        if pretrained:
            print(f"[OK] Loaded {arch} from Hugging Face hub (Internet works).")
        return model
    except Exception as e:
        if not pretrained:
            raise e
        print(f"Failed to download {arch} weights. Looking for offline weights...")
        
        offline_path = find_offline_weights(arch)
        if offline_path:
            # Create unpretrained model, then load weights
            model = timm.create_model(arch, pretrained=False, num_classes=num_classes, **kwargs)
            try:
                state_dict = torch.load(offline_path, map_location="cpu")
                # Handle standard timm format or raw state dicts
                if "model" in state_dict:
                    state_dict = state_dict["model"]
                elif "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                
                # Drop classifier keys to avoid size mismatches
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith(('head.', 'classifier.', 'fc.'))}
                model.load_state_dict(state_dict, strict=False)
                print(f"[OK] Loaded offline weights from {offline_path}")
                return model
            except Exception as e2:
                raise RuntimeError(f"Found offline weights at {offline_path} but failed to load: {e2}")
        else:
            raise RuntimeError(
                f"\n!!! PRETRAINED WEIGHTS FAILED !!!\n"
                f"Could not download {arch} (Internet off?) and no offline weights found in {DATA_ROOT}.\n"
                f"Fix: Enable Internet in Notebook settings OR attach a dataset with '{arch}' weights.\n"
                f"Do not train from random initialization!"
            )

class TimeBudget:
    """Manages time budget for Kaggle notebooks (12h max)."""
    def __init__(self, budget_hours):
        self.start_time = time.time()
        self.budget_seconds = budget_hours * 3600
        
    def get_elapsed(self):
        return time.time() - self.start_time
        
    def get_remaining(self):
        return self.budget_seconds - self.get_elapsed()
        
    def check_epoch(self, epoch_time):
        """Returns True if there's enough time for another epoch + safety margin (15 mins)."""
        return self.get_remaining() > (epoch_time + 900)
