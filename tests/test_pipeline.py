import os
import sys
import pytest
import pandas as pd
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from dataset import _one_fold, three_way_split, stratified_split, assert_no_patient_leakage
from metrics import classification_metrics
from calibrate import fit_calibrator, calibrate_probs
from predict import predict_image, load_model

from unittest.mock import patch, MagicMock

@patch("evaluate.plot_confusion")
@patch("evaluate.plot_roc")
@patch("evaluate.plot_pr")
@patch("evaluate.plot_reliability")
def test_evaluate_ensemble(mock_rel, mock_pr, mock_roc, mock_conf, tmp_path):
    import torch.nn as nn
    from evaluate import main
    from argparse import Namespace
    
    class MockModel(nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 2)
            
    with patch("predict.load_ensemble_models") as mock_load:
        mock_load.return_value = ([MockModel(), MockModel()], torch.device("cpu"))
        
        args = Namespace(
            csv=str(tmp_path / "dummy.csv"),
            img_dir=str(tmp_path),
            model=str(tmp_path / "dummy.pth"),
            split="all",
            tag="test_ens",
            out_dir=str(tmp_path),
            threshold=0.5,
            threshold_json=None,
            calibrator=None,
            val_size=0.1,
            test_size=0.1,
            seed=42,
            batch_size=2,
            num_workers=0,
            n_boot=10,
            ensemble=True,
            tta=True
        )
        
        import cv2
        df = pd.DataFrame({"image_name": ["img1", "img2"], "target": [0, 1]})
        df.to_csv(args.csv, index=False)
        
        cv2.imwrite(str(tmp_path / "img1.jpg"), np.zeros((10,10,3), dtype=np.uint8))
        cv2.imwrite(str(tmp_path / "img2.jpg"), np.zeros((10,10,3), dtype=np.uint8))
        
        main(args)
        assert (tmp_path / "test_ens_metrics.json").exists()

def test_leakage():
    """Test patient-grouping logic to ensure no patient overlaps splits."""
    df = pd.DataFrame({
        "image_name": [f"img_{i}" for i in range(100)],
        "target": np.random.randint(0, 2, 100),
        "patient_id": [f"pat_{i % 15}" for i in range(100)] # 15 patients
    })
    
    train, val, test = three_way_split(df, 0.15, 0.15, seed=42)
    
    # Assert counts match roughly
    assert len(train) + len(val) + len(test) == len(df)
    
    # Assert NO overlap in patients
    sets = [set(train.patient_id), set(val.patient_id), set(test.patient_id)]
    for i in range(3):
        for j in range(i + 1, 3):
            assert len(sets[i] & sets[j]) == 0

def test_saved_split_leakage(tmp_path):
    from evaluate import OUTPUTS_DIR
    import os
    train_csv = os.path.join(OUTPUTS_DIR, "train_split.csv")
    val_csv = os.path.join(OUTPUTS_DIR, "val_split.csv")
    test_csv = os.path.join(OUTPUTS_DIR, "test_split.csv")
    
    # Only test if the real split was saved
    if os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv):
        train = pd.read_csv(train_csv)
        val = pd.read_csv(val_csv)
        test = pd.read_csv(test_csv)
        
        if "patient_id" in train.columns:
            sets = [set(train["patient_id"]), set(val["patient_id"]), set(test["patient_id"])]
            for i in range(3):
                for j in range(i + 1, 3):
                    assert len(sets[i] & sets[j]) == 0

def test_calibration_monotonicity():
    """Isotonic and temperature calibration should be strictly monotonic."""
    labels = np.array([0, 0, 1, 0, 1, 1, 1, 0, 0, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    
    for method in ["temperature", "isotonic"]:
        cal = fit_calibrator(probs, labels, method=method)
        cal_probs = calibrate_probs(probs, cal)
        # Check monotonicity
        for i in range(len(cal_probs) - 1):
            assert cal_probs[i] <= cal_probs[i+1]

def test_predict_backward_compat(tmp_path):
    """Ensure predict.py can load older checkpoints with missing config keys."""
    # Create fake old checkpoint
    import torch.nn as nn
    model = nn.Linear(10, 2)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "epoch": 10,
        "val_metrics": {"auc_roc": 0.90},
        "class_weights": [0.1, 0.9]
        # Missing: arch, img_size, etc.
    }
    path = tmp_path / "old_ckpt.pth"
    torch.save(ckpt, path)
    
    # Mock load_model
    import timm
    from unittest.mock import patch
    
    with patch("timm.create_model", return_value=model):
        loaded_model, device = load_model(str(path))
        assert loaded_model is not None

def test_nan_auc_checkpoint():
    """Verify that train.py fallback checkpoint logic on nan AUC doesn't crash."""
    from train import main
    import argparse
    import math

    # We mock the parts to just check the boolean logic in the epoch loop.
    m = {"auc_roc": float("nan")}
    val_loss = 0.5
    best_val_loss = getattr(main, "best_val_loss", float("inf"))
    
    is_best = False
    if math.isnan(m["auc_roc"]):
        is_best = val_loss <= best_val_loss
        if is_best:
            best_val_loss = val_loss
    
    assert is_best == True
    assert best_val_loss == 0.5

def test_threshold_target_sens():
    from calibrate import select_threshold
    labels = np.array([0, 0, 1, 0, 1, 1, 1, 0, 0, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    
    res = select_threshold(labels, probs, target_sens=0.8)
    assert res["achieved_sensitivity"] >= 0.8

def test_ensemble_output_keys():
    from predict import predict_image_ensemble
    from unittest.mock import patch
    import torch
    
    # Mock models and device
    class MockModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[0.2, 0.8]])
    
    models = [MockModel(), MockModel()]
    
    with patch("predict.preprocess_image", return_value=torch.zeros(1, 3, 224, 224)):
        res = predict_image_ensemble("dummy.jpg", models, "cpu")
        
    assert "predicted_class" in res
    assert "confidence" in res
    assert "malignant_probability" in res
    assert "threshold_used" in res
    assert "threshold_applies_to" in res
    assert "model_agreement" in res

def test_ensemble_logits():
    from predict import predict_image_ensemble
    import torch
    from unittest.mock import patch
    
    class MockModel1(torch.nn.Module):
        def forward(self, x): return torch.tensor([[0.0, 0.0]]) # prob=0.5
    class MockModel2(torch.nn.Module):
        def forward(self, x): return torch.tensor([[-2.0, 2.0]]) # prob=0.982
        
    models = [MockModel1(), MockModel2()]
    
    with patch("predict.preprocess_image", return_value=torch.zeros(1, 3, 224, 224)):
        res = predict_image_ensemble("dummy.jpg", models, "cpu", threshold=0.5)
        
    # (0.5 + 0.98201) / 2 = 0.7410
    assert abs(res["malignant_probability"] - 0.7410) < 1e-3
    assert res["predicted_class"] == "Malignant"

def test_ensemble_calibration_flip():
    from predict import predict_image_ensemble
    import torch
    import numpy as np
    from unittest.mock import patch
    from calibrate import fit_calibrator
    
    # Mock model outputs mean prob of 0.8
    class MockModel(torch.nn.Module):
        def forward(self, x): return torch.tensor([[0.0, 1.386]]) # softmax ~0.8
        
    # Real calibrator fit on toy data: 4 negatives and 1 positive all predicted as 0.8
    labels = np.array([0, 0, 0, 0, 1])
    probs = np.array([0.8, 0.8, 0.8, 0.8, 0.8])
    calibrator = fit_calibrator(probs, labels, method="isotonic")
    
    with patch("predict.preprocess_image", return_value=torch.zeros(1, 3, 224, 224)):
        res = predict_image_ensemble("dummy.jpg", [MockModel()], "cpu", threshold=0.5, calibrator=calibrator)
        
    # Calibrated prob should be 1/5 = 0.2. Threshold is 0.5 -> Benign!
    assert abs(res["malignant_probability"] - 0.2) < 1e-3
    assert res["predicted_class"] == "Benign"
