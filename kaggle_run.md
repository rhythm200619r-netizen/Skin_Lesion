# Kaggle Training Notebook - Skin Lesion Risk (ISIC)

This notebook is designed to run the full training pipeline in the Kaggle environment. 
Follow the setup instructions precisely to avoid OOMs or network issues.

## Kaggle Environment Setup
1. **Accelerator**: GPU T4 x2 (or P100). The code defaults to 1 GPU as `DataParallel` is disabled for stability.
2. **Internet**: Turn **ON** for the first run so `timm` can download pretrained weights. If you must run offline, you MUST attach a dataset containing the `.pth` weights for the chosen architecture (e.g., `efficientnet_b0`) and place it in the `DATA_ROOT`.
3. **Persistence**: Set to **Files only** if you want `/kaggle/working` to survive notebook restarts, though saving a version will also persist it.
4. **Data Attached**: Ensure the ISIC 2020 dataset is attached (this usually mounts at `/kaggle/input/isic-2020-resized` or similar).

To fully save the outputs, you must click **"Save & Run All (Commit)"**.

---

### Cell 0: Environment Check (Runtime: < 1 min)
*Run this cell to verify GPU, internet, and data paths.*

```python
import os
import torch
import psutil
import pandas as pd
from PIL import Image

# Basic checks
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", torch.cuda.get_device_properties(0).total_memory / 1e9)
print("RAM (GB):", psutil.virtual_memory().total / 1e9)
print("CPU Cores:", os.cpu_count())

# Verify Internet
import urllib.request
try:
    urllib.request.urlopen('http://huggingface.co', timeout=3)
    print("Internet is ON (pretrained downloads will work)")
except:
    print("⚠ Internet is OFF. Pretrained models will fail unless offline weights are attached.")

# Discover dataset
DATA_ROOT = "/kaggle/input"
print("\nScanning /kaggle/input for dataset...")
csv_path, img_dir = None, None
for root, dirs, files in os.walk(DATA_ROOT):
    for f in files:
        if f.endswith(".csv") and "train" in f.lower():
            csv_path = os.path.join(root, f)
            break
    if csv_path: break

for root, dirs, files in os.walk(DATA_ROOT):
    jpgs = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(jpgs) > 100:
        img_dir = root
        break

print(f"Discovered CSV: {csv_path}")
print(f"Discovered Img Dir: {img_dir}")
if not csv_path or not img_dir:
    raise FileNotFoundError("❌ ERROR: Could not find dataset. Did you attach it?")

# Print CSV columns and assert patient_id exists
df = pd.read_csv(csv_path)
print(f"CSV Columns: {list(df.columns)}")
assert "patient_id" in df.columns, "patient_id must exist in the dataset for leakage-free splitting!"

# Print size of 3 sample images
print("\nSample image sizes:")
jpgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
for f in jpgs[:3]:
    img = Image.open(os.path.join(img_dir, f))
    print(f"{f}: {img.size}")

# Pretrained-weights load test
try:
    import timm
    _ = timm.create_model('efficientnet_b0', pretrained=True)
    print("\npretrained weights loaded: yes")
except Exception as e:
    print(f"\npretrained weights loaded: no ({e})")
```

### Cell 1: Install & Copy Repo (Runtime: < 1 min)
*The dataset paths on Kaggle are read-only. We copy our scripts to the writable `/kaggle/working` directory.*

```python
import os
import shutil

# Replace this with the actual path if the repo is attached as a dataset
REPO_SOURCE = "/kaggle/input/skin-lesion-risk-repo"  
WORK_DIR = "/kaggle/working/skin-lesion-risk"

if os.path.exists(REPO_SOURCE) and not os.path.exists(WORK_DIR):
    print(f"Copying repo from {REPO_SOURCE} to {WORK_DIR}...")
    shutil.copytree(REPO_SOURCE, WORK_DIR)
    
os.chdir(WORK_DIR)
print(f"Current working directory: {os.getcwd()}")
```

### Cell 2: Data Splitting (Runtime: < 1 min)
*Merge metadata and run the real patient-grouped split.*

```python
import sys
import subprocess
try:
    subprocess.run([sys.executable, "merge_and_split.py", "--metadata", csv_path], check=True)
except subprocess.CalledProcessError:
    raise RuntimeError("Splitting failed! Halting notebook.")
```

### Cell 3: Smoke Test & Pytest (Runtime: < 3 min)
*Ensure nothing is broken before starting a massive training run.*

```bash
# Run tests
!python -m pytest tests/

# Run the 2-epoch tiny-subset smoke test
!python src/train.py --smoke
```

### Cell 4: Resolution Decision & 1-Epoch Timing Run (Runtime: < 5 mins)
*Decide on image size and run a fast 1-epoch test to project total runtime and check initial loss.*

```python
import os
import time
import subprocess

IMG_SIZE = 224
print(f"Chosen image size: {IMG_SIZE}x{IMG_SIZE}")

start_t = time.time()
res = subprocess.run([
    "python", "src/train.py", 
    "--arch", "efficientnet_b0", 
    "--img-size", str(IMG_SIZE), 
    "--batch-size", "64", 
    "--epochs", "1", 
    "--amp", 
    "--imbalance", "sampler", 
    "--num-workers", "2",
    "--samples-per-epoch", "100"  # short run
], capture_output=True, text=True)
end_t = time.time()

duration = end_t - start_t
print(res.stdout)
if res.returncode != 0:
    print(res.stderr)
    raise RuntimeError("Timing run failed!")

print(f"\nTime per epoch (estimated for 100 steps): {duration:.1f} seconds")
print(f"Projected time for full 15 epochs (~375 steps/epoch): {duration * 3.75 * 15 / 3600:.2f} hours")
```

### Cell 5: 5-Fold Cross Validation - Fold 0 (Runtime: ~2-3 hours)
*K-fold CV is fully resumable per fold.*

```bash
!python src/train_kfold.py \
    --arch efficientnet_b0 \
    --img-size 224 \
    --batch-size 64 \
    --epochs 15 \
    --amp \
    --imbalance sampler \
    --folds 0 \
    --time-budget-hours 9.0 \
    --num-workers 2
```

### Cell 6: 5-Fold Cross Validation - Fold 1
```bash
!python src/train_kfold.py --arch efficientnet_b0 --img-size 224 --batch-size 64 --epochs 15 --amp --imbalance sampler --folds 1 --time-budget-hours 9.0 --num-workers 2
```

### Cell 7: 5-Fold Cross Validation - Fold 2
```bash
!python src/train_kfold.py --arch efficientnet_b0 --img-size 224 --batch-size 64 --epochs 15 --amp --imbalance sampler --folds 2 --time-budget-hours 9.0 --num-workers 2
```

### Cell 8: 5-Fold Cross Validation - Fold 3
```bash
!python src/train_kfold.py --arch efficientnet_b0 --img-size 224 --batch-size 64 --epochs 15 --amp --imbalance sampler --folds 3 --time-budget-hours 9.0 --num-workers 2
```

### Cell 9: 5-Fold Cross Validation - Fold 4
```bash
!python src/train_kfold.py --arch efficientnet_b0 --img-size 224 --batch-size 64 --epochs 15 --amp --imbalance sampler --folds 4 --time-budget-hours 9.0 --num-workers 2
```

### Cell 10: Fit Global Calibrator on OOF (Runtime: < 1 min)
*Calibrate the final ensemble using the Out-Of-Fold predictions.*

```bash
!python src/calibrate.py --val-probs /kaggle/working/skin-lesion-risk/outputs/oof_preds.csv --target-sens 0.90
```

### Cell 11: Final Test Evaluation & Ensemble (Runtime: ~10 mins)
*Evaluate the 5-fold ensemble with Test-Time Augmentation on the held-out test set.*

```bash
!python src/evaluate.py \
    --split test \
    --tag final_ensemble_test \
    --ensemble \
    --tta \
    --model /kaggle/working/skin-lesion-risk/models \
    --calibrator /kaggle/working/skin-lesion-risk/models/calibrator.json \
    --threshold-json /kaggle/working/skin-lesion-risk/models/threshold.json
```

### Cell 12: Final Report & Zip Outputs (Runtime: 1 min)
*Prints summary table and zips all results.*

```python
import os
import glob
import json
import pandas as pd

eval_files = glob.glob("/kaggle/working/skin-lesion-risk/outputs/eval/*_metrics.json")
records = []
for f in eval_files:
    with open(f) as fp:
        data = json.load(fp)
    records.append({
        "Tag": os.path.basename(f).replace("_metrics.json", ""),
        "AUC": f"{data['auc_roc']:.3f} [{data['auc_roc_95ci'][0]:.3f}, {data['auc_roc_95ci'][1]:.3f}]",
        "Sensitivity": f"{data['sensitivity']:.3f} [{data['sensitivity_95ci'][0]:.3f}, {data['sensitivity_95ci'][1]:.3f}]",
        "Specificity": f"{data['specificity']:.3f} [{data['specificity_95ci'][0]:.3f}, {data['specificity_95ci'][1]:.3f}]",
        "Brier": f"{data['brier_calibrated']:.4f}",
        "Brier (Const)": f"{data['brier_constant_baseline']:.4f}",
    })

if records:
    print("================ FINAL REPORT ================")
    df_rep = pd.DataFrame(records)
    print(df_rep.to_string(index=False))
    print("==============================================")
else:
    print("No evaluation metrics found.")

print("\nZipping results (excluding last.pth and caches)...")
os.system("zip -r /kaggle/working/results.zip /kaggle/working/skin-lesion-risk/outputs /kaggle/working/skin-lesion-risk/models -x \"*last.pth\" -x \"*.dat\"")
print("Done. Download results.zip from the output sidebar.")
```
