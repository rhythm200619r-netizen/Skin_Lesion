# 🩺 DermaVision — Skin Lesion Malignancy Risk Prediction

An end-to-end deep learning framework and screening application for classifying dermoscopic skin lesions as **Benign** or **Malignant** using the ISIC dataset. The pipeline is engineered with patient-grouped data splitting to prevent leakage, class-imbalance mitigation, threshold calibration for high melanoma recall, comprehensive statistical evaluation, and an interactive modern desktop GUI.

---

## 📌 Table of Contents

- [Overview & Clinical Motivation](#-overview--clinical-motivation)
- [Key Features](#-key-features)
- [Project Architecture](#-project-architecture)
- [Installation & Setup](#-installation--setup)
- [Dataset Preparation](#-dataset-preparation)
- [Usage Guide](#-usage-guide)
  - [1. Generate Stratified Subset](#1-generate-stratified-subset)
  - [2. Train EfficientNet-B0](#2-train-efficientnet-b0)
  - [3. Comprehensive Evaluation](#3-comprehensive-evaluation)
  - [4. Single Image CLI Inference](#4-single-image-cli-inference)
  - [5. Run Interactive Desktop GUI](#5-run-interactive-desktop-gui)
- [Evaluation Metrics & Validation Results](#-evaluation-metrics--validation-results)
- [Quality Assessment & Confidence Calibration Architecture](#-quality-assessment--confidence-calibration-architecture)
- [Disclaimer](#-disclaimer)

---

## 🔬 Overview & Clinical Motivation

Melanoma is one of the most aggressive forms of skin cancer, where early and accurate detection significantly improves patient survival rates. However, clinical dermoscopy workflows face major challenges:
1. **Severe Class Imbalance**: Malignant lesions typically constitute only ~1.7% of public screening datasets like ISIC 2020 (~580 malignant vs. ~32,500 benign).
2. **Patient Data Leakage**: Multiple lesion photos from the same patient appearing across training and testing splits inflate benchmark metrics artificially.
3. **Clinical Priority (Recall > Accuracy)**: Missing a malignant lesion (false negative) carries catastrophic clinical consequences compared to a false positive benign biopsy referral.
4. **Image Quality Degradation**: Blurry, overexposed, or poorly framed photos degrade deep learning predictions without alerting the operator.

**DermaVision** addresses these challenges by combining:
- Patient-grouped stratified 3-way partitioning (`Train / Val / Test`).
- ImageNet-pretrained `EfficientNet-B0` with class-frequency loss reweighting.
- Multi-metric evaluation (AUC-ROC, PR-AUC, ECE calibration error, Brier score, and 95% bootstrap confidence intervals).
- Configurable decision thresholding prioritized for melanoma sensitivity.
- A modern desktop GUI (`DermaVision`) built with CustomTkinter for real-time visual screening.

---

## ✨ Key Features

- **🛡️ Leakage-Free Stratification**: Employs `StratifiedGroupKFold` on `patient_id` to guarantee no patient is split across train, validation, and test sets.
- **⚡ Fast Iteration Subset Builder**: `src/make_subset.py` extracts all rare malignant images while sampling a balanced ratio of benign images for rapid experimentation.
- **🧠 EfficientNet-B0 Backbone**: Pretrained on ImageNet via `timm`, optimized with Adam and `ReduceLROnPlateau`, with optional mixed precision (`--amp`) for accelerated training.
- **⚖️ Weighted Cross-Entropy**: Inversely weights benign vs. malignant loss based on class frequencies to counter severe data imbalance.
- **📊 Robust Statistical Evaluation**: `src/evaluate.py` outputs full classification metrics, ROC curves, confusion matrices, reliability curves, Expected Calibration Error (ECE), and 1000-fold bootstrap 95% confidence intervals.
- **🖥️ DermaVision Desktop App**: Sleek, dark-mode desktop screening interface built with CustomTkinter featuring real-time image preview, probability meters, and risk summaries.

---

## 📁 Project Architecture

```
skin-lesion-risk/
├── data/                         # ISIC metadata and images (ignored by git)
│   ├── train.csv                 # Full metadata (image_name, patient_id, target)
│   ├── subset_train.csv          # Generated stratified subset metadata
│   └── jpeg/train/               # Full/subset dermoscopy JPEG images
├── models/
│   └── best_model.pth            # Best checkpoint saved by validation AUC-ROC
├── outputs/                      # Training and evaluation artifacts
│   ├── confusion_matrix.png      # Validation confusion matrix
│   ├── threshold_sweep.json      # Metric sweeps across cutoffs (0.10 - 0.90)
│   ├── training_history.json     # Epoch-by-epoch loss and metrics
│   ├── train_split.csv           # Exact train split metadata
│   ├── val_split.csv             # Exact validation split metadata
│   ├── test_split.csv            # Exact test split metadata
│   └── eval/                     # Evaluation outputs (ROC, ECE, reliability plots)
├── src/
│   ├── dataset.py                # Dataset loader, augmentations & patient-grouped split
│   ├── make_subset.py            # Stratified subset generator
│   ├── train.py                  # Training pipeline with AMP and AUC checkpointing
│   ├── evaluate.py               # Test split evaluation, bootstrap CIs & reliability
│   ├── metrics.py                # Pure NumPy/Scikit-Learn metric computation & ECE
│   ├── predict.py                # Single-image inference module & CLI tool
│   ├── demo.py                   # DermaVision CustomTkinter GUI application
│   └── quality_stub.py           # Contract interface for image quality & calibration
├── test_images/                  # Sample images for testing inference
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9+ (Python 3.10 / 3.11 recommended)
- Optional: NVIDIA GPU with CUDA support for mixed precision training

### 1. Clone the Repository & Create Virtual Environment
```bash
git clone https://github.com/rhythm200619r-netizen/Skin_Lesion.git
cd Skin_Lesion

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🗂️ Dataset Preparation

The pipeline is designed for the **ISIC 2020 Challenge** dataset (or compatible ISIC archives containing `image_name`, `patient_id`, and `target`):

1. Download the dataset from [ISIC Archive / Kaggle](https://www.kaggle.com/c/isic-2020-challenge/data).
2. Extract the files into the `data/` folder:
   ```
   data/
   ├── train.csv
   └── jpeg/
       └── train/
           ├── ISIC_0015719.jpg
           ├── ISIC_0052212.jpg
           └── ...
   ```

---

## 🚀 Usage Guide

### 1. Generate Stratified Subset
For fast experimentation without processing all 33k images, build a subset that **keeps 100% of malignant lesions** and samples benign lesions at a designated ratio:

```bash
python src/make_subset.py \
    --csv data/train.csv \
    --img-dir data/jpeg/train \
    --out-csv data/subset_train.csv \
    --target-size 6000 \
    --benign-malignant-ratio 8.0
```

---

### 2. Train EfficientNet-B0
Train the model with patient-grouped splitting, class weighting, and validation AUC-ROC checkpointing:

```bash
python src/train.py \
    --csv data/subset_train.csv \
    --img-dir data/jpeg/train \
    --epochs 12 \
    --batch-size 32 \
    --lr 0.0001 \
    --val-size 0.15 \
    --test-size 0.15 \
    --amp
```

**Key Arguments:**
- `--csv`: Path to dataset CSV.
- `--img-dir`: Path to folder containing image files.
- `--epochs`: Number of training epochs (default: `12`).
- `--batch-size`: Mini-batch size (default: `32`).
- `--lr`: Learning rate for Adam optimizer (default: `1e-4`).
- `--val-size` / `--test-size`: Proportions for patient-grouped splits (default: `0.15` each).
- `--amp`: Enable automatic mixed precision (NVIDIA CUDA only).

---

### 3. Comprehensive Evaluation
Evaluate the trained checkpoint on the held-out test split, generating ROC curves, calibration diagrams, confusion matrices, and bootstrap confidence intervals:

```bash
python src/evaluate.py \
    --csv data/subset_train.csv \
    --img-dir data/jpeg/train \
    --model models/best_model.pth \
    --split test \
    --tag test_eval \
    --out-dir outputs/eval \
    --threshold 0.5 \
    --n-boot 1000
```

**Generated Artifacts (`outputs/eval/`):**
- `test_eval_metrics.json`: Accuracy, Precision, Recall/Sensitivity, Specificity, F1, AUC-ROC, ECE, Brier score, and 95% CIs.
- `test_eval_roc.png`: Receiver Operating Characteristic (ROC) curve with area under curve.
- `test_eval_reliability.png`: Calibration curve plotting observed vs. predicted probabilities.
- `test_eval_confusion.png`: Confusion matrix at the chosen threshold.
- `test_eval_probs.csv`: Sample-by-sample probabilities for downstream analysis.

---

### 4. Single Image CLI Inference
Run predictions on individual dermoscopy images:

```bash
python src/predict.py \
    --image test_images/ISIC_0052212.jpg \
    --model models/best_model.pth \
    --threshold 0.5
```

**Example Output:**
```text
test_images/ISIC_0052212.jpg
  |
  v
model
  |
  v
Benign
Confidence: 91.4%
(raw malignant probability: 8.6%, threshold: 0.5)
```

---

### 5. Run Interactive Desktop GUI

Launch the **DermaVision** desktop application:

```bash
python src/demo.py
```

#### GUI Highlights:
- **Image Upload & Live Preview**: Browse and load local dermoscopic images.
- **Instant Risk Classification**: Clear visual status indicator (**BENIGN** in emerald green vs. **MALIGNANT** in crimson red).
- **Confidence & Probability Bars**: Visual progress bars showing model certainty and raw malignant probability.
- **Screening Summary**: Automatic clinical risk guidance and model diagnostics.

---

## 📈 Evaluation Metrics & Validation Results

### Baseline Model Performance (EfficientNet-B0)

| Metric | Validation Score | Note |
| :--- | :---: | :--- |
| **AUC-ROC** | **~0.850** | Strong discrimination across classification thresholds |
| **Sensitivity / Recall (Malignant)** | **~75.0%** (up to **>90%** with threshold tuning) | Prioritizes minimizing false negatives in clinical triage |
| **Validation Accuracy** | **~84.5%** | Robust overall performance despite class imbalance |
| **Macro F1-Score** | **~0.446** | Substantially higher than standard unweighted baselines |

### Decision Threshold Optimization (`outputs/threshold_sweep.json`)
In clinical screening workflows, standard `0.50` decision cutoffs can be tuned depending on deployment objectives:
- **Screening Mode (Threshold = 0.25 - 0.35)**: Maximizes melanoma recall (>85-92%) to minimize missed malignancies.
- **Balanced Mode (Threshold = 0.50)**: Standard operating threshold balancing sensitivity and precision.

---

## 🔬 Quality Assessment & Confidence Calibration Architecture

The project architecture includes a contract interface (`src/quality_stub.py`) designed for quality-aware clinical decision support:

```
                ┌────────────────────────────────┐
                │     Dermoscopy Lesion Image    │
                └───────┬────────────────┬───────┘
                        │                │
                        ▼                ▼
         ┌─────────────────────┐  ┌───────────────────────┐
         │ EfficientNet-B0 CNN │  │ Image Quality Module  │
         │  (Classification)   │  │   (OpenCV Analysis)   │
         └──────────┬──────────┘  └───────────┬───────────┘
                    │                         │
     P(Malignant)   │                         │ Quality Subscores
                    ▼                         ▼
         ┌────────────────────────────────────────────────┐
         │          Confidence Calibration Layer          │
         │      (Distinguishes Ambiguity vs. Bad Photo)   │
         └───────────────────────┬────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       ┌────────────────────┐          ┌────────────────────┐
       │ Reliable Prediction│          │   Retake Guidance  │
       │ (Benign/Malignant) │          │ (Blur, Dark, Crop) │
       └────────────────────┘          └────────────────────┘
```

- **Image Quality Analysis**: Evaluates Laplacian variance (blur), luminance histograms (under/overexposure), Weber contrast, and lesion bounding ratios.
- **Dual-Reason Guidance**: Differentiates between a model being uncertain due to genuine clinical ambiguity versus low-fidelity imaging requiring a retake.

---

## ⚠️ Disclaimer

> [!IMPORTANT]
> **Research and Educational Use Only**: This software and its associated models are developed for academic research and educational experimentation. It is **not** a certified medical diagnostic device and should **never** replace professional clinical examination, dermatological consultation, or histopathological biopsy.
