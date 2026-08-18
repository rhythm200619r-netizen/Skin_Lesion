# Skin Lesion Malignancy Risk Prediction

## Problem
Classify dermoscopic skin lesion images as benign or malignant, with a
quality-aware confidence layer that distinguishes "bad photo, please retake"
from "genuinely ambiguous lesion, refer to specialist" — rather than treating
every model prediction as equally trustworthy.

## Dataset
ISIC 2020 (full ~33,000 images, `image_name` + `target` in `train.csv`).
Heavily imbalanced: malignant is a small minority of the full set.

For fast iteration, `src/make_subset.py` builds a smaller stratified subset
(keeps all malignant images, samples a fixed multiple of benign images).

## Current Progress (Person 1 — Dataset + Model)
- [x] Dataset loader (`src/dataset.py`): stratified train/val split, resize
      to 224x224, ImageNet normalization, flip/rotate/brightness augmentation
      on the training split only
- [x] Stratified subset builder (`src/make_subset.py`) for fast iteration on
      the full 33k-image set
- [x] EfficientNet-B0 fine-tuning (`src/train.py`): ImageNet-pretrained via
      `timm`, weighted `CrossEntropyLoss` for class imbalance, saves best
      checkpoint by validation malignant-recall (not accuracy)
- [x] Evaluation: accuracy, precision, recall, F1, AUC-ROC, confusion matrix,
      threshold sweep (0.10–0.90) for tuning the melanoma-recall-priority
      cutoff — all written to `outputs/`
- [x] Prediction script (`src/predict.py`): single image in, class + confidence out
- [x] End-to-end pipeline sanity-tested on synthetic data (structure/mechanics
      verified; real training run pending on full dataset — GPU/local machine)

## Current Model
`EfficientNet-B0`, ImageNet pretrained, binary classification head, fine-tuned
with a weighted loss to counter class imbalance.

## Not Yet Done (Person 2 / joint work)
- Grad-CAM attribution
- Image quality assessment module (blur/brightness/contrast/framing via OpenCV)
- Dual-reason retake guidance logic (bad photo vs. ambiguous lesion)
- Confidence calibration combining quality score + classifier confidence
- Streamlit/Gradio demo integration

## How to Run

```bash
pip install -r requirements.txt

# 1. Unzip your ISIC 2020 download so you have:
#    data/train.csv
#    data/jpeg/train/*.jpg

# 2. Build a manageable subset (keeps all malignant + a benign sample)
python src/make_subset.py \
    --csv data/train.csv \
    --img-dir data/jpeg/train \
    --out-csv data/subset_train.csv \
    --target-size 6000 \
    --benign-malignant-ratio 8

# 3. Train
python src/train.py \
    --csv data/subset_train.csv \
    --img-dir data/jpeg/train \
    --epochs 10 \
    --batch-size 32

# 4. Predict on a single image
python src/predict.py --image path/to/lesion.jpg --model models/best_model.pth
```

## Project Structure
```
skin-lesion-risk/
├── data/                    # train.csv + jpeg/train/ go here (not tracked in git)
├── models/
│   └── best_model.pth       # saved after training
├── src/
│   ├── dataset.py
│   ├── make_subset.py
│   ├── train.py
│   └── predict.py
├── outputs/
│   ├── confusion_matrix.png
│   ├── threshold_sweep.json
│   └── training_history.json
├── requirements.txt
└── README.md
```

## Notes for Review 1
Full inference pipeline (dataset → EfficientNet-B0 → training → checkpoint →
prediction) is implemented and verified end-to-end. Next stage: run real
training on the full/subset ISIC data, then integrate Person 2's Grad-CAM +
quality assessment module, followed by the confidence calibration / retake
logic that is the project's core novelty.
