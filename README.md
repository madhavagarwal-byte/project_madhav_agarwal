# IDC Breast Cancer Classification
### project_madhav_agarwal

A lightweight CNN-based binary classifier that detects **Invasive Ductal Carcinoma (IDC)** from breast histopathology image patches. This is a supervised image classification task where the input is a 50×50 RGB image patch and the output is a binary label — IDC Negative or IDC Positive.

---

## Task Description

**Invasive Ductal Carcinoma (IDC)** is the most common form of breast cancer. Pathologists manually examine tissue slides to identify cancerous regions, which is time-consuming and prone to error. This project automates that process using a CNN trained on labelled histopathology image patches.

- **Input:** 50×50 px RGB image patch from a breast tissue slide
- **Output:** Binary label — `IDC Negative (0)` or `IDC Positive (1)`
- **Type:** Binary image classification

---

## Project Structure

```
project_madhav_agarwal/
├── checkpoints/
│   └── final_weights.pth       # Best trained model weights (saved during training)
├── data/
│   ├── IDC_Negative/           # 10 sample IDC Negative patches
│   │   ├── img01.jpg
│   │   └── ... img10.jpg
│   └── IDC_Positive/           # 10 sample IDC Positive patches
│       ├── img01.jpg
│       └── ... img10.jpg
├── config.py                   # All hyperparameters and path configs
├── dataset.py                  # IDCDataset class + the_dataloader factory
├── model.py                    # IDCClassifier CNN architecture
├── train.py                    # Training loop with early stopping + LR scheduling
├── predict.py                  # Single image and batch inference
├── interface.py                # Standardised aliases for grading
└── README.md
```

---

## Dataset

**Breast Histopathology Images** — [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)

- 277,524 patches (50×50 px) extracted from 162 whole-slide images
- Originally highly imbalanced (~198K negative, ~78K positive)
- **Balanced subset used:** 12,000 images (6,000 per class)
- **Split:** 70% train / 15% validation / 15% test (stratified)
- Patches are stored as `.png` files in nested folders: `patient_id/class_label/image.png`

### Data Augmentation (training only)
- Random horizontal and vertical flips
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation, hue)
- Normalize with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]

---

## Model Architecture

A custom 3-block CNN built from scratch — no pretrained weights used.

```
Input  (B, 3, 50, 50)
  │
  ├─ Block 1: Conv2d(3→32, 3×3)   BatchNorm2d  ReLU  MaxPool2d(2×2)  →  (B, 32, 25, 25)
  ├─ Block 2: Conv2d(32→64, 3×3)  BatchNorm2d  ReLU  MaxPool2d(2×2)  →  (B, 64, 12, 12)
  ├─ Block 3: Conv2d(64→128, 3×3) BatchNorm2d  ReLU  MaxPool2d(2×2)  →  (B, 128, 6, 6)
  │
  ├─ Flatten  →  4608
  ├─ Linear(4608→256)  BatchNorm1d  ReLU  Dropout(0.5)
  └─ Linear(256→1)  ← raw logit (sigmoid applied at inference)
```

- **Loss:** BCEWithLogitsLoss
- **Output:** Single raw logit → sigmoid → threshold at 0.5
- **Total parameters:** ~1.2M trainable

---

## Setup & Installation

```bash
git clone https://github.com/madhav-agarwal/project_madhav_agarwal.git
cd project_madhav_agarwal

pip install torch torchvision pillow scikit-learn matplotlib
```

---

## How to Train

```bash
python train.py
```

All hyperparameters are controlled from `config.py`:

| Parameter | Value |
|---|---|
| Image size | 50 × 50 px |
| Input channels | 3 (RGB) |
| Batch size | 64 |
| Epochs | 20 |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Optimizer | Adam |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early stopping | patience=5 |
| Random seed | 42 |

The best weights (lowest validation loss) are automatically saved to `checkpoints/final_weights.pth` during training.

---

## How to Run Inference

**Single image:**
```python
from predict import predict_image

label, confidence = predict_image("data/IDC_Negative/img01.jpg")
print(f"{label}  (confidence: {confidence:.4f})")
# IDC Negative  (confidence: 0.0213)
```

**Batch of images (list of paths):**
```python
from predict import predict_image

results = predict_image(["data/IDC_Positive/img01.jpg", "data/IDC_Positive/img02.jpg"])
for label, conf in results:
    print(f"{label}  ({conf:.4f})")
```

**Run on all images in `data/` from terminal:**
```bash
python predict.py
```

---

## How to Use interface.py

`interface.py` exposes standardised names for all key components:

```python
from interface import (
    TheModel,        # IDCClassifier  — the CNN class
    the_trainer,     # train_model()  — runs the training loop
    the_predictor,   # predict_image() — runs inference
    TheDataset,      # IDCDataset     — the Dataset class
    the_dataloader,  # the_dataloader() — returns train/val/test loaders
    the_batch_size,  # 64
    total_epochs,    # 20
)
```

---

## Results

| Metric | Score |
|---|---|
| Test Accuracy | — |
| Precision (IDC Positive) | — |
| Recall (IDC Positive) | ≥ 0.90 (target) |
| F1 Score | — |
| AUC-ROC | — |

> Run the evaluation cell in the Kaggle notebook to fill in these values.

---

## Requirements

```
torch
torchvision
pillow
scikit-learn
matplotlib
```

---

## Author

**Madhav Agarwal**

---

## License

This project is for academic purposes only.
