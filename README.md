# PAAL Sow Posture Classification Pipeline

**Automated sow posture classification (standing / sitting / lying) using OAK-D ToF camera data with deep learning.**

Built for the Precision Animal Agriculture Lab (PAAL) at the University of Missouri. Classifies sow postures from multi-modal camera feeds (RGB, IR, Depth) to enable automated welfare monitoring — tracking how much time each sow spends in each posture throughout the day.

## Key Results

| Metric | Value |
|---|---|
| **Test Accuracy** | 98.4% (190 held-out frames from unseen pigs) |
| **Best Model** | MobileNetV2 + IR modality (2.2M params) |
| **Classes** | Standing, Sitting, Lying |
| **Backbones Tested** | MobileNetV2, Xception, DenseNet121 |
| **Modalities** | RGB, IR (infrared), Depth (ToF) |

## Pipeline Overview

```
OAK-D ToF Camera → Depth Prefilter (pig present?) → Crop ROI → MobileNetV2 → Posture Prediction
```

The full pipeline:

```text
scan_data.py          Scan raw OAK exports → labels/metadata.csv
        ↓
prefilter_depth.py    Depth-based pig presence filter → labels/presence_filter.csv
        ↓
label_tool.py         Manual posture labeling GUI → labels/labels_posture3.csv
        ↓
crop_images.py        Crop ROI to remove neighbor pigs & ceiling
        ↓
train.py              Train classifier (3 backbones × 3 modalities) → models/*.pth
        ↓
eval.py               Evaluate on held-out test pigs → confusion matrices, per-class metrics
        ↓
pretrain_sowbot.py    (Optional) Pre-train on external sowbot 2022 IR dataset
```

## Features

- **Multi-modal**: Train on RGB, IR, depth, or any combination
- **3 backbone architectures**: MobileNetV2 (2.2M), Xception (20.8M), DenseNet121 (7.0M)
- **Transfer learning**: ImageNet pre-trained weights + optional sowbot IR pre-training
- **Pig-ID-based splitting**: Train/val/test split by pig identity (no data leakage)
- **Class-weighted loss**: Handles severe class imbalance (sitting class)
- **Depth prefiltering**: Automatically detects pig presence using depth statistics
- **Cropped ROI**: Removes neighbor pigs and ceiling from ToF field of view
- **Presentation-ready visuals**: Comparison charts, heatmaps, confusion matrices

## Quick Start

### 1. Build metadata from raw OAK exports

```bash
python3 scan_data.py
```

### 2. (Optional) Filter out empty frames

```bash
python3 prefilter_depth.py
```

### 3. Label data

```bash
python3 label_tool.py --class-set posture3
```

### 4. Crop images (remove neighbor pigs)

```bash
python3 crop_images.py
```

### 5. Train (e.g., MobileNetV2 on IR with class weights)

```bash
python3 train.py --modality ir --backbone mobilenet_v2 --num-classes 3 \
  --labels-csv labels/labels_posture3.csv --model-prefix posture3 \
  --use-cropped --class-weights
```

### 6. Evaluate on held-out test pigs

```bash
python3 eval.py --model-prefix posture3 --class-set posture3 --num-classes 3 \
  --labels-csv labels/labels_posture3.csv --use-cropped
```

### 7. (Optional) Pre-train on sowbot 2022 dataset, then fine-tune

```bash
python3 pretrain_sowbot.py --backbone mobilenet_v2 --epochs 30
python3 train.py --modality ir --backbone mobilenet_v2 \
  --init-weights models/sowbot_mobilenet_v2_best.pth \
  --model-prefix posture3ft --labels-csv labels/labels_posture3.csv \
  --num-classes 3 --use-cropped --class-weights --lr 1e-5
```

## Setup for New Machine

The code is on GitHub, but data (15GB) and models (914MB) are too large for git.

1. **Clone the repo**: `git clone https://github.com/AkbarDevop/paal-pipeline.git`
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Get data**: Download `data/` from the shared Google Drive folder and place it inside the repo root
4. **Get models** (optional): Download `models/` from Google Drive for pre-trained checkpoints
5. **Run pipeline**: `python3 run_pipeline.py --all` or run individual steps below

Label CSVs in `labels/` contain absolute paths from the original machine. The pipeline resolves these automatically via `config.resolve_path()` — no manual path editing needed.

## Task B: Vulva Segmentation (In Progress)

Annotate vulva regions on standing pig images for swelling detection.

```bash
# Label vulva polygons on standing frames (uses RGB for labeling)
python3 label_vulva.py                    # All standing frames
python3 label_vulva.py --pig 5            # Only pig 5
python3 label_vulva.py --show-depth       # Show depth side-by-side
python3 label_vulva.py --use-cropped      # Use cropped images
```

Workflow: classify frame quality (good/tail_closed/too_close/bad) → if good, draw polygon → saves binary mask PNG + coordinates to CSV.

Output: `labels/vulva_labels.csv` + `labels/vulva_masks/*.png`

## Project Structure

```
paal_pipeline/
├── config.py                  # Central configuration (paths, splits, classes)
├── models.py                  # SingleModalModel (MobileNetV2/Xception/DenseNet121)
├── data_loader.py             # Dataset & dataloader with pig-ID splits
├── train.py                   # Training script with class weights & fine-tuning
├── eval.py                    # Evaluation with per-class metrics & plots
├── pretrain_sowbot.py         # Pre-train on sowbot 2022 IR dataset
├── scan_data.py               # Index raw OAK timestamp folders
├── prefilter_depth.py         # Depth-based pig presence detection
├── crop_images.py             # Crop ROI from raw frames
├── label_tool.py              # Manual posture labeling GUI (Task A)
├── label_vulva.py             # Vulva polygon annotation tool (Task B)
├── find_sitting.py            # Scan for sitting candidates in unlabeled data
├── infer_random.py            # Quick inference on random samples
├── align_images.py            # Align RGB/IR/depth modalities
├── make_slides.py             # Generate weekly meeting slides
├── validate_pig_detection.py  # Validate depth prefilter accuracy
├── plot_pretrain_comparison.py  # Sowbot vs from-scratch comparison visuals
├── plot_pipeline_summary.py     # Presentation-ready summary figures
├── labels/                    # Label CSVs and backups
│   ├── vulva_labels.csv       # Vulva annotations (Task B)
│   └── vulva_masks/           # Binary mask PNGs (Task B)
├── models/                    # Saved model checkpoints (not tracked)
├── outputs/                   # Evaluation results and figures (not tracked)
└── data/                      # Raw OAK exports (not tracked)
```

## Modalities

| Modality | Channels | Description |
|---|---|---|
| `rgb` | 3 | Color camera (1280×800) |
| `ir` | 3 | Infrared camera (640×480), works in darkness |
| `depth` | 3 | Time-of-Flight depth map (640×480), colorized |

## Data Split

Pig-ID-based split to prevent data leakage:

| Split | Pig IDs | Purpose |
|---|---|---|
| Train | 0–11 | Model training |
| Validation | 12–15 | Hyperparameter tuning, early stopping |
| Test | 16–19 | Final evaluation (completely held out) |

## Requirements

- Python 3.10+
- PyTorch, torchvision
- OpenCV, NumPy, scikit-learn, matplotlib
- timm (for Xception backbone)

## License

Research use — University of Missouri, PAAL Lab.
