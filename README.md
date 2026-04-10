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

### Task B.1: Pig depth map / 3D surface

Build a pig-scale 3D reconstruction from one depth frame plus the matching IR image. This creates the 3D foundation for the next step, where a vulva-only mask can isolate the vulva surface instead of just the 2D outline.

```bash
# Build a pig point cloud + textured mesh from one frame
python3 point_cloud.py \
  --depth-raw images/20260211-09-17-32/pig0_depth_20260211-09-17-49.raw \
  --ir-img images/20260211-09-17-32/pig0_ir_vis_20260211-09-17-49.jpg

# Later: pass a binary mask to keep only the vulva region
python3 point_cloud.py \
  --depth-raw /path/to/pig_depth.raw \
  --ir-img /path/to/pig_ir_vis.jpg \
  --mask /path/to/vulva_mask.png
```

Outputs are written to `outputs/<frame>_depthmap/`:
- `pig_point_cloud.ply` - IR-textured point cloud
- `pig_surface.obj` + `.mtl` + `mesh_texture.png` - textured surface mesh
- `depthmap_bundle.npz` - arrays for downstream vulva-only segmentation / measurement
- `mask.png`, `segmented_depth.png`, `summary.json`

### Task B.2: Vulva mask in depth coordinates

Project vulva masks from RGB annotation space into the ToF/depth frame, then build vulva-only 3D outputs from the projected masks.

```bash
# After labels/vulva_labels.csv exists
python3 build_vulva_depthmaps.py

# Optional: only create depth-coordinate masks
python3 build_vulva_depthmaps.py --no-3d

# View one 3D point cloud / mesh interactively
python3 view_pointcloud.py depthmap/20260211-09-17-32/pig0_20260211-09-17-49
```

Outputs:
- `labels/vulva_masks_depth/*.png` - vulva masks in ToF/depth coordinates
- `labels/vulva_depth_labels.csv` - manifest of projected masks and 3D outputs
- `depthmap_vulva/<timestamp>/<frame>/` - vulva-only point cloud, mesh, previews, and summaries

### Task B.3: Manual vulva length and width on every live image

Label every current dataset frame directly on the IR image, with a simple point-based workflow.

```bash
# Annotate the live images/ dataset
python3 label_vulva_dataset.py

# Optional: only work on one pig or a smaller slice
python3 label_vulva_dataset.py --pig 0
python3 label_vulva_dataset.py --start-at 25 --limit 40
```

Annotation statuses:
- `present` - click 2 points for vulva length, then 2 points for vulva width
- `not_clear` - vulva is not clear enough to measure
- `skip` - leave the frame for later

Outputs:
- `labels/vulva_length_width_labels.csv` - per-frame timeslot, pig ID, status, `L`, `W`, and estimated volume

The tool uses the IR/depth-aligned frame plus ToF intrinsics to convert the clicked point pairs into metric `L/W` values. Volume is estimated with an ellipsoid assumption using diameters `L x W x W`.

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
├── point_cloud.py             # Depth raw + IR -> pig point cloud + textured mesh
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
