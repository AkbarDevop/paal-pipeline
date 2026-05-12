# PAAL Sow Posture Classification Pipeline

**Automated sow posture classification (standing / sitting / lying) using OAK-D ToF camera data with deep learning.**

Built for the Precision Animal Agriculture Lab (PAAL) at the University of Missouri. Classifies sow postures from multi-modal camera feeds (RGB, IR, Depth) to enable automated welfare monitoring — tracking how much time each sow spends in each posture throughout the day.

> **Continuing this project? Read [HANDOFF.md](HANDOFF.md) first.** It documents the one-command monthly inference flow, known data quirks, and where Task A ends and Task B begins.

## Quick start

```bash
# Posture inference on a month of OAK-D data (the main thing)
python posture/run_inference.py /path/to/data

# Or use the Colab notebook (zero setup):
# https://colab.research.google.com/github/AkbarDevop/paal-pipeline/blob/main/PAAL_Posture_Classification.ipynb
```

## Repository layout

```
paal-pipeline/
├── HANDOFF.md                ← read this first
├── config.py                 ← shared paths/constants
├── posture/                  ← Task A: posture classification (production, 98.4%)
├── vulva/                    ← Task B: vulva measurement (in progress)
├── archive/                  ← exploratory/dead scripts (see archive/README.md)
├── labels/  models/  outputs/  docs/
└── PAAL_Posture_Classification.ipynb   ← Colab notebook (self-contained)
```

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
OAK-D ToF Camera → Crop ROI → CNN Pig-Presence Detector → MobileNetV2 Posture Classifier
                                                                      ↓
                                          Heatmap + Predictions CSV + Excel Reports
```

The production inference pipeline is wrapped in a single command:

```bash
python posture/run_inference.py /path/to/data
```

It performs: crop → pig detection → posture classification → heatmap → CSV → per-pig Excel + master workbook.

To re-train the model from scratch:

```bash
python posture/train.py --modality ir --class-set posture3   # Trains posture classifier
python posture/train_pig_detector.py                          # Trains pig presence detector
python posture/eval.py                                        # Evaluates on held-out pigs 16–19
```

Earlier exploratory scripts (`scan_data.py`, `prefilter_depth.py`, `label_tool.py`, etc.) are preserved in [`archive/`](archive/README.md) for reference.

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

### Inference (production)

```bash
python posture/run_inference.py /path/to/data
```

This is the one-command monthly pipeline: crop → detect pig presence → classify posture → heatmap + CSV + Excel reports. Output lands in `outputs/`.

### Re-training the models

```bash
# Posture classifier (3 classes)
python posture/train.py --modality ir --backbone mobilenet_v2 --num-classes 3 \
  --labels-csv labels/labels_posture3.csv --model-prefix posture3 \
  --use-cropped --class-weights

# Evaluate on held-out pigs 16–19
python posture/eval.py --model-prefix posture3 --class-set posture3 --num-classes 3 \
  --labels-csv labels/labels_posture3.csv --use-cropped

# Pig-presence detector (binary)
python posture/train_pig_detector.py
```

## Setup for New Machine

The code is on GitHub, but data (~15 GB) and models (~30 MB) are not committed.

1. **Clone**: `git clone https://github.com/AkbarDevop/paal-pipeline.git`
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Get data**: Download the OAK-D timestamp folders from the shared Google Drive / OneDrive and place them anywhere on disk (you'll pass the path to `run_inference.py`)
4. **Get models**: Download from the GitHub release: <https://github.com/AkbarDevop/paal-pipeline/releases/tag/v1.0> and place in `models/`
5. **Run inference**: `python posture/run_inference.py /path/to/data`

Label CSVs in `labels/` contain absolute paths from the original machine. The pipeline resolves these automatically via `config.resolve_path()` — no manual path editing needed.

## Task B: Vulva Segmentation (In Progress)

Annotate vulva regions on standing pig images for swelling detection.

```bash
# Label vulva polygons on standing frames (uses RGB for labeling)
python vulva/label_vulva.py                    # All standing frames
python vulva/label_vulva.py --pig 5            # Only pig 5
python vulva/label_vulva.py --show-depth       # Show depth side-by-side
python vulva/label_vulva.py --use-cropped      # Use cropped images
```

Workflow: classify frame quality (good/tail_closed/too_close/bad) → if good, draw polygon → saves binary mask PNG + coordinates to CSV.

Output: `labels/vulva_labels.csv` + `labels/vulva_masks/*.png`

### Task B.1: Pig depth map / 3D surface

Build a pig-scale 3D reconstruction from one depth frame plus the matching IR image. This creates the 3D foundation for the next step, where a vulva-only mask can isolate the vulva surface instead of just the 2D outline.

```bash
# Build a pig point cloud + textured mesh from one frame
python vulva/point_cloud.py \
  --depth-raw images/20260211-09-17-32/pig0_depth_20260211-09-17-49.raw \
  --ir-img images/20260211-09-17-32/pig0_ir_vis_20260211-09-17-49.jpg

# Later: pass a binary mask to keep only the vulva region
python vulva/point_cloud.py \
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
paal-pipeline/
├── HANDOFF.md                 # Onboarding doc for the next maintainer
├── README.md                  # This file
├── CLAUDE.md                  # Project notes, decisions, history
├── config.py                  # Shared paths/constants (used by both tasks)
├── requirements.txt
├── PAAL_Posture_Classification.ipynb   # Colab notebook (self-contained)
│
├── posture/                   # Task A — posture classification (production)
│   ├── run_inference.py       # MAIN ENTRY: one-command monthly pipeline
│   ├── crop_images.py
│   ├── models.py              # MobileNetV2 / Xception / DenseNet121 wrappers
│   ├── data_loader.py
│   ├── train.py
│   ├── train_pig_detector.py  # CNN pig-presence detector (replaces depth prefilter)
│   ├── filter_predictions.py
│   ├── eval.py
│   ├── infer_batch.py
│   ├── count_posture_changes.py     # Transitions vs estrus ground truth
│   ├── generate_master_report.py    # Per-pig and master Excel workbook
│   ├── regenerate_heatmap.py
│   ├── audit_data.py
│   ├── fix_pig_ids.py               # Camera magnet/overflow corrections
│   └── fix_predictions_csv.py
│
├── vulva/                     # Task B — vulva segmentation/measurement (in progress)
│   ├── point_cloud.py         # Core 3D engine
│   ├── build_depthmaps.py
│   ├── build_vulva_rect_crops.py
│   ├── build_paper_vulva_surfaces.py
│   ├── crop_vulva_pointcloud.py
│   ├── label_vulva.py
│   ├── label_vulva_length_width.py
│   ├── measure_vulva_dataset.py
│   ├── measure_vulva_manual.py
│   ├── view_pointcloud.py
│   ├── extract_rear_view_pig_manual.py
│   └── render_depthmaps.py
│
├── archive/                   # Exploratory / dead scripts (see archive/README.md)
│
├── labels/                    # Label CSVs and binary masks
├── models/                    # Trained model weights (not tracked, download from release)
├── outputs/                   # Inference outputs (not tracked, generated locally)
└── docs/                      # Methodology and reference notes
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
