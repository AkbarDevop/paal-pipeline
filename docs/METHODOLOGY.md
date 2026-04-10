# Technical Methodology — PAAL Sow Posture Classification & Vulva Measurement

This document explains every step, decision, metric, and library in the pipeline.
Written for scientific reproducibility and paper-readiness.

---

## Reference Papers

This project replicates and extends the methodology from:

1. **Xu et al. (2023)** — "Detecting sow vulva size change around estrus using machine vision technology." *Smart Agricultural Technology*, Vol. 3, 100090. DOI: 10.1016/j.atech.2022.100090
   - Foundational vulva measurement paper: LiDAR camera (Intel RealSense L515), 8 sows, 19 days, CV = W×L×H (R²=0.92)

2. **Xu et al. (2024)** — "Automated oestrous detection in sows using a robotic imaging system." *Biosystems Engineering*, Vol. 244, 134-145. DOI: 10.1016/j.biosystemseng.2024.05.018
   - Full pipeline: posture classification + vulva volume → 1D CNN estrus detection (98.0% accuracy with DFW + behavior + vulva volume)

3. **Xu et al. (2024)** — "Developing a Sow Vulva Volume Estimation Pipeline Based on LiDAR Imagery and Deep Learning." *J. ASABE*, Vol. 67(3), 649-661.
   - Vulva segmentation pipeline: MobileNet filter → U-Net segment → MobileNet QA → volume calculation

---

## Table of Contents

### Part A — Posture Classification (COMPLETE, 98.4%)
1. [Hardware Setup](#1-hardware-setup)
2. [Data Collection & Structure](#2-data-collection--structure)
3. [Step 1: scan_data.py — Data Indexing](#3-step-1-scan_datapy--data-indexing)
4. [Step 2: prefilter_depth.py — Pig Presence Detection](#4-step-2-prefilter_depthpy--pig-presence-detection)
5. [Step 3: label_tool.py — Manual Annotation](#5-step-3-label_toolpy--manual-annotation)
6. [Step 4: crop_images.py — Region of Interest Extraction](#6-step-4-crop_imagespy--region-of-interest-extraction)
7. [Step 5: train.py — Model Training](#7-step-5-trainpy--model-training)
8. [Step 6: eval.py — Evaluation & Metrics](#8-step-6-evalpy--evaluation--metrics)

### Part B — Vulva 3D Measurement (IN PROGRESS)
9. [Task B Overview & Reference Paper Mapping](#9-task-b-overview--reference-paper-mapping)
10. [Step B.1: point_cloud.py — Depth-to-3D Reconstruction](#10-step-b1-point_cloudpy--depth-to-3d-reconstruction)
11. [Step B.2: build_depthmaps.py — Batch 3D Processing](#11-step-b2-build_depthmapspy--batch-3d-processing)
12. [Step B.3: crop_vulva_pointcloud.py — Vulva Rectangle Selection](#12-step-b3-crop_vulva_pointcloudpy--vulva-rectangle-selection)
13. [Step B.4: build_paper_vulva_surfaces.py — Paper-Style Surface Analysis](#13-step-b4-build_paper_vulva_surfacespy--paper-style-surface-analysis)
14. [Step B.5: label_vulva_length_width.py — Manual L/W Annotation](#14-step-b5-label_vulva_length_widthpy--manual-lw-annotation)
15. [Step B.6: compare_vulva_ground_truths.py — Validation](#15-step-b6-compare_vulva_ground_truthspy--validation)

### Shared
16. [Libraries & Technologies](#16-libraries--technologies)
17. [Metric Definitions](#17-metric-definitions)
18. [Key Design Decisions & Justifications](#18-key-design-decisions--justifications)

### Appendix
19. [File Creation & Update Log](#19-file-creation--update-log)

---

## 1. Hardware Setup

**Camera**: Luxonis OAK-D ToF (Time-of-Flight)
- Mounted on the side of each farrowing stall, facing the vulva side of the sow
- Captures 3 synchronized image streams:
  - **RGB**: 1280×800 pixels, 3-channel color (standard camera)
  - **IR (Infrared)**: 640×480 pixels, active infrared illumination — works in complete darkness
  - **Depth**: 640×480 pixels, Time-of-Flight sensor — measures distance (mm) to each pixel

**Why 3 modalities?**
- RGB gives the most visual detail but fails in darkness (barns are dark at night)
- IR works 24/7 regardless of lighting
- Depth captures 3D shape regardless of color/lighting — a pig is always a "blob" at a certain distance

**Data format on disk**:
- Each recording session is a timestamp folder (e.g., `20260123-14-46-37/`)
- Each pig gets files like: `pig0_rgb_20260123-14-47-04.jpg`, `pig0_ir_vis_20260123-14-47-04.jpg`, `pig0_depth_vis_20260123-14-47-04.jpg`
- Raw sensor data is also saved as `.raw` files (uint16 binary, 640×480, some with 8-byte headers)

---

## 2. Data Collection & Structure

**Dataset**: 20 sows, recorded across multiple sessions
- Total raw frames: **4,852** (before filtering)
- After labeling: **1,372 frames** annotated with posture

**Posture classes**:
| Label | Posture | Count | Percentage | Description |
|-------|---------|-------|------------|-------------|
| 0 | Standing | 656 | 47.8% | Sow upright on all 4 legs |
| 1 | Sitting | 52 | 3.8% | Sow on haunches, front legs extended |
| 2 | Lying | 664 | 48.4% | Sow on side or belly |

**How posture labels were obtained**: Labels are **manual human annotations**, not automated classifications. A single annotator viewed each RGB image and assigned a posture class based on visual judgment. The depth sensor is only used for pig *presence* detection (Section 4), not for posture classification. This is the standard supervised learning approach — human-labeled ground truth is used to train and evaluate the model. See [FAQ Q1](#q1-how-were-posture-labels-created-and-how-should-they-be-documented-for-a-paper) for paper-readiness requirements.

**Class imbalance**: Sitting is extremely rare (3.8%). This is a real-world reflection — sows rarely sit. This creates a challenge: a model that never predicts "sitting" would still get 96.2% accuracy. We address this with class-weighted loss (see Section 7).

---

## 3. Step 1: scan_data.py — Data Indexing

**What it does**: Walks all timestamp folders in `data/`, matches files by pig ID and timestamp, and writes `labels/metadata.csv`.

**Output columns**: `timestamp_folder`, `pig_id`, `pig_timestamp`, `rgb_jpg`, `ir_jpg`, `depth_jpg`, `rgb_cropped_jpg`, `ir_cropped_jpg`, `depth_cropped_jpg`, `rgb_raw`, `ir_raw`, `depth_raw`

**Why**: The rest of the pipeline reads from this CSV instead of scanning the filesystem each time. This makes the pipeline reproducible — same CSV = same data order.

---

## 4. Step 2: prefilter_depth.py — Pig Presence Detection

**Problem**: Not every frame contains a pig. Some stalls are empty, or the pig has moved out of view. Labeling empty frames wastes time.

**Method**: Depth-based median thresholding.

**How it works** (line by line):
1. Load the raw depth image (uint16, values = distance in millimeters)
2. Crop to the stall region: `CROP_TOF = (120, 30, 500, 480)` — this is the area where the pig would be
3. Mask out stall bars: set left 50px and right 50px to False (metal bars at edges give misleading depth values)
4. Filter valid depth range: keep only pixels where 200mm < depth < 5000mm (removes noise and zero-depth pixels — see [FAQ Q3](#q3-how-were-the-crop-box-coordinates-120-30-500-480-and-depth-filter-range-2005000mm-determined) for why these values)
5. Compute the **median** of remaining depth values
6. **Decision rule**: `median_depth < 1463mm` → pig present (see [FAQ Q3](#q3-how-were-the-crop-box-coordinates-120-30-500-480-and-depth-filter-range-2005000mm-determined) for how this threshold was validated)

**Why median, not mean?**
- Median is robust to outliers (a few noisy pixels don't affect it)
- Mean would be skewed by a single very-far or very-near pixel

**Why 1463mm threshold?**
- The camera is side-mounted at a known distance from the opposite wall
- When a pig is present, its body blocks the view → median depth is low (600-1200mm)
- When empty, the camera sees the far wall → median depth is high (1500mm+)
- 1463mm was determined empirically by inspecting the depth histogram of frames with and without pigs (see `validate_pig_detection.py` which checks against manual ground truth)

**Output**: `labels/presence_filter.csv` with columns: `pig_present` (0 or 1), `median_depth`, `threshold`

> **Note**: The prefilter crops the depth data **in memory** using `CROP_TOF` but does not save cropped files to disk. That's done later by `crop_images.py` (Step 4). See [FAQ Q2](#q2-why-does-prefilter_depthpy-crop-in-memory-but-crop_imagespy-saves-to-disk-isnt-that-redundant) for details.

---

## 5. Step 3: label_tool.py — Manual Annotation

**What it does**: Displays images one at a time. Human presses a key to label the posture.

**Interface**:
- Press `0` = standing, `1` = sitting, `2` = lying
- Press `s` = skip (unsure), `q` = quit
- Each label is immediately flushed to CSV (no data loss on crash)

**Output**: Appends rows to `labels/labels_posture3.csv` with the image paths and label.

**Why manual labeling?**
- This is the ground truth that all model performance is measured against
- 1,372 frames were labeled by a single annotator for consistency
- Ambiguous frames (pig mid-transition, partially occluded) were skipped

---

## 6. Step 4: crop_images.py — Region of Interest Extraction

**Problem**: The raw camera image shows more than just the target pig:
- **Neighboring pigs** are visible through stall bars on the left/right
- **Ceiling hardware** (pipes, feeders) appears at the top
- These distractions confuse the model

**Solution**: Crop to a fixed ROI (Region of Interest) that contains only the target pig.

### How the crop box was determined

The crop box is defined in `config.py`:

```python
CROP_TOF = (120, 30, 500, 480)  # (x_left, y_top, x_right, y_bottom) in ToF coords
```

This means: from the 640×480 depth/IR image, keep only x=120..500, y=30..480. (See [FAQ Q3](#q3-how-were-the-crop-box-coordinates-120-30-500-480-and-depth-filter-range-2005000mm-determined) for full justification.)

**How these numbers were chosen**:
- **x=120 (left boundary)**: The left stall bar is at approximately x=100-120 in the ToF image. Cropping at x=120 removes the neighboring pig visible through the left gap.
- **x=500 (right boundary)**: The right stall bar is at approximately x=490-510. Cropping at x=500 removes the right neighbor.
- **y=30 (top boundary)**: Removes the ceiling pipes and overhead hardware.
- **y=480 (bottom)**: Keeps the full bottom (floor level) — no need to crop.

**These values were determined by visual inspection**: opening multiple frames across different pigs and sessions, identifying where the stall bars consistently appear, and setting the crop just inside the bars. The stall dimensions are physically fixed (same stall design for all pigs), so one crop box works for all frames.

**For RGB images** (1280×800, higher resolution than ToF's 640×480):
```python
sx, sy = w / 640, h / 480  # scale factors: 2.0 and 1.667
# Scaled crop: (240, 50, 1000, 800)
```

**Output**: Cropped images saved alongside originals with `_cropped` suffix.

**Result**: The cropped region is 380×450 pixels (ToF: 500−120 × 480−30) or 760×750 pixels (RGB: 1000−240 × 800−50 after scaling). This is then resized to 224×224 for model input (the standard ImageNet input size that all three backbones were pre-trained on).

> **Note**: The same `CROP_TOF` box is also used by `prefilter_depth.py` (Step 2), but only in memory — it doesn't save files. See [FAQ Q2](#q2-why-does-prefilter_depthpy-crop-in-memory-but-crop_imagespy-saves-to-disk-isnt-that-redundant) for why.

---

## 7. Step 5: train.py — Model Training

### 7.1 Data Split — Pig-ID Based

```python
TRAIN_PIG_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # 12 pigs
VAL_PIG_IDS   = [12, 13, 14, 15]                            # 4 pigs
TEST_PIG_IDS  = [16, 17, 18, 19]                             # 4 pigs
```

| Split | Pig IDs | Frames | Purpose |
|-------|---------|--------|---------|
| Train | 0–11 | 918 | Model learns from these |
| Validation | 12–15 | 264 | Tune hyperparameters, early stopping |
| Test | 16–19 | 190 | Final evaluation (NEVER seen during training) |

**Why split by pig ID, not randomly?**

If we split randomly, frames from the same pig could appear in both train and test sets. The model would learn to recognize individual pigs (their unique markings, body shape) rather than general posture patterns. This is called **data leakage** — the test accuracy would be artificially inflated because the model has already "seen" those pigs.

By splitting by pig ID, the test pigs (16–19) are completely new to the model. If it still achieves high accuracy, it means the model learned posture features, not pig identity.

### 7.2 Image Preprocessing (data_loader.py)

For each image:
1. **Load**: `cv2.imread(path)` — reads JPG as BGR numpy array
2. **Resize**: `cv2.resize(img, (224, 224))` — standard input size for ImageNet-pretrained models
3. **Color convert**: `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` — OpenCV loads BGR, but models expect RGB
4. **Normalize**: `img / 255.0` — scale pixel values from [0, 255] to [0.0, 1.0]
5. **To tensor**: `torch.from_numpy(img).permute(2, 0, 1)` — convert to PyTorch format (channels, height, width)

**Why 224×224?**
All three backbones (MobileNetV2, Xception, DenseNet121) were originally designed for 224×224 ImageNet inputs. Their pretrained weights expect this size. Using a different size would require architecture changes.

### 7.3 Model Architecture (models.py)

All three backbones follow the same pattern:

```
Input image (224×224×3)
    → Feature extractor (pretrained backbone — convolutional layers)
    → Global Average Pooling (collapse spatial dimensions → single vector)
    → Dropout (0.2 probability — randomly zero 20% of features to prevent overfitting)
    → Linear classifier (feature vector → 3 class scores)
```

**Backbones**:

| Backbone | Parameters | Source | ImageNet Top-1 | Key Design |
|----------|-----------|--------|----------------|------------|
| MobileNetV2 | 3.4M | torchvision | 71.8% | Inverted residuals, depthwise separable convolutions. Designed to be small and fast. |
| Xception | 22.9M | timm library | 79.0% | "Extreme Inception" — replaces standard convolutions with depthwise separable convolutions throughout. |
| DenseNet121 | 8.0M | torchvision | 74.4% | Dense connections — each layer receives input from ALL previous layers. Promotes feature reuse. |

**What is transfer learning?**
Instead of initializing weights randomly, we start with weights pre-trained on ImageNet (1.2 million images, 1,000 classes — dogs, cats, cars, etc.). These weights already know how to detect edges, textures, shapes. We then **fine-tune** — continue training on our pig data so the model adapts these generic features to recognize pig postures.

**Why transfer learning instead of training from scratch?**
We only have 918 training images. Training a CNN from scratch with so few images would likely overfit (memorize training data without generalizing). Pre-trained features provide a strong starting point.

### 7.4 Loss Function — Class-Weighted Cross-Entropy

**Cross-entropy loss** measures how wrong the model's predictions are. For a single sample:

```
Loss = -log(predicted_probability_of_correct_class)
```

If the model predicts 95% confidence for the correct class → loss = -log(0.95) = 0.05 (low, good).
If the model predicts 10% confidence for the correct class → loss = -log(0.10) = 2.30 (high, bad).

**Class weighting** multiplies each sample's loss by a weight inversely proportional to class frequency:

```python
weight[class] = total_samples / (num_classes × count_of_class)
```

For our training set (918 samples: 451 standing, 40 sitting, 427 lying):
- Standing weight: 918 / (3 × 451) = **0.68**
- Sitting weight: 918 / (3 × 40) = **7.65** ← 11× higher than standing
- Lying weight: 918 / (3 × 427) = **0.72**

**Effect**: A mistake on a sitting sample costs 11× more than a mistake on standing. This forces the model to pay attention to the rare class instead of ignoring it.

**Without class weights**: The model could predict "standing" or "lying" for everything and still get ~96% accuracy (since sitting is only 3.8%). With class weights, ignoring sitting incurs a large penalty.

### 7.5 Optimizer — Adam

**Adam** (Adaptive Moment Estimation) is the optimization algorithm that updates model weights after each batch.

- Learning rate: `1e-4` (0.0001) — how big each update step is
- Adam automatically adapts the learning rate per-parameter based on gradient history
- More stable than basic SGD for small datasets

**Why Adam?**
Standard choice for fine-tuning. It converges faster than SGD and requires less hyperparameter tuning. For our small dataset and 30 epochs, fast convergence matters.

### 7.6 Learning Rate Scheduler — ReduceLROnPlateau

```python
scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
```

**What it does**: Monitors validation loss after each epoch. If validation loss hasn't improved for 5 consecutive epochs (`patience=5`), it halves the learning rate (`factor=0.5`).

**Why?** Early in training, a larger learning rate makes big improvements. Later, the model needs smaller adjustments — a large learning rate would overshoot the optimal weights. This scheduler automatically transitions from "big steps" to "small steps."

### 7.7 Training Loop

For each of 30 epochs:
1. **Train phase**: Show all 918 training images (in batches of 16). For each batch:
   - Forward pass: model predicts posture
   - Compute loss (class-weighted cross-entropy)
   - Backward pass: compute gradients (how to adjust each weight)
   - Update weights using Adam
2. **Validation phase**: Run all 264 validation images through the model (no weight updates). Record accuracy and loss.
3. **Save best model**: If validation accuracy improved, save the model weights to disk.
4. **Adjust learning rate**: If validation loss stalled for 5 epochs, halve the learning rate.

**Best model selection**: We save the model from the epoch with the highest validation accuracy — NOT the final epoch. This prevents using an overfitted model.

---

## 8. Step 6: eval.py — Evaluation & Metrics

### How accuracy is calculated

After training, we load the saved best model and run it on the **test set** (pigs 16–19, 190 frames).

```python
accuracy = correct_predictions / total_predictions
# 187 correct out of 190 = 98.4%
```

The model has NEVER seen these 4 pigs during training. This is the number reported in the paper. See [FAQ Q4](#q4-how-does-the-model-predict-postures-on-images-it-has-never-seen-and-how-do-we-know-the-984-accuracy-is-real) for why this number is trustworthy and how the evaluation works.

### Confusion matrix

A 3×3 grid showing what the model predicted vs. the true label:

```
                    Predicted
                Standing  Sitting  Lying
True Standing      83        1       0
True Sitting        0        3       1
True Lying          1        0     101
```

**How to read it**:
- Diagonal = correct (83 + 3 + 101 = 187 correct)
- Off-diagonal = errors (1 + 1 + 1 = 3 errors)
- Row = what the true label was
- Column = what the model predicted

### Per-class metrics (from sklearn.classification_report)

For each class, we compute:

| Metric | Formula | What it means |
|--------|---------|---------------|
| **Precision** | TP / (TP + FP) | Of all frames the model called "sitting," what fraction truly were sitting? |
| **Recall** | TP / (TP + FN) | Of all truly sitting frames, what fraction did the model catch? |
| **F1-score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of precision and recall — balances both |

Where: TP = true positive, FP = false positive, FN = false negative.

**Example for sitting class** (MobileNetV2/IR):
- True sitting frames: 4 (test set has only 4 sitting frames)
- Model predicted sitting: 4 (3 correct + 1 false positive from standing)
- TP=3, FP=1, FN=1
- Precision = 3/(3+1) = 0.75 — "75% of frames called sitting were actually sitting"
- Recall = 3/(3+1) = 0.75 — "model caught 75% of the sitting frames"
- F1 = 2×(0.75×0.75)/(0.75+0.75) = 0.75

### Why sitting metrics matter most

Overall accuracy (98.4%) is dominated by standing and lying (which together are 96.2% of the data). A model could get 96.2% by never predicting sitting. The sitting F1 score tells us whether the model actually learned to recognize the rare posture.

---

---

## Part B — Vulva 3D Measurement

---

## 9. Task B Overview & Reference Paper Mapping

Task B aims to replicate the vulva volume estimation pipeline from Xu et al. (2023, 2024) using our OAK-D ToF camera instead of the Intel RealSense L515. The goal is to measure vulva swelling (length, width, height, volume) over time as an automated estrus indicator.

### Paper Pipeline vs Our Implementation

| Paper Step | Paper Method | Our File | Our Method | Status |
|---|---|---|---|---|
| Image acquisition | LiDAR camera at 0.7-1.0m, manual | OAK-D ToF, side-mounted, automated | Different hardware, same concept | Done |
| Posture filter | MobileNet filters defective postures (94.8%) | prefilter_depth.py + run_inference.py | Depth prefilter + posture classifier | Done |
| Vulva segmentation | U-Net segments 3D vulva surfaces (96.7%) | build_paper_vulva_surfaces.py | Geometric: plane fitting + seed detection + inpainting | Done |
| QA filter | MobileNet excludes bad segmentations (100%) | Manual review in crop_vulva_pointcloud.py | Interactive rectangle verification | Done |
| 3D surface creation | Separate vulva from body | build_paper_vulva_surfaces.py | Original Surface → No-Vulva Surface → Vulva-Only Surface | Done |
| Feature extraction | 5 features: W, L, H, area, volume; CV=W×L×H | build_paper_vulva_surfaces.py | 9+ features including CV=W×L×H | Done |
| Ground truth comparison | Caliper measurements | compare_vulva_ground_truths.py | Excel ground truth vs measured, delta analysis | Done |
| Estrus detection | 1D CNN: DFW + behavior + vulva volume → estrus | Not yet implemented | — | Pending |

### Key Differences from Papers

1. **Camera**: OAK-D ToF (640×480 depth) vs Intel RealSense L515 (up to 1024×768). Our sensor has lower depth resolution but similar accuracy at stall distances.
2. **Vulva segmentation**: Papers use U-Net (deep learning); we use geometric methods (plane fitting + DoG seed detection + morphological dilation + diffuse inpainting). Both produce a Vulva-Only Surface.
3. **4 vs 3 posture classes**: Papers distinguish sternal vs lateral lying; we merge into one lying class.
4. **Additional measurements**: We compute surface area (Jacobian-corrected), integral volume, horizontal/vertical rect areas beyond the paper's 5 features.

---

## 10. Step B.1: point_cloud.py — Depth-to-3D Reconstruction

**Purpose**: Core 3D engine. Converts raw OAK-D depth frames into textured point clouds and meshes.

### Depth Backprojection

For each pixel (u, v) with depth value z (in mm), the 3D coordinate is:
```
X = (u - cx) * z / fx
Y = (v - cy) * z / fy
Z = z
```
Where (fx, fy, cx, cy) are camera intrinsics from calibration.json:
- fx=471.583, fy=471.430 (focal lengths in pixels)
- cx=323.905, cy=246.816 (principal point)

### Auto Pig Segmentation

1. Load depth raw (uint16, 640×480, optional 8-byte header)
2. Create ROI mask using CROP_TOF box with 50px bar margin
3. Compute wall distance as 90th percentile of valid depth in ROI
4. Set foreground threshold = wall_distance - 140mm (foreground margin)
5. Extract pig mask: depth < foreground threshold AND valid range (200-5000mm)
6. Morphological cleaning → largest connected component

### Vulva Detection via IR Local Darkness

The vulva appears as a locally dark region in IR images. Detection uses Difference-of-Gaussians (DoG):
1. Compute small blur: Gaussian σ=3 pixels
2. Compute large blur: Gaussian σ=17 pixels
3. Difference = small_blur - large_blur (highlights local dark features)
4. Threshold at 95th percentile → candidate mask
5. Select largest connected component within posterior region
6. Output: focus_box [y_min, x_min, y_max, x_max] around vulva

### Output Files (per pig)

| File | Content |
|---|---|
| pig_point_cloud.ply | ASCII PLY with XYZ + RGB per vertex |
| pig_surface.obj/mtl | Textured OBJ mesh with face culling |
| raw_3d_color_image.png | RGB-textured 3D visualization |
| depthmap_bundle.npz | NumPy archive: depth_mm, mask, texture arrays |
| summary.json | Metadata: vertex count, bounds, intrinsics, mask source |

---

## 11. Step B.2: build_depthmaps.py — Batch 3D Processing

**Purpose**: Batch driver that iterates all depth_*.raw files in images/ and calls point_cloud.py for each.

**Validation**: Checks raw file size is exactly 614,400 bytes (640×480×2) or 614,408 bytes (with 8-byte header).

**Output modes**:
- `paper_raw_acquisition` (default): Raw valid-depth masking, no heuristic segmentation — preserves all depth data in ROI
- `full_outputs`: Complete assets (point cloud, mesh, renders)
- `point_cloud_only`: Minimal (PLY + JSON)

**Output structure**: depthmap/{timestamp}/{pigN_timestamp}/ with index.csv and index.json at root.

**Current state**: 232 pig folders processed across 12 timestamp sessions.

---

## 12. Step B.3: crop_vulva_pointcloud.py — Vulva Rectangle Selection

**Purpose**: Interactive manual rectangle selection on rear-view acquisition images. User draws a rectangle around the vulva region, and the tool exports the 3D points within that box.

### Workflow

1. Load depth raw, calibration, IR texture, RGB image
2. Build context mask (valid depth within ROI)
3. Auto-suggest vulva box using IR local darkness detection
4. **Interactive GUI**: User draws/adjusts rectangle on color image
   - Left-drag = draw rectangle
   - Enter = accept, R = reset, Q = cancel
   - Minimum rectangle: 10×10 pixels
5. Crop depth and texture to selected rectangle
6. Build strided point cloud from cropped region
7. Export: cropped PLY, RGB/IR rect crops, overlay with rectangle, summary.json

### Context Mask Priority

The mask that determines which pixels are valid follows this priority chain:
1. External mask file (--mask argument)
2. Manual pig mask from previous summary.json
3. Paper-style raw valid depth (all depth in valid range)
4. Auto pig depth segmentation (heuristic)

**Output structure**: depthmap_rect/{timestamp}/{pigN_timestamp}/ with index.csv/json.

---

## 13. Step B.4: build_paper_vulva_surfaces.py — Paper-Style Surface Analysis

**Purpose**: Implements the paper's vulva measurement methodology. This is the core replication of Xu et al.'s approach, adapted from U-Net segmentation to geometric methods.

### Pipeline (replicating paper methodology)

#### Step 1: Load Cropped Vulva Rectangle
- Input: depthmap_rect frame (from Step B.3)
- Load depth, intrinsics, IR texture, RGB image
- Backproject cropped rectangle to 3D point cloud

#### Step 2: Fit Rotating Plane (SVD)
- `fit_rotating_plane()`: Iterative robust plane fitting (3 iterations)
- Each iteration: select support points between 10th-80th depth percentile → SVD → refine
- Output: plane_center, normal vector, orthonormal axes (u, v, normal)
- Orient normal so vulva center is on the "inward" side vs outer margin
- Align axes to match image x/y coordinates using correlation coefficients

#### Step 3: Rasterize to Original Surface (OS)
- Project all 3D points onto the fitted plane → (u, v, w) coordinates
- Rasterize to 300×300 grid (DEFAULT_GRID_SIZE) using bilinear interpolation
- **Original Surface (OS)**: The w-values (height above plane) form the depth map
- Track grid spacing: du_mm, dv_mm (physical size of each grid cell)

#### Step 4: Detect Vulva Seed Region
- `detect_vulva_seed_region()`: Implements paper's "largest round region" detection
- Compute signal = OS height values
- Compute prominence = DoG (2σ - 12σ Gaussian blur difference)
- Threshold and extract connected components
- Filter: min_area=0.15% of grid, max_area=20% of grid
- Score each candidate: `2.4*circularity + 1.4*aspect + 3.8*centrality + 1.4*area_preference + 0.9*signal + 0.5*prominence + 1.2*border_penalty`
- Select highest-scoring component as vulva seed

#### Step 5: Dilate Mask by 35%
- `grow_mask_by_percent(mask, growth_fraction=0.35)`
- Morphological dilation until pixel count reaches seed_pixels × 1.35
- This expanded region defines the measurement area

#### Step 6: Inpaint No-Vulva Surface
- `diffuse_fill()`: Iterative Gaussian blur inpainting (180-220 iterations, σ=2.0)
- Fills the vulva region with estimated "what the surface would look like without the vulva"
- Known pixels (outside vulva mask) are preserved each iteration
- Result: No-Vulva Surface — a smooth baseline estimate

#### Step 7: Compute Vulva-Only Surface
```
Vulva-Only Surface = Original Surface - No-Vulva Surface
```
- Positive values = vulva protrusion above body surface
- This isolates the vulva's 3D shape from the surrounding body curvature

#### Step 8: Ellipse-Fitted Measurements

**Measurement extraction** using cv2.fitEllipse on the measurement mask contour:

| Measurement | Formula | Paper Equivalent |
|---|---|---|
| **length_mm** | Ellipse major axis length | Vulva Length (L) |
| **width_mm** | Ellipse minor axis length | Vulva Width (W) |
| **peak_height_mm** | 99th percentile of Vulva-Only Surface | Vulva Height (H) |
| **mean_height_mm** | Mean of positive Vulva-Only values | — |
| **base_area_mm2** | count(VOS > 0) × du × dv | Base Area |
| **surface_area_mm2** | Σ √(1 + (∂h/∂u)² + (∂h/∂v)²) × du × dv | Surface Area (Jacobian-corrected) |
| **volume_mm3** | Σ VOS_height × du × dv | Vulva Volume (integral) |
| **cubic_volume_mm3** | W × L × H | **CV** (paper's key metric, R²=0.92) |
| **horizontal_rect_area_mm2** | W × L | HRA |
| **vertical_rect_area_mm2** | W × H | VRA |

### Output Files (per pig, in paper_vulva/ subfolder)

| File | Content |
|---|---|
| original_surface_top_view.png | RGB top view of vulva rectangle |
| original_surface_depth_os.png | 300×300 OS depth colormap |
| vulva_seed_region.png | Detected seed mask |
| vulva_mask_scaled_35pct.png | Dilated measurement mask |
| no_vulva_surface.png | Inpainted baseline surface |
| vulva_only_surface.png | Difference surface (VOS) colormap |
| vulva_measurements.png | VOS with ellipse axes, L/W/H overlaid |
| paper_vulva_process_panel.png | 9-panel overview of entire pipeline |
| paper_vulva_summary.json | All measurements + metadata |
| paper_vulva_surface_bundle.npz | NumPy arrays for all surfaces |
| *_rgb_overlay.png, *_os_overlay.png | Mask overlays on RGB and OS |

---

## 14. Step B.5: label_vulva_length_width.py — Manual L/W Annotation

**Purpose**: Alternative measurement approach. User clicks 2 points for vulva length and 2 for width directly on the IR image. System backprojects to 3D metric coordinates and computes Euclidean distance.

### Workflow

1. Scan images/ for RGB/IR/depth frame triplets
2. Load existing measurements from CSV, skip already-labeled
3. For each frame:
   - Display review panel (IR large + RGB small + Depth small)
   - User classifies: p=present, u=not_clear, s=skip
   - If "present": open IR annotation tool
   - Click 2 points for length (red lines), 2 for width (blue lines)
   - Backproject each point to 3D using `nearest_valid_depth()` within 5px radius
   - Compute 3D Euclidean distance: `√((x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)²)`
   - Area estimate: 0.5 × L × W (half-ellipse approximation)
4. Enrich with ground truth from ground_truths/vulva_ground_truths.csv
5. Save to labels/vulva_length_width_labels.csv

### Output CSV Fields
`timeslot_name, pig_id, status, L_mm, GT_L_mm, diff_L_mm, W_mm, GT_W_mm, diff_W_mm, area_mm2, area_method`

---

## 15. Step B.6: compare_vulva_ground_truths.py — Validation

**Purpose**: Compare measured vulva dimensions against caliper ground truth from Excel workbooks.

### Ground Truth Source
- File: ground_truths/Feb_2026.xlsx (openpyxl)
- Each sheet = one pig, rows starting at row 15
- Columns: day_index, date, width_mm (col 7), length_mm (col 8), volume (col 9)
- Ground truth formula: vulva_volume = L × W × 0.005

### Comparison Method
1. Extract ground truth rows from Excel → build index by (date, pig_id)
2. Load measurement rows from label CSV
3. Join by (measurement_date, pig_id) — date derived from timeslot name
4. Compute deltas: diff_L = measured_L - GT_L, diff_W = measured_W - GT_W
5. Report: matched rows, mean absolute errors per measurement type

---

## 16. Libraries & Technologies

| Library | Version | What it does in our pipeline |
|---------|---------|------------------------------|
| **PyTorch** | 2.x | Deep learning framework — defines models, computes gradients, runs training loop |
| **torchvision** | 0.x | Provides pre-trained MobileNetV2 and DenseNet121 backbones |
| **timm** | 1.x | Provides pre-trained Xception backbone (not in torchvision) |
| **OpenCV (cv2)** | 4.x | Image loading (`imread`), resizing, color conversion, display for labeling GUI |
| **NumPy** | 1.x | Array operations — depth data processing, image manipulation |
| **scikit-learn** | 1.x | Evaluation metrics — `accuracy_score`, `classification_report`, `confusion_matrix` |
| **matplotlib** | 3.x | Plotting — confusion matrices, training curves, comparison charts |
| **python-pptx** | 0.6.x | PowerPoint generation for weekly meeting slides |

### Key PyTorch concepts used

| Concept | Where | What it does |
|---------|-------|--------------|
| `nn.Module` | `models.py` | Base class for all neural network models |
| `nn.CrossEntropyLoss` | `train.py:93` | Loss function that combines softmax + negative log-likelihood |
| `optim.Adam` | `train.py:96` | Optimizer that updates weights |
| `ReduceLROnPlateau` | `train.py:97` | Scheduler that reduces learning rate when validation stalls |
| `DataLoader` | `data_loader.py:176` | Batches and shuffles data for training |
| `model.train()` / `model.eval()` | `train.py:32` | Switches dropout and batch normalization between training/inference modes |
| `torch.no_grad()` | `eval.py:58` | Disables gradient computation during evaluation (saves memory) |
| `model.state_dict()` | `train.py:123` | Serializes all model weights for saving/loading |

---

## 17. Metric Definitions

### Accuracy
```
Accuracy = (Number of correct predictions) / (Total predictions)
         = 187 / 190 = 98.4%
```
Simple and intuitive, but can be misleading with imbalanced classes.

### Precision (per-class)
"Of everything the model labeled as class X, how many were actually class X?"
```
Precision_sitting = True_Positives / (True_Positives + False_Positives)
```
High precision = few false alarms.

### Recall (per-class)
"Of all actual class X samples, how many did the model find?"
```
Recall_sitting = True_Positives / (True_Positives + False_Negatives)
```
High recall = few missed detections.

### F1-Score (per-class)
Harmonic mean of precision and recall:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Ranges from 0 (worst) to 1 (perfect). Penalizes models that sacrifice one for the other.

### Macro Average
Average the per-class metrics equally:
```
Macro_F1 = (F1_standing + F1_sitting + F1_lying) / 3
```
Treats all classes equally regardless of size — gives sitting equal weight to standing.

### Weighted Average
Average weighted by class support (number of samples):
```
Weighted_F1 = (F1_standing × 84 + F1_sitting × 4 + F1_lying × 102) / 190
```
Reflects overall performance proportional to class sizes.

---

## 18. Key Design Decisions & Justifications

### Why pig-ID split instead of random split?
**Justification**: Random splitting causes data leakage. Frames from the same pig share visual features (body shape, markings, stall position). A model trained on pig 5's standing frames and tested on pig 5's lying frames has already learned pig 5's appearance — inflating accuracy. Pig-ID split ensures generalization to truly unseen pigs.

**How to verify**: Compare pig-ID split accuracy vs. random split accuracy. If random split gives significantly higher numbers, data leakage is confirmed.

### Why a fixed crop box instead of per-frame detection?
**Justification**: The camera is mounted at a fixed position relative to the stall. The stall bars are always in the same pixel locations across all frames. A fixed crop is simpler, faster, and more reproducible than running an object detector per frame. The crop box coordinates were determined by visual inspection across multiple pigs and sessions.

**Trade-off**: If the camera shifts position, the crop would need recalibration. For our fixed setup, this is not an issue.

### Why 224×224 input size?
**Justification**: All three backbone architectures were designed and pre-trained on 224×224 ImageNet images. Using a different size would require architectural modifications and would not benefit from the pre-trained weights. 224×224 retains sufficient detail for posture classification (the overall body shape is clearly visible at this resolution).

### Why class-weighted loss instead of oversampling?
**Justification**: Both approaches address class imbalance. We chose class weighting because:
- Simpler to implement (one line of code vs. custom sampler)
- Doesn't duplicate training samples (oversampling can cause overfitting on repeated minority images)
- Equivalent mathematical effect: both make minority class errors cost more

### Why Adam optimizer instead of SGD?
**Justification**: Adam converges faster than SGD, which matters with small datasets (918 training images) and limited epochs (30). SGD with momentum can achieve slightly better final accuracy on large datasets but requires more careful learning rate tuning.

### Why ReduceLROnPlateau instead of a fixed schedule?
**Justification**: Adaptive — responds to actual training dynamics rather than a pre-set schedule. If the model converges quickly (within 10 epochs), the scheduler reduces the learning rate early. If it's still improving at epoch 20, the learning rate stays high.

### Why MobileNetV2 is recommended despite lower ImageNet accuracy?
**Justification**: On our specific task (3-class pig posture), MobileNetV2 matches larger models at 98.4% test accuracy. This is because pig posture classification is much simpler than 1,000-class ImageNet — the "capacity gap" between MobileNetV2 and Xception doesn't matter here. MobileNetV2's advantages (10× fewer parameters, faster inference, deployable on edge devices like OAK-D) make it the practical choice.

### Why depth prefilter threshold = 1463mm?
**Justification**: Determined empirically using `validate_pig_detection.py`:
1. Manually labeled 51 frames as pig-present or empty (ground truth)
2. Computed median depth for each frame
3. Found the threshold that maximizes detection accuracy against ground truth
4. 1463mm gave the best separation between pig-present (median ~600-1200mm) and empty (median ~1500mm+)

---

## Reproducibility Checklist

- [ ] All random seeds are fixed (PyTorch, NumPy)
- [x] Data split is deterministic (pig IDs, not random)
- [x] Best model saved by validation accuracy (not final epoch)
- [x] All hyperparameters documented in `config.py`
- [x] Raw data preserved alongside processed versions
- [x] Label CSV tracks which frames were labeled and by whom
- [x] Training history (loss/accuracy per epoch) saved as JSON
- [x] Evaluation metrics saved as JSON
- [x] Confusion matrices saved as plots
- [ ] Random seed should be added for full reproducibility

---

## FAQ

<details>
<summary><strong>Q1: How were posture labels created, and how should they be documented for a paper?</strong> — <a href="#2-data-collection--structure">See Section 2</a></summary>

**Short answer**: Posture labels (standing/sitting/lying) were created by **manual human annotation** — a single annotator viewed each RGB image and assigned a class. No algorithm was used for posture classification during labeling. The depth sensor is only used for pig *presence* detection (is a pig in the stall? yes/no), not for posture.

**Why manual labeling?** This is the standard approach in supervised learning for animal behavior classification. The model learns to replicate human-provided labels. Published works in precision livestock farming follow the same protocol — see references below.

**What a paper needs for the labeling protocol**:

1. **Written posture definitions** (agreed upon *before* labeling begins):
   - **Standing**: Sow's torso is elevated with at least 3 legs visibly supporting body weight. The ventral body surface is not in contact with the floor.
   - **Sitting**: Hindquarters in contact with the floor, front legs extended or supporting the anterior body. A transitional posture between standing and lying.
   - **Lying**: Torso in contact with the floor, either lateral recumbency (on side) or sternal recumbency (on belly).

2. **Inter-annotator agreement** (recommended for publication):
   - Have a second independent annotator label a random subset of 50–100 frames
   - Compute **Cohen's kappa (κ)** to measure agreement beyond chance:
     - κ > 0.80 = almost perfect agreement
     - κ = 0.61–0.80 = substantial agreement
     - κ < 0.60 = needs discussion and re-calibration of definitions
   - Report kappa in the paper's methods section

3. **Edge case handling** (document how these were resolved):
   - Mid-transition frames (pig moving from standing to lying): **skipped** during labeling
   - Partially occluded frames: **skipped**
   - Ambiguous postures: **skipped** — only clearly identifiable postures were labeled

4. **Annotator information**: single annotator, trained on example images before labeling

**Current status**: We have single-annotator labels with ambiguous frames skipped. For a full paper, inter-annotator agreement on a subset should be added.

**References**:
- Zheng, C., et al. (2018). "Automatic recognition of lactating sow postures from depth images by deep learning detector." *Computers and Electronics in Agriculture*, 147, 51–63. — Used manual posture labels as ground truth for CNN training.
- Nasirahmadi, A., et al. (2019). "Using machine vision for investigation of changes in pig group lying patterns." *Computers and Electronics in Agriculture*, 157, 495–503. — Manual labeling protocol with inter-annotator agreement for posture classification.
- Riekert, M., et al. (2020). "Automatically detecting pig position and posture by 2D camera imaging and deep learning." *Computers and Electronics in Agriculture*, 174, 105391. — Defined posture classes with written criteria, used manual annotation as ground truth.


</details>

<details>
<summary><strong>Q2: Why does prefilter_depth.py crop in memory but crop_images.py saves to disk? Isn't that redundant?</strong> — <a href="#4-step-2-prefilter_depthpy--pig-presence-detection">See Section 4</a> / <a href="#6-step-4-crop_imagespy--region-of-interest-extraction">See Section 6</a></summary>

**Short answer**: No, they serve different purposes. Both use the same crop box (`CROP_TOF` from `config.py`), but:

- **`prefilter_depth.py`** (Step 2): Crops the raw depth data **in memory temporarily** to compute the median depth for pig presence detection. It doesn't save any image — it just needs the depth values in the stall region to decide "pig present or not." The cropped data is discarded after the median is computed.

- **`crop_images.py`** (Step 4): Crops the JPG images and **saves them to disk** as new `_cropped.jpg` files. These are the actual inputs to the training pipeline — the model trains on cropped images so it only sees the target pig, not neighbor pigs or ceiling hardware.

**Why this is good design**:
- The crop box is defined **once** in `config.py` (`CROP_TOF = (120, 30, 500, 480)`) — single source of truth. If the camera position changes, you update one number.
- Prefilter runs before labeling (to skip empty frames), so it doesn't need saved crops — saving files would be wasteful at that stage.
- Cropping to disk happens later, only for frames that will be used for training.

**Pipeline order**:
```
scan_data.py → prefilter_depth.py → label_tool.py → crop_images.py → train.py
                (crops in memory)                    (saves to disk)
```

</details>

<details>
<summary><strong>Q3: How were the crop box coordinates (120, 30, 500, 480) and depth filter range (200–5000mm) determined?</strong> — <a href="#6-step-4-crop_imagespy--region-of-interest-extraction">See Section 6</a> / <a href="#4-step-2-prefilter_depthpy--pig-presence-detection">See Section 4</a></summary>

These are empirically determined values based on human analysis of the physical stall setup and sensor specifications. Here's the justification for each:

### Crop box: `CROP_TOF = (120, 30, 500, 480)`

The OAK-D ToF camera is side-mounted at a fixed position on each farrowing stall. The stall partitions (metal bars) are visible at consistent pixel locations across all frames because the camera-to-stall geometry is identical for every stall.

**How the coordinates were determined:**
1. Representative frames were opened across multiple pigs and recording sessions
2. The left stall partition bar was identified at approximately x=110–120 pixels in the 640×480 ToF image
3. The right stall partition bar was identified at approximately x=500–510 pixels
4. Ceiling hardware (pipes, feeders) was identified at approximately y=0–30 pixels
5. The crop was set to **(120, 30, 500, 480)** — just inside the bars on all sides, keeping the full floor level (y=480 = bottom of frame)

**Why a fixed box works:** The stall dimensions are physically standardized (same design for all 20 stalls). The camera mount position is identical across stalls. Therefore, the bar positions in pixel coordinates are consistent across all frames. This was verified by overlaying the crop box on images from different pigs and sessions.

**For the paper:** "The ROI was defined by identifying the pixel coordinates of the stall partition bars across representative frames from all 20 stalls. The partitions appeared consistently at x≈115 (left) and x≈505 (right) in the 640×480 ToF coordinate frame. The crop region was set to (120, 30, 500, 480) to exclude the partitions, neighboring stall contents, and overhead infrastructure while preserving the full target sow."

### Depth filter: `200mm < depth < 5000mm`

This filter removes invalid depth readings before computing the median for pig presence detection.

- **200mm minimum**: The OAK-D ToF sensor has a minimum operating range of approximately 200mm. Depth values below this are sensor noise or invalid readings caused by objects too close to the camera. This is a hardware specification, not an arbitrary choice.
- **5000mm maximum**: The farrowing stalls are approximately 2 meters deep. Any depth reading above 5000mm (5 meters) is physically impossible in the stall environment and represents sensor artifacts — multipath reflections, interference, or readings through gaps in the stall structure.

**For the paper:** "Depth values outside the 200–5000mm range were excluded from analysis. The lower bound corresponds to the OAK-D ToF sensor's minimum operating range; the upper bound exceeds the maximum physical stall depth (~2m) and filters multipath reflection artifacts."

### Pig presence threshold: `1463mm`

Unlike the crop box and depth range, this threshold was **empirically validated**:
1. 51 frames were manually labeled as pig-present or empty (ground truth)
2. The median depth was computed for each frame using the filtered, cropped region
3. The distribution of median depths showed clear separation: pig-present frames clustered at 600–1200mm, empty frames at 1500mm+
4. The threshold of 1463mm was selected to maximize classification accuracy against the ground truth
5. This was verified using `validate_pig_detection.py`

**For the paper:** "The pig presence threshold (1463mm) was determined empirically by computing the median depth of 51 manually annotated frames and selecting the value that maximized detection accuracy. Pig-present frames exhibited median depths of 600–1200mm (body occluding the camera's view), while empty stall frames showed median depths exceeding 1500mm (camera viewing the far wall)."

</details>

<details>
<summary><strong>Q4: How does the model predict postures on images it has never seen, and how do we know the 98.4% accuracy is real?</strong> — <a href="#8-step-6-evalpy--evaluation--metrics">See Section 8</a></summary>

### Why should we trust these numbers?

The 98.4% accuracy is not self-reported by the model. It's computed by an independent comparison:

1. The model sees 190 test images (pigs 16-19) — **only the pixels, never the labels**
2. For each image, the model outputs a prediction based on patterns it learned from training pigs (0-11)
3. We compare those predictions against the human-annotated ground truth labels
4. 187 of 190 predictions matched → 187/190 = 98.4%

The model and the ground truth are completely separate. The model guesses, then `eval.py` grades it.

### How can the model predict postures on pigs it has never seen?

**Analogy**: A teacher gives students 80 practice math problems WITH answer keys. Students study the underlying patterns ("quadratic equations → use the quadratic formula"). Then the teacher gives 20 NEW problems with no answers. Students solve them using the patterns they learned. The teacher grades the answers against the answer key. Students never see the answer key.

In our pipeline:
- **Practice problems with answers** = 918 training images with manual posture labels
- **New problems** = 190 test images (model sees pixels only, NOT the labels)
- **Student answers** = model predictions (`eval.py` line 60: `model(x.to(device)).argmax(1)`)
- **Answer key** = human labels (`eval.py` line 61: `y.numpy()`)
- **Grading** = `accuracy_score(labels, preds)` — scikit-learn counts how many match

During training, the model learned general visual patterns:
- "Legs visible under elevated body" → standing
- "Body flat on floor, no leg gap" → lying
- "Hindquarters down, front legs extended" → sitting

These patterns are the same regardless of which specific pig it is. So when the model sees pig 17 for the first time, it applies the same rules.

### What makes the evaluation scientifically valid?

| Concern | How we address it |
|---------|-------------------|
| Data leakage? | Pig-ID split — test pigs (16-19) never appear in training (0-11). Enforced in `config.py` lines 22-24. |
| Cherry-picked results? | We report ALL 9 models (3 backbones × 3 modalities), not just the best one. |
| Custom metrics that could be buggy? | Metrics computed by scikit-learn (`accuracy_score`, `classification_report`, `confusion_matrix`) — open-source, peer-reviewed, used in thousands of papers. |
| Black box? | Confusion matrix shows the exact 3 errors. Each misclassified frame can be inspected individually. |
| Overfitting? | Best model selected by validation accuracy (pigs 12-15), not by test accuracy. Test set is only used once, at the end. |

### What established methods does the training use?

Every component maps to published, peer-reviewed work:

- **Transfer learning from ImageNet** — Yosinski et al. (2014). "How transferable are features in deep neural networks?" *NeurIPS*. Cited 12,000+ times.
- **Cross-entropy loss** — Standard classification loss function since the 1990s, derived from information theory (Shannon, 1948).
- **Adam optimizer** — Kingma & Ba (2015). "Adam: A Method for Stochastic Optimization." *ICLR*. Cited 180,000+ times.
- **Class weighting for imbalanced data** — King & Zeng (2001). Standard approach; scikit-learn and PyTorch both implement it natively.
- **MobileNetV2** — Sandler et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks." *CVPR*. Cited 15,000+ times.
- **ReduceLROnPlateau** — Standard adaptive learning rate scheduling, built into PyTorch.

None of these methods are novel. The contribution is the application to sow posture classification using OAK-D ToF multi-modal data — not the methods themselves.

</details>

---

## 19. File Creation & Update Log

### Task A Files (committed, stable)

| File | Created | Last Modified | Purpose |
|---|---|---|---|
| config.py | 2026-02 | 2026-04-07 | Central config: paths, crop box, splits, hyperparams |
| scan_data.py | 2026-02 | 2026-04-02 | Build metadata.csv from OAK-D folder structure |
| prefilter_depth.py | 2026-02 | 2026-04-02 | Depth-based pig presence detection |
| crop_images.py | 2026-02 | 2026-04-02 | Crop images to stall ROI |
| label_tool.py | 2026-02 | 2026-04-02 | Interactive posture labeling GUI |
| data_loader.py | 2026-02 | 2026-04-02 | PyTorch Dataset/DataLoader, multi-modal, pig-ID splits |
| models.py | 2026-02 | 2026-04-02 | SingleModalModel: MobileNetV2/Xception/DenseNet121 |
| train.py | 2026-02 | 2026-04-02 | Training loop with class-weighted loss |
| eval.py | 2026-02 | 2026-04-02 | Evaluation: confusion matrix, classification report |
| run_inference.py | 2026-02 | 2026-04-02 | One-command pipeline: crop + predict + heatmap |
| infer_batch.py | 2026-02 | 2026-04-02 | Batch inference with multi-modal support |
| run_pipeline.py | 2026-02 | 2026-04-02 | 7-step orchestrator (scan → eval) |
| audit_data.py | 2026-03 | 2026-04-02 | Detect missing/shifted pig IDs per folder |
| find_sitting.py | 2026-02 | 2026-04-02 | Active learning for rare sitting class |
| validate_pig_detection.py | 2026-02 | 2026-04-02 | Validate depth threshold vs manual ground truth |
| inspect_prefilter.py | 2026-02 | 2026-04-02 | Visual QA for pig presence classifications |
| depth_profile_viewer.py | 2026-02 | 2026-04-02 | Interactive depth cross-section viewer |
| normalize_posture3_labels.py | 2026-02 | 2026-04-02 | Remap 4-class labels to 3-class |
| align_images.py | 2026-02 | 2026-04-02 | Warp RGB into ToF frame using calibration |
| check_alignment.py | 2026-02 | 2026-04-02 | Visual alignment QA |
| depthai_tof_align_official.py | 2026-02 | 2026-04-02 | Live OAK-D hardware alignment demo |
| pretrain_sowbot.py | 2026-02 | 2026-04-02 | Pre-train on Sowbot 2022 IR dataset |
| plot_pipeline_summary.py | 2026-02 | 2026-04-02 | Pipeline flowchart + results figures |
| plot_pretrain_comparison.py | 2026-02 | 2026-04-02 | Sowbot pre-training comparison figures |
| make_slides.py | 2026-02 | 2026-04-02 | 17-slide PowerPoint for weekly meetings |

### Task B Files (active development)

| File | Created | Last Modified | Purpose | Notes |
|---|---|---|---|---|
| point_cloud.py | 2026-03 | 2026-04-08 | Core 3D engine: backprojection, segmentation, vulva detection, PLY/OBJ export | 1200+ lines. Implements depth→3D, auto pig segmentation, IR local darkness vulva detection, stylized depth portraits. Heavily iterated. |
| build_depthmaps.py | 2026-03 | 2026-04-08 | Batch driver for point_cloud.py | Processes all depth_*.raw → depthmap/ folders. 232 pig folders generated. Validates raw file sizes. |
| crop_vulva_pointcloud.py | 2026-03 | 2026-04-08 | Interactive vulva rectangle selection + cropped 3D export | RectangleSelector GUI, context mask priority chain, auto-box suggestion via IR darkness. |
| build_vulva_rect_crops.py | 2026-03 | 2026-04-08 | Batch driver for crop_vulva_pointcloud.py | Generates depthmap_rect/ folders with vulva-centered crops. |
| build_paper_vulva_surfaces.py | 2026-03 | 2026-04-08 | Paper-style vulva measurement pipeline | 800+ lines. Core replication of Xu et al.: plane fitting (SVD), 300×300 OS rasterization, vulva seed detection, 35% dilation, diffuse inpainting, ellipse-fitted measurements. |
| label_vulva.py | 2026-03 | 2026-04-03 | Interactive vulva annotation: quality + polygon mask | Quality classification (good/tail_closed/too_close/bad/skip), polygon drawing for mask creation. |
| label_vulva_length_width.py | 2026-03 | 2026-04-07 | Manual L/W measurement via IR point-click | 600+ lines. Click 2 points for length + 2 for width, backproject to 3D metric distance, ground truth enrichment. |
| compare_vulva_ground_truths.py | 2026-03 | 2026-04-07 | Validate measured vs caliper ground truth | Parses Excel workbooks (openpyxl), joins by (date, pig_id), computes deltas. |
| measure_vulva_dataset.py | 2026-03 | 2026-04-07 | DEPRECATED — redirects to label_vulva_length_width.py | Old batch workflow, replaced. |
| measure_vulva_manual.py | 2026-03 | 2026-04-07 | DEPRECATED — redirects to label_vulva_length_width.py | Old interactive measurement, replaced. |
| extract_rear_view_pig_manual.py | 2026-03 | 2026-04-08 | Manual polygon selection for rear-view pig extraction | Imports PolygonDrawer from label_vulva, generates manually-masked point clouds. |
| view_pointcloud.py | 2026-03 | 2026-04-03 | Interactive 3D viewer for PLY/OBJ files | Tkinter + matplotlib rendering, mouse rotation/pan/zoom, IR preview panel. |
| render_depthmaps.py | 2026-03 | 2026-04-03 | Backfill utility: regenerate grayscale depth portraits | Imports render_stylized_depthmap from point_cloud.py. |

### Data & Output Files

| File/Directory | Created | Content |
|---|---|---|
| images/ (12 sessions) | 2026-02-11 | Raw OAK-D data: 678 pig captures, ~2,054 image files |
| depthmap/ (232 folders) | 2026-03 | 3D point clouds + meshes for all pigs |
| depthmap_rect/ | 2026-04 | Vulva rectangle crops + paper_vulva/ measurement outputs |
| rear_view_pig_manual/ | 2026-04 | Manual polygon rear-view extractions (pig0 only) |
| labels/vulva_length_width_labels.csv | 2026-04 | Manual L/W measurements with ground truth deltas |
| ground_truths/vulva_ground_truths.csv | 2026-04 | Extracted caliper measurements from Excel |
| ground_truths/Feb_2026.xlsx | 2026-02 | Source caliper measurements from animal science team |
| models/posture3_ir_best.pth | 2026-03 | Trained MobileNetV2/IR model (8.7MB, 98.4% test acc) |
| outputs/posture_heatmap.png | 2026-03 | Time × pig posture heatmap (majority vote per hour) |
| outputs/timestamp_summary.csv | 2026-03 | Per-session posture distribution summary |

### Key Iteration History

- **2026-02**: Task A built end-to-end (scan → train → eval). Initial binary classifier, then expanded to 3-class.
- **2026-03-03**: Labels discovered to have ~40% error rate. Decision: relabel all 796 pig-present frames from scratch. Old labels archived to labels/backup/.
- **2026-03-30**: Task A finalized at 98.4%. Fed_pig dataset (75k frames) processed. Started Task B.
- **2026-04-02**: point_cloud.py initial version. Depth backprojection + auto segmentation working.
- **2026-04-03**: label_vulva.py, view_pointcloud.py, render_depthmaps.py created. Vulva polygon annotation tool functional.
- **2026-04-07**: label_vulva_length_width.py, compare_vulva_ground_truths.py, measure_vulva_dataset.py, config.py updated. Ground truth comparison pipeline working. Old measurement scripts deprecated.
- **2026-04-08**: Major iteration on build_paper_vulva_surfaces.py (800+ lines), build_depthmaps.py, build_vulva_rect_crops.py, crop_vulva_pointcloud.py, extract_rear_view_pig_manual.py, point_cloud.py. Paper-style surface pipeline fully operational with plane fitting, seed detection, inpainting, and ellipse measurements.
- **2026-04-09**: Full project analysis. METHODOLOGY.md updated with Task B documentation and file logs.
