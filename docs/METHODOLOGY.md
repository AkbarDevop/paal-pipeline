# Technical Methodology — PAAL Sow Posture Classification

This document explains every step, decision, metric, and library in the pipeline.
Written for scientific reproducibility and paper-readiness.

---

## Table of Contents

1. [Hardware Setup](#1-hardware-setup)
2. [Data Collection & Structure](#2-data-collection--structure)
3. [Step 1: scan_data.py — Data Indexing](#3-step-1-scan_datapy--data-indexing)
4. [Step 2: prefilter_depth.py — Pig Presence Detection](#4-step-2-prefilter_depthpy--pig-presence-detection)
5. [Step 3: label_tool.py — Manual Annotation](#5-step-3-label_toolpy--manual-annotation)
6. [Step 4: crop_images.py — Region of Interest Extraction](#6-step-4-crop_imagespy--region-of-interest-extraction)
7. [Step 5: train.py — Model Training](#7-step-5-trainpy--model-training)
8. [Step 6: eval.py — Evaluation & Metrics](#8-step-6-evalpy--evaluation--metrics)
9. [Libraries & Technologies](#9-libraries--technologies)
10. [Metric Definitions](#10-metric-definitions)
11. [Key Design Decisions & Justifications](#11-key-design-decisions--justifications)

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
4. Filter valid depth range: keep only pixels where 200mm < depth < 5000mm (removes noise and zero-depth pixels)
5. Compute the **median** of remaining depth values
6. **Decision rule**: `median_depth < 1463mm` → pig present

**Why median, not mean?**
- Median is robust to outliers (a few noisy pixels don't affect it)
- Mean would be skewed by a single very-far or very-near pixel

**Why 1463mm threshold?**
- The camera is side-mounted at a known distance from the opposite wall
- When a pig is present, its body blocks the view → median depth is low (600-1200mm)
- When empty, the camera sees the far wall → median depth is high (1500mm+)
- 1463mm was determined empirically by inspecting the depth histogram of frames with and without pigs (see `validate_pig_detection.py` which checks against manual ground truth)

**Output**: `labels/presence_filter.csv` with columns: `pig_present` (0 or 1), `median_depth`, `threshold`

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

This means: from the 640×480 depth/IR image, keep only x=120..500, y=30..480.

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

**Result**: The cropped region is approximately 380×450 pixels (ToF) or 760×750 pixels (RGB), containing only the target pig. This is then resized to 224×224 for model input.

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

The model has NEVER seen these 4 pigs during training. This is the number reported in the paper.

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

## 9. Libraries & Technologies

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

## 10. Metric Definitions

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

## 11. Key Design Decisions & Justifications

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
