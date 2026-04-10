# PAAL Pipeline — Posture Classification & Vulva Measurement

## What This Is
OAK-D ToF camera pipeline for (A) classifying pig postures and (B) measuring vulva dimensions for automated estrus detection. Replicates methodology from Xu et al. (2023, 2024) — see docs/METHODOLOGY.md for full reference paper mapping.

## Quick Commands

### Task A: Run inference on new dataset
```bash
python run_inference.py "C:\PAAL_data\Fed_pig\Pictures_OAk"   # Windows lab PC
python run_inference.py /path/to/data                          # Mac
```

### Task B: Vulva measurement pipeline
```bash
python build_depthmaps.py                                       # Step 1: depth → 3D point clouds (depthmap/)
python build_vulva_rect_crops.py                                # Step 2: interactive vulva rectangle crop (depthmap_rect/)
python build_paper_vulva_surfaces.py depthmap_rect/SESSION/PIG  # Step 3: paper-style surface analysis + measurements
python label_vulva_length_width.py                              # Alt: manual L/W via IR point-click
python compare_vulva_ground_truths.py                           # Validate vs caliper ground truth
```

### Other commands
```bash
python infer_batch.py --data-dir /path/to/data          # Batch inference only
python crop_images.py --data-dir /path/to/data           # Crop only
python point_cloud.py --depth-raw X --ir-img Y           # Single pig point cloud + mesh
python view_pointcloud.py depthmap/SESSION/PIG            # Interactive 3D viewer
python extract_rear_view_pig_manual.py INPUT              # Manual polygon rear-view extraction
```

## Project Structure

### Task A (Posture — COMPLETE)
- `run_inference.py` — Single-command pipeline (crop + predict + heatmap)
- `infer_batch.py` — Batch inference with depth prefilter
- `crop_images.py` — Crop OAK-D images to stall region
- `train.py` / `eval.py` — Model training and evaluation
- `models.py` — MobileNetV2/Xception/DenseNet121 architectures
- `data_loader.py` — PyTorch Dataset with pig-ID splits
- `config.py` — All paths, crop boxes, class definitions
- `models/posture3_ir_best.pth` — Trained MobileNetV2 IR model (8.7MB, 98.4%)

### Task B (Vulva — IN PROGRESS)
- `point_cloud.py` — Core 3D engine: depth backprojection, pig segmentation, vulva IR detection
- `build_depthmaps.py` — Batch: all depth_*.raw → depthmap/ (232 pigs processed)
- `crop_vulva_pointcloud.py` — Interactive vulva rectangle selection + cropped 3D export
- `build_vulva_rect_crops.py` — Batch driver for rectangle crops → depthmap_rect/
- `build_paper_vulva_surfaces.py` — Paper-style: plane fit → 300×300 OS → seed detect → dilate 35% → inpaint → measure
- `label_vulva_length_width.py` — Manual L/W annotation via IR point-click with 3D backprojection
- `compare_vulva_ground_truths.py` — Validate measured vs caliper ground truth
- `label_vulva.py` — Interactive vulva quality + polygon annotation
- `extract_rear_view_pig_manual.py` — Manual polygon rear-view extraction
- `view_pointcloud.py` — Interactive 3D PLY/OBJ viewer
- `render_depthmaps.py` — Regenerate depth portraits from bundles

## Key Technical Details
- **Task A Model**: MobileNetV2 on IR modality, 3 classes, 98.4% test accuracy
- **Depth prefilter**: median depth < 1463mm = pig present (50px bar margin)
- **Crop box (TOF)**: (120, 30, 500, 480) in config.py
- **Pig ID overflow**: IDs > 19 normalized with `% 20`
- **Vulva detection**: IR local darkness via DoG (3σ − 17σ), 95th percentile threshold
- **Paper measurements**: CV = W × L × H (cubic volume, R²=0.92 in reference paper)
- **Camera intrinsics**: fx=471.6, fy=471.4, cx=323.9, cy=246.8

## Reference Papers
1. Xu et al. (2023) "Detecting sow vulva size change around estrus using machine vision" — Smart Agri. Tech.
2. Xu et al. (2024) "Automated oestrous detection in sows using a robotic imaging system" — Biosystems Eng.
3. Xu et al. (2024) "Developing a Sow Vulva Volume Estimation Pipeline" — J. ASABE

## Current Status (2026-04-10)
- **Task A** (posture classification): COMPLETE at 98.4%
- **Task B** (vulva measurement): Paper-style surface pipeline operational, detection tuning in progress
  - 232 pig 3D point clouds generated (depthmap/)
  - Vulva rectangle crops done on IR images (not RGB) — IR is pixel-aligned with depth
  - 48 valid crops (186 cancelled = no visible vulva), stored in depthmap_rect/
  - Paper-style surface pipeline: plane fit → auto-grid OS → Open3D DBSCAN vulva detection → ellipse fit → 35% dilate → inpaint → measure
  - Open3D DBSCAN + ellipse-fit detection gives strictly oval/circular vulva masks
  - Escalating threshold strategy: starts at 75th pct, raises to 80/85/90 if ellipse covers >40% of grid
  - Ground truth caliper data in `ground_truths/vulva_ground_truths.csv` (20 pigs, 11 days)
  - Sample results: pig0 L=55.86mm W=41.62mm (GT: W=41.0, L=79.3) — width matches well
  - Known issue: some crops (e.g., pig15) where entire surface protrudes uniformly still produce oversized ellipses — threshold escalation not fully solving this yet

## Key Design Decisions (Task B)
- **IR over RGB**: IR images used for rectangle selection and texture — pixel-aligned with depth (same sensor). RGB is from a separate camera with ~1.73cm baseline and calibration.json is empty (0 bytes), so no extrinsic alignment available.
- **Auto grid sizing**: `grid = sqrt(n_points / 0.40)` clamped to [64, 300] for ~40% coverage (OAK-D gives ~3k-8k points per crop vs paper's L515 which gives much denser data)
- **No frequency filtering**: Paper (Xu 2023) does NOT use frequency filtering or U-Net — those are from the 2024 follow-up. Current implementation follows the 2023 paper: regionprops → largest round region.
- **Vulva shape constraint**: Vulva is always circular or oval. Detection uses DBSCAN + circularity scoring, then cv2.fitEllipse to enforce strictly oval mask.
- **Cancelled crops**: When user cancels rectangle selection (no visible vulva), `error.txt` is written — these are skipped in batch processing.
- **JET colormap**: Paper uses linear JET for depth visualization: linear normalization min→max, cv2.COLORMAP_JET
- **Open3D for detection**: User requested Open3D (3D library) over OpenCV for vulva detection — works in 3D point space via DBSCAN clustering

## Data Cleaning (Akbar — Task 2)
Audited all 3,766 timestamp folders. 97.6% (3,674) have exactly 20 pigs.

### Camera Magnet Corrections Applied
All corrections are scripted in `fix_pig_ids.py` and `fix_predictions_csv.py`:

| Range | Folders | Issue | Fix |
|-------|---------|-------|-----|
| 02/13 13:41–19:22 | 32 | pig19=dup of pig18, pig20=real pig19 | delete pig19, rename pig20→pig19, delete pig21+ |
| 02/17 06:01 | 1 | overflow | same pattern |
| 02/18 10:54–11:15 | 3 | pig19+pig20 dups | delete pig19+pig20, rename pig21→pig19, delete pig22+ |
| 02/18 15:33–19:00 | 14 | pig19=dup, pig20=real pig19 | delete pig19, rename pig20→pig19, delete pig21+ |
| 02/19 15:09–17:58 | ~10 | pig19=pig17 dup, pig20=pig18 dup | delete pig19+pig20, rename pig21→pig19, delete pig22+ |
| 02/20, 03/08, 03/10-03/11 | few | simple overflow | delete pig20+ |

### Data Gaps
- Camera down: 02/16–02/17, 02/20–02/22 (no frames exist)
- Pig 18 & 19: empty stalls throughout
- **Clean data window: 02/23–03/12** (17 days, clear posture cycles)

### Tools Built
- `audit_data.py` — scan folders, report missing/overflow pig IDs
- `detect_missing_pigs.py` — depth-based comparison for < 20 pig folders
- `detect_id_shifts.py` — stall depth fingerprinting (inconclusive — stalls equidistant)
- `fix_pig_ids.py` — apply file rename/delete corrections (with --dry-run)
- `fix_predictions_csv.py` — apply same corrections to predictions CSV
- `regenerate_heatmap.py` — rebuild heatmaps from CSV in seconds (no 30-min re-run)

## Pending Tasks
### Task A (Akbar)
1. Compare predictions with Lucas's ground truth
2. Re-run inference on fully cleaned data for final heatmap
3. Share 3D point clouds with Dr. Zhou

### Task B (Srikar)
1. Fix oversized detection for uniformly-protruding crops (pig15-type cases)
2. Characterize OAK-D ToF depth accuracy (paper reports 3.4±3.0mm for L515)
3. Compute formal daily behavioral indices (SI, SLI, LLI, PCF) for estrus model input
4. Build 1D CNN estrus detection model (DFW + behavior + vulva volume)
5. Multi-day longitudinal vulva tracking across estrus cycle

### Together
1. Manual vulva labeling (40-50 samples) — label_vulva.py ready
2. Compare labels with ground truth

## Data Locations
- **Lab PC (Windows):** `C:\PAAL_data\Fed_pig\Pictures_OAk\`
- **Mac:** `~/Downloads/paal_pipeline/data/`
- **Model on Google Drive:** `posture3_ir_best.pth` (8.7MB)
- **Ground truth calipers:** `ground_truths/vulva_ground_truths.csv` and `ground_truths/Feb_2026.xlsx`
- **Reference paper PDF:** `papers/1-s2.0-S2772375522000557-main.pdf`
