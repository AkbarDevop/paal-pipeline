# PAAL Pipeline — Pig Posture Classification

## What This Is
OAK-D camera pipeline for classifying pig postures (standing/sitting/lying) using MobileNetV2 on IR images. Task A is complete at 98.4% accuracy.

## Quick Commands

### Run inference on new dataset (single command)
```bash
# Windows lab PC:
python run_inference.py "C:\PAAL_data\Fed_pig\Pictures_OAk"

# Mac:
python run_inference.py /path/to/data
```
This crops images, runs the IR model, and outputs CSV + heatmaps (overall + per-pig) + timestamp summary.

### Other useful commands
```bash
python infer_batch.py --data-dir /path/to/data          # Batch inference only
python crop_images.py --data-dir /path/to/data           # Crop only
python point_cloud.py --depth-raw X --color-img Y        # 3D point cloud
```

## Project Structure
- `run_inference.py` — Single-command pipeline (crop + predict + heatmap)
- `infer_batch.py` — Batch inference with depth prefilter
- `crop_images.py` — Crop OAK-D images to stall region
- `point_cloud.py` — 3D point cloud generation (plotly)
- `train.py` — Model training
- `models/posture3_ir_best.pth` — Trained MobileNetV2 IR model (8.7MB)
- `config.py` — All paths, crop boxes, class definitions
- `data_loader.py` — Dataset loading and preprocessing
- `labels/` — Label CSVs and detection results

## Key Technical Details
- Model: MobileNetV2 on IR modality, 3 classes, 98.4% test accuracy
- Camera files: `ir_vis`/`depth_vis` in filenames map to `ir`/`depth` model keys
- Pig IDs > 19 get normalized with `% 20` (camera counter overflow)
- Heatmap uses majority vote per hour cell
- Depth prefilter: median depth < 1463mm = pig present
- Crop box (TOF): defined in config.py as CROP_TOF

## Current Status (2026-03-30)
- Task A (posture classification): COMPLETE
- Fed_pig dataset results: 75,120 frames, 73,521 predicted, 30.4% standing, 3.2% sitting, 66.4% lying, avg confidence 0.952
- Per-pig heatmaps and timestamp summary just added (commit 1baec57) — need to re-run on lab PC
- Point cloud files generated (plotly HTML + PLY) — need to upload to Google Drive for supervisor
- Task B (vulva segmentation): labeling tool built (`label_vulva.py`), labeling not started

## Pending Tasks
1. Re-run `run_inference.py` on fed_pig dataset on lab PC to get per-pig heatmaps + timestamp summary
2. Upload point cloud files to Google Drive for supervisor/Dr. Zhou
3. Post X/Twitter thread (draft ready)
4. Start Task B vulva labeling

## Data Locations
- **Mac:** `~/Downloads/paal_pipeline/data/`
- **Lab PC (Windows):**
  - Old dataset: `C:\back\`
  - New dataset: `C:\PAAL_data\Fed_pig\Pictures_OAk\`
- **Model on Google Drive:** `posture3_ir_best.pth` (8.7MB) — also committed to repo
