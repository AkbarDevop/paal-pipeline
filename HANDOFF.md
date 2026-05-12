# HANDOFF.md

This document is the single source of truth for continuing work on this repository. Read this first.

---

## What this repo does

OAK-D ToF camera pipeline for monitoring sows (mother pigs). Two tasks:

- **Task A — Posture classification** (standing / sitting / lying). Shipped, 98.4% test accuracy.
- **Task B — Vulva segmentation / measurement** for automated estrus detection. In progress.

The two tasks are independent. You can work on either without touching the other.

---

## Repository layout

```
paal-pipeline/
├── HANDOFF.md                 ← you are here
├── README.md                  ← general overview
├── CLAUDE.md                  ← project notes (status, decisions, history)
├── config.py                  ← shared paths and constants (used by both tasks)
├── requirements.txt
├── PAAL_Posture_Classification.ipynb   ← Colab notebook (self-contained)
│
├── posture/                   ← Task A: all scripts that classify pig posture
│   └── run_inference.py       ← MAIN ENTRY POINT for monthly inference
│
├── vulva/                     ← Task B: vulva segmentation + measurement scripts
│
├── archive/                   ← exploratory / dead scripts (see archive/README.md)
│
├── labels/                    ← label CSVs and binary masks
├── models/                    ← trained model weights (.pth files)
├── outputs/                   ← inference outputs (CSVs, heatmaps, Excel reports)
└── docs/                      ← methodology notes
```

All scripts in `posture/` and `vulva/` are runnable as `python posture/<script>.py` or `python vulva/<script>.py` from the repo root.

---

## Task A — Posture Classification (PRODUCTION READY)

### Run the pipeline on new monthly data

```bash
python posture/run_inference.py /path/to/new/month/of/data
```

That single command does the whole pipeline:
1. Crops images to the stall region
2. Filters out empty stalls (CNN pig-presence detector)
3. Runs the posture classifier (MobileNetV2 on IR, 98.4% test accuracy)
4. Generates the heatmap (`outputs/posture_heatmap.png`)
5. Generates the predictions CSV (`outputs/predictions.csv`)
6. Generates per-pig Excel reports + master workbook with daily statistics

### Required model files

Both must exist in `models/`:
- `posture3_ir_best.pth` (8.7 MB) — the posture classifier (3 classes)
- `pig_detector.pth` — the CNN pig-presence detector

If missing, both can be downloaded from the GitHub release: <https://github.com/AkbarDevop/paal-pipeline/releases/tag/v1.0>

### Data format expected

The input folder must contain timestamp subfolders in OAK-D format:
```
/path/to/data/
├── 20260211-14-30-00/
│   ├── ir_pig0_20260211-14-30-17.jpg
│   ├── ir_pig1_20260211-14-30-18.jpg
│   ├── ...
│   ├── depth_pig0_20260211-14-30-17.raw
│   └── rgb_pig0_20260211-14-30-17.jpg
├── 20260211-14-45-00/
└── ...
```

Files per pig: `ir_*.jpg` (used by classifier), `depth_*.raw` (used by detector), `rgb_*.jpg` (optional).

### Camera ID quirks — already handled

The OAK-D camera occasionally generates pig IDs > 19 due to magnet/overflow issues. The pipeline normalizes `pid % 20` automatically. For known specific date ranges where pig 19/20 are duplicates of pig 17/18, see `posture/fix_pig_ids.py` and `posture/fix_predictions_csv.py`. If you encounter new date ranges where IDs are corrupt:
1. Add a new rule to the `CORRECTIONS` list in both `fix_pig_ids.py` (renames files) and `fix_predictions_csv.py` (rewrites predictions CSV)
2. Re-run inference, or run `fix_predictions_csv.py` to patch existing predictions

### Re-generating the heatmap without re-running inference (30 min saved)

```bash
python posture/regenerate_heatmap.py
```

Reads `outputs/predictions.csv` and rebuilds heatmaps. Use this after editing pig IDs or tweaking colors.

### Comparing posture against Lucas's estrus ground truth

```bash
python posture/count_posture_changes.py --estrus path/to/estrus.xlsx
```

Counts nighttime (19:00–08:00) posture transitions per pig per night. Cross-references with Lucas's heat-day labels. Validated biological signal: pig 9 = +12.8 transitions on heat days, pig 14 = +10.1.

### Master Excel report

```bash
python posture/generate_master_report.py
```

Produces one Excel workbook per pig group with: overview sheet, group plots, per-pig daily summary + embedded plots + raw observations (including the per-frame "posture changed since previous frame" column).

### Colab notebook for non-technical users

The notebook `PAAL_Posture_Classification.ipynb` is self-contained — every function it needs is bundled inline, and the model auto-downloads from the GitHub release. Anyone can:

1. Open <https://colab.research.google.com/github/AkbarDevop/paal-pipeline/blob/main/PAAL_Posture_Classification.ipynb>
2. Paste their Google Drive data path into the text field
3. Click play on each cell

All code cells are auto-collapsed (Colab form mode) so non-CS users see only titles + the path input.

### Re-training the posture model

Only needed if labeled data changes substantially.

```bash
python posture/train.py --modality ir --class-set posture3
python posture/eval.py
```

### Re-training the pig-presence detector

```bash
python posture/train_pig_detector.py
```

Uses auto-labeled IR frames (depth < 1463 mm = pig present, else empty stall). Balanced across standing / sitting / lying. Outputs `models/pig_detector.pth`.

---

## Task B — Vulva Segmentation (IN PROGRESS)

Status as of handoff: **paper-style surface pipeline operational, detection tuning in progress**.

### Pipeline steps

```bash
# 1. Build 3D point clouds from depth raw files
python vulva/build_depthmaps.py

# 2. Interactive rectangle crop around vulva on IR images
python vulva/build_vulva_rect_crops.py

# 3. Paper-style surface analysis + measurements (Xu et al. 2023)
python vulva/build_paper_vulva_surfaces.py depthmap_rect/SESSION/PIG

# Alternative: manual L/W via IR point-click
python vulva/label_vulva_length_width.py

# Validate against caliper ground truth
python vulva/compare_vulva_ground_truths.py
```

### Current state

- **232 pig 3D point clouds** generated in `depthmap/`
- **48 valid vulva crops** in `depthmap_rect/` (186 cancelled = no visible vulva)
- Paper-style pipeline: plane fit → auto-grid OS → Open3D DBSCAN vulva detection → ellipse fit → 35% dilate → inpaint → measure
- **Known issue:** uniformly-protruding crops (pig15-type cases) still produce oversized ellipses. Threshold escalation strategy (75th → 80/85/90 pct) helps but doesn't fully solve.

### Key design decisions (locked)

- **IR over RGB for rectangle selection.** IR is pixel-aligned with depth (same sensor); RGB has unresolvable calibration baseline.
- **JET colormap** for depth visualization (matches reference paper).
- **Open3D DBSCAN + ellipse fit** for strictly oval vulva masks.
- **No frequency filtering / U-Net.** Reference paper (Xu et al. 2023) uses regionprops → largest round region. The 2024 follow-up adds U-Net — not implemented here.

### Ground truth

`ground_truths/vulva_ground_truths.csv` — caliper measurements from 20 pigs over 11 days.

Sample result: pig0 L=55.86mm W=41.62mm (GT: W=41.0, L=79.3). Width matches well, length under-predicts.

### Reference papers

In `papers/`:
1. Xu et al. (2023) — *Detecting sow vulva size change around estrus* — main reference for the surface pipeline
2. Xu et al. (2024) — *Automated oestrous detection in sows* — robotic imaging system
3. Xu et al. (2024) — *Developing a Sow Vulva Volume Estimation Pipeline* — CV = W × L × H volume estimator

### What's next for vulva work

1. Fix oversized detection for uniformly-protruding crops
2. Characterize OAK-D ToF depth accuracy (paper reports 3.4 ± 3.0 mm on Intel L515)
3. Compute formal daily behavioral indices (SI, SLI, LLI, PCF) for estrus model input
4. Build 1D CNN estrus detection model (DFW + behavior + vulva volume)
5. Multi-day longitudinal vulva tracking across the estrus cycle

---

## Data locations

- **Lab Windows PC:** `C:\PAAL_data\Fed_pig\Pictures_OAK\` and `C:\PAAL_data\Fed_pig\Pictures_RS\`
- **Google Drive:** shared folder with sample data
- **OneDrive:** processed outputs uploaded as `PAAL Pipeline Outputs (<date range>)`
- **Shared lab drive `E:\Pig_data\`** structure: `Pictures_OAK/` + `Pictures_RS/`, each containing `YYYY-MM-DD_MM-DD/` collection-round folders with the timestamp subfolders inside

---

## Known issues / gotchas

| Issue | Workaround |
|---|---|
| Camera magnet glitch creates pig IDs > 19 | Use `fix_pig_ids.py` / `fix_predictions_csv.py`. Already covers Feb 13, Feb 17, Feb 18, Feb 19, Feb 20, Mar 8, Mar 10–11. |
| Camera downtime Feb 16–17, Feb 20–22 | No frames exist for those dates. Pipeline handles cleanly (empty folders). |
| Pig 18 / pig 19 empty stalls in Fed_Pig 2026 round | CNN pig-detector filters them automatically. |
| `_cropped.jpg` files in legacy data | `posture/run_inference.py` detects and skips double-cropping. |
| `models/` and `outputs/` are gitignored | Required artifacts download from GitHub release; outputs are generated locally. |
| Colab notebook needs Drive mount | Standard `from google.colab import drive` flow. If your data is in "Shared with me", create a shortcut to "My Drive" first. |

---

## Environment

```bash
pip install -r requirements.txt
```

Key deps: `torch`, `torchvision`, `opencv-python`, `numpy`, `pandas`, `openpyxl`, `matplotlib`, `tqdm`, `open3d` (vulva only), `python-pptx` (slides only).

Tested on Python 3.10+, both Mac and Windows lab PC.

---

## Reference

- **GitHub repo:** <https://github.com/AkbarDevop/paal-pipeline>
- **Methodology document:** `docs/METHODOLOGY.md` — maps each pipeline step to the relevant Xu et al. paper section
- **Last working snapshot:** tag `v1.0-handoff`

---

## Quick sanity check

After cloning fresh, verify everything imports:

```bash
python posture/run_inference.py --help
python posture/regenerate_heatmap.py --help
python posture/count_posture_changes.py --help
python vulva/point_cloud.py --help
```

All four should print their usage without errors.
