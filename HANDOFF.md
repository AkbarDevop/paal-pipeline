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
1. Scans every capture and cleans camera IDs (known magnet fixes from `posture/id_corrections.py`, overflow IDs 20+ and double captures set aside — no more `% 20`)
2. Crops images to the stall region in memory (raw folders are never written to)
3. Filters out empty stalls (depth prefilter + CNN pig-presence detector)
4. Runs the posture classifier in batches (MobileNetV2 on IR: 98.4% on test pigs 16–19, 97.6% in 5-fold CV by pig, 91.3% against Lucas's independent labels for Feb 6–11, 94.1% at night)
5. Writes `outputs/predictions.csv` (one row per stall per folder, with class probabilities and a `posture_changed` column), `predictions_excluded.csv`, `data_quality.csv` (folders with missing/extra/duplicate IDs) and `coverage_by_pig_day.csv`
6. Generates heatmaps, per-pig Excel reports + master workbook with daily statistics

Useful flags: `--start/--end` (folder date range, e.g. `--start 20260206 --end 20260211`), `--smooth 3` adds `smoothed_prediction_name` (do **not** use it for transition counts: against Lucas's labels it cut nightly transitions from 8.0 to 2.5 per sow), `--model PATH` to try another checkpoint, `--id-policy modulo` to reproduce v1.0 numbers, `--batch-size/--workers` for speed.

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
│   ├── pig0_ir_vis_20260211-14-30-17.jpg
│   ├── pig0_depth_20260211-14-30-17.raw
│   ├── pig0_rgb_20260211-14-30-17.jpg
│   ├── pig1_ir_vis_20260211-14-30-18.jpg
│   └── ...
├── 20260211-14-45-00/
└── ...
```

Files per pig (pattern in `FILENAME_RE`): `pigN_ir_vis_*.jpg` (used by the classifier), `pigN_depth_*.raw` (used by the depth prefilter), `pigN_rgb_*.jpg` (optional). Legacy `*_cropped.jpg` files are recognised and never cropped twice.

### Camera ID quirks — already handled (v1.1)

The OAK-D numbers pigs sequentially as it walks the stalls, so a misfiring stall magnet causes IDs > 19 and double captures. All rules live in one file, `posture/id_corrections.py`, and `run_inference.py` applies them in memory:

- known bad ranges (Feb 13, 17, 18, 19, 20, Mar 8, Mar 10–11) are renamed/dropped per rule
- any other ID ≥ 20 is set aside as `overflow` (v1.0 mapped it with `% 20`, which silently added pig 20's frames to pig 0, pig 21's to pig 1, …)
- a second capture of the same stall in the same folder is set aside as `duplicate`
- everything set aside goes to `predictions_excluded.csv`; folders that need a human look are flagged in `data_quality.csv`

If you find a new bad range: add one rule to `CORRECTIONS` in `posture/id_corrections.py` and re-run inference (or `regenerate_heatmap.py` / `count_posture_changes.py`, which apply the same policy to older CSVs). `fix_pig_ids.py` (renames raw files) is dry-run by default and should only ever be used on a copy of the data. `fix_predictions_csv.py` is only for CSVs made before v1.1.

Still open: a stall that is *skipped* (19 captures) shifts every later ID by one. `data_quality.csv` lists those folders (`missing_stalls`), but deciding which physical stall was skipped still needs image comparison.

### Re-generating the heatmap without re-running inference (30 min saved)

```bash
python posture/regenerate_heatmap.py
```

Reads `outputs/predictions.csv` and rebuilds heatmaps. Use this after editing pig IDs or tweaking colors.

### Scoring against Lucas's time-stamped posture ground truth (independent test)

Lucas's `sow position.xlsx` (his Apr 16 email; Teams → Ground Truth) labels all 20 stalls on every robot pass from Feb 6 to Feb 15 (codes 0 = lying, 1 = sitting, 2 = standing, X = empty stall; 35,816 labels). The OAK frames for those days are on the RDE share (`\\umad.umsystem.edu\rde\zhoujianf-lab\Pig project\Sow Data\Pictures_OAK`, flat timestamp folders) and on the lab PC under `E:\Pig_data\Pictures_OAK\OAK_0206_0211` / `OAK_0211_0312`:

```bash
python posture/run_inference.py "<share>\Pictures_OAK" --start 20260206 --end 20260211 --output-dir outputs/gt_feb06
python posture/eval_ground_truth.py --gt "sow position.xlsx" --pred outputs/gt_feb06/predictions.csv --out outputs/gt_feb06/ground_truth_eval.csv
python posture/review_presence.py --misses outputs/gt_feb06/ground_truth_misses.csv
```

Results on Feb 6–11 are in CHANGELOG.md (v1.1 → Validation). Lucas's row time is the start of the robot pass (one time per row for all 20 stalls); the robot passes every 10.65 min and he logs about two rows per pass, so the script scores the closest row per pass and stall.

`run_inference.py` also writes `posture_matrix.csv` (rows = timestamp folders, columns = pig0…pig19), the format Fengkai asked for to test the model himself.

### Comparing posture against Lucas's estrus ground truth

```bash
python posture/count_posture_changes.py --estrus path/to/estrus.xlsx
```

Counts nighttime (19:00–08:00) posture transitions per pig per night. Cross-references with Lucas's heat-day labels. v1.0 signal: pig 9 = +12.8 transitions on heat days, pig 14 = +10.1 — re-check these with v1.1, because v1.0 counted overflow captures merged into pigs 0–3 as extra transitions. Options: `--use-smoothed`, `--min-confidence 0.7`, `--id-policy modulo` (reproduce v1.0).

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

### Re-training / benchmarking the posture model

Labeled frames: `labels/labels_posture3.csv` (1,372 frames, Jan 23 – Feb 3; standing 656 / sitting 52 / lying 664; 24 of the 52 sitting frames are pig 4). Its image paths are from the Mac, so point `--data-root` at any folder that contains those timestamp folders — files are found by name.

```bash
# Before/after comparison (shipped model, v1.0 recipe vs v1.1 recipe, grouped 5-fold CV by pig)
python posture/benchmark.py --data-root "C:\PAAL_data\Fed_pig\Pictures_OAK"

# Train one model on the fixed split (train 0–11 / val 12–15 / test 16–19)
python posture/train.py --class-set posture3 --modality ir --recipe improved --data-root <folder>
python posture/eval.py --model-prefix posture3 --class-set posture3 --num-classes 3 --data-root <folder>
```

The fixed test split has only ~190 frames and 5 sitting frames, so compare recipes with `--cv 5` (every pig tested once) rather than the single split. New checkpoints record their preprocessing (`normalize`, `crop`) and inference reads it, so v1.0 and v1.1 models both work everywhere, including the Colab notebook (MobileNetV2 checkpoints only).

### Re-training the pig-presence detector

```bash
python posture/train_pig_detector.py
```

Uses auto-labeled IR frames (depth < 1463 mm = pig present, else empty stall). Balanced across standing / sitting / lying. Outputs `models/pig_detector.pth`.

---

## Task B — Vulva Segmentation (IN PROGRESS)

**v2 (October 2026): learned pipeline.** The posture model picks standing frames, a U-Net finds the vulva on the IR image, the matching ToF depth gives width, length, height and volume in mm, and each sow gets one robust value per day. The hand-crop surface tools below are kept for labelling and comparison.

### Why it is built this way (data checked 2026-10-10)

- **The estrus signal is real but small.** Lucas's caliper workbook (Teams > AI Imaging > Ground Truth > `Feb_2026.xlsx`, 20 sows, Feb 6–17) shows the W × L index rising a median 20% from each sow's day 1–2 baseline to her heat peak (more than 10% in 14 of 19 sows that came into heat). Width rose a median 9%, length 10%. Caliper day-to-day noise after heat: SD 2.6 mm (width), 4.5 mm (length).
- **Single camera frames are too noisy.** Srikar's manual masks on Feb 11 (09:06–11:07, 12 passes, 48 measured frames) gave a within-morning SD of 8.7 mm (width) and 7.7 mm (length) for the same sow, and per-frame errors against the Feb 11 calipers of 9.9 mm (width) and 13.9 mm (length). One frame cannot see a 4–6 mm change, so the pipeline takes the median of many frames per sow-day.
- **Most frames show no vulva.** In the 09:06 pass 15 of 20 sows were measurable; after 09:50 it was 0–3 per pass. Feeding time is the window; `--windows` restricts the run to it.

### Pipeline (new)

```bash
# 0. Ground truth: Lucas's caliper + heat workbook -> tidy sow-day table and per-sow heat response
python vulva/truth.py "path\to\Feb_2026.xlsx" --out outputs/vulva/sow_day_truth.csv

# 1. Label index from the manual masks we have (measured -> present, no_vulva -> absent)
python vulva/make_seg_labels.py --manual labels/vulva_measurements_manual_dataset.csv ^
    --data-root "\\umad.umsystem.edu\rde\zhoujianf-lab\Pig project\Sow Data\Pictures_OAK" ^
    --mask-root labels/vulva_masks_manual

# 2. Train the U-Net (5-fold CV grouped by sow, then a final model -> models/vulva_seg.pth)
python vulva/train_seg.py --labels labels/vulva_seg_labels.csv

# 3. Run on posture predictions (standing frames only), compare with calipers
python vulva/run_vulva.py outputs/gt_feb06/predictions.csv --windows 06:00-11:00,14:00-18:00 ^
    --truth outputs/vulva/sow_day_truth.csv --save-masks

# Noise check on any per-frame table (manual or run_vulva output)
python vulva/repeatability.py outputs/vulva/vulva_frames.csv --truth outputs/vulva/sow_day_truth.csv
```

| Script | What it does |
|---|---|
| `vulva/truth.py` | Reads Lucas's per-sow sheets (DAY, DATE, V. W., V. L., V. V., MORNING/EVENING heat = 20). V. V. is W × L / 200. |
| `vulva/make_seg_labels.py` | Manual table + masks → `labels/vulva_seg_labels.csv` (finds IR frames again under `--data-root`). |
| `vulva/seg_model.py` | U-Net with ResNet-18 encoder; stall crop 380 × 450 → 320 × 384; masks mapped back to 640 × 480. |
| `vulva/train_seg.py` | CV by sow: presence precision / recall / F1 + confusion matrix, IoU / Dice on visible frames; picks the mask threshold on held-out sows; saves the final model. |
| `vulva/select_frames.py` | Standing frames from `predictions.csv` (confidence ≥ 0.6, ID problems dropped), optional time windows and cap per sow-day. |
| `vulva/measure3d.py` | Skin plane from a ring around the mask (RANSAC); W and L along the in-plane axes (tilt removed), height above the plane, volume, quality numbers. Checked on synthetic tilted vulvas: within 1 mm and 4% volume up to 60° tilt. |
| `vulva/daily.py` | Good frames → per sow-day median and MAD, % change from the first 2 days, join with calipers and heat days. |
| `vulva/run_vulva.py` | Steps 1–4 in one command; writes `vulva_frames.csv`, `vulva_daily.csv`, `vulva_vs_caliper.csv`, `vulva_review.jpg`, `vulva_summary.json`. |
| `vulva/repeatability.py` | Within-day SD, error vs calipers, frames needed per sow-day. |

Checks: `python tests/test_vulva_pipeline.py` (17 synthetic tests, including a tiny train → run).

### Labelling (the masks from the hand-crop work are not on the lab PC)

```bash
# lab PC: copy standing frames (IR + depth) out of the raw share into a label set
python vulva/export_label_set.py outputs\gt_feb06\predictions.csv \\tsclient\paal_exchange\vulva_label_set --per-day 3

# any computer: 3 clicks per visible vulva (top, bottom, side), n = not visible; resumes where you stopped.
# Shows IR | 3D shading | height (mm above the sow's own skin, from the aligned depth); click on any of them.
python vulva/label_ellipse.py ~/Downloads/paal_exchange/vulva_label_set

# train straight from the label set (paths in labels.csv are relative to its folder)
python vulva/train_seg.py --labels ~/Downloads/paal_exchange/vulva_label_set/labels.csv
# lab PC: --jobs 6 trains the 5 folds and the final model at once on the two GPUs; --copy-to puts model,
# report and per-frame CV file on the Mac share (PowerShell copy gets "access denied" on \\tsclient; Python
# writes work while the Mac is awake). Labels from a local copy of the set, so the Mac is only needed for --copy-to.
python vulva\train_seg.py --labels C:\Users\aakdbx\vulva_label_set\labels_r4.csv --jobs 6 --out-model models\vulva_seg_r6.pth --out-dir outputs\vulva_seg_r6 --copy-to \\tsclient\paal_exchange\vulva_label_set
# same, with depth as extra input channels (height above the skin + 3D shading)
python vulva/train_seg.py --labels ~/Downloads/paal_exchange/vulva_label_set/labels.csv --inputs ir+depth

# colour images too (lab PC; keeps the frames and labels): aligned RGB becomes a 4th clickable panel,
# plain RGB from the separate colour camera is shown whole for reference
python vulva/export_label_set.py outputs\gt_feb06\predictions.csv \\tsclient\paal_exchange\vulva_label_set --add-rgb

# re-check saved labels against the outline rule (whole vulva to its outer edge; t = tail covers part)
python vulva/label_ellipse.py ~/Downloads/paal_exchange/vulva_label_set --review r6
# after a first model: it suggests each ellipse (Enter accept, n agree, click to redo), hardest frames first
python vulva/label_ellipse.py ~/Downloads/paal_exchange/vulva_label_set --model models/vulva_seg_r5.pth --hard-first
```

Status (2026-10-26): all 360 Feb 6–11 frames labelled (234 visible), snapshot `labels_r4.csv`. Current model `models/vulva_seg_r5.pth` (IR only; Mac and lab PC), which carries a confidence cut for "vulva visible". CV by sow at that cut: precision 0.90, recall 0.86, IoU 0.79 overall; on the 161 human-drawn labels (the rest are model outlines accepted as drawn) precision 0.75, recall 0.67, IoU 0.72. Sizes are unbiased. The cut drops large vulvas more often than small ones, so check heat days in the caliper comparison. History: r1 (105 labels) F1 0.71, IoU 0.59; r2 (215) F1 0.75, IoU 0.65, drew about 10% small; r3 (r2's labels, 100 epochs) same as r2, so more labels help and longer training does not; r4 (360) like r5 without the cut, calling a vulva visible on about half the hard frames without one; IR + depth input (r2d) scored lower, so depth stays a labelling aid. Details in CHANGELOG.md.

### Next

1. Blind test done 2026-10-27 (148 Feb 12–15 frames in `vulva_test_feb12`, hand-labelled without suggestions; keep it as the test set and do not train on it): r5 finds 99% of vulvas, but its outlines are about 15% small and it draws tail-covered vulvas. Next: review the 234 visible training labels with the outline rule (whole vulva to its outer edge; tail covering any part = `t`): `python vulva/label_ellipse.py ~/Downloads/paal_exchange/vulva_label_set --review r6`. Then snapshot `labels_r6.csv`, train r6 on the lab PC (`--jobs 6`), and run `eval_seg.py` for r6 and r5 on the blind set.
2. Done 2026-10-26: posture on Feb 12–17 (`outputs\gt_feb12`) and `run_vulva.py --model models\vulva_seg_r5.pth` on Feb 6–17 (`outputs\vulva_r5_feb06_17` on the lab PC, copied to the Mac). First look in CHANGELOG.md: volume peaks on Feb 10, W × L does not rise. Still to do: download `Feb_2026.xlsx`, run `truth.py`, and compare each sow's daily values with her calipers and heat days, including whether her heat days lose more frames to the cut.
3. Add the vulva change to the heat model next to night transitions and days since move (`estrus/`).
4. More labels from other days and other pig groups (new sows matter most), drawn or corrected by hand rather than accepted; the least confident frames of the Feb 6–17 run are the ones to label. Also redraw the two sliver labels noted in CHANGELOG.md.
5. Keep the label set on the lab PC's own disk (`robocopy \\tsclient\paal_exchange\vulva_label_set C:\Users\aakdbx\vulva_label_set /E`) so training does not need the Mac awake.

### Heat detection groundwork (`estrus/`)

```bash
python estrus/sow_days.py outputs/gt_feb06/predictions.csv outputs/vulva/sow_day_truth.csv [--vulva outputs/vulva/vulva_daily.csv]
python estrus/evaluate.py outputs/estrus/sow_day_features.csv
```

One row per sow per day: posture for the night ending that morning (19:00–08:00, the `count_posture_changes.py` convention) and the daytime, days since the sows were moved in, heat from Lucas's MORNING/EVENING checks, and vulva values when `run_vulva.py` has produced them. `evaluate.py` scores each input on its own (AUC) and in leave-one-sow-out logistic regression, always next to days-since-move alone, because heat clusters a few days after weaning and any other input has to beat that.

### Legacy manual tools (v1)


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
- **No frequency filtering / U-Net (v1 only).** Xu et al. (2023) uses regionprops → largest round region. The 2024 follow-up adds a U-Net, which v2 now uses (`vulva/seg_model.py`).

### Ground truth

`ground_truths/vulva_ground_truths.csv` — caliper measurements from 20 pigs over 11 days. The source is Lucas's `Feb_2026.xlsx` (Teams > AI Imaging > Ground Truth); read it with `vulva/truth.py`.

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
- **RDE share:** `\\umad.umsystem.edu\rde\zhoujianf-lab\Pig project\Sow Data\Pictures_OAK` (flat timestamp folders; 5,223 folders in Oct 2026, Jan 23 – Jan 30 and Feb 6 onward; Jan 31 – Feb 3 is not there)
- **Lab PC drives (CAFNR-8M42H04):** `E:\Pig_data\Pictures_OAK\OAK_<MMDD>_<MMDD>\` round folders (Jan 23 – Jun 11), `F:\Pig_barn_data\Pictures_OAK\` (Apr 9 onward). `C:\PAAL_data` does not exist on this PC.
- **Shared lab drive `E:\Pig_data\`** structure: `Pictures_OAK/` + `Pictures_RS/`, each containing `YYYY-MM-DD_MM-DD/` collection-round folders with the timestamp subfolders inside

---

## Known issues / gotchas

| Issue | Workaround |
|---|---|
| Camera magnet glitch creates pig IDs > 19 | Handled in memory by `posture/id_corrections.py` (known rules for Feb 13, 17, 18, 19, 20, Mar 8, Mar 10–11; other overflow/duplicates set aside and reported in `data_quality.csv`). |
| Mac disk nearly full / raw data only on lab PC | Run training, benchmark and full inference on the lab PC; the Mac only has the repo. |
| Camera downtime Feb 16–17, Feb 20–22 | No frames exist for those dates. Pipeline handles cleanly (empty folders). |
| Pig 18 / pig 19 empty stalls in Fed_Pig 2026 round | CNN pig-detector filters them automatically. |
| `_cropped.jpg` files in legacy data | `posture/run_inference.py` detects and skips double-cropping. |
| `models/` and `outputs/` are gitignored | Required artifacts download from GitHub release; outputs are generated locally. |
| Colab notebook needs Drive mount | Standard `from google.colab import drive` flow. If your data is in "Shared with me", create a shortcut to "My Drive" first. |

---

## Environment

```bash
pip install -r requirements.txt
python tests/test_posture_pipeline.py   # 13 quick checks on synthetic data, ~20 s, no lab data needed
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
