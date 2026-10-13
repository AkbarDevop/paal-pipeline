# Changelog

## Vulva v2 — learned pipeline (October 2026, branch `vulva/v2-learned-pipeline`)

New scripts in `vulva/`, all reading raw folders only and writing under `outputs/`:

- `truth.py` reads Lucas's caliper + heat workbook (`Feb_2026.xlsx`) into a sow-day table and prints each sow's vulva rise during heat.
- `make_seg_labels.py`, `seg_model.py`, `train_seg.py`: U-Net (ResNet-18 encoder) trained on the existing manual masks, cross-validated by sow; reports presence precision / recall / F1 with the confusion matrix, and IoU / Dice. The mask threshold is tuned on the held-out sows (over-sized masks inflate width and length) and stored in the checkpoint.
- `select_frames.py`: standing frames from the posture predictions, optional feeding-time windows.
- `measure3d.py`: width, length, height and volume in mm from the ToF depth, measured on the fitted skin plane so camera tilt does not shorten the length.
- `daily.py`, `run_vulva.py`: one command from posture predictions to one value per sow per day, compared with the calipers and heat days.
- `repeatability.py`: within-day noise of any per-frame table and how many frames a daily value needs.
- `tests/test_vulva_pipeline.py`: 7 synthetic checks.
- `estrus/sow_days.py`, `estrus/evaluate.py`: the sow-day table for the heat model (night transitions, standing share, days since move, heat, vulva when available, each as change from the sow's own first days) and a leave-one-sow-out check of which inputs pick out heat days, with days-since-move alone as the reference. `tests/test_estrus_baseline.py`: 4 checks.

### Findings that shaped it (2026-10-10)

- Calipers, 20 sows, Feb 6–17: the W × L index rose a median 20% from baseline to the heat peak (> 10% in 14 of 19 sows in heat); width +9%, length +10%. Caliper day-to-day SD after heat: 2.6 mm width, 4.5 mm length.
- Manual camera masks, Feb 11 morning (48 frames): same-sow SD within two hours 8.7 mm width, 7.7 mm length; per-frame error against calipers 9.9 mm width, 13.9 mm length. The change to detect is 4–6 mm, so daily medians of many frames are required.
- Only the 09:06 and 09:17 passes showed most vulvas (15 and 11 of 20 sows); later passes showed 0–5.

## v1.1 — posture module review (October 2026)

Merged into `main` on 2026-10-10 after validation on the lab PC. The shipped model (`posture3_ir_best.pth`, v1.0 release) is unchanged; v1.1 changes the pipeline around it.

### Validation (lab PC, 2026-10-10)

- **Grouped 5-fold CV by pig** (`posture/benchmark.py`, 748 of the 1,372 labelled frames; Jan 31 – Feb 3 frames are not on the RDE share or the lab PC drives):

  | Recipe | Accuracy | Macro-F1 | Balanced acc. | Sitting recall | Lying → sitting errors |
  |---|---|---|---|---|---|
  | baseline (v1.0) | 97.6% | 89.5% | 86.9% | 20/32 | 3 |
  | improved | 97.5% | 91.4% | 95.3% | 29/32 | 13 |

  The improved recipe is not adopted: it catches sitting but labels more lying sows as sitting.
- **Independent test against Lucas's `sow position.xlsx`** (`posture/eval_ground_truth.py`, Feb 6–11, 12,745 pass × stall comparisons, shipped model):
  - accuracy 91.3%, macro-F1 77.3%; night (19:00–08:00) 94.1%, day 86.9%
  - recall / precision: standing 84.7 / 81.5%, sitting 50.5 / 56.9%, lying 95.2 / 95.5%
  - nightly posture transitions per sow: truth 8.0, predicted 7.3, r = 0.84 (100 sow-nights). With `--smooth 3` only 2.5 (r = 0.68), so **leave smoothing off** for transition counts.
  - presence: 99.9% of captured frames with a sow reached the posture model. The 5 depth-gate misses were all at 1,464–1,467 mm, just past the 1,463 mm cutoff.
- Inference on the lab PC: 14,825 stall frames for Feb 6–11 in 20 min reading from the RDE share (2× RTX A6000).

### Bugs fixed

- **Pig IDs ≥ 20 were folded into pigs 0–3.**
  - What happened: `run_inference.py` and the Colab notebook mapped every leftover camera ID with `pid % 20`. A pig-20 overflow capture became a second "pig 0" frame in the same folder.
  - Effect: transitions were inflated for low IDs, and pigs "had data" on days the camera skipped them. This matches Fengkai's Apr 23 review comments (duplicate sow tags, implausible all-day postures).
  - Fix: `posture/id_corrections.py` applies the known rules. It then sets aside overflow IDs and double captures, lists them in `predictions_excluded.csv`, and flags their folders in `data_quality.csv`.
- **The Colab notebook never used the pig-presence detector.** It loaded `pig_detector.pth` but never called it. Colab outputs therefore still contained empty-stall frames that passed the depth prefilter. Lucas uses the Colab outputs.
- **The two correction scripts had drifted apart.** `fix_pig_ids.py` had 2 rules and `fix_predictions_csv.py` had 8, with different end times for Feb 13. There is now one list in `id_corrections.py`.
- **`fix_pig_ids.py` could lose the real pig 19.** It renamed `pig20→pig19` in `os.listdir` order, so the rename could hit the pig-19 duplicate before that duplicate was deleted. It now plans every change first, deletes before renaming, and runs as a dry run unless `--apply` is given. The lab rule is to never modify raw folders.
- **Running `fix_predictions_csv.py` twice deleted real pig-19 rows.** It now refuses to run on v1.1 CSVs, which are already corrected.
- **`count_posture_changes.py` dropped the evening half of the first night.** It filtered the Day-0 window on calendar date but grouped counts by night date. It now filters on night date.
- **The documented retrain command failed.** `train.py --class-set posture3` did not exist. It does now.
- **The documented data format was wrong.** HANDOFF.md listed file names as `ir_pig0_…`. The real pattern is `pig0_ir_vis_…`.

### Improvements

- **Shared preprocessing (`posture/preprocess.py`).** Training, inference, eval and the notebook now use the same crop, resize and normalization.
  - Checkpoints record `normalize` and `crop`.
  - v1.0 checkpoints load unchanged and give identical predictions (checked: 761/761 frames, max confidence difference 0.0).
- **Batched inference.** Default batch size is 64 on GPU/MPS and 4 on CPU. Measured on the Mac (MPS, 1,141 synthetic frames): prediction step 7 s vs 15 s for v1.0, whole run 10.5 s vs 19.6 s, identical predictions. Loader worker processes are off by default on macOS/Windows (they made the same run 3x slower); `--workers 4` is opt-in.
- **New outputs.**
  - Per-class probabilities and the pig-presence probability for each frame.
  - A `posture_changed` column.
  - `coverage_by_pig_day.csv`, which shows frames per stall per day.
- **Optional temporal smoothing (`--smooth 3`).** A majority vote across neighbouring frames that removes one-frame flips. Frames more than 30 minutes apart are never voted together.
  - Off by default until the benchmark shows it helps.
  - `count_posture_changes.py --use-smoothed` counts on the smoothed column.
- **Training (`--recipe improved`).** The `baseline` recipe still reproduces v1.0. The improved recipe adds:
  - ImageNet normalization (the backbones are ImageNet-pretrained)
  - flip, small affine and brightness/contrast augmentation
  - AdamW with a cosine schedule
  - label smoothing and square-root class weights
  - model selection on validation macro-F1, with early stopping
  - seeds, and images cached in RAM, so they are decoded once rather than every epoch
- **Honest evaluation (`--cv K`).** K-fold cross-validation grouped by pig. Metrics are pooled and every pig is tested once. The fixed test split has only 190 frames and 5 sitting frames.
- **`posture/benchmark.py`.** One command for the before/after report, written to `outputs/benchmark_report.md`. It compares:
  - the shipped model on its test pigs
  - the v1.0 recipe against the v1.1 recipe under the same folds
  - transitions in the labels against predicted transitions, raw and smoothed
- **`posture/eval_ground_truth.py`.** Scores predictions against Lucas's time-stamped `sow position.xlsx` (Feb 6–10, all 20 pigs every ~10 min). This is the first independent test set: different days and a different labeller from training. It also compares nightly transition counts: truth vs predicted vs smoothed. Reads Lucas's real layout: one row per robot pass (~5.5 min) with the time typed on every other row (rows between are interpolated), note rows and 'X' (empty stall) cells skipped, his per-sow totals detected as the end of each block, 'completo' / 'alternado' sheets merged and 'Cópia' copies skipped. Checked against his own per-sow totals: exact on every sheet where his formulas cover the whole block. The workbook has 35,816 labelled sow-times, Feb 6–15. Default matching tolerance is now 3 min. Rows are matched to the start of the robot pass (`--match pass`, the default): Lucas writes one time per row for all 20 stalls, while each stall's capture time runs up to ~3 min into the pass, and he logs about two rows per 10.65-min pass, so only the closest row per pass and stall is scored. It also reports night vs day, per-class precision/recall, and nightly transitions per sow (correlation and MAE against his counts).
- **`posture_matrix.csv` output.** Rows are timestamp folders, columns are pig0…pig19 — the format Fengkai asked for on Mar 15.
- **`--data-root`.** Label CSVs (Mac absolute paths) now resolve on any machine by file name. Lookup goes by timestamp folder (directory listing only, also one level down for round folders), so the RDE network share is never walked file by file. `posture/check_label_paths.py` reports which labelled frames are found, missing from an existing folder, or in a missing folder, by day and pig.
- **Tests.** `tests/test_posture_pipeline.py` has 16 checks on synthetic OAK-D folders with no lab data needed. It covers:
  - the ID policy
  - smoothing
  - the end-to-end CLI
  - the notebook staying in sync with the repo
  - transition counting on legacy CSVs

### Hardening after an independent review of v1.1

- **Correction rules run only once per window.** A rule is applied only if its time window still contains IDs that need fixing. CSVs and folders that v1.0's fix scripts already patched keep their real pig 19.
- **Duplicates are resolved among usable captures.** Duplicate captures are only compared once a frame has a prediction. A corrupt or empty first capture no longer hides a good second one, so the CLI and Colab now agree.
- **Smoothed columns only exist after smoothing.** They are written only when `--smooth` is used. `count_posture_changes --use-smoothed` stops if the column is empty, and `eval_ground_truth` skips the smoothed line.
- **Crop setting is honoured.** Inference respects each checkpoint's `crop` flag, both in the CLI and in Colab. The pig detector always gets the stall crop.
- **Fused-modality augmentation is aligned.** Flip and shift/rotate are applied once to all channels, so RGB and IR blocks stay aligned. Colour jitter is still applied per block.
- **`eval.py --class-set posture3` now works.** It switches to `labels_posture3.csv`, so the documented command runs.
- **Safe cross-validation fold count.** `--cv` caps the number of folds at the number of pigs and refuses empty splits.
- **Empty-result guard.** When every frame is set aside, the CLI and notebook print a clear message instead of crashing.
- **Smaller changes:**
  - `train.py` now defaults to `--class-set posture3`.
  - `infer_batch.py` has a `--no-crop` option.
  - Sheet names like `02_06` now parse.
- **Tests.** There are now 16 checks.
- **Presence audit.** The fixed depth gate (median stall depth under 1,463 mm) separates lying sows from an empty floor by only a few centimetres: in the labelled set, lying frames have a median of 1,318 mm, the deepest is 1,461 mm, and the nearest frame called empty is 1,466 mm. Because frames were labelled only after passing the gate, any lying sows it rejected are invisible in the labels. To measure this:
  - `run_inference.py` writes `skipped_frames.csv`: every frame that never reached the posture model, with the reason (depth, detector, corrupt), median depth and detector probability.
  - `eval_ground_truth.py` adds a presence check. Every cell in Lucas's sheet has a sow, so a cell with no prediction is a miss. Each miss is explained (depth gate, detector, unreadable, ID policy, no capture) and broken down by the recorded posture, then written to `ground_truth_misses.csv`.
  - `posture/review_presence.py` makes contact sheets of those misses, or of frames the gate called empty near the threshold, plus a `review.csv` to mark has_sow by eye.
- **`--start` / `--end` date range in `run_inference.py`.** Ported from the lab PC's local edit (used for the Mar 17–27 run). Bounds are inclusive timestamp-folder names; a date alone, like `--end 20260327`, includes that whole day.

### Not changed / still open

- Which physical stall is missing when a folder has 19 captures. Every later ID shifts. `data_quality.csv` lists these folders, but resolving them still needs image comparison.
- The pig-presence detector:
  - It was trained on depth-threshold auto-labels with a random frame split, so its "100%" validation score is optimistic.
  - It is not in the Mac's `models/` folder. It is on the GitHub release and the lab PC.
- Task B (vulva) code was not touched.
