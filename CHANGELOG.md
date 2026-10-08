# Changelog

## v1.1 — posture module review (October 2026, branch `improve/posture-v1.1`)

Local branch. Not pushed until `posture/benchmark.py` has been run on the lab PC and the numbers confirm the new training recipe.

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
- **Batched inference with parallel image loading.** Default batch size is 64 on GPU/MPS and 4 on CPU. Large batches are slower than single frames for MobileNetV2 on a 2-core CPU.
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
- **`posture/eval_ground_truth.py`.** Scores predictions against Lucas's time-stamped `sow position.xlsx` (Feb 6–10, all 20 pigs every ~10 min). This is the first independent test set: different days and a different labeller from training. It also compares nightly transition counts: truth vs predicted vs smoothed.
- **`posture_matrix.csv` output.** Rows are timestamp folders, columns are pig0…pig19 — the format Fengkai asked for on Mar 15.
- **`--data-root`.** Label CSVs (Mac absolute paths) now resolve on any machine by file name.
- **Tests.** `tests/test_posture_pipeline.py` has 13 checks on synthetic OAK-D folders with no lab data needed. It covers:
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
- **Tests.** There are now 13 checks.

### Not changed / still open

- Which physical stall is missing when a folder has 19 captures. Every later ID shifts. `data_quality.csv` lists these folders, but resolving them still needs image comparison.
- The pig-presence detector:
  - It was trained on depth-threshold auto-labels with a random frame split, so its "100%" validation score is optimistic.
  - It is not in the Mac's `models/` folder. It is on the GitHub release and the lab PC.
- Task B (vulva) code was not touched.
