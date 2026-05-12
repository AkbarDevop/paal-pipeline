# archive/

Scripts that were used during development but are no longer part of the active pipeline. Kept here (instead of deleted) so git history stays clean and future readers can see what was tried.

| File | Why archived |
|---|---|
| `align_images.py`, `check_alignment.py`, `depthai_tof_align_official.py` | Early RGB↔ToF alignment work. Now handled by official DepthAI align pipeline; these were exploration. |
| `depth_profile_viewer.py` | One-off debug viewer for depth histograms. |
| `detect_id_shifts.py` | Stall-depth fingerprinting attempt to detect pig ID shifts. Inconclusive (stalls are equidistant). |
| `detect_missing_pigs.py` | Diagnostic tool used once during data audit. |
| `find_sitting.py` | Uncertainty-sampling helper used during 3-class labeling. Labels are now complete. |
| `infer_random.py` | Exploratory inference on random frames during model selection. |
| `inspect_prefilter.py` | Debug script for the old depth-only prefilter (replaced by CNN pig detector). |
| `label_tool.py` | Used to create `labels/labels.csv` and `labels_posture3.csv`. Labels are complete. |
| `make_slides.py` | Weekly meeting slide generator. Not needed for inference pipeline. |
| `normalize_posture3_labels.py` | One-time label normalization run. |
| `plot_pipeline_summary.py`, `plot_pretrain_comparison.py` | One-off plots for meeting decks. |
| `prefilter_depth.py` | Old depth-only pig presence filter. Replaced by `posture/train_pig_detector.py` + `posture/filter_predictions.py`. |
| `pretrain_sowbot.py` | Sowbot transfer-learning experiment. Abandoned — fine-tuning from sowbot hurt accuracy vs from-scratch. |
| `run_pipeline.py` | Earlier version of the inference entry point. Superseded by `posture/run_inference.py`. |
| `scan_data.py` | Initial dataset discovery script. Used once at project start. |
| `validate_pig_detection.py` | Standalone validation harness. Functionality now covered by training script's eval split. |

To revive any of these, just `git mv archive/<file>.py posture/` and add the path-fix at the top.
