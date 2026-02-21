# PAAL Sow Posture Classification Pipeline

Standing vs not-standing binary classification from OAK RGB/IR/Depth exports.

## Main Scripts
- `scan_data.py` builds `labels/metadata.csv`
- `label_tool.py` creates/updates `labels/labels.csv`
- `prefilter_depth.py` builds depth-based `no_pig` filter CSV
- `train.py` trains one modality model
- `eval.py` evaluates saved models on held-out test pigs
- `depthai_tof_align_official.py` official on-device ToF/IR -> RGB alignment demo
- `align_images.py` deprecated stub (intentionally disabled)

## Quick Start
1. Put timestamp folders under `data/`
2. `python3 scan_data.py`
3. `python3 label_tool.py`
4. `python3 train.py --modality rgb`
5. `python3 eval.py`

## Label Tool Options
- `python3 label_tool.py --pig 0`
- `python3 label_tool.py --show-ir`
- `python3 label_tool.py --show-depth`
- `python3 label_tool.py --class-set posture4`
- `python3 label_tool.py --class-set posture4 --present-only --presence-csv labels/presence_filter.csv`

## Modality Options
- `rgb`
- `ir`
- `depth`
- `rgb_depth`
- `rgb_ir`
- `all`

## Alignment Policy
- Do not use custom offline RGB->ToF warping for reportable results.
- Use official DepthAI alignment nodes for future capture (`depthai_tof_align_official.py`).
- For current offline experiments, train directly from exported JPG modalities.

## Quick Checks
1. Deprecated script is blocked:
   - `python3 align_images.py`
2. Eval run:
   - `python3 eval.py --modality rgb ir depth rgb_depth rgb_ir all`
3. Official device alignment demo (OAK connected):
   - `python3 depthai_tof_align_official.py`

## No-Pig Prefilter (Depth Rule)
- Generate presence file:
  - `python3 prefilter_depth.py`
- Output:
  - `labels/presence_filter.csv` with `pig_present` (1/0)

## 4-Class Posture Workflow
1. Generate depth prefilter:
   - `python3 prefilter_depth.py`
2. Label only pig-present frames into posture4 CSV:
   - `python3 label_tool.py --class-set posture4 --present-only --presence-csv labels/presence_filter.csv`
3. Train posture4 model:
   - `python3 train.py --modality rgb --num-classes 4 --labels-csv labels/labels_posture4.csv --model-prefix posture4 --allowed-labels 0,1,2,3`
4. Evaluate posture4 model:
   - `python3 eval.py --modality rgb --class-set posture4 --num-classes 4 --labels-csv labels/labels_posture4.csv --model-prefix posture4 --allowed-labels 0,1,2,3`
