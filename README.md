# PAAL Posture Pipeline

This repo trains sow posture classifiers from OAK exports (`RGB`, `IR`, `Depth`).

## Pipeline

```text
OAK timestamp folders (data/)
        |
        v
scan_data.py -> labels/metadata.csv
        |
        +--> prefilter_depth.py -> labels/presence_filter.csv (optional no-pig filter)
        |
        v
label_tool.py -> labels CSV
        |
        v
train.py -> models/*.pth + outputs/history_*.json
        |
        v
eval.py -> outputs/eval_summary.json + plots
```

## Active Scripts

- `scan_data.py`: index raw folders into `labels/metadata.csv`
- `prefilter_depth.py`: rule-based presence filter (`pig_present`)
- `inspect_prefilter.py`: visual QA for prefilter output
- `label_tool.py`: manual labeling (`binary`, `posture3`)
- `train.py`: training for one modality at a time
- `eval.py`: held-out test evaluation + plots
- `normalize_posture3_labels.py`: convert legacy posture3 labels to strict 3-class labels
- `depthai_tof_align_official.py`: official DepthAI live demo (device required)
- `check_alignment.py`: offline visual overlay check for exported JPGs

## Quick Start

1. Build metadata

```bash
python3 scan_data.py
```

2. (Optional) Build no-pig filter and inspect

```bash
python3 prefilter_depth.py
python3 inspect_prefilter.py --target present -n 20
```

3. Label data

```bash
python3 label_tool.py --class-set posture3
```

4. Train

```bash
python3 train.py --modality depth --num-classes 3 --labels-csv labels/labels_posture3_clean.csv --model-prefix posture3
```

5. Evaluate + generate plots

```bash
python3 eval.py --modality depth --class-set posture3 --num-classes 3 --labels-csv labels/labels_posture3_clean.csv --model-prefix posture3
```

## Modalities

`rgb`, `ir`, `depth`, `rgb_depth`, `rgb_ir`, `all`

## Outputs

- `outputs/eval_summary.json`
- `outputs/cm_<prefix>_<modality>.png`
- `outputs/per_class_<prefix>_<modality>.png`
- `outputs/curves_<modality>.png`
- `outputs/modality_comparison_test.png` (when evaluating multiple modalities)
