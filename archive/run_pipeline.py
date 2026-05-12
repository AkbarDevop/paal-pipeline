#!/usr/bin/env python3
"""
PAAL Sow Posture Classification — Full Pipeline
=================================================

Run each step in order. Each step can also be run independently.

Usage:
    python run_pipeline.py              # Interactive — choose which steps to run
    python run_pipeline.py --all        # Run all steps (except labeling, which is manual)
    python run_pipeline.py --step 4     # Run a specific step

Pipeline steps:
    1. Scan data       — Build metadata.csv from raw OAK-D folders
    2. Prefilter       — Mark pig-present vs empty frames using depth
    3. Label           — Manual labeling GUI (interactive, skipped in --all)
    4. Crop            — Crop images to remove neighbor pigs / ceiling
    5. Train           — Train 9 models (3 backbones × 3 modalities)
    6. Evaluate        — Evaluate all models on held-out test pigs
    7. Visualize       — Generate summary plots for presentation

Requires: data/ folder with OAK-D recordings, labels/ with labeled CSVs
"""

import argparse
import itertools
import os
import subprocess
import sys

# ── Configuration ────────────────────────────────────────────────────────────

BACKBONES  = ["mobilenet_v2", "xception", "densenet121"]
MODALITIES = ["rgb", "ir", "depth"]

TRAIN_ARGS = dict(
    num_classes=3,
    labels_csv="labels/labels_posture3.csv",
    model_prefix="posture3",
    epochs=30,
    use_cropped=True,
    class_weights=True,
)

EVAL_ARGS = dict(
    model_prefix="posture3",
    class_set="posture3",
    num_classes=3,
    labels_csv="labels/labels_posture3.csv",
    use_cropped=True,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def run(cmd, desc=None):
    """Run a command and stream output."""
    if desc:
        print(f"\n{'='*60}")
        print(f"  {desc}")
        print(f"{'='*60}\n")
    result = subprocess.run(cmd, shell=isinstance(cmd, str))
    if result.returncode != 0:
        print(f"\n  ✗ Failed (exit code {result.returncode})")
        return False
    return True


def ask_continue(step_name):
    """Ask user whether to run a step."""
    reply = input(f"\n→ Run step: {step_name}? [Y/n] ").strip().lower()
    return reply in ("", "y", "yes")


# ── Pipeline Steps ───────────────────────────────────────────────────────────

def step1_scan():
    """Step 1: Scan data folders → metadata.csv"""
    return run(
        [sys.executable, "scan_data.py"],
        "Step 1/7 — Scanning data folders → metadata.csv"
    )


def step2_prefilter():
    """Step 2: Depth-based pig presence filter"""
    return run(
        [sys.executable, "prefilter_depth.py"],
        "Step 2/7 — Prefiltering empty frames (depth-based)"
    )


def step3_label():
    """Step 3: Manual labeling (interactive GUI)"""
    print(f"\n{'='*60}")
    print("  Step 3/7 — Manual Labeling (interactive)")
    print(f"{'='*60}")
    print()
    print("  This opens a GUI for manual labeling.")
    print("  Keys: 0=standing, 1=sitting, 2=lying, s=skip, q=quit")
    print()
    reply = input("  Open labeling tool? [y/N] ").strip().lower()
    if reply in ("y", "yes"):
        return run([sys.executable, "label_tool.py", "--class-set", "posture3"])
    print("  Skipped (using existing labels)")
    return True


def step4_crop():
    """Step 4: Crop images to isolate target pig"""
    return run(
        [sys.executable, "crop_images.py"],
        "Step 4/7 — Cropping images (removing neighbor pigs / ceiling)"
    )


def step5_train():
    """Step 5: Train all 9 models"""
    print(f"\n{'='*60}")
    print("  Step 5/7 — Training 9 models (3 backbones × 3 modalities)")
    print(f"{'='*60}\n")

    total = len(BACKBONES) * len(MODALITIES)
    for i, (bb, mod) in enumerate(itertools.product(BACKBONES, MODALITIES), 1):
        print(f"\n--- [{i}/{total}] {bb} / {mod} ---\n")
        cmd = [
            sys.executable, "train.py",
            "--modality", mod,
            "--backbone", bb,
            "--num-classes", str(TRAIN_ARGS["num_classes"]),
            "--labels-csv", TRAIN_ARGS["labels_csv"],
            "--model-prefix", TRAIN_ARGS["model_prefix"],
            "--epochs", str(TRAIN_ARGS["epochs"]),
        ]
        if TRAIN_ARGS["use_cropped"]:
            cmd.append("--use-cropped")
        if TRAIN_ARGS["class_weights"]:
            cmd.append("--class-weights")
        ok = run(cmd)
        if not ok:
            print(f"  Training failed for {bb}/{mod}, continuing...")
    return True


def step6_evaluate():
    """Step 6: Evaluate all trained models"""
    cmd = [
        sys.executable, "eval.py",
        "--model-prefix", EVAL_ARGS["model_prefix"],
        "--class-set", EVAL_ARGS["class_set"],
        "--num-classes", str(EVAL_ARGS["num_classes"]),
        "--labels-csv", EVAL_ARGS["labels_csv"],
    ]
    if EVAL_ARGS["use_cropped"]:
        cmd.append("--use-cropped")
    return run(cmd, "Step 6/7 — Evaluating all models on test set")


def step7_visualize():
    """Step 7: Generate presentation figures"""
    ok = True
    if os.path.exists("plot_pipeline_summary.py"):
        ok = run(
            [sys.executable, "plot_pipeline_summary.py"],
            "Step 7/7 — Generating summary figures"
        )
    else:
        print("\n  plot_pipeline_summary.py not found, skipping visualization")
    return ok


# ── Main ─────────────────────────────────────────────────────────────────────

STEPS = [
    ("1. Scan data",    step1_scan),
    ("2. Prefilter",    step2_prefilter),
    ("3. Label (GUI)",  step3_label),
    ("4. Crop images",  step4_crop),
    ("5. Train models", step5_train),
    ("6. Evaluate",     step6_evaluate),
    ("7. Visualize",    step7_visualize),
]


def main():
    parser = argparse.ArgumentParser(
        description="PAAL Sow Posture Pipeline — run all steps in order",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--all", action="store_true",
                        help="Run all steps non-interactively (skips labeling GUI)")
    parser.add_argument("--step", type=int, choices=range(1, 8),
                        help="Run a single step (1-7)")
    parser.add_argument("--from-step", type=int, choices=range(1, 8), default=1,
                        help="Start from this step (default: 1)")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   PAAL Sow Posture Classification Pipeline              ║")
    print("║   3 backbones × 3 modalities → 9 models                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print("  Steps:")
    for name, _ in STEPS:
        print(f"    {name}")
    print()

    if args.step:
        # Run single step
        name, func = STEPS[args.step - 1]
        print(f"Running: {name}")
        func()
        return

    # Run steps sequentially
    for i, (name, func) in enumerate(STEPS, 1):
        if i < args.from_step:
            continue

        if args.all:
            if i == 3:  # Skip labeling GUI in --all mode
                print(f"\n  Skipping: {name} (use existing labels in --all mode)")
                continue
            func()
        else:
            if ask_continue(name):
                func()
            else:
                print(f"  Skipped: {name}")

    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    print()
    print("  Results:")
    print("    models/    — trained model checkpoints (.pth)")
    print("    outputs/   — evaluation plots, metrics, history")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
