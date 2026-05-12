"""Evaluate trained models on held-out test pigs."""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import BATCH_SIZE, BINARY_CLASSES, POSTURE3_CLASSES, LABELS_CSV, MODEL_DIR, OUTPUT_DIR, TEST_PIG_IDS
from data_loader import MODALITY_CHANNELS, SowPostureDataset
from models import BACKBONES, SingleModalModel


def parse_allowed_labels(raw):
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip() != ""]


def evaluate_model(modality, device, args, class_names, allowed_labels, model_path=None):
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, f"{args.model_prefix}_{modality}_best.pth")
    if not os.path.exists(model_path):
        print(f"  No model for '{modality}'")
        return None

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    in_channels = ckpt.get("in_channels", MODALITY_CHANNELS[modality])
    num_classes = ckpt.get("num_classes", args.num_classes)
    backbone = ckpt.get("backbone", "mobilenet_v2")
    model = SingleModalModel(in_channels=in_channels, num_classes=num_classes, pretrained=False, backbone=backbone).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_set = SowPostureDataset(
        modality=modality, labels_csv=args.labels_csv, pig_ids=TEST_PIG_IDS,
        allowed_labels=allowed_labels,
        use_cropped=args.use_cropped,
        silent=True,
    )
    if len(test_set) == 0:
        print(f"  No test samples for '{modality}'")
        return None

    loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.extend(model(x.to(device)).argmax(1).cpu().numpy())
            labels.extend(y.numpy())

    preds, labels = np.array(preds), np.array(labels)
    label_ids = sorted(class_names.keys())
    report_dict = classification_report(
        labels, preds, labels=label_ids,
        target_names=[class_names[i] for i in label_ids], zero_division=0, output_dict=True,
    )
    report_text = classification_report(
        labels, preds, labels=label_ids,
        target_names=[class_names[i] for i in label_ids], zero_division=0,
    )
    return {
        "modality": modality,
        "accuracy": accuracy_score(labels, preds),
        "n_test": len(test_set),
        "best_epoch": ckpt.get("epoch", "?"),
        "test_pigs": TEST_PIG_IDS,
        "confusion_matrix": confusion_matrix(labels, preds, labels=label_ids),
        "report_text": report_text,
        "report_dict": report_dict,
    }


def plot_training_curves(modality, model_prefix):
    if not HAS_PLT:
        return
    hist_path = os.path.join(OUTPUT_DIR, f"history_{model_prefix}_{modality}.json")
    if not os.path.exists(hist_path):
        return

    with open(hist_path) as f:
        h = json.load(f)
    epochs = range(1, len(h["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, h["train_loss"], label="Train")
    ax1.plot(epochs, h["val_loss"], label="Val")
    ax1.set_title(f"{modality} loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, [x * 100 for x in h["train_acc"]], label="Train")
    ax2.plot(epochs, [x * 100 for x in h["val_acc"]], label="Val")
    if h.get("test_acc") is not None:
        ax2.axhline(h["test_acc"] * 100, linestyle="--", label=f"Test {h['test_acc']*100:.1f}%")
    ax2.set_title(f"{modality} accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"curves_{modality}.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(cm, class_names, modality, model_prefix):
    if not HAS_PLT:
        return
    label_ids = sorted(class_names.keys())
    names = [class_names[i] for i in label_ids]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(names)), yticks=range(len(names)),
        xticklabels=names, yticklabels=names,
        ylabel="True label", xlabel="Predicted label",
        title=f"{modality} confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"cm_{model_prefix}_{modality}.png"), dpi=150)
    plt.close()


def plot_per_class_accuracy(report_dict, class_names, modality, model_prefix):
    if not HAS_PLT:
        return
    label_ids = sorted(class_names.keys())
    names = [class_names[i] for i in label_ids]
    precisions = [float(report_dict.get(name, {}).get("precision", 0.0)) for name in names]
    recalls = [float(report_dict.get(name, {}).get("recall", 0.0)) for name in names]
    f1s = [float(report_dict.get(name, {}).get("f1-score", 0.0)) for name in names]

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, [p * 100 for p in precisions], width, label="Precision")
    ax.bar(x, [r * 100 for r in recalls], width, label="Recall")
    ax.bar(x + width, [f * 100 for f in f1s], width, label="F1-score")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)")
    ax.set_title(f"{modality} per-class metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"per_class_{model_prefix}_{modality}.png"), dpi=150)
    plt.close()


def plot_comparison(results):
    if not HAS_PLT:
        return
    names = [r["modality"] for r in results]
    vals = [r["accuracy"] * 100 for r in results]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, vals)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"Test comparison (pigs {TEST_PIG_IDS})")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{v:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "modality_comparison_test.png"), dpi=150)
    plt.close()


def main(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Test pigs: {TEST_PIG_IDS}\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    class_map = {"binary": BINARY_CLASSES, "posture3": POSTURE3_CLASSES}
    base_class_names = class_map[args.class_set]
    allowed_labels = parse_allowed_labels(args.allowed_labels)
    if allowed_labels:
        ids = sorted(set(allowed_labels))
        class_names = {i: base_class_names.get(lbl, str(lbl)) for i, lbl in enumerate(ids)}
    else:
        class_names = base_class_names

    # Discover model files: {prefix}_{backbone}_{modality}_best.pth  or  {prefix}_{modality}_best.pth (legacy)
    model_files = sorted(
        f for f in os.listdir(MODEL_DIR)
        if f.startswith(f"{args.model_prefix}_") and f.endswith("_best.pth")
    )

    # Parse each file into (backbone, modality, path)
    eval_targets = []
    for f in model_files:
        stem = f.replace(f"{args.model_prefix}_", "", 1).replace("_best.pth", "")
        # Try backbone_modality pattern first
        matched = False
        for bb in BACKBONES:
            if stem.startswith(bb + "_"):
                mod = stem[len(bb) + 1:]
                if mod in MODALITY_CHANNELS:
                    eval_targets.append((bb, mod, os.path.join(MODEL_DIR, f)))
                    matched = True
                    break
        if not matched and stem in MODALITY_CHANNELS:
            eval_targets.append(("mobilenet_v2", stem, os.path.join(MODEL_DIR, f)))

    # Filter by --backbone and --modality if specified
    if args.backbone:
        eval_targets = [t for t in eval_targets if t[0] == args.backbone]
    if args.modality:
        eval_targets = [t for t in eval_targets if t[1] in args.modality]

    if not eval_targets:
        print("No trained models found.")
        return

    print(f"Evaluating {len(eval_targets)} model(s):\n")
    results = []
    for bb, m, mpath in eval_targets:
        label = f"{bb}/{m}" if len(set(t[0] for t in eval_targets)) > 1 else m
        print(f"== {label.upper()} ==")
        r = evaluate_model(m, device, args, class_names, allowed_labels, model_path=mpath)
        if not r:
            print()
            continue
        r["backbone"] = bb
        r["label"] = label
        results.append(r)
        print(f"Test accuracy: {r['accuracy']:.4f} ({r['n_test']} frames, best epoch {r['best_epoch']})")
        print(f"Confusion matrix:\n{r['confusion_matrix']}")
        print(f"\n{r['report_text']}")
        tag = f"{bb}_{m}" if bb != "mobilenet_v2" else m
        plot_training_curves(m, f"{args.model_prefix}_{bb}" if bb != "mobilenet_v2" else args.model_prefix)
        plot_confusion_matrix(r["confusion_matrix"], class_names, tag, args.model_prefix)
        plot_per_class_accuracy(r["report_dict"], class_names, tag, args.model_prefix)
        print()

    if len(results) > 1:
        print("== COMPARISON ==")
        print(f"{'Model':<30} {'Test Acc':>10} {'N_test':>8}")
        print("-" * 50)
        for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
            print(f"{r['label']:<30} {r['accuracy']*100:>9.1f}% {r['n_test']:>7}")
        plot_comparison(results)

    summary = [
        {
            "modality": r["modality"],
            "test_accuracy": r["accuracy"],
            "n_test_frames": r["n_test"],
            "test_pigs": r["test_pigs"],
            "best_epoch": r["best_epoch"],
            "confusion_matrix": r["confusion_matrix"].tolist(),
        }
        for r in results
    ]
    out = os.path.join(OUTPUT_DIR, "eval_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--modality", nargs="+", default=None)
    parser.add_argument("--class-set", choices=["binary", "posture3"], default="binary")
    parser.add_argument("--num-classes", type=int, default=2, choices=[2, 3])
    parser.add_argument("--labels-csv", default=LABELS_CSV)
    parser.add_argument("--model-prefix", default="standing")
    parser.add_argument("--allowed-labels", default=None)
    parser.add_argument("--use-cropped", action="store_true", help="Use cropped images")
    parser.add_argument("--backbone", default=None, choices=BACKBONES, help="Filter by backbone")
    main(parser.parse_args())
