"""Evaluate trained models on held-out test pigs."""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    BINARY_CLASSES,
    POSTURE4_CLASSES,
    LABELS_CSV,
    MODEL_DIR,
    OUTPUT_DIR,
    TEST_PIG_IDS,
)
from data_loader import MODALITY_CHANNELS, SowPostureDataset
from models import SingleModalModel


def evaluate_model(modality, device, args, class_names):
    model_path = os.path.join(MODEL_DIR, f"{args.model_prefix}_{modality}_best.pth")
    if not os.path.exists(model_path):
        print(f"  No model for '{modality}'")
        return None

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    in_channels = ckpt.get("in_channels", MODALITY_CHANNELS[modality])
    num_classes = ckpt.get("num_classes", args.num_classes)
    model = SingleModalModel(in_channels=in_channels, num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_set = SowPostureDataset(
        modality=modality,
        labels_csv=args.labels_csv,
        pig_ids=TEST_PIG_IDS,
        allowed_labels=None if not args.allowed_labels else [int(x) for x in args.allowed_labels.split(",")],
        silent=True,
    )
    if len(test_set) == 0:
        print(f"  No test samples for '{modality}'")
        return None

    loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            yhat = model(x.to(device)).argmax(1).cpu().numpy()
            preds.extend(yhat)
            labels.extend(y.numpy())

    preds = np.array(preds)
    labels = np.array(labels)
    acc = accuracy_score(labels, preds)
    label_ids = sorted(class_names.keys())
    cm = confusion_matrix(labels, preds, labels=label_ids)
    report = classification_report(
        labels,
        preds,
        labels=label_ids,
        target_names=[class_names[i] for i in label_ids],
        zero_division=0,
    )

    return {
        "modality": modality,
        "accuracy": acc,
        "n_test": len(test_set),
        "best_epoch": ckpt.get("epoch", "?"),
        "test_pigs": TEST_PIG_IDS,
        "confusion_matrix": cm,
        "report": report,
    }


def plot_training_curves(modality, model_prefix):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
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


def plot_comparison(results):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
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
    class_names = BINARY_CLASSES if args.class_set == "binary" else POSTURE4_CLASSES

    if args.modality:
        modalities = args.modality
    else:
        modalities = sorted(
            x.replace("standing_", "").replace("_best.pth", "")
            for x in os.listdir(MODEL_DIR)
            if x.startswith("standing_") and x.endswith("_best.pth")
        )
        modalities = [m for m in modalities if m in MODALITY_CHANNELS]

    if not modalities:
        print("No trained models found.")
        return

    print(f"Evaluating: {modalities}\n")
    results = []
    for m in modalities:
        print(f"== {m.upper()} ==")
        r = evaluate_model(m, device, args, class_names)
        if not r:
            print()
            continue
        results.append(r)
        print(f"Test accuracy: {r['accuracy']:.4f} ({r['n_test']} frames, best epoch {r['best_epoch']})")
        print(f"Confusion matrix:\n{r['confusion_matrix']}")
        print(f"\n{r['report']}")
        plot_training_curves(m, args.model_prefix)
        print()

    if len(results) > 1:
        print("== COMPARISON ==")
        print(f"{'Modality':<15} {'Test Acc':>10} {'N_test':>8}")
        print("-" * 35)
        for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
            print(f"{r['modality']:<15} {r['accuracy']*100:>9.1f}% {r['n_test']:>7}")
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
    parser.add_argument("--class-set", choices=["binary", "posture4"], default="binary")
    parser.add_argument("--num-classes", type=int, default=2, choices=[2, 4])
    parser.add_argument("--labels-csv", default=LABELS_CSV)
    parser.add_argument("--model-prefix", default="standing")
    parser.add_argument("--allowed-labels", default=None,
                        help="Comma-separated label ids to include, e.g. 0,1,2,3")
    main(parser.parse_args())
