"""Generate visual comparison: sowbot pre-trained+fine-tuned vs from-scratch models."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import MODEL_DIR, OUTPUT_DIR

BACKBONES = ["mobilenet_v2", "xception", "densenet121"]
MODALITIES = ["rgb", "ir", "depth"]
BB_DISPLAY = {"mobilenet_v2": "MobileNetV2", "xception": "Xception", "densenet121": "DenseNet121"}
MOD_DISPLAY = {"rgb": "RGB", "ir": "IR", "depth": "Depth"}

# Colors
C_SCRATCH = "#2563eb"   # blue
C_FINETUNE = "#dc2626"  # red
C_SITTING_S = "#60a5fa" # light blue
C_SITTING_F = "#f87171" # light red


def load_history(prefix, backbone, modality):
    if backbone == "mobilenet_v2":
        path = os.path.join(OUTPUT_DIR, f"history_{prefix}_{modality}.json")
    else:
        path = os.path.join(OUTPUT_DIR, f"history_{prefix}_{backbone}_{modality}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_checkpoint_info(prefix, backbone, modality):
    if backbone == "mobilenet_v2" and prefix == "posture3":
        path = os.path.join(MODEL_DIR, f"{prefix}_{modality}_best.pth")
    else:
        path = os.path.join(MODEL_DIR, f"{prefix}_{backbone}_{modality}_best.pth")
    if not os.path.exists(path):
        return None
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt


def get_test_acc(prefix, backbone, modality):
    h = load_history(prefix, backbone, modality)
    if h and "test_acc" in h:
        return h["test_acc"]
    return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Collect data ──
    scratch_acc = {}
    finetune_acc = {}
    scratch_sitting_f1 = {}
    finetune_sitting_f1 = {}

    # From-scratch sitting F1 and test acc (from eval output we already ran)
    # Hard-code from eval results since we need per-class metrics not in history
    scratch_results = {
        ("mobilenet_v2", "rgb"):   {"acc": 0.9789, "sitting_f1": 0.67},
        ("mobilenet_v2", "ir"):    {"acc": 0.9842, "sitting_f1": 0.75},
        ("mobilenet_v2", "depth"): {"acc": 0.9842, "sitting_f1": 0.75},
        ("xception", "rgb"):       {"acc": 0.9842, "sitting_f1": 0.75},
        ("xception", "ir"):        {"acc": 0.9842, "sitting_f1": 0.75},
        ("xception", "depth"):     {"acc": 0.9842, "sitting_f1": 0.75},
        ("densenet121", "rgb"):    {"acc": 0.9842, "sitting_f1": 0.75},
        ("densenet121", "ir"):     {"acc": 0.9842, "sitting_f1": 0.75},
        ("densenet121", "depth"):  {"acc": 0.9842, "sitting_f1": 0.75},
    }
    finetune_results = {
        ("mobilenet_v2", "rgb"):   {"acc": 0.9632, "sitting_f1": 0.50},
        ("mobilenet_v2", "ir"):    {"acc": 0.9842, "sitting_f1": 0.75},
        ("mobilenet_v2", "depth"): {"acc": 0.9684, "sitting_f1": 0.60},
        ("xception", "rgb"):       {"acc": 0.9684, "sitting_f1": 0.55},
        ("xception", "ir"):        {"acc": 0.9842, "sitting_f1": 0.75},
        ("xception", "depth"):     {"acc": 0.9737, "sitting_f1": 0.60},
        ("densenet121", "rgb"):    {"acc": 0.9737, "sitting_f1": 0.75},
        ("densenet121", "ir"):     {"acc": 0.9789, "sitting_f1": 0.67},
        ("densenet121", "depth"):  {"acc": 0.9842, "sitting_f1": 0.75},
    }

    # ── Figure 1: Test Accuracy Comparison (grouped bar chart) ──
    fig, ax = plt.subplots(figsize=(14, 6))
    labels = []
    for bb in BACKBONES:
        for mod in MODALITIES:
            labels.append(f"{BB_DISPLAY[bb]}\n{MOD_DISPLAY[mod]}")

    x = np.arange(len(labels))
    width = 0.35

    vals_scratch = [scratch_results[(bb, mod)]["acc"] * 100
                    for bb in BACKBONES for mod in MODALITIES]
    vals_ft = [finetune_results[(bb, mod)]["acc"] * 100
               for bb in BACKBONES for mod in MODALITIES]

    bars1 = ax.bar(x - width/2, vals_scratch, width, label="From Scratch (ImageNet)", color=C_SCRATCH, alpha=0.85)
    bars2 = ax.bar(x + width/2, vals_ft, width, label="Sowbot Pre-trained + Fine-tuned", color=C_FINETUNE, alpha=0.85)

    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title("Posture Classification: From-Scratch vs Sowbot Pre-trained\n(Test set: pigs 16-19, n=190)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(93, 100.5)
    ax.legend(fontsize=11, loc="lower left")
    ax.axhline(98.4, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.15, f"{h:.1f}", ha="center", va="bottom", fontsize=8, color=C_SCRATCH)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.15, f"{h:.1f}", ha="center", va="bottom", fontsize=8, color=C_FINETUNE)

    # Backbone group separators
    for i in [2.5, 5.5]:
        ax.axvline(i, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "comparison_accuracy.png")
    plt.savefig(path1, dpi=200)
    plt.close()
    print(f"Saved: {path1}")

    # ── Figure 2: Sitting F1 Comparison ──
    fig, ax = plt.subplots(figsize=(14, 6))

    vals_sit_s = [scratch_results[(bb, mod)]["sitting_f1"]
                  for bb in BACKBONES for mod in MODALITIES]
    vals_sit_f = [finetune_results[(bb, mod)]["sitting_f1"]
                  for bb in BACKBONES for mod in MODALITIES]

    bars1 = ax.bar(x - width/2, vals_sit_s, width, label="From Scratch (ImageNet)", color=C_SCRATCH, alpha=0.85)
    bars2 = ax.bar(x + width/2, vals_sit_f, width, label="Sowbot Pre-trained + Fine-tuned", color=C_FINETUNE, alpha=0.85)

    ax.set_ylabel("Sitting F1 Score", fontsize=13)
    ax.set_title("Sitting Class F1: From-Scratch vs Sowbot Pre-trained\n(Only 4 sitting samples in test set)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11, loc="lower left")
    ax.axhline(0.75, color="gray", linestyle=":", alpha=0.4, linewidth=0.8, label="_nolegend_")

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=8, color=C_SCRATCH)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=8, color=C_FINETUNE)

    for i in [2.5, 5.5]:
        ax.axvline(i, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "comparison_sitting_f1.png")
    plt.savefig(path2, dpi=200)
    plt.close()
    print(f"Saved: {path2}")

    # ── Figure 3: Sowbot pre-training curves (3 backbones) ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, bb in enumerate(BACKBONES):
        h = load_history(f"sowbot_{bb}", None, None)
        # Try direct path
        hpath = os.path.join(OUTPUT_DIR, f"history_sowbot_{bb}.json")
        if os.path.exists(hpath):
            with open(hpath) as f:
                h = json.load(f)
        if h is None:
            continue
        epochs = range(1, len(h["train_acc"]) + 1)
        ax = axes[i]
        ax.plot(epochs, [v * 100 for v in h["train_acc"]], label="Train", color=C_SCRATCH, linewidth=2)
        ax.plot(epochs, [v * 100 for v in h["val_acc"]], label="Val", color=C_FINETUNE, linewidth=2)
        if h.get("test_acc") is not None:
            ax.axhline(h["test_acc"] * 100, linestyle="--", color="green", alpha=0.7, label=f"Test {h['test_acc']*100:.1f}%")
        ax.set_title(f"{BB_DISPLAY[bb]}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)" if i == 0 else "")
        ax.set_ylim(80, 101)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)

    fig.suptitle("Sowbot Pre-training Accuracy (9,072 IR images, 3 classes)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, "sowbot_pretrain_curves.png")
    plt.savefig(path3, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path3}")

    # ── Figure 4: Summary heatmap ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy heatmap
    acc_scratch = np.array([[scratch_results[(bb, mod)]["acc"] * 100 for mod in MODALITIES] for bb in BACKBONES])
    acc_ft = np.array([[finetune_results[(bb, mod)]["acc"] * 100 for mod in MODALITIES] for bb in BACKBONES])
    delta = acc_ft - acc_scratch

    im1 = ax1.imshow(delta, cmap="RdYlGn", vmin=-2.5, vmax=1.0, aspect="auto")
    ax1.set_xticks(range(3))
    ax1.set_xticklabels([MOD_DISPLAY[m] for m in MODALITIES], fontsize=11)
    ax1.set_yticks(range(3))
    ax1.set_yticklabels([BB_DISPLAY[b] for b in BACKBONES], fontsize=11)
    ax1.set_title("Accuracy Change (%)\n(Fine-tuned minus From-Scratch)", fontsize=12, fontweight="bold")
    for i in range(3):
        for j in range(3):
            val = delta[i, j]
            color = "white" if abs(val) > 1.0 else "black"
            ax1.text(j, i, f"{val:+.1f}%", ha="center", va="center", fontsize=12, fontweight="bold", color=color)
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # Sitting F1 heatmap
    f1_scratch = np.array([[scratch_results[(bb, mod)]["sitting_f1"] for mod in MODALITIES] for bb in BACKBONES])
    f1_ft = np.array([[finetune_results[(bb, mod)]["sitting_f1"] for mod in MODALITIES] for bb in BACKBONES])
    delta_f1 = f1_ft - f1_scratch

    im2 = ax2.imshow(delta_f1, cmap="RdYlGn", vmin=-0.3, vmax=0.1, aspect="auto")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels([MOD_DISPLAY[m] for m in MODALITIES], fontsize=11)
    ax2.set_yticks(range(3))
    ax2.set_yticklabels([BB_DISPLAY[b] for b in BACKBONES], fontsize=11)
    ax2.set_title("Sitting F1 Change\n(Fine-tuned minus From-Scratch)", fontsize=12, fontweight="bold")
    for i in range(3):
        for j in range(3):
            val = delta_f1[i, j]
            color = "white" if abs(val) > 0.1 else "black"
            ax2.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=12, fontweight="bold", color=color)
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle("Impact of Sowbot Pre-training on OAK Performance", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path4 = os.path.join(OUTPUT_DIR, "comparison_heatmap.png")
    plt.savefig(path4, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path4}")

    # ── Figure 5: Dataset overview ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Sowbot class distribution
    sowbot_counts = [1796, 1840, 5436]
    sowbot_labels = ["Standing\n(folder 4)", "Sitting\n(folder 2)", "Lying\n(folders 0,1,3)"]
    colors_sowbot = ["#3b82f6", "#f59e0b", "#10b981"]
    bars = ax1.bar(sowbot_labels, sowbot_counts, color=colors_sowbot, edgecolor="white", linewidth=1.5)
    ax1.set_title("Sowbot 2022 Dataset\n(9,072 IR images)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Number of images")
    for bar, count in zip(bars, sowbot_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f"{count:,}", ha="center", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 6200)

    # OAK class distribution (train + val + test)
    oak_train = [451, 40, 427]
    oak_val = [121, 8, 135]
    oak_test = [84, 4, 102]
    oak_labels = ["Standing", "Sitting", "Lying"]
    x_oak = np.arange(3)
    w = 0.25
    b1 = ax2.bar(x_oak - w, oak_train, w, label="Train (pigs 0-11)", color="#3b82f6", alpha=0.8)
    b2 = ax2.bar(x_oak, oak_val, w, label="Val (pigs 12-15)", color="#f59e0b", alpha=0.8)
    b3 = ax2.bar(x_oak + w, oak_test, w, label="Test (pigs 16-19)", color="#10b981", alpha=0.8)
    ax2.set_title("OAK 2026 Dataset\n(1,372 labeled frames)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Number of frames")
    ax2.set_xticks(x_oak)
    ax2.set_xticklabels(oak_labels)
    ax2.legend(fontsize=9)
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, h + 5, str(int(h)), ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    path5 = os.path.join(OUTPUT_DIR, "dataset_overview.png")
    plt.savefig(path5, dpi=200)
    plt.close()
    print(f"Saved: {path5}")

    print(f"\nAll figures saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
