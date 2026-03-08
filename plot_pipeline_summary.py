"""Generate a presentation-ready pipeline summary figure."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

from config import OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)


def fig_pipeline_overview():
    """Figure 1: Recommended pipeline flowchart."""
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Box style
    box_kw = dict(boxstyle="round,pad=0.4", linewidth=2)
    arrow_kw = dict(arrowstyle="->,head_width=0.4,head_length=0.2", lw=2.5, color="#475569")

    # Boxes: x, y, label, color
    boxes = [
        (1.0, 2.5, "OAK-D ToF\nCamera", "#e0f2fe", "#0284c7"),
        (4.0, 2.5, "Depth\nPrefilter", "#fef3c7", "#d97706"),
        (7.0, 2.5, "Crop ROI\n(remove neighbors)", "#fce7f3", "#be185d"),
        (10.0, 2.5, "MobileNetV2\n(IR modality)", "#dbeafe", "#2563eb"),
        (13.5, 2.5, "Posture\nPrediction", "#dcfce7", "#16a34a"),
    ]

    for x, y, text, facecolor, edgecolor in boxes:
        bbox = mpatches.FancyBboxPatch((x - 1.1, y - 0.7), 2.2, 1.4,
                                        boxstyle="round,pad=0.15",
                                        facecolor=facecolor, edgecolor=edgecolor, linewidth=2.5)
        ax.add_patch(bbox)
        ax.text(x, y, text, ha="center", va="center", fontsize=12, fontweight="bold", color=edgecolor)

    # Arrows between boxes
    for x1, x2 in [(2.1, 2.9), (5.1, 5.9), (8.1, 8.9), (11.1, 12.4)]:
        ax.annotate("", xy=(x2, 2.5), xytext=(x1, 2.5), arrowprops=arrow_kw)

    # Output labels under last box
    outputs = ["Standing", "Sitting", "Lying"]
    colors_out = ["#3b82f6", "#f59e0b", "#10b981"]
    for i, (label, color) in enumerate(zip(outputs, colors_out)):
        yy = 1.3 - i * 0.5
        ax.annotate("", xy=(13.5 + (i - 1) * 0.8, yy + 0.15), xytext=(13.5, 1.8),
                     arrowprops=dict(arrowstyle="->", lw=1.5, color=color))
        ax.text(13.5 + (i - 1) * 0.8, yy - 0.1, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, linewidth=1.5))

    # Annotation: pig present?
    ax.text(4.0, 1.3, "Pig present?", ha="center", fontsize=10, style="italic", color="#92400e")
    ax.annotate("", xy=(4.0, 1.55), xytext=(4.0, 1.8), arrowprops=dict(arrowstyle="->", lw=1.2, color="#92400e"))

    # Title
    ax.text(8.0, 4.5, "Recommended Pipeline: Sow Posture Classification (PAAL)",
            ha="center", va="center", fontsize=16, fontweight="bold", color="#1e293b")
    ax.text(8.0, 4.0, "MobileNetV2 + IR  |  98.4% accuracy  |  2.2M parameters  |  Works day & night",
            ha="center", va="center", fontsize=11, color="#64748b")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pipeline_overview.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


def fig_model_selection():
    """Figure 2: Why MobileNetV2/IR — comparison table as figure."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    # Table data
    col_labels = ["Backbone", "Modality", "Params", "Test Acc", "Sitting F1", "Recommended"]
    rows = [
        ["MobileNetV2", "IR", "2.2M", "98.4%", "0.75", "BEST"],
        ["MobileNetV2", "Depth", "2.2M", "98.4%", "0.75", ""],
        ["MobileNetV2", "RGB", "2.2M", "97.9%", "0.67", ""],
        ["Xception", "IR", "20.8M", "98.4%", "0.75", ""],
        ["Xception", "Depth", "20.8M", "98.4%", "0.75", ""],
        ["Xception", "RGB", "20.8M", "98.4%", "0.75", ""],
        ["DenseNet121", "IR", "7.0M", "98.4%", "0.75", ""],
        ["DenseNet121", "Depth", "7.0M", "98.4%", "0.75", ""],
        ["DenseNet121", "RGB", "7.0M", "98.4%", "0.75", ""],
    ]

    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#1e40af")
        cell.set_text_props(color="white", fontweight="bold", fontsize=12)

    # Highlight recommended row
    for j in range(len(col_labels)):
        cell = table[1, j]
        cell.set_facecolor("#dbeafe")
        cell.set_text_props(fontweight="bold")

    # Color the "BEST" cell
    best_cell = table[1, 5]
    best_cell.set_facecolor("#16a34a")
    best_cell.set_text_props(color="white", fontweight="bold")

    # Alternate row shading
    for i in range(2, len(rows) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[i, j].set_facecolor("#f8fafc")

    ax.set_title("Model Selection: All 9 Configurations (From-Scratch, ImageNet Pre-trained)\n"
                 "MobileNetV2/IR recommended: same accuracy as larger models, 10x fewer parameters",
                 fontsize=14, fontweight="bold", pad=20, color="#1e293b")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "model_selection_table.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


def fig_experiment_summary():
    """Figure 3: Full experiment summary — what we tried and the conclusion."""
    fig = plt.figure(figsize=(16, 9))

    # Layout: 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # ── Panel A: Sowbot pre-training result (heatmap) ──
    ax1 = fig.add_subplot(gs[0, 0])
    backbones = ["MobileNetV2", "Xception", "DenseNet121"]
    modalities = ["RGB", "IR", "Depth"]
    delta = np.array([
        [-1.6, 0.0, -1.6],
        [-1.6, 0.0, -1.0],
        [-1.0, -0.5, 0.0],
    ])
    im = ax1.imshow(delta, cmap="RdYlGn", vmin=-2.5, vmax=1.0, aspect="auto")
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(modalities, fontsize=10)
    ax1.set_yticks(range(3))
    ax1.set_yticklabels(backbones, fontsize=10)
    ax1.set_title("A. Sowbot Pre-training Effect\n(accuracy change vs from-scratch)", fontsize=12, fontweight="bold")
    for i in range(3):
        for j in range(3):
            color = "white" if abs(delta[i, j]) > 1.0 else "black"
            ax1.text(j, i, f"{delta[i, j]:+.1f}%", ha="center", va="center", fontsize=12, fontweight="bold", color=color)
    plt.colorbar(im, ax=ax1, shrink=0.8)

    # ── Panel B: Best model per-class metrics ──
    ax2 = fig.add_subplot(gs[0, 1])
    classes = ["Standing", "Sitting", "Lying"]
    precision = [0.99, 0.75, 0.99]
    recall = [0.99, 0.75, 0.99]
    f1 = [0.99, 0.75, 0.99]
    support = [84, 4, 102]

    x = np.arange(3)
    w = 0.22
    b1 = ax2.bar(x - w, [p * 100 for p in precision], w, label="Precision", color="#3b82f6", alpha=0.85)
    b2 = ax2.bar(x, [r * 100 for r in recall], w, label="Recall", color="#f59e0b", alpha=0.85)
    b3 = ax2.bar(x + w, [f * 100 for f in f1], w, label="F1", color="#10b981", alpha=0.85)
    ax2.set_ylim(0, 115)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{c}\n(n={s})" for c, s in zip(classes, support)], fontsize=10)
    ax2.set_ylabel("Score (%)")
    ax2.set_title("B. Best Model Per-Class Metrics\n(MobileNetV2/IR, test set)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right")
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.0f}", ha="center", fontsize=8, fontweight="bold")

    # ── Panel C: Data challenge ──
    ax3 = fig.add_subplot(gs[1, 0])
    categories = ["Standing", "Sitting", "Lying"]
    sowbot = [1796, 1840, 5436]
    oak = [451 + 121 + 84, 40 + 8 + 4, 427 + 135 + 102]

    x = np.arange(3)
    w = 0.3
    ax3.bar(x - w / 2, sowbot, w, label="Sowbot 2022 (9,072)", color="#6366f1", alpha=0.85)
    ax3.bar(x + w / 2, oak, w, label="OAK 2026 (1,372)", color="#f97316", alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, fontsize=10)
    ax3.set_ylabel("Number of samples")
    ax3.set_title("C. Dataset Comparison\n(Sitting severely underrepresented in OAK)", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)

    # Highlight sitting
    ax3.annotate("Only 52 total\n(4 in test!)", xy=(1 + w / 2, 52), xytext=(1.6, 1500),
                 fontsize=10, fontweight="bold", color="#dc2626",
                 arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2))

    # ── Panel D: Key takeaways ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    takeaways = [
        ("Recommended Model", "MobileNetV2 + IR modality\n98.4% accuracy, 2.2M params, works day & night"),
        ("Sowbot Pre-training", "Did not improve OAK performance\nImageNet features > sowbot-specific features"),
        ("Main Limitation", "Sitting class: only 52 samples (4 in test)\nF1 = 0.75 — needs more labeled data"),
        ("Next Steps", "1) Label more sitting frames (20-30 minimum)\n2) Deploy MobileNetV2/IR for real-time inference\n3) Consider IR+Depth ensemble for robustness"),
    ]

    ax4.set_title("D. Key Takeaways & Next Steps", fontsize=12, fontweight="bold")
    colors = ["#16a34a", "#dc2626", "#d97706", "#2563eb"]
    for i, ((title, body), color) in enumerate(zip(takeaways, colors)):
        y = 0.88 - i * 0.25
        ax4.text(0.02, y, title, fontsize=11, fontweight="bold", color=color,
                 transform=ax4.transAxes, va="top")
        ax4.text(0.02, y - 0.06, body, fontsize=9.5, color="#334155",
                 transform=ax4.transAxes, va="top", linespacing=1.4)

    fig.suptitle("PAAL Sow Posture Classification — Experiment Summary",
                 fontsize=16, fontweight="bold", color="#0f172a", y=0.98)

    path = os.path.join(OUTPUT_DIR, "experiment_summary.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


def fig_confusion_matrices():
    """Figure 4: Confusion matrices for the recommended model + runner-ups."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    cms = {
        "MobileNetV2/IR\n(Recommended)": np.array([[83, 1, 0], [0, 3, 1], [1, 0, 101]]),
        "MobileNetV2/Depth": np.array([[83, 1, 0], [0, 3, 1], [1, 0, 101]]),
        "Xception/IR": np.array([[83, 1, 0], [0, 3, 1], [1, 0, 101]]),
    }
    classes = ["Standing", "Sitting", "Lying"]

    for idx, (title, cm) in enumerate(cms.items()):
        ax = axes[idx]
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(classes, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True" if idx == 0 else "")
        ax.set_title(title, fontsize=12, fontweight="bold")
        thresh = cm.max() / 2.0
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, fontweight="bold",
                        color="white" if cm[i, j] > thresh else "black")

    fig.suptitle("Confusion Matrices — Top Models (98.4% accuracy, 3/190 errors)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "top_confusion_matrices.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    fig_pipeline_overview()
    fig_model_selection()
    fig_experiment_summary()
    fig_confusion_matrices()
    print(f"\nAll presentation figures saved to: {OUTPUT_DIR}/")
