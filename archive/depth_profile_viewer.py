"""View depth profiles with stall-bar masking for pig detection analysis.

Shows depth maps, histograms, and slice plots. Optionally label frames
as pig/no-pig for ground truth collection.

Camera is side-mounted (vulva side), so:
  - No pig: uniform depth to far wall → low std, narrow histogram
  - Pig present: mixed depths (pig body + wall) → high std, wide histogram

Modes:
  View mode (default):  just browse frames
  Label mode (--label): press y/n after each frame, saves to CSV
"""

import argparse
import csv
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

from config import CROP_TOF, METADATA_CSV, LABEL_DIR

TOF_W, TOF_H = 640, 480

# Stall bar mask: pixels to exclude on left/right edges of cropped region
BAR_MARGIN = 50  # pixels to mask on each side

GROUND_TRUTH_CSV = os.path.join(LABEL_DIR, "pig_detection_gt.csv")


def load_depth_raw(path):
    """Load depth raw file (uint16, 640x480, optional 8-byte header)."""
    if not path or not os.path.exists(path):
        return None
    size = os.path.getsize(path)
    expected = TOF_W * TOF_H * 2
    if size == expected + 8:
        raw = np.fromfile(path, dtype=np.uint16, offset=8)
    elif size == expected:
        raw = np.fromfile(path, dtype=np.uint16)
    else:
        return None
    if raw.size != TOF_W * TOF_H:
        return None
    return raw.reshape(TOF_H, TOF_W)


def compute_depth_stats(cropped, bar_margin):
    """Compute depth stats for the masked (bar-free) region."""
    ch, cw = cropped.shape
    bar_mask = np.ones((ch, cw), dtype=bool)
    bar_mask[:, :bar_margin] = False
    bar_mask[:, cw - bar_margin:] = False

    valid_masked = (cropped > 200) & (cropped < 5000) & bar_mask
    masked_depths = cropped[valid_masked]

    stats = {}
    if masked_depths.size > 0:
        stats["mean"] = masked_depths.mean()
        stats["median"] = float(np.median(masked_depths))
        stats["std"] = masked_depths.std()
        stats["q25"] = float(np.percentile(masked_depths, 25))
        stats["q75"] = float(np.percentile(masked_depths, 75))
        stats["iqr"] = stats["q75"] - stats["q25"]
        stats["p10"] = float(np.percentile(masked_depths, 10))
        stats["p90"] = float(np.percentile(masked_depths, 90))
        stats["range_p10_p90"] = stats["p90"] - stats["p10"]
        stats["valid_pct"] = valid_masked.mean() * 100
    else:
        for k in ["mean", "median", "std", "q25", "q75", "iqr", "p10", "p90", "range_p10_p90", "valid_pct"]:
            stats[k] = 0

    stats["masked_depths"] = masked_depths
    stats["valid_masked"] = valid_masked
    stats["bar_mask"] = bar_mask
    return stats


def make_depth_image(depth_2d, mask=None):
    """Create a continuous colormap depth image. Masked pixels shown as dark gray."""
    h, w = depth_2d.shape
    valid = (depth_2d > 200) & (depth_2d < 5000)
    if mask is not None:
        valid = valid & mask

    img = np.full((h, w, 3), 40, dtype=np.uint8)

    if valid.any():
        vals = depth_2d[valid].astype(np.float32)
        normed = (vals - 200) / (5000 - 200)
        cmap = plt.cm.RdYlBu
        colors = (cmap(normed)[:, :3] * 255).astype(np.uint8)
        img[valid] = colors

    if mask is not None:
        excluded = ~mask & (depth_2d > 0)
        img[excluded] = [20, 20, 20]

    return img


def show_frame(row, i, total, cropped, stats, bar_margin, label_mode=False):
    """Display one frame with all panels. Returns label if label_mode."""
    ch, cw = cropped.shape
    pig_id = row.get("pig_id", "?")
    ts = row.get("timestamp_folder", "?")

    h_mid_y = ch // 2
    h_bot_y = int(ch * 0.75)
    v_slice_x = cw // 2

    h_mid_values = cropped[h_mid_y, bar_margin:cw - bar_margin]
    h_bot_values = cropped[h_bot_y, bar_margin:cw - bar_margin]
    v_values = cropped[:, v_slice_x]

    masked_depths = stats["masked_depths"]
    bar_mask = stats["bar_mask"]

    if label_mode:
        # 2x3 layout: add IR image in top-left, shift depth panels
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))

        # ── Top-left: IR image for visual reference ──
        ax_ir = axes[0, 0]
        ir_path = row.get("ir_jpg", "")
        if ir_path and os.path.exists(ir_path):
            ir_img = cv2.imread(ir_path)
            ir_img = cv2.cvtColor(ir_img, cv2.COLOR_BGR2RGB)
            ax_ir.imshow(ir_img)
        else:
            ax_ir.text(0.5, 0.5, "No IR image", ha="center", va="center", transform=ax_ir.transAxes)
        ax_ir.set_title(f"IR image (visual reference)")
        ax_ir.set_xlabel("Press Y=pig, N=no pig, S=skip")

        # ── Top-mid: Masked depth ──
        ax_masked = axes[0, 1]
        masked_img = make_depth_image(cropped, mask=bar_mask)
        cv2.line(masked_img, (bar_margin, h_mid_y), (cw - bar_margin, h_mid_y), (255, 255, 255), 1)
        cv2.line(masked_img, (bar_margin, h_bot_y), (cw - bar_margin, h_bot_y), (200, 200, 200), 1)
        cv2.line(masked_img, (v_slice_x, 0), (v_slice_x, ch), (200, 200, 200), 1)
        ax_masked.imshow(masked_img)
        ax_masked.set_title("Masked depth (bars removed)")

        # ── Top-right: Histogram ──
        ax_hist = axes[0, 2]
    else:
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))

        # ── Top-left: Raw depth ──
        ax_raw = axes[0, 0]
        raw_img = make_depth_image(cropped)
        cv2.line(raw_img, (bar_margin, 0), (bar_margin, ch), (255, 255, 255), 1)
        cv2.line(raw_img, (cw - bar_margin, 0), (cw - bar_margin, ch), (255, 255, 255), 1)
        ax_raw.imshow(raw_img)
        ax_raw.set_title("Raw cropped depth")
        ax_raw.set_xlabel("White lines = stall bar boundaries")

        # ── Top-mid: Masked depth ──
        ax_masked = axes[0, 1]
        masked_img = make_depth_image(cropped, mask=bar_mask)
        cv2.line(masked_img, (bar_margin, h_mid_y), (cw - bar_margin, h_mid_y), (255, 255, 255), 1)
        cv2.line(masked_img, (bar_margin, h_bot_y), (cw - bar_margin, h_bot_y), (200, 200, 200), 1)
        cv2.line(masked_img, (v_slice_x, 0), (v_slice_x, ch), (200, 200, 200), 1)
        ax_masked.imshow(masked_img)
        ax_masked.set_title("Masked depth (bars removed)")
        ax_masked.set_xlabel("Red=close  Yellow=mid  Blue=far")

        # ── Top-right: Histogram ──
        ax_hist = axes[0, 2]

    # Histogram
    if masked_depths.size > 0:
        bins = np.linspace(200, 5000, 101)
        counts, edges = np.histogram(masked_depths, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        cmap = plt.cm.RdYlBu
        normed = (centers - 200) / (5000 - 200)
        colors = cmap(normed)
        ax_hist.bar(centers, counts, width=(edges[1] - edges[0]),
                    color=colors, edgecolor="none", alpha=0.85)
        ax_hist.axvline(x=stats["mean"], color="black", linestyle="-", linewidth=1.5,
                        label=f"Mean={stats['mean']:.0f}")
        ax_hist.axvline(x=stats["median"], color="gray", linestyle="--", linewidth=1.5,
                        label=f"Median={stats['median']:.0f}")
        ax_hist.axvspan(stats["q25"], stats["q75"], color="black", alpha=0.1,
                        label=f"IQR={stats['iqr']:.0f}")
        ax_hist.legend(fontsize=8)
    ax_hist.set_title("Depth histogram (bars masked)")
    ax_hist.set_xlabel("Depth (mm)")
    ax_hist.set_ylabel("Pixel count")
    ax_hist.set_xlim(200, 5000)
    ax_hist.grid(True, alpha=0.3)

    # ── Bottom-left: Box plot ──
    ax_box = axes[1, 0]
    if masked_depths.size > 0:
        ax_box.boxplot(masked_depths, vert=True, widths=0.5,
                       patch_artist=True,
                       boxprops=dict(facecolor="#FFC800", alpha=0.6),
                       medianprops=dict(color="red", linewidth=2),
                       whiskerprops=dict(color="white"),
                       capprops=dict(color="white"),
                       flierprops=dict(marker='.', markersize=2, color='gray', alpha=0.3))
        ax_box.set_facecolor("#1a1a1a")
        ax_box.set_ylabel("Depth (mm)")
        ax_box.set_ylim(200, 5000)
        ax_box.set_title(f"Depth spread  |  Std={stats['std']:.0f}  IQR={stats['iqr']:.0f}")
        ax_box.set_xticks([])
        ax_box.text(1.3, stats["median"], f"Median={stats['median']:.0f}", fontsize=9, color="red", va="center")
        ax_box.text(1.3, stats["q25"], f"Q25={stats['q25']:.0f}", fontsize=8, color="white", va="center")
        ax_box.text(1.3, stats["q75"], f"Q75={stats['q75']:.0f}", fontsize=8, color="white", va="center")
    ax_box.grid(True, alpha=0.3)

    # ── Bottom-mid: Horizontal slices ──
    ax_h = axes[1, 1]
    slice_x = np.arange(bar_margin, cw - bar_margin)
    ax_h.plot(slice_x, h_mid_values, color="white", linewidth=1.2, label=f"Mid (y={h_mid_y})")
    ax_h.plot(slice_x, h_bot_values, color="lightgray", linewidth=1.2, label=f"Bottom 75% (y={h_bot_y})")
    ax_h.set_title("Horizontal slices (bars excluded)")
    ax_h.set_xlabel("X pixel")
    ax_h.set_ylabel("Depth (mm)")
    ax_h.set_ylim(0, 5500)
    ax_h.set_facecolor("#1a1a1a")
    ax_h.legend(fontsize=9)
    ax_h.grid(True, alpha=0.3)

    # ── Bottom-right: Vertical slice ──
    ax_v = axes[1, 2]
    ax_v.plot(range(ch), v_values, color="white", linewidth=1.2)
    ax_v.set_title(f"Vertical slice (x={v_slice_x}, top → bottom)")
    ax_v.set_xlabel("Y pixel")
    ax_v.set_ylabel("Depth (mm)")
    ax_v.set_ylim(0, 5500)
    ax_v.set_facecolor("#1a1a1a")
    ax_v.grid(True, alpha=0.3)

    fig.suptitle(
        f"[{i+1}/{total}] pig{pig_id} | {ts}\n"
        f"Std: {stats['std']:.0f}mm  IQR: {stats['iqr']:.0f}mm  Range(p10-p90): {stats['range_p10_p90']:.0f}mm  |  "
        f"Mean: {stats['mean']:.0f}mm  Median: {stats['median']:.0f}mm  Valid: {stats['valid_pct']:.0f}%",
        fontsize=11,
    )

    plt.tight_layout()

    if label_mode:
        plt.show(block=False)
        plt.pause(0.1)
        while True:
            inp = input(f"  [{i+1}/{total}] pig{pig_id} | {ts}  →  (y)pig / (n)no pig / (s)skip / (q)quit: ").strip().lower()
            if inp in ("y", "n", "s", "q"):
                plt.close(fig)
                return inp
            print("    Invalid input. Press y, n, s, or q.")
    else:
        plt.show()
        return None


def load_existing_labels(csv_path):
    """Load already-labeled frame keys to skip them."""
    labeled = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
                labeled.add((row["timestamp_folder"], row["pig_id"]))
    return labeled


def main(args):
    global BAR_MARGIN
    BAR_MARGIN = args.bar_margin

    if not os.path.exists(args.metadata_csv):
        print(f"Missing metadata: {args.metadata_csv}")
        return

    with open(args.metadata_csv, "r") as f:
        rows = list(csv.DictReader(f))

    rows = [r for r in rows if r.get("depth_raw") and os.path.exists(r["depth_raw"])]
    print(f"Found {len(rows)} frames with depth raw files")

    if not rows:
        return

    # In label mode, skip already-labeled frames
    if args.label:
        already_labeled = load_existing_labels(args.output_csv)
        if already_labeled:
            before = len(rows)
            rows = [r for r in rows if (r["timestamp_folder"], r["pig_id"]) not in already_labeled]
            print(f"Skipping {before - len(rows)} already-labeled frames ({len(rows)} remaining)")

    random.shuffle(rows)
    rows = rows[: args.n]

    if args.label:
        print(f"\n=== LABEL MODE ===")
        print(f"For each frame: y=pig present, n=no pig, s=skip, q=quit")
        print(f"Labels saved to: {args.output_csv}\n")

    x1, y1, x2, y2 = CROP_TOF
    labeled_count = 0

    for i, row in enumerate(rows):
        depth = load_depth_raw(row["depth_raw"])
        if depth is None:
            print(f"  Skipping {row['depth_raw']} (failed to load)")
            continue

        cropped = depth[y1:y2, x1:x2]
        stats = compute_depth_stats(cropped, BAR_MARGIN)

        result = show_frame(row, i, len(rows), cropped, stats, BAR_MARGIN, label_mode=args.label)

        if args.label:
            if result == "q":
                print(f"\nQuit. Labeled {labeled_count} frames this session.")
                break
            if result == "s":
                continue
            if result in ("y", "n"):
                pig_present = 1 if result == "y" else 0
                # Append to CSV
                file_exists = os.path.exists(args.output_csv)
                with open(args.output_csv, "a", newline="") as f:
                    fields = [
                        "timestamp_folder", "pig_id", "pig_present",
                        "depth_mean", "depth_median", "depth_std",
                        "depth_iqr", "depth_range_p10_p90", "valid_pct",
                    ]
                    w = csv.DictWriter(f, fieldnames=fields)
                    if not file_exists:
                        w.writeheader()
                    w.writerow({
                        "timestamp_folder": row["timestamp_folder"],
                        "pig_id": row["pig_id"],
                        "pig_present": pig_present,
                        "depth_mean": f"{stats['mean']:.1f}",
                        "depth_median": f"{stats['median']:.1f}",
                        "depth_std": f"{stats['std']:.1f}",
                        "depth_iqr": f"{stats['iqr']:.1f}",
                        "depth_range_p10_p90": f"{stats['range_p10_p90']:.1f}",
                        "valid_pct": f"{stats['valid_pct']:.1f}",
                    })
                labeled_count += 1
                print(f"    → {'PIG' if pig_present else 'NO PIG'} (saved)")

    if args.label:
        # Show summary
        total_labeled = 0
        pig_count = 0
        if os.path.exists(args.output_csv):
            with open(args.output_csv, "r") as f:
                for r in csv.DictReader(f):
                    total_labeled += 1
                    if r["pig_present"] == "1":
                        pig_count += 1
        print(f"\nTotal labeled: {total_labeled} (pig: {pig_count}, no pig: {total_labeled - pig_count})")
        print(f"Saved to: {args.output_csv}")
    else:
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View depth cross-section profiles")
    parser.add_argument("-n", type=int, default=20, help="Number of random frames to show")
    parser.add_argument("--metadata-csv", default=METADATA_CSV)
    parser.add_argument("--bar-margin", type=int, default=BAR_MARGIN,
                        help="Pixels to mask on left/right edges for stall bars")
    parser.add_argument("--label", action="store_true",
                        help="Enable labeling mode: press y/n for each frame")
    parser.add_argument("--output-csv", default=GROUND_TRUTH_CSV,
                        help="Where to save ground truth labels")
    args = parser.parse_args()
    main(args)
