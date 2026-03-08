"""Find sitting candidates using trained model inference.

Runs the trained model on all unlabeled pig-present frames, ranks by
sitting probability (class 1), and presents top candidates for quick
labeling. Confirmed labels get appended to labels_posture3.csv.
"""

import argparse
import csv
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config import (
    IMG_SIZE, LABELS_POSTURE3_CSV, METADATA_CSV, MODEL_DIR,
    POSTURE3_CLASSES, PRESENCE_CSV,
)
from data_loader import load_and_preprocess, MODALITY_CHANNELS
from models import SingleModalModel


def load_unlabeled_pig_present():
    """Get all pig-present frames that haven't been labeled yet."""
    with open(METADATA_CSV) as f:
        metadata = list(csv.DictReader(f))

    with open(PRESENCE_CSV) as f:
        present = {
            (r["timestamp_folder"], r["pig_id"])
            for r in csv.DictReader(f)
            if r["pig_present"] == "1"
        }

    labeled = set()
    if os.path.exists(LABELS_POSTURE3_CSV):
        with open(LABELS_POSTURE3_CSV) as f:
            labeled = {(r["timestamp_folder"], r["pig_id"]) for r in csv.DictReader(f)}

    unlabeled = [
        r for r in metadata
        if (r["timestamp_folder"], r["pig_id"]) in present
        and (r["timestamp_folder"], r["pig_id"]) not in labeled
    ]
    return unlabeled


def run_inference(rows, modality, device):
    """Run model on all rows, return predictions with probabilities."""
    model_path = os.path.join(MODEL_DIR, f"posture3_{modality}_best.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, f"posture3_mobilenet_v2_{modality}_best.pth")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    in_channels = ckpt.get("in_channels", MODALITY_CHANNELS[modality])
    num_classes = ckpt.get("num_classes", 3)
    backbone = ckpt.get("backbone", "mobilenet_v2")
    model = SingleModalModel(in_channels=in_channels, num_classes=num_classes, pretrained=False, backbone=backbone).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    results = []
    for i, row in enumerate(rows):
        if modality == "rgb":
            path = row.get("rgb_cropped_jpg") or row.get("rgb_jpg", "")
        elif modality == "ir":
            path = row.get("ir_cropped_jpg") or row.get("ir_jpg", "")
        elif modality == "depth":
            path = row.get("depth_cropped_jpg") or row.get("depth_jpg", "")
        else:
            path = row.get("rgb_cropped_jpg") or row.get("rgb_jpg", "")

        if not path or not os.path.exists(path):
            continue

        img = load_and_preprocess(path)
        if img is None:
            continue

        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        results.append({
            "row": row,
            "probs": probs,
            "sitting_prob": float(probs[1]),
            "max_prob": float(probs.max()),
            "pred": int(probs.argmax()),
        })

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(rows)}")

    return results


def show_candidates(candidates):
    """Show sitting candidates for user confirmation. Saves labels."""
    labels_csv = LABELS_POSTURE3_CSV
    file_exists = os.path.exists(labels_csv)
    fields = ["timestamp_folder", "pig_id", "pig_timestamp", "rgb_jpg", "ir_jpg", "depth_jpg", "label"]

    sitting_count = 0
    reviewed = 0

    print(f"\nShowing {len(candidates)} sitting candidates")
    print("Keys: 1=SITTING (confirm), 0=standing, 2=lying, s=skip, q=quit\n")

    with open(labels_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()

        for i, c in enumerate(candidates):
            row = c["row"]
            probs = c["probs"]

            img_path = row.get("rgb_cropped_jpg") or row.get("rgb_jpg", "")
            if not img_path or not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            scale = min(600 / h, 800 / w, 1.0)
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)))

            bar_h = 40
            bar = np.zeros((bar_h, img.shape[1], 3), dtype=np.uint8) + 30
            prob_text = f"Stand:{probs[0]:.0%}  SIT:{probs[1]:.0%}  Lie:{probs[2]:.0%}"
            cv2.putText(bar, prob_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            display = np.vstack([bar, img])

            title = f"[{i+1}/{len(candidates)}] pig{row['pig_id']} | {row['timestamp_folder']} | sit={probs[1]:.1%}"
            cv2.imshow(title, display)

            label = None
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord("1"):
                    label = 1
                    sitting_count += 1
                    break
                if key == ord("0"):
                    label = 0
                    break
                if key == ord("2"):
                    label = 2
                    break
                if key == ord("s"):
                    label = -1
                    break
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    print(f"\nReviewed: {reviewed}, Sitting confirmed: {sitting_count}")
                    return

            cv2.destroyAllWindows()
            reviewed += 1

            if label == -1:
                continue

            writer.writerow({
                "timestamp_folder": row["timestamp_folder"],
                "pig_id": row["pig_id"],
                "pig_timestamp": row["pig_timestamp"],
                "rgb_jpg": row.get("rgb_jpg", ""),
                "ir_jpg": row.get("ir_jpg", ""),
                "depth_jpg": row.get("depth_jpg", ""),
                "label": label,
            })
            f.flush()

    print(f"\nReviewed: {reviewed}, Sitting confirmed: {sitting_count}")
    print(f"Saved to: {labels_csv}")


def main():
    parser = argparse.ArgumentParser(description="Find sitting candidates via model inference")
    parser.add_argument("--modality", default="rgb", choices=["rgb", "ir", "depth"])
    parser.add_argument("--top-n", type=int, default=200, help="Show top N candidates")
    parser.add_argument("--sort-by", choices=["sitting", "uncertainty"], default="sitting",
                        help="sitting = highest P(sitting); uncertainty = lowest max confidence")
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    print("Loading unlabeled pig-present frames...")
    rows = load_unlabeled_pig_present()
    print(f"  {len(rows)} unlabeled frames")

    if not rows:
        print("Nothing to process.")
        return

    print(f"\nRunning inference ({args.modality} model)...")
    results = run_inference(rows, args.modality, device)
    print(f"  {len(results)} frames processed")

    # Summary
    pred_counts = {}
    for r in results:
        p = POSTURE3_CLASSES[r["pred"]]
        pred_counts[p] = pred_counts.get(p, 0) + 1
    print(f"\nPrediction distribution: {pred_counts}")

    # Sort
    if args.sort_by == "sitting":
        candidates = sorted(results, key=lambda x: x["sitting_prob"], reverse=True)
    else:
        candidates = sorted(results, key=lambda x: x["max_prob"])

    candidates = candidates[:args.top_n]

    if not candidates:
        print("No candidates found.")
        return

    top_sit = candidates[0]["sitting_prob"]
    bot_sit = candidates[-1]["sitting_prob"]
    print(f"\nTop {len(candidates)} candidates (sorted by {args.sort_by}):")
    print(f"  Sitting prob range: {bot_sit:.1%} — {top_sit:.1%}")

    show_candidates(candidates)


if __name__ == "__main__":
    main()
