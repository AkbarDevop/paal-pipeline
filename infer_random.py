"""Random-sample inference sanity check for trained posture models."""

import argparse
import csv
import os
import random
from datetime import datetime

import cv2
import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

from config import (
    LABELS_CSV,
    LABELS_POSTURE3_CSV,
    MODEL_DIR,
    OUTPUT_DIR,
    TRAIN_PIG_IDS,
    VAL_PIG_IDS,
    TEST_PIG_IDS,
    BINARY_CLASSES,
    POSTURE3_CLASSES,
)
from data_loader import SowPostureDataset, MODALITY_CHANNELS
from models import SingleModalModel


def parse_allowed_labels(raw):
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def default_labels_csv(class_set):
    if class_set == "binary":
        return LABELS_CSV
    clean = os.path.join("labels", "labels_posture3_clean.csv")
    return clean if os.path.exists(clean) else LABELS_POSTURE3_CSV


def split_to_pigs(name):
    if name == "train":
        return TRAIN_PIG_IDS
    if name == "val":
        return VAL_PIG_IDS
    if name == "test":
        return TEST_PIG_IDS
    return None


def choose_display_path(sample, modality):
    if modality == "ir":
        return sample.get("ir_jpg", "")
    if modality == "depth":
        return sample.get("depth_jpg", "")
    if modality in ("rgb", "rgb_ir", "rgb_depth", "all"):
        if sample.get("rgb_jpg", ""):
            return sample["rgb_jpg"]
        if sample.get("ir_jpg", ""):
            return sample["ir_jpg"]
        return sample.get("depth_jpg", "")
    return sample.get("rgb_jpg", "")


def choose_rgb_display_path(sample):
    if sample.get("rgb_jpg", ""):
        return sample["rgb_jpg"]
    if sample.get("ir_jpg", ""):
        return sample["ir_jpg"]
    return sample.get("depth_jpg", "")


def class_names_for(class_set, allowed_labels):
    base = BINARY_CLASSES if class_set == "binary" else POSTURE3_CLASSES
    if not allowed_labels:
        return base
    ids = sorted(set(allowed_labels))
    return {i: base.get(lbl, str(lbl)) for i, lbl in enumerate(ids)}


def load_model(modality, model_prefix, num_classes, device):
    model_path = os.path.join(MODEL_DIR, f"{model_prefix}_{modality}_best.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model: {model_path}")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    in_channels = ckpt.get("in_channels", MODALITY_CHANNELS[modality])
    ckpt_classes = ckpt.get("num_classes", num_classes)
    backbone = ckpt.get("backbone", "mobilenet_v2")
    model = SingleModalModel(in_channels=in_channels, num_classes=ckpt_classes, pretrained=False, backbone=backbone).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, model_path, ckpt_classes


def save_grid(images, captions, out_path, cols=4):
    if not HAS_PLT or not images:
        return
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for i, ax in enumerate(axes.flat):
        if i >= len(images):
            ax.axis("off")
            continue
        img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(captions[i], fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def interactive_review(rows, review_csv, window_name="RGB Random Review"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_exists = os.path.exists(review_csv)
    with open(review_csv, "a", newline="") as f:
        fields = [
            "timestamp",
            "index",
            "pig_id",
            "path",
            "true_label",
            "pred_label",
            "confidence",
            "model_correct",
            "human_judgement",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            w.writeheader()

        print("\nInteractive review mode:")
        print("  y = prediction is TRUE")
        print("  n = prediction is FALSE")
        print("  s = skip")
        print("  q = quit")

        reviewed = 0
        yes = 0
        no = 0
        for i, row in enumerate(rows, 1):
            img = cv2.imread(row["path"]) if row["path"] else None
            if img is None:
                img = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                h, w0 = img.shape[:2]
                scale = min(900 / w0, 650 / h, 1.0)
                if scale < 1.0:
                    img = cv2.resize(img, (int(w0 * scale), int(h * scale)))

            l1 = f"[{i}/{len(rows)}] pig{row['pig_id']} | true={row['true_name']} | pred={row['pred_name']} | conf={row['conf']:.3f}"
            l2 = "Press: y=true, n=false, s=skip, q=quit"
            cv2.putText(img, l1, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, l2, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.imshow(window_name, img)

            human = None
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord("y"):
                    human = "true"
                    yes += 1
                    break
                if key == ord("n"):
                    human = "false"
                    no += 1
                    break
                if key == ord("s"):
                    human = "skip"
                    break
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    print(f"Stopped early. reviewed={reviewed}, true={yes}, false={no}")
                    return

            w.writerow(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "index": i,
                    "pig_id": row["pig_id"],
                    "path": row["path"],
                    "true_label": row["true_name"],
                    "pred_label": row["pred_name"],
                    "confidence": f"{row['conf']:.4f}",
                    "model_correct": int(row["model_correct"]),
                    "human_judgement": human,
                }
            )
            f.flush()
            reviewed += 1

        cv2.destroyAllWindows()
        print(f"Review complete. reviewed={reviewed}, true={yes}, false={no}")


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    labels_csv = args.labels_csv or default_labels_csv(args.class_set)
    allowed_labels = parse_allowed_labels(args.allowed_labels)
    class_names = class_names_for(args.class_set, allowed_labels)

    model, model_path, _ = load_model(args.modality, args.model_prefix, args.num_classes, device)
    pig_ids = split_to_pigs(args.split)
    ds = SowPostureDataset(
        modality=args.modality,
        labels_csv=labels_csv,
        pig_ids=pig_ids,
        allowed_labels=allowed_labels,
        silent=True,
    )

    if len(ds) == 0:
        print("No samples found for requested split/labels.")
        return

    n = min(args.n, len(ds))
    picks = random.sample(range(len(ds)), n)

    images = []
    captions = []
    correct = 0
    review_rows = []

    print(f"Model: {model_path}")
    print(f"Samples: {len(ds)} | split={args.split} | random_n={n}")

    with torch.no_grad():
        for k, idx in enumerate(picks, 1):
            x, y = ds[idx]
            sample = ds.samples[idx]
            logits = model(x.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
            true = int(y.item())
            conf = float(probs[pred])
            ok = pred == true
            correct += int(ok)

            true_name = class_names.get(true, str(true))
            pred_name = class_names.get(pred, str(pred))
            print(f"[{k:02d}] pig{sample['pig_id']:>2} | true={true_name:<10} pred={pred_name:<10} conf={conf:.3f} {'OK' if ok else 'MISS'}")

            p = choose_rgb_display_path(sample) if args.display_rgb_only else choose_display_path(sample, args.modality)
            img = cv2.imread(p) if p else None
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (320, 240))

            title = f"pig{sample['pig_id']} T:{true_name} P:{pred_name} {conf:.2f}"
            images.append(img)
            captions.append(title)
            review_rows.append(
                {
                    "pig_id": sample["pig_id"],
                    "path": p,
                    "true_name": true_name,
                    "pred_name": pred_name,
                    "conf": conf,
                    "model_correct": ok,
                }
            )

    acc = correct / n
    print(f"\nRandom-sample accuracy: {correct}/{n} = {acc:.3f}")

    out_path = os.path.join(OUTPUT_DIR, f"random_preds_{args.model_prefix}_{args.modality}_{args.split}.png")
    save_grid(images, captions, out_path, cols=args.cols)
    if HAS_PLT:
        print(f"Saved grid: {out_path}")
    else:
        print("matplotlib not installed; skipped grid image save.")

    if args.review:
        review_csv = args.review_csv or os.path.join(
            OUTPUT_DIR,
            f"review_{args.model_prefix}_{args.modality}_{args.split}.csv",
        )
        interactive_review(review_rows, review_csv)
        print(f"Saved review log: {review_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Random image inference sanity check")
    p.add_argument("--modality", choices=["rgb", "ir", "depth", "rgb_depth", "rgb_ir", "all"], default="depth")
    p.add_argument("--model-prefix", default="posture3clean")
    p.add_argument("--class-set", choices=["binary", "posture3"], default="posture3")
    p.add_argument("--num-classes", type=int, choices=[2, 3], default=3)
    p.add_argument("--labels-csv", default=None)
    p.add_argument("--allowed-labels", default=None)
    p.add_argument("--split", choices=["all", "train", "val", "test"], default="test")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--display-rgb-only", action="store_true")
    p.add_argument("--review", action="store_true")
    p.add_argument("--review-csv", default=None)
    main(p.parse_args())
