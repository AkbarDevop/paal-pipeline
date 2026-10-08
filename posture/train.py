"""Train the posture classifier.

Two recipes (pick with --recipe):
  baseline  what v1.0 did (README command --use-cropped --class-weights):
            stall-cropped 0..1 inputs (no ImageNet normalization), no
            augmentation, inverse-frequency class weights, Adam 1e-4 +
            ReduceLROnPlateau, keep the epoch with the best validation accuracy.
  improved  v1.1: ImageNet normalization (the backbones are ImageNet
            pretrained), light augmentation (flip, small shift/rotate,
            brightness/contrast for day/night IR), AdamW + cosine schedule,
            label smoothing, sqrt-inverse-frequency class weights for the rare
            sitting class, keep the epoch with the best validation macro-F1,
            early stopping.

Two evaluation modes:
  default   fixed pig split from config.py (train 0-11 / val 12-15 / test 16-19),
            saves models/{prefix}_{backbone}_{modality}_best.pth
  --cv K    grouped K-fold by pig: every pig is tested once by a model that never
            saw it, metrics are pooled over all folds. Use this to compare recipes;
            the fixed test split has only ~190 frames and 5 sitting frames.

Examples:
  python posture/train.py --class-set posture3 --modality ir --recipe improved \\
      --data-root "C:\\PAAL_data\\labeled_frames"
  python posture/train.py --class-set posture3 --modality ir --cv 5 --recipe baseline --no-save
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
import pathlib
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import (BATCH_SIZE, BINARY_CLASSES, LABELS_CSV, LABELS_POSTURE3_CSV, LEARNING_RATE,
                    MODEL_DIR, NUM_EPOCHS, OUTPUT_DIR, POSTURE3_CLASSES, TEST_PIG_IDS,
                    TRAIN_PIG_IDS, VAL_PIG_IDS, IMG_SIZE)
from data_loader import MODALITY_CHANNELS, cache_images, read_label_rows
from models import BACKBONES, build_model
from preprocess import normalize_batch, pick_device

RECIPES = {
    # README v1.0 command: --use-cropped --class-weights (inverse frequency), Adam, plateau
    "baseline": dict(normalize=None, augment=False, optimizer="adam", lr=LEARNING_RATE,
                     weight_decay=0.0, schedule="plateau", label_smoothing=0.0,
                     class_weights="inverse", select="val_acc", patience=None, epochs=NUM_EPOCHS),
    "improved": dict(normalize="imagenet", augment=True, optimizer="adamw", lr=3e-4,
                     weight_decay=1e-4, schedule="cosine", label_smoothing=0.05,
                     class_weights="sqrt", select="val_macro_f1", patience=10, epochs=NUM_EPOCHS),
}


# ── utilities ──────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def parse_allowed_labels(raw):
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip() != ""]


def compute_metrics(y_true, y_pred, n_classes):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    per_class = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class.append({"precision": prec, "recall": rec, "f1": f1, "support": int(cm[c, :].sum())})
    present = [c for c in range(n_classes) if per_class[c]["support"] > 0]
    return {
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "macro_f1": float(np.mean([per_class[c]["f1"] for c in present])) if present else 0.0,
        "balanced_accuracy": float(np.mean([per_class[c]["recall"] for c in present])) if present else 0.0,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def format_metrics(m, class_names):
    lines = [f"  frames={m['n']}  accuracy={m['accuracy']*100:.2f}%  macro-F1={m['macro_f1']*100:.2f}%  "
             f"balanced-acc={m['balanced_accuracy']*100:.2f}%",
             f"  {'class':<12}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}"]
    for c, pc in enumerate(m["per_class"]):
        lines.append(f"  {class_names[c]:<12}{pc['precision']*100:>9.1f}%{pc['recall']*100:>8.1f}%"
                     f"{pc['f1']*100:>7.1f}%{pc['support']:>9}")
    lines.append("  confusion (rows=true, cols=pred): " + str(m["confusion_matrix"]))
    return "\n".join(lines)


# ── data ───────────────────────────────────────────────────────────────────

class ArrayDataset(torch.utils.data.Dataset):
    """Cached uint8 images -> float tensors in 0..1 (normalization happens on device)."""

    def __init__(self, images, labels, augment=False):
        self.images, self.labels, self.augment = images, labels, augment
        self.geo = self.color = None
        if augment:
            from torchvision.transforms import v2 as T
            # geometry is applied once to all channels so fused modalities stay aligned
            self.geo = T.Compose([
                T.RandomHorizontalFlip(0.5),
                T.RandomAffine(degrees=5, translate=(0.04, 0.04), scale=(0.95, 1.05)),
            ])
            self.color = T.ColorJitter(brightness=0.25, contrast=0.25)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = torch.from_numpy(np.ascontiguousarray(self.images[i].transpose(2, 0, 1)))
        if self.geo is not None:
            x = self.geo(x)
            c = x.shape[0]
            # ColorJitter works on 3-channel images: jitter each modality block on its own
            x = torch.cat([self.color(x[k:k + 3]) for k in range(0, c, 3)], 0) if c > 3 else self.color(x)
        return x.float().div_(255.0), int(self.labels[i])


def class_weight_tensor(labels, n_classes, mode, device):
    if mode == "none":
        return None
    counts = Counter(int(l) for l in labels)
    n = len(labels)
    w = []
    for c in range(n_classes):
        base = n / (n_classes * max(counts.get(c, 0), 1))
        w.append(math.sqrt(base) if mode == "sqrt" else base)
    return torch.tensor(w, dtype=torch.float32, device=device)


# ── training ───────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, device, normalize, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss, preds, labels = 0.0, [], []
    with torch.set_grad_enabled(train_mode):
        for x, y in loader:
            x = normalize_batch(x.to(device), normalize)
            y = y.to(device)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            if train_mode:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            preds.append(out.argmax(1).cpu())
            labels.append(y.cpu())
    if not labels:
        return 0.0, np.array([]), np.array([])
    return total_loss / sum(len(l) for l in labels), torch.cat(labels).numpy(), torch.cat(preds).numpy()


def predict(model, images, device, normalize, batch=64):
    model.eval()
    probs = []
    with torch.inference_mode():
        for i in range(0, len(images), batch):
            x = torch.from_numpy(np.ascontiguousarray(images[i:i + batch].transpose(0, 3, 1, 2)))
            x = normalize_batch(x.to(device).float().div_(255.0), normalize)
            probs.append(torch.softmax(model(x), 1).cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros((0, 1))


def train_one(images, labels, tr_idx, va_idx, args, recipe, n_classes, device, log_prefix=""):
    """Train on tr_idx, select on va_idx. Returns (model, info)."""
    model = build_model(modality=args.modality, num_classes=n_classes, pretrained=not args.no_pretrained,
                        backbone=args.backbone).to(device)
    if args.init_weights:
        ckpt = torch.load(args.init_weights, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    g = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        ArrayDataset(images[tr_idx], labels[tr_idx], augment=recipe["augment"]),
        batch_size=args.batch, shuffle=True, generator=g, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        ArrayDataset(images[va_idx], labels[va_idx]), batch_size=64, shuffle=False, num_workers=0)

    cw = class_weight_tensor(labels[tr_idx], n_classes, recipe["class_weights"], device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=recipe["label_smoothing"])
    if recipe["optimizer"] == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"])
    else:
        optimizer = optim.Adam(model.parameters(), lr=recipe["lr"])
    epochs = recipe["epochs"]
    if recipe["schedule"] == "cosine":
        warm = 2
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda e: (e + 1) / warm if e < warm else
                                                0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, epochs - warm))))
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best = {"score": -1.0, "loss": float("inf"), "epoch": 0, "state": None}
    history = []
    stale = 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_y, tr_p = run_epoch(model, train_loader, criterion, device, recipe["normalize"], optimizer)
        va_loss, va_y, va_p = run_epoch(model, val_loader, criterion, device, recipe["normalize"])
        if recipe["schedule"] == "cosine":
            scheduler.step()
        else:
            scheduler.step(va_loss)
        va = compute_metrics(va_y, va_p, n_classes)
        score = va["macro_f1"] if recipe["select"] == "val_macro_f1" else va["accuracy"]
        improved = score > best["score"] or (recipe["select"] == "val_macro_f1"
                                            and score == best["score"] and va_loss < best["loss"])
        if improved:
            best = {"score": score, "loss": va_loss, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": float((tr_y == tr_p).mean()),
                        "val_loss": va_loss, "val_acc": va["accuracy"], "val_macro_f1": va["macro_f1"]})
        if args.verbose:
            print(f"{log_prefix}epoch {epoch:3d} | train {history[-1]['train_acc']:.3f} ({tr_loss:.4f}) | "
                  f"val acc {va['accuracy']:.3f} f1 {va['macro_f1']:.3f} ({va_loss:.4f}) | "
                  f"{time.time() - t0:.1f}s{' *' if improved else ''}")
        if recipe["patience"] and stale >= recipe["patience"]:
            break
    model.load_state_dict(best["state"])
    return model, {"best_epoch": best["epoch"], "best_val_score": best["score"], "history": history}


def assign_folds(pig_counts, k, seed):
    """Greedy balanced assignment of pigs to k folds by frame count."""
    rng = random.Random(seed)
    pigs = list(pig_counts)
    rng.shuffle(pigs)
    pigs.sort(key=lambda p: -pig_counts[p])
    folds, sizes = [[] for _ in range(k)], [0] * k
    for p in pigs:
        i = sizes.index(min(sizes))
        folds[i].append(p)
        sizes[i] += pig_counts[p]
    return [sorted(f) for f in folds]


def main(args):
    if args.class_set == "posture3":
        args.num_classes = 3
        if args.labels_csv == LABELS_CSV:
            args.labels_csv = LABELS_POSTURE3_CSV
        class_names = POSTURE3_CLASSES
    elif args.class_set == "binary":
        args.num_classes = 2
        class_names = BINARY_CLASSES
        if args.model_prefix == "posture3":
            args.model_prefix = "standing"
    else:
        class_names = POSTURE3_CLASSES if args.num_classes == 3 else BINARY_CLASSES

    recipe = dict(RECIPES[args.recipe])
    if args.epochs:
        recipe["epochs"] = args.epochs
    if args.lr:
        recipe["lr"] = args.lr
    if args.class_weights:
        recipe["class_weights"] = "inverse"
    set_seed(args.seed)
    device = pick_device()
    allowed = parse_allowed_labels(args.allowed_labels)
    n_classes = args.num_classes
    names = [class_names.get(i, str(i)) for i in range(n_classes)]
    print(f"Device: {device} | recipe: {args.recipe} | modality: {args.modality} | backbone: {args.backbone}")
    print(f"Recipe settings: { {k: v for k, v in recipe.items()} }")

    samples = read_label_rows(args.labels_csv, None, allowed, args.data_root)
    t0 = time.time()
    images, found = cache_images(samples, args.modality, IMG_SIZE, crop=not args.no_crop,
                                 use_cropped=args.use_cropped)
    print(f"Loaded {found.sum()}/{len(samples)} labeled frames in {time.time() - t0:.1f}s "
          f"({len(samples) - found.sum()} missing on disk, dropped)")
    keep = np.where(found)[0]
    samples = [samples[i] for i in keep]
    images = images[keep]
    labels = np.array([s["label"] for s in samples])
    pigs = np.array([s["pig_id"] for s in samples])
    if len(samples) == 0:
        print("No images found. Pass --data-root pointing at the folders that contain the labeled timestamp folders.")
        sys.exit(1)

    report = {"recipe": args.recipe, "recipe_settings": recipe, "modality": args.modality,
              "backbone": args.backbone, "labels_csv": args.labels_csv, "seed": args.seed,
              "n_frames": int(len(samples)), "class_counts": dict(Counter(map(int, labels))),
              "git_commit": git_commit()}

    if args.cv:
        pig_counts = Counter(map(int, pigs))
        if len(pig_counts) < 3:
            sys.exit(f"--cv needs at least 3 pigs with labels (found {len(pig_counts)})")
        if args.cv > len(pig_counts):
            print(f"Only {len(pig_counts)} pigs: using {len(pig_counts)} folds instead of {args.cv}")
            args.cv = len(pig_counts)
        folds = assign_folds(pig_counts, args.cv, args.seed)
        print(f"\nGrouped {args.cv}-fold CV by pig: {folds}")
        oof = np.full(len(samples), -1)
        oof_prob = np.zeros((len(samples), n_classes))
        fold_scores = []
        for i in range(args.cv):
            test_p, val_p = set(folds[i]), set(folds[(i + 1) % args.cv])
            te = np.where(np.isin(pigs, list(test_p)))[0]
            va = np.where(np.isin(pigs, list(val_p)))[0]
            tr = np.where(~np.isin(pigs, list(test_p | val_p)))[0]
            if len(te) == 0 or len(va) == 0 or len(tr) == 0:
                sys.exit(f"fold {i + 1}: empty train/val/test split — use fewer folds")
            set_seed(args.seed + i)
            model, info = train_one(images, labels, tr, va, args, recipe, n_classes, device,
                                    log_prefix=f"[fold {i + 1}] ")
            prob = predict(model, images[te], device, recipe["normalize"])
            oof_prob[te] = prob
            oof[te] = prob.argmax(1)
            fm = compute_metrics(labels[te], oof[te], n_classes)
            fold_scores.append(fm)
            print(f"[fold {i + 1}/{args.cv}] test pigs {sorted(test_p)} | best epoch {info['best_epoch']} | "
                  f"acc {fm['accuracy']*100:.2f}% macro-F1 {fm['macro_f1']*100:.2f}% (n={fm['n']})")
        pooled = compute_metrics(labels, oof, n_classes)
        report["cv"] = {
            "k": args.cv, "folds": folds, "pooled": pooled,
            "fold_accuracy": [f["accuracy"] for f in fold_scores],
            "fold_macro_f1": [f["macro_f1"] for f in fold_scores],
        }
        report["oof_predictions"] = [
            {"pig_id": int(s["pig_id"]), "pig_timestamp": s["pig_timestamp"],
             "timestamp_folder": s["timestamp_folder"], "label": int(l), "pred": int(p),
             "probs": [round(float(v), 4) for v in pr]}
            for s, l, p, pr in zip(samples, labels, oof, oof_prob)]
        print(f"\n=== Pooled out-of-fold results ({args.recipe}) ===\n{format_metrics(pooled, names)}")
        accs = np.array(report["cv"]["fold_accuracy"])
        f1s = np.array(report["cv"]["fold_macro_f1"])
        print(f"  per-fold accuracy {accs.mean()*100:.2f} ± {accs.std()*100:.2f}% | "
              f"macro-F1 {f1s.mean()*100:.2f} ± {f1s.std()*100:.2f}%")
    else:
        tr = np.where(np.isin(pigs, TRAIN_PIG_IDS))[0]
        va = np.where(np.isin(pigs, VAL_PIG_IDS))[0]
        te = np.where(np.isin(pigs, TEST_PIG_IDS))[0]
        print(f"\nFixed split: train {len(tr)} / val {len(va)} / test {len(te)} frames")
        model, info = train_one(images, labels, tr, va, args, recipe, n_classes, device)
        prob = predict(model, images[te], device, recipe["normalize"])
        tm = compute_metrics(labels[te], prob.argmax(1), n_classes)
        report["fixed_split"] = {"train_pigs": TRAIN_PIG_IDS, "val_pigs": VAL_PIG_IDS, "test_pigs": TEST_PIG_IDS,
                                 "best_epoch": info["best_epoch"], "test": tm, "history": info["history"]}
        print(f"\n=== Test pigs {TEST_PIG_IDS} ({args.recipe}, best epoch {info['best_epoch']}) ===\n"
              f"{format_metrics(tm, names)}")
        if not args.no_save:
            os.makedirs(MODEL_DIR, exist_ok=True)
            tag = f"{args.model_prefix}_{args.backbone}_{args.modality}"
            model_path = os.path.join(MODEL_DIR, f"{tag}_best.pth")
            torch.save({
                "epoch": info["best_epoch"], "model_state_dict": model.state_dict(),
                "val_acc": info["best_val_score"] if recipe["select"] == "val_acc" else None,
                "val_score": info["best_val_score"], "select": recipe["select"],
                "modality": args.modality, "in_channels": MODALITY_CHANNELS[args.modality],
                "num_classes": n_classes, "class_names": names, "labels_csv": args.labels_csv,
                "backbone": args.backbone, "normalize": recipe["normalize"], "crop": not args.no_crop,
                "img_size": IMG_SIZE, "recipe": args.recipe, "seed": args.seed,
                "test_metrics": tm, "git_commit": report["git_commit"],
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, model_path)
            print(f"Model saved: {model_path}")
            report["model_path"] = model_path

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = args.out_json or os.path.join(
        OUTPUT_DIR, f"train_{args.recipe}_{args.backbone}_{args.modality}{'_cv' + str(args.cv) if args.cv else ''}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"Report: {out}")
    return report


def build_parser():
    p = argparse.ArgumentParser(description="Train posture classifier")
    p.add_argument("--modality", default="ir", choices=list(MODALITY_CHANNELS))
    p.add_argument("--class-set", choices=["binary", "posture3"], default="posture3",
                   help="posture3 (default) = 3 classes + labels/labels_posture3.csv; "
                        "binary = standing vs not (labels/labels.csv)")
    p.add_argument("--num-classes", type=int, default=2, choices=[2, 3])
    p.add_argument("--labels-csv", default=LABELS_CSV)
    p.add_argument("--model-prefix", default="posture3")
    p.add_argument("--allowed-labels", default=None, help="Comma-separated label ids to include, e.g. 0,1,2")
    p.add_argument("--recipe", choices=list(RECIPES), default="improved")
    p.add_argument("--cv", type=int, default=0, help="Grouped K-fold by pig (0 = fixed split)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch", type=int, default=BATCH_SIZE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", action="append", default=None,
                   help="Folder containing the labeled timestamp folders (repeatable)")
    p.add_argument("--no-crop", action="store_true", help="Feed the whole frame (v1.0 used pre-cropped files)")
    p.add_argument("--use-cropped", action="store_true", help="Prefer legacy *_cropped.jpg files if present")
    p.add_argument("--class-weights", action="store_true", help="Force plain inverse-frequency class weights")
    p.add_argument("--backbone", default="mobilenet_v2", choices=BACKBONES)
    p.add_argument("--init-weights", default=None, help="Checkpoint to fine-tune from")
    p.add_argument("--no-pretrained", action="store_true", help="Random init (offline smoke tests only)")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--out-json", default=None)
    p.add_argument("--verbose", action="store_true", help="Print every epoch")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
