"""Train posture classifier."""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    MODEL_DIR,
    NUM_EPOCHS,
    OUTPUT_DIR,
    LABELS_CSV,
)
from data_loader import get_dataloaders
from models import BACKBONES, build_model


def parse_allowed_labels(raw):
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip() != ""]


def run_epoch(model, loader, criterion, device, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.set_grad_enabled(train_mode):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if train_mode:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if train_mode:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


def main(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, in_channels = get_dataloaders(
        modality=args.modality,
        batch_size=args.batch,
        labels_csv=args.labels_csv,
        allowed_labels=parse_allowed_labels(args.allowed_labels),
        use_cropped=args.use_cropped,
    )

    model = build_model(modality=args.modality, num_classes=args.num_classes, pretrained=True, backbone=args.backbone).to(device)

    if args.init_weights:
        ckpt = torch.load(args.init_weights, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        src = ckpt.get("source", "unknown")
        src_acc = ckpt.get("val_acc", 0)
        print(f"Initialized weights from: {args.init_weights} (source={src}, val_acc={src_acc:.4f})")

    # Compute class weights from training set for imbalanced data
    if args.class_weights:
        label_counts = {}
        for _, lbl in train_loader.dataset:
            lbl = lbl.item() if hasattr(lbl, 'item') else int(lbl)
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        n_samples = sum(label_counts.values())
        n_classes = len(label_counts)
        weights = torch.tensor(
            [n_samples / (n_classes * label_counts.get(i, 1)) for i in range(args.num_classes)],
            dtype=torch.float32,
        ).to(device)
        print(f"Class weights: {weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_tag = f"{args.model_prefix}_{args.backbone}_{args.modality}"
    model_path = os.path.join(MODEL_DIR, f"{model_tag}_best.pth")
    hist_path = os.path.join(OUTPUT_DIR, f"history_{model_tag}.json")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val = -1.0

    print(f"\nTraining modality={args.modality}, epochs={args.epochs}, lr={args.lr}\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        saved = ""
        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "modality": args.modality,
                    "in_channels": in_channels,
                    "num_classes": args.num_classes,
                    "labels_csv": args.labels_csv,
                    "backbone": args.backbone,
                },
                model_path,
            )
            saved = " ★ saved"

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train {train_acc:.3f} ({train_loss:.4f}) | "
            f"Val {val_acc:.3f} ({val_loss:.4f}) | "
            f"{time.time() - t0:.1f}s{saved}"
        )

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device)

    history["test_loss"] = test_loss
    history["test_acc"] = test_acc
    history["num_classes"] = args.num_classes
    history["labels_csv"] = args.labels_csv
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete")
    print(f"Best val accuracy: {best_val:.4f}")
    print(f"Test accuracy:     {test_acc:.4f}")
    print(f"Model:   {model_path}")
    print(f"History: {hist_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train posture classifier")
    parser.add_argument("--modality", default="rgb", choices=["rgb", "ir", "depth", "rgb_depth", "rgb_ir", "all"])
    parser.add_argument("--num-classes", type=int, default=2, choices=[2, 3])
    parser.add_argument("--labels-csv", default=LABELS_CSV)
    parser.add_argument("--model-prefix", default="standing")
    parser.add_argument("--allowed-labels", default=None,
                        help="Comma-separated label ids to include, e.g. 0,1,2")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--use-cropped", action="store_true", help="Use cropped images")
    parser.add_argument("--class-weights", action="store_true", help="Use inverse-frequency class weights")
    parser.add_argument("--backbone", default="mobilenet_v2", choices=BACKBONES)
    parser.add_argument("--init-weights", default=None, help="Path to pretrained checkpoint for fine-tuning")
    main(parser.parse_args())
