#!/usr/bin/env python3
"""Backfill grayscale depth portraits for existing depthmap folders."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from point_cloud import render_stylized_depthmap


def find_bundles(depthmap_root: Path):
    return sorted(depthmap_root.rglob("depthmap_bundle.npz"))


def update_summary(summary_path: Path, render_path: Path) -> None:
    data = {}
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data["depthmap_render"] = str(render_path)
    summary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def render_one(bundle_path: Path, image_size: int, force: bool) -> str:
    out_dir = bundle_path.parent
    render_path = out_dir / "depthmap_render.png"
    summary_path = out_dir / "summary.json"

    if render_path.exists() and not force:
        update_summary(summary_path, render_path)
        return "skipped_existing"

    bundle = np.load(bundle_path, allow_pickle=False)
    depth_mm = bundle["depth_mm"]
    mask = bundle["mask"].astype(bool)
    if not mask.any():
        return "empty"

    render = render_stylized_depthmap(
        depth_mm=depth_mm,
        mask=mask,
        image_size=max(256, image_size),
    )
    cv2.imwrite(str(render_path), render)
    update_summary(summary_path, render_path)
    return "generated"


def main():
    parser = argparse.ArgumentParser(description="Render grayscale depth portraits for existing depthmap outputs")
    parser.add_argument("--depthmap-root", default="depthmap", help="Root directory containing depthmap outputs")
    parser.add_argument("--force", action="store_true", help="Re-render even if depthmap_render.png already exists")
    parser.add_argument("--render-size", type=int, default=768, help="Output size in pixels")
    args = parser.parse_args()

    depthmap_root = Path(args.depthmap_root)
    if not depthmap_root.exists():
        raise FileNotFoundError(f"Depthmap root not found: {depthmap_root}")

    bundles = find_bundles(depthmap_root)
    print(f"Depthmap root: {depthmap_root}")
    print(f"Bundles:       {len(bundles)}")

    counts = {"generated": 0, "skipped_existing": 0, "empty": 0}
    for idx, bundle_path in enumerate(bundles, 1):
        rel = bundle_path.relative_to(depthmap_root)
        status = render_one(
            bundle_path,
            image_size=args.render_size,
            force=args.force,
        )
        counts[status] = counts.get(status, 0) + 1
        print(f"[{idx}/{len(bundles)}] {rel} -> {status}")

    print()
    print(
        "Done. "
        f"generated={counts.get('generated', 0)} "
        f"skipped_existing={counts.get('skipped_existing', 0)} "
        f"empty={counts.get('empty', 0)}"
    )


if __name__ == "__main__":
    main()
