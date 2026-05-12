"""Visual RGB/IR/Depth alignment check from saved JPG files."""

import argparse
import os

import cv2
import numpy as np

from config import DATA_DIR


def find_sample(folder_hint=None, pig_id=0):
    folders = sorted(d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)))
    if folder_hint:
        folders = [f for f in folders if folder_hint in f] + folders

    for folder in folders:
        path = os.path.join(DATA_DIR, folder)
        files = os.listdir(path)
        rgb = [f for f in files if f.startswith(f"pig{pig_id}_rgb_") and f.endswith(".jpg")]
        ir = [f for f in files if f.startswith(f"pig{pig_id}_ir_vis_") and f.endswith(".jpg")]
        depth = [f for f in files if f.startswith(f"pig{pig_id}_depth_vis_") and f.endswith(".jpg")]
        if rgb and ir and depth:
            return {
                "folder": folder,
                "rgb": os.path.join(path, rgb[0]),
                "ir": os.path.join(path, ir[0]),
                "depth": os.path.join(path, depth[0]),
            }
    return None


def edges(gray):
    return cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.5), 50, 150)


def fit_for_display(img, max_w=1200, max_h=700):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(img, None, fx=s, fy=s) if s < 1.0 else img


def main(folder_hint=None, pig_id=0):
    sample = find_sample(folder_hint, pig_id)
    if not sample:
        print("No sample found.")
        return

    rgb = cv2.imread(sample["rgb"])
    ir = cv2.imread(sample["ir"])
    depth = cv2.imread(sample["depth"])
    if rgb is None or ir is None or depth is None:
        print("Failed to load sample images.")
        return

    print(f"Sample: pig{pig_id} in {sample['folder']}")
    print(f"RGB: {rgb.shape}, IR: {ir.shape}, Depth: {depth.shape}")

    shapes = [rgb.shape[:2], ir.shape[:2], depth.shape[:2]]
    if len(set(shapes)) > 1:
        h = min(s[0] for s in shapes)
        w = min(s[1] for s in shapes)
        rgb, ir, depth = cv2.resize(rgb, (w, h)), cv2.resize(ir, (w, h)), cv2.resize(depth, (w, h))
        print(f"Resized to common shape: {(h, w)}")

    rgb_e = edges(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY))
    ir_e = edges(cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY))
    depth_e = edges(cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY))

    top = np.hstack([rgb, ir, depth])
    cv2.putText(top, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    w = rgb.shape[1]
    cv2.putText(top, "IR", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(top, "DEPTH", (2 * w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    ov = np.zeros((*rgb_e.shape, 3), dtype=np.uint8)
    ov[:, :, 2] = rgb_e
    ov[:, :, 1] = ir_e
    ov[:, :, 0] = depth_e
    cv2.putText(ov, "RED=RGB GREEN=IR BLUE=Depth", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    rgb_depth = rgb.copy()
    rgb_depth[depth_e > 0] = [0, 255, 0]
    cv2.putText(rgb_depth, "RGB + depth edges", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Side by Side - RGB | IR | Depth", fit_for_display(top))
    cv2.imshow("Edge Overlay - Alignment Check", fit_for_display(ov, max_w=900, max_h=650))
    cv2.imshow("RGB + Depth Edges Overlay", fit_for_display(rgb_depth, max_w=900, max_h=650))
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Visual alignment check")
    p.add_argument("--folder", type=str, default=None)
    p.add_argument("--pig", type=int, default=0)
    a = p.parse_args()
    main(a.folder, a.pig)
