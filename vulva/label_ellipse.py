#!/usr/bin/env python3
"""Fast vulva labelling: three clicks draw an ellipse, one key marks "not visible".

For each frame of a label set (export_label_set.py) the IR image is shown,
cropped to the stall and contrast-boosted:

    click 1  top end of the vulva       (the view then zooms in below it; --zoom 0 turns that off)
    click 2  bottom end
    click 3  one side, at the widest point
    Enter    save the ellipse            n  not visible (tail, lying, blur)
    u        undo the last click         s  skip for now
    q        save and quit (run again to continue where you stopped)

The vulva is close to an ellipse, so three clicks give a usable training mask
in a few seconds, and the clicked length and width are kept as image-space
measurements. Results go next to the frames:
    <set>/labels.csv          one row per labelled frame (paths relative to <set>)
    <set>/masks/*.png         filled ellipse masks, 640 x 480

Usage:
    python vulva/label_ellipse.py ~/Downloads/paal_exchange/vulva_label_set
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import cv2
import numpy as np

FIELDS = ["ir_path", "mask_path", "label", "pig_id", "timestamp_folder", "pig_timestamp",
          "cx", "cy", "half_length_px", "half_width_px", "angle_deg"]
FULL_W, FULL_H = 640, 480
CROP = (120, 30, 500, 480)


def ellipse_from_clicks(top, bottom, side):
    """Three (x, y) clicks -> (cx, cy, half_length, half_width, angle_deg) in pixels.
    half_width is the side click's distance from the top-bottom line."""
    (x1, y1), (x2, y2), (x3, y3) = top, bottom, side
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    half_len = math.hypot(dx, dy) / 2
    if half_len == 0:
        return None
    half_w = abs(dy * (x3 - x1) - dx * (y3 - y1)) / (2 * half_len)
    return cx, cy, half_len, max(half_w, 1.0), math.degrees(math.atan2(dy, dx))


def ellipse_mask(params, shape=(FULL_H, FULL_W)):
    cx, cy, a, b, ang = params
    m = np.zeros(shape, np.uint8)
    cv2.ellipse(m, (int(round(cx)), int(round(cy))), (int(round(a)), int(round(b))), ang, 0, 360, 255, -1)
    return m


def load_done(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {r["ir_path"]: r for r in csv.DictReader(f)}


def save_rows(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def display_image(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    if g.shape != (FULL_H, FULL_W):
        g = cv2.resize(g, (FULL_W, FULL_H))
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set_dir", help="folder from export_label_set.py")
    ap.add_argument("--zoom", type=int, default=100, help="half-width of the zoomed view after the first click, px (0 = no zoom)")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    plt.rcParams["keymap.save"] = []      # 's' and 'q' are ours
    plt.rcParams["keymap.quit"] = []

    set_dir = os.path.abspath(os.path.expanduser(args.set_dir))
    with open(os.path.join(set_dir, "label_set.csv"), newline="", encoding="utf-8-sig") as f:
        frames = list(csv.DictReader(f))
    out_csv = os.path.join(set_dir, "labels.csv")
    done = load_done(out_csv)
    os.makedirs(os.path.join(set_dir, "masks"), exist_ok=True)
    todo = [fr for fr in frames if fr["ir_path"] not in done]
    print(f"{len(frames)} frames, {len(done)} already labelled, {len(todo)} to go")
    if not todo:
        return

    state = {"i": 0, "clicks": [], "quit": False}
    fig, ax = plt.subplots(figsize=(8, 9))
    fig.canvas.manager.set_window_title("PAAL vulva labelling")

    def current():
        return todo[state["i"]]

    def reset_view():
        ax.set_xlim(CROP[0], CROP[2])
        ax.set_ylim(CROP[3], CROP[1])

    def draw():
        ax.clear()
        fr = current()
        img = display_image(os.path.join(set_dir, fr["ir_path"]))
        if img is None:
            img = np.zeros((FULL_H, FULL_W), np.uint8)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        for k, (x, y) in enumerate(state["clicks"]):
            ax.plot(x, y, "+", color=("#00e5ff", "#00e5ff", "#ffd400")[k], ms=12, mew=1.5)
        if len(state["clicks"]) == 3:
            e = ellipse_from_clicks(*state["clicks"])
            if e:
                ax.add_patch(Ellipse((e[0], e[1]), 2 * e[2], 2 * e[3], angle=e[4], fill=False,
                                     ec="#00e5ff", lw=1.5))
        n_done = len(done)
        ax.set_title(f"pig {fr['pig_id']}   {fr['pig_timestamp']}   [{n_done + 1}/{len(frames)}]\n"
                     "click top, bottom, side   Enter save   n not visible   u undo   s skip   q quit",
                     fontsize=10)
        if state["clicks"] and args.zoom:
            x, y = state["clicks"][0]          # zoom around the top end, mostly below it
            z = args.zoom
            ax.set_xlim(x - z, x + z)
            ax.set_ylim(y + 2.0 * z, y - 0.4 * z)
        else:
            reset_view()
        ax.set_xticks([])
        ax.set_yticks([])
        fig.canvas.draw_idle()

    def record(label, params=None):
        fr = current()
        row = {"ir_path": fr["ir_path"], "mask_path": "", "label": label, "pig_id": fr["pig_id"],
               "timestamp_folder": fr["timestamp_folder"], "pig_timestamp": fr["pig_timestamp"]}
        if params:
            name = f"{fr['timestamp_folder']}_pig{fr['pig_id']}_{fr['pig_timestamp']}.png"
            rel = os.path.join("masks", name).replace("\\", "/")
            cv2.imwrite(os.path.join(set_dir, rel), ellipse_mask(params))
            row.update({"mask_path": rel, "cx": round(params[0], 1), "cy": round(params[1], 1),
                        "half_length_px": round(params[2], 2), "half_width_px": round(params[3], 2),
                        "angle_deg": round(params[4], 1)})
        done[fr["ir_path"]] = row
        save_rows(out_csv, list(done.values()))

    def advance():
        state["clicks"] = []
        state["i"] += 1
        if state["i"] >= len(todo):
            print("All frames labelled.")
            plt.close(fig)
            return
        draw()

    def on_click(ev):
        if ev.inaxes != ax or ev.button != 1 or ev.xdata is None:
            return
        if fig.canvas.manager.toolbar is not None and getattr(fig.canvas.manager.toolbar, "mode", ""):
            return            # zoom / pan tool active
        if len(state["clicks"]) < 3:
            state["clicks"].append((ev.xdata, ev.ydata))
            draw()

    def on_key(ev):
        k = (ev.key or "").lower()
        if k == "enter" and len(state["clicks"]) == 3:
            e = ellipse_from_clicks(*state["clicks"])
            if e:
                record("present", e)
                advance()
        elif k == "n":
            record("absent")
            advance()
        elif k == "u" and state["clicks"]:
            state["clicks"].pop()
            draw()
        elif k == "s":
            advance()
        elif k == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show()
    pres = sum(r["label"] == "present" for r in done.values())
    print(f"Saved {len(done)} labels ({pres} present, {len(done) - pres} not visible) -> {out_csv}")


if __name__ == "__main__":
    main()
