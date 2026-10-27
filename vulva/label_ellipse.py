#!/usr/bin/env python3
"""Fast vulva labelling: three clicks draw an ellipse, one key marks "not visible".

For each frame of a label set (export_label_set.py) the IR image is shown,
cropped to the stall and contrast-boosted, next to two pictures made from the
pixel-aligned depth (depth_views.py): "3D shading" (the vulva is a lit oval
lump; flat feces and dark marks vanish) and "height" (mm above the sow's own
body surface, red = sticks out). Click on any of the three; the views zoom
together. --no-depth shows the IR image alone.

If the set has colour images (export_label_set.py --rgb / --add-rgb), run
register_rgb.py on it once: the colour image is then matched onto the IR pixel
grid and shown as the second panel, clickable and zooming with the others. It
shows the vulva's outer edge far better than the IR. Otherwise the aligned RGB
(if the camera saved one) is that panel. Plain RGB from the separate colour
camera is not pixel-aligned: it is shown for reference, cropped to roughly the
same area with an approximate map (--rgb-map, fitted by hand on the stall bars
and anus of three Feb 6-7 frames; off by up to ~30 IR pixels because of
parallax), and it zooms with the others. --no-rgb hides it.

    click 1  top end of the vulva       (the view then zooms in below it; --zoom 0 turns that off)
    click 2  bottom end
    click 3  one side, at the widest point
    Enter    save the ellipse            n  not visible (turned, lying, blur, feces)
    t        tail covers part of it      (saved as not visible, note "tail")
    u        undo the last click         s  skip for now
    q        save and quit (run again to continue where you stopped)

Outline rule (from 2026-10-27): the whole vulva to its outer edge, lips and dark centre
together, not just the dark centre. If the tail covers any part of it, press t: a partly
hidden vulva gives a wrong size, so it counts as not visible.

The vulva is close to an ellipse, so three clicks give a usable training mask
in a few seconds, and the clicked length and width are kept as image-space
measurements. Results go next to the frames:
    <set>/labels.csv          one row per labelled frame (paths relative to <set>)
    <set>/masks/*.png         filled ellipse masks, 640 x 480

Model-assisted mode (--model): a trained checkpoint draws its own ellipse
first. Enter accepts it, n agrees there is no vulva, a click starts over from
your own top click, u removes the suggested side point so you can re-click it.
--hard-first shows the frames the model is least sure about first, which are
the ones that teach the next model the most. The "how" column records whether
a label was clicked by hand or accepted from the model.

Review mode (--review TAG): goes through frames that are already labelled and shows the
saved ellipse (white, dashed). Enter keeps it, a click redraws it, n / t change it to not
visible. Each reviewed row gets review = "TAG kept" / "TAG redrawn" / "TAG not visible" /
"TAG tail", so a review can stop and resume. Only frames labelled visible are reviewed
unless --review-absent is given.

Usage:
    python vulva/label_ellipse.py ~/Downloads/paal_exchange/vulva_label_set
    python vulva/label_ellipse.py ~/Downloads/paal_exchange/vulva_label_set --model models/vulva_seg_r1.pth --hard-first
    python vulva/label_ellipse.py ~/Downloads/paal_exchange/vulva_label_set --review r6
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import cv2
import numpy as np

FIELDS = ["ir_path", "mask_path", "label", "pig_id", "timestamp_folder", "pig_timestamp",
          "cx", "cy", "half_length_px", "half_width_px", "angle_deg", "how", "note", "review"]
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


def clicks_from_params(cx, cy, half_len, half_w, angle_deg):
    """Saved ellipse -> the (top, bottom, side) clicks that redraw it with ellipse_from_clicks."""
    t = math.radians(angle_deg)
    ux, uy = math.cos(t), math.sin(t)
    p1, p2 = (cx - ux * half_len, cy - uy * half_len), (cx + ux * half_len, cy + uy * half_len)
    top, bottom = (p1, p2) if p1[1] <= p2[1] else (p2, p1)
    return [top, bottom, (cx - uy * half_w, cy + ux * half_w)]


def saved_clicks(row):
    try:
        return clicks_from_params(*(float(row[k]) for k in ("cx", "cy", "half_length_px", "half_width_px", "angle_deg")))
    except (KeyError, TypeError, ValueError):
        return None


def clicks_from_mask(mask, min_area=30):
    """Largest blob of a mask -> the (top, bottom, side) clicks of its fitted ellipse,
    so ellipse_from_clicks(*clicks) gives the same ellipse back. None if no usable blob."""
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5 or cv2.contourArea(c) < min_area:
        return None
    (cx, cy), (w, h), ang = cv2.fitEllipse(c)
    t = math.radians(ang)
    u = (math.cos(t), math.sin(t))        # direction of the 'w' axis
    v = (-math.sin(t), math.cos(t))       # direction of the 'h' axis
    (major, a), (minor, b) = ((u, w / 2), (v, h / 2)) if w >= h else ((v, h / 2), (u, w / 2))
    p1 = (cx - major[0] * a, cy - major[1] * a)
    p2 = (cx + major[0] * a, cy + major[1] * a)
    top, bottom = (p1, p2) if p1[1] <= p2[1] else (p2, p1)
    side = (cx + minor[0] * b, cy + minor[1] * b)
    return [top, bottom, side]


class Suggester:
    """Wraps a vulva_seg checkpoint: frame path -> (clicks or None, max probability)."""

    def __init__(self, ckpt_path):
        import torch
        from seg_model import load_checkpoint, read_input
        self.read_input = read_input
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model, ck = load_checkpoint(ckpt_path, self.device)
        self.thr, self.min_area = ck.get("threshold", 0.5), ck.get("min_area", 25)
        self.inputs = ck.get("inputs", "ir")
        self.cache = {}

    def __call__(self, paths):
        from seg_model import predict
        need = [p for p in paths if p not in self.cache]
        for i in range(0, len(need), 16):
            chunk = need[i:i + 16]
            grays = [self.read_input(p, None, self.inputs) for p in chunk]
            ok = [j for j, g in enumerate(grays) if g is not None]
            res = predict(self.model, [grays[j] for j in ok], self.device, self.thr, self.min_area) if ok else []
            for p in chunk:
                self.cache[p] = (None, 0.0)
            for j, (m, pmax, *_rest) in zip(ok, res):
                self.cache[chunk[j]] = (clicks_from_mask(m) if m.any() else None, pmax)
        return [self.cache[p] for p in paths]


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
    ap.add_argument("--model", default="", help="vulva_seg checkpoint: show its ellipse as a suggestion")
    ap.add_argument("--hard-first", action="store_true",
                    help="with --model: least certain frames first (max probability closest to the threshold)")
    ap.add_argument("--no-depth", action="store_true", help="IR image only (no 3D shading / height panels)")
    ap.add_argument("--no-rgb", action="store_true", help="hide the colour image even if the set has one")
    ap.add_argument("--rgb-map", default="1.69,30,-12",
                    help="approximate ToF-pixel -> RGB-pixel map for 1280x800 RGB: scale,dx,dy (used to show "
                         "the same area of the unaligned RGB image)")
    ap.add_argument("--height-span", type=float, default=8.0, help="height colour scale, +/- mm")
    ap.add_argument("--review", default="", metavar="TAG",
                    help="review frames already labelled (saved ellipse shown; Enter keeps, click redraws)")
    ap.add_argument("--review-absent", action="store_true", help="with --review: also frames labelled not visible")
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
    review = args.review.strip()
    if review:
        want = ("present", "absent") if args.review_absent else ("present",)
        todo = [fr for fr in frames if fr["ir_path"] in done and done[fr["ir_path"]]["label"] in want
                and not (done[fr["ir_path"]].get("review") or "").startswith(review + " ")]
        n_rev = sum((r.get("review") or "").startswith(review + " ") for r in done.values())
        print(f"Review {review}: {len(todo)} to go, {n_rev} already reviewed")
    else:
        todo = [fr for fr in frames if fr["ir_path"] not in done]
        print(f"{len(frames)} frames, {len(done)} already labelled, {len(todo)} to go")
    if not todo:
        return

    suggest = None
    if args.model and not review:
        suggest = Suggester(os.path.expanduser(args.model))
        print(f"Model suggestions from {args.model} on {suggest.device}; predicting {len(todo)} frames ...")
        preds = suggest([os.path.join(set_dir, fr["ir_path"]) for fr in todo])
        n_yes = sum(c is not None for c, _ in preds)
        print(f"  model sees a vulva in {n_yes} of {len(todo)} (threshold {suggest.thr:.2f})")
        if args.hard_first:
            order = sorted(range(len(todo)), key=lambda k: abs(preds[k][1] - suggest.thr))
            todo = [todo[k] for k in order]

    state = {"i": 0, "clicks": [], "quit": False, "suggested": False}
    use_depth = not args.no_depth
    if use_depth:
        from depth_views import views_for
    # colour: aligned RGB shares the IR pixel grid (clickable, zooms with the others);
    # plain RGB is a separate camera, so it is shown whole, for reference only
    rgb_mode = None
    if not args.no_rgb:
        if any(fr.get("rgb_aligned_path") for fr in frames):
            rgb_mode = "aligned"
        elif any(fr.get("rgb_path") for fr in frames):
            rgb_mode = "reference"
    n_linked = 1 + 2 * use_depth + (rgb_mode == "aligned")
    n_total = n_linked + (rgb_mode == "reference")
    multi = n_total > 1
    fig = plt.figure(figsize=(3.55 * n_total + 0.3, 7.2) if multi else (8, 9))
    axes = []
    for k in range(n_linked):
        axes.append(fig.add_subplot(1, n_total, k + 1, sharex=axes[0] if axes else None,
                                    sharey=axes[0] if axes else None))
    ref_ax = fig.add_subplot(1, n_total, n_total) if rgb_mode == "reference" else None
    if multi:
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.86, wspace=0.03)
    ax = axes[0]
    fig.canvas.manager.set_window_title("PAAL vulva labelling")
    view_cache = {}

    rgb_s, rgb_dx, rgb_dy = (float(v) for v in args.rgb_map.split(","))

    def to_rgb_box(x0, x1, y0, y1, w, h, pad=0.25):
        """ToF-pixel box -> RGB-pixel limits (the RGB camera is not calibrated, so the box is
        widened by `pad` on each side to keep the same area in view)."""
        kx, ky = w / 1280.0, h / 800.0
        X0, X1 = (rgb_s * x0 + rgb_dx) * kx, (rgb_s * x1 + rgb_dx) * kx
        Y0, Y1 = (rgb_s * y0 + rgb_dy) * ky, (rgb_s * y1 + rgb_dy) * ky
        px, py = pad * (X1 - X0), pad * (Y1 - Y0)
        return max(0, X0 - px), min(w, X1 + px), min(h, Y1 + py), max(0, Y0 - py)

    def read_rgb(rel, aligned):
        if not rel:
            return None
        im = cv2.imread(os.path.join(set_dir, rel), cv2.IMREAD_COLOR)
        if im is None:
            return None
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if aligned and im.shape[:2] != (FULL_H, FULL_W):
            im = cv2.resize(im, (FULL_W, FULL_H))
        return im

    def current():
        return todo[state["i"]]

    def depth_views(fr):
        key = fr["ir_path"]
        if key not in view_cache:
            view_cache.clear()
            dp = fr.get("depth_path", "")
            view_cache[key] = views_for(os.path.join(set_dir, dp), args.height_span) if dp else None
        return view_cache[key]

    def reset_view():
        ax.set_xlim(CROP[0], CROP[2])
        ax.set_ylim(CROP[3], CROP[1])

    def load_suggestion():
        state["clicks"], state["suggested"], state["hint"] = [], False, ""
        if review:
            row = done[current()["ir_path"]]
            clicks = saved_clicks(row) if row["label"] == "present" else None
            if clicks:
                state["clicks"], state["suggested"] = clicks, True
                state["hint"] = "saved outline: Enter keep, click to redraw, t tail covers part, n not visible"
            else:
                state["hint"] = "saved: not visible   n keep, t tail covers part, or click to draw it"
            return
        if suggest is None:
            return
        clicks, pmax = suggest([os.path.join(set_dir, current()["ir_path"])])[0]
        if clicks:
            state["clicks"], state["suggested"] = list(clicks), True
            state["hint"] = f"model: vulva (p={pmax:.2f})  Enter accept, click to redo"
        else:
            state["hint"] = f"model: not visible (p={pmax:.2f})  n agree, or click it"

    def draw():
        fr = current()
        img = display_image(os.path.join(set_dir, fr["ir_path"]))
        if img is None:
            img = np.zeros((FULL_H, FULL_W), np.uint8)
        panels = [("IR", img)]
        if use_depth:
            v = depth_views(fr)
            if v is None:
                blank = np.full((FULL_H, FULL_W, 3), 30, np.uint8)
                panels += [("3D shading (no depth)", blank), ("height (no depth)", blank)]
            else:
                panels += [("3D shading", v[0]), (f"height, mm (+/-{args.height_span:g}, red = sticks out)", v[1])]
        if rgb_mode == "aligned":                 # colour next to the IR: it shows the vulva edge best
            rgb = read_rgb(fr.get("rgb_aligned_path", ""), True)
            panels.insert(1, ("colour (matched to IR)" if rgb is not None else "colour (none for this frame)",
                              rgb if rgb is not None else np.full((FULL_H, FULL_W, 3), 30, np.uint8)))
        e = ellipse_from_clicks(*state["clicks"]) if len(state["clicks"]) == 3 else None
        hand = not state["suggested"]
        for a, (name, im) in zip(axes, panels):
            a.clear()
            if im.ndim == 2:
                a.imshow(im, cmap="gray", vmin=0, vmax=255)
            else:
                a.imshow(im)
            for k, (x, y) in enumerate(state["clicks"]):
                a.plot(x, y, "+", color=("#ff3df2", "#ff3df2", "#ffffff")[k], ms=12, mew=1.8)
            if e:
                a.add_patch(Ellipse((e[0], e[1]), 2 * e[2], 2 * e[3], angle=e[4], fill=False,
                                    ec="#ff3df2" if hand else "#ffffff", lw=1.6, ls="-" if hand else "--"))
            if multi:
                a.set_title(name, fontsize=9)
            a.set_xticks([])
            a.set_yticks([])
        if ref_ax is not None:
            ref_ax.clear()
            rgb = read_rgb(fr.get("rgb_path", ""), False)
            if rgb is not None:
                ref_ax.imshow(rgb)
                ref_ax.set_title("RGB (separate camera, approx. same area)", fontsize=9)
                h, w = rgb.shape[:2]
                if state["clicks"] and args.zoom:
                    x, y = state["clicks"][0]
                    z = args.zoom
                    bx = to_rgb_box(x - z, x + z, y - 0.4 * z, y + 2.0 * z, w, h, pad=0.3)
                else:
                    bx = to_rgb_box(CROP[0], CROP[2], CROP[1], CROP[3], w, h, pad=0.05)
                ref_ax.set_xlim(bx[0], bx[1])
                ref_ax.set_ylim(bx[2], bx[3])
            else:
                ref_ax.set_title("RGB (none for this frame)", fontsize=9)
            ref_ax.set_xticks([])
            ref_ax.set_yticks([])
        hint = state.get("hint", "")
        where = f"review {state['i'] + 1}/{len(todo)}" if review else f"{len(done) + 1}/{len(frames)}"
        title = (f"pig {fr['pig_id']}   {fr['pig_timestamp']}   [{where}]\n"
                 + (hint + "\n" if hint else "")
                 + "click top, bottom, side   Enter save   n not visible   t tail covers part   u undo   s skip   q quit")
        if multi:
            fig.suptitle(title, fontsize=10)
        else:
            ax.set_title(title, fontsize=10)
        if state["clicks"] and args.zoom:
            x, y = state["clicks"][0]          # zoom around the top end, mostly below it
            z = args.zoom
            ax.set_xlim(x - z, x + z)
            ax.set_ylim(y + 2.0 * z, y - 0.4 * z)
        else:
            reset_view()
        fig.canvas.draw_idle()

    def record(label, params=None, note=""):
        fr = current()
        how = "hand"
        old = done.get(fr["ir_path"], {})
        if review:
            kept = (label == "present" and state["suggested"]) or (label == "absent" and old.get("label") == "absent"
                                                                    and note == (old.get("note") or ""))
            how = old.get("how", "") if kept else "hand"
        elif suggest is not None:
            model_said = "present" if (state["suggested"] or state.get("hint", "").startswith("model: vulva")) \
                else "absent"
            if label == "present":
                how = "accepted" if state["suggested"] else ("corrected" if model_said == "present" else "hand")
            else:
                how = "accepted" if model_said == "absent" else "corrected"
        row = {"ir_path": fr["ir_path"], "mask_path": "", "label": label, "pig_id": fr["pig_id"],
               "timestamp_folder": fr["timestamp_folder"], "pig_timestamp": fr["pig_timestamp"], "how": how,
               "note": note, "review": old.get("review", "")}
        if review:
            what = ("kept" if state["suggested"] else "redrawn") if label == "present" else (note or "not visible")
            row["review"] = f"{review} {what}"
        if params and review and state["suggested"] and old.get("mask_path"):
            # kept as it was: same mask file and numbers, nothing rewritten
            row.update({k: old.get(k, "") for k in ("mask_path", "cx", "cy", "half_length_px", "half_width_px",
                                                     "angle_deg")})
            params = None
        if params:
            # a review writes its own mask file, so earlier label snapshots keep pointing at their masks
            tag = f"_{review}" if review else ""
            name = f"{fr['timestamp_folder']}_pig{fr['pig_id']}_{fr['pig_timestamp']}{tag}.png"
            rel = os.path.join("masks", name).replace("\\", "/")
            cv2.imwrite(os.path.join(set_dir, rel), ellipse_mask(params))
            row.update({"mask_path": rel, "cx": round(params[0], 1), "cy": round(params[1], 1),
                        "half_length_px": round(params[2], 2), "half_width_px": round(params[3], 2),
                        "angle_deg": round(params[4], 1)})
        done[fr["ir_path"]] = row
        save_rows(out_csv, list(done.values()))

    def advance():
        state["i"] += 1
        if state["i"] >= len(todo):
            print("All frames labelled.")
            plt.close(fig)
            return
        load_suggestion()
        draw()

    def on_click(ev):
        if ev.inaxes not in axes or ev.button != 1 or ev.xdata is None:
            return
        if fig.canvas.manager.toolbar is not None and getattr(fig.canvas.manager.toolbar, "mode", ""):
            return            # zoom / pan tool active
        if state["suggested"]:                 # first click on a suggestion: start over by hand
            state["clicks"], state["suggested"] = [], False
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
        elif k == "t":
            record("absent", note="tail")
            advance()
        elif k == "u" and state["clicks"]:
            state["clicks"].pop()
            state["suggested"] = False
            draw()
        elif k == "s":
            advance()
        elif k == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    load_suggestion()
    draw()
    plt.show()
    pres = sum(r["label"] == "present" for r in done.values())
    print(f"Saved {len(done)} labels ({pres} present, {len(done) - pres} not visible) -> {out_csv}")
    if review:
        tags = [(r.get("review") or "")[len(review) + 1:] for r in done.values()
                if (r.get("review") or "").startswith(review + " ")]
        print(f"  review {review}: " + ", ".join(f"{tags.count(t)} {t}" for t in
                                                  ("kept", "redrawn", "tail", "not visible") if tags.count(t)))
    if suggest is not None:
        hows = [r.get("how", "") for r in done.values() if r.get("how") in ("accepted", "corrected")]
        if hows:
            print(f"  model suggestions this far: {hows.count('accepted')} accepted, {hows.count('corrected')} corrected")


if __name__ == "__main__":
    main()
