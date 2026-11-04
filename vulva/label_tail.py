#!/usr/bin/env python3
r"""Outline tails on labelled frames, for the 3-class model (background / vulva / tail).

Tails come in every shape (hanging, curled, swinging to one side, covering part of the vulva),
so the tool has three ways to draw, all on the same mask and freely mixed:

  stroke   (default) click along the middle of the tail; each click extends a thick line
           (round ends) of the current width, so a curved tail is a few clicks along its curve.
           [ / ] or the mouse wheel change the width. n starts a new stroke (a second piece,
           a curl that doubles back). One click alone paints a round blob (a tail tip).
  polygon  p toggles: clicks are corners of an outline; n (or p again) closes and fills it.
           Best for a wide tail base or an odd shape.
  brush    b toggles: drag with the left button to paint, with the right button to erase,
           at the current width. For touching up edges.
  SAM      (optional, --sam sam2.1_t.pt, needs `pip install ultralytics`, which downloads the
           model weights the first time) a outlines the tail from the clicks of the current
           stroke as hints; right-click adds a "not tail" hint. Fix the result with the brush.

    Enter  save the tail mask        x  no tail visible in this frame (saved as tail = 0)
    u      undo the last step        c  clear this frame
    s      skip for now              q  save and quit (run again to continue)

The IR image and the colour image (matched onto the IR grid) are shown side by side; draw on
either. The saved vulva ellipse is drawn in green for reference. Frames labelled "tail" (key t
in label_ellipse.py) come first, then frames with a visible vulva (tails often hang next to it),
then the rest. A frame you never open stays "unknown" and the 3-class training does not use it
to learn tail versus background, so labelling only part of a set is fine.

Results go next to the labels CSV:
    tail_labels.csv     one row per frame: tail (1/0), tail_mask_path, covers_vulva, how, width_px
    tail_masks/*.png    640 x 480, 255 = tail (same pixel grid as the IR and the vulva masks)

Usage (CAFNR PC):
    python vulva\label_tail.py C:\Users\aakdbx\vulva_r9_set\labels.csv
    python vulva\label_tail.py C:\Users\aakdbx\vulva_jan_set\labels.csv
    python vulva\label_tail.py C:\Users\aakdbx\Downloads\paal_labels_feb\vulva_label_set\labels_r7.csv --frames-root "\\umad.umsystem.edu\rde\zhoujianf-lab\Pig project\Sow Data\Pictures_OAK"
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FULL_W, FULL_H = 640, 480
CROP = (120, 30, 500, 480)
TAIL_FIELDS = ["ir_path", "pig_id", "timestamp_folder", "pig_timestamp", "tail", "tail_mask_path",
               "covers_vulva", "how", "width_px", "date"]


# ---------------------------------------------------------------------------------------------
# mask drawing (kept free of any GUI so it can be tested)

def stroke_mask(points, width, shape=(FULL_H, FULL_W)):
    """A thick line through the points with round ends and joints; one point = a round blob."""
    m = np.zeros(shape, np.uint8)
    if not points:
        return m
    w = max(int(round(width)), 1)
    pts = [(int(round(x)), int(round(y))) for x, y in points]
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(m, a, b, 255, w, cv2.LINE_8)
    for p in pts:
        cv2.circle(m, p, max(w // 2, 1), 255, -1)
    return m


def polygon_mask(points, shape=(FULL_H, FULL_W)):
    m = np.zeros(shape, np.uint8)
    if len(points) >= 3:
        cv2.fillPoly(m, [np.round(np.array(points)).astype(np.int32)], 255)
    return m


class TailCanvas:
    """The tail mask of one frame, built from strokes, polygons, brush dabs and SAM layers,
    with undo. Everything is in full-frame IR pixels (640 x 480)."""

    def __init__(self, width=12, shape=(FULL_H, FULL_W)):
        self.shape, self.width = shape, float(width)
        self.base = np.zeros(shape, np.uint8)      # finished parts
        self.stroke, self.polygon = [], []         # parts being drawn
        self.mode = "stroke"
        self.hows = set()
        self.history = []                          # (base, stroke, polygon) snapshots for undo
        self.neg = []                              # SAM "not tail" hints

    def _push(self):
        self.history.append((self.base.copy(), list(self.stroke), list(self.polygon), set(self.hows)))
        if len(self.history) > 60:
            self.history.pop(0)

    def mask(self):
        m = self.base.copy()
        if self.stroke:
            m |= stroke_mask(self.stroke, self.width, self.shape)
        if len(self.polygon) >= 3:
            m |= polygon_mask(self.polygon, self.shape)
        return m

    def click(self, x, y):
        self._push()
        if self.mode == "polygon":
            self.polygon.append((x, y))
        else:
            self.stroke.append((x, y))
            self.hows.add("stroke")

    def finish_part(self):
        """End the current stroke / close the polygon into the finished mask."""
        if not self.stroke and len(self.polygon) < 3:
            self.polygon = []
            return
        self._push()
        if self.stroke:
            self.base |= stroke_mask(self.stroke, self.width, self.shape)
            self.stroke = []
        if len(self.polygon) >= 3:
            self.base |= polygon_mask(self.polygon, self.shape)
            self.hows.add("polygon")
        self.polygon = []

    def set_mode(self, mode):
        self.finish_part()
        self.mode = mode if self.mode != mode else "stroke"

    def brush(self, x, y, erase=False, new_drag=False):
        if new_drag:
            self.finish_part()
            self._push()
        r = max(int(round(self.width / 2)), 1)
        cv2.circle(self.base, (int(round(x)), int(round(y))), r, 0 if erase else 255, -1)
        self.hows.add("brush")

    def add_layer(self, m, how):
        self._push()
        self.base |= (m > 0).astype(np.uint8) * 255
        self.stroke = []
        self.hows.add(how)

    def set_width(self, w):
        self.width = float(min(max(w, 2), 80))

    def undo(self):
        if self.history:
            self.base, self.stroke, self.polygon, self.hows = self.history.pop()

    def clear(self):
        self._push()
        self.base[:] = 0
        self.stroke, self.polygon, self.neg = [], [], []


# ---------------------------------------------------------------------------------------------
# files

def tail_csv_for(labels_csv):
    return os.path.join(os.path.dirname(os.path.abspath(labels_csv)), "tail_labels.csv")


def load_tail_labels(path):
    """tail_labels.csv -> {ir_path as written in the labels CSV: row}."""
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {r["ir_path"]: r for r in csv.DictReader(f)}


def save_tail_labels(path, done):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TAIL_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(done.values())
    os.replace(tmp, path)


def save_tail(set_dir, done, row, mask, vulva_mask=None, how="", width=0.0):
    """Write one frame's tail mask (or 'no tail') and its row; returns the row."""
    tail = int(mask is not None and mask.any())
    out = {k: row.get(k, "") for k in ("ir_path", "pig_id", "timestamp_folder", "pig_timestamp")}
    out.update({"tail": tail, "tail_mask_path": "", "covers_vulva": "", "how": how if tail else "none",
                "width_px": round(width, 1) if tail else "", "date": time.strftime("%Y-%m-%d")})
    if tail:
        os.makedirs(os.path.join(set_dir, "tail_masks"), exist_ok=True)
        name = f"{row.get('timestamp_folder', '')}_pig{row.get('pig_id', '')}_{row.get('pig_timestamp', '')}.png"
        rel = os.path.join("tail_masks", name).replace("\\", "/")
        cv2.imwrite(os.path.join(set_dir, rel), (mask > 0).astype(np.uint8) * 255)
        out["tail_mask_path"] = rel
        if vulva_mask is not None and vulva_mask.any():
            out["covers_vulva"] = int(bool(((mask > 0) & (vulva_mask > 0)).any()))
    done[row["ir_path"]] = out
    return out


def frame_order(rows, done, redo=False):
    """Tail-labelled frames first, then visible vulvas, then the rest; already done ones skipped."""
    def rank(r):
        if r.get("label") == "absent" and "tail" in (r.get("note") or ""):
            return 0
        return 1 if r.get("label") == "present" else 2
    todo = [r for r in rows if redo or r["ir_path"] not in done]
    return sorted(todo, key=lambda r: (rank(r), r.get("pig_timestamp", "")))


def read_label_rows(labels_csv, frames_roots=None):
    """Rows of a labels CSV with absolute image paths (relocated under --frames-root when the
    image is not next to the labels); 'key' keeps the ir_path as written in the CSV."""
    from select_frames import index_folders
    from train_seg import relocate_row
    index = index_folders(frames_roots) if frames_roots else {}
    base = os.path.dirname(os.path.abspath(labels_csv))
    res = lambda p: p if (not p or os.path.isabs(p) or p.startswith("\\\\")) else os.path.join(base, p)  # noqa: E731
    with open(labels_csv, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("label") in ("present", "absent")]
    colour = {}
    ls = os.path.join(base, "label_set.csv")
    if os.path.exists(ls):
        with open(ls, newline="", encoding="utf-8-sig") as f:
            colour = {fr["ir_path"]: fr.get("rgb_aligned_path", "") for fr in csv.DictReader(f)}
    for r in rows:
        r["key"] = r["ir_path"]
        r["rgb_reg"] = res(colour.get(r["ir_path"], ""))
        r["ir_path"], r["mask_path"], r["depth_path"] = res(r["ir_path"]), res(r.get("mask_path", "")), ""
        relocate_row(r, index)
    return rows


def colour_for(row, ir):
    """Colour image on the IR grid: the set's registered one, else matched on the fly, else None."""
    p = row.get("rgb_reg", "")
    if p and os.path.exists(p):
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is not None and im.shape[:2] == (FULL_H, FULL_W):
            return im
    d, name = os.path.split(row["ir_path"])
    for ext in (".jpg", ".png"):
        cand = os.path.join(d, name.replace("_ir_vis_", "_rgb_").rsplit(".", 1)[0] + ext)
        if os.path.exists(cand):
            rgb = cv2.imread(cand, cv2.IMREAD_COLOR)
            if rgb is not None:
                from rgb_align import register
                return register(ir, rgb, None)[0]
    return None


class SamAssist:
    """Optional: outline the tail with Segment Anything from click hints (ultralytics)."""

    def __init__(self, weights):
        from ultralytics import SAM          # pip install ultralytics; weights download on first use
        self.model = SAM(weights)

    def __call__(self, image_bgr, pos, neg):
        pts = [[float(x), float(y)] for x, y in list(pos) + list(neg)]
        labels = [1] * len(pos) + [0] * len(neg)
        r = self.model(image_bgr, points=[pts], labels=[labels], verbose=False)
        if not r or r[0].masks is None or len(r[0].masks.data) == 0:
            return None
        data = r[0].masks.data.cpu().numpy()
        m = data[int(np.argmax([d.sum() for d in data]))] > 0.5 if len(data) > 1 else data[0] > 0.5
        if m.shape != (FULL_H, FULL_W):
            m = cv2.resize(m.astype(np.uint8), (FULL_W, FULL_H), interpolation=cv2.INTER_NEAREST) > 0
        return m


# ---------------------------------------------------------------------------------------------
# GUI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels_csv", help="labels CSV of a label set (label_ellipse.py output)")
    ap.add_argument("--frames-root", default="", help="';'-separated raw folders, for frames not next to the labels")
    ap.add_argument("--width", type=float, default=14, help="starting stroke / brush width, IR px")
    ap.add_argument("--only", choices=("all", "tail", "present"), default="all",
                    help="tail = only frames labelled tail; present = only frames with a visible vulva")
    ap.add_argument("--redo", action="store_true", help="also show frames already done (to fix them)")
    ap.add_argument("--sam", default="", help="optional SAM weights for the 'a' key, e.g. sam2.1_t.pt")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    plt.rcParams["keymap.save"] = []
    plt.rcParams["keymap.quit"] = []
    for k in ("keymap.pan", "keymap.zoom", "keymap.back", "keymap.forward", "keymap.home", "keymap.grid",
              "keymap.xscale", "keymap.yscale", "keymap.fullscreen"):
        if k in plt.rcParams:
            plt.rcParams[k] = []

    labels_csv = os.path.abspath(os.path.expanduser(args.labels_csv))
    set_dir = os.path.dirname(labels_csv)
    out_csv = tail_csv_for(labels_csv)
    done = load_tail_labels(out_csv)
    rows = read_label_rows(labels_csv, [r for r in args.frames_root.split(";") if r.strip()])
    for r in rows:                        # done is keyed by the ir_path as written in the labels CSV
        r["ir_path_abs"], r["ir_path"] = r["ir_path"], r["key"]
    todo = frame_order(rows, done, args.redo)
    if args.only == "tail":
        todo = [r for r in todo if r.get("label") == "absent" and "tail" in (r.get("note") or "")]
    elif args.only == "present":
        todo = [r for r in todo if r.get("label") == "present"]
    n_tail = sum(1 for r in rows if r.get("label") == "absent" and "tail" in (r.get("note") or ""))
    print(f"{len(rows)} labelled frames ({n_tail} marked tail), {len(done)} already done, {len(todo)} to go")
    if not todo:
        return
    sam = None
    if args.sam:
        try:
            sam = SamAssist(args.sam)
            print(f"SAM ready ({args.sam}): press a to outline from your clicks")
        except Exception as e:  # noqa: BLE001
            print(f"SAM not available ({e}); strokes, polygon and brush still work")

    fig, axes = plt.subplots(1, 2, figsize=(13, 7.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.86, wspace=0.03)
    fig.canvas.manager.set_window_title("PAAL tail outlining")
    state = {"i": 0, "drag": None}
    cache = {}

    def current():
        return todo[state["i"]]

    def load(row):
        k = row["ir_path"]
        if k not in cache:
            cache.clear()
            ir = cv2.imread(row["ir_path_abs"], cv2.IMREAD_GRAYSCALE)
            if ir is not None and ir.shape != (FULL_H, FULL_W):
                ir = cv2.resize(ir, (FULL_W, FULL_H))
            if ir is None:
                ir = np.zeros((FULL_H, FULL_W), np.uint8)
            disp = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(ir)
            col = colour_for(dict(row, ir_path=row["ir_path_abs"]), ir)
            vm = None
            if row.get("label") == "present" and row.get("mask_path") and os.path.exists(row["mask_path"]):
                vm = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
                if vm is not None and vm.shape != (FULL_H, FULL_W):
                    vm = cv2.resize(vm, (FULL_W, FULL_H), interpolation=cv2.INTER_NEAREST)
            cache[k] = (disp, col, vm)
        return cache[k]

    def new_canvas():
        cv_ = TailCanvas(args.width)
        old = done.get(current()["ir_path"])
        if old and old.get("tail_mask_path"):
            m = cv2.imread(os.path.join(set_dir, old["tail_mask_path"]), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                cv_.base = (m > 127).astype(np.uint8) * 255
        state["canvas"] = cv_

    def draw():
        row, cv_ = current(), state["canvas"]
        disp, col, vm = load(row)
        m = cv_.mask() > 0
        xl, yl = axes[0].get_xlim(), axes[0].get_ylim()
        keep_view = state.get("view_of") == row["ir_path"]
        for a, (name, im) in zip(axes, (("IR", cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)),
                                        ("colour (matched to IR)" if col is not None else "colour (none)",
                                         cv2.cvtColor(col, cv2.COLOR_BGR2RGB) if col is not None
                                         else np.full((FULL_H, FULL_W, 3), 30, np.uint8)))):
            a.clear()
            show = im.copy()
            show[m] = (0.55 * show[m] + 0.45 * np.array([255, 40, 40])).astype(np.uint8)
            a.imshow(show)
            if vm is not None:
                cnts, _ = cv2.findContours((vm > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for c in cnts:
                    a.plot(c[:, 0, 0], c[:, 0, 1], "--", color="#3dff6a", lw=1.2)
            for pts, style in ((cv_.stroke, "o-"), (cv_.polygon, "s--")):
                if pts:
                    xs, ys = zip(*pts)
                    a.plot(xs, ys, style, color="#ffe600", ms=4, lw=1)
            for x, y in cv_.neg:
                a.plot(x, y, "x", color="#00e5ff", ms=8, mew=2)
            a.set_title(name, fontsize=9)
            a.set_xticks([])
            a.set_yticks([])
        if keep_view:
            axes[0].set_xlim(xl)
            axes[0].set_ylim(yl)
        else:
            axes[0].set_xlim(CROP[0], CROP[2])
            axes[0].set_ylim(CROP[3], CROP[1])
            state["view_of"] = row["ir_path"]
        lab = row.get("label", "")
        what = "TAIL" if "tail" in (row.get("note") or "") else ("vulva visible (green)" if lab == "present"
                                                               else "not visible")
        n_done = sum(1 for r in done.values())
        fig.suptitle(
            f"pig {row.get('pig_id')}   {row.get('pig_timestamp')}   labelled: {what}   "
            f"[{state['i'] + 1}/{len(todo)}, {n_done} done]   mode: {cv_.mode}   width {cv_.width:.0f} px\n"
            "stroke: click along the tail   n new stroke   [ ] or wheel width   p polygon   b brush (right = erase)"
            + ("   a SAM" if sam else "") + "\nEnter save   x no tail   u undo   c clear   s skip   q quit",
            fontsize=9)
        fig.canvas.draw_idle()

    def advance():
        state["i"] += 1
        if state["i"] >= len(todo):
            print("All frames done.")
            plt.close(fig)
            return
        new_canvas()
        draw()

    def toolbar_busy():
        tb = fig.canvas.manager.toolbar
        return tb is not None and getattr(tb, "mode", "")

    def on_press(ev):
        if ev.inaxes not in axes or ev.xdata is None or toolbar_busy():
            return
        cv_ = state["canvas"]
        if cv_.mode == "brush" and ev.button in (1, 3):
            state["drag"] = ev.button
            cv_.brush(ev.xdata, ev.ydata, erase=ev.button == 3, new_drag=True)
        elif ev.button == 1:
            cv_.click(ev.xdata, ev.ydata)
        elif ev.button == 3 and sam is not None:
            cv_.neg.append((ev.xdata, ev.ydata))
        draw()

    def on_motion(ev):
        if state["drag"] and ev.inaxes in axes and ev.xdata is not None:
            state["canvas"].brush(ev.xdata, ev.ydata, erase=state["drag"] == 3)
            state["moves"] = state.get("moves", 0) + 1
            if state["moves"] % 4 == 0:          # redrawing on every mouse move is slow over Remote Desktop
                draw()

    def on_release(ev):
        if state["drag"]:
            state["drag"] = None
            draw()

    def on_scroll(ev):
        cv_ = state["canvas"]
        cv_.set_width(cv_.width + (2 if ev.button == "up" else -2))
        draw()

    def on_key(ev):
        k = (ev.key or "").lower()
        cv_, row = state["canvas"], current()
        if k == "enter":
            cv_.finish_part()
            m = cv_.mask()
            if not m.any():
                print("  nothing drawn: press x if there is no tail")
                return
            save_tail(set_dir, done, row, m, load(row)[2], "+".join(sorted(cv_.hows)) or "stroke", cv_.width)
            save_tail_labels(out_csv, done)
            advance()
        elif k == "x":
            save_tail(set_dir, done, row, None)
            save_tail_labels(out_csv, done)
            advance()
        elif k == "n":
            cv_.finish_part()
            draw()
        elif k == "p":
            cv_.set_mode("polygon")
            draw()
        elif k == "b":
            cv_.set_mode("brush")
            draw()
        elif k in ("]", "+", "="):
            cv_.set_width(cv_.width + 2)
            draw()
        elif k in ("[", "-"):
            cv_.set_width(cv_.width - 2)
            draw()
        elif k == "a" and sam is not None:
            pos = list(cv_.stroke) or list(cv_.polygon)
            if not pos:
                print("  click a few points on the tail first, then a")
                return
            disp, col, _ = load(row)
            img = col if col is not None else cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
            try:
                m = sam(img, pos, cv_.neg)
            except Exception as e:  # noqa: BLE001
                print(f"  SAM failed: {e}")
                return
            if m is not None:
                cv_.add_layer(m, "sam")
                cv_.neg = []
            draw()
        elif k == "u":
            cv_.undo()
            draw()
        elif k == "c":
            cv_.clear()
            draw()
        elif k == "s":
            advance()
        elif k == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)
    new_canvas()
    draw()
    plt.show()
    tails = sum(1 for r in done.values() if str(r.get("tail")) == "1")
    cover = sum(1 for r in done.values() if str(r.get("covers_vulva")) == "1")
    print(f"Saved {len(done)} frames ({tails} with a tail outlined, {len(done) - tails} no tail; "
          f"{cover} tails touch the vulva outline) -> {out_csv}")


if __name__ == "__main__":
    main()
