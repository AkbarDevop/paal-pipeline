"""Fast checks for the vulva pipeline (no lab data needed).

Run:  python tests/test_vulva_pipeline.py      (or: pytest tests/)

Synthetic OAK-D frames: a tilted rump plane in depth with an ellipsoid bump of
known width / length / height, and the matching IR image with the vulva drawn
in. Checks the 3D measurement against the known size, the readers for Lucas's
workbook and the manual labels, frame selection, daily aggregation, and the
whole train -> run path on a tiny dataset.
"""

import csv
import json
from collections import Counter
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "vulva"))
sys.path.insert(0, ROOT)

from daily import add_baseline_change, aggregate, compare_truth  # noqa: E402
from measure3d import DEFAULT_INTRINSICS as K  # noqa: E402
from measure3d import measure  # noqa: E402
from select_frames import parse_windows, select  # noqa: E402

PY = sys.executable


def render(tilt_deg=30, dist=1200, W=40.0, L=60.0, H=15.0, noise=0.0, seed=0, cx=320, cy=260):
    """Depth (uint16 mm) and vulva mask for a rump plane tilted about x with an ellipsoid bump."""
    t = np.radians(tilt_deg)
    n = np.array([0, np.sin(t), -np.cos(t)])
    ex = np.array([1.0, 0, 0])
    ey = np.cross(n, ex)
    ey /= np.linalg.norm(ey)
    if ey[1] < 0:
        ey = -ey
    r0 = np.array([(cx - K["cx"]) / K["fx"], (cy - K["cy"]) / K["fy"], 1.0])
    C = r0 / r0[2] * dist
    ys, xs = np.mgrid[0:480, 0:640]
    rays = np.stack([(xs - K["cx"]) / K["fx"], (ys - K["cy"]) / K["fy"], np.ones_like(xs, float)], -1)
    s = (C @ n) / (rays @ n)
    P = rays * s[..., None]
    rel = P - C
    a, b = rel @ ex, rel @ ey
    q = 1 - (a / (W / 2)) ** 2 - (b / (L / 2)) ** 2
    h = np.where(q > 0, H * np.sqrt(np.clip(q, 0, 1)), 0.0)
    unit = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    cos = np.abs(unit @ n)
    z = P[..., 2] - h / cos * unit[..., 2]
    z = z + np.random.default_rng(seed).normal(0, noise, z.shape)
    return np.clip(z, 0, 65535).astype(np.uint16), q > 0


def test_measure_recovers_known_size_under_tilt():
    for tilt in (0, 45, 60):
        depth, mask = render(tilt_deg=tilt)
        m = measure(depth, mask)
        assert m["measure_status"] == "ok"
        assert abs(m["width_mm"] - 40) < 2.0, (tilt, m["width_mm"])
        assert abs(m["length_mm"] - 60) < 2.5, (tilt, m["length_mm"])
        assert abs(m["peak_height_mm"] - 15) < 1.5, (tilt, m["peak_height_mm"])
        true_vol = 2 / 3 * np.pi * 20 * 30 * 15
        assert abs(m["volume_mm3"] - true_vol) / true_vol < 0.10, (tilt, m["volume_mm3"])
        assert abs(m["view_angle_deg"] - tilt) < 1.0


def test_measure_handles_noise_bad_masks_and_missing_depth():
    depth, mask = render(tilt_deg=30, noise=2.0, seed=3)
    m = measure(depth, mask)
    assert abs(m["width_mm"] - 40) < 3 and abs(m["length_mm"] - 60) < 4
    assert measure(depth, np.zeros_like(mask))["measure_status"] == "empty_mask"
    assert measure(None, mask)["measure_status"] == "no_depth"
    tiny = np.zeros_like(mask)
    tiny[260:263, 320:323] = True
    assert measure(depth, tiny)["measure_status"] == "too_few_points"


def _lucas_workbook(path):
    from openpyxl import Workbook
    wb = Workbook()
    wb.remove(wb.active)
    days = [(0, "MOVED", None, None, 0, 0), (1, "OK", 40.0, 60.0, 0, 0), (2, "", 40.4, 60.6, 0, 0),
            (3, "OK", 44.0, 66.0, 20, 20), (4, "", 43.0, 65.0, 0, 20), (5, "OK", 40.2, 60.1, 0, 0),
            (6, "", 40.0, 60.0, 0, 0), (7, "OK", 41.0, 59.0, 0, 0)]
    for pid, tag in ((0, "200196"), (1, "182")):
        ws = wb.create_sheet(str(pid))
        ws["A1"], ws["A2"], ws["B2"] = "SOW 20", "NUMBER", int(tag)
        ws["A13"], ws["B13"], ws["C13"], ws["H13"], ws["P13"] = "DAY", "DATE", "BODY CONDITION", "VULVA", "HEAT"
        for col, name in zip("CDEFGHIJKLMNOPQ", ["CALIPER", "BS", "BC SOLID", "WEIGHT", "BACKFAT", "V. W.", "V. L.",
                                                  "V. V.", "V. S. O. A.", "V. S. N. A.", "V. V. S. B. N. E.",
                                                  "I. I. T. B", "BPTM", "MORNING", "EVENING"]):
            ws[f"{col}14"] = name
        ws["S12"] = "ACTION"
        for i, (day, action, w, l, hm, he) in enumerate(days):
            r = 15 + i
            ws[f"A{r}"], ws[f"B{r}"] = day, datetime(2026, 2, 6 + day)
            if w:
                ws[f"C{r}"], ws[f"D{r}"], ws[f"H{r}"], ws[f"I{r}"] = 14.5, 4, w, l
                ws[f"J{r}"] = w * l / 200
            ws[f"P{r}"], ws[f"Q{r}"], ws[f"S{r}"] = hm, he, action
        ws["A40"], ws["B40"], ws["J40"] = 30, datetime(2026, 3, 8), 0     # empty future row
    wb.save(path)


def test_truth_reads_lucas_workbook_layout():
    from truth import heat_response, read_workbook
    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, "Feb_2026.xlsx")
        _lucas_workbook(p)
        rows = read_workbook(p)
        assert len(rows) == 16 and {r["pig_id"] for r in rows} == {0, 1}
        r3 = next(r for r in rows if r["pig_id"] == 0 and r["day"] == 3)
        assert r3["date"] == "2026-02-09" and r3["heat"] == 1 and r3["vulva_width_mm"] == 44.0
        assert abs(r3["vulva_index"] - 44 * 66 / 200) < 1e-6
        assert next(r for r in rows if r["day"] == 0)["action"] == "MOVED"
        assert rows[0]["tag"] == "200196"
        summary, noise = heat_response(rows)
        s0 = summary[0]
        assert s0["heat_days"] == "3|4"
        assert abs(s0["vulva_index_rise_pct"] - 100 * (44 * 66 - 40.2 * 60.3) / (40.2 * 60.3)) < 1.5
        assert noise["vulva_width_mm"] is not None
    finally:
        shutil.rmtree(tmp)


def test_select_frames_standing_windows_and_cap():
    rows = []
    for h in range(0, 24):
        for pid in (0, 1):
            rows.append({"timestamp_folder": f"20260211-{h:02d}-00-00", "pig_id": str(pid),
                         "pig_timestamp": f"20260211-{h:02d}-00-{10 + pid:02d}",
                         "prediction_name": "standing" if h % 2 == 0 else "lying",
                         "confidence": "0.9", "id_status": "ok",
                         "image_path": f"/d/20260211-{h:02d}-00-00/pig{pid}_ir_vis_20260211-{h:02d}-00-{10 + pid:02d}.jpg"})
    rows.append(dict(rows[0], id_status="overflow"))
    picked = select(rows, windows=parse_windows("06:00-11:00"))
    assert {p["pig_timestamp"][9:11] for p in picked} == {"06", "08", "10"}
    assert picked[0]["depth_path"].endswith("pig0_depth_20260211-06-00-10.raw")
    capped = select(rows, max_per_day=4)
    assert len(capped) == 8 and len({p["pig_timestamp"] for p in capped if p["pig_id"] == 0}) == 4
    one = select(rows, max_per_day=1)                   # 12 standing hours: the middle one, not the first
    assert [p["pig_timestamp"][9:11] for p in one if p["pig_id"] == 0] == ["12"]


def test_relocate_finds_frames_after_the_data_moved():
    """Predictions made before the raw folders moved: frames are found again by timestamp folder name."""
    from select_frames import relocate
    tmp = tempfile.mkdtemp()
    try:
        new = os.path.join(tmp, "E", "Pictures_OAK", "OAK_0206_0211", "20260209-09-00-00")
        os.makedirs(new)
        open(os.path.join(new, "pig3_ir_vis_20260209-09-00-13.jpg"), "wb").close()
        picked = [{"timestamp_folder": "20260209-09-00-00", "pig_id": 3, "pig_timestamp": "20260209-09-00-13",
                   "ir_path": "C:\\PAAL_data\\Fed_pig\\Pictures_OAK\\20260209-09-00-00\\pig3_ir_vis_20260209-09-00-13.jpg",
                   "depth_path": ""},
                  dict(timestamp_folder="20260209-09-10-00", pig_id=4, pig_timestamp="20260209-09-10-14",
                       ir_path="C:\\gone\\pig4_ir_vis_20260209-09-10-14.jpg", depth_path="")]
        found, missing = relocate(picked, [os.path.join(tmp, "E", "Pictures_OAK")])
        assert missing == 1 and len(found) == 1
        assert found[0]["ir_path"] == os.path.join(new, "pig3_ir_vis_20260209-09-00-13.jpg")
        assert found[0]["depth_path"].endswith("pig3_depth_20260209-09-00-13.raw")
    finally:
        shutil.rmtree(tmp)


def test_review_run_panels_and_point_cloud():
    """review_run.py: panels for a run's frames, and the 3D surface / point cloud around a vulva."""
    from review_run import local_surface, write_ply
    from seg_model import VulvaUNet, save_checkpoint
    depth, mask = render(tilt_deg=30, W=40, L=60, H=12, noise=0.0)
    surf = local_surface(depth, mask, 320, 260)
    assert surf is not None
    u, v, h, vul, pts, ys, xs = surf
    assert 8 < np.percentile(h[vul], 95) < 16 and abs(np.median(h[~vul])) < 2      # bump on a flat rump
    tmp = tempfile.mkdtemp()
    try:
        write_ply(os.path.join(tmp, "a.ply"), surf, np.full((480, 640), 128, np.uint8))
        with open(os.path.join(tmp, "a.ply")) as f:
            head = f.read(400)
        assert head.startswith("ply") and f"element vertex {len(pts)}" in head
        rng = np.random.default_rng(2)
        folder = os.path.join(tmp, "20260124-09-00-00")
        os.makedirs(folder)
        rows = []
        for pid in range(2):
            ts = f"20260124-09-00-{10 + pid:02d}"
            _write_frame(folder, pid, ts, True, rng)
            rows.append({"timestamp_folder": "20260124-09-00-00", "pig_id": pid, "pig_timestamp": ts,
                         "measure_status": "ok", "width_mm": 40, "length_mm": 60,
                         "ir_path": os.path.join(folder, f"pig{pid}_ir_vis_{ts}.jpg")})
        fcsv = os.path.join(tmp, "vulva_frames.csv")
        with open(fcsv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        ck = os.path.join(tmp, "m.pth")
        save_checkpoint(VulvaUNet(pretrained=False), ck, {"threshold": 0.5, "min_area": 25, "inputs": "ir"})
        out = subprocess.run([PY, os.path.join(ROOT, "vulva", "review_run.py"), fcsv, "--model", ck, "--n", "2"],
                             capture_output=True, text=True, cwd=tmp)
        assert out.returncode == 0, out.stderr[-2000:]
        assert len(os.listdir(os.path.join(tmp, "review_ok", "panels"))) == 2
        assert cv2.imread(os.path.join(tmp, "review_ok", "index.jpg")).shape[1] == 600
        assert os.path.exists(os.path.join(tmp, "review_ok", "sheet_01.jpg"))
        rows2 = [dict(r, measure_status="not_visible") for r in rows]            # fast contact sheets only
        with open(fcsv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows2[0]))
            w.writeheader()
            w.writerows(rows2)
        out = subprocess.run([PY, os.path.join(ROOT, "vulva", "review_run.py"), fcsv, "--model", ck, "--which",
                              "not_visible", "--n", "2", "--sheet-only"], capture_output=True, text=True, cwd=tmp)
        assert out.returncode == 0, out.stderr[-2000:]
        sh = cv2.imread(os.path.join(tmp, "review_not_visible", "sheet_01.jpg"))
        assert sh.shape[:2] == (230, 230 * 8) and not os.path.exists(os.path.join(tmp, "review_not_visible", "panels"))
    finally:
        shutil.rmtree(tmp)


def test_two_label_sets_frames_root_and_sow_split():
    """Training on several label CSVs; labels.csv + masks/ alone, images found under --frames-root;
    split_by_sow keeps every sow on one side."""
    from split_by_sow import split
    from train_seg import load_labels
    tmp = tempfile.mkdtemp()
    try:
        rng = np.random.default_rng(4)
        raw = os.path.join(tmp, "share", "OAK_0124", "20260124-09-00-00")
        os.makedirs(raw)
        sets = []
        for k, pids in enumerate(((0, 1), (2, 3))):
            d = os.path.join(tmp, f"set{k}")
            os.makedirs(os.path.join(d, "masks"))
            rows = []
            for pid in pids:
                ts = f"20260124-09-00-{10 + pid:02d}"
                m = _write_frame(raw, pid, ts, True, rng)
                mp = f"masks/{pid}.png"
                cv2.imwrite(os.path.join(d, mp), m.astype(np.uint8) * 255)
                rows.append({"ir_path": f"20260124-09-00-00/pig{pid}_ir_vis_{ts}.jpg", "mask_path": mp,
                             "label": "present", "pig_id": pid, "timestamp_folder": "20260124-09-00-00",
                             "pig_timestamp": ts})
            with open(os.path.join(d, "labels.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            sets.append(os.path.join(d, "labels.csv"))
        assert load_labels(sets) == []                                       # images not next to the labels
        items = load_labels(sets, "ir", [os.path.join(tmp, "share")])
        assert sorted(it["pig_id"] for it in items) == [0, 1, 2, 3] and all(it["mask"].any() for it in items)
        items_d = load_labels(sets, "ir+depth", [os.path.join(tmp, "share")])  # depth found next to the moved IR
        assert len(items_d) == 4 and items_d[0]["image"].ndim == 3
        rows = [{"pig_id": str(p), "label": "present"} for p in (1, 1, 2, 3, 3, 3, 4, 5)]
        test, train, ts, tr = split(rows, 0.5, 42)
        assert set(ts).isdisjoint(tr) and len(test) + len(train) == len(rows) and len(ts) == 2
    finally:
        shutil.rmtree(tmp)


def test_daily_median_baseline_and_caliper_join():
    frames = []
    for day, base in ((7, 40.0), (8, 40.0), (9, 44.0), (10, 40.0)):
        for k in range(5):
            frames.append({"timestamp_folder": f"202602{day:02d}-09-{k:02d}-00", "pig_id": "3",
                           "measure_status": "ok", "width_mm": base + (k - 2) * 0.5, "length_mm": 60.0,
                           "peak_height_mm": 10, "volume_mm3": 1000, "plane_rmse_mm": 1, "view_angle_deg": 30,
                           "depth_coverage": 0.9})
    frames.append(dict(frames[0], width_mm=99, plane_rmse_mm=50))     # bad plane fit, dropped
    daily = add_baseline_change(aggregate(frames, min_frames=3), baseline_days=2)
    assert [d["n_frames"] for d in daily] == [5, 5, 5, 5]
    assert daily[0]["width_mm"] == 40.0 and daily[2]["width_mm_change_pct"] == 10.0
    truth = [{"pig_id": 3, "date": f"2026-02-{d:02d}", "vulva_width_mm": w, "vulva_length_mm": 60,
              "vulva_index": w * 60 / 200, "heat": int(d == 9)} for d, w in ((7, 41), (8, 41), (9, 45), (10, 41))]
    joined, s = compare_truth(daily, truth)
    assert s["sow_days"] == 4 and s["width_mm_mae"] == 1.0 and s["width_mm_bias"] == -1.0
    assert s["index_change_heat_days"] > s["index_change_other_days"]


def test_repeatability_reports_frames_needed():
    from repeatability import analyse
    rows = [{"timestamp_folder": "20260211-09-00-00", "pig_id": "1", "status": "measured",
             "width_mm": w, "length_mm": 60} for w in (36, 40, 44, 40)]
    per, s = analyse(rows, truth=[{"pig_id": 1, "date": "2026-02-11", "vulva_width_mm": 40, "vulva_length_mm": 60}])
    assert s["width_mm_within_day_sd"] > 3 and s["width_mm_frames_needed"] == 3
    assert s["width_mm_mae_vs_caliper"] == 2.0 and s["length_mm_mae_vs_caliper"] == 0.0


def _write_frame(folder, pid, ts, present, rng):
    """IR jpg + depth raw for one stall; returns the full-frame mask when present."""
    tilt = rng.uniform(20, 50)
    W, L = rng.uniform(34, 46), rng.uniform(52, 70)
    cx, cy = int(rng.uniform(280, 360)), int(rng.uniform(220, 300))
    depth, mask = render(tilt_deg=tilt, W=W, L=L, H=14, cx=cx, cy=cy, noise=1.0, seed=int(rng.integers(1e6)))
    ir = np.clip(rng.normal(110, 12, (480, 640)), 0, 255).astype(np.uint8)
    ir = cv2.GaussianBlur(ir, (5, 5), 0)
    if present:
        ir[mask] = 35
        ring = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        ir[ring & ~mask] = 200
    else:
        mask = np.zeros_like(mask)
    cv2.imwrite(os.path.join(folder, f"pig{pid}_ir_vis_{ts}.jpg"), ir)
    with open(os.path.join(folder, f"pig{pid}_depth_{ts}.raw"), "wb") as f:
        f.write(b"\0" * 8)
        f.write(depth.tobytes())
    return mask


def test_train_and_run_end_to_end():
    tmp = tempfile.mkdtemp()
    try:
        rng = np.random.default_rng(0)
        data = os.path.join(tmp, "Pictures_OAK")
        masks = os.path.join(tmp, "labels", "vulva_masks_manual")
        os.makedirs(masks)
        manual, preds = [], []
        for k in range(6):
            folder_name = f"20260211-09-{10 * k:02d}-00"
            folder = os.path.join(data, folder_name)
            os.makedirs(folder)
            for pid in range(4):
                ts = f"20260211-09-{10 * k:02d}-{10 + pid:02d}"
                present = (k + pid) % 3 != 0
                m = _write_frame(folder, pid, ts, present, rng)
                mp = ""
                if present:
                    mp = os.path.join("C:\\elsewhere\\labels\\vulva_masks_manual",
                                      f"{folder_name}_pig{pid}_{ts}_vulva_mask.png")
                    cv2.imwrite(os.path.join(masks, os.path.basename(mp.replace("\\", "/"))), m.astype(np.uint8) * 255)
                manual.append({"timestamp_folder": folder_name, "pig_id": pid, "pig_timestamp": ts,
                               "status": "measured" if present else "no_vulva", "mask_path": mp,
                               "texture_input": ""})
                preds.append({"timestamp_folder": folder_name, "pig_id": pid, "pig_timestamp": ts,
                              "prediction_name": "standing", "confidence": 0.95, "id_status": "ok",
                              "image_path": os.path.join(folder, f"pig{pid}_ir_vis_{ts}.jpg")})
        manual.append(dict(manual[0], status="failed: Vulva mask is empty after cleanup"))
        man_csv = os.path.join(tmp, "manual.csv")
        with open(man_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manual[0]))
            w.writeheader()
            w.writerows(manual)
        pred_csv = os.path.join(tmp, "predictions.csv")
        with open(pred_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(preds[0]))
            w.writeheader()
            w.writerows(preds)

        labels = os.path.join(tmp, "seg_labels.csv")
        out = subprocess.run([PY, os.path.join(ROOT, "vulva", "make_seg_labels.py"), "--manual", man_csv,
                              "--data-root", data, "--mask-root", masks, "--out", labels],
                             capture_output=True, text=True, cwd=tmp)
        assert out.returncode == 0, out.stderr
        assert "24 labelled frames (16 present, 8 absent)" in out.stdout, out.stdout

        model = os.path.join(tmp, "models", "vulva_seg.pth")
        share = os.path.join(tmp, "share")
        os.makedirs(share)
        out = subprocess.run([PY, os.path.join(ROOT, "vulva", "train_seg.py"), "--labels", labels, "--folds", "2",
                              "--epochs", "4", "--no-pretrained", "--out-model", model,
                              "--out-dir", os.path.join(tmp, "seg"), "--copy-to", share],
                             capture_output=True, text=True, cwd=tmp)
        assert out.returncode == 0, out.stderr[-2000:]
        assert os.path.exists(model) and "Cross-validation" in out.stdout
        assert os.path.exists(os.path.join(tmp, "seg", "seg_report.json"))
        assert sorted(os.listdir(share)) == ["vulva_seg.pth", "vulva_seg_cv.csv", "vulva_seg_report.txt"]
        # --jobs: folds and final model as parallel workers, same outputs
        model_p = os.path.join(tmp, "models", "vulva_seg_par.pth")
        out = subprocess.run([PY, os.path.join(ROOT, "vulva", "train_seg.py"), "--labels", labels, "--folds", "2",
                              "--epochs", "3", "--no-pretrained", "--jobs", "3", "--out-model", model_p,
                              "--out-dir", os.path.join(tmp, "seg_par")], capture_output=True, text=True, cwd=tmp)
        assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
        assert "All 3 models trained" in out.stdout and "Cross-validation" in out.stdout, out.stdout
        assert os.path.exists(model_p)
        with open(os.path.join(tmp, "seg_par", "cv_predictions.csv")) as f:
            assert len(list(csv.DictReader(f))) == 24
        out_train = out.stdout

        outdir = os.path.join(tmp, "out")
        out = subprocess.run([PY, os.path.join(ROOT, "vulva", "run_vulva.py"), pred_csv, "--model", model,
                              "--out-dir", outdir, "--min-frames", "1", "--save-masks"],
                             capture_output=True, text=True, cwd=tmp)
        assert out.returncode == 0, out.stderr[-2000:]
        with open(os.path.join(outdir, "vulva_frames.csv")) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 24
        assert all(r["measure_status"] for r in rows)
        assert "vulva_conf" in rows[0]
        with open(os.path.join(tmp, "seg", "cv_predictions.csv")) as f:
            assert "peak_prob,blob_conf" in f.readline()
        assert "With confidence cut" in out_train or "No confidence cut" in out_train
        assert os.path.exists(os.path.join(outdir, "vulva_daily.csv"))
        assert os.path.exists(os.path.join(outdir, "vulva_summary.json"))
        # IR + depth fusion model (5 channels): trains, records its inputs, and run_vulva feeds it depth
        model_d = os.path.join(tmp, "models", "vulva_seg_d.pth")
        out = subprocess.run([PY, os.path.join(ROOT, "vulva", "train_seg.py"), "--labels", labels, "--folds", "0",
                              "--epochs", "3", "--no-pretrained", "--inputs", "fusion", "--out-model", model_d,
                              "--out-dir", os.path.join(tmp, "seg_d")], capture_output=True, text=True, cwd=tmp)
        assert out.returncode == 0, out.stderr[-2000:]
        assert "(inputs: fusion" in out.stdout
        import torch
        assert torch.load(model_d, map_location="cpu", weights_only=False)["inputs"] == "fusion"
        assert "presence_threshold" in torch.load(model, map_location="cpu", weights_only=False)
        out = subprocess.run([PY, os.path.join(ROOT, "vulva", "run_vulva.py"), pred_csv, "--model", model_d,
                              "--out-dir", os.path.join(tmp, "out_d"), "--min-frames", "1"],
                             capture_output=True, text=True, cwd=tmp)
        assert out.returncode == 0, out.stderr[-2000:]
        # nothing written next to the raw frames
        assert all(not n.endswith(".png") for _, _, fs in os.walk(data) for n in fs)
    finally:
        shutil.rmtree(tmp)


def test_ellipse_from_three_clicks():
    from label_ellipse import ellipse_from_clicks, ellipse_mask
    e = ellipse_from_clicks((300, 200), (300, 260), (320, 230))      # vertical, 60 long, 40 wide
    assert e[0] == 300 and e[1] == 230 and e[2] == 30 and abs(e[3] - 20) < 1e-9 and abs(e[4] - 90) < 1e-9
    m = ellipse_mask(e) > 0
    ys, xs = np.nonzero(m)
    assert ys.min() == 200 and ys.max() == 260 and xs.min() == 280 and xs.max() == 320
    assert abs(m.sum() - np.pi * 30 * 20) / (np.pi * 30 * 20) < 0.05


def test_model_mask_to_clicks_round_trip():
    """A predicted mask becomes three clicks that redraw the same ellipse (model-assisted labelling)."""
    from label_ellipse import clicks_from_mask, ellipse_from_clicks, ellipse_mask
    assert clicks_from_mask(np.zeros((480, 640), np.uint8)) is None
    for ang in (90, 75, 110, 30, 0):
        truth = (330.0, 240.0, 40.0, 22.0, float(ang))
        clicks = clicks_from_mask(ellipse_mask(truth))
        top, bottom, _ = clicks
        assert top[1] <= bottom[1]
        e = ellipse_from_clicks(*clicks)
        assert abs(e[0] - 330) < 1.5 and abs(e[1] - 240) < 1.5, (ang, e)
        assert abs(e[2] - 40) < 1.5 and abs(e[3] - 22) < 1.5, (ang, e)
        a = ellipse_mask(truth) > 0
        b = ellipse_mask(e) > 0
        assert (a & b).sum() / (a | b).sum() > 0.93, (ang, e)


def test_presence_cut_reaches_target_precision():
    """The 'vulva visible' cut is the lowest blob confidence whose precision reaches the target."""
    from train_seg import pick_presence_cut
    vis, none = np.ones((4, 4), np.uint8), np.zeros((4, 4), np.uint8)
    items = [{"mask": vis}] * 7 + [{"mask": none}] * 4             # 7 real vulvas, 4 frames without
    pred = [vis] * 6 + [none] + [vis] * 4                           # one missed, four false blobs
    confs = [.9, .8, .7, .6, .5, .4, 0, .65, .45, .35, .3]
    assert pick_presence_cut(items, pred, confs, 0.8) == 0.5        # 5 real, 1 false kept
    assert pick_presence_cut(items, pred, confs, 1.0) == 0.7
    assert pick_presence_cut(items, pred, confs, 0) == 0.0
    assert pick_presence_cut(items, pred, [.1] * 6 + [0] + [.95] * 4, 0.9) is None


def test_saved_ellipse_round_trip_for_review():
    """Review mode redraws a saved ellipse from its stored parameters; Enter must save the same one."""
    from label_ellipse import ellipse_from_clicks, saved_clicks
    for ang in (90.0, 75.0, 110.0, -80.0):
        row = {"cx": "330", "cy": "250", "half_length_px": "38", "half_width_px": "21", "angle_deg": str(ang)}
        e = ellipse_from_clicks(*saved_clicks(row))
        assert abs(e[0] - 330) < 1e-6 and abs(e[1] - 250) < 1e-6 and abs(e[2] - 38) < 1e-6 and abs(e[3] - 21) < 1e-6
        assert abs(((e[4] - ang) + 90) % 180 - 90) < 1e-6          # same axis (direction may flip)
    assert saved_clicks({"cx": "", "cy": ""}) is None


def test_review_mode_keeps_or_changes_saved_labels():
    """--review: Enter keeps the saved ellipse (how unchanged), t turns it into 'tail' (not visible)."""
    tmp = tempfile.mkdtemp()
    try:
        rng = np.random.default_rng(3)
        folder = os.path.join(tmp, "20260209-09-00-00")
        os.makedirs(folder)
        fields = ["timestamp_folder", "pig_id", "pig_timestamp", "ir_path", "depth_path", "posture_confidence"]
        frames, labels = [], []
        for pid in range(3):
            ts = f"20260209-09-00-{10 + pid:02d}"
            _write_frame(folder, pid, ts, True, rng)
            ir = f"20260209-09-00-00/pig{pid}_ir_vis_{ts}.jpg"
            frames.append({"timestamp_folder": "20260209-09-00-00", "pig_id": pid, "pig_timestamp": ts, "ir_path": ir,
                           "depth_path": "", "posture_confidence": 0.9})
            labels.append({"ir_path": ir, "label": "present", "pig_id": pid,
                           "timestamp_folder": "20260209-09-00-00", "pig_timestamp": ts, "cx": 320, "cy": 250,
                           "half_length_px": 30, "half_width_px": 18, "angle_deg": 90, "how": "accepted",
                           "mask_path": f"masks/old{pid}.png"})
        for name, rows in (("label_set.csv", frames), ("labels.csv", labels)):
            with open(os.path.join(tmp, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        driver = os.path.join(tmp, "drive.py")
        with open(driver, "w") as f:
            f.write(
                "import sys, matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
                "from matplotlib.backend_bases import KeyEvent\n"
                "def show():\n    fig = plt.gcf()\n"
                "    for k in ('enter', 't', 'q'):\n"
                "        fig.canvas.callbacks.process('key_press_event', KeyEvent('key_press_event', fig.canvas, k))\n"
                "plt.show = show\n"
                f"sys.path.insert(0, {os.path.join(ROOT, 'vulva')!r})\n"
                f"sys.argv = ['label_ellipse.py', {tmp!r}, '--review', 'r6', '--no-depth', '--no-rgb']\n"
                "import label_ellipse\nlabel_ellipse.main()\n")
        out = subprocess.run([PY, driver], capture_output=True, text=True, cwd=tmp)
        assert out.returncode == 0, out.stderr[-2000:]
        assert "Review r6: 3 to go" in out.stdout and "1 kept, 1 tail" in out.stdout, out.stdout
        with open(os.path.join(tmp, "labels.csv")) as f:
            rows = {r["pig_id"]: r for r in csv.DictReader(f)}
        assert rows["0"]["review"] == "r6 kept" and rows["0"]["how"] == "accepted" and rows["0"]["label"] == "present"
        assert rows["0"]["mask_path"] == "masks/old0.png"                  # kept: the saved mask is not rewritten
        assert rows["1"]["label"] == "absent" and rows["1"]["note"] == "tail" and rows["1"]["review"] == "r6 tail"
        assert rows["2"]["review"] == "" and rows["2"]["label"] == "present"
    finally:
        shutil.rmtree(tmp)


def test_rgb_align_recovers_a_known_shift():
    """rgb_align: a colour image made from the IR with a known offset is put back on the IR grid;
    register_rgb.py writes it into a label set without touching frames or labels."""
    from rgb_align import MAP, register
    from register_rgb import register_set
    rng = np.random.default_rng(11)
    ir = cv2.GaussianBlur(rng.integers(0, 255, (480, 640)).astype(np.uint8), (0, 0), 6)
    ir = cv2.normalize(ir, None, 0, 255, cv2.NORM_MINMAX)
    for _ in range(25):
        cv2.ellipse(ir, (int(rng.integers(120, 520)), int(rng.integers(80, 420))),
                    (int(rng.integers(8, 30)), int(rng.integers(8, 30))), 0, 0, 360, int(rng.integers(0, 255)), -1)
    s, dx, dy = MAP
    true = (21, 4)
    A = np.array([[1 / s, 0, -dx / s - true[0]], [0, 1 / s, -dy / s - true[1]]], np.float32)
    rgb = cv2.cvtColor(cv2.warpAffine(ir, A, (1280, 800), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP),
                       cv2.COLOR_GRAY2BGR)
    img, shift, score, how = register(ir, rgb, (330, 250))
    assert how == "vulva" and abs(shift[0] - true[0]) <= 1 and abs(shift[1] - true[1]) <= 1, (shift, score)
    back = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[100:400, 150:500].astype(float)
    assert np.abs(back - ir[100:400, 150:500]).mean() < 6
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "f"))
        cv2.imwrite(os.path.join(tmp, "f", "pig1_ir_vis_20260213-09-00-11.jpg"), ir)
        cv2.imwrite(os.path.join(tmp, "f", "pig1_rgb_20260213-09-00-11.jpg"), rgb)
        with open(os.path.join(tmp, "label_set.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["timestamp_folder", "pig_id", "pig_timestamp", "ir_path", "rgb_path"])
            w.writeheader()
            w.writerow({"timestamp_folder": "f", "pig_id": 1, "pig_timestamp": "20260213-09-00-11",
                        "ir_path": "f/pig1_ir_vis_20260213-09-00-11.jpg", "rgb_path": "f/pig1_rgb_20260213-09-00-11.jpg"})
        counts = register_set(tmp)
        assert counts["none"] == 0 and sum(counts.values()) == 1
        with open(os.path.join(tmp, "label_set.csv")) as f:
            row = next(csv.DictReader(f))
        assert row["rgb_aligned_path"] == "f/pig1_rgbreg_20260213-09-00-11.jpg"
        assert cv2.imread(os.path.join(tmp, row["rgb_aligned_path"])).shape == (480, 640, 3)
        assert os.path.exists(os.path.join(tmp, "label_set_before_rgbreg.csv"))
    finally:
        shutil.rmtree(tmp)


def test_depth_input_stack():
    """ir+depth input: channel 0 is the IR, channel 1 height (128 = level skin, higher on the bump),
    channel 2 shading; missing depth gives a flat, cue-free stack."""
    from seg_model import make_input, to_model_space, to_tensor
    depth, mask = render(tilt_deg=20, dist=700, W=40, L=70, H=15, noise=0.5, cx=320, cy=240)
    ir = np.full((480, 640), 90, np.uint8)
    x = make_input(ir, depth, "ir+depth")
    assert x.shape == (480, 640, 3) and (x[..., 0] == 90).all()
    assert np.median(x[..., 1][mask]) > 150 and abs(int(np.median(x[..., 1][380:450, 400:560])) - 128) < 10
    flat = make_input(ir, None, "ir+depth")
    assert (flat[..., 1] == 128).all() and (flat[..., 2] == 0).all()
    assert make_input(ir, depth, "ir") is ir
    im, _ = to_model_space(x)
    assert to_tensor(im).shape == (3, 384, 320) and to_tensor(to_model_space(ir)[0]).shape == (3, 384, 320)
    # the new inputs: depth only (height, shape index, curvedness) and fusion (IR + height + shape index)
    xd, xf = make_input(ir, depth, "depth"), make_input(ir, depth, "fusion")
    assert xd.shape == (480, 640, 3) and xf.shape == (480, 640, 3) and (xf[..., 0] == 90).all()
    assert np.median(xd[..., 0][mask]) > 140 and abs(int(np.median(xd[..., 0][380:450, 400:560])) - 128) < 12
    assert (xf[..., 1] == xd[..., 0]).all() and (xf[..., 2] == xd[..., 1]).all()
    assert make_input(ir, None, "depth") is None and make_input(ir, None, "fusion") is None
    assert to_tensor(to_model_space(xf)[0], "fusion").shape == (5, 384, 320)
    assert to_tensor(to_model_space(xd)[0], "depth").shape == (3, 384, 320)


def test_depth_features_tell_a_dome_from_a_ridge():
    """depth_features: the shape index reads a round bump (vulva) as a dome (near +1), a long
    narrow bump (tail) as a ridge (near +0.5) and flat skin as neutral (128); height is high on both."""
    from seg_model import depth_features
    yy, xx = np.mgrid[0:480, 0:640].astype(np.float64)
    px = 800 / 471.583                                              # mm per pixel at 800 mm
    dome = 10 * np.exp(-(((xx - 260) ** 2 + (yy - 260) ** 2) * px ** 2) / (2 * 15.0 ** 2))
    ridge = 10 * np.exp(-((xx - 400) * px) ** 2 / (2 * 8.0 ** 2)) * ((yy > 120) & (yy < 400))
    z = 800 - dome - ridge + np.random.default_rng(1).normal(0, 0.3, xx.shape)
    h8, s8, c8 = depth_features(np.clip(z, 0, 65535).astype(np.uint16))
    si = lambda y, x: (float(s8[y, x]) / 255) * 2 - 1                 # noqa: E731
    assert si(260, 260) > 0.75, si(260, 260)
    assert 0.3 < si(260, 400) < 0.7, si(260, 260)
    assert abs(si(420, 160)) < 0.1 and h8[260, 260] > 150 and h8[260, 400] > 150
    assert c8[260, 260] > c8[420, 160]


def test_fusion_model_starts_from_the_ir_filters():
    """VulvaUNet(in_ch=5): the two depth channels of the first layer start at zero (the network
    starts as the IR model); a fusion checkpoint loads back as 5 channels with its inputs, and
    predict_parts skips frames without an input (no depth file) instead of failing."""
    import torch
    from seg_model import VulvaUNet, load_checkpoint, make_input, predict_parts, save_checkpoint
    m = VulvaUNet(pretrained=False, n_out=3, in_ch=5)
    w = m.stem[0].weight.detach()
    assert w.shape[1] == 5 and float(w[:, 3:].abs().sum()) == 0 and float(w[:, :3].abs().sum()) > 0
    assert m(torch.zeros(1, 5, 384, 320)).shape == (1, 3, 384, 320)
    tmp = tempfile.mkdtemp()
    try:
        ck = os.path.join(tmp, "f.pth")
        save_checkpoint(m, ck, {"threshold": 0.5, "min_area": 25, "inputs": "fusion", "classes": 3})
        m2, ckpt = load_checkpoint(ck)
        assert m2.in_ch == 5 and m2.inputs == "fusion" and ckpt["inputs"] == "fusion"
        depth, _ = render(tilt_deg=20, dist=700, W=40, L=70, H=15, noise=0.5, cx=320, cy=240)
        x = make_input(np.full((480, 640), 90, np.uint8), depth, "fusion")
        out = predict_parts(m2, [x, None])
        assert len(out) == 2 and out[0]["vulva"].shape == (480, 640) and out[1].get("no_input")
    finally:
        shutil.rmtree(tmp)


def test_depth_views_show_the_bump_and_drop_the_background():
    """Height view: the vulva bump reads as several mm proud of the skin, flat skin near 0,
    and stall background far behind the sow is masked out (NaN)."""
    from depth_views import relief, shade_image, height_image
    depth, mask = render(tilt_deg=20, dist=700, W=40, L=70, H=15, noise=1.0, cx=320, cy=240)
    depth = depth.copy()
    depth[:, :150] = 2000                                    # far stall wall at the left
    shade, height, zc = relief(depth)
    assert 600 < zc < 850, zc
    assert np.isnan(height[:, :140]).all()                   # background is not part of the sow
    ys, xs = np.nonzero(mask)
    peak = np.nanmedian(height[ys.min() + (ys.max() - ys.min()) // 3: ys.max() - (ys.max() - ys.min()) // 3,
                               xs.min() + 5: xs.max() - 5])
    skin = np.nanmedian(np.abs(height[380:450, 400:560]))
    assert peak > 5 and skin < 1.5, (peak, skin)
    assert shade_image(shade).shape == (480, 640, 3) and height_image(height).dtype == np.uint8


def test_add_rgb_keeps_frames_and_labels():
    """--add-rgb copies pigN_rgb_aligned / pigN_rgb next to the frames already in the set and
    adds their columns; the frame list and labels.csv stay as they were."""
    from export_label_set import export, add_rgb
    tmp = tempfile.mkdtemp()
    try:
        rng = np.random.default_rng(9)
        raw = os.path.join(tmp, "raw", "20260210-09-00-00")
        os.makedirs(raw)
        preds = []
        for pid in range(3):
            ts = f"20260210-09-00-{10 + pid:02d}"
            _write_frame(raw, pid, ts, True, rng)
            cv2.imwrite(os.path.join(raw, f"pig{pid}_rgb_{ts}.jpg"), np.full((720, 1280, 3), 90, np.uint8))
            if pid != 1:                                   # one frame without an aligned image
                cv2.imwrite(os.path.join(raw, f"pig{pid}_rgb_aligned_{ts}.jpg"), np.full((480, 640, 3), 120, np.uint8))
            preds.append({"timestamp_folder": "20260210-09-00-00", "pig_id": pid, "pig_timestamp": ts,
                          "prediction_name": "standing", "confidence": 0.9, "id_status": "ok",
                          "image_path": os.path.join(raw, f"pig{pid}_ir_vis_{ts}.jpg")})
        out = os.path.join(tmp, "set")
        rows, _ = export(preds, out, per_day=3)
        assert "rgb_path" not in rows[0]
        with open(os.path.join(out, "labels.csv"), "w") as f:
            f.write("ir_path,label\n" + rows[0]["ir_path"] + ",absent\n")
        before = open(os.path.join(out, "label_set.csv")).read().splitlines()
        n, counts, no_source = add_rgb(preds, out)
        assert n == 3 and no_source == 0 and counts == {"rgb_aligned_path": 2, "rgb_path": 3}, counts
        with open(os.path.join(out, "label_set.csv"), newline="") as f:
            new = list(csv.DictReader(f))
        assert [r["ir_path"] for r in new] == [ln.split(",")[3] for ln in before[1:]]
        assert all(os.path.exists(os.path.join(out, r["rgb_path"])) for r in new)
        assert new[1]["rgb_aligned_path"] == "" and os.path.exists(os.path.join(out, new[0]["rgb_aligned_path"]))
        assert open(os.path.join(out, "labels.csv")).read().count("absent") == 1
        assert len(os.listdir(raw)) == 3 * 3 + 2                    # raw folder only read: ir, depth, rgb (+2 aligned)
    finally:
        shutil.rmtree(tmp)


def test_export_label_set_and_train_on_relative_paths():
    from export_label_set import export
    from train_seg import load_labels
    tmp = tempfile.mkdtemp()
    try:
        rng = np.random.default_rng(5)
        raw = os.path.join(tmp, "raw", "20260209-09-00-00")
        os.makedirs(raw)
        preds = []
        for pid in range(3):
            ts = f"20260209-09-00-{10 + pid:02d}"
            _write_frame(raw, pid, ts, True, rng)
            preds.append({"timestamp_folder": "20260209-09-00-00", "pig_id": pid, "pig_timestamp": ts,
                          "prediction_name": "standing" if pid < 2 else "lying", "confidence": 0.9,
                          "id_status": "ok", "image_path": os.path.join(raw, f"pig{pid}_ir_vis_{ts}.jpg")})
        out = os.path.join(tmp, "set")
        rows, missing = export(preds, out, per_day=3)
        assert len(rows) == 2 and missing == 0
        assert os.path.exists(os.path.join(out, rows[0]["depth_path"]))
        assert not os.path.isabs(rows[0]["ir_path"])
        # labels written the way label_ellipse.py writes them: relative to the set folder
        os.makedirs(os.path.join(out, "masks"))
        m = np.zeros((480, 640), np.uint8)
        m[250:270, 310:325] = 255
        cv2.imwrite(os.path.join(out, "masks", "a.png"), m)
        with open(os.path.join(out, "labels.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["ir_path", "mask_path", "label", "pig_id",
                                              "timestamp_folder", "pig_timestamp"])
            w.writeheader()
            w.writerow({"ir_path": rows[0]["ir_path"], "mask_path": "masks/a.png", "label": "present",
                        "pig_id": 0, "timestamp_folder": rows[0]["timestamp_folder"],
                        "pig_timestamp": rows[0]["pig_timestamp"]})
            w.writerow({"ir_path": rows[1]["ir_path"], "mask_path": "", "label": "absent", "pig_id": 1,
                        "timestamp_folder": rows[1]["timestamp_folder"], "pig_timestamp": rows[1]["pig_timestamp"]})
        items = load_labels(os.path.join(out, "labels.csv"))
        assert len(items) == 2 and items[0]["mask"].any() and not items[1]["mask"].any()
    finally:
        shutil.rmtree(tmp)


def test_eval_seg_scores_a_model_end_to_end():
    """eval_seg.py: a model against hand labels it was not trained on, with sizes in mm from both masks."""
    from eval_seg import evaluate, read_rows, summarise
    from seg_model import VulvaUNet
    tmp = tempfile.mkdtemp()
    try:
        rng = np.random.default_rng(7)
        folder = os.path.join(tmp, "set", "20260213-09-00-00")
        os.makedirs(folder)
        os.makedirs(os.path.join(tmp, "set", "masks"))
        rows = []
        for pid in range(4):
            ts = f"20260213-09-00-{10 + pid:02d}"
            present = pid < 3
            m = _write_frame(folder, pid, ts, present, rng)
            mp = ""
            if present:
                mp = f"masks/{pid}.png"
                cv2.imwrite(os.path.join(tmp, "set", mp), m.astype(np.uint8) * 255)
            rows.append({"ir_path": f"20260213-09-00-00/pig{pid}_ir_vis_{ts}.jpg", "mask_path": mp,
                         "label": "present" if present else "absent", "pig_id": pid,
                         "timestamp_folder": "20260213-09-00-00", "pig_timestamp": ts, "how": "hand"})
        labels = os.path.join(tmp, "set", "labels.csv")
        with open(labels, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        model = VulvaUNet(pretrained=False).eval()
        ckpt = {"threshold": 0.5, "min_area": 25, "presence_threshold": 0.0, "inputs": "ir"}
        frames = evaluate(model, ckpt, read_rows(labels), "cpu")
        assert len(frames) == 4 and {f["verdict"] for f in frames} <= {
            "false alarm", "missed", "dropped by cut", "both visible", "both not visible"}
        assert all(f["human_status"] == "ok" for f in frames if f["label"] == "present")
        assert abs(float(frames[0]["human_width_mm"]) - 40) < 8          # hand mask measured on the depth
        report = summarise(frames, {"conf_high": 0.0, "conf_low": 0.0, "min_peak_mm": 0.0})
        assert report[0].startswith("4 labelled frames: 3 with a visible vulva, 1 without")
        assert report[1].startswith("Mask mAP50 ")
        from seg_model import save_checkpoint
        ck = os.path.join(tmp, "m.pth")
        save_checkpoint(model, ck, ckpt)
        out = subprocess.run([PY, os.path.join(ROOT, "vulva", "eval_seg.py"), "--model", ck, "--labels", labels,
                              "--out-dir", os.path.join(tmp, "ev")], capture_output=True, text=True, cwd=tmp)
        assert out.returncode == 0, out.stderr[-2000:]
        assert all(os.path.exists(os.path.join(tmp, "ev", n)) for n in ("eval_report.txt", "eval_frames.csv",
                                                                         "eval_sheet.jpg"))
        with open(os.path.join(tmp, "ev", "eval_frames.csv")) as f:
            assert "model_peak_height_mm" in next(csv.reader(f))       # keep_rule.py tunes on it
    finally:
        shutil.rmtree(tmp)


def test_mask_map_coco_style():
    """eval_seg.py mAP: perfect masks give 1.0; a false outline ranked first costs precision;
    outlines that overlap by IoU 0.72 count up to the 0.70 threshold only (mAP50-95 = 0.5)."""
    from eval_seg import map_scores
    fr = [{"label": "present", "model_visible_before_cut": 1, "blob_conf": 0.9 + 0.01 * i, "iou": 0.97}
          for i in range(4)]
    fr += [{"label": "absent", "model_visible_before_cut": 0, "blob_conf": 0, "iou": ""} for _ in range(2)]
    assert map_scores(fr)[:2] == (1.0, 1.0)
    fa = fr + [{"label": "absent", "model_visible_before_cut": 1, "blob_conf": 0.999, "iou": ""}]
    assert abs(map_scores(fa)[1] - 0.8) < 1e-6
    loose = [dict(f, iou=0.72) if f["label"] == "present" else f for f in fr]
    m50, m5095, by_t = map_scores(loose)
    assert m50 == 1.0 and abs(m5095 - 0.5) < 1e-9 and by_t[0.75] == 0.0
    missed = fr[:2] + [dict(f, model_visible_before_cut=0, iou="") for f in fr[2:4]] + fr[4:]
    assert abs(map_scores(missed)[1] - 51 / 101) < 1e-9          # half the vulvas never drawn: recall stops at 0.5

def test_keep_rule_bump_check_tunes_applies_and_saves():
    """keep_rule.py: a fairly sure blob on a bump is kept, a fairly sure flat one is not; the rule is
    tuned on two tests, saved into a checkpoint copy and re-applied to a run's vulva_frames.csv."""
    from keep_rule import apply_to_run, keep, rule_from_checkpoint, score, tune
    old = rule_from_checkpoint({"presence_threshold": 0.98})
    assert old == {"conf_high": 0.98, "conf_low": 0.98, "min_peak_mm": 0.0}
    rule = {"conf_high": 0.98, "conf_low": 0.7, "min_peak_mm": 6.0}
    assert keep(0.99, None, False, rule) and keep(0.8, 9.0, True, rule)
    assert not keep(0.8, 3.0, True, rule) and not keep(0.8, 9.0, False, rule) and not keep(0.5, 9.0, True, rule)

    def make_test(seed, n=60):
        rng = np.random.default_rng(seed)
        out = []
        for i in range(n):                       # real vulvas: half sure, half only fairly sure, all on a bump
            out.append({"present": True, "blob": True, "conf": float(rng.uniform(0.985, 1.0) if i % 2 else
                                                                      rng.uniform(0.75, 0.97)),
                        "peak": float(rng.uniform(7, 15)), "measured": True, "iou": 0.7, "tail": False})
        for i in range(n):                       # tails and skin: fairly sure but flat, a few very sure
            out.append({"present": False, "blob": i % 3 != 0, "conf": float(rng.uniform(0.75, 0.97) if i % 10
                                                                            else 0.99),
                        "peak": float(rng.uniform(0, 4)), "measured": True, "iou": None, "tail": True})
        return out
    tests = [make_test(1), make_test(2)]
    tuned = tune(tests, precision=0.9)
    assert tuned["conf_low"] < tuned["conf_high"] and 4 <= tuned["min_peak_mm"] <= 7, tuned
    for t in tests:
        a, b = score(t, old), score(t, tuned)
        assert b["recall"] > a["recall"] + 0.3 and b["precision"] >= 0.9, (a, b)

    tmp = tempfile.mkdtemp()
    try:
        evs = []
        for k, t in enumerate(tests):            # the same frames as eval_seg.py would write them
            p = os.path.join(tmp, f"ev{k}", "eval_frames.csv")
            os.makedirs(os.path.dirname(p))
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["label", "note", "model_visible_before_cut", "blob_conf",
                                                  "model_peak_height_mm", "model_status", "iou"])
                w.writeheader()
                for fr in t:
                    w.writerow({"label": "present" if fr["present"] else "absent", "note": "tail" * fr["tail"],
                                "model_visible_before_cut": int(fr["blob"]), "blob_conf": fr["conf"],
                                "model_peak_height_mm": fr["peak"], "model_status": "ok",
                                "iou": fr["iou"] if fr["iou"] is not None else ""})
            evs.append(p)
        import torch
        ck, ck2 = os.path.join(tmp, "m.pth"), os.path.join(tmp, "m_keep.pth")
        torch.save({"presence_threshold": 0.98, "state_dict": {}}, ck)
        out = subprocess.run([PY, os.path.join(ROOT, "vulva", "keep_rule.py"), *evs, "--model", ck,
                              "--precision", "0.9", "--write", ck2], capture_output=True, text=True, cwd=tmp)
        assert out.returncode == 0, out.stderr[-2000:]
        assert "Carry-over check" in out.stdout and "tail frames kept" in out.stdout
        saved = torch.load(ck2, weights_only=False)
        assert saved["keep_rule"]["conf_low"] == tuned["conf_low"] and saved["presence_threshold"] == 0.98
        assert "keep_rule" not in torch.load(ck, weights_only=False)        # original untouched

        # re-apply to a run: a dropped blob on a bump comes back, a flat one stays dropped
        run = os.path.join(tmp, "vulva_frames.csv")
        base = {"timestamp_folder": "20260213-09-00-00", "pig_id": 1, "width_mm": 40, "length_mm": 70,
                "plane_rmse_mm": 1, "view_angle_deg": 10, "depth_coverage": 0.9}
        rows = [dict(base, pig_timestamp=f"t{i}", vulva_visible=0, vulva_conf=0.85, peak_height_mm=10,
                     measure_status="low_confidence", blob_status="ok") for i in range(3)]
        rows += [dict(base, pig_timestamp="flat", vulva_visible=0, vulva_conf=0.85, peak_height_mm=1,
                      measure_status="low_confidence", blob_status="ok"),
                 dict(base, pig_timestamp="none", vulva_visible=0, vulva_conf=0, measure_status="not_visible")]
        with open(run, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
        kept, n, days = apply_to_run(run, tuned)
        assert (kept, n, days) == (3, 5, 1)
        with open(os.path.join(tmp, "vulva_frames_rekept.csv")) as f:
            st = {r["pig_timestamp"]: r["measure_status"] for r in csv.DictReader(f)}
        assert st == {"t0": "ok", "t1": "ok", "t2": "ok", "flat": "low_confidence", "none": "not_visible"}
    finally:
        shutil.rmtree(tmp)


def test_size_vs_caliper_ratios_and_within_sow_tracking():
    """size_vs_caliper.py: hand outlines 15% big and a model that tracks the calipers come out as such."""
    from size_vs_caliper import compare, sources_from
    rng = np.random.default_rng(3)
    truth, rows = {}, []
    for pig in range(6):
        for d in range(6, 12):
            date = f"2026-02-{d:02d}"
            w, l = 38 + pig + (d - 6) * 1.5, 70 + 2 * pig + (d - 6) * 2.5      # each sow grows towards heat
            truth[(pig, date)] = (w, l)
            for k in range(2):
                rows.append({"timestamp_folder": f"202602{d:02d}-09-00-00", "pig_id": pig, "label": "present",
                             "human_status": "ok", "human_width_mm": w * 1.15 + rng.normal(0, 0.5),
                             "human_length_mm": l * 1.15 + rng.normal(0, 0.5), "model_visible": 1,
                             "model_status": "ok", "model_width_mm": w + rng.normal(0, 0.5),
                             "model_length_mm": l + rng.normal(0, 0.5)})
    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, "ev", "eval_frames.csv")
        os.makedirs(os.path.dirname(p))
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        (hn, hand, _), (mn, model, _) = sources_from(p)
        _, sh = compare(hand, truth)
        _, sm = compare(model, truth)
        assert sh["sow_days"] == 36 and abs(sh["width_mm"]["ratio"] - 1.15) < 0.02, sh
        assert abs(sm["length_mm"]["ratio"] - 1.0) < 0.02 and sm["length_mm"]["within_r"] > 0.9, sm
    finally:
        shutil.rmtree(tmp)


def test_pick_hard_frames_buckets_holdouts_and_export():
    """pick_hard_frames.py: unsure / borderline / nothing / sure frames, spread over sows and days, never
    from held-out test days or test sows or already-labelled frames; export_label_set.py copies them."""
    import random as _random
    from export_label_set import export
    from pick_hard_frames import parse_holdout, parse_ranges, pick
    tmp = tempfile.mkdtemp()
    try:
        rows = []
        for day in ("20260124", "20260125", "20260213", "20260302"):
            for pig in range(6):
                for k in range(6):
                    ts = f"{day}-09-{10 + k:02d}-{pig:02d}"
                    folder = os.path.join(tmp, "raw", f"{day}-09-00-00")
                    os.makedirs(folder, exist_ok=True)
                    ir = os.path.join(folder, f"pig{pig}_ir_vis_{ts}.jpg")
                    cv2.imwrite(ir, np.zeros((8, 8), np.uint8))
                    status, vis, conf = [("low_confidence", 0, 0.95), ("not_visible", 0, 0.0), ("ok", 1, 0.990),
                                         ("ok", 1, 0.9995), ("ok", 1, 0.997), ("low_confidence", 0, 0.9)][k]
                    rows.append({"timestamp_folder": f"{day}-09-00-00", "pig_id": str(pig), "pig_timestamp": ts,
                                 "posture_confidence": "0.9", "measure_status": status, "vulva_visible": str(vis),
                                 "vulva_conf": str(conf), "ir_path": ir})
        done = rows[0]
        lab = os.path.join(tmp, "labels.csv")
        with open(lab, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["pig_id", "pig_timestamp"])
            w.writeheader()
            w.writerow({"pig_id": done["pig_id"], "pig_timestamp": done["pig_timestamp"]})
        from pick_hard_frames import labelled_keys
        picked, pools = pick(rows, 0.9888, {"unsure": 10, "borderline": 4, "nothing": 4, "sure": 2},
                             _random.Random(0), parse_ranges("20260213-20260213,20260302"),
                             parse_holdout("0,1@20260123-20260205"), labelled_keys([lab]))
        days = {r["pig_timestamp"][:8] for r in picked}
        assert days <= {"20260124", "20260125"}, days                      # test days never picked
        assert not any(r["pig_id"] in ("0", "1") for r in picked)          # January test sows never picked
        assert pools == {"unsure": 16, "borderline": 8, "nothing": 8, "sure": 8}, pools
        got = Counter(r["bucket"] for r in picked)
        assert got == {"unsure": 10, "borderline": 4, "nothing": 4, "sure": 2}, got
        assert len({(r["pig_id"], r["pig_timestamp"][:8]) for r in picked if r["bucket"] == "unsure"}) == 8
        preds = [dict(r, prediction_name="standing", confidence="0.9", image_path=r["ir_path"]) for r in picked]
        out_rows, missing = export(preds, os.path.join(tmp, "set"), None, 0, 0.6, None, False)
        assert len(out_rows) == 20 and missing == 0
        assert os.path.exists(os.path.join(tmp, "set", "label_set.csv"))
    finally:
        shutil.rmtree(tmp)


def test_retune_cut_on_normal_frames_only():
    """retune_cut.py: hand-picked hard frames (tails at high confidence) push the cut up; left out,
    the cut set on the normal frames is lower and keeps more vulvas; a checkpoint copy gets it."""
    from retune_cut import at_cut, cut_for, read_cv
    tmp = tempfile.mkdtemp()
    try:
        rows, hard = [], []
        for i in range(60):                                    # normal: vulvas 0.95-0.999, absent mostly undrawn
            rows.append({"pig_id": i % 10, "pig_timestamp": f"n{i}", "label": "present", "pred_area_px": 300,
                         "blob_conf": 0.95 + 0.049 * (i / 59)})
            rows.append({"pig_id": i % 10, "pig_timestamp": f"a{i}", "label": "absent",
                         "pred_area_px": 200 if i < 6 else 0, "blob_conf": 0.96 if i < 6 else 0})
        for i in range(30):                                    # hand-picked: tails drawn with high confidence
            rows.append({"pig_id": i % 10, "pig_timestamp": f"h{i}", "label": "absent", "pred_area_px": 250,
                         "blob_conf": 0.95 + 0.04 * (i / 29)})
            hard.append({"pig_id": i % 10, "pig_timestamp": f"h{i}"})
        cv = os.path.join(tmp, "cv_predictions.csv")
        with open(cv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        lab = os.path.join(tmp, "labels.csv")
        with open(lab, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["pig_id", "pig_timestamp"])
            w.writeheader()
            w.writerows(hard)
        excl = {(str(h["pig_id"]), h["pig_timestamp"]) for h in hard}
        c_all, c_norm = cut_for(read_cv(cv), 0.9), cut_for(read_cv(cv, excl), 0.9)
        assert c_norm < c_all, (c_norm, c_all)
        assert at_cut(read_cv(cv, excl), c_norm)["recall"] > at_cut(read_cv(cv, excl), c_all)["recall"]
        import torch
        ck, ck2 = os.path.join(tmp, "m.pth"), os.path.join(tmp, "m2.pth")
        torch.save({"presence_threshold": c_all, "state_dict": {}}, ck)
        out = subprocess.run([PY, os.path.join(ROOT, "vulva", "retune_cut.py"), cv, "--model", ck,
                              "--exclude-labels", lab, "--write", ck2], capture_output=True, text=True, cwd=tmp)
        assert out.returncode == 0, out.stderr[-2000:]
        saved = torch.load(ck2, weights_only=False)
        assert saved["presence_threshold"] == c_norm and saved["presence_threshold_train_all"] == c_all
        assert torch.load(ck, weights_only=False)["presence_threshold"] == c_all        # original untouched
    finally:
        shutil.rmtree(tmp)


def test_label_tail_canvas_files_and_order():
    """label_tail.py: strokes (curved tail = clicks along it), polygon, brush erase and undo build
    the mask; save / load round trip; tail-labelled frames come first; covers_vulva is set."""
    from label_tail import TailCanvas, frame_order, load_tail_labels, save_tail, save_tail_labels, stroke_mask
    one = stroke_mask([(100, 100)], 10)
    assert 60 < one.astype(bool).sum() < 110                     # a round blob of diameter ~10
    cv_ = TailCanvas(width=10)
    for p in [(300, 100), (310, 160), (330, 220), (320, 280)]:   # a curved tail, four clicks along it
        cv_.click(*p)
    m = cv_.mask() > 0
    assert m[100, 300] and m[220, 330] and m[190, 320] and not m[100, 400]
    cv_.finish_part()
    cv_.set_mode("polygon")
    for p in [(200, 300), (240, 300), (240, 340), (200, 340)]:
        cv_.click(*p)
    cv_.finish_part()
    assert cv_.mask()[320, 220] > 0 and "polygon" in cv_.hows
    cv_.set_mode("brush")
    cv_.brush(220, 320, erase=True, new_drag=True)
    assert cv_.mask()[320, 220] == 0
    cv_.undo()
    assert cv_.mask()[320, 220] > 0
    tmp = tempfile.mkdtemp()
    try:
        done = {}
        vm = np.zeros((480, 640), np.uint8)
        cv2.ellipse(vm, (320, 290), (12, 20), 0, 0, 360, 255, -1)
        rows = [{"ir_path": "a/pig1_ir_vis_x.jpg", "pig_id": "1", "timestamp_folder": "a", "pig_timestamp": "t1",
                 "label": "present", "note": ""},
                {"ir_path": "a/pig2_ir_vis_x.jpg", "pig_id": "2", "timestamp_folder": "a", "pig_timestamp": "t2",
                 "label": "absent", "note": "tail"},
                {"ir_path": "a/pig3_ir_vis_x.jpg", "pig_id": "3", "timestamp_folder": "a", "pig_timestamp": "t3",
                 "label": "absent", "note": ""}]
        assert [r["pig_id"] for r in frame_order(rows, done)] == ["2", "1", "3"]
        out = save_tail(tmp, done, rows[0], cv_.mask(), vm, "stroke+polygon", 10)
        assert out["tail"] == 1 and out["covers_vulva"] == 1
        save_tail(tmp, done, rows[2], None)
        path = os.path.join(tmp, "tail_labels.csv")
        save_tail_labels(path, done)
        back = load_tail_labels(path)
        assert back["a/pig1_ir_vis_x.jpg"]["tail"] == "1" and back["a/pig3_ir_vis_x.jpg"]["tail"] == "0"
        assert cv2.imread(os.path.join(tmp, back["a/pig1_ir_vis_x.jpg"]["tail_mask_path"]), 0).max() == 255
        assert [r["pig_id"] for r in frame_order(rows, back)] == ["2"]
    finally:
        shutil.rmtree(tmp)


def test_tail_next_to_vulva_is_not_covering_it():
    """A tail hanging right above the vulva (normal: the tail root is just above it) does not count;
    a tail lying across part of the vulva does."""
    from seg_model import covered_by_tail
    vul = np.zeros((384, 320), np.uint8)
    cv2.ellipse(vul, (160, 220), (14, 24), 0, 0, 360, 1, -1)
    above = np.zeros_like(vul)
    cv2.line(above, (160, 60), (160, 194), 1, 16)                  # ends just above the vulva top (196)
    assert not covered_by_tail(vul, above)
    across = np.zeros_like(vul)
    cv2.line(across, (160, 60), (160, 230), 1, 16)                 # hangs down over its upper half
    cut = vul & (1 - across)                                       # the model sees only the uncovered part
    assert covered_by_tail(cut, across)
    assert not covered_by_tail(vul, None) and not covered_by_tail(np.zeros_like(vul), across)

def test_three_class_loss_ignores_unlabelled_tails():
    """Frames whose tail was never outlined: swapping the background and tail scores on non-vulva
    pixels must not change the loss (an unlabelled tail is not taught as background)."""
    import torch
    from train_seg import loss_fn
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 16, 16)
    target = torch.zeros(2, 1, 16, 16)
    target[:, :, 4:8, 4:8] = 1
    swapped = logits[:, [2, 1, 0]]
    unknown = torch.zeros(2)
    a, b = loss_fn(logits, target, known=unknown), loss_fn(swapped, target, known=unknown)
    assert abs(float(a) - float(b)) < 1e-5
    known = torch.ones(2)
    assert abs(float(loss_fn(logits, target, known=known)) - float(loss_fn(swapped, target, known=known))) > 1e-3


def test_three_class_train_predict_and_eval():
    """train_seg.py --classes 3 on a set with tail outlines for part of the frames: checkpoint with
    3 classes, CV report with the tail rule, predict_parts gives a tail mask, eval_seg runs."""
    from seg_model import load_checkpoint, predict_parts, read_input
    tmp = tempfile.mkdtemp()
    try:
        rng = np.random.default_rng(5)
        set_dir = os.path.join(tmp, "set")
        os.makedirs(os.path.join(set_dir, "masks"))
        os.makedirs(os.path.join(set_dir, "tail_masks"))
        rows, tails = [], []
        for k in range(12):
            folder_name = f"20260124-09-{k:02d}-00"
            folder = os.path.join(set_dir, folder_name)
            os.makedirs(folder)
            pid = k % 6
            ts = f"20260124-09-{k:02d}-10"
            present = k % 3 != 0
            m = _write_frame(folder, pid, ts, present, rng)
            ir_rel = f"{folder_name}/pig{pid}_ir_vis_{ts}.jpg"
            ir = cv2.imread(os.path.join(set_dir, ir_rel), 0)
            tail = np.zeros((480, 640), np.uint8)
            if k % 2 == 0:                                      # a bright tail hanging above
                cv2.line(tail, (300, 60), (320, 200), 255, 14)
                ir[tail > 0] = 230
                cv2.imwrite(os.path.join(set_dir, ir_rel), ir)
            mp = ""
            if present:
                mp = f"masks/{k}.png"
                cv2.imwrite(os.path.join(set_dir, mp), m.astype(np.uint8) * 255)
            rows.append({"ir_path": ir_rel, "mask_path": mp, "label": "present" if present else "absent",
                         "pig_id": pid, "timestamp_folder": folder_name, "pig_timestamp": ts,
                         "note": "" if present else "tail"})
            if k < 8:                                            # tails looked at in 8 of 12 frames
                tp = ""
                if tail.any():
                    tp = f"tail_masks/{k}.png"
                    cv2.imwrite(os.path.join(set_dir, tp), tail)
                tails.append({"ir_path": ir_rel, "pig_id": pid, "timestamp_folder": folder_name,
                              "pig_timestamp": ts, "tail": int(tail.any()), "tail_mask_path": tp})
        for name, data in (("labels.csv", rows), ("tail_labels.csv", tails)):
            with open(os.path.join(set_dir, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
        ck = os.path.join(tmp, "m3.pth")
        out = subprocess.run([PY, os.path.join(ROOT, "vulva", "train_seg.py"), "--labels",
                              os.path.join(set_dir, "labels.csv"), "--classes", "3", "--folds", "2", "--epochs", "2",
                              "--no-pretrained", "--out-model", ck, "--out-dir", os.path.join(tmp, "o"), "--jobs", "2"],
                             capture_output=True, text=True, cwd=tmp)
        assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
        assert "tail looked at in 8 frames, outlined in 4" in out.stdout, out.stdout[-2000:]
        assert "With the tail rule" in out.stdout and "Tails (frames whose tail" in out.stdout
        model, ckpt = load_checkpoint(ck, "cpu")
        assert ckpt["classes"] == 3 and model.n_out == 3 and ckpt["presence_threshold"] == 0.0
        img = read_input(os.path.join(set_dir, rows[0]["ir_path"]))
        d = predict_parts(model, [img], "cpu", ckpt["threshold"], ckpt["min_area"])[0]
        assert d["tail"] is not None and d["tail"].shape == (480, 640) and d["vulva"].shape == (480, 640)
        out = subprocess.run([PY, os.path.join(ROOT, "vulva", "train_seg.py"), "--rescore", "--folds", "2",
                              "--out-dir", os.path.join(tmp, "o")], capture_output=True, text=True, cwd=tmp)
        assert out.returncode == 0, out.stderr[-3000:]
        assert "Re-scored 12 frames" in out.stdout and "classes 3" in out.stdout and "With the tail rule" in out.stdout
        with open(os.path.join(tmp, "o", "cv_predictions.csv")) as f:
            head = next(csv.reader(f))
        assert "covered_by_tail" in head and "tail_known" in head
        out = subprocess.run([PY, os.path.join(ROOT, "vulva", "eval_seg.py"), "--model", ck, "--labels",
                              os.path.join(set_dir, "labels.csv"), "--out-dir", os.path.join(tmp, "ev")],
                             capture_output=True, text=True, cwd=tmp)
        assert out.returncode == 0, out.stderr[-3000:]
        with open(os.path.join(tmp, "ev", "eval_frames.csv")) as f:
            assert "covered_by_tail" in next(csv.reader(f))
    finally:
        shutil.rmtree(tmp)


def test_label_tail_gui_smoke():
    """label_tail.py window, driven by simulated clicks and keys (no screen): a curved stroke with a
    width change, a polygon, a brush erase, Enter saves, x saves 'no tail', q quits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import KeyEvent, MouseEvent
    import label_tail
    tmp = tempfile.mkdtemp()
    try:
        rng = np.random.default_rng(1)
        set_dir = os.path.join(tmp, "set")
        folder = os.path.join(set_dir, "20260124-09-00-00")
        os.makedirs(folder)
        os.makedirs(os.path.join(set_dir, "masks"))
        rows = []
        for pid in range(3):
            ts = f"20260124-09-00-{10 + pid:02d}"
            m = _write_frame(folder, pid, ts, pid == 0, rng)
            mp = ""
            if pid == 0:
                mp = "masks/0.png"
                cv2.imwrite(os.path.join(set_dir, mp), m.astype(np.uint8) * 255)
            rows.append({"ir_path": f"20260124-09-00-00/pig{pid}_ir_vis_{ts}.jpg", "mask_path": mp,
                         "label": "present" if pid == 0 else "absent", "pig_id": pid,
                         "timestamp_folder": "20260124-09-00-00", "pig_timestamp": ts,
                         "note": "tail" if pid == 1 else ""})
        lab = os.path.join(set_dir, "labels.csv")
        with open(lab, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

        def drive():
            fig = plt.gcf()
            ax = fig.axes[0]
            cv = fig.canvas

            def click(x, y, button=1):
                px, py = ax.transData.transform((x, y))
                cv.callbacks.process("button_press_event", MouseEvent("button_press_event", cv, px, py, button))
                cv.callbacks.process("button_release_event", MouseEvent("button_release_event", cv, px, py, button))

            def key(k):
                cv.callbacks.process("key_press_event", KeyEvent("key_press_event", cv, k))

            # frame 1 (pig 1, labelled tail): curved stroke, wider, then a polygon, a brush erase, save
            for p in [(300, 80), (310, 140), (325, 200)]:
                click(*p)
            key("]")
            key("n")
            key("p")
            for p in [(250, 300), (280, 300), (280, 330)]:
                click(*p)
            key("n")
            key("b")
            click(325, 200, button=3)
            key("enter")
            key("x")         # frame 2 (pig 0, vulva visible): no tail
            key("q")         # quit before frame 3
        orig_show, orig_argv = plt.show, sys.argv
        plt.show = drive
        sys.argv = ["label_tail.py", lab]
        try:
            label_tail.main()
        finally:
            plt.show, sys.argv = orig_show, orig_argv
        done = label_tail.load_tail_labels(os.path.join(set_dir, "tail_labels.csv"))
        assert set(done) == {rows[1]["ir_path"], rows[0]["ir_path"]}, done
        r1 = done[rows[1]["ir_path"]]
        assert r1["tail"] == "1" and "polygon" in r1["how"] and "brush" in r1["how"]
        tm = cv2.imread(os.path.join(set_dir, r1["tail_mask_path"]), 0)
        assert tm[140, 310] == 255 and tm[320, 270] == 255 and tm[200, 325] == 0     # stroke, polygon, erased
        assert done[rows[0]["ir_path"]]["tail"] == "0"
    finally:
        shutil.rmtree(tmp)


def test_agreement_set_cvat_round_trip_and_compare():
    """make_agreement_set.py pictures -> a CVAT 'for images 1.1' export drawn on them -> import_cvat.py
    puts the ellipses back on the IR grid -> compare_labels.py: the same ellipses agree perfectly,
    shifted ones less; tags become not visible / tail; an oval drawn as a "Track" (copied by CVAT
    onto every later picture) is dropped on the copies."""
    import zipfile as _zip
    from compare_labels import compare, load
    from import_cvat import import_export
    from label_ellipse import ellipse_mask
    tmp = tempfile.mkdtemp()
    try:
        rng = np.random.default_rng(4)
        set_dir = os.path.join(tmp, "test_set")
        os.makedirs(os.path.join(set_dir, "masks"))
        rows = []
        for k in range(10):
            folder_name = f"20260213-09-{k:02d}-00"
            os.makedirs(os.path.join(set_dir, folder_name))
            pid, ts = k % 5, f"20260213-09-{k:02d}-10"
            present = k < 7
            _write_frame(os.path.join(set_dir, folder_name), pid, ts, present, rng)
            row = {"ir_path": f"{folder_name}/pig{pid}_ir_vis_{ts}.jpg", "mask_path": "",
                   "label": "present" if present else "absent", "pig_id": pid, "timestamp_folder": folder_name,
                   "pig_timestamp": ts, "cx": "", "cy": "", "half_length_px": "", "half_width_px": "",
                   "angle_deg": "", "note": "tail" if k == 9 else ""}
            if present:
                params = (300 + 3 * k, 250 + 2 * k, 26.0, 13.0, 80.0 + k)
                mp = f"masks/{k}.png"
                cv2.imwrite(os.path.join(set_dir, mp), ellipse_mask(params))
                row.update(mask_path=mp, cx=params[0], cy=params[1], half_length_px=params[2],
                           half_width_px=params[3], angle_deg=params[4])
            rows.append(row)
        ref = os.path.join(set_dir, "labels.csv")
        with open(ref, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        agr = os.path.join(tmp, "agreement")
        out = subprocess.run([PY, os.path.join(ROOT, "vulva", "make_agreement_set.py"), ref, agr, "--n", "8",
                              "--absent", "2"], capture_output=True, text=True, cwd=tmp)
        assert out.returncode == 0, out.stderr[-3000:]
        with open(os.path.join(agr, "manifest.csv")) as f:
            man = list(csv.DictReader(f))
        assert len(man) == 8 and len(os.listdir(os.path.join(agr, "images"))) == 8
        img = cv2.imread(os.path.join(agr, "images", man[0]["image"]))
        assert img.shape == (900, 1520, 3)
        assert {"vulva", "not_visible", "tail_covers"} <= {d["name"] for d in json.load(open(os.path.join(agr, "cvat_labels.json")))}
        by_ts = {r["pig_timestamp"]: r for r in rows}

        def export(path, shift=0.0, empty_one=False, track=False):
            parts = ['<?xml version="1.0" encoding="utf-8"?><annotations><version>1.1</version>']
            copy = None
            for i, m in enumerate(man):
                r = by_ts[m["pig_timestamp"]]
                parts.append(f'<image id="{i}" name="images/{m["image"]}" width="1520" height="900">')
                if copy:                                    # a "Track" drawn earlier, copied on by CVAT
                    parts.append(copy)
                if r["label"] == "present":
                    hw = 760 if i % 2 else 0                    # draw on the IR half for every other picture
                    cx = (float(r["cx"]) + shift - 120) * 2 + hw
                    cy = (float(r["cy"]) - 30) * 2
                    parts.append(f'<ellipse label="vulva" source="manual" occluded="0" cx="{cx}" cy="{cy}" '
                                 f'rx="{float(r["half_length_px"]) * 2}" ry="{float(r["half_width_px"]) * 2}" '
                                 f'rotation="{r["angle_deg"]}" z_order="0"></ellipse>')
                    if track and copy is None:
                        copy = parts[-1]
                elif not (empty_one and i == len(man) - 1):
                    lab = "tail_covers" if r["note"] == "tail" else "not_visible"
                    parts.append(f'<tag label="{lab}" source="manual"></tag>')
                parts.append("</image>")
            parts.append("</annotations>")
            with _zip.ZipFile(path, "w") as z:
                z.writestr("annotations.xml", "".join(parts))
        export(os.path.join(tmp, "same.zip"))
        export(os.path.join(tmp, "shifted.zip"), shift=8.0)
        export(os.path.join(tmp, "track.zip"), track=True)
        _, got, unknown, empty, on_colour, carried = import_export(os.path.join(tmp, "same.zip"), agr, "same")
        assert len(got) == 8 and unknown == 0 and empty == 0 and carried == 0
        sides = [r["how"].rsplit(":", 1)[1] for r in got if r["label"] == "present"]
        assert on_colour == sides.count("colour") and "ir" in sides and "colour" in sides, sides
        import_export(os.path.join(tmp, "shifted.zip"), agr, "shifted")
        _, got_t, _, _, _, carried = import_export(os.path.join(tmp, "track.zip"), agr, "track")
        first = next(i for i, m in enumerate(man) if by_ts[m["pig_timestamp"]]["label"] == "present")
        assert carried == len(man) - first - 1, carried       # every copy after the Track's own picture
        assert [(r["label"], r.get("cx")) for r in got_t] == [(r["label"], r.get("cx")) for r in got]
        keep = {(str(int(m["pig_id"])), m["pig_timestamp"]) for m in man}
        a = load(ref, keep=keep)
        s_same = compare(a, load(os.path.join(agr, "labels_same.csv"), keep=keep))
        s_shift = compare(a, load(os.path.join(agr, "labels_shifted.csv"), keep=keep))
        assert s_same["frames"] == 8 and s_same["agree"] == 1.0 and min(s_same["ious"]) > 0.95, s_same["ious"]
        assert s_same["map5095"] > 0.95 and s_same["tail_agree"] == 1.0
        assert 0.4 < np.mean(s_shift["ious"]) < 0.8 and s_shift["map5095"] < s_same["map5095"]
    finally:
        shutil.rmtree(tmp)



def test_depth_outline_finds_the_bump_like_the_paper():
    """depth_outline.py: a vulva-shaped bump on a curved, tilted rump is found from a rough seed,
    its outline comes from the bump (not the seed), a tail ridge next to it is left out when its
    mask is given, a flat area is called flat and a too-small seed is reported as clipped."""
    from depth_outline import depth_outline
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:480, 0:640].astype(np.float64)
    cx, cy, a, b = 300.0, 250.0, 14.0, 24.0                      # bump half-axes (x, y) in pixels
    base = 800 + 0.05 * (xx - 320) + 0.0015 * ((xx - 300) ** 2 + (yy - 250) ** 2)

    def ell(scale, dx=0.0, dy=0.0):
        return ((xx - cx - dx) / (a * scale)) ** 2 + ((yy - cy - dy) / (b * scale)) ** 2 <= 1

    r = np.sqrt(((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2)
    bump = 10 * np.exp(-r ** 4)
    tail = (np.abs(xx - 300) < 5) & (yy > 180) & (yy < 222)       # ridge ending just above the bump
    tail_h = np.where(tail, 9.0, 0.0)

    def frame(with_bump=True, with_tail=False):
        z = base - (bump if with_bump else 0) - (tail_h if with_tail else 0) + rng.normal(0, 0.7, base.shape)
        return np.clip(np.round(z), 0, 65535).astype(np.uint16)

    res = depth_outline(frame(), ell(1.1, 2, -2))
    assert res["status"] == "ok", res["status"]
    ex, ey, h1, h2, _ = res["ellipse"]
    big, small = max(h1, h2), min(h1, h2)
    # threshold max(2 mm, 20% of 10 mm) on 10 exp(-r^4) -> r = (ln 5)^(1/4) = 1.126
    assert abs(ex - cx) < 3 and abs(ey - cy) < 3, (ex, ey)
    assert abs(big / (1.126 * b) - 1) < 0.15 and abs(small / (1.126 * a) - 1) < 0.15, (big, small)
    assert 8 < res["peak_mm"] < 12 and res["roundness"] > 0.85 and res["clipped"] < 0.3, res

    tail_mask = tail.copy()
    res_t = depth_outline(frame(with_tail=True), ell(1.1, 2, -2), tail=tail_mask)
    assert res_t["status"] == "ok"
    big_t = max(res_t["ellipse"][2:4])
    assert abs(big_t / (1.126 * b) - 1) < 0.15, big_t

    assert depth_outline(frame(with_bump=False), ell(1.0))["status"] == "flat"
    res_c = depth_outline(frame(), ell(0.6))
    assert res_c["status"] == "ok" and res_c["clipped"] > 0.3, res_c["clipped"]


def test_build_dataset_splits_leak_checks_and_split_guard():
    """build_dataset.py: label sets become one dataset (copies + one table with a split column,
    tails and colour carried over); a frame or an animal-day in both train and test stops the
    build; train_seg / eval_seg refuse a split table without --split and use only that split."""
    tmp = tempfile.mkdtemp()
    try:
        rng = np.random.default_rng(11)

        def make_set(name, frames, tail_for=None):
            d = os.path.join(tmp, name)
            os.makedirs(os.path.join(d, "masks"))
            rows = []
            for pid, ts in frames:
                folder = ts[:9] + "00-00-00"
                os.makedirs(os.path.join(d, folder), exist_ok=True)
                m = _write_frame(os.path.join(d, folder), pid, ts, True, rng)
                mp = f"masks/{pid}_{ts}.png"
                cv2.imwrite(os.path.join(d, mp), m.astype(np.uint8) * 255)
                cv2.imwrite(os.path.join(d, folder, f"pig{pid}_rgbreg_{ts}.jpg"), np.full((480, 640, 3), 99, np.uint8))
                rows.append({"ir_path": f"{folder}/pig{pid}_ir_vis_{ts}.jpg", "mask_path": mp, "label": "present",
                             "pig_id": pid, "timestamp_folder": folder, "pig_timestamp": ts, "how": "hand"})
            with open(os.path.join(d, "labels.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            if tail_for is not None:
                r = rows[tail_for]
                os.makedirs(os.path.join(d, "tail_masks"))
                cv2.imwrite(os.path.join(d, "tail_masks", "t.png"), np.zeros((480, 640), np.uint8))
                with open(os.path.join(d, "tail_labels.csv"), "w", newline="") as f:
                    f.write("ir_path,pig_id,timestamp_folder,pig_timestamp,tail,tail_mask_path,covers_vulva,how,width_px,date\n")
                    f.write(f"{r['ir_path']},{r['pig_id']},{r['timestamp_folder']},{r['pig_timestamp']},1,"
                            f"tail_masks/t.png,0,stroke,8,2026-11-05\n")
            return os.path.join(d, "labels.csv")

        tr = make_set("feb_train", [(0, "20260210-08-00-01"), (1, "20260210-08-00-02"), (0, "20260211-08-00-01")], 0)
        te = make_set("feb_test", [(0, "20260213-08-00-01"), (2, "20260213-14-00-02")])
        bad = make_set("bad_test", [(1, "20260210-15-00-05")])                  # sow 1 on a training day
        tool = os.path.join(ROOT, "vulva", "build_dataset.py")
        ds = os.path.join(tmp, "ds")
        out = subprocess.run([PY, tool, ds, "--add", "train", "feb", tr, "--add", "test_newdays", "feb12", te],
                             capture_output=True, text=True, cwd=tmp)
        assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
        with open(os.path.join(ds, "labels.csv")) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5 and {r["split"] for r in rows} == {"train", "test_newdays"}
        assert all(os.path.exists(os.path.join(ds, r["ir_path"])) and os.path.exists(os.path.join(ds, r["mask_path"]))
                   and os.path.exists(os.path.join(ds, r["depth_path"])) for r in rows)
        assert {r["animal_id"] for r in rows} == {"b2_s00", "b2_s01", "b2_s02"}
        with open(os.path.join(ds, "tail_labels.csv")) as f:
            tails = list(csv.DictReader(f))
        assert len(tails) == 1 and os.path.exists(os.path.join(ds, tails[0]["tail_mask_path"]))
        with open(os.path.join(ds, "label_set.csv")) as f:
            assert len(list(csv.DictReader(f))) == 5
        assert os.path.exists(os.path.join(ds, "README.md")) and "no animal-day" in out.stdout
        rep = os.path.join(tmp, "seg_report.json")                # train split taken from a run's report
        with open(rep, "w") as f:
            json.dump({"labels": [tr]}, f)
        o = subprocess.run([PY, tool, os.path.join(tmp, "ds2"), "--train-from", rep, "--add", "test_newdays", "feb12", te],
                           capture_output=True, text=True, cwd=tmp)
        assert o.returncode == 0 and "train <- feb_train (3)" in o.stdout, o.stdout[-1500:]
        # leaks stop the build: the same frame in two splits, or an animal on a training day
        for extra in (["--add", "test_newsows", "dup", tr], ["--add", "test_newsows", "bad", bad]):
            o = subprocess.run([PY, tool, os.path.join(tmp, "ds_bad"), "--add", "train", "feb", tr] + extra,
                               capture_output=True, text=True, cwd=tmp)
            assert o.returncode != 0 and "Not built" in o.stderr, o.stdout[-1500:]
        assert not os.path.exists(os.path.join(tmp, "ds_bad", "labels.csv"))
        # train / eval by split; without --split they refuse
        lab = os.path.join(ds, "labels.csv")
        trainer = [PY, os.path.join(ROOT, "vulva", "train_seg.py"), "--labels", lab, "--folds", "0", "--epochs", "1",
                   "--no-pretrained", "--out-model", os.path.join(tmp, "m.pth"), "--out-dir", os.path.join(tmp, "seg")]
        o = subprocess.run(trainer, capture_output=True, text=True, cwd=tmp)
        assert o.returncode != 0 and "--split" in o.stderr
        o = subprocess.run(trainer + ["--split", "train"], capture_output=True, text=True, cwd=tmp)
        assert o.returncode == 0, o.stderr[-2000:]
        assert "3 frames from" in o.stdout, o.stdout[-800:]
        ev = [PY, os.path.join(ROOT, "vulva", "eval_seg.py"), "--model", os.path.join(tmp, "m.pth"), "--labels", lab,
              "--out-dir", os.path.join(tmp, "ev")]
        assert subprocess.run(ev, capture_output=True, text=True, cwd=tmp).returncode != 0
        o = subprocess.run(ev + ["--split", "test_newdays"], capture_output=True, text=True, cwd=tmp)
        assert o.returncode == 0, o.stderr[-2000:]
        assert "(split test_newdays)" in o.stdout and "2 labelled frames" in o.stdout, o.stdout[:600]
    finally:
        shutil.rmtree(tmp)


def test_inventory_raw_counts_passes_months_night_and_gaps():
    """inventory_raw.py: timestamp folders (also one level down) counted by month and day, with
    night passes, sow images per day and the days without any pass."""
    tmp = tempfile.mkdtemp()
    try:
        root = os.path.join(tmp, "Pictures_OAK")
        for name in ["20260130-08-00-00", "20260130-22-10-00", "20260201-09-00-00", "OAK_feb/20260204-03-00-00"]:
            d = os.path.join(root, name)
            os.makedirs(d)
            for pid in range(3):
                open(os.path.join(d, f"pig{pid}_ir_vis_{os.path.basename(name)}.jpg"), "wb").close()
            open(os.path.join(d, "pig0_depth_x.raw"), "wb").close()
        out_csv = os.path.join(tmp, "inv.csv")
        o = subprocess.run([PY, os.path.join(ROOT, "vulva", "inventory_raw.py"), root, "--out", out_csv],
                           capture_output=True, text=True, cwd=tmp)
        assert o.returncode == 0, o.stderr[-1500:]
        assert "4 robot passes on 3 days" in o.stdout, o.stdout
        assert "2026-01      1       2             1           6" in o.stdout, o.stdout
        assert "01-31" in o.stdout and "02-02 to 02-03" in o.stdout, o.stdout
        with open(out_csv) as f:
            rows = {r["date"]: r for r in csv.DictReader(f)}
        assert rows["20260204"]["night_passes"] == "1" and rows["20260130"]["sow_images"] == "6"
        assert rows["20260130"]["highest_stall"] == "2"
    finally:
        shutil.rmtree(tmp)


def test_check_view_learns_tails_from_held_out_candidates():
    """check_view.py fit: from a finder run's held-out predictions (jobs/fold*.npz) it learns to keep
    round, dome-shaped candidates (vulvas) and drop long, ridge-shaped ones lying on the tail
    class (tails), cross-validated by the same folds; eval runs finder + check on a labelled set."""
    from seg_model import VulvaUNet, save_checkpoint
    tmp = tempfile.mkdtemp()
    try:
        rng = np.random.default_rng(5)
        set_dir = os.path.join(tmp, "set")
        os.makedirs(os.path.join(set_dir, "masks"))
        os.makedirs(os.path.join(tmp, "run", "jobs"))
        yy, xx = np.mgrid[0:384, 0:320].astype(np.float64)
        fy, fx = np.mgrid[0:480, 0:640].astype(np.float64)
        rows, folds = [], {0: ([], []), 1: ([], [])}
        for i in range(32):
            pid, real = i % 4, i % 2 == 0
            ts = f"20260210-08-{i:02d}-00"
            folder = os.path.join(set_dir, ts)
            os.makedirs(folder)
            mx, my = 150 + rng.uniform(-15, 15), 200 + rng.uniform(-15, 15)        # model-space centre
            ax, ay = (14, 20) if real else (5, 42)
            blob = ((xx - mx) / ax) ** 2 + ((yy - my) / ay) ** 2 <= 1
            tail = (((xx - mx) / 6) ** 2 + ((yy - (my - 60 if real else my)) / 45) ** 2 <= 1) & ~(blob if real else 0)
            probs = np.stack([np.where(blob, 0.9, 0.02), np.where(tail, 0.8, 0.01)]).astype(np.float16)
            Fx, Fy = mx * 380 / 320 + 120, my * 450 / 384 + 30                     # full-frame centre
            px = 800 / 471.583
            if real:
                bump = 10 * np.exp(-(((fx - Fx) ** 2 + (fy - Fy) ** 2) * px ** 2) / (2 * 15.0 ** 2))
            else:
                bump = 9 * np.exp(-((fx - Fx) * px) ** 2 / (2 * 6.0 ** 2)) * (np.abs(fy - Fy) < 45)
            z = np.clip(800 - bump + rng.normal(0, 0.3, fx.shape), 0, 65535).astype(np.uint16)
            ir = np.full((480, 640), 110, np.uint8)
            full_blob = (((fx - Fx) / (ax * 380 / 320)) ** 2 + ((fy - Fy) / (ay * 450 / 384)) ** 2) <= 1
            ir[full_blob] = 40 if real else 95
            cv2.imwrite(os.path.join(folder, f"pig{pid}_ir_vis_{ts}.jpg"), ir)
            with open(os.path.join(folder, f"pig{pid}_depth_{ts}.raw"), "wb") as f:
                f.write(b"\0" * 8)
                f.write(z.tobytes())
            row = {"ir_path": f"{ts}/pig{pid}_ir_vis_{ts}.jpg", "mask_path": "", "label": "absent", "pig_id": pid,
                   "timestamp_folder": ts, "pig_timestamp": ts}
            if real:
                from seg_model import to_full_frame
                cv2.imwrite(os.path.join(set_dir, "masks", f"{i}.png"), to_full_frame(blob.astype(np.uint8)) * 255)
                row.update(mask_path=f"masks/{i}.png", label="present")
            rows.append(row)
            k = 0 if pid < 2 else 1
            folds[k][0].append(os.path.join(folder, f"pig{pid}_ir_vis_{ts}.jpg"))
            folds[k][1].append(probs)
        lab = os.path.join(set_dir, "labels.csv")
        with open(lab, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        for k, (keys, probs) in folds.items():
            np.savez(os.path.join(tmp, "run", "jobs", f"fold{k}.npz"), keys=np.array(keys), probs=np.stack(probs))
        tool = os.path.join(ROOT, "vulva", "check_view.py")
        chk = os.path.join(tmp, "check.json")
        o = subprocess.run([PY, tool, "fit", "--run", os.path.join(tmp, "run"), "--labels", lab, "--out", chk,
                            "--precision", "0.9"], capture_output=True, text=True, cwd=tmp)
        assert o.returncode == 0, o.stdout[-1500:] + o.stderr[-1500:]
        model = json.load(open(chk))
        cv = model["cv"]
        assert model["candidates"] == 32 and cv["real"] == 16 and cv["false"] == 16, cv
        assert cv["kept_real"] >= 14 and cv["kept_false"] <= 1, o.stdout
        ck = os.path.join(tmp, "f.pth")
        save_checkpoint(VulvaUNet(pretrained=False, n_out=3), ck, {"threshold": 0.5, "min_area": 25, "inputs": "ir",
                                                                    "classes": 3})
        o = subprocess.run([PY, tool, "eval", "--model", ck, "--check", chk, "--labels", lab,
                            "--out-dir", os.path.join(tmp, "ev")], capture_output=True, text=True, cwd=tmp)
        assert o.returncode == 0, o.stderr[-1500:]
        assert "with the check" in o.stdout and os.path.exists(os.path.join(tmp, "ev", "check_frames.csv"))
    finally:
        shutil.rmtree(tmp)

if __name__ == "__main__":
    fails = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{'All' if not fails else fails} {'checks passed' if not fails else 'failed'}")
    sys.exit(1 if fails else 0)
