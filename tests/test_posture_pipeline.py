"""Fast checks for the posture pipeline (no lab data needed).

Run:  python tests/test_posture_pipeline.py      (or: pytest tests/)

Builds a tiny synthetic OAK-D folder tree with the camera quirks we have seen
in Fed_Pig (overflow IDs, double captures, empty stalls, corrupt jpgs) and
checks the ID policy, smoothing, transition counting and the end-to-end CLI.
"""

import csv
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "posture"))

from id_corrections import apply_corrections, assign_ids, folder_quality  # noqa: E402
from temporal import smooth_sequence  # noqa: E402


def _frame(folder, pid, ts=None):
    return {"timestamp_folder": folder, "pig_id": pid, "pig_timestamp": ts or folder}


def test_known_rule_renames_and_drops():
    folder = "20260213-14-00-00"
    assert apply_corrections(folder, 19)[0] is None          # dup of 18
    assert apply_corrections(folder, 20)[0] == 19            # real pig 19
    assert apply_corrections(folder, 23)[0] is None          # >= 21
    assert apply_corrections("20260101-00-00-00", 20)[0] == 20  # outside every rule


def test_strict_policy_never_merges_overflow_into_low_ids():
    folder = "20260301-01-00-00"  # no correction rule covers this folder
    frames = [_frame(folder, i) for i in range(20)] + [_frame(folder, 20), _frame(folder, 21)]
    assign_ids(frames, policy="strict")
    kept = [f for f in frames if f["include"]]
    assert len(kept) == 20
    assert {f["pig_id"] for f in kept} == set(range(20))
    assert [f["id_status"] for f in frames[-2:]] == ["overflow", "overflow"]

    legacy = [_frame(folder, i) for i in range(20)] + [_frame(folder, 20)]
    assign_ids(legacy, policy="modulo")
    # v1.0 behaviour: pig20 became a second pig0 capture -> now caught as duplicate
    assert sum(1 for f in legacy if f["pig_id"] == 0 and f["include"]) == 1


def test_double_capture_is_deduplicated_keeping_earliest():
    folder = "20260301-02-00-00"
    a = _frame(folder, 5, "20260301-02-00-10")
    b = _frame(folder, 5, "20260301-02-00-55")
    assign_ids([b, a])
    assert a["include"] and not b["include"] and b["id_status"] == "duplicate"


def test_quality_report_flags_short_folders():
    frames = [_frame("20260301-03-00-00", i) for i in range(19)]
    assign_ids(frames)
    q = folder_quality(frames)[0]
    assert q["needs_review"] == 1 and q["missing_stalls"] == "19"


def test_smoothing_removes_single_frame_flips_only():
    t0 = datetime(2026, 3, 1, 22, 0)
    times = [t0 + timedelta(minutes=10 * i) for i in range(7)]
    labels = ["lying", "lying", "standing", "lying", "lying", "standing", "standing"]
    out = smooth_sequence(labels, times, window=3)
    assert out == ["lying", "lying", "lying", "lying", "lying", "standing", "standing"]
    # a real 2-frame bout survives
    labels2 = ["lying", "standing", "standing", "lying", "lying"]
    assert smooth_sequence(labels2, times[:5], window=3) == labels2
    # frames separated by a big gap are never voted together
    gap_times = [t0, t0 + timedelta(minutes=10), t0 + timedelta(hours=5)]
    assert smooth_sequence(["lying", "lying", "standing"], gap_times, window=3)[2] == "standing"


# ── End-to-end on synthetic data ─────────────────────────────────────────

def make_synthetic_dataset(root, n_folders=12, seed=0):
    """OAK-style folders: 20 stalls, plus one overflow, one double capture,
    one empty stall (far depth) and one corrupt jpg."""
    rng = np.random.default_rng(seed)
    t0 = datetime(2026, 3, 1, 20, 0, 0)
    for k in range(n_folders):
        t = t0 + timedelta(minutes=10 * k)
        folder = t.strftime("%Y%m%d-%H-%M-%S")
        fdir = os.path.join(root, folder)
        os.makedirs(fdir)
        pids = list(range(20)) + ([20] if k == 3 else []) + ([7] if k == 5 else [])
        for j, pid in enumerate(pids):
            ts = (t + timedelta(seconds=2 * j)).strftime("%Y%m%d-%H-%M-%S")
            img = (rng.random((480, 640)) * 60 + 40 + pid * 5).astype(np.uint8)
            cv2.rectangle(img, (200, 150 + pid * 5), (420, 400), int(120 + 6 * (pid % 3)), -1)
            img = cv2.merge([img, img, img])
            path = os.path.join(fdir, f"pig{pid}_ir_vis_{ts}.jpg")
            if k == 2 and pid == 3:
                open(path, "wb").write(b"not a jpg")
            else:
                cv2.imwrite(path, img)
            depth = np.full((480, 640), 2200 if pid == 18 else 1100, dtype=np.uint16)
            depth.tofile(os.path.join(fdir, f"pig{pid}_depth_{ts}.raw"))
    return root


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_scan_folder_date_range():
    from run_inference import scan_folder
    tmp = tempfile.mkdtemp(prefix="paal_test_")
    try:
        data = make_synthetic_dataset(os.path.join(tmp, "data"))   # 20:00 .. 21:50 on Mar 1
        folders = lambda **kw: sorted({f["timestamp_folder"] for f in scan_folder(data, **kw)})
        assert len(folders()) == 12
        assert folders(start="20260301-20-30-00", end="20260301-21-00-00") == [
            "20260301-20-30-00", "20260301-20-40-00", "20260301-20-50-00", "20260301-21-00-00"]
        assert len(folders(end="20260301")) == 12, "a date-only end includes that whole day"
        assert folders(start="20260302") == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_end_to_end_cli(tmp=None):
    model = os.path.join(ROOT, "models", "posture3_ir_best.pth")
    if not os.path.exists(model):
        print("  (skipped: models/posture3_ir_best.pth not present)")
        return
    tmp = tmp or tempfile.mkdtemp(prefix="paal_test_")
    try:
        data = make_synthetic_dataset(os.path.join(tmp, "data"))
        out = os.path.join(tmp, "out")
        cmd = [sys.executable, os.path.join(ROOT, "posture", "run_inference.py"), data,
               "--output-dir", out, "--workers", "0", "--smooth", "3", "--no-detector"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
        rows = _read_csv(os.path.join(out, "predictions.csv"))
        keys = [(r["timestamp_folder"], r["pig_id"]) for r in rows]
        assert len(keys) == len(set(keys)), "one row per stall per folder"
        assert all(0 <= int(r["pig_id"]) < 20 for r in rows)
        assert not any(r["pig_id"] == "18" for r in rows), "empty stall filtered by depth"
        excluded = _read_csv(os.path.join(out, "predictions_excluded.csv"))
        assert {r["id_status"] for r in excluded} == {"overflow", "duplicate"}
        quality = _read_csv(os.path.join(out, "data_quality.csv"))
        assert sum(int(q["needs_review"]) for q in quality) == 2
        assert all(r["smoothed_prediction_name"] for r in rows)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Colab notebook stays in sync with the repo ───────────────────────────

def _notebook_pipeline_namespace():
    import json
    nb = json.load(open(os.path.join(ROOT, "PAAL_Posture_Classification.ipynb")))
    src = "".join(nb["cells"][8]["source"])
    src = src.replace('OUTPUT_DIR = "/content/paal_outputs"', 'OUTPUT_DIR = tempfile.mkdtemp(prefix="paal_nb_")')
    src = src.replace("from tqdm.notebook import tqdm", "from tqdm import tqdm")  # no Jupyter widgets here
    ns = {"os": os, "tempfile": tempfile}
    exec(compile(src, "notebook_cell_8", "exec"), ns)
    return ns


def test_notebook_has_same_corrections_and_no_modulo():
    import inspect
    import json
    import id_corrections
    ns = _notebook_pipeline_namespace()
    assert ns["CORRECTIONS"] == id_corrections.CORRECTIONS, "notebook CORRECTIONS out of sync"
    for fn in ("apply_corrections", "active_rules", "dedupe", "assign_ids"):
        assert inspect.getsource(getattr(id_corrections, fn)).strip() in "".join(
            json.load(open(os.path.join(ROOT, "PAAL_Posture_Classification.ipynb")))["cells"][8]["source"]), \
            f"notebook copy of {fn} out of sync with posture/id_corrections.py"
    nb = json.load(open(os.path.join(ROOT, "PAAL_Posture_Classification.ipynb")))
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "% 20" not in code, "notebook still maps pig IDs with % 20"


def test_notebook_pipeline_matches_cli_ids():
    import torch
    model_path = os.path.join(ROOT, "models", "posture3_ir_best.pth")
    if not os.path.exists(model_path):
        print("  (skipped: model not present)")
        return
    ns = _notebook_pipeline_namespace()
    tmp = tempfile.mkdtemp(prefix="paal_nbdata_")
    try:
        data = make_synthetic_dataset(os.path.join(tmp, "data"), n_folders=6)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model = ns["SingleModalModel"](in_channels=3, num_classes=3)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        results = ns["run_predictions"](data, torch.device("cpu"), model, detector=None)
        ns["assign_ids"](results)
        kept = [r for r in results if r["include"]]
        keys = [(r["timestamp_folder"], r["pig_id"]) for r in kept]
        assert len(keys) == len(set(keys))
        assert all(0 <= r["pig_id"] < 20 for r in kept)
        assert not any(r["pig_id"] == 18 for r in kept)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_transition_counting_on_legacy_csv_drops_overflow():
    from count_posture_changes import count_transitions, load_predictions
    tmp = tempfile.mkdtemp(prefix="paal_cpc_")
    try:
        path = os.path.join(tmp, "legacy.csv")
        t0 = datetime(2026, 3, 1, 22, 0)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_folder", "pig_id", "pig_timestamp", "prediction", "prediction_name",
                        "confidence", "median_depth", "image_path"])
            for k in range(6):
                folder = (t0 + timedelta(minutes=10 * k)).strftime("%Y%m%d-%H-%M-%S")
                w.writerow([folder, 0, folder, 2, "lying", 0.9, 1100, ""])
                # camera overflow: pig20 "standing" in the same folder (v1.0 merged it into pig 0)
                w.writerow([folder, 20, folder[:-2] + "30", 0, "standing", 0.9, 1100, ""])
        rows = load_predictions(path)
        assert {r["pig_id"] for r in rows} == {0}
        res = count_transitions(rows, night_only=True)
        assert sum(d["total_transitions"] for d in res[0].values()) == 0
        legacy = load_predictions(path, id_policy="none")
        legacy = [dict(r, pig_id=r["pig_id"] % 20) for r in legacy]
        res_old = count_transitions(legacy, night_only=True)
        assert sum(d["total_transitions"] for d in res_old[0].values()) > 0  # the v1.0 artefact
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_ground_truth_sheet_parsing():
    import openpyxl
    from eval_ground_truth import read_ground_truth, sheet_date
    assert str(sheet_date("26", 2026)) == "2026-02-06"
    assert str(sheet_date("210", 2026)) == "2026-02-10"
    tmp = tempfile.mkdtemp(prefix="paal_gt_")
    try:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "26"
        ws.append(["Sow position Feb 6"])
        ws.append(["Time"] + list(range(20)))
        ws.append(["08:00"] + [0] * 19 + [2])
        ws.append(["08:10"] + [1] * 20)
        path = os.path.join(tmp, "gt.xlsx")
        wb.save(path)
        gt = read_ground_truth(path, {"0": "lying", "1": "sitting", "2": "standing"}, 2026)
        assert len(gt) == 40
        first = [g for g in gt if g["time"].hour == 8 and g["time"].minute == 0]
        assert {g["label"] for g in first if g["pig_id"] == 19} == {"standing"}
        assert {g["label"] for g in first if g["pig_id"] == 0} == {"lying"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_rules_not_applied_twice_to_corrected_data():
    # v1.0 fix scripts already turned pig20 into pig19 here: re-applying the 02/13
    # rule would delete the real pig 19 (delete_pids [19]).
    folder = "20260213-14-00-00"
    frames = [_frame(folder, i) for i in range(20)]
    assign_ids(frames)
    assert all(f["include"] for f in frames)
    raw = [_frame(folder, i) for i in range(22)]          # uncorrected: 19 dup, 20 real, 21 dup
    assign_ids(raw)
    assert {f["pig_id"] for f in raw if f["include"]} == set(range(20))


def test_duplicates_resolved_among_usable_captures():
    from id_corrections import dedupe
    folder = "20260301-04-00-00"
    first = _frame(folder, 5, "20260301-04-00-10")
    second = _frame(folder, 5, "20260301-04-00-40")
    assign_ids([first, second], dedupe_captures=False)
    usable = [second]                                     # first capture was corrupt
    dedupe(usable)
    assert second["include"] and second["id_status"] == "ok"


def test_multimodal_augmentation_keeps_blocks_aligned():
    import torch
    sys.path.insert(0, os.path.join(ROOT, "posture"))
    from train import ArrayDataset
    img = (np.random.default_rng(0).random((1, 64, 64, 3)) * 255).astype(np.uint8)
    six = np.concatenate([img, img], axis=3)               # identical RGB and IR blocks
    ds = ArrayDataset(six, np.array([0]), augment=True)
    ds.color = lambda t: t                                 # isolate the geometric part
    for _ in range(10):
        x, _ = ds[0]
        assert torch.equal(x[:3], x[3:]), "flip/affine must be shared across modality blocks"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")
