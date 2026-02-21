# PAAL Project — Progress Report
**Student:** Akbar | **Date:** 19 Feb 2026 | **Task A:** Sow Posture Classifier

---

## What I did this week

**Data collection & labelling**
- Collected ~40 timestamp folders across 5 days at different times of day
- Manually labelled 621 frames: 306 standing / 315 not-standing (binary: 1=standing, 0=not-standing)
- Split by pig ID — training on pig 0–14, validation on 15–16, test on 17–19 (held-out, unseen pigs)

**Camera alignment**
- Found that RGB (1280×800, fx=798) and ToF sensors (640×480, fx=472) have different resolutions, focal lengths, and a ~1.7 cm physical offset between them (from calibration.json translation vector)
- Wrote an alignment script using OpenCV remap to warp RGB into the ToF frame so pixels correspond spatially across modalities
- *(Fengkai — happy to have you review the alignment code in case I've misread the calibration format)*

**Model training**
- Trained 6 variants of a MobileNetV2 classifier using different input modalities: RGB, IR, Depth, RGB+Depth, RGB+IR, RGB+IR+Depth
- Test accuracy on held-out pigs:

| Modality | Test Acc |
|---|---|
| RGB only | **98.0%** |
| IR / Depth / Fusion variants | 96.1% |

---

## Next steps
- Extend to 4-class posture classification (standing, sitting, lateral lying, sternal lying) — pending discussion with Fengkai
- Start Task B: vulva size estimation from depth / point cloud data (`pig18_pointcloud.ply`)

Happy to share code, alignment overlays, or training curves on request.
