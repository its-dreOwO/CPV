# Design Spec — Pivot to "How can vehicles avoid pedestrians and vehicles?"

> **Status:** approved design, pre-implementation. Supersedes the drone
> obstacle-avoidance design in `docs/training_pipeline.md`, which will be
> rewritten to match this spec during implementation.
> **Date:** 2026-06-22

---

## 1. Problem & system identity

The CPV301 project pivots from **drone obstacle avoidance** to a road-vehicle
research question:

> **"How can vehicles avoid pedestrians and vehicles?"**

The deliverable is a **forward-facing dashcam perception + risk-advisory
system** (ADAS Forward-Collision-Warning style). The vehicle does **not** act —
the system perceives obstacles, tracks them, and classifies each tracked object
as **SAFE / CAUTION / DANGER**. This keeps the project squarely a computer-vision
problem (perception + risk reasoning), honest about running on a single
monocular camera, and avoids claiming a control loop that cannot be validated
without a real vehicle.

### Viewpoint change vs the drone project

| Aspect | Drone (old) | Vehicle (new) |
|--------|-------------|---------------|
| Camera | Top-down / aerial | Forward-facing dashcam, monocular |
| "Avoid" output | `(yaw_delta, altitude_delta)` control | Per-object **risk label** (advisory) |
| Depth source | AirSim ground-truth depth | None at runtime (mono); KITTI GT for offline validation |
| Danger cue | Proximity to frame center | In ego-path **and** closing (bbox growth) |

---

## 2. Pipeline architecture

The three-stage chain keeps its **shape**; only the viewpoint and the third
stage's meaning change.

```
frame (dashcam, monocular np.ndarray)
  -> BaseDetector.detect(frame)           -> List[Detection]    (vehicle / person / two_wheeler)
  -> BaseTracker.update(detections)       -> List[Track]        (track_id, bbox, velocity, scale-growth)
  -> BaseRiskAssessor.assess(tracks, hw)  -> List[RiskedTrack]  (each tagged SAFE/CAUTION/DANGER)
```

### Reuse vs replace map

| Component | Fate | Notes |
|-----------|------|-------|
| `detection/detector.py` (`BaseDetector`, `Detection`) | **Keep as-is** | Contract unchanged |
| `detection/yolo_detector.py` (`YoloDetector`) | **Keep code, retrain weights** | Retrain on BDD100K |
| `tracking/` (`KalmanTracker`, `Track`) | **Keep as-is** | 2D image tracking is dataset-agnostic; already exposes bbox scale + velocity needed for the TTC proxy |
| `avoidance/planner.py` (`BaseAvoidancePlanner`) | **Replace** | → `src/risk/assessor.py` (`BaseRiskAssessor.assess(tracks, frame_shape) -> List[RiskedTrack]`) |
| `avoidance/geometric_planner.py` (`GeometricPlanner`) | **Replace** | → `src/risk/zone_assessor.py` (`RiskZoneAssessor`, Approach A) |
| `utils/visualizer.py` | **Adapt** | Draw ego-path trapezoid + color-coded risk boxes/labels |
| `main.py` | **Adapt** | Risk overlay instead of yaw/altitude printout |
| `prototype/web_app.py` (Streamlit) | **Adapt** | New copy, 3 classes, new demo overlay |
| `scripts/preprocess.py`, `validate_data.py`, `configs/*`, `modal_train.py` | **Adapt** | BDD100K + 3 classes + KITTI hold-out; keep 3-model lineup |

**Package rename:** `src/avoidance/` → `src/risk/` (the word "avoidance" implied
control; the stage is now risk assessment). All imports updated accordingly.

### New data contract

`RiskedTrack` extends a `Track` with:
- `risk: Literal["SAFE", "CAUTION", "DANGER"]`
- `in_path: bool` — object overlaps the ego-path region
- `ttc_proxy: float` — closing rate from bbox-area growth (unitless proxy, not seconds)

---

## 3. Datasets & class scheme

### Primary — BDD100K (detection)

- Train / val / test from the BDD100K detection split. BDD100K ships official
  train + val; we carve a held-out **test** split from train (stratified) or use
  val as test — decided during implementation. **Seed = 42** preserved everywhere.
- Native 10 classes collapsed to **3 obstacle classes**:

  | Coarse class | BDD100K source classes |
  |--------------|------------------------|
  | `vehicle` | car, truck, bus, train |
  | `person` | pedestrian, rider |
  | `two_wheeler` | bicycle, motorcycle |

  Dropped: `traffic light`, `traffic sign` (not collision obstacles; do not feed
  risk logic).

### Held-out — KITTI (cross-dataset)

KITTI is **not** folded into training. It is used two ways in R3:
1. **Zero-shot generalization** — BDD100K-trained models evaluated on KITTI →
   domain-gap finding (mAP drop).
2. **Risk-zone validation (Approach C)** — KITTI ground-truth 3D boxes / depth
   give true object distances, used to validate the heuristic risk labels.

### Storage

`data/raw/` and `data/processed/` remain gitignored. Same layout convention as
before: `data/processed/<dataset>/{train,val,test}/{images,labels}`.

---

## 4. Risk assessment (Approach A) + validation (Approach C)

A dashcam is monocular — no runtime depth. "Dangerous" decomposes into **WHERE**
(is the object in the path the car will drive through?) and **HOW CLOSE / closing**
(is it near and getting nearer?).

### Approach A — 2D image heuristics (the runtime engine)

`RiskZoneAssessor.assess(tracks, frame_shape)`:

1. **Ego-path region** — a trapezoid parameterized by frame size (narrow near the
   horizon line, wide at the bottom edge), approximating where the lane ahead
   projects onto the image. Config-driven; no per-frame compute.
2. **In-path test** — the object's ground-contact point (bbox bottom-center)
   inside the trapezoid → `in_path = True`.
3. **Closing proxy (TTC)** — from the Kalman tracker's bbox-area growth rate. A
   large and rapidly-growing box ⇒ closing fast.
4. **Risk label:**
   - `DANGER` = `in_path` ∧ (large ∨ fast-growing)
   - `CAUTION` = `in_path` ∧ slow/stable, **or** near the path edge
   - `SAFE` = off-path
   All thresholds live in the config; no magic numbers in code.
5. **Size gate on growth** — the growth proxy normalizes by box area
   (`scale_velocity / area`), so for a distant (tiny) box the signal is
   noise-dominated: a 1–2 px Kalman jitter reads as explosive growth and would
   false-trigger `DANGER`. A `min_danger_area_frac` floor requires the box to
   clear a minimum size before *growth alone* may escalate to `DANGER`; below it
   an in-path object caps at `CAUTION`. The `large` term is unaffected (a big box
   right ahead is `DANGER` regardless). This is the runtime correlate of the
   "no metric depth" honesty — distance is inferred from apparent size, and we
   refuse to trust the closing signal where apparent size is too small to be
   reliable.

No extra models, real-time, reuses tracker output directly.

### Approach C — validate A against KITTI ground truth (offline, R3)

`scripts/validate_risk.py` runs the assessor on KITTI images and compares its
labels to KITTI ground-truth distances, reporting agreement (e.g. "of objects
flagged DANGER, X% are within Y meters"). This converts the heuristic into a
**measured** finding. Offline only — zero runtime cost. Mirrors the role AirSim's
depth maps played in the drone design.

### Out of scope (stretch only)

A monocular depth model (Depth-Anything / MiDaS) in the core pipeline
("Approach B") is explicitly **not** in this design — a risk-advisory CV demo
does not need metric depth, and stacking an unvalidated depth estimate adds
machinery without commensurate value. May be revisited as an R4 stretch goal.

---

## 5. Model lineup (unchanged — R3 needs the comparison)

Three detectors along two controlled axes, retrained on BDD100K with an
**identical protocol** (same split, epochs, image size, augmentation, seed) so any
performance gap is attributable to the model, not the training loop.

| Model | Role | Axis |
|-------|------|------|
| **YOLOv8n** | Speed baseline / embedded floor (shared anchor) | capacity (n → m) |
| **YOLOv8m** | Primary demo model (ships in R4) | capacity / architecture pivot |
| **YOLOv10n** | NMS-free end-to-end / architecture contrast | version·paradigm (v8n → v10n) |

> **Amendment (2026-06-23):** RT-DETR-L was replaced by **YOLOv10n**. At
> operator-run time the compute budget ($28.57) could not fund a full RT-DETR
> run (~$0.47/epoch ≈ $23, over half the budget, on any GPU). YOLOv10n is an
> NMS-free, end-to-end CNN detector — it preserves RT-DETR's core contribution
> (a non-NMS detection paradigm) at ~$5.6/run, and its latency focus aligns with
> the FPS ≥ 30 rule. With v8n as the shared anchor the contrast becomes
> v8n → v10n (version/paradigm, matched nano scale) rather than CNN → transformer.

`configs/` keeps `yolov8n.yaml`, `yolov8m.yaml`, `yolov10n.yaml` (adapted to
BDD100K) plus a new dataset config (e.g. `bdd100k.yaml`, `nc=3`). Image size
640×640, seed 42, selection rule **highest mAP@0.5 subject to FPS ≥ 30**.
`modal_train.py` kept and adapted for the BDD100K 3-model runs.

---

## 6. R-round mapping (full rewrite)

| Round | Content |
|-------|---------|
| **R1** | New problem statement + this locked design + BDD100K data-validation report |
| **R2** | Preprocess + one model (YOLOv8n) trained on BDD100K + initial risk-overlay demo |
| **R3** | **3-model comparison** (v8n / v8m / v10n) on BDD100K test **+** KITTI cross-dataset generalization **+** risk-zone validation vs KITTI ground truth **+** day/night & weather robustness breakdown (from BDD100K per-image attributes) |
| **R4** | Best model wired into the live dashcam demo + end-to-end FPS + Streamlit showcase |

`docs/training_pipeline.md` and `CLAUDE.md` are rewritten to this spec during
implementation; old drone content is replaced (git history preserves it).

---

## 7. Cleanup completed (2026-06-22, pre-implementation)

Done on branch `pivot/vehicle-avoidance` before any pivot code:

- Removed obsolete VisDrone data (~4.1 GB), drone demo videos/outputs (~7.2 GB),
  `configs/visdrone5.yaml`, `lab_data/`, stale notes (`findings.md`,
  `progress.md`), `install_dependencies.sh`, old drone R1 PDFs, stray Kaggle
  metadata.
- `git gc --aggressive --prune=now` reclaimed ~3.3 GB of unreachable objects
  (no history rewrite; commit SHAs unchanged — safe for the public repo).
- Fixed `.gitignore` so the `.venv` symlink is no longer tracked.
- Result: repo ~11 GB → ~141 MB; `.git` 3.3 GB → 760 KB.

---

## 8. Open implementation details (resolved during planning, not blocking design)

- BDD100K test split: carve from train vs reuse val — decide in preprocess step.
- Exact ego-path trapezoid parameters and risk thresholds — tuned empirically,
  stored in config.
- BDD100K → YOLO label conversion (BDD100K ships JSON; needs a converter in
  `scripts/preprocess.py`).
- KITTI label parsing for the risk-validation script.
