# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This is a **CPV301 (Computer Vision) course project** at FPT University. As of **2026-06-22 the project pivoted** away from drone obstacle avoidance to a road-vehicle research question:

> **"How can vehicles avoid pedestrians and vehicles?"**

The deliverable is a **forward-facing dashcam perception + risk-advisory system** (ADAS Forward-Collision-Warning style): it detects and tracks road obstacles and tags each tracked object **SAFE / CAUTION / DANGER**. It is perception + risk reasoning, not a vehicle-control loop (no steering/braking output) — honest about running on a single monocular camera.

Submissions are organized into four rounds (`reports/R1`-`R4`), each requiring slides + report (PDF, English) plus a public GitHub link. R2 onward must include source code; R4 is the final demo. The full rubric is committed at `docs/SU26_AI2013_CPV301.xlsx`.

**Source of truth for the pivot:**
- Design spec: `docs/superpowers/specs/2026-06-22-vehicle-avoidance-pivot-design.md`
- Implementation plans: `docs/superpowers/plans/` (Plan 1 = risk-assessor code pivot, Plan 2 = BDD100K data pipeline, Plan 3 = training + eval + docs rewrite)

> **Note:** the old drone `docs/training_pipeline.md` was removed; the design spec under `docs/superpowers/specs/` is the source of truth. A vehicle training pipeline doc is authored in Plan 3 when training lands. Active work is on branch `pivot/vehicle-avoidance`.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate   # Fish: source .venv/bin/activate.fish
pip install -r requirements.txt

# Run demo pipeline (webcam or dashcam video) — shows ego-path + risk overlay
python main.py --source 0
python main.py --source path/to/dashcam.mp4
python main.py --source path/to/dashcam.mp4 --weights models/best.pt --device cuda

# Streamlit showcase prototype (run from repo root so `src.*` imports resolve)
streamlit run prototype/web_app.py

# Data pipeline (BDD100K) — scripts are being adapted from VisDrone in Plan 2
python scripts/validate_data.py --images data/raw/bdd100k/images --labels data/raw/bdd100k/labels --num-classes 3
python scripts/preprocess.py     # BDD100K JSON -> YOLO, 10 -> 3 class remap, stratified split -> data/processed/

# Train / evaluate (sanity: --epochs 5; full: --epochs 50)
python scripts/train.py --config configs/yolov8m.yaml --epochs 50 --device cuda
python scripts/evaluate.py --weights models/best.pt --data data/processed/test

# Tests (pytest with coverage on src/ is configured in setup.cfg)
pytest                                    # all tests
pytest tests/test_risk.py                 # single file
pytest tests/test_risk.py::test_off_path_object_is_safe  # single test
pytest -k "tracking"                      # by keyword

# Lint / format (matches CI exactly)
black --check .
flake8 .
black .                                   # apply formatting
```

CI (`.github/workflows/ci.yml`) runs `black --check .` then `flake8 .` on every push and PR. Both must pass.

## Architecture

The pipeline is a three-stage CV chain: **detection -> tracking -> risk assessment**, with each stage defined as a base class that concrete implementations subclass. This separation lets the team swap models (YOLO variants, different trackers) without changing downstream code — important because R3 compares >=3 different detectors.

**Data flow:**
```
frame (dashcam, monocular np.ndarray)
  -> BaseDetector.detect()        -> List[Detection]   (bbox, confidence, class_id: vehicle/person/two_wheeler)
  -> BaseTracker.update()         -> List[Track]       (track_id, bbox, velocity, scale_velocity, age)
  -> BaseRiskAssessor.assess()    -> List[RiskedTrack] (each tagged SAFE/CAUTION/DANGER)
```

**Key contracts** (in `src/`):
- `detection/detector.py` — `Detection` dataclass + `BaseDetector.detect(frame) -> List[Detection]`
- `tracking/tracker.py` — `Track` dataclass (carries `velocity` and `scale_velocity` — bbox-area growth rate — for the closing/TTC proxy) + `BaseTracker.update(detections) -> List[Track]`
- `risk/assessor.py` — `RiskedTrack` dataclass + `RiskLevel` constants + `BaseRiskAssessor.assess(tracks, frame_shape) -> List[RiskedTrack]`
- `utils/visualizer.py` — rendering helpers (decoupled from pipeline logic)

**Concrete implementations:**
- `detection/yolo_detector.py` — `YoloDetector(model_path)` wraps Ultralytics YOLO; returns `List[Detection]` in original-frame pixel coordinates
- `tracking/kalman_tracker.py` — `KalmanTracker` uses per-object Kalman filters (7-state) with Hungarian/IoU data association (SORT-style); exposes area velocity as `Track.scale_velocity`
- `risk/zone_assessor.py` — `RiskZoneAssessor` (Approach A): projects an **ego-path trapezoid** (narrow at the horizon, wide at the frame bottom); an object is **in-path** if its bbox bottom-center falls inside it, and **closing** if its bbox area is large or growing (`scale_velocity`). Tags SAFE (off-path) / CAUTION (in-path, stable) / DANGER (in-path, large or fast-growing).

`main.py` wires these together for the live demo and overlays the ego-path region + color-coded risk boxes. `prototype/web_app.py` is a Streamlit showcase app reusing the same `src.*` pipeline classes; it inserts the repo root onto `sys.path` so `from src.*` resolves and reads its theme from the repo-root `.streamlit/config.toml` — always launch it from the repo root. `scripts/train.py` loads a model config YAML and calls `ultralytics.YOLO.train()`. `scripts/preprocess.py` does the class remap + stratified split; `scripts/validate_data.py` runs pre-training integrity checks (helpers in `src/utils/data_validation.py`).

> The `src/risk/` package replaces the former `src/avoidance/` (the word "avoidance" implied a control loop). `scripts/preprocess.py`, `validate_data.py`, `train.py`, `evaluate.py`, and `modal_train.py` are still VisDrone/drone-shaped and are adapted for BDD100K in Plans 2–3.

**Import style:** modules import from `src.*` (absolute), so always run pytest/scripts from the repo root.

## Datasets & training plan

Only the **detector** is trained. Tracker = Kalman + Hungarian (rule-based). Risk assessor = geometric ego-path heuristic (rule-based).

**Datasets:**
- **Primary — BDD100K** (dashcam detection). Native 10 classes collapsed to **3 coarse classes**: `vehicle` (car/truck/bus/train), `person` (pedestrian/rider), `two_wheeler` (bicycle/motorcycle). Traffic light/sign dropped.
- **Held-out — KITTI** (cross-dataset). Used in R3 for (a) zero-shot generalization (domain gap) and (b) **validating the risk-zone heuristic against ground-truth 3D/depth** distances.
- Both gitignored under `data/raw/` and `data/processed/`.

**Model lineup for the R3 >=3-model comparison (kept):**
- **YOLOv8n** — speed baseline / embedded floor
- **YOLOv8m** — primary demo model (ships in R4)
- **RT-DETR-L** — accuracy ceiling / architecture contrast (transformer vs CNN)

Two controlled axes — capacity (n -> m) and architecture (m -> RT-DETR-L) — so any gap is attributable to the model, not the training loop. All use the same Ultralytics API.

**Locked decisions:** image size 640x640, epochs 50 (full) / 5 (sanity), seed = 42 everywhere, 70/15/15 stratified split, selection rule **highest mAP@0.5 subject to FPS >= 30**. Compute target: Nvidia L4 (24 GB); RT-DETR-L needs L40S/A100 for batch=16.

## Pipeline status (as of 2026-06-22)

The pivot is mid-implementation. Old drone training results (VisDrone) are **superseded** and the VisDrone data/weights were removed in the cleanup.

| Stage | Status |
|-------|--------|
| Cleanup (drone artifacts, git gc) | ✅ done |
| Design spec | ✅ approved + committed |
| Plan 1 — risk-assessor code pivot (`src/risk/`) | 🔄 executing (TDD, no dataset/GPU needed) |
| Plan 2 — BDD100K data pipeline (download, JSON→YOLO, 3-class remap, KITTI prep) | ⬜ next |
| Plan 3 — training (3 models on BDD100K) + KITTI eval + docs/Streamlit rewrite | ⬜ |

## R-round mapping

| Round | What ships |
|-------|-----------|
| **R1** | New problem statement + design + BDD100K data-validation report |
| **R2** | Preprocess + YOLOv8n trained on BDD100K + initial risk-overlay demo |
| **R3** | 3-model comparison + KITTI cross-dataset generalization + risk-zone validation + day/night & weather robustness |
| **R4** | Best model wired into live dashcam demo + end-to-end FPS + Streamlit showcase |

## Conventions

- **Line length: 88** (Black default, matched in `setup.cfg` flake8 config with `E203,W503` ignored for Black compatibility).
- **Data and weights are gitignored** — `data/raw/`, `data/processed/`, and `models/*.pt|*.pth|*.onnx` never get committed. Only `data/samples/` is intended for tracked tiny test fixtures.
- **Course materials are gitignored** — `CPV_notes/`, `*.pptx`, `*.xlsx` stay local only, **except** the project rubric at `docs/SU26_AI2013_CPV301.xlsx` (whitelisted via `!docs/*.xlsx`).
- **Reports directory** (`reports/R1`-`R4`) is for final PDFs only; drafts and working files belong elsewhere.
- **Design docs and plans** live in `docs/` (`docs/superpowers/specs/`, `docs/superpowers/plans/`) — the source of truth that R-round reports reference. Update them when the design changes.
- **`prototype/` holds the Streamlit showcase app** and is excluded from `flake8` in `setup.cfg` (the `sys.path` shim trips E402), so it is not lint-gated by CI — keep it formatted with `black` manually.
