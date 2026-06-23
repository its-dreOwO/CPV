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

> **Note:** the design spec under `docs/superpowers/specs/` is the source of truth. The vehicle training pipeline doc was authored in Plan 3 — see `docs/training_pipeline.md` (Modal training runbook, evaluation, KITTI zero-shot, risk validation). Active work is on branch `pivot/vehicle-avoidance`.

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

# Data pipeline (BDD100K) — JSON -> YOLO, 10 -> 3 class remap, stratified split
python scripts/preprocess.py
python scripts/validate_data.py --images data/processed/bdd100k/train/images --labels data/processed/bdd100k/train/labels --num-classes 3
python scripts/validate_data.py --images data/processed/bdd100k/val/images --labels data/processed/bdd100k/val/labels --num-classes 3
python scripts/validate_data.py --images data/processed/bdd100k/test/images --labels data/processed/bdd100k/test/labels --num-classes 3

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

> The `src/risk/` package replaces the former `src/avoidance/` (the word "avoidance" implied a control loop). Plan 3 cleaned up the vehicle pipeline: `evaluate.py` (per-class mAP, drop visdrone refs) and `modal_train.py` (correct `--data-root`, vehicle naming) are done; `scripts/evaluate_robustness.py`, `scripts/preprocess_kitti.py`, `scripts/validate_risk.py`, and `src/utils/kitti.py` were added.

**Import style:** modules import from `src.*` (absolute), so always run pytest/scripts from the repo root.

## Datasets & training plan

Only the **detector** is trained. Tracker = Kalman + Hungarian (rule-based). Risk assessor = geometric ego-path heuristic (rule-based).

**Datasets:**
- **Primary — BDD100K** (dashcam detection). Native 10 classes collapsed to **3 coarse classes**: `vehicle` (car/truck/bus/train), `person` (pedestrian/rider), `two_wheeler` (bicycle/motorcycle). Traffic light/sign dropped.
- **Held-out — KITTI** (cross-dataset). Used in R3 for (a) zero-shot generalization (domain gap) and (b) **validating the risk-zone heuristic against ground-truth 3D/depth** distances.
- Both gitignored under `data/raw/` and `data/processed/`.

**Model lineup for the R3 >=3-model comparison:**
- **YOLOv8n** — speed baseline / embedded floor (shared anchor)
- **YOLOv8m** — primary demo model (ships in R4)
- **YOLOv10n** — NMS-free, end-to-end CNN; architecture/paradigm contrast at matched (nano) capacity

With YOLOv8n as the shared anchor, two clean axes isolate the cause of any gap: capacity (v8n -> v8m, same architecture) and version/paradigm (v8n -> v10n, same nano scale). All use the same Ultralytics API.

> **Lineup change (2026-06-23, budget-driven):** RT-DETR-L was dropped in favor of **YOLOv10n**. RT-DETR cost ~$0.47/epoch (≈$23 for a full run) regardless of GPU — over half the available compute budget — while YOLOv10n trains for ~$5.6. YOLOv10n preserves RT-DETR's key contribution (a *non-NMS, end-to-end* detection paradigm) at a fraction of the cost, and its latency focus fits the FPS>=30 selection rule. `configs/rtdetr.yaml` was removed.

**Locked decisions:** image size 640x640, epochs 50 (full) / 5 (sanity), seed = 42 everywhere, 70/15/15 stratified split, selection rule **highest mAP@0.5 subject to FPS >= 30**. Compute target: Nvidia L4 (24 GB) — all three models fit on L4 (YOLOv10n at batch 64 like the other nano).

## Pipeline status (as of 2026-06-23)

The pivot is mid-implementation. Old drone training results (VisDrone) are **superseded** and the VisDrone data/weights were removed in the cleanup.

| Stage | Status |
|-------|--------|
| Cleanup (drone artifacts, git gc) | ✅ done |
| Design spec | ✅ approved + committed |
| Plan 1 — risk-assessor code pivot (`src/risk/`) | ✅ done |
| Plan 2 — BDD100K data pipeline (download, JSON→YOLO, 3-class remap) | ✅ done |
| Plan 3 — training/eval/risk-validation scripts + docs/Streamlit rewrite | ✅ code done |
| Plan 3 — GPU training runs (3 models) | 🔄 in progress — yolov8n + yolov10n on Modal (detached), yolov8m on a free GCP T4 VM; see `docs/superpowers/plans/2026-06-23-HANDOFF.md` |
| Plan 3 — fetch/verify weights + BDD100K/KITTI eval + risk validation + final review | ⬜ operator-pending (handoff has exact commands) |

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
