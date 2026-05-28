# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This is a **CPV301 (Computer Vision) course project** at FPT University implementing **RT-DR-003: Obstacle Avoidance for Drones**. The research question is *"How can drones avoid dynamic obstacles during flight?"* and the expected output is a working prototype.

Submissions are organized into four rounds (`reports/R1`-`R4`), each requiring slides + report (PDF, English) plus a public GitHub link. R2 onward must include source code; R4 is the final demo. The full rubric is committed at `docs/SU26_AI2013_CPV301.xlsx`. The locked training pipeline design (model lineup, decisions, R-round mapping) is in `docs/training_pipeline.md` - read it before changing anything related to training.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run demo pipeline (webcam or video file)
python main.py --source 0
python main.py --source path/to/video.mp4
python main.py --source path/to/video.mp4 --weights models/best.pt --device cuda

# Phase 2 — validate raw data (run from repo root; labels are already YOLO-format)
python scripts/validate_data.py \
    --images data/raw/VisDrone_Dataset/VisDrone2019-DET-train/images \
    --labels data/raw/VisDrone_Dataset/VisDrone2019-DET-train/labels \
    --num-classes 10
python scripts/validate_data.py --videos data/raw/airsim/sequences

# Phase 3 — remap to 5 coarse classes + 70/15/15 stratified split -> data/processed/
python scripts/preprocess.py
python scripts/preprocess.py --raw-root data/raw/VisDrone_Dataset --seed 42

# Train / evaluate (Phase 5 sanity: --epochs 5; Phase 6 full: --epochs 50)
python scripts/train.py --config configs/yolov8m.yaml --epochs 50 --device cuda
# On Kaggle, override the dataset path:
python scripts/train.py --config configs/yolov8m.yaml --epochs 50 --device cuda \
    --data-root /kaggle/input/<dataset-slug>
python scripts/evaluate.py --weights models/best.pt --data data/processed/test

# Tests (pytest with coverage on src/ is configured in setup.cfg)
pytest                                    # all tests
pytest tests/test_detection.py            # single file
pytest tests/test_detection.py::test_detection_fields  # single test
pytest -k "tracking"                      # by keyword

# Lint / format (matches CI exactly)
black --check .
flake8 .
black .                                   # apply formatting
```

CI (`.github/workflows/lint.yml`) runs `black --check .` then `flake8 .` on every push and PR. Both must pass.

## Architecture

The pipeline is a three-stage CV chain: **detection -> tracking -> avoidance planning**, with each stage defined as a base class that concrete implementations subclass. This separation lets the team swap models (YOLO variants, different trackers, planners) without changing downstream code - important because R3 requires comparing >=3 different models.

**Data flow:**
```
frame (np.ndarray)
  -> BaseDetector.detect()    -> List[Detection]   (bbox, confidence, class_id)
  -> BaseTracker.update()     -> List[Track]       (track_id, bbox, velocity, age)
  -> BaseAvoidancePlanner.plan() -> (yaw_delta, altitude_delta)
```

**Key contracts** (in `src/`):
- `detection/detector.py` - `Detection` dataclass + `BaseDetector.detect(frame) -> List[Detection]`
- `tracking/tracker.py` - `Track` dataclass (carries velocity for dynamic obstacle prediction) + `BaseTracker.update(detections) -> List[Track]`
- `avoidance/planner.py` - `BaseAvoidancePlanner.plan(tracks) -> (yaw_delta, altitude_delta)`
- `utils/visualizer.py` - rendering helpers (decoupled from pipeline logic)

**Concrete implementations** (all three pipeline stages are now implemented):
- `detection/yolo_detector.py` — `YoloDetector(model_path)` wraps Ultralytics YOLO; returns `List[Detection]` in original-frame pixel coordinates
- `tracking/kalman_tracker.py` — `KalmanTracker` uses per-object Kalman filters (7-state: cx, cy, area, aspect ratio + velocities) with Hungarian/IoU data association (SORT-style)
- `avoidance/geometric_planner.py` — `GeometricPlanner` projects each track's future position by `time_horizon` seconds using its velocity, applies a repulsive force if within `safe_distance` pixels of frame center, outputs clamped `(yaw_delta, altitude_delta)`

`main.py` wires these together for the live demo; it reads actual frame dimensions at startup and passes them to the planner. `scripts/train.py` loads a model config YAML and calls `ultralytics.YOLO.train()` — supports `--data-root` to override the dataset path on Kaggle. `scripts/preprocess.py` does the Phase 3 class remap (10 → 5 classes) and 70/15/15 stratified split, writing to `data/processed/`. `scripts/validate_data.py` runs pre-training integrity checks (image readability, YOLO label format, image/label pairing, video integrity); reusable helpers live in `src/utils/data_validation.py`.

**Import style:** modules import from `src.*` (absolute), so always run pytest/scripts from the repo root.

## Training Plan (locked - see `docs/training_pipeline.md`)

Only the **detector** is trained. Tracker = Kalman + Hungarian (rule-based). Planner = geometric (rule-based).

**Model lineup for the R3 >=3-model comparison:**
- **YOLOv8n** - speed baseline / embedded floor
- **YOLOv8m** - primary demo model (the one that ships in R4)
- **RT-DETR-L** - accuracy ceiling / architecture contrast (transformer vs CNN)

The lineup is along two controlled axes - capacity (n -> m) and architecture (m -> RT-DETR-L) - so any performance gap is attributable to the model, not the training loop. All three use the same Ultralytics API.

**Locked decisions** (full table in `docs/training_pipeline.md`):
- Primary dataset: **VisDrone-DET** — raw data lives at `data/raw/VisDrone_Dataset/` with YOLO-format labels (10 original classes); processed 5-class split at `data/processed/` (both gitignored)
- **AirSim is held out as a cross-dataset test set**, not folded into training - gives R3 a real generalization finding (sim-to-real gap) and lets the avoidance planner be validated against AirSim's ground-truth depth maps. Only fold in if Phase 6 looks under-fit.
- Class scheme: **5 coarse classes** - `vehicle`, `person`, `static`, `flying`, `other` (collapsed from VisDrone's 10)
- Split: 70/15/15 stratified, **seed = 42** everywhere
- Image size: 640x640, batch tuned for L4 24 GB (YOLOv8n: 128, YOLOv8m: 64, RT-DETR-L: 20)
- Epochs: 50 (full), 5 (sanity)
- Selection rule: highest mAP@0.5 subject to FPS >= 30

**Compute target: Nvidia L4 (24 GB, $0.80/h).** `--device cuda` applies inside the training environment.

## Pipeline Status (as of 2026-05-28)

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0 — env | ✅ done | venv + requirements.txt |
| Phase 1 — raw data | ✅ done | `data/raw/VisDrone_Dataset/` (8629 labeled images, YOLO-format) |
| Phase 2 — validate raw | ✅ done | All three splits PASS; fixed 1 zero-height bbox in train |
| Phase 3 — preprocess | ✅ done | `scripts/preprocess.py`; `data/processed/` 6040/1294/1295 split |
| Phase 4 — configs | ✅ done | `configs/visdrone5.yaml`, `yolov8n/m/rtdetr.yaml`; tuned for L4 24 GB |
| Phase 5 — sanity run | ✅ done | YOLOv8n 100 epochs on Modal L4; mAP@0.5=47.4% confirmed pipeline healthy |
| Phase 6 — full training | 🔄 partial | YOLOv8n ✅ (47.4% mAP@0.5), YOLOv8m ✅ (59.2% mAP@0.5), RT-DETR-L ⏸ paused (VRAM constraints at batch=4 on L4 — needs L40S/A100 to run batch=16) |
| Phase 7 — evaluation | ⬜ next | `scripts/evaluate.py`; produce `reports/R3/model_comparison.md` |
| Phase 8 — integration | ⬜ | Wire best.pt into `main.py` + AirSim closed-loop demo for R4 |

## Training Results (Phase 6)

| Model | mAP@0.5 | mAP@0.5:0.95 | vehicle | person | other | Inference |
|-------|---------|--------------|---------|--------|-------|-----------|
| YOLOv8n | 47.4% | 24.4% | 74.5% | 32.9% | 34.9% | 0.8ms (~1250 FPS) |
| YOLOv8m | 59.2% | 32.3% | 82.2% | 46.3% | 49.1% | ~5-7ms (~150 FPS) |
| RT-DETR-L | ⏸ paused | — | — | — | — | — |

Weights stored in Modal volume `cpv-data` under `/runs/<model>/weights/best.pt`. Fetch locally with `modal run modal_train.py::fetch --model <name>`.

## Conventions

- **Line length: 88** (Black default, matched in `setup.cfg` flake8 config with `E203,W503` ignored for Black compatibility).
- **Data and weights are gitignored** - `data/raw/`, `data/processed/`, and `models/*.pt|*.pth|*.onnx` never get committed. Only `data/samples/` is intended for tracked tiny test fixtures.
- **Course materials are gitignored** - `CPV_notes/`, `*.pptx`, `*.xlsx` stay local only, **except** the project rubric at `docs/SU26_AI2013_CPV301.xlsx` (whitelisted via `!docs/*.xlsx`).
- **Reports directory** (`reports/R1`-`R4`) is for final PDFs only; drafts and working files belong elsewhere.
- **Design docs** live in `docs/` (e.g., `docs/training_pipeline.md`). Update the doc when the design changes - it's the source of truth that R-round reports reference.
