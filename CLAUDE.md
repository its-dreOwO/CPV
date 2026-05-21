# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This is a **CPV301 (Computer Vision) course project** at FPT University implementing **RT-DR-003: Obstacle Avoidance for Drones**. The research question is *"How can drones avoid dynamic obstacles during flight?"* and the expected output is a working prototype.

Submissions are organized into four rounds (`reports/R1`–`R4`), each requiring slides + report (PDF, English) plus a public GitHub link. R2 onward must include source code; R4 is the final demo. The full rubric is committed at `docs/SU26_AI2013_CPV301.xlsx`. The locked training pipeline design (model lineup, decisions, R-round mapping) is in `docs/training_pipeline.md` — read it before changing anything related to training.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run demo pipeline (webcam or video file)
python main.py --source 0
python main.py --source path/to/video.mp4

# Validate dataset before training (images + YOLO labels, and/or videos)
python scripts/validate_data.py --images data/raw/visdrone/images \
    --labels data/raw/visdrone/labels --num-classes 5
python scripts/validate_data.py --videos data/raw/airsim/sequences

# Train / evaluate
python scripts/train.py --config configs/yolov8m.yaml --epochs 50 --device cuda
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

The pipeline is a three-stage CV chain: **detection → tracking → avoidance planning**, with each stage defined as a base class that concrete implementations subclass. This separation lets the team swap models (YOLO variants, different trackers, planners) without changing downstream code — important because R3 requires comparing ≥3 different models.

**Data flow:**
```
frame (np.ndarray)
  → BaseDetector.detect()    → List[Detection]   (bbox, confidence, class_id)
  → BaseTracker.update()     → List[Track]       (track_id, bbox, velocity, age)
  → BaseAvoidancePlanner.plan() → (yaw_delta, altitude_delta)
```

**Key contracts** (in `src/`):
- `detection/detector.py` — `Detection` dataclass + `BaseDetector.detect(frame) -> List[Detection]`
- `tracking/tracker.py` — `Track` dataclass (carries velocity for dynamic obstacle prediction) + `BaseTracker.update(detections) -> List[Track]`
- `avoidance/planner.py` — `BaseAvoidancePlanner.plan(tracks) -> (yaw_delta, altitude_delta)`
- `utils/visualizer.py` — rendering helpers (decoupled from pipeline logic)

`main.py` wires these together for the live demo. `scripts/train.py` and `scripts/evaluate.py` are standalone entry points for model lifecycle, not part of the inference pipeline. `scripts/validate_data.py` runs pre-training integrity checks (image readability, YOLO label format, image/label pairing, video integrity); reusable helpers live in `src/utils/data_validation.py`.

**Import style:** modules import from `src.*` (absolute), so always run pytest/scripts from the repo root.

## Training Plan (locked — see `docs/training_pipeline.md`)

Only the **detector** is trained. Tracker = Kalman + Hungarian (rule-based). Planner = geometric (rule-based).

**Model lineup for the R3 ≥3-model comparison:**
- **YOLOv8n** — speed baseline / embedded floor
- **YOLOv8m** — primary demo model (the one that ships in R4)
- **RT-DETR-L** — accuracy ceiling / architecture contrast (transformer vs CNN)

The lineup is along two controlled axes — capacity (n→m) and architecture (m→RT-DETR-L) — so any performance gap is attributable to the model, not the training loop. All three use the same Ultralytics API.

**Locked decisions** (full table in `docs/training_pipeline.md`):
- Primary dataset: **VisDrone-DET** (Kaggle: `banuprasadb/visdrone-dataset`)
- **AirSim is held out as a cross-dataset test set**, not folded into training — gives R3 a real generalization finding (sim-to-real gap) and lets the avoidance planner be validated against AirSim's ground-truth depth maps. Only fold in if Phase 6 looks under-fit.
- Class scheme: **5 coarse classes** — `vehicle`, `person`, `static`, `flying`, `other` (collapsed from VisDrone's 10)
- Split: 70/15/15 stratified, **seed = 42** everywhere
- Image size: 640×640, batch 16 (drop to 8 if RT-DETR OOMs)
- Epochs: 50 (full), 5 (sanity)
- Selection rule: highest mAP@0.5 subject to FPS ≥ 30

**Compute target: Kaggle Kernels (free GPU).** Training runs are pushed to Kaggle via the kaggle-cli — don't assume a local CUDA card. The `--device cuda` flag in the commands above applies inside the Kaggle notebook environment.

## Conventions

- **Line length: 88** (Black default, matched in `setup.cfg` flake8 config with `E203,W503` ignored for Black compatibility).
- **Data and weights are gitignored** — `data/raw/`, `data/processed/`, and `models/*.pt|*.pth|*.onnx` never get committed. Only `data/samples/` is intended for tracked tiny test fixtures.
- **Course materials are gitignored** — `CPV_notes/`, `*.pptx`, `*.xlsx` stay local only, **except** the project rubric at `docs/SU26_AI2013_CPV301.xlsx` (whitelisted via `!docs/*.xlsx`).
- **Reports directory** (`reports/R1`–`R4`) is for final PDFs only; drafts and working files belong elsewhere.
- **Design docs** live in `docs/` (e.g., `docs/training_pipeline.md`). Update the doc when the design changes — it's the source of truth that R-round reports reference.
