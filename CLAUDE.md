# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This is a **CPV301 (Computer Vision) course project** at FPT University implementing **RT-DR-003: Obstacle Avoidance for Drones**. The research question is *"How can drones avoid dynamic obstacles during flight?"* and the expected output is a working prototype.

Submissions are organized into four rounds (`reports/R1`–`R4`), each requiring slides + report (PDF, English) plus a public GitHub link. R2 onward must include source code; R4 is the final demo. See `SU26_AI2013_CPV301.xlsx` (gitignored, local only) for full rubric.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run demo pipeline (webcam or video file)
python main.py --source 0
python main.py --source path/to/video.mp4

# Train / evaluate
python scripts/train.py --config configs/yolo.yaml --epochs 50 --device cuda
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

`main.py` wires these together for the live demo. `scripts/train.py` and `scripts/evaluate.py` are standalone entry points for model lifecycle, not part of the inference pipeline.

**Import style:** modules import from `src.*` (absolute), so always run pytest/scripts from the repo root.

## Conventions

- **Line length: 88** (Black default, matched in `setup.cfg` flake8 config with `E203,W503` ignored for Black compatibility).
- **Data and weights are gitignored** — `data/raw/`, `data/processed/`, and `models/*.pt|*.pth|*.onnx` never get committed. Only `data/samples/` is intended for tracked tiny test fixtures.
- **Course materials are gitignored** — `CPV_notes/`, `*.pptx`, `*.xlsx` stay local only.
- **Reports directory** (`reports/R1`–`R4`) is for final PDFs only; drafts and working files belong elsewhere.
