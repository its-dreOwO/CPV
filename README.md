# CPV301 - Vehicle Pedestrian & Vehicle Avoidance

**Research Question:** How can vehicles avoid pedestrians and vehicles?

A forward-facing dashcam **perception + risk-advisory** system built for the FPT
University CPV301 course project. A three-stage CV pipeline —
**detection → tracking → risk assessment** — detects road obstacles, tracks them,
and tags each as **SAFE / CAUTION / DANGER** (no control output; it is a
perception system, not a vehicle controller).

The detector is trained on **BDD100K** (collapsed to 3 coarse classes:
`vehicle`, `person`, `two_wheeler`); the tracker is Kalman + Hungarian
(SORT-style); the risk assessor is a geometric ego-path heuristic over tracked
obstacles. **KITTI** is held out for cross-dataset generalization and
ground-truth risk validation.

See [`docs/superpowers/specs/`](docs/superpowers/specs/) for the locked design
spec, [`docs/superpowers/plans/`](docs/superpowers/plans/) for the implementation
plans, and [`CLAUDE.md`](CLAUDE.md) for the working contract.

> **Pivot note:** this project pivoted from a drone obstacle-avoidance problem.
> The perception→track pipeline and the 3-model comparison were retained; the
> avoidance planner was replaced by the risk assessor. The data pipeline
> (`scripts/`, model `configs/`, `modal_train.py`) is being repointed from
> VisDrone to BDD100K across Plans 2–3 and still references the old dataset until
> then.

## Project Structure

```
CPV/
|-- configs/                  # per-model YAMLs (yolov8n/m, rtdetr) + bdd100k dataset YAML (Plan 2)
|-- data/
|   |-- raw/                  # original BDD100K / KITTI trees (gitignored)
|   |-- processed/            # 3-class, 70/15/15 stratified split (gitignored)
|   `-- samples/              # small tracked test fixtures
|-- docs/
|   |-- superpowers/          # design specs + implementation plans (source of truth)
|   `-- SU26_AI2013_CPV301.xlsx  # course rubric
|-- models/                   # trained weights (*.pt, gitignored)
|-- reports/R1..R4/           # round submissions (slides + PDF + GitHub link)
|-- scripts/                  # validate_data / preprocess / train / evaluate (BDD100K adaptation: Plan 2)
|-- src/
|   |-- detection/            # BaseDetector + YoloDetector
|   |-- tracking/             # BaseTracker + KalmanTracker (exposes bbox-area growth)
|   |-- risk/                 # BaseRiskAssessor + RiskZoneAssessor (SAFE/CAUTION/DANGER)
|   `-- utils/                # visualizer, data_validation helpers
|-- prototype/                # Streamlit showcase app (re-skin to vehicle: Plan 3)
|-- tests/                    # pytest suite
|-- main.py                   # live demo: detection -> tracking -> risk overlay
|-- requirements.txt
`-- setup.cfg                 # flake8 + pytest config
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Fish: source .venv/bin/activate.fish
pip install -r requirements.txt
```

> `setuptools>=70` is pinned at the top of `requirements.txt` because
> `filterpy 1.4.5` breaks on the setuptools API removed in 68+. Install
> requirements as a group rather than `pip install filterpy` first.

## Run the live demo

```bash
python main.py --source 0                          # webcam
python main.py --source path/to/dashcam.mp4
python main.py --source clip.mp4 --weights models/best.pt --device cuda
```

The demo overlays the ego-path region and color-coded risk boxes
(green = SAFE, amber = CAUTION, red = DANGER).

## Train (R3 model lineup)

Three detectors trained with an **identical protocol** (same split, seed, image
size, epochs) so any performance gap is attributable to the model:

| Config                  | Model       | Role                              |
|-------------------------|-------------|-----------------------------------|
| `configs/yolov8n.yaml`  | YOLOv8n     | Speed baseline / embedded floor   |
| `configs/yolov8m.yaml`  | YOLOv8m     | Primary R4 demo model             |
| `configs/rtdetr.yaml`   | RT-DETR-L   | Accuracy ceiling / arch contrast  |

```bash
python scripts/train.py --config configs/yolov8m.yaml --epochs 50 --device cuda
```

> The data pipeline (`scripts/preprocess.py`, `scripts/validate_data.py`,
> `configs/bdd100k.yaml`) is implemented for BDD100K in Plan 2; until then the
> training entry points still reference the old VisDrone layout.

## Tests & lint

```bash
pytest                          # coverage on src/
black --check . && flake8 .     # exactly what CI runs
```

## CI

Every push and pull request runs **Black** (format check), **Flake8** (lint),
and **pytest** via GitHub Actions. All three must pass.
