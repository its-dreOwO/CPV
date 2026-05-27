# CPV301 - Drone Obstacle Avoidance (RT-DR-003)

**Research Question:** How can drones avoid dynamic obstacles during flight?

A three-stage CV pipeline — **detection → tracking → avoidance planning** — built
for the FPT University CPV301 course project. The detector is trained on
**VisDrone-DET** (remapped to 5 coarse classes); the tracker is Kalman + Hungarian
(SORT-style), and the avoidance planner is a geometric controller over tracked
velocities.

See [`docs/training_pipeline.md`](docs/training_pipeline.md) for the locked
training design and [`CLAUDE.md`](CLAUDE.md) for the working contract.

## Project Structure

```
CPV/
|-- configs/                  # dataset + per-model YAMLs (visdrone5, yolov8n/m, rtdetr)
|-- data/
|   |-- raw/                  # original VisDrone-DET tree (gitignored)
|   |-- processed/            # 5-class, 70/15/15 stratified split (gitignored)
|   `-- samples/              # small tracked test fixtures
|-- docs/                     # design docs (training_pipeline.md, rubric xlsx)
|-- models/                   # trained weights (*.pt, gitignored)
|-- notebooks/                # EDA + Kaggle training kernel
|-- reports/R1..R4/           # round submissions (slides + PDF + GitHub link)
|-- scripts/
|   |-- validate_data.py      # Phase 2 — image/label/video integrity checks
|   |-- preprocess.py         # Phase 3 — VisDrone 10 -> 5 classes + split
|   |-- train.py              # Phase 5/6 — Ultralytics training entry point
|   `-- evaluate.py           # Phase 7 — mAP / precision / recall / FPS dump
|-- src/
|   |-- detection/            # BaseDetector + YoloDetector
|   |-- tracking/             # BaseTracker + KalmanTracker
|   |-- avoidance/            # BaseAvoidancePlanner + GeometricPlanner
|   `-- utils/                # visualizer, data_validation helpers
|-- tests/                    # pytest suite
|-- main.py                   # live demo: detection -> tracking -> planning
|-- requirements.txt
`-- setup.cfg                 # flake8 + pytest config
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> `setuptools>=70` is pinned at the top of `requirements.txt` because
> `filterpy 1.4.5` breaks on the setuptools API removed in 68+. Install
> requirements as a group rather than `pip install filterpy` first.

## Run the live demo

```bash
python main.py --source 0                          # webcam
python main.py --source path/to/video.mp4
python main.py --source clip.mp4 --weights models/best.pt --device cuda
```

## Data pipeline (Phase 2 → 3)

```bash
# Validate raw VisDrone splits (YOLO-format labels, 10 source classes)
python scripts/validate_data.py \
    --images data/raw/VisDrone_Dataset/VisDrone2019-DET-train/images \
    --labels data/raw/VisDrone_Dataset/VisDrone2019-DET-train/labels \
    --num-classes 10

# Remap 10 -> 5 coarse classes and write a 70/15/15 stratified split
python scripts/preprocess.py
```

## Train

The R3 model lineup is three detectors trained with an **identical protocol**
(same split, same seed, same image size, same epochs):

| Config                  | Model       | Role                              |
|-------------------------|-------------|-----------------------------------|
| `configs/yolov8n.yaml`  | YOLOv8n     | Speed baseline / embedded floor   |
| `configs/yolov8m.yaml`  | YOLOv8m     | Primary R4 demo model             |
| `configs/rtdetr.yaml`   | RT-DETR-L   | Accuracy ceiling / arch contrast  |

```bash
# Phase 5 sanity (5 epochs)
python scripts/train.py --config configs/yolov8n.yaml --epochs 5 --device cuda

# Phase 6 full (50 epochs × 3 models)
python scripts/train.py --config configs/yolov8m.yaml --epochs 50 --device cuda

# On Kaggle, override the dataset path:
python scripts/train.py --config configs/yolov8m.yaml --epochs 50 --device cuda \
    --data-root /kaggle/input/<dataset-slug>
```

## Evaluate

Runs Ultralytics `model.val()` on the held-out split and prints the R3
comparison-table metrics (mAP@0.5, mAP@0.5:0.95, precision, recall, FPS, size).

```bash
python scripts/evaluate.py --weights models/yolov8m_best.pt \
    --data configs/visdrone5.yaml --split test --device cuda \
    --output reports/R3/yolov8m_metrics.json
```

**Selection rule:** highest mAP@0.5 subject to FPS ≥ 30 → saved as `models/best.pt`.

## Tests & lint

```bash
pytest                          # 13 tests, coverage on src/
black --check . && flake8 .     # exactly what CI runs
```

## CI

Every push and pull request runs **Black** (format check), **Flake8** (lint),
and **pytest** via GitHub Actions. All three must pass.
