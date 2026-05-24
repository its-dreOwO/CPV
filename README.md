# CPV301 - Drone Obstacle Avoidance (RT-DR-003)

**Research Question:** How can drones avoid dynamic obstacles during flight?

## Project Structure

```
CPV/
|-- data/
|   |-- raw/          # original, unprocessed data
|   |-- processed/    # cleaned / augmented data
|   |__ samples/      # small sample files for quick testing
|-- models/           # saved weights (not tracked by git)
|-- notebooks/        # EDA and experiment notebooks
|-- reports/          # R1-R4 submission slides & reports
|   |-- R1/
|   |-- R2/
|   |-- R3/
|   |__ R4/
|-- scripts/
|   |-- train.py      # training entry point
|   |__ evaluate.py   # evaluation entry point
|-- src/
|   |-- detection/    # object detection module
|   |-- tracking/     # multi-object tracking module
|   |-- avoidance/    # avoidance planning module
|   |__ utils/        # shared helpers (visualizer, I/O, ...)
|-- tests/            # unit tests (pytest)
|-- main.py           # demo pipeline
|-- requirements.txt
|__ setup.cfg         # flake8 + pytest config
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Demo

```bash
python main.py --source 0              # webcam
python main.py --source path/to/video.mp4
```

## Train

```bash
python scripts/train.py --config configs/yolo.yaml --epochs 50 --device cuda
```

## Evaluate

```bash
python scripts/evaluate.py --weights models/best.pt --data data/processed/test
```

## Tests

```bash
pytest
```

## CI

Every push and pull request runs **Black** (format check) and **Flake8** (lint) via GitHub Actions.
