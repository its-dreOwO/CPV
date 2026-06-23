# Vehicle Detection Training Pipeline

Operational guide for training, evaluating, and validating the CPV301 vehicle-perception
system. This document covers the three-model BDD100K training run, cross-dataset
KITTI evaluation, attribute-sliced robustness tests, and risk-zone validation.

The design spec (`docs/superpowers/specs/2026-06-22-vehicle-avoidance-pivot-design.md`)
is the source of truth for architecture and problem framing. This document is the
*runbook* — exact commands, expected artifacts, gotchas.

---

## Contents

1. [Locked decisions](#1-locked-decisions)
2. [Model lineup](#2-model-lineup)
3. [Dataset recap](#3-dataset-recap)
4. [Phase A — Modal training](#4-phase-a--modal-training)
5. [Phase B — Evaluation](#5-phase-b--evaluation)
6. [Phase C — Risk-zone validation](#6-phase-c--risk-zone-validation)
7. [R-round artifact map](#7-r-round-artifact-map)

---

## 1. Locked decisions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Image size | 640 × 640 | Standard YOLO inference size; fits all three models |
| Epochs (full) | 50 | Sufficient convergence on BDD100K at this scale |
| Epochs (sanity) | 5 | Quick smoke-test before committing GPU hours |
| Seed | 42 | Applied to all training runs for reproducibility |
| Split ratio | 70 / 15 / 15 | Stratified; applied once by `scripts/preprocess.py` |
| Model selection rule | Highest mAP@0.5 subject to FPS ≥ 30 | Speed floor keeps R4 demo real-time |
| Protocol | Identical across all three models | Differences attributable to model, not training loop |

---

## 2. Model lineup

Three models provide two controlled comparison axes for R3:

| Handle | Architecture | Config | Role |
|--------|-------------|--------|------|
| `yolov8n` | YOLOv8-nano (CNN) | `configs/yolov8n.yaml` | Speed floor / embedded baseline |
| `yolov8m` | YOLOv8-medium (CNN) | `configs/yolov8m.yaml` | Primary R4 demo model |
| `rtdetr` | RT-DETR-L (transformer) | `configs/rtdetr.yaml` | Accuracy ceiling / architecture contrast |

Axis 1 (capacity): `yolov8n` → `yolov8m` — answers "how much does model scale help?"
Axis 2 (architecture): `yolov8m` → `rtdetr` — answers "CNN vs transformer, same data?"

RT-DETR-L is configured for `batch: 4` (L4 24 GB). YOLOv8n uses `batch: 64`;
YOLOv8m uses `batch: 16` (see its YAML). If L4 is unavailable for RT-DETR-L,
request an L40S or A100 via the Modal GPU override.

---

## 3. Dataset recap

BDD100K was preprocessed to three coarse detection classes:

| Class ID | Name | BDD100K source labels |
|----------|------|-----------------------|
| 0 | `vehicle` | car, truck, bus, train |
| 1 | `person` | pedestrian, rider |
| 2 | `two_wheeler` | bicycle, motorcycle |

Traffic lights and traffic signs are dropped (not relevant to collision risk).

**Split sizes:**

| Split | Images | Labels | Produced from |
|-------|--------|--------|---------------|
| train | 57,287 | 57,287 | BDD100K official train split, stratified 82% |
| val | 12,576 | 12,576 | BDD100K official train split, stratified 18% |
| test | 10,000 | 10,000 | BDD100K official val split (held-out) |

Processed layout:

```
data/processed/bdd100k/
  train/images/   train/labels/
  val/images/     val/labels/
  test/images/    test/labels/
  attributes.csv   # name, split, weather, scene, timeofday (79,863 rows)
```

Dataset config: `configs/bdd100k.yaml`

To regenerate from raw:

```bash
python scripts/preprocess.py
python scripts/validate_data.py --images data/processed/bdd100k/train/images \
    --labels data/processed/bdd100k/train/labels --num-classes 3
python scripts/validate_data.py --images data/processed/bdd100k/val/images \
    --labels data/processed/bdd100k/val/labels --num-classes 3
python scripts/validate_data.py --images data/processed/bdd100k/test/images \
    --labels data/processed/bdd100k/test/labels --num-classes 3
```

See `docs/bdd100k_data_validation.md` for the full validation report (all three
splits passed).

---

## 4. Phase A — Modal training

Training runs on Modal GPU cloud via `modal_train.py`. The in-container entrypoint
is `scripts/train.py`. All commands run from the repo root.

### 4.1 One-time setup

Install Modal and authenticate:

```bash
pip install modal
modal setup
```

### 4.2 Upload the dataset (one-time)

Tar the processed data first — a single archive is far faster to upload than
~80,000 individual files. **Use `-h`** so the symlinked images under
`data/processed/bdd100k/*/images` are dereferenced into the archive (a plain
`tar` would store dangling symlinks that don't resolve inside the container):

```bash
tar czhf processed.tar.gz -C data processed
modal volume create cpv-bdd100k
modal volume put cpv-bdd100k processed.tar.gz /processed.tar.gz
rm processed.tar.gz   # optional local cleanup
```

The archive stays in the volume as a single file. Each training run unpacks it
to fast **local** container disk (`/root/data/processed/bdd100k`) on first use —
Modal Volumes are a network filesystem and far too slow for YOLO's per-epoch
reads of ~80k small images (the unpack alone times out against the volume). No
separate extract step is needed.

**GOTCHA — `--data-root` must point at the BDD100K directory, not its parent.**
`scripts/train.py` accepts `--data-root` to override the `path:` field in the
dataset YAML at runtime. The correct value is `/vol/processed/bdd100k` (the
directory that contains `train/`, `val/`, `test/`). Passing `/vol/processed`
(one level up) causes Ultralytics to fail to find the split subdirectories.
`modal_train.py` already passes the correct path — do not change it.

### 4.3 Sanity run (5 epochs)

Verify the pipeline end-to-end before committing to a full training run (~15 min,
~$0.15 on L4):

```bash
modal run modal_train.py::main --model yolov8n --epochs 5
```

### 4.4 Full training runs (50 epochs each)

Run each model independently. Use `--fresh` to discard any prior partial run
for that model:

```bash
modal run modal_train.py::main --model yolov8n --epochs 50 --fresh
modal run modal_train.py::main --model yolov8m --epochs 50 --fresh
modal run modal_train.py::main --model rtdetr  --epochs 50 --fresh
```

### 4.5 Download trained weights

```bash
modal run modal_train.py::fetch --model yolov8n   # saves models/yolov8n-best.pt
modal run modal_train.py::fetch --model yolov8m   # saves models/yolov8m-best.pt
modal run modal_train.py::fetch --model rtdetr    # saves models/rtdetr-best.pt
```

Weights files are gitignored (`models/*.pt`). Store them locally or in cloud
storage; do not commit them.

### 4.6 Local training (alternative)

For CPU smoke-tests or if Modal is unavailable:

```bash
python scripts/train.py --config configs/yolov8n.yaml --epochs 5 --device cpu
python scripts/train.py --config configs/yolov8m.yaml --epochs 50 --device cuda
```

To resume an interrupted run:

```bash
python scripts/train.py --config configs/yolov8m.yaml --resume
```

### 4.7 GPU / cost reference

| GPU | $/hr | VRAM | Notes |
|-----|------|------|-------|
| L4 | $0.80 | 24 GB | Default; best value for all three models |
| A10G | $1.10 | 24 GB | Fallback if L4 unavailable |
| T4 | $0.59 | 16 GB | Budget option for YOLOv8n/m; RT-DETR-L is an OOM risk on T4 (16 GB) — `modal_train.py` notes it may OOM around batch 8; the shipped config uses `batch: 4` tuned for the L4 (24 GB), not T4 |
| L40S / A100 | higher | 48 GB+ | Recommended for RT-DETR-L; eliminates the memory risk |

Estimated full-run cost per model on L4: ~$1–2 (50 epochs, BDD100K 57k images).
Modal volume storage: ~$0.20/GB/month (~$0.40/mo for the ~1.9 GB dataset).

---

## 5. Phase B — Evaluation

All evaluation scripts run from the repo root. Always use `--device 0` (GPU) for
accurate FPS measurements; CPU timing is not comparable.

### 5.1 Standard evaluation (BDD100K test split)

Run for each trained model:

```bash
python scripts/evaluate.py \
    --weights models/yolov8n-best.pt \
    --data configs/bdd100k.yaml \
    --split test --device 0 \
    --classwise \
    --output reports/R3/yolov8n_metrics.json

python scripts/evaluate.py \
    --weights models/yolov8m-best.pt \
    --data configs/bdd100k.yaml \
    --split test --device 0 \
    --classwise \
    --output reports/R3/yolov8m_metrics.json

python scripts/evaluate.py \
    --weights models/rtdetr-best.pt \
    --data configs/bdd100k.yaml \
    --split test --device 0 \
    --classwise \
    --output reports/R3/rtdetr_metrics.json
```

**Metrics emitted** (in the JSON and printed to stdout):

| Field | Description |
|-------|-------------|
| `map50` | mAP@0.5 (primary selection criterion) |
| `map50_95` | mAP@0.5:0.95 |
| `precision` | Mean precision across classes |
| `recall` | Mean recall across classes |
| `fps_inference_only` | 1000 / inference_ms |
| `fps_end_to_end` | 1000 / (pre + infer + post ms) |
| `params_millions` | Parameter count in millions |
| `weights_size_mb` | On-disk weight file size |
| `per_class` | Per-class mAP@0.5 for vehicle / person / two_wheeler (with `--classwise`) |

**Model selection rule:** highest `map50` subject to `fps_end_to_end >= 30`. The
winning model ships as the R4 live demo.

### 5.2 Robustness evaluation (attribute-sliced mAP)

Slice by time of day and weather using `attributes.csv`:

```bash
# Time of day slices (daytime / night / dawn/dusk / undefined)
python scripts/evaluate_robustness.py \
    --weights models/yolov8m-best.pt \
    --data configs/bdd100k.yaml \
    --by timeofday --device 0 \
    --output reports/R3/yolov8m_robustness_timeofday.json

# Weather slices (clear / overcast / rainy / snowy / foggy / partly cloudy)
python scripts/evaluate_robustness.py \
    --weights models/yolov8m-best.pt \
    --data configs/bdd100k.yaml \
    --by weather --device 0 \
    --output reports/R3/yolov8m_robustness_weather.json
```

Run these for all three models. Each JSON contains per-slice `map50` and
`n_images`. This drives the R3 day/night and weather robustness analysis.

### 5.3 KITTI zero-shot evaluation (cross-dataset generalization)

KITTI is a **held-out** dataset — it is never trained on. It measures how well
a BDD100K-trained model generalizes to a different domain (different camera,
country, image statistics).

**Prerequisites:** Download KITTI object detection data manually from
https://www.cvlibs.net/datasets/kitti/eval_object.php (registration required).
Place the files at:

```
data/raw/kitti/image_2/   # left color images (*.png)
data/raw/kitti/label_2/   # training labels (*.txt)
```

Convert to the 3-class YOLO layout:

```bash
python scripts/preprocess_kitti.py
# Output: data/processed/kitti/{images,labels}/
```

Run zero-shot evaluation:

```bash
python scripts/evaluate.py \
    --weights models/yolov8m-best.pt \
    --data configs/kitti.yaml \
    --split val --device 0 \
    --classwise \
    --output reports/R3/yolov8m_kitti_metrics.json
```

Repeat for the other two models. The KITTI-vs-BDD100K mAP gap quantifies the
domain shift (R3 cross-dataset section).

---

## 6. Phase C — Risk-zone validation

`scripts/validate_risk.py` validates the `RiskZoneAssessor` heuristic against
KITTI ground-truth 3D distances. It does not require trained weights — the risk
assessor is a geometric rule, not a learned model.

**What it measures:** For each KITTI frame, ground-truth bounding boxes are fed
into `RiskZoneAssessor` (with zero velocity, since KITTI is single-frame). The
script reports what fraction of DANGER-labelled objects are within a configurable
distance threshold (`--near-thresh-m`, default 15 m). The key metric is
`precision_danger_near`: "of all objects the heuristic flagged as DANGER, X%
were truly within 15 m."

```bash
python scripts/validate_risk.py \
    --kitti data/processed/kitti \
    --raw-labels data/raw/kitti/label_2 \
    --near-thresh-m 15 \
    --output reports/R3/risk_validation.json
```

**Single-frame caveat:** KITTI provides one frame at a time with no temporal
context. The `RiskZoneAssessor` has two DANGER signals: (1) in-path object
that is large (close to camera), and (2) in-path object whose bbox area is
growing (`scale_velocity > threshold`). Because KITTI is single-frame,
`scale_velocity` is always zero — the closing/TTC proxy never fires. This
validation tests the **WHERE** geometry (is the ego-path trapezoid correctly
predicting in-path objects near the vehicle?) but does **not** validate the
TTC proximity proxy. TTC validation requires a video sequence with real
closing objects, such as BDD100K clips.

**Interpreting the output:**

```json
{
  "danger_total": 412,
  "danger_near": 389,
  "caution_total": 107,
  "safe_total": 1843,
  "precision_danger_near": 0.944
}
```

A `precision_danger_near` close to 1.0 means the ego-path geometry reliably
coincides with nearby objects (good WHERE accuracy). Lower values indicate
the trapezoid is too wide or not angled correctly.

---

## 7. R-round artifact map

| Round | What ships | Artifact paths |
|-------|-----------|----------------|
| **R1** | Problem statement, design, BDD100K data-validation report | `docs/bdd100k_data_validation.md` |
| **R2** | YOLOv8n trained on BDD100K, initial risk-overlay demo | `models/yolov8n-best.pt`, `reports/R2/` slides + report |
| **R3** | 3-model comparison, KITTI cross-dataset, robustness, risk validation | See below |
| **R4** | Best model in live dashcam demo + Streamlit showcase | `models/<best>-best.pt`, `prototype/web_app.py` |

**R3 artifact breakdown:**

```
reports/R3/
  yolov8n_metrics.json          # BDD100K test mAP, FPS, params
  yolov8m_metrics.json
  rtdetr_metrics.json
  yolov8n_kitti_metrics.json    # KITTI zero-shot (cross-dataset)
  yolov8m_kitti_metrics.json
  rtdetr_kitti_metrics.json
  yolov8n_robustness_timeofday.json   # day / night / dawn / dusk slices
  yolov8m_robustness_timeofday.json
  rtdetr_robustness_timeofday.json
  yolov8n_robustness_weather.json     # clear / rainy / foggy / etc.
  yolov8m_robustness_weather.json
  rtdetr_robustness_weather.json
  risk_validation.json          # Approach C: DANGER precision vs KITTI GT distance
```

---

## Quick-reference command index

```bash
# Preprocess BDD100K
python scripts/preprocess.py

# Validate splits
python scripts/validate_data.py --images data/processed/bdd100k/train/images \
    --labels data/processed/bdd100k/train/labels --num-classes 3

# Upload dataset to Modal (one-time; -h dereferences the symlinked images)
tar czhf processed.tar.gz -C data processed
modal volume create cpv-bdd100k
modal volume put cpv-bdd100k processed.tar.gz /processed.tar.gz

# Sanity training run
modal run modal_train.py::main --model yolov8n --epochs 5

# Full training (all three models)
modal run modal_train.py::main --model yolov8n --epochs 50 --fresh
modal run modal_train.py::main --model yolov8m --epochs 50 --fresh
modal run modal_train.py::main --model rtdetr  --epochs 50 --fresh

# Download weights
modal run modal_train.py::fetch --model yolov8n
modal run modal_train.py::fetch --model yolov8m
modal run modal_train.py::fetch --model rtdetr

# BDD100K evaluation (repeat for each model)
python scripts/evaluate.py --weights models/yolov8m-best.pt \
    --data configs/bdd100k.yaml --split test --device 0 \
    --classwise --output reports/R3/yolov8m_metrics.json

# Robustness slices
python scripts/evaluate_robustness.py --weights models/yolov8m-best.pt \
    --data configs/bdd100k.yaml --by timeofday --device 0 \
    --output reports/R3/yolov8m_robustness_timeofday.json
python scripts/evaluate_robustness.py --weights models/yolov8m-best.pt \
    --data configs/bdd100k.yaml --by weather --device 0 \
    --output reports/R3/yolov8m_robustness_weather.json

# KITTI zero-shot
python scripts/preprocess_kitti.py
python scripts/evaluate.py --weights models/yolov8m-best.pt \
    --data configs/kitti.yaml --split val --device 0 \
    --classwise --output reports/R3/yolov8m_kitti_metrics.json

# Risk-zone validation
python scripts/validate_risk.py \
    --kitti data/processed/kitti \
    --raw-labels data/raw/kitti/label_2 \
    --output reports/R3/risk_validation.json
```
