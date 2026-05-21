# Training Pipeline — RT-DR-003 Drone Obstacle Avoidance

> **Status:** locked design, pre-implementation. This document is the source of
> truth for the R1 prototype pipeline and the R3 multi-model comparison.

Only the **detector** is trained. The tracker is a Kalman + Hungarian assignment
(rule-based); the avoidance planner is a geometric controller over tracked
velocities. Neither requires training data.

---

## 1. Model lineup (locked)

Three detectors chosen along two controlled axes — capacity and architecture —
so the R3 comparison isolates *why* a model wins, not just *which*.

| Model | Role | Weights | Params | Size | ~FPS (T4) |
|-------|------|---------|--------|------|-----------|
| **YOLOv8n** | Speed baseline / embedded floor | `yolov8n.pt` | 3.2 M | 3 MB | ~80 |
| **YOLOv8m** | Sweet spot / primary demo model | `yolov8m.pt` | 25.9 M | 25 MB | ~50 |
| **RT-DETR-L** | Accuracy ceiling / architecture contrast | `rtdetr-l.pt` | 32 M | 33 MB | ~30 |

**Comparisons R3 will report:**

- **YOLOv8n → YOLOv8m** — same architecture, capacity scales 8× → answers
  *"does scale help on drone data?"*
- **YOLOv8m → RT-DETR-L** — comparable capacity, CNN vs transformer →
  answers *"does the architectural paradigm matter for aerial detection?"*

All three trained with **identical protocol** (same data split, same epochs,
same image size, same augmentation, same seed) so any performance gap is
attributable to the model, not the training loop.

---

## 2. Critical decisions (locked)

| Decision | Value | Rationale |
|----------|-------|-----------|
| Primary dataset | VisDrone-DET | Standard benchmark, citable, ~2 GB |
| Supplementary | AirSim RGB+depth, Bird-vs-Drone | Dynamic-obstacle stress sets |
| Class scheme | 5 coarse classes: `vehicle`, `person`, `static`, `flying`, `other` | Avoidance doesn't need fine-grained taxonomy; reduces class imbalance |
| Image size | 640 × 640 | YOLO default; balances speed vs small-object recall |
| Epochs | 50 (full) / 5 (sanity) | Standard for fine-tuning from COCO-pretrained weights |
| Batch size | 16 default, drop to 8 if RT-DETR OOMs | RT-DETR uses ~1.5× the VRAM of YOLOv8m |
| Optimizer | SGD (Ultralytics default) | Don't fight the framework |
| Seed | 42 | Single source of randomness across all runs |
| Selection criterion | `mAP@0.5 ≥ X subject to FPS ≥ 30` | Locks the speed/accuracy tradeoff up front |
| Training framework | Ultralytics (`pip install ultralytics`) | Same API for YOLOv8 and RT-DETR |

---

## 3. Pipeline phases

### Phase 0 — Environment

- Activate venv, `pip install -r requirements.txt`
- Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Set seed = 42 in every training entry point

**Output:** working env, known device.

---

### Phase 1 — Acquire raw data

Land datasets read-only under `data/raw/`:

```
data/raw/
├── visdrone/      # VisDrone-DET (primary)
├── airsim/        # AirSim RGB+depth (sim, depth ground truth)
└── bird-drone/    # Dynamic-obstacle stress set
```

Download via `kaggle datasets download` or `kagglehub.dataset_download()`.

**Output:** read-only raw trees. Never edited after this point.

---

### Phase 2 — Validate raw data

```bash
python scripts/validate_data.py --images data/raw/visdrone/images \
    --labels data/raw/visdrone/labels --num-classes 10
python scripts/validate_data.py --videos data/raw/airsim/sequences
```

**Decision gate:** PASS → continue. FAIL → fix or drop the dataset.

**Output:** `reports/R1/data_validation.md` (summary tables + class
distribution + image size stats).

---

### Phase 3 — Process raw → training-ready

| Sub-step | Operation |
|----------|-----------|
| Annotation conversion | VisDrone format → YOLO format (`class x y w h` normalized) |
| Class harmonization | Map 10 VisDrone classes → 5 coarse classes (see decision table) |
| Train/val/test split | 70 / 15 / 15, stratified by class, seed = 42 |
| Output layout | `data/processed/<dataset>/{train,val,test}/{images,labels}` |

Image resize happens at dataloader time, not on disk — keep native resolution
in `processed/` so we can experiment with input sizes later.

**Re-validate after this step** — corruption sneaks in during conversion.

---

### Phase 4 — Config files

`configs/` will hold:

```
configs/
├── visdrone.yaml      # dataset paths, class names, nc=5
├── yolov8n.yaml       # model + hyperparameters
├── yolov8m.yaml
└── rtdetr.yaml
```

Every training run is reproducible from `(config, seed)`. No hyperparameters
inside `scripts/train.py`.

---

### Phase 5 — Baseline sanity run

Train **YOLOv8n** for **5 epochs** on a **~500-image subset**.

Pass criteria:
- Loss decreases monotonically
- Val mAP > random (>10% mAP@0.5 on 5 classes)
- No NaN, no OOM
- Total wall time < 15 minutes

If anything fails here, **stop** — debug before launching the full Phase 6.

**Output:** `models/sanity_yolov8n.pt` (throwaway).

---

### Phase 6 — Full training (the R3 deliverable)

Run identical 50-epoch training for all three models on the full processed
dataset:

```bash
python scripts/train.py --config configs/yolov8n.yaml --epochs 50 --device cuda
python scripts/train.py --config configs/yolov8m.yaml --epochs 50 --device cuda
python scripts/train.py --config configs/rtdetr.yaml  --epochs 50 --device cuda
```

Each run produces:
- `models/<name>_best.pt`  — best mAP checkpoint
- `models/<name>_last.pt`  — final-epoch checkpoint
- `runs/<name>/`           — TensorBoard logs, training curves, sample predictions

**Time budget per run:** ~2–4 hours on a single T4. RT-DETR is the slowest.

---

### Phase 7 — Evaluation & model selection

Per-model metrics, computed on the held-out **test** split:

| Category | Metrics |
|----------|---------|
| Accuracy | mAP@0.5, mAP@0.5:0.95, precision, recall, F1 |
| Per-class | AP per class, confusion matrix |
| Speed | FPS on target HW, model size (MB), params (M) |
| Robustness | mAP under test-time noise / motion blur / low-light |

```bash
python scripts/evaluate.py --weights models/yolov8n_best.pt --data configs/visdrone.yaml
python scripts/evaluate.py --weights models/yolov8m_best.pt --data configs/visdrone.yaml
python scripts/evaluate.py --weights models/rtdetr_best.pt  --data configs/visdrone.yaml
```

Selection rule: **highest mAP@0.5 subject to FPS ≥ 30 on the target hardware.**
Save chosen weights as `models/best.pt`.

**Output:** `reports/R3/model_comparison.md` with the full comparison table.

---

### Phase 8 — Pipeline integration

Subclass `BaseDetector` with the chosen weights:

```python
# src/detection/yolo_detector.py (Phase 8, not yet written)
class YoloDetector(BaseDetector):
    def __init__(self, weights: str): ...
    def detect(self, frame): ...
```

Wire into `main.py`, test on a held-out video clip, measure **end-to-end FPS**
(detection + tracking + planning combined — not just detection).

**Output:** working R4 demo.

---

## 4. Visual flow

```
[ Phase 0: env ]
       ↓
[ Phase 1: raw download ] ─────────→ data/raw/
       ↓
[ Phase 2: validate ] ──FAIL──→ fix / drop
       ↓ PASS
[ Phase 3: convert + 70/15/15 split ] → data/processed/
       ↓
[ Phase 4: configs ] ──────────────→ configs/*.yaml
       ↓
[ Phase 5: sanity (5 ep, subset) ] ──FAIL──→ debug
       ↓ PASS
[ Phase 6: train YOLOv8n + YOLOv8m + RT-DETR-L (50 ep each) ]
       ↓
[ Phase 7: evaluate + select by mAP@0.5 s.t. FPS≥30 ] → models/best.pt
       ↓
[ Phase 8: wire into BaseDetector + main.py ] → R4 demo
```

---

## 5. Open questions (need user input before Phase 0)

1. **GPU availability** — local CUDA card, Colab, or Kaggle Kernels? Determines
   batch size and whether Phase 6 takes 6 hours or 2 days.
2. **Exact 5-class mapping from VisDrone's 10 classes** — proposal below, open
   to revision:

   | Coarse class | VisDrone source classes |
   |--------------|-------------------------|
   | `vehicle` | car, van, truck, bus |
   | `person` | pedestrian, person |
   | `static` | (none in VisDrone — reserved for AirSim trees/buildings) |
   | `flying` | (none in VisDrone — reserved for Bird-vs-Drone) |
   | `other` | bicycle, tricycle, awning-tricycle, motor |

3. **AirSim integration timing** — bring in during Phase 3 (joint training) or
   keep as a held-out test set (cross-dataset generalization story for R3)?

---

## 6. R-round mapping

| Round | What from this pipeline is in the submission |
|-------|----------------------------------------------|
| **R1** | Phases 0–4 designed (this doc), Phase 2 validation report on raw data |
| **R2** | Phases 0–5 executed, one model trained (YOLOv8n), initial results |
| **R3** | Phases 6–7 executed, full 3-model comparison table |
| **R4** | Phase 8 wired into the live demo |
