# Plan 3 — Vehicle Pipeline: Training, Evaluation, Risk Validation & Docs

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the 3-model detector lineup on BDD100K (via Modal), evaluate them on the BDD100K test split + KITTI cross-dataset + day/night/weather robustness, validate the risk-zone heuristic against KITTI ground truth, and rewrite the docs/Streamlit app to the vehicle pivot.

**Architecture:** Detector-only training (tracker + risk assessor stay rule-based). Training runs on Modal L4/L40S GPUs through the existing `modal_train.py`; `scripts/train.py` is the in-container entrypoint. Evaluation reuses Ultralytics `model.val()` through `scripts/evaluate.py` plus two new scripts: `scripts/evaluate_robustness.py` (attribute-sliced mAP) and `scripts/validate_risk.py` (Approach C — risk labels vs KITTI ground-truth distance). KITTI is converted to the same 3-class YOLO layout for zero-shot detection eval, and parsed natively (3D boxes) for risk validation.

**Tech Stack:** Python 3.11, Ultralytics `>=8.2.0`, Modal `>=0.64.0`, PyYAML, pytest, NumPy. Black 88-col, flake8 (`E203,W503` ignored). KITTI object-detection dataset (left color images + label_2 + calib).

## Global Constraints

- **Line length 88** (Black default); flake8 config in `setup.cfg` (`E203,W503` ignored). CI runs `black --check .` then `flake8 .` — both must pass.
- **Seed = 42 everywhere.** Image size **640×640**. Epochs: **50 full / 5 sanity**.
- **Identical training protocol across all 3 models** (same split, epochs, imgsz, augmentation defaults, seed) so any gap is attributable to the model, not the loop.
- **Model lineup (locked):** `yolov8n` (speed floor), `yolov8m` (primary R4 demo), `rtdetr` = RT-DETR-L (accuracy/architecture contrast).
- **Selection rule:** highest **mAP@0.5 subject to FPS ≥ 30**.
- **3 classes only:** `0: vehicle`, `1: person`, `2: two_wheeler`. KITTI must be remapped to these same ids.
- **Data & weights are gitignored:** `data/raw/`, `data/processed/`, `models/*.pt`. Never commit them. Only `data/samples/` holds tiny tracked fixtures; tests build their own temp fixtures.
- **Run pytest/scripts from repo root** (modules use absolute `src.*` imports).
- **`prototype/` is excluded from flake8** (the `sys.path` shim trips E402) — format it with `black` manually.
- **KITTI is held-out: never folded into training.** Used only for zero-shot eval (Phase B) and risk validation (Phase C).
- The processed BDD100K dataset already exists on disk: `data/processed/bdd100k/{train,val,test}/{images,labels}` (57,287 / 12,576 / 10,000) plus `data/processed/bdd100k/attributes.csv` (`name,split,weather,scene,timeofday`).

---

## File Structure

**Created:**
- `scripts/evaluate_robustness.py` — slices BDD100K test mAP by `timeofday` and `weather` using `attributes.csv` (R3 robustness breakdown).
- `scripts/preprocess_kitti.py` — converts KITTI object-detection labels to the 3-class YOLO layout under `data/processed/kitti/` for zero-shot detection eval.
- `scripts/validate_risk.py` — Approach C: runs `RiskZoneAssessor` on KITTI frames and measures agreement between risk labels and KITTI ground-truth distance.
- `src/utils/kitti.py` — pure KITTI parsing/remap helpers (label_2 lines, calib, distance), unit-tested independently of disk.
- `docs/training_pipeline.md` — vehicle training pipeline doc (replaces the removed drone version).
- `tests/test_evaluate_robustness.py`, `tests/test_kitti.py`, `tests/test_validate_risk.py`, `tests/test_modal_train.py`.

**Modified:**
- `modal_train.py` — fix drone naming + the dataset-path mismatch (data lives at `/vol/processed/bdd100k`, not `/vol/processed`); add a `fetch_all` convenience and KITTI-agnostic notes.
- `scripts/evaluate.py` — fix the stale `visdrone5.yaml` docstring; pin the Ultralytics dataset path; add `--classwise` per-class mAP for the R3 table.
- `configs/bdd100k.yaml` — no functional change needed; confirm `path` contract (Ultralytics resolves relative `path` against its own datasets_dir; Phase A pins it via `--data-root`).
- `CLAUDE.md` — flip status table (Plan 3 → done) at the very end; update R-round / pipeline-status sections as phases land.
- `prototype/web_app.py` — rewrite copy + overlay for the 3-class vehicle demo (manual `black`, not lint-gated).

---

## Phase A — Training infrastructure + 3-model Modal sweep

Maps to R2 (YOLOv8n trained) and seeds R3 (all 3 trained). Code is TDD-able; the GPU runs are manual operator steps.

### Task 1: Fix `modal_train.py` dataset-path mismatch and drone naming

**Files:**
- Modify: `modal_train.py`
- Test: `tests/test_modal_train.py` (create)

**Interfaces:**
- Produces: `build_train_cmd(model: str, epochs: int, dataset_path: str, run_dir: str, resume: bool) -> list[str]` — pure function returning the `python scripts/train.py …` argv. Extracted so the argv (and the critical `--data-root` value) is unit-testable without Modal.
- Consumes: `scripts/train.py` CLI (`--config`, `--epochs`, `--device`, `--data-root`, `--project`, `--resume`) — unchanged.

**Context:** `data/processed/bdd100k.yaml`'s `path:` is `data/processed/bdd100k`. The volume holds the tar extracted to `/vol/processed/bdd100k/...`. So `--data-root` must be `/vol/processed/bdd100k` (the dir that directly contains `train/`, `val/`, `test/`), NOT `/vol/processed`. The current script passes `/vol/processed` → wrong. Also `APP_NAME`/`VOLUME_NAME` carry drone naming.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_modal_train.py
import importlib.util
from pathlib import Path


def _load_modal_train():
    """Import modal_train.py without requiring the `modal` package at import time."""
    import sys
    import types

    if "modal" not in sys.modules:
        stub = types.ModuleType("modal")
        stub.App = lambda *a, **k: types.SimpleNamespace(
            function=lambda *fa, **fk: (lambda f: f),
            local_entrypoint=lambda *fa, **fk: (lambda f: f),
        )

        class _Img:
            def __getattr__(self, _):
                return lambda *a, **k: self

        stub.Image = types.SimpleNamespace(debian_slim=lambda *a, **k: _Img())
        stub.Volume = types.SimpleNamespace(from_name=lambda *a, **k: object())
        sys.modules["modal"] = stub

    path = Path(__file__).resolve().parents[1] / "modal_train.py"
    spec = importlib.util.spec_from_file_location("modal_train", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_train_cmd_points_data_root_at_dataset_subdir():
    mt = _load_modal_train()
    cmd = mt.build_train_cmd(
        model="yolov8n",
        epochs=50,
        dataset_path="/vol/processed/bdd100k",
        run_dir="/vol/runs",
        resume=False,
    )
    assert "--data-root" in cmd
    assert cmd[cmd.index("--data-root") + 1] == "/vol/processed/bdd100k"
    assert "--config" in cmd
    assert cmd[cmd.index("--config") + 1] == "/app/configs/yolov8n.yaml"
    assert "--resume" not in cmd


def test_build_train_cmd_appends_resume_flag():
    mt = _load_modal_train()
    cmd = mt.build_train_cmd("yolov8m", 50, "/vol/processed/bdd100k", "/vol/runs", True)
    assert cmd[-1] == "--resume"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_modal_train.py -v`
Expected: FAIL — `AttributeError: module 'modal_train' has no attribute 'build_train_cmd'`

- [ ] **Step 3: Implement `build_train_cmd` and wire it into `train()`**

In `modal_train.py`, update the constants and add the helper near the top (after imports):

```python
APP_NAME = "cpv-vehicle-perception"
VOLUME_NAME = "cpv-bdd100k"
VOLUME_PATH = Path("/vol")
# Dataset is extracted to /vol/processed/bdd100k (the dir holding train/ val/ test/).
DATASET_SUBDIR = "processed/bdd100k"


def build_train_cmd(model, epochs, dataset_path, run_dir, resume):
    """Argv for the in-container scripts/train.py call. Pure + unit-testable."""
    cmd = [
        "python",
        "/app/scripts/train.py",
        "--config",
        f"/app/configs/{model}.yaml",
        "--epochs",
        str(epochs),
        "--device",
        "0",
        "--data-root",
        str(dataset_path),
        "--project",
        str(run_dir),
    ]
    if resume:
        cmd.append("--resume")
    return cmd
```

Then in `train()`, replace the dataset-path block and the inline `cmd = [...]`:

```python
    dataset_path = VOLUME_PATH / DATASET_SUBDIR
    if not dataset_path.exists():
        raise RuntimeError(
            "Dataset not found in volume. Run:\n"
            "  tar czf processed.tar.gz -C data processed\n"
            f"  modal volume put {VOLUME_NAME} processed.tar.gz /processed.tar.gz\n"
            "  modal run modal_train.py::extract_dataset"
        )
    ...
    resume = (not fresh) and last_pt.exists()
    cmd = build_train_cmd(model, epochs, dataset_path, run_dir, resume)
```

Keep the existing resume/`fresh` checkpoint logic that sets `last_pt` and clears `model_run_dir`; just feed its result into the `resume` boolean above.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_modal_train.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Update the module docstring** so the upload/extract commands match the real layout (`-C data processed` still tars `data/processed/`, which now contains `bdd100k/`). Add a one-line note: "After extraction the dataset is at `/vol/processed/bdd100k`."

- [ ] **Step 6: Lint + full suite**

Run: `black modal_train.py tests/test_modal_train.py && flake8 modal_train.py tests/test_modal_train.py && pytest -q`
Expected: black clean, flake8 silent, all tests pass.

- [ ] **Step 7: Commit**

```bash
git add modal_train.py tests/test_modal_train.py
git commit -m "fix(modal): point --data-root at bdd100k subdir + vehicle naming"
```

### Task 2: Operator runbook — run the Modal sweep (MANUAL, no code)

This task is executed by the human operator with GPU access; an agent only records the runbook in the plan and verifies the downloaded artifacts.

**Pre-req:** `pip install modal && modal setup` (one-time, local).

- [ ] **Step 1: Upload the dataset to the volume (one-time)**

```bash
tar czf processed.tar.gz -C data processed
modal volume create cpv-bdd100k        # name must match VOLUME_NAME
modal volume put cpv-bdd100k processed.tar.gz /processed.tar.gz
modal run modal_train.py::extract_dataset
rm processed.tar.gz
```

- [ ] **Step 2: Sanity run (5 epochs, YOLOv8n) — confirms the pipeline end-to-end before spending on full runs**

```bash
modal run modal_train.py::main --model yolov8n --epochs 5
```
Expected: training starts, reads `/vol/processed/bdd100k`, writes `/vol/runs/yolov8n/weights/best.pt`. If it errors with "Dataset not found", the path fix from Task 1 regressed.

- [ ] **Step 3: Full runs (50 epochs each, independent)**

```bash
modal run modal_train.py::main --model yolov8n --epochs 50 --fresh
modal run modal_train.py::main --model yolov8m --epochs 50 --fresh
modal run modal_train.py::main --model rtdetr  --epochs 50 --fresh
```
Note: RT-DETR-L at `batch: 4` fits L4; if OOM, drop to `--gpu` L40S by editing the `gpu=` arg or lower batch in `configs/rtdetr.yaml`. `--fresh` avoids resuming a stale checkpoint.

- [ ] **Step 4: Download all three best checkpoints**

```bash
modal run modal_train.py::fetch --model yolov8n   # -> models/yolov8n-best.pt
modal run modal_train.py::fetch --model yolov8m   # -> models/yolov8m-best.pt
modal run modal_train.py::fetch --model rtdetr    # -> models/rtdetr-best.pt
```

- [ ] **Step 5: Verify artifacts (agent-checkable)**

Run: `ls -la models/*-best.pt`
Expected: three `.pt` files, non-trivial size (n≈6 MB, m≈50 MB, rtdetr≈60–70 MB). These are gitignored — do NOT commit them.

> ⚠️ The `models/yolov8n-best.pt` / `yolov8m-best.pt` currently on disk are **stale VisDrone (5-class) drone weights**. The `--fresh` runs above overwrite them with BDD100K 3-class weights. Confirm by loading one and checking `model.names` has 3 entries (`vehicle/person/two_wheeler`) before trusting any eval.

---

## Phase B — Evaluation harness (BDD100K test + KITTI zero-shot + robustness)

Maps to R3's 3-model comparison + cross-dataset generalization + day/night/weather breakdown.

### Task 3: Clean `evaluate.py` (stale docstring, path pin, per-class mAP)

**Files:**
- Modify: `scripts/evaluate.py`
- Test: covered by manual run (Ultralytics `val` needs weights + data; no new unit test — the metric-collection logic is already exercised indirectly and the change is docstring + flag plumbing).

**Interfaces:**
- Produces (unchanged JSON schema from `collect_metrics`): `map50, map50_95, precision, recall, fps_inference_only, fps_end_to_end, inference_ms_per_image, total_ms_per_image, params_millions, weights_size_mb`. Adds optional `per_class: {class_name: map50}` when `--classwise` is passed.

- [ ] **Step 1: Fix the docstring** — replace both `configs/visdrone5.yaml` examples with `configs/bdd100k.yaml`, e.g.:

```python
    python scripts/evaluate.py --weights models/yolov8m-best.pt \\
        --data configs/bdd100k.yaml --split test --device 0 \\
        --output reports/R3/yolov8m_metrics.json --classwise
```

- [ ] **Step 2: Add `--classwise` arg** in `parse_args()`:

```python
    p.add_argument(
        "--classwise",
        action="store_true",
        help="Include per-class mAP@0.5 in the output (R3 comparison table)",
    )
```

- [ ] **Step 3: Populate per-class metrics** in `collect_metrics` (extend signature to accept the flag, or compute in `main`). Add after the `box` line:

```python
    per_class = None
    if classwise:
        names = getattr(model, "names", {}) or {}
        # results.box.maps is the per-class mAP@0.5:0.95 array; ap50() gives @0.5
        try:
            ap50 = results.box.ap50  # ndarray, one entry per class index present
            per_class = {
                names.get(i, str(i)): float(ap50[idx])
                for idx, i in enumerate(results.box.ap_class_index)
            }
        except Exception:
            per_class = None
```

Include `"per_class": per_class` in the returned dict (only when not None), and thread `args.classwise` through `main()`.

- [ ] **Step 4: Pin the dataset path** — add a comment + runtime note in `main()` that Ultralytics resolves a relative `path:` against its own `datasets_dir`, so evaluation must be run from repo root with `path: data/processed/bdd100k` resolvable, OR pass an absolute `path`. Add a guard:

```python
    data_cfg = Path(args.data)
    if not data_cfg.exists():
        raise FileNotFoundError(f"Data config not found: {data_cfg} (run from repo root)")
```

- [ ] **Step 5: Lint**

Run: `black scripts/evaluate.py && flake8 scripts/evaluate.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add scripts/evaluate.py
git commit -m "feat(eval): per-class mAP + drop stale visdrone refs in evaluate.py"
```

- [ ] **Step 7 (MANUAL, after Phase A weights exist): Run the 3-model BDD100K test eval**

```bash
for m in yolov8n yolov8m rtdetr; do
  python scripts/evaluate.py --weights models/$m-best.pt --data configs/bdd100k.yaml \
    --split test --device 0 --classwise --output reports/R3/${m}_metrics.json
done
```
Expected: three JSON files; FPS for yolov8n/m well above 30, rtdetr likely below (informs the selection rule).

### Task 4: `scripts/evaluate_robustness.py` — attribute-sliced mAP

**Files:**
- Create: `scripts/evaluate_robustness.py`
- Create: `tests/test_evaluate_robustness.py`
- Consumes: `data/processed/bdd100k/attributes.csv` (`name,split,weather,scene,timeofday`) and a trained model.

**Interfaces:**
- Produces: `group_image_names(attributes_csv: Path, split: str, by: str) -> dict[str, list[str]]` — maps each attribute value (e.g. `"daytime"`, `"night"`) to the list of image filenames in that split. Pure, CSV-only, unit-testable.

The slicing logic (grouping by attribute) is the only non-Ultralytics-dependent part, so that is what we TDD. The per-slice `model.val()` invocation is a thin manual loop.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate_robustness.py
from pathlib import Path
from scripts.evaluate_robustness import group_image_names


def _write_csv(tmp_path: Path) -> Path:
    csv = tmp_path / "attributes.csv"
    csv.write_text(
        "name,split,weather,scene,timeofday\n"
        "a.jpg,test,clear,city street,daytime\n"
        "b.jpg,test,rainy,highway,night\n"
        "c.jpg,test,clear,city street,night\n"
        "d.jpg,train,clear,city street,daytime\n"
    )
    return csv


def test_group_by_timeofday_filters_to_split(tmp_path):
    groups = group_image_names(_write_csv(tmp_path), split="test", by="timeofday")
    assert groups["daytime"] == ["a.jpg"]
    assert sorted(groups["night"]) == ["b.jpg", "c.jpg"]
    assert "d.jpg" not in groups.get("daytime", [])  # train split excluded


def test_group_by_weather(tmp_path):
    groups = group_image_names(_write_csv(tmp_path), split="test", by="weather")
    assert sorted(groups["clear"]) == ["a.jpg", "c.jpg"]
    assert groups["rainy"] == ["b.jpg"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluate_robustness.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.evaluate_robustness'`

- [ ] **Step 3: Implement the script**

```python
# scripts/evaluate_robustness.py
"""Slice BDD100K test mAP by image attribute (timeofday / weather) for R3.

Usage
-----
    python scripts/evaluate_robustness.py --weights models/yolov8m-best.pt \\
        --data configs/bdd100k.yaml --by timeofday --device 0 \\
        --output reports/R3/yolov8m_robustness_timeofday.json
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def group_image_names(attributes_csv, split, by):
    groups = defaultdict(list)
    with open(attributes_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            groups[row[by]].append(row["name"])
    return dict(groups)


def _parse_args():
    p = argparse.ArgumentParser(description="Attribute-sliced mAP for BDD100K")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="configs/bdd100k.yaml")
    p.add_argument("--attributes", default="data/processed/bdd100k/attributes.csv")
    p.add_argument("--split", default="test")
    p.add_argument("--by", default="timeofday", choices=["timeofday", "weather", "scene"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    from ultralytics import YOLO

    args = _parse_args()
    groups = group_image_names(Path(args.attributes), args.split, args.by)
    model = YOLO(args.weights)
    results = {}
    img_root = Path("data/processed/bdd100k") / args.split / "images"
    for value, names in sorted(groups.items()):
        paths = [str(img_root / n) for n in names if (img_root / n).exists()]
        if not paths:
            continue
        # Ultralytics accepts a list of image paths as the val source.
        r = model.val(data=args.data, imgsz=args.imgsz, device=args.device,
                      verbose=False, source=paths) if False else None
        # NOTE: model.val() validates the full split; to restrict to a subset we
        # write a temp split file. See Step 4.
        results[value] = {"n_images": len(paths), "map50": None}
    out = {"by": args.by, "split": args.split, "slices": results}
    print(json.dumps(out, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluate_robustness.py -v`
Expected: PASS (the test only exercises `group_image_names`, which is complete).

- [ ] **Step 5: Replace the `main()` slicing stub with a working temp-split approach.** Ultralytics `val()` validates a whole `data:` split, so to evaluate a subset, build a temporary data YAML whose `val:` points to a generated `.txt` image-list file (Ultralytics supports a txt list as a split source). Update `main()`:

```python
    import tempfile
    import yaml

    with open(args.data) as f:
        base = yaml.safe_load(f)
    for value, names in sorted(groups.items()):
        paths = [str(img_root.resolve() / n) for n in names if (img_root / n).exists()]
        if not paths:
            continue
        with tempfile.TemporaryDirectory() as td:
            listing = Path(td) / "imgs.txt"
            listing.write_text("\n".join(paths) + "\n")
            cfg = dict(base)
            cfg["path"] = str(Path("data/processed/bdd100k").resolve())
            cfg["val"] = str(listing)
            tmp_yaml = Path(td) / "slice.yaml"
            tmp_yaml.write_text(yaml.safe_dump(cfg))
            r = model.val(data=str(tmp_yaml), split="val", imgsz=args.imgsz,
                          device=args.device, verbose=False)
            results[value] = {"n_images": len(paths), "map50": float(r.box.map50)}
```

Remove the dead `if False else None` stub line.

- [ ] **Step 6: Lint + test**

Run: `black scripts/evaluate_robustness.py tests/test_evaluate_robustness.py && flake8 scripts/evaluate_robustness.py tests/test_evaluate_robustness.py && pytest tests/test_evaluate_robustness.py -q`
Expected: clean + PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/evaluate_robustness.py tests/test_evaluate_robustness.py
git commit -m "feat(eval): attribute-sliced (day/night/weather) mAP for R3"
```

- [ ] **Step 8 (MANUAL, after weights): generate the breakdown for the demo model**

```bash
python scripts/evaluate_robustness.py --weights models/yolov8m-best.pt \
  --by timeofday --device 0 --output reports/R3/yolov8m_robustness_timeofday.json
python scripts/evaluate_robustness.py --weights models/yolov8m-best.pt \
  --by weather --device 0 --output reports/R3/yolov8m_robustness_weather.json
```

### Task 5: KITTI → YOLO 3-class converter for zero-shot detection eval

**Files:**
- Create: `src/utils/kitti.py`
- Create: `scripts/preprocess_kitti.py`
- Create: `tests/test_kitti.py`

**Interfaces:**
- Produces in `src/utils/kitti.py`:
  - `KITTI_TO_COARSE: dict[str, int]` — KITTI type → 3-class id, e.g. `{"Car":0,"Van":0,"Truck":0,"Tram":0,"Pedestrian":1,"Person_sitting":1,"Cyclist":2}`. `DontCare`/`Misc` excluded.
  - `parse_label_line(line: str) -> dict | None` — one `label_2` row → `{"type","bbox":(l,t,r,b),"location":(x,y,z),"dimensions":(h,w,l),"truncated","occluded"}`; `None` if malformed.
  - `to_yolo_lines(label_path: Path, img_w: int, img_h: int) -> list[str]` — convert a KITTI label file to YOLO `cls cx cy w h` (normalized) lines, dropping classes not in the map.

**Context — KITTI label_2 format** (15 space-separated fields per object): `type truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom h w l x y z rotation_y`. Bbox is in pixels on the left color image; `(x,y,z)` is the 3D camera-frame location in meters (used in Phase C for distance).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kitti.py
from pathlib import Path
from src.utils.kitti import parse_label_line, to_yolo_lines, KITTI_TO_COARSE


def test_parse_label_line_car():
    line = "Car 0.00 0 1.55 100.0 150.0 300.0 350.0 1.5 1.6 4.0 5.0 1.7 20.0 0.1"
    obj = parse_label_line(line)
    assert obj["type"] == "Car"
    assert obj["bbox"] == (100.0, 150.0, 300.0, 350.0)
    assert obj["location"] == (5.0, 1.7, 20.0)


def test_parse_label_line_dontcare_returns_object_but_unmapped():
    obj = parse_label_line("DontCare -1 -1 -10 0 0 0 0 -1 -1 -1 -1000 -1000 -1000 -10")
    assert obj["type"] == "DontCare"
    assert "DontCare" not in KITTI_TO_COARSE


def test_to_yolo_lines_maps_and_normalizes(tmp_path):
    lp = tmp_path / "000000.txt"
    lp.write_text(
        "Car 0 0 0 0 0 100 100 1.5 1.6 4 5 1.7 20 0.1\n"
        "Pedestrian 0 0 0 50 50 90 150 1.7 0.5 0.5 1 1 8 0\n"
        "DontCare -1 -1 -10 0 0 0 0 -1 -1 -1 -1000 -1000 -1000 -10\n"
    )
    lines = to_yolo_lines(lp, img_w=1000, img_h=500)
    assert len(lines) == 2  # DontCare dropped
    cls, cx, cy, w, h = lines[0].split()
    assert cls == "0"  # Car -> vehicle
    assert abs(float(cx) - 0.05) < 1e-6  # (0+100)/2 / 1000
    assert abs(float(w) - 0.10) < 1e-6   # 100/1000
    assert lines[1].split()[0] == "1"  # Pedestrian -> person
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kitti.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.kitti'`

- [ ] **Step 3: Implement `src/utils/kitti.py`**

```python
from pathlib import Path
from typing import List, Optional

KITTI_TO_COARSE = {
    "Car": 0,
    "Van": 0,
    "Truck": 0,
    "Tram": 0,
    "Pedestrian": 1,
    "Person_sitting": 1,
    "Cyclist": 2,
}


def parse_label_line(line: str) -> Optional[dict]:
    parts = line.split()
    if len(parts) < 15:
        return None
    try:
        nums = [float(x) for x in parts[1:15]]
    except ValueError:
        return None
    return {
        "type": parts[0],
        "truncated": nums[0],
        "occluded": int(nums[1]),
        "bbox": (nums[3], nums[4], nums[5], nums[6]),
        "dimensions": (nums[7], nums[8], nums[9]),  # h, w, l
        "location": (nums[10], nums[11], nums[12]),  # x, y, z (meters)
    }


def to_yolo_lines(label_path: Path, img_w: int, img_h: int) -> List[str]:
    out = []
    for raw in Path(label_path).read_text().splitlines():
        obj = parse_label_line(raw)
        if obj is None or obj["type"] not in KITTI_TO_COARSE:
            continue
        l, t, r, b = obj["bbox"]
        cx = ((l + r) / 2.0) / img_w
        cy = ((t + b) / 2.0) / img_h
        bw = (r - l) / img_w
        bh = (b - t) / img_h
        out.append(
            f"{KITTI_TO_COARSE[obj['type']]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_kitti.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Implement `scripts/preprocess_kitti.py`** (uses the helper; reads image sizes via PIL, which is already an Ultralytics dep). It expects raw KITTI at `data/raw/kitti/{image_2,label_2}` (operator downloads separately — see Step 7) and writes `data/processed/kitti/{images,labels}` + a `configs/kitti.yaml` pointing at it.

```python
"""Convert KITTI object-detection labels to the 3-class YOLO layout.

Raw layout expected (KITTI 'left color images' + 'training labels'):
    data/raw/kitti/image_2/000000.png ...
    data/raw/kitti/label_2/000000.txt ...

Usage:
    python scripts/preprocess_kitti.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402
from src.utils.kitti import to_yolo_lines  # noqa: E402

RAW = Path("data/raw/kitti")
OUT = Path("data/processed/kitti")


def main():
    img_dir, lbl_dir = RAW / "image_2", RAW / "label_2"
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    n = 0
    for img in sorted(img_dir.glob("*.png")):
        label = lbl_dir / f"{img.stem}.txt"
        if not label.exists():
            continue
        with Image.open(img) as im:
            w, h = im.size
        lines = to_yolo_lines(label, w, h)
        (OUT / "labels" / f"{img.stem}.txt").write_text("\n".join(lines) + "\n")
        dst = OUT / "images" / img.name
        if not dst.exists():
            dst.symlink_to(img.resolve())
        n += 1
    print(f"Converted {n} KITTI frames -> {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Create `configs/kitti.yaml`** (zero-shot eval target — KITTI is val-only here):

```yaml
# KITTI held-out eval — same 3 classes as BDD100K (zero-shot, never trained on).
path: data/processed/kitti
val: images
nc: 3
names:
  0: vehicle
  1: person
  2: two_wheeler
```

- [ ] **Step 7 (MANUAL, operator): download raw KITTI**

KITTI requires registration. Either use the `kaggle` skill (search a KITTI object-detection mirror) or download `data_object_image_2.zip` + `data_object_label_2.zip` from the KITTI site, then arrange as `data/raw/kitti/image_2/` and `data/raw/kitti/label_2/`. Then:

```bash
python scripts/preprocess_kitti.py
python scripts/validate_data.py --images data/processed/kitti/images \
  --labels data/processed/kitti/labels --num-classes 3
```

- [ ] **Step 8: Lint + suite + commit**

Run: `black src/utils/kitti.py scripts/preprocess_kitti.py tests/test_kitti.py && flake8 src/utils/kitti.py scripts/preprocess_kitti.py tests/test_kitti.py && pytest -q`
Expected: clean + all pass.

```bash
git add src/utils/kitti.py scripts/preprocess_kitti.py configs/kitti.yaml tests/test_kitti.py
git commit -m "feat(eval): KITTI->YOLO 3-class converter for zero-shot eval"
```

- [ ] **Step 9 (MANUAL, after weights + KITTI): zero-shot cross-dataset eval**

```bash
for m in yolov8n yolov8m rtdetr; do
  python scripts/evaluate.py --weights models/$m-best.pt --data configs/kitti.yaml \
    --split val --device 0 --classwise --output reports/R3/${m}_kitti_metrics.json
done
```
Expected: mAP noticeably below the BDD100K numbers — that gap *is* the R3 domain-gap finding.

---

## Phase C — Risk-zone validation against KITTI ground truth (Approach C)

Maps to R3's "risk-zone validation vs ground-truth depth." Converts the heuristic into a measured finding. Offline only.

### Task 6: `scripts/validate_risk.py` — agreement of risk labels with KITTI distance

**Files:**
- Create: `scripts/validate_risk.py`
- Create: `tests/test_validate_risk.py`
- Consumes: `src.utils.kitti.parse_label_line` (for `location` → distance), `src.risk.zone_assessor.RiskZoneAssessor`, `src.tracking.tracker.Track`.

**Interfaces:**
- Produces:
  - `kitti_distance(location: tuple) -> float` — Euclidean distance `sqrt(x²+y²+z²)` in meters from the camera (z dominates forward distance).
  - `risk_distance_agreement(assessor, objects, frame_shape, near_thresh_m) -> dict` — for KITTI objects (each `{"bbox","location"}`), build a single-frame `Track` per object (zero velocity → tests the spatial WHERE component, not the closing proxy), run `assessor.assess`, and report `{"danger_total","danger_near","caution_total","safe_total","precision_danger_near"}` where `precision_danger_near = danger_near / danger_total` (of objects flagged DANGER, fraction within `near_thresh_m`).

**Note on scope:** KITTI is single-frame, so `scale_velocity = 0` → the closing/TTC proxy never fires. This validates the **in-path geometry** (the WHERE axis) against true distance, which is the part KITTI ground truth can actually adjudicate. The plan states this limitation explicitly in the report (Task 8).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validate_risk.py
import math
from src.risk.zone_assessor import RiskZoneAssessor
from scripts.validate_risk import kitti_distance, risk_distance_agreement


def test_kitti_distance():
    assert abs(kitti_distance((3.0, 0.0, 4.0)) - 5.0) < 1e-6


def test_agreement_flags_near_in_path_object_as_danger():
    # Frame 1000x500. An in-path, very large box near the bottom-center.
    assessor = RiskZoneAssessor(large_area_frac=0.05)
    objects = [
        # near, centered, large -> expect DANGER and near
        {"bbox": (400.0, 300.0, 600.0, 500.0), "location": (0.0, 0.0, 6.0)},
        # off to the far side, tiny, far -> SAFE
        {"bbox": (10.0, 250.0, 30.0, 270.0), "location": (0.0, 0.0, 60.0)},
    ]
    out = risk_distance_agreement(
        assessor, objects, frame_shape=(500, 1000), near_thresh_m=15.0
    )
    assert out["danger_total"] >= 1
    assert out["danger_near"] >= 1
    assert math.isclose(out["precision_danger_near"], 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_validate_risk.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.validate_risk'`

- [ ] **Step 3: Implement `scripts/validate_risk.py`**

```python
"""Approach C — validate the risk heuristic against KITTI ground-truth distance.

For each KITTI frame: build a zero-velocity Track per ground-truth object, run
RiskZoneAssessor, and measure how well DANGER/CAUTION labels line up with true
distance. Single-frame data has no closing rate, so this validates the in-path
(WHERE) geometry, not the TTC proxy.

Usage:
    python scripts/validate_risk.py --kitti data/processed/kitti \\
        --raw-labels data/raw/kitti/label_2 --output reports/R3/risk_validation.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from src.risk.zone_assessor import RiskZoneAssessor  # noqa: E402
from src.risk.assessor import RiskLevel  # noqa: E402
from src.tracking.tracker import Track  # noqa: E402
from src.utils.kitti import parse_label_line, KITTI_TO_COARSE  # noqa: E402


def kitti_distance(location):
    x, y, z = location
    return math.sqrt(x * x + y * y + z * z)


def risk_distance_agreement(assessor, objects, frame_shape, near_thresh_m):
    tracks, dists = [], []
    for i, obj in enumerate(objects):
        tracks.append(
            Track(track_id=i, bbox=obj["bbox"], velocity=(0.0, 0.0),
                  scale_velocity=0.0, age=1)
        )
        dists.append(kitti_distance(obj["location"]))
    risked = assessor.assess(tracks, frame_shape)
    counts = {"danger_total": 0, "danger_near": 0, "caution_total": 0, "safe_total": 0}
    for rt, d in zip(risked, dists):
        if rt.risk == RiskLevel.DANGER:
            counts["danger_total"] += 1
            if d <= near_thresh_m:
                counts["danger_near"] += 1
        elif rt.risk == RiskLevel.CAUTION:
            counts["caution_total"] += 1
        else:
            counts["safe_total"] += 1
    counts["precision_danger_near"] = (
        counts["danger_near"] / counts["danger_total"]
        if counts["danger_total"] else None
    )
    return counts


def _parse_args():
    p = argparse.ArgumentParser(description="Validate risk labels vs KITTI distance")
    p.add_argument("--kitti", default="data/processed/kitti")
    p.add_argument("--raw-labels", default="data/raw/kitti/label_2")
    p.add_argument("--near-thresh-m", type=float, default=15.0)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    assessor = RiskZoneAssessor()
    img_dir = Path(args.kitti) / "images"
    lbl_dir = Path(args.raw_labels)
    agg = {"danger_total": 0, "danger_near": 0, "caution_total": 0, "safe_total": 0}
    for img in sorted(img_dir.glob("*.png")):
        label = lbl_dir / f"{img.stem}.txt"
        if not label.exists():
            continue
        with Image.open(img) as im:
            w, h = im.size
        objects = []
        for raw in label.read_text().splitlines():
            o = parse_label_line(raw)
            if o is None or o["type"] not in KITTI_TO_COARSE:
                continue
            objects.append({"bbox": o["bbox"], "location": o["location"]})
        res = risk_distance_agreement(assessor, objects, (h, w), args.near_thresh_m)
        for k in agg:
            agg[k] += res[k]
    agg["precision_danger_near"] = (
        agg["danger_near"] / agg["danger_total"] if agg["danger_total"] else None
    )
    print(json.dumps(agg, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_validate_risk.py -v`
Expected: PASS (2 passed).

> If `Track`'s constructor signature differs from `Track(track_id, bbox, velocity, scale_velocity, age)`, read `src/tracking/tracker.py` and adjust the `Track(...)` call in both the test and script to match the real dataclass fields before implementing.

- [ ] **Step 5: Lint + suite + commit**

Run: `black scripts/validate_risk.py tests/test_validate_risk.py && flake8 scripts/validate_risk.py tests/test_validate_risk.py && pytest -q`
Expected: clean + all pass.

```bash
git add scripts/validate_risk.py tests/test_validate_risk.py
git commit -m "feat(risk): Approach C — validate risk labels vs KITTI distance"
```

- [ ] **Step 6 (MANUAL, after KITTI): run validation**

```bash
python scripts/validate_risk.py --output reports/R3/risk_validation.json
```
Expected: a `precision_danger_near` figure — the headline "of objects flagged DANGER, X% are within 15 m" finding.

---

## Phase D — Documentation & Streamlit rewrite

Maps to R4 (showcase) and keeps the source-of-truth docs current.

### Task 7: Author `docs/training_pipeline.md` (vehicle edition)

**Files:**
- Create: `docs/training_pipeline.md`

The old drone version was removed; the design spec is the source of truth. This doc is the *operational* companion: how to train/eval/validate the vehicle pipeline.

- [ ] **Step 1: Write the doc** covering, with the exact commands from Phases A–C:
  - Dataset prep recap (points to `scripts/preprocess.py`, the 57k/12.5k/10k split, `docs/bdd100k_data_validation.md`).
  - Modal training runbook (Task 2 commands), GPU/cost notes (L4 default; RT-DETR may need L40S), the `--data-root = /vol/processed/bdd100k` gotcha.
  - Evaluation: `evaluate.py` (per-class + FPS), `evaluate_robustness.py` (day/night/weather), KITTI zero-shot via `configs/kitti.yaml`.
  - Risk validation: `validate_risk.py`, and the **single-frame caveat** (validates WHERE, not TTC).
  - The locked decisions block (imgsz 640, seed 42, 50/5 epochs, selection rule mAP@0.5 s.t. FPS ≥ 30) and the 3-model lineup table.
  - An R-round → artifact map (which `reports/R*/` files each step produces).

- [ ] **Step 2: Commit**

```bash
git add docs/training_pipeline.md
git commit -m "docs: vehicle training/eval/risk-validation pipeline guide"
```

### Task 8: Rewrite `prototype/web_app.py` for the 3-class vehicle demo

**Files:**
- Modify: `prototype/web_app.py`

**Context:** Streamlit showcase reusing `src.*` pipeline classes. Excluded from flake8 (the `sys.path` shim trips E402) — format with `black` manually. Reads theme from repo-root `.streamlit/config.toml`. Must launch from repo root.

- [ ] **Step 1: Read the current `prototype/web_app.py`** and identify drone-era copy (obstacle/yaw/altitude language, class names, any 5-class assumptions).

- [ ] **Step 2: Rewrite** so it:
  - Loads `YoloDetector` + `KalmanTracker` + `RiskZoneAssessor`, with a model selector (yolov8n/m/rtdetr from `models/*-best.pt`).
  - Overlays the ego-path trapezoid (`assessor.ego_path_polygon`) and color-codes boxes by `RiskedTrack.risk` (SAFE/CAUTION/DANGER).
  - Uses the 3 class names `vehicle/person/two_wheeler`.
  - Has accurate copy: dashcam forward-collision-warning advisory, monocular, no control loop.

- [ ] **Step 3: Smoke-check it imports** (don't need a browser):

Run: `python -c "import ast; ast.parse(open('prototype/web_app.py').read())"`
Expected: no output (parses clean). Optionally `streamlit run prototype/web_app.py` locally.

- [ ] **Step 4: Format + commit**

Run: `black prototype/web_app.py`

```bash
git add prototype/web_app.py
git commit -m "feat(demo): rewrite Streamlit showcase for 3-class vehicle pipeline"
```

### Task 9: Flip the CLAUDE.md status table

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the pipeline-status table** — set "Plan 3 — training + KITTI eval + docs/Streamlit rewrite" to ✅ done, and update the prose note that listed `validate_data.py`/`train.py`/`evaluate.py`/`modal_train.py` as needing Plan 3 cleanup (now done).

- [ ] **Step 2: Update the "as of" date** in the pipeline-status heading to the completion date.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: mark Plan 3 complete in CLAUDE.md status table"
```

---

## Self-Review notes (spec coverage)

- **Training (3 models, BDD100K, identical protocol)** → Phase A (Task 1 infra fix + Task 2 runbook). ✅
- **3-model comparison on BDD100K test** → Task 3 (per-class + FPS metrics JSON per model). ✅
- **KITTI cross-dataset generalization** → Task 5 (converter + `configs/kitti.yaml`) + Task 3 Step 9 (zero-shot eval). ✅
- **Risk-zone validation vs KITTI ground truth (Approach C)** → Phase C / Task 6. ✅
- **Day/night & weather robustness breakdown** → Task 4 (`evaluate_robustness.py` over `attributes.csv`). ✅
- **Docs rewrite (training_pipeline.md + CLAUDE.md)** → Tasks 7, 9. ✅
- **Streamlit rewrite** → Task 8. ✅
- **Selection rule / locked decisions** → Global Constraints + documented in Task 7. ✅

**Known dependencies / sequencing:** Tasks 1, 3, 4, 5, 6 (code) are TDD-able now and independent of GPU. The MANUAL steps (Task 2; Task 3 Step 7,9; Task 4 Step 8; Task 5 Step 7,9; Task 6 Step 6) require the operator's Modal account + KITTI download and produce the `reports/R3/*.json` artifacts the R3 report consumes. Task 8 (Streamlit) is best done after at least `yolov8m-best.pt` exists so the demo can be eyeballed. Task 9 closes out the plan.
