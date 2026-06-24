# BDD100K Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn raw BDD100K detection data into a YOLO-format 3-class dataset (`vehicle` / `person` / `two_wheeler`) with stratified train/val/test splits, a day-night/weather attributes manifest, and a passing pre-training validation report — the R1 data deliverable and the input Plan 3 trains on.

**Architecture:** BDD100K ships detection labels as JSON (one entry per 1280×720 image, pixel `box2d` boxes + per-image `attributes`). A pure conversion layer in `src/utils/bdd100k.py` collapses the native 10 categories to 3 coarse obstacle classes and emits normalized YOLO lines; thin CLIs in `scripts/` download (Kaggle) and orchestrate (convert → split → materialize). The official **train** set is split (stratified by dominant class, seed 42) into our train/val; the official **val** set becomes the held-out **test** (its labels are public; the official test labels are withheld). Images are **symlinked** into the processed tree to avoid duplicating ~5 GB on a disk with ~22 GB free. The existing `src/utils/data_validation.py` validates the result unchanged at `num_classes=3`.

**Tech Stack:** Python, NumPy, scikit-learn (`train_test_split`), OpenCV (validation only), Kaggle CLI, Ultralytics dataset YAML.

---

## Scope & locked decisions

- **Source:** Kaggle mirror `solesensei/solesensei_bdd100k` (canonical; ships `images/100k/{train,val,test}/` + `labels/bdd100k_labels_images_{train,val}.json` with weather/scene/timeofday attributes). Downloaded via `scripts/download_bdd100k.py`. The machine is **already authenticated** (`kaggle config view` → user `itsdreowo`, `auth_method: ACCESS_TOKEN`); there is **no `kaggle.json` file**, so the download script must not pre-flight-check for one.
- **Split:** official train → our train/val (stratified, `--val-ratio 0.18`, seed 42); official val → our test. No model selection ever touches the test split.
- **Classes (index order = training order):** `0 vehicle` ← car/truck/bus/train; `1 person` ← pedestrian(/person)/rider; `2 two_wheeler` ← bicycle(/bike)/motorcycle(/motor). Dropped: traffic light, traffic sign, lane, drivable area. The converter accepts **both** the 2018 (`person`/`bike`/`motor`) and 2020 (`pedestrian`/`bicycle`/`motorcycle`) category spellings.
- **Attributes:** captured during conversion into `data/processed/bdd100k/attributes.csv` (`name,split,weather,scene,timeofday`) for the R3 robustness slice.
- **Images symlinked** by default (`--copy` flag falls back to copying).
- **KITTI prep deferred to Plan 3.** KITTI is only used offline in R3 (cross-dataset generalization + risk-zone validation), is ~12 GB (tight against 22 GB free), and needs the `validate_risk.py` machinery that lives in Plan 3. This is a deliberate narrowing of CLAUDE.md's "KITTI prep" line; raise it now if you disagree.
- **Branch:** continue on `pivot/vehicle-avoidance` (already checked out, clean tree). No new worktree.

## File structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/utils/bdd100k.py` | **Create** | Pure conversion (category map, box2d→YOLO, frame convert, JSON load) **and** dataset-prep helpers (collect/split/materialize/attributes). Mirrors the `data_validation.py` pattern: logic in `src/utils`, thin CLI in `scripts/`. |
| `tests/test_bdd100k.py` | **Create** | Unit tests for conversion + prep helpers. |
| `scripts/preprocess.py` | **Rewrite** | Thin CLI: convert → split → materialize → write attributes. Replaces the VisDrone version. |
| `scripts/download_bdd100k.py` | **Create** | Thin Kaggle-CLI wrapper (credential check + `kaggle datasets download --unzip`). |
| `configs/bdd100k.yaml` | **Create** | Ultralytics dataset config (`nc=3`, names). Already referenced by `configs/yolov8{n,m}.yaml` + `rtdetr.yaml`. |
| `requirements.txt` | **Modify** | Add `kaggle`. |
| `docs/bdd100k_data_validation.md` | **Create** | R1 data-validation report (validator output + split sizes + class distribution). |
| `CLAUDE.md` | **Modify** | Flip Plan 2 → done in the status table; drop the "preprocess still VisDrone-shaped" note. |

`src/utils/data_validation.py` and `scripts/validate_data.py` are **reused unchanged** (already generic over `--num-classes`).

---

### Task 1: Dependency + dataset config

**Files:**
- Modify: `requirements.txt`
- Create: `configs/bdd100k.yaml`

- [ ] **Step 1: Add the Kaggle client to requirements**

In `requirements.txt`, add a line (keep the file's existing ordering/style):

```
kaggle>=1.6.0
```

- [ ] **Step 2: Install it**

Run: `pip install kaggle`
Expected: `Successfully installed kaggle-...` (CLI already on PATH at `~/.local/bin/kaggle`).

- [ ] **Step 3: Create the Ultralytics dataset config**

Create `configs/bdd100k.yaml`:

```yaml
# BDD100K detection dataset — 3 coarse obstacle classes.
# Produced by scripts/preprocess.py. Train from the repo root.
# NOTE: Ultralytics resolves a relative `path` against its own datasets_dir
# (~/.config/Ultralytics/settings.json), not the cwd. Plan 3 pins this at
# train time; for now the layout below is the contract preprocess.py writes.
path: data/processed/bdd100k
train: train/images
val: val/images
test: test/images
nc: 3
names:
  0: vehicle
  1: person
  2: two_wheeler
```

- [ ] **Step 4: Confirm the model configs already point here**

Run: `grep -l "configs/bdd100k.yaml" configs/*.yaml`
Expected: `configs/yolov8n.yaml`, `configs/yolov8m.yaml`, `configs/rtdetr.yaml` (all three). If any does not reference it, set its `data:` field to `configs/bdd100k.yaml`.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt configs/bdd100k.yaml
git commit -m "feat(data): add kaggle dep + BDD100K 3-class dataset config"
```

---

### Task 2: BDD100K conversion core (pure functions, TDD)

**Files:**
- Create: `src/utils/bdd100k.py`
- Test: `tests/test_bdd100k.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bdd100k.py`:

```python
from src.utils.bdd100k import (
    CLASS_NAMES,
    FrameLabels,
    box2d_to_yolo,
    category_to_class_id,
    convert_frame,
)


def test_class_names_order():
    assert CLASS_NAMES == ["vehicle", "person", "two_wheeler"]


def test_category_map_vehicle_case_insensitive():
    assert category_to_class_id("car") == 0
    assert category_to_class_id("Truck") == 0
    assert category_to_class_id("bus") == 0
    assert category_to_class_id("train") == 0


def test_category_map_person_both_spellings():
    assert category_to_class_id("pedestrian") == 1
    assert category_to_class_id("person") == 1
    assert category_to_class_id("rider") == 1


def test_category_map_two_wheeler_both_spellings():
    assert category_to_class_id("bicycle") == 2
    assert category_to_class_id("bike") == 2
    assert category_to_class_id("motorcycle") == 2
    assert category_to_class_id("motor") == 2


def test_category_map_dropped_and_unknown():
    assert category_to_class_id("traffic light") is None
    assert category_to_class_id("traffic sign") is None
    assert category_to_class_id("lane") is None
    assert category_to_class_id("banana") is None


def test_box2d_to_yolo_full_frame():
    box = {"x1": 0, "y1": 0, "x2": 1280, "y2": 720}
    assert box2d_to_yolo(box, 1280, 720) == (0.5, 0.5, 1.0, 1.0)


def test_box2d_to_yolo_quadrant():
    box = {"x1": 0, "y1": 0, "x2": 640, "y2": 360}
    assert box2d_to_yolo(box, 1280, 720) == (0.25, 0.25, 0.5, 0.5)


def test_box2d_to_yolo_clamps_overflow():
    box = {"x1": -10, "y1": -10, "x2": 1300, "y2": 740}
    assert box2d_to_yolo(box, 1280, 720) == (0.5, 0.5, 1.0, 1.0)


def test_box2d_to_yolo_degenerate_returns_none():
    assert box2d_to_yolo({"x1": 5, "y1": 5, "x2": 5, "y2": 9}, 1280, 720) is None


def test_convert_frame_filters_drops_and_keeps_attrs():
    entry = {
        "name": "x.jpg",
        "attributes": {
            "weather": "clear",
            "scene": "city street",
            "timeofday": "daytime",
        },
        "labels": [
            {"category": "car", "box2d": {"x1": 0, "y1": 0, "x2": 640, "y2": 360}},
            {"category": "traffic sign", "box2d": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
            {"category": "lane", "poly2d": [[0, 0]]},  # no box2d
        ],
    }
    fl = convert_frame(entry)
    assert isinstance(fl, FrameLabels)
    assert fl.name == "x.jpg"
    assert fl.attributes["timeofday"] == "daytime"
    assert len(fl.yolo_lines) == 1
    assert fl.yolo_lines[0] == "0 0.250000 0.250000 0.500000 0.500000"


def test_convert_frame_empty_labels():
    fl = convert_frame({"name": "y.jpg", "labels": []})
    assert fl.yolo_lines == []
    assert fl.attributes == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_bdd100k.py -q`
Expected: collection/import error — `No module named 'src.utils.bdd100k'`.

- [ ] **Step 3: Implement the conversion core**

Create `src/utils/bdd100k.py`:

```python
"""BDD100K detection-label conversion to YOLO format with a 10->3 class remap.

BDD100K ships detection labels as JSON (one entry per image, pixel ``box2d``
boxes and per-image ``attributes``). We collapse the native categories to three
coarse obstacle classes and emit normalized YOLO lines.

Coarse classes (index order is the training class order)::

    0 vehicle      <- car, truck, bus, train
    1 person       <- pedestrian / person, rider
    2 two_wheeler  <- bicycle / bike, motorcycle / motor

Dropped (not collision obstacles): traffic light, traffic sign, lane,
drivable area. Handles both the 2018 ("person"/"bike"/"motor") and 2020
("pedestrian"/"bicycle"/"motorcycle") category spellings.

This module holds the pure conversion helpers; dataset-prep IO (collect,
split, materialize) is added in the same file by the next task. Mirrors the
``data_validation.py`` pattern: logic here, thin CLI in ``scripts/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CLASS_NAMES: List[str] = ["vehicle", "person", "two_wheeler"]

CATEGORY_MAP: Dict[str, int] = {
    # vehicle
    "car": 0,
    "truck": 0,
    "bus": 0,
    "train": 0,
    # person
    "pedestrian": 1,
    "person": 1,
    "rider": 1,
    # two_wheeler
    "bicycle": 2,
    "bike": 2,
    "motorcycle": 2,
    "motor": 2,
}

# All BDD100K frames are 1280x720.
BDD_IMG_W = 1280
BDD_IMG_H = 720


@dataclass
class FrameLabels:
    name: str
    yolo_lines: List[str] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)


def category_to_class_id(category: str) -> Optional[int]:
    """Map a BDD100K category to a coarse class id, or None to drop it."""
    return CATEGORY_MAP.get(category.strip().lower())


def box2d_to_yolo(
    box: Dict[str, float], img_w: int, img_h: int
) -> Optional[Tuple[float, float, float, float]]:
    """Convert a pixel ``box2d`` ({x1,y1,x2,y2}) to clamped normalized YOLO
    ``(x_center, y_center, w, h)``. Returns ``None`` for a degenerate box."""
    x1 = max(0.0, min(float(box["x1"]), img_w))
    y1 = max(0.0, min(float(box["y1"]), img_h))
    x2 = max(0.0, min(float(box["x2"]), img_w))
    y2 = max(0.0, min(float(box["y2"]), img_h))
    if x2 <= x1 or y2 <= y1:
        return None
    xc = (x1 + x2) / 2.0 / img_w
    yc = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return xc, yc, w, h


def convert_frame(
    entry: dict, img_w: int = BDD_IMG_W, img_h: int = BDD_IMG_H
) -> FrameLabels:
    """Convert one BDD100K JSON entry to ``FrameLabels``, dropping non-obstacle
    and degenerate boxes and preserving the frame's attributes."""
    name = entry["name"]
    attributes = dict(entry.get("attributes") or {})
    lines: List[str] = []
    for label in entry.get("labels") or []:
        box = label.get("box2d")
        if not box:
            continue  # lane / drivable-area entries carry poly2d, not box2d
        cls = category_to_class_id(label.get("category", ""))
        if cls is None:
            continue
        yolo = box2d_to_yolo(box, img_w, img_h)
        if yolo is None:
            continue
        xc, yc, w, h = yolo
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return FrameLabels(name=name, yolo_lines=lines, attributes=attributes)


def load_bdd_json(path: Path) -> List[dict]:
    """Load a BDD100K detection-label JSON (a top-level list of frame entries)."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(
            f"expected a top-level JSON list in {path}, got {type(data).__name__}"
        )
    return data
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_bdd100k.py -q`
Expected: all PASS.

- [ ] **Step 5: Lint**

Run: `black src/utils/bdd100k.py tests/test_bdd100k.py && flake8 src/utils/bdd100k.py tests/test_bdd100k.py`
Expected: no changes needed / no errors.

- [ ] **Step 6: Commit**

```bash
git add src/utils/bdd100k.py tests/test_bdd100k.py
git commit -m "feat(data): BDD100K JSON->YOLO conversion core (10->3 remap)"
```

---

### Task 3: Dataset-prep helpers — collect, split, materialize (TDD)

Adds IO/orchestration helpers to `src/utils/bdd100k.py` so `scripts/preprocess.py` stays a thin CLI and the split/materialize logic is unit-tested.

**Files:**
- Modify: `src/utils/bdd100k.py`
- Test: `tests/test_bdd100k.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_bdd100k.py`:

```python
import json as _json

import cv2
import numpy as np

from src.utils.bdd100k import (
    collect_frames,
    dominant_class,
    stratified_train_val,
    write_attributes,
    write_split,
)


def _write_img(path, w=16, h=16):
    cv2.imwrite(str(path), np.zeros((h, w, 3), dtype=np.uint8))


def _raw_tree(tmp_path):
    imgs = tmp_path / "images"
    imgs.mkdir()
    _write_img(imgs / "a.jpg")
    _write_img(imgs / "b.jpg")
    labels = [
        {
            "name": "a.jpg",
            "attributes": {
                "weather": "clear",
                "scene": "city street",
                "timeofday": "daytime",
            },
            "labels": [
                {"category": "car", "box2d": {"x1": 0, "y1": 0, "x2": 640, "y2": 360}}
            ],
        },
        {
            "name": "b.jpg",
            "attributes": {
                "weather": "rainy",
                "scene": "highway",
                "timeofday": "night",
            },
            "labels": [
                {
                    "category": "pedestrian",
                    "box2d": {"x1": 10, "y1": 10, "x2": 50, "y2": 200},
                }
            ],
        },
    ]
    js = tmp_path / "det.json"
    js.write_text(_json.dumps(labels))
    return imgs, js


def test_collect_frames_pairs_existing_images(tmp_path):
    imgs, js = _raw_tree(tmp_path)
    pairs = collect_frames(imgs, js)
    assert {p.name for p, _ in pairs} == {"a.jpg", "b.jpg"}


def test_collect_frames_skips_missing_image(tmp_path):
    imgs, js = _raw_tree(tmp_path)
    (imgs / "b.jpg").unlink()
    pairs = collect_frames(imgs, js)
    assert {p.name for p, _ in pairs} == {"a.jpg"}


def test_dominant_class():
    fl = FrameLabels(
        name="x",
        yolo_lines=[
            "0 0.5 0.5 0.2 0.2",
            "0 0.1 0.1 0.1 0.1",
            "1 0.5 0.5 0.1 0.1",
        ],
    )
    assert dominant_class(fl) == 0
    assert dominant_class(FrameLabels(name="y")) == -1


def test_write_split_materializes_images_and_labels(tmp_path):
    imgs, js = _raw_tree(tmp_path)
    pairs = collect_frames(imgs, js)
    out = tmp_path / "processed"
    write_split("train", pairs, out, copy=True)
    assert (out / "train" / "images" / "a.jpg").exists()
    label = (out / "train" / "labels" / "a.txt").read_text().strip()
    assert label == "0 0.250000 0.250000 0.500000 0.500000"


def test_write_split_symlinks_by_default(tmp_path):
    imgs, js = _raw_tree(tmp_path)
    pairs = collect_frames(imgs, js)
    out = tmp_path / "processed"
    write_split("val", pairs, out, copy=False)
    assert (out / "val" / "images" / "a.jpg").is_symlink()


def test_write_attributes_csv(tmp_path):
    rows = [
        (
            "a.jpg",
            "train",
            {"weather": "clear", "scene": "city street", "timeofday": "daytime"},
        )
    ]
    out = tmp_path / "attributes.csv"
    write_attributes(rows, out)
    text = out.read_text()
    assert text.splitlines()[0] == "name,split,weather,scene,timeofday"
    assert "a.jpg,train,clear,city street,daytime" in text


def test_stratified_train_val_is_proportional_and_seeded():
    pairs = []
    for i in range(5):
        pairs.append(
            (Path(f"c{i}.jpg"), FrameLabels(f"c{i}.jpg", ["0 0.5 0.5 0.1 0.1"]))
        )
    for i in range(5):
        pairs.append(
            (Path(f"p{i}.jpg"), FrameLabels(f"p{i}.jpg", ["1 0.5 0.5 0.1 0.1"]))
        )
    train, val = stratified_train_val(pairs, val_ratio=0.4, seed=42)
    assert len(train) == 6 and len(val) == 4
    # deterministic under a fixed seed
    train2, val2 = stratified_train_val(pairs, val_ratio=0.4, seed=42)
    assert [p.name for p, _ in val] == [p.name for p, _ in val2]
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `pytest tests/test_bdd100k.py -q`
Expected: import error for `collect_frames` (and the other new names).

- [ ] **Step 3: Implement the prep helpers**

Append to `src/utils/bdd100k.py`:

```python
import csv
import os
import shutil
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split

Pair = Tuple[Path, FrameLabels]


def collect_frames(images_dir: Path, labels_json: Path) -> List[Pair]:
    """Return ``(image_path, FrameLabels)`` for every JSON frame whose image
    exists under ``images_dir``. Frames with no matching image are skipped."""
    pairs: List[Pair] = []
    missing = 0
    for entry in load_bdd_json(labels_json):
        fl = convert_frame(entry)
        img_path = images_dir / fl.name
        if not img_path.exists():
            missing += 1
            continue
        pairs.append((img_path, fl))
    if missing:
        print(f"  warning: {missing} JSON frames had no image in {images_dir}")
    return pairs


def dominant_class(fl: FrameLabels) -> int:
    """Most frequent class id in a frame, or -1 when it has no obstacle boxes."""
    counts = Counter(int(line.split()[0]) for line in fl.yolo_lines)
    if not counts:
        return -1
    return counts.most_common(1)[0][0]


def stratified_train_val(
    pairs: List[Pair], val_ratio: float, seed: int
) -> Tuple[List[Pair], List[Pair]]:
    """Split ``pairs`` into (train, val), stratified by dominant class."""
    rng = np.random.default_rng(seed)
    seed_int = int(rng.integers(0, 2**31))
    strata = [dominant_class(fl) for _, fl in pairs]
    train, val = train_test_split(
        pairs,
        test_size=val_ratio,
        stratify=strata,
        random_state=seed_int,
    )
    return train, val


def write_split(
    split_name: str, pairs: List[Pair], out_root: Path, copy: bool = False
) -> None:
    """Materialize a split: symlink (or copy) images and write YOLO labels."""
    img_out = out_root / split_name / "images"
    lbl_out = out_root / split_name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for img_path, fl in pairs:
        dst_img = img_out / img_path.name
        if not dst_img.exists():
            if copy:
                shutil.copy2(img_path, dst_img)
            else:
                os.symlink(img_path.resolve(), dst_img)
        body = "\n".join(fl.yolo_lines)
        (lbl_out / f"{img_path.stem}.txt").write_text(
            body + ("\n" if body else "")
        )


def write_attributes(rows, out_csv: Path) -> None:
    """Write the ``name,split,weather,scene,timeofday`` manifest."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "split", "weather", "scene", "timeofday"])
        for name, split, attrs in rows:
            writer.writerow(
                [
                    name,
                    split,
                    attrs.get("weather", ""),
                    attrs.get("scene", ""),
                    attrs.get("timeofday", ""),
                ]
            )
```

> Note: the `from __future__ import annotations` at the top of the file lets the `Pair`/`List[Pair]` annotations resolve even though `csv`/`os` are imported lower down. Keep the top-of-file imports (`json`, dataclass, typing) where they are; the new stdlib + third-party imports may be grouped here at the bottom of the existing import block instead if `flake8` prefers — run black/flake8 in Step 5 and move them up if E402 complains.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_bdd100k.py -q`
Expected: all PASS.

- [ ] **Step 5: Lint (watch for E402 import-order)**

Run: `black src/utils/bdd100k.py tests/test_bdd100k.py && flake8 src/utils/bdd100k.py tests/test_bdd100k.py`
Expected: no errors. If flake8 reports `E402 module level import not at top of file`, move the `csv`/`os`/`shutil`/`Counter`/`numpy`/`sklearn` imports up into the top import block and re-run.

- [ ] **Step 6: Commit**

```bash
git add src/utils/bdd100k.py tests/test_bdd100k.py
git commit -m "feat(data): BDD100K collect/split/materialize prep helpers"
```

---

### Task 4: Rewrite `scripts/preprocess.py` as a thin CLI

**Files:**
- Modify (full rewrite): `scripts/preprocess.py`

- [ ] **Step 1: Replace the file contents**

Overwrite `scripts/preprocess.py` with:

```python
"""Preprocess BDD100K detection labels into a YOLO-format 3-class dataset.

Pipeline
--------
1. Convert each BDD100K JSON frame to YOLO lines with the 10->3 remap
   (src/utils/bdd100k).
2. Split the official **train** set into our train/val (stratified by the
   frame's dominant class, seed=42). The official **val** set becomes our
   held-out **test** (its labels are public; the official test labels are not).
3. Materialize each split under data/processed/bdd100k/<split>/{images,labels}
   (images symlinked by default) and write a combined attributes.csv
   (name,split,weather,scene,timeofday) for the R3 day/night & weather slice.

Usage
-----
    python scripts/preprocess.py
    python scripts/preprocess.py --val-ratio 0.18 --copy
"""

import argparse
from pathlib import Path

from src.utils.bdd100k import (
    collect_frames,
    stratified_train_val,
    write_attributes,
    write_split,
)


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess BDD100K for 3-class training")
    p.add_argument(
        "--images-train",
        type=Path,
        default=Path("data/raw/bdd100k/images/100k/train"),
    )
    p.add_argument(
        "--images-val",
        type=Path,
        default=Path("data/raw/bdd100k/images/100k/val"),
    )
    p.add_argument(
        "--labels-train",
        type=Path,
        default=Path("data/raw/bdd100k/labels/det_20/det_train.json"),
    )
    p.add_argument(
        "--labels-val",
        type=Path,
        default=Path("data/raw/bdd100k/labels/det_20/det_val.json"),
    )
    p.add_argument(
        "--out-root", type=Path, default=Path("data/processed/bdd100k")
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.18,
        help="Fraction of the official train set held out as our val split",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--copy",
        action="store_true",
        help="Copy images instead of symlinking (uses ~5GB more disk)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("Collecting train frames...")
    train_pool = collect_frames(args.images_train, args.labels_train)
    print(f"  {len(train_pool)} frames")
    print("Collecting val frames (-> our held-out test split)...")
    test_pairs = collect_frames(args.images_val, args.labels_val)
    print(f"  {len(test_pairs)} frames")

    print("Splitting train -> train/val (stratified, seed=42)...")
    train_pairs, val_pairs = stratified_train_val(
        train_pool, args.val_ratio, args.seed
    )

    splits = {"train": train_pairs, "val": val_pairs, "test": test_pairs}
    attr_rows = []
    for name, pairs in splits.items():
        print(f"Writing {name}: {len(pairs)} frames...")
        write_split(name, pairs, args.out_root, copy=args.copy)
        attr_rows.extend((fl.name, name, fl.attributes) for _, fl in pairs)

    write_attributes(attr_rows, args.out_root / "attributes.csv")

    print("\nDone. Split sizes:")
    for name, pairs in splits.items():
        print(f"  {name}: {len(pairs)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it imports and the CLI parses (no data needed)**

Run: `python scripts/preprocess.py --help`
Expected: usage text listing `--images-train`, `--val-ratio`, `--copy`, etc. — no import errors.

- [ ] **Step 3: Lint**

Run: `black scripts/preprocess.py && flake8 scripts/preprocess.py`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add scripts/preprocess.py
git commit -m "feat(data): rewrite preprocess.py for BDD100K 3-class pipeline"
```

---

### Task 5: Kaggle download wrapper

A thin wrapper around the `kaggle` binary (shells out — manually verified, not unit-tested, matching `scripts/train.py`/`evaluate.py` which have no pytest). **No `kaggle.json` pre-flight check** — this machine authenticates via an ACCESS_TOKEN (no json file), so a file check would falsely block. The kaggle CLI surfaces its own clear error if auth is missing.

**Files:**
- Create: `scripts/download_bdd100k.py`

- [ ] **Step 1: Write the script**

Create `scripts/download_bdd100k.py`:

```python
"""Download BDD100K (images + detection labels) from a Kaggle mirror.

Prerequisite: Kaggle API auth configured (``kaggle config view`` shows your
username). Create a token at https://www.kaggle.com/settings if needed.

Default mirror: solesensei/solesensei_bdd100k — ships
``images/100k/{train,val,test}/`` and
``labels/bdd100k_labels_images_{train,val}.json`` (with weather/scene/timeofday
attributes). The Kaggle CLI unzips into --dest.

Usage::

    python scripts/download_bdd100k.py
    python scripts/download_bdd100k.py --dataset owner/other-bdd100k

Inspect the extracted layout and pass matching --images-*/--labels-* paths to
scripts/preprocess.py.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Download BDD100K from a Kaggle mirror")
    p.add_argument(
        "--dataset",
        default="solesensei/solesensei_bdd100k",
        help="Kaggle dataset slug (default: the canonical solesensei mirror)",
    )
    p.add_argument("--dest", type=Path, default=Path("data/raw"))
    return p.parse_args()


def main():
    args = parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        args.dataset,
        "-p",
        str(args.dest),
        "--unzip",
    ]
    print("Running:", " ".join(cmd))
    # No pre-flight credential check: auth may be a kaggle.json, an env var,
    # or an ACCESS_TOKEN under ~/.kaggle. The kaggle CLI emits its own clear
    # error if credentials are missing or invalid.
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it imports and the CLI parses (no network)**

Run: `python scripts/download_bdd100k.py --help`
Expected: usage text showing `--dataset` (default `solesensei/solesensei_bdd100k`) and `--dest` — no import errors.

- [ ] **Step 3: Lint**

Run: `black scripts/download_bdd100k.py && flake8 scripts/download_bdd100k.py`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add scripts/download_bdd100k.py
git commit -m "feat(data): add Kaggle BDD100K download wrapper"
```

---

### Task 6: Acquire data + run the pipeline (network checkpoint)

This task downloads ~8 GB and runs the pipeline. The machine is **already authenticated** (`kaggle config view` → user `itsdreowo`). Disk note: ~22 GB free; the solesensei archive is ~8 GB and includes `drivable_maps`/segmentation we don't need — delete those after extraction to reclaim space. Run the download in the background (it is long).

- [ ] **Step 1: Confirm Kaggle auth**

Run: `kaggle config view`
Expected: shows `username: itsdreowo`. (If not, configure auth before continuing.)

- [ ] **Step 2: Download + unzip (background)**

Run: `python scripts/download_bdd100k.py` (pinned default `solesensei/solesensei_bdd100k`, extracting under `data/raw/`).
Expected: downloads + unzips to `data/raw/bdd100k/bdd100k/...`. The `--unzip` flag removes the zip automatically.

- [ ] **Step 3: Reclaim disk (drop unused modalities)**

Run: `rm -rf data/raw/bdd100k/bdd100k/drivable_maps data/raw/bdd100k/bdd100k/seg 2>/dev/null; du -sh data/raw/bdd100k`
We only need detection images + JSON labels.

- [ ] **Step 4: Locate the actual image + label paths**

Run: `find data/raw/bdd100k -maxdepth 5 -type d | grep -iE "images/100k|labels" ` and `find data/raw/bdd100k -name "*.json"`.
For solesensei expect:
- images: `data/raw/bdd100k/bdd100k/images/100k/{train,val}`
- labels: `data/raw/bdd100k/bdd100k/labels/bdd100k_labels_images_{train,val}.json`

- [ ] **Step 5: Run preprocessing (pass the solesensei paths explicitly)**

```bash
python scripts/preprocess.py \
  --images-train data/raw/bdd100k/bdd100k/images/100k/train \
  --images-val   data/raw/bdd100k/bdd100k/images/100k/val \
  --labels-train data/raw/bdd100k/bdd100k/labels/bdd100k_labels_images_train.json \
  --labels-val   data/raw/bdd100k/bdd100k/labels/bdd100k_labels_images_val.json
```

Expected: prints train/val/test frame counts and final split sizes (~57k train / ~13k val / ~10k test), and writes `data/processed/bdd100k/{train,val,test}/{images,labels}` + `attributes.csv`. (Adjust paths if Step 4 shows a different layout.)

- [ ] **Step 6: Validate each split at num_classes=3**

```bash
python scripts/validate_data.py --images data/processed/bdd100k/train/images --labels data/processed/bdd100k/train/labels --num-classes 3
python scripts/validate_data.py --images data/processed/bdd100k/val/images   --labels data/processed/bdd100k/val/labels   --num-classes 3
python scripts/validate_data.py --images data/processed/bdd100k/test/images  --labels data/processed/bdd100k/test/labels  --num-classes 3
```

Expected: each ends with `Result: PASS` (no missing labels, no out-of-range class ids, class distribution over `{0,1,2}`). Capture the three outputs for the report in Task 7. If any split FAILs, **stop** and debug with superpowers:systematic-debugging before proceeding.

---

### Task 7: R1 validation report + docs + final commit

**Files:**
- Create: `docs/bdd100k_data_validation.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Write the data-validation report**

Create `docs/bdd100k_data_validation.md` with: the pinned Kaggle slug (provenance), the three split sizes, the per-split class distribution and image-size stats from Task 6 Step 6, and a one-line PASS/FAIL verdict per split. This is the artifact R1 references for the "BDD100K data-validation report" deliverable.

- [ ] **Step 2: Update the pipeline status in CLAUDE.md**

In `CLAUDE.md`:
- In the "Pipeline status" table, change the **Plan 2** row to `✅ done` and the **Plan 3** row to `🔄 next`.
- In the architecture note, remove `preprocess.py` from the "still VisDrone/drone-shaped" sentence (it is now BDD100K; `validate_data.py`, `train.py`, `evaluate.py`, `modal_train.py` remain for Plan 3).

- [ ] **Step 3: Run the full test suite + lint (CI parity)**

```bash
pytest -q
black --check .
flake8 .
```
Expected: all tests pass; black/flake8 clean (matches `.github/workflows/ci.yml`).

- [ ] **Step 4: Commit**

```bash
git add docs/bdd100k_data_validation.md CLAUDE.md
git commit -m "docs(data): BDD100K validation report + Plan 2 status"
```

---

## Self-review notes

- **Spec coverage:** download (Task 5/6), JSON→YOLO conversion (Task 2), 10→3 remap (Task 2), stratified split with official-val-as-test (Task 3/4), attributes manifest for R3 robustness (Task 3), `nc=3` validation + R1 report (Task 6/7), dataset config consumed by the R3 model lineup (Task 1). KITTI is explicitly deferred to Plan 3 (documented above).
- **Type consistency:** `FrameLabels(name, yolo_lines, attributes)`, `Pair = (Path, FrameLabels)`, and the helper signatures (`collect_frames`, `stratified_train_val(pairs, val_ratio, seed)`, `write_split(split, pairs, out_root, copy)`, `write_attributes(rows, out_csv)`) are used identically across `src/utils/bdd100k.py`, `scripts/preprocess.py`, and the tests.
- **No placeholders:** every code step is complete and runnable; the only execution-time unknowns (Kaggle slug, exact extracted dir names) are discovered by explicit `kaggle datasets list` / `find` commands in Task 6, not left as TODOs.
- **Reuse:** `src/utils/data_validation.py` and `scripts/validate_data.py` are used unchanged.
</content>
</invoke>
