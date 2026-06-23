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
