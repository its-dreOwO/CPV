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
            Track(
                track_id=i,
                bbox=obj["bbox"],
                velocity=(0.0, 0.0),
                scale_velocity=0.0,
                age=1,
            )
        )
        dists.append(kitti_distance(obj["location"]))
    risked = assessor.assess(tracks, frame_shape)
    counts = {
        "danger_total": 0,
        "danger_near": 0,
        "caution_total": 0,
        "safe_total": 0,
    }
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
        if counts["danger_total"]
        else None
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
