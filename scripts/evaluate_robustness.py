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
import tempfile
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
    p.add_argument(
        "--by",
        default="timeofday",
        choices=["timeofday", "weather", "scene"],
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    import yaml

    from ultralytics import YOLO

    args = _parse_args()
    groups = group_image_names(Path(args.attributes), args.split, args.by)
    model = YOLO(args.weights)
    results = {}
    img_root = Path("data/processed/bdd100k") / args.split / "images"

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
            r = model.val(
                data=str(tmp_yaml),
                split="val",
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )
            results[value] = {"n_images": len(paths), "map50": float(r.box.map50)}

    out = {"by": args.by, "split": args.split, "slices": results}
    print(json.dumps(out, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
