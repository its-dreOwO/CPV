"""Train a detection model using an Ultralytics config YAML.

Usage
-----
    python scripts/train.py --config configs/yolov8m.yaml --epochs 50 --device cuda
    python scripts/train.py --config configs/yolov8n.yaml --epochs 5 --device cuda
"""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Train obstacle detection model")
    p.add_argument(
        "--config", type=str, required=True, help="Path to model config YAML"
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override the dataset path (e.g. /kaggle/input/<slug>)",
    )
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    data_yaml = cfg.pop("data")
    if args.data_root:
        # Rewrite path at runtime without mutating the file on disk
        with open(data_yaml) as f:
            data_cfg = yaml.safe_load(f)
        data_cfg["path"] = args.data_root
        tmp = Path(data_yaml).with_suffix(".tmp.yaml")
        with open(tmp, "w") as f:
            yaml.dump(data_cfg, f)
        data_yaml = str(tmp)

    model_weights = cfg.pop("model")
    model = YOLO(model_weights)

    model.train(
        data=data_yaml,
        epochs=args.epochs,
        device=args.device,
        **cfg,
    )

    if args.data_root:
        Path(data_yaml).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
