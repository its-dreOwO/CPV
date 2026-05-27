"""Evaluate a trained detector on the held-out test split.

Runs Ultralytics' built-in ``model.val()`` on a data YAML and prints the
metrics R3's ``reports/R3/model_comparison.md`` table needs:

    - mAP@0.5
    - mAP@0.5:0.95
    - precision
    - recall
    - FPS (1000 / per-image inference ms)
    - params (M) and model size (MB)

Usage
-----
    python scripts/evaluate.py --weights models/yolov8m_best.pt \\
        --data configs/visdrone5.yaml
    python scripts/evaluate.py --weights models/yolov8m_best.pt \\
        --data configs/visdrone5.yaml --split test --device cuda \\
        --output reports/R3/yolov8m_metrics.json
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate obstacle detection model")
    p.add_argument(
        "--weights", type=str, required=True, help="Path to trained weights (.pt)"
    )
    p.add_argument(
        "--data",
        type=str,
        required=True,
        help="Dataset YAML (e.g. configs/visdrone5.yaml)",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Which split to evaluate on",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to dump metrics as JSON",
    )
    return p.parse_args()


def collect_metrics(model: YOLO, results) -> dict:
    """Pull the R3 comparison-table fields out of an Ultralytics results object."""
    box = results.box
    speed = results.speed  # ms per image: {'preprocess', 'inference', 'postprocess'}
    inference_ms = float(speed.get("inference", 0.0))
    total_ms = sum(float(v) for v in speed.values())

    n_params = sum(p.numel() for p in model.model.parameters())
    weights_path = Path(model.ckpt_path) if getattr(model, "ckpt_path", None) else None
    size_mb = (
        weights_path.stat().st_size / (1024 * 1024)
        if weights_path and weights_path.exists()
        else None
    )

    return {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "fps_inference_only": 1000.0 / inference_ms if inference_ms > 0 else None,
        "fps_end_to_end": 1000.0 / total_ms if total_ms > 0 else None,
        "inference_ms_per_image": inference_ms,
        "total_ms_per_image": total_ms,
        "params_millions": n_params / 1e6,
        "weights_size_mb": size_mb,
    }


def print_report(metrics: dict, weights: str, data: str, split: str) -> None:
    print()
    print(f"Evaluation report — {weights}")
    print(f"  data:  {data}  split={split}")
    print("-" * 60)
    print(f"  mAP@0.5            : {metrics['map50']:.4f}")
    print(f"  mAP@0.5:0.95       : {metrics['map50_95']:.4f}")
    print(f"  precision          : {metrics['precision']:.4f}")
    print(f"  recall             : {metrics['recall']:.4f}")
    fps_inf = metrics["fps_inference_only"]
    fps_e2e = metrics["fps_end_to_end"]
    fps_inf_s = f"{fps_inf:.1f}" if fps_inf else "n/a"
    fps_e2e_s = f"{fps_e2e:.1f}" if fps_e2e else "n/a"
    print(f"  FPS (inference)    : {fps_inf_s}")
    print(f"  FPS (end-to-end)   : {fps_e2e_s}")
    print(f"  inference ms/img   : {metrics['inference_ms_per_image']:.2f}")
    print(f"  params (M)         : {metrics['params_millions']:.2f}")
    size = metrics["weights_size_mb"]
    if size is not None:
        print(f"  weights size (MB)  : {size:.2f}")


def main():
    args = parse_args()

    model = YOLO(args.weights)
    results = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        verbose=False,
    )

    metrics = collect_metrics(model, results)
    print_report(metrics, args.weights, args.data, args.split)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))
        print(f"\nMetrics written to {out}")


if __name__ == "__main__":
    main()
