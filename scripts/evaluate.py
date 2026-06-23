"""Evaluate a trained detector on the held-out test split.

Runs Ultralytics' built-in ``model.val()`` on a data YAML and prints the
metrics R3's ``reports/R3/model_comparison.md`` table needs:

    - mAP@0.5
    - mAP@0.5:0.95
    - precision
    - recall
    - FPS (1000 / per-image inference ms)
    - params (M) and model size (MB)
    - per-class mAP@0.5 (optional, via --classwise)

Usage
-----
    python scripts/evaluate.py --weights models/yolov8m-best.pt \\
        --data configs/bdd100k.yaml --split test --device 0 \\
        --output reports/R3/yolov8m_metrics.json --classwise
    python scripts/evaluate.py --weights models/yolov8m-best.pt \\
        --data configs/bdd100k.yaml --split test --device cuda \\
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
        help="Dataset YAML (e.g. configs/bdd100k.yaml)",
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
    p.add_argument(
        "--classwise",
        action="store_true",
        help="Include per-class mAP@0.5 in the output (R3 comparison table)",
    )
    return p.parse_args()


def collect_metrics(model: YOLO, results, classwise: bool = False) -> dict:
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

    metrics = {
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
    if per_class is not None:
        metrics["per_class"] = per_class
    return metrics


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

    # Ultralytics resolves a relative `path:` in the data YAML against its own
    # datasets_dir, so always run from repo root with data/processed/bdd100k
    # reachable, or pass an absolute path via --data.
    data_cfg = Path(args.data)
    if not data_cfg.exists():
        raise FileNotFoundError(
            f"Data config not found: {data_cfg} (run from repo root)"
        )

    model = YOLO(args.weights)
    results = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        verbose=False,
    )

    metrics = collect_metrics(model, results, classwise=args.classwise)
    print_report(metrics, args.weights, args.data, args.split)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))
        print(f"\nMetrics written to {out}")


if __name__ == "__main__":
    main()
