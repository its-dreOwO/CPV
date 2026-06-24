# R3 — Model Comparison & Robustness Results

**Generated:** 2026-06-24 · BDD100K **test** split (10,000 images) · imgsz 640 · seed 42
FPS measured on an RTX 4060 Laptop GPU. Per-model JSON lives alongside this file
(`<model>_metrics.json`, `<model>_robustness_{timeofday,weather}.json`).

## Headline comparison (BDD100K test)

| Model | mAP@0.5 | mAP@0.5:0.95 | FPS (e2e) | params (M) | vehicle | person | two_wheeler |
|-------|--------:|-------------:|----------:|-----------:|--------:|-------:|------------:|
| yolov8n | 0.570 | 0.306 | 555 | 3.0 | 0.760 | 0.545 | 0.405 |
| **yolov8m** | **0.683** | **0.378** | 110 | 25.9 | 0.818 | 0.673 | 0.558 |
| yolov10n | 0.552 | 0.299 | 631 | 2.7 | 0.759 | 0.516 | 0.381 |

**Selection rule:** highest mAP@0.5 subject to FPS ≥ 30. All three clear 30 FPS, so
**yolov8m wins on accuracy (0.683 mAP@0.5)** and ships in R4.

**Reading the two clean axes** (yolov8n is the shared anchor):
- **Capacity** (v8n → v8m, same architecture): +0.113 mAP@0.5 for ~8.6× the params and ~5× the latency.
- **Version/paradigm** (v8n → v10n, same nano scale): v10n is NMS-free/end-to-end and ~14% faster
  (631 vs 555 FPS) but slightly less accurate (−0.018 mAP@0.5) at this capacity on BDD100K.
- Hardest class is consistently `two_wheeler` (smallest, rarest); `vehicle` is easiest.

## Robustness — mAP@0.5 by time of day

| time of day | n_img | yolov8n | yolov8m | yolov10n |
|-------------|------:|--------:|--------:|---------:|
| daytime | 5258 | 0.579 | 0.693 | 0.562 |
| dawn/dusk | 778 | 0.564 | 0.682 | 0.552 |
| night | 3929 | 0.550 | 0.660 | 0.528 |

## Robustness — mAP@0.5 by weather

| weather | n_img | yolov8n | yolov8m | yolov10n |
|---------|------:|--------:|--------:|---------:|
| clear | 5346 | 0.559 | 0.674 | 0.537 |
| overcast | 1239 | 0.588 | 0.689 | 0.570 |
| partly cloudy | 738 | 0.561 | 0.693 | 0.548 |
| rainy | 738 | 0.550 | 0.659 | 0.541 |
| snowy | 769 | 0.523 | 0.661 | 0.511 |
| foggy | 13 | 0.557 | 0.777 | 0.518 |

**Takeaways:** yolov8m is best in every slice. Night costs all models ~0.03 mAP@0.5 vs daytime;
snowy is the hardest weather. The `foggy` slice (n=13) is too small to be meaningful — report with a caveat.

## Still pending (blocked on KITTI download)

KITTI raw data is not yet downloaded, so these R3 items remain:
- KITTI zero-shot cross-dataset generalization (`scripts/evaluate.py --data configs/kitti.yaml --split val`)
- Risk-zone heuristic validation against KITTI 3D/depth (`scripts/validate_risk.py`)

See `docs/training_pipeline.md` and the handoff for the exact runbook.
