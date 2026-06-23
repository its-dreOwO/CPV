# BDD100K Data Validation Report

Date: 2026-06-23

## Provenance

- Source dataset: Kaggle mirror `solesensei/solesensei_bdd100k`
- Raw images: `data/raw/bdd100k/bdd100k/images/100k/{train,val,test}`
- Raw labels: `data/raw/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_{train,val}.json`
- Conversion: `scripts/preprocess.py`
- Class remap: native BDD100K detection labels collapsed to `vehicle`, `person`, `two_wheeler`
- Split policy: official train split stratified into train/val with `seed=42` and `val_ratio=0.18`; official val split used as held-out test
- Materialization: images symlinked under `data/processed/bdd100k`, labels written in YOLO format, attributes written to `attributes.csv`

## Split Summary

| Split | Images | Labels | Class 0 vehicle | Class 1 person | Class 2 two_wheeler | Image size | Verdict |
|-------|--------|--------|-----------------|----------------|---------------------|------------|---------|
| train | 57,287 | 57,287 | 619,565 | 79,036 | 8,373 | 1280x720 | PASS |
| val | 12,576 | 12,576 | 135,425 | 16,830 | 1,839 | 1280x720 | PASS |
| test | 10,000 | 10,000 | 108,363 | 13,911 | 1,459 | 1280x720 | PASS |

`attributes.csv` contains 79,863 data rows plus the header.

## Validator Results

Each split was validated with `scripts/validate_data.py --num-classes 3`.

| Split | Unreadable images | Missing labels | Orphan labels | Invalid label lines | Duplicate images | Result |
|-------|-------------------|----------------|---------------|---------------------|------------------|--------|
| train | OK | OK | OK | OK | OK | PASS |
| val | OK | OK | OK | OK | OK | PASS |
| test | OK | OK | OK | OK | OK | PASS |

The class distribution is imbalanced as expected for road scenes: vehicles dominate, pedestrians are secondary, and two-wheelers are sparse. This is acceptable for R1 data validation and should be handled during model evaluation with per-class metrics in R3.
