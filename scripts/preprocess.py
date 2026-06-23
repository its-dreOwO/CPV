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
    p.add_argument("--out-root", type=Path, default=Path("data/processed/bdd100k"))
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
    train_pairs, val_pairs = stratified_train_val(train_pool, args.val_ratio, args.seed)

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
