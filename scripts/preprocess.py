"""Phase 3: remap VisDrone 10-class labels to 5 coarse classes and produce
a stratified 70/15/15 train/val/test split under data/processed/.

Class mapping
-------------
    vehicle (0) <- car(3), van(4), truck(5), bus(8)
    person  (1) <- pedestrian(0), people(1)
    static  (2)  -- reserved; no VisDrone source
    flying  (3)  -- reserved; no VisDrone source
    other   (4) <- bicycle(2), tricycle(6), awning-tricycle(7), motor(9)

Usage
-----
    python scripts/preprocess.py
    python scripts/preprocess.py --raw-root data/raw/VisDrone_Dataset --seed 42
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

COARSE_MAP = {0: 1, 1: 1, 2: 4, 3: 0, 4: 0, 5: 0, 6: 4, 7: 4, 8: 0, 9: 4}

RAW_SPLITS = [
    "VisDrone2019-DET-train",
    "VisDrone2019-DET-val",
    "VisDrone2019-DET-test-dev",
]


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess VisDrone for 5-class training")
    p.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/VisDrone_Dataset"),
        help="Root of the raw VisDrone dataset",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/processed"),
        help="Output root for processed splits",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--val-ratio", type=float, default=0.15, help="Fraction of data for val"
    )
    p.add_argument(
        "--test-ratio", type=float, default=0.15, help="Fraction of data for test"
    )
    return p.parse_args()


def dominant_coarse_class(label_path: Path) -> int:
    """Return the most frequent coarse class in a label file."""
    from collections import Counter

    counts: Counter = Counter()
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            orig = int(parts[0])
            counts[COARSE_MAP[orig]] += 1
    if not counts:
        return -1  # no annotations — treated as its own stratum
    return counts.most_common(1)[0][0]


def remap_label(src: Path, dst: Path) -> None:
    """Write a copy of src with class IDs remapped to the 5-class scheme."""
    lines = []
    for line in src.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            parts[0] = str(COARSE_MAP[int(parts[0])])
            lines.append(" ".join(parts))
    dst.write_text("\n".join(lines) + ("\n" if lines else ""))


def collect_samples(raw_root: Path):
    """Return parallel lists of (image_path, label_path) from all labeled splits."""
    images, labels = [], []
    for split in RAW_SPLITS:
        img_dir = raw_root / split / "images"
        lbl_dir = raw_root / split / "labels"
        for img in sorted(img_dir.glob("*.jpg")):
            lbl = lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                images.append(img)
                labels.append(lbl)
    return images, labels


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    seed_int = int(rng.integers(0, 2**31))

    print("Collecting samples...")
    images, labels = collect_samples(args.raw_root)
    print(f"  {len(images)} labeled images found")

    print("Computing dominant class per image (stratification key)...")
    strata = [dominant_coarse_class(lbl) for lbl in labels]

    # 70 / (15+15) split, then split remainder 50/50
    train_ratio = 1.0 - args.val_ratio - args.test_ratio
    imgs_train, imgs_tmp, lbls_train, lbls_tmp, str_train, str_tmp = train_test_split(
        images,
        labels,
        strata,
        train_size=train_ratio,
        stratify=strata,
        random_state=seed_int,
    )
    val_frac_of_tmp = args.val_ratio / (args.val_ratio + args.test_ratio)
    imgs_val, imgs_test, lbls_val, lbls_test = train_test_split(
        imgs_tmp,
        lbls_tmp,
        train_size=val_frac_of_tmp,
        stratify=str_tmp,
        random_state=seed_int,
    )

    splits = {
        "train": (imgs_train, lbls_train),
        "val": (imgs_val, lbls_val),
        "test": (imgs_test, lbls_test),
    }

    for split_name, (split_imgs, split_lbls) in splits.items():
        img_out = args.out_root / split_name / "images"
        lbl_out = args.out_root / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        print(f"Writing {split_name}: {len(split_imgs)} images...")
        for img, lbl in zip(split_imgs, split_lbls):
            shutil.copy2(img, img_out / img.name)
            remap_label(lbl, lbl_out / lbl.name)

    print("\nDone. Split sizes:")
    for split_name, (split_imgs, _) in splits.items():
        print(f"  {split_name}: {len(split_imgs)}")


if __name__ == "__main__":
    main()
