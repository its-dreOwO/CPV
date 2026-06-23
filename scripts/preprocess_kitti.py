"""Convert KITTI object-detection labels to the 3-class YOLO layout.

Raw layout expected (KITTI 'left color images' + 'training labels'):
    data/raw/kitti/image_2/000000.png ...
    data/raw/kitti/label_2/000000.txt ...

Usage:
    python scripts/preprocess_kitti.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402
from src.utils.kitti import to_yolo_lines  # noqa: E402

RAW = Path("data/raw/kitti")
OUT = Path("data/processed/kitti")


def main():
    img_dir, lbl_dir = RAW / "image_2", RAW / "label_2"
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    n = 0
    for img in sorted(img_dir.glob("*.png")):
        label = lbl_dir / f"{img.stem}.txt"
        if not label.exists():
            continue
        with Image.open(img) as im:
            w, h = im.size
        lines = to_yolo_lines(label, w, h)
        (OUT / "labels" / f"{img.stem}.txt").write_text("\n".join(lines) + "\n")
        dst = OUT / "images" / img.name
        if not dst.exists():
            dst.symlink_to(img.resolve())
        n += 1
    print(f"Converted {n} KITTI frames -> {OUT}")


if __name__ == "__main__":
    main()
