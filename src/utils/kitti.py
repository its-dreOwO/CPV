from pathlib import Path
from typing import List, Optional

KITTI_TO_COARSE = {
    "Car": 0,
    "Van": 0,
    "Truck": 0,
    "Tram": 0,
    "Pedestrian": 1,
    "Person_sitting": 1,
    "Cyclist": 2,
}


def parse_label_line(line: str) -> Optional[dict]:
    parts = line.split()
    if len(parts) < 15:
        return None
    try:
        nums = [float(x) for x in parts[1:15]]
    except ValueError:
        return None
    return {
        "type": parts[0],
        "truncated": nums[0],
        "occluded": int(nums[1]),
        "bbox": (nums[3], nums[4], nums[5], nums[6]),
        "dimensions": (nums[7], nums[8], nums[9]),  # h, w, l
        "location": (nums[10], nums[11], nums[12]),  # x, y, z (meters)
    }


def to_yolo_lines(label_path: Path, img_w: int, img_h: int) -> List[str]:
    out = []
    for raw in Path(label_path).read_text().splitlines():
        obj = parse_label_line(raw)
        if obj is None or obj["type"] not in KITTI_TO_COARSE:
            continue
        l, t, r, b = obj["bbox"]
        cx = ((l + r) / 2.0) / img_w
        cy = ((t + b) / 2.0) / img_h
        bw = (r - l) / img_w
        bh = (b - t) / img_h
        out.append(
            f"{KITTI_TO_COARSE[obj['type']]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        )
    return out
