from typing import List

import cv2
import numpy as np

from src.tracking.tracker import Track


def draw_tracks(frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t.bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"ID:{t.track_id}",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return out
