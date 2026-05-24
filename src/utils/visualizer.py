from typing import List

import cv2
import numpy as np

from src.tracking.tracker import Track


def draw_tracks(
    frame: np.ndarray, tracks: List[Track], avoidance_cmd: tuple = None
) -> np.ndarray:
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t.bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"ID:{t.track_id} V:[{t.velocity[0]:.1f},{t.velocity[1]:.1f}]",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    if avoidance_cmd:
        yaw, alt = avoidance_cmd
        cv2.putText(
            out,
            f"CMD -> Yaw: {yaw:+.1f} Alt: {alt:+.1f}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
    return out
