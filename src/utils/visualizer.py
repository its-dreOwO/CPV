from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.risk.assessor import RiskedTrack, RiskLevel

_COLORS = {
    RiskLevel.SAFE: (0, 200, 0),  # green
    RiskLevel.CAUTION: (0, 215, 255),  # amber
    RiskLevel.DANGER: (0, 0, 255),  # red
}


def draw_risk(
    frame: np.ndarray,
    risked: List[RiskedTrack],
    ego_polygon: Optional[List[Tuple[int, int]]] = None,
) -> np.ndarray:
    """Render the ego-path region and color-coded per-object risk boxes."""
    out = frame.copy()

    if ego_polygon:
        pts = np.array(ego_polygon, dtype=np.int32).reshape((-1, 1, 2))
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], (255, 180, 0))
        out = cv2.addWeighted(overlay, 0.2, out, 0.8, 0)
        cv2.polylines(out, [pts], True, (255, 180, 0), 2)

    for rt in risked:
        x1, y1, x2, y2 = [int(v) for v in rt.track.bbox]
        color = _COLORS.get(rt.risk, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"{rt.risk} ID:{rt.track.track_id}",
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return out
