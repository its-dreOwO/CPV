from typing import List, Tuple

from src.risk.assessor import BaseRiskAssessor, RiskedTrack, RiskLevel
from src.tracking.tracker import Track


class RiskZoneAssessor(BaseRiskAssessor):
    """Monocular dashcam risk assessor (Approach A).

    WHERE: is the object's ground-contact point inside the ego-path trapezoid?
    HOW CLOSE: is its bounding box large or growing (Kalman area velocity)?
    """

    def __init__(
        self,
        horizon_ratio: float = 0.5,
        top_width: float = 0.1,
        bottom_width: float = 0.9,
        large_area_frac: float = 0.05,
        growth_thresh: float = 0.02,
        min_danger_area_frac: float = 0.01,
    ):
        self.horizon_ratio = horizon_ratio
        self.top_width = top_width
        self.bottom_width = bottom_width
        self.large_area_frac = large_area_frac
        self.growth_thresh = growth_thresh
        self.min_danger_area_frac = min_danger_area_frac

    def _half_width_at(self, py: float, h: int, w: int):
        """Half-width of the trapezoid at image row py, or None if above horizon."""
        horizon_y = self.horizon_ratio * h
        if py < horizon_y:
            return None
        t = (py - horizon_y) / max(h - horizon_y, 1e-6)
        t = min(max(t, 0.0), 1.0)
        top_half = self.top_width * w / 2.0
        bottom_half = self.bottom_width * w / 2.0
        return top_half + (bottom_half - top_half) * t

    def ego_path_polygon(self, frame_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Four trapezoid corners (top-left, top-right, bottom-right, bottom-left)."""
        h, w = frame_shape
        cx = w / 2.0
        horizon_y = self.horizon_ratio * h
        top_half = self.top_width * w / 2.0
        bottom_half = self.bottom_width * w / 2.0
        return [
            (int(cx - top_half), int(horizon_y)),
            (int(cx + top_half), int(horizon_y)),
            (int(cx + bottom_half), int(h)),
            (int(cx - bottom_half), int(h)),
        ]

    def assess(
        self, tracks: List[Track], frame_shape: Tuple[int, int]
    ) -> List[RiskedTrack]:
        h, w = frame_shape
        cx = w / 2.0
        frame_area = float(h * w)
        out = []
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            contact_x = (x1 + x2) / 2.0
            contact_y = y2
            area = max((x2 - x1) * (y2 - y1), 0.0)

            half_width = self._half_width_at(contact_y, h, w)
            in_path = half_width is not None and abs(contact_x - cx) <= half_width

            growth_rate = track.scale_velocity / area if area > 0 else 0.0
            ttc_proxy = growth_rate

            if not in_path:
                risk = RiskLevel.SAFE
            else:
                area_frac = area / frame_area
                big = area_frac >= self.large_area_frac
                # A distant (tiny) box's area-growth signal is noise-dominated:
                # scale_velocity / area explodes as area -> 0. Require the box to
                # clear a minimum size before growth may escalate to DANGER, so
                # far objects cannot false-trigger on Kalman jitter.
                near_enough = area_frac >= self.min_danger_area_frac
                growing = near_enough and growth_rate >= self.growth_thresh
                risk = RiskLevel.DANGER if (big or growing) else RiskLevel.CAUTION

            out.append(
                RiskedTrack(
                    track=track,
                    risk=risk,
                    in_path=in_path,
                    ttc_proxy=ttc_proxy,
                )
            )
        return out
