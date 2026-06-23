from dataclasses import dataclass
from typing import List, Tuple

from src.tracking.tracker import Track


class RiskLevel:
    """String constants for the three advisory risk levels."""

    SAFE = "SAFE"
    CAUTION = "CAUTION"
    DANGER = "DANGER"


@dataclass
class RiskedTrack:
    track: Track
    risk: str
    in_path: bool
    ttc_proxy: float


class BaseRiskAssessor:
    def assess(
        self, tracks: List[Track], frame_shape: Tuple[int, int]
    ) -> List[RiskedTrack]:
        """Tag each track SAFE/CAUTION/DANGER. frame_shape is (height, width)."""
        raise NotImplementedError
