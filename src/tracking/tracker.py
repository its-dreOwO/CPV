from dataclasses import dataclass, field
from typing import List

from src.detection.detector import Detection


@dataclass
class Track:
    track_id: int
    bbox: List[float]
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0])
    scale_velocity: float = 0.0
    age: int = 0


class BaseTracker:
    def update(self, detections: List[Detection]) -> List[Track]:
        raise NotImplementedError
