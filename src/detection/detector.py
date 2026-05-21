from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Detection:
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str = ""


class BaseDetector:
    def detect(self, frame: np.ndarray) -> List[Detection]:
        raise NotImplementedError
