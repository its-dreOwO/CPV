from typing import List, Tuple

from src.tracking.tracker import Track


class BaseAvoidancePlanner:
    def plan(self, tracks: List[Track]) -> Tuple[float, float]:
        """Return (yaw_delta, altitude_delta) command."""
        raise NotImplementedError
