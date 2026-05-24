import numpy as np
from typing import List, Tuple
from src.tracking.tracker import Track
from src.avoidance.planner import BaseAvoidancePlanner


class GeometricPlanner(BaseAvoidancePlanner):
    """
    Geometric-based avoidance planner.
    Calculates drone movements (yaw and altitude) to maintain a safe distance.
    """

    def __init__(
        self,
        safe_distance: float = 150.0,
        time_horizon: float = 2.0,
        frame_width: int = 640,
        frame_height: int = 640,
    ):
        """
        Initialize the planner.
        """
        self.safe_distance = safe_distance
        self.time_horizon = time_horizon
        self.frame_width = frame_width
        self.frame_height = frame_height

    def plan(self, tracks: List[Track]) -> Tuple[float, float]:
        """
        Calculates (yaw_delta, altitude_delta) based on current tracks.
        """
        if not tracks:
            return 0.0, 0.0

        yaw_delta_acc = 0.0
        altitude_delta_acc = 0.0

        frame_center_x = self.frame_width / 2
        frame_center_y = self.frame_height / 2

        for track in tracks:
            # Get current center
            x1, y1, x2, y2 = track.bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Predict future position based on velocity
            vx, vy = track.velocity
            future_cx = cx + vx * self.time_horizon
            future_cy = cy + vy * self.time_horizon

            # Distance from center
            dist = np.sqrt(
                (future_cx - frame_center_x) ** 2 + (future_cy - frame_center_y) ** 2
            )

            if dist < self.safe_distance:
                # Calculate repulsive force
                force = (self.safe_distance - dist) / self.safe_distance

                # Direction away from obstacle
                dx = frame_center_x - future_cx
                dy = frame_center_y - future_cy

                # Normalize direction
                mag = np.sqrt(dx**2 + dy**2)
                if mag > 0:
                    dx /= mag
                    dy /= mag

                # Accumulate deltas (simplified)
                yaw_delta_acc += dx * force * 10.0  # scale factor
                altitude_delta_acc += dy * force * 10.0

        # Clamp output
        yaw_delta = np.clip(yaw_delta_acc, -30.0, 30.0)
        altitude_delta = np.clip(altitude_delta_acc, -20.0, 20.0)

        return float(yaw_delta), float(altitude_delta)
