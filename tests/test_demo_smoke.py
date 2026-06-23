import numpy as np

from src.risk.zone_assessor import RiskZoneAssessor
from src.tracking.kalman_tracker import KalmanTracker
from src.utils.visualizer import draw_risk


def test_pipeline_smoke_without_detector():
    """Track + assess + draw on a synthetic frame, no model download required."""
    from src.detection.detector import Detection

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracker = KalmanTracker(min_hits=1)
    assessor = RiskZoneAssessor()

    tracks = tracker.update(
        [Detection(bbox=[280, 300, 360, 440], confidence=0.9, class_id=0)]
    )
    risked = assessor.assess(tracks, frame_shape=frame.shape[:2])
    vis = draw_risk(frame, risked, ego_polygon=assessor.ego_path_polygon((480, 640)))

    assert vis.shape == frame.shape
    assert vis.sum() > 0
