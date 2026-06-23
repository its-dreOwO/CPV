from src.detection.detector import Detection
from src.risk.assessor import RiskLevel
from src.risk.zone_assessor import RiskZoneAssessor
from src.tracking.kalman_tracker import KalmanTracker


def test_tracker_assessor_integration():
    """Tracker output flows into the risk assessor without error."""
    tracker = KalmanTracker(min_hits=1)
    assessor = RiskZoneAssessor()

    det1 = Detection(bbox=[280, 400, 360, 480], confidence=0.9, class_id=0)
    tracker.update([det1])
    det2 = Detection(bbox=[275, 395, 365, 490], confidence=0.9, class_id=0)
    tracks = tracker.update([det2])

    assert len(tracks) == 1
    risked = assessor.assess(tracks, frame_shape=(640, 640))
    assert len(risked) == 1
    assert risked[0].risk in (RiskLevel.SAFE, RiskLevel.CAUTION, RiskLevel.DANGER)


def test_centered_approaching_object_is_in_path():
    """A centered object near the bottom is flagged in-path."""
    tracker = KalmanTracker(min_hits=1)
    assessor = RiskZoneAssessor()
    det = Detection(bbox=[220, 420, 420, 620], confidence=0.9, class_id=0)
    tracks = tracker.update([det])
    risked = assessor.assess(tracks, frame_shape=(640, 640))
    assert risked[0].in_path is True
