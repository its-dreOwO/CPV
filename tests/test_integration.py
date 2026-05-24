from src.detection.detector import Detection
from src.tracking.kalman_tracker import KalmanTracker
from src.avoidance.geometric_planner import GeometricPlanner


def test_tracker_planner_integration():
    """Test that the tracker and planner can handle a basic detection sequence."""
    tracker = KalmanTracker(
        min_hits=1
    )  # Set min_hits to 1 for immediate tracking in test
    planner = GeometricPlanner(frame_width=640, frame_height=640)

    # 1. First frame: Object approaching center
    det1 = Detection(bbox=[100, 100, 200, 200], confidence=0.9, class_id=0)
    tracks = tracker.update([det1])

    assert len(tracks) == 1
    t1 = tracks[0]
    assert t1.track_id == 1

    # 2. Second frame: Object moved
    det2 = Detection(bbox=[110, 110, 210, 210], confidence=0.9, class_id=0)
    tracks = tracker.update([det2])

    assert len(tracks) == 1
    t2 = tracks[0]
    # Check velocity estimation ( KalmabBoxTracker should estimate vx, vy)
    assert abs(t2.velocity[0]) > 0
    assert abs(t2.velocity[1]) > 0

    # 3. Plan avoidance
    yaw, alt = planner.plan(tracks)

    # Since object is in top-left (100,100) and moving down-right,
    # and frame center is (320,320), the "repulsive force" should
    # move us away from it if it's within safe distance.
    # In this test, it's far enough potentially, but let's check it doesn't crash.
    assert isinstance(yaw, float)
    assert isinstance(alt, float)


def test_planner_repulsion():
    """Test the planner's repulsion logic specifically."""
    planner = GeometricPlanner(safe_distance=500.0, frame_width=640, frame_height=640)

    # Object right in the center future path
    from src.tracking.tracker import Track

    track = Track(track_id=1, bbox=[310, 310, 330, 330], velocity=[0.0, 0.0])

    yaw, alt = planner.plan([track])

    # Should stay 0 if exactly pinned? No, the code has a dx = center - future_cx
    # If cx = 320, dx = 0.
    assert yaw == 0.0
    assert alt == 0.0

    # Object to the left
    track_left = Track(track_id=2, bbox=[100, 310, 120, 330], velocity=[0.0, 0.0])
    yaw, alt = planner.plan([track_left])
    # Should move right (positive yaw)
    assert yaw > 0
