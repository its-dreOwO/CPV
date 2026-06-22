from src.tracking.tracker import Track


def test_track_default_velocity():
    t = Track(track_id=1, bbox=[10, 10, 50, 50])
    assert t.velocity == [0.0, 0.0]
    assert t.age == 0


def test_track_has_scale_velocity_default():
    t = Track(track_id=1, bbox=[10, 10, 50, 50])
    assert t.scale_velocity == 0.0


def test_kalman_tracker_reports_growing_scale_velocity():
    from src.detection.detector import Detection
    from src.tracking.kalman_tracker import KalmanTracker

    tracker = KalmanTracker(min_hits=1)
    # Box grows each frame (area increasing) -> object approaching the camera.
    tracker.update([Detection(bbox=[100, 100, 200, 200], confidence=0.9, class_id=0)])
    tracker.update([Detection(bbox=[95, 95, 210, 210], confidence=0.9, class_id=0)])
    tracks = tracker.update(
        [Detection(bbox=[90, 90, 220, 220], confidence=0.9, class_id=0)]
    )
    assert len(tracks) == 1
    assert tracks[0].scale_velocity > 0
