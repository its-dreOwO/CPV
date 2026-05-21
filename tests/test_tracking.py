from src.tracking.tracker import Track


def test_track_default_velocity():
    t = Track(track_id=1, bbox=[10, 10, 50, 50])
    assert t.velocity == [0.0, 0.0]
    assert t.age == 0
