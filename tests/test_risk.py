from src.risk.assessor import BaseRiskAssessor, RiskedTrack, RiskLevel
from src.risk.zone_assessor import RiskZoneAssessor
from src.tracking.tracker import Track


def test_risk_level_constants():
    assert RiskLevel.SAFE == "SAFE"
    assert RiskLevel.CAUTION == "CAUTION"
    assert RiskLevel.DANGER == "DANGER"


def test_risked_track_wraps_a_track():
    t = Track(track_id=7, bbox=[0, 0, 10, 10])
    rt = RiskedTrack(track=t, risk=RiskLevel.SAFE, in_path=False, ttc_proxy=0.0)
    assert rt.track.track_id == 7
    assert rt.risk == "SAFE"
    assert rt.in_path is False


def test_base_assessor_is_abstract():
    import pytest

    with pytest.raises(NotImplementedError):
        BaseRiskAssessor().assess([], (640, 640))


FRAME = (640, 640)  # (height, width)


def _assess_one(track):
    return RiskZoneAssessor().assess([track], FRAME)[0]


def test_off_path_object_is_safe():
    # Bottom-center x=10 is left of the trapezoid edge (half-width ~272 at y=620,
    # so the in-path band is x in [48, 592]); 10 is outside -> off-path.
    t = Track(track_id=1, bbox=[0, 580, 20, 620])  # bottom-center x=10
    rt = _assess_one(t)
    assert rt.in_path is False
    assert rt.risk == RiskLevel.SAFE


def test_object_above_horizon_is_safe():
    # Bottom-center y=100 is above the horizon line (320) -> not in path.
    t = Track(track_id=2, bbox=[300, 60, 340, 100])
    rt = _assess_one(t)
    assert rt.in_path is False
    assert rt.risk == RiskLevel.SAFE


def test_large_in_path_object_is_danger():
    # Centered, near bottom, large box (~0.1 of frame area).
    t = Track(track_id=3, bbox=[220, 420, 420, 620])  # 200x200, bottom-center (320,620)
    rt = _assess_one(t)
    assert rt.in_path is True
    assert rt.risk == RiskLevel.DANGER


def test_small_growing_in_path_object_is_danger():
    # Small box but area growing fast: growth_rate = 20/400 = 0.05 >= 0.02.
    t = Track(track_id=4, bbox=[310, 590, 330, 610], scale_velocity=20.0)
    rt = _assess_one(t)
    assert rt.in_path is True
    assert rt.risk == RiskLevel.DANGER
    assert rt.ttc_proxy > 0


def test_small_stable_in_path_object_is_caution():
    # Small box (area frac ~0.001 < 0.05), not growing -> caution.
    t = Track(track_id=5, bbox=[310, 590, 330, 610], scale_velocity=0.0)
    rt = _assess_one(t)
    assert rt.in_path is True
    assert rt.risk == RiskLevel.CAUTION


def test_ego_path_polygon_has_four_points():
    poly = RiskZoneAssessor().ego_path_polygon(FRAME)
    assert len(poly) == 4
    assert all(len(pt) == 2 for pt in poly)
