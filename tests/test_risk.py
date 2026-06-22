from src.risk.assessor import BaseRiskAssessor, RiskedTrack, RiskLevel
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
