import numpy as np

from src.risk.assessor import RiskedTrack, RiskLevel
from src.tracking.tracker import Track
from src.utils.visualizer import draw_risk


def test_draw_risk_returns_same_shape_without_mutating_input():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    risked = [
        RiskedTrack(
            track=Track(track_id=1, bbox=[100, 300, 200, 460]),
            risk=RiskLevel.DANGER,
            in_path=True,
            ttc_proxy=0.1,
        )
    ]
    poly = [(310, 240), (330, 240), (600, 480), (40, 480)]
    out = draw_risk(frame, risked, ego_polygon=poly)
    assert out.shape == frame.shape
    # Original frame is untouched; output has drawn pixels.
    assert frame.sum() == 0
    assert out.sum() > 0
