import math
from src.risk.zone_assessor import RiskZoneAssessor
from scripts.validate_risk import kitti_distance, risk_distance_agreement


def test_kitti_distance():
    assert abs(kitti_distance((3.0, 0.0, 4.0)) - 5.0) < 1e-6


def test_agreement_flags_near_in_path_object_as_danger():
    # Frame 1000x500. An in-path, very large box near the bottom-center.
    assessor = RiskZoneAssessor(large_area_frac=0.05)
    objects = [
        # near, centered, large -> expect DANGER and near
        {"bbox": (400.0, 300.0, 600.0, 500.0), "location": (0.0, 0.0, 6.0)},
        # off to the far side, tiny, far -> SAFE
        {"bbox": (10.0, 250.0, 30.0, 270.0), "location": (0.0, 0.0, 60.0)},
    ]
    out = risk_distance_agreement(
        assessor, objects, frame_shape=(500, 1000), near_thresh_m=15.0
    )
    assert out["danger_total"] >= 1
    assert out["danger_near"] >= 1
    assert math.isclose(out["precision_danger_near"], 1.0)
