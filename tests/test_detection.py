from src.detection.detector import Detection


def test_detection_fields():
    d = Detection(bbox=[0, 0, 100, 100], confidence=0.9, class_id=0, class_name="person")
    assert d.confidence == 0.9
    assert len(d.bbox) == 4
