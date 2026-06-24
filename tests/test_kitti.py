from src.utils.kitti import parse_label_line, to_yolo_lines, KITTI_TO_COARSE


def test_parse_label_line_car():
    line = "Car 0.00 0 1.55 100.0 150.0 300.0 350.0 1.5 1.6 4.0 5.0 1.7 20.0 0.1"
    obj = parse_label_line(line)
    assert obj["type"] == "Car"
    assert obj["bbox"] == (100.0, 150.0, 300.0, 350.0)
    assert obj["location"] == (5.0, 1.7, 20.0)


def test_parse_label_line_dontcare_returns_object_but_unmapped():
    obj = parse_label_line("DontCare -1 -1 -10 0 0 0 0 -1 -1 -1 -1000 -1000 -1000 -10")
    assert obj["type"] == "DontCare"
    assert "DontCare" not in KITTI_TO_COARSE


def test_to_yolo_lines_maps_and_normalizes(tmp_path):
    lp = tmp_path / "000000.txt"
    lp.write_text(
        "Car 0 0 0 0 0 100 100 1.5 1.6 4 5 1.7 20 0.1\n"
        "Pedestrian 0 0 0 50 50 90 150 1.7 0.5 0.5 1 1 8 0\n"
        "DontCare -1 -1 -10 0 0 0 0 -1 -1 -1 -1000 -1000 -1000 -10\n"
    )
    lines = to_yolo_lines(lp, img_w=1000, img_h=500)
    assert len(lines) == 2  # DontCare dropped
    cls, cx, cy, w, h = lines[0].split()
    assert cls == "0"  # Car -> vehicle
    assert abs(float(cx) - 0.05) < 1e-6  # (0+100)/2 / 1000
    assert abs(float(w) - 0.10) < 1e-6  # 100/1000
    assert lines[1].split()[0] == "1"  # Pedestrian -> person
