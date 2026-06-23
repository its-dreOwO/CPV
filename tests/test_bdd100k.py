from src.utils.bdd100k import (
    CLASS_NAMES,
    FrameLabels,
    box2d_to_yolo,
    category_to_class_id,
    convert_frame,
)


def test_class_names_order():
    assert CLASS_NAMES == ["vehicle", "person", "two_wheeler"]


def test_category_map_vehicle_case_insensitive():
    assert category_to_class_id("car") == 0
    assert category_to_class_id("Truck") == 0
    assert category_to_class_id("bus") == 0
    assert category_to_class_id("train") == 0


def test_category_map_person_both_spellings():
    assert category_to_class_id("pedestrian") == 1
    assert category_to_class_id("person") == 1
    assert category_to_class_id("rider") == 1


def test_category_map_two_wheeler_both_spellings():
    assert category_to_class_id("bicycle") == 2
    assert category_to_class_id("bike") == 2
    assert category_to_class_id("motorcycle") == 2
    assert category_to_class_id("motor") == 2


def test_category_map_dropped_and_unknown():
    assert category_to_class_id("traffic light") is None
    assert category_to_class_id("traffic sign") is None
    assert category_to_class_id("lane") is None
    assert category_to_class_id("banana") is None


def test_box2d_to_yolo_full_frame():
    box = {"x1": 0, "y1": 0, "x2": 1280, "y2": 720}
    assert box2d_to_yolo(box, 1280, 720) == (0.5, 0.5, 1.0, 1.0)


def test_box2d_to_yolo_quadrant():
    box = {"x1": 0, "y1": 0, "x2": 640, "y2": 360}
    assert box2d_to_yolo(box, 1280, 720) == (0.25, 0.25, 0.5, 0.5)


def test_box2d_to_yolo_clamps_overflow():
    box = {"x1": -10, "y1": -10, "x2": 1300, "y2": 740}
    assert box2d_to_yolo(box, 1280, 720) == (0.5, 0.5, 1.0, 1.0)


def test_box2d_to_yolo_degenerate_returns_none():
    assert box2d_to_yolo({"x1": 5, "y1": 5, "x2": 5, "y2": 9}, 1280, 720) is None


def test_convert_frame_filters_drops_and_keeps_attrs():
    entry = {
        "name": "x.jpg",
        "attributes": {
            "weather": "clear",
            "scene": "city street",
            "timeofday": "daytime",
        },
        "labels": [
            {"category": "car", "box2d": {"x1": 0, "y1": 0, "x2": 640, "y2": 360}},
            {
                "category": "traffic sign",
                "box2d": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            },
            {"category": "lane", "poly2d": [[0, 0]]},  # no box2d
        ],
    }
    fl = convert_frame(entry)
    assert isinstance(fl, FrameLabels)
    assert fl.name == "x.jpg"
    assert fl.attributes["timeofday"] == "daytime"
    assert len(fl.yolo_lines) == 1
    assert fl.yolo_lines[0] == "0 0.250000 0.250000 0.500000 0.500000"


def test_convert_frame_empty_labels():
    fl = convert_frame({"name": "y.jpg", "labels": []})
    assert fl.yolo_lines == []
    assert fl.attributes == {}
