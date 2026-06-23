import json as _json
from pathlib import Path

import cv2
import numpy as np

from src.utils.bdd100k import (
    CLASS_NAMES,
    FrameLabels,
    box2d_to_yolo,
    category_to_class_id,
    collect_frames,
    convert_frame,
    dominant_class,
    stratified_train_val,
    write_attributes,
    write_split,
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


def _write_img(path, w=16, h=16):
    cv2.imwrite(str(path), np.zeros((h, w, 3), dtype=np.uint8))


def _raw_tree(tmp_path):
    imgs = tmp_path / "images"
    imgs.mkdir()
    _write_img(imgs / "a.jpg")
    _write_img(imgs / "b.jpg")
    labels = [
        {
            "name": "a.jpg",
            "attributes": {
                "weather": "clear",
                "scene": "city street",
                "timeofday": "daytime",
            },
            "labels": [
                {"category": "car", "box2d": {"x1": 0, "y1": 0, "x2": 640, "y2": 360}}
            ],
        },
        {
            "name": "b.jpg",
            "attributes": {
                "weather": "rainy",
                "scene": "highway",
                "timeofday": "night",
            },
            "labels": [
                {
                    "category": "pedestrian",
                    "box2d": {"x1": 10, "y1": 10, "x2": 50, "y2": 200},
                }
            ],
        },
    ]
    js = tmp_path / "det.json"
    js.write_text(_json.dumps(labels))
    return imgs, js


def test_collect_frames_pairs_existing_images(tmp_path):
    imgs, js = _raw_tree(tmp_path)
    pairs = collect_frames(imgs, js)
    assert {p.name for p, _ in pairs} == {"a.jpg", "b.jpg"}


def test_collect_frames_skips_missing_image(tmp_path):
    imgs, js = _raw_tree(tmp_path)
    (imgs / "b.jpg").unlink()
    pairs = collect_frames(imgs, js)
    assert {p.name for p, _ in pairs} == {"a.jpg"}


def test_dominant_class():
    fl = FrameLabels(
        name="x",
        yolo_lines=[
            "0 0.5 0.5 0.2 0.2",
            "0 0.1 0.1 0.1 0.1",
            "1 0.5 0.5 0.1 0.1",
        ],
    )
    assert dominant_class(fl) == 0
    assert dominant_class(FrameLabels(name="y")) == -1


def test_write_split_materializes_images_and_labels(tmp_path):
    imgs, js = _raw_tree(tmp_path)
    pairs = collect_frames(imgs, js)
    out = tmp_path / "processed"
    write_split("train", pairs, out, copy=True)
    assert (out / "train" / "images" / "a.jpg").exists()
    label = (out / "train" / "labels" / "a.txt").read_text().strip()
    assert label == "0 0.250000 0.250000 0.500000 0.500000"


def test_write_split_symlinks_by_default(tmp_path):
    imgs, js = _raw_tree(tmp_path)
    pairs = collect_frames(imgs, js)
    out = tmp_path / "processed"
    write_split("val", pairs, out, copy=False)
    assert (out / "val" / "images" / "a.jpg").is_symlink()


def test_write_attributes_csv(tmp_path):
    rows = [
        (
            "a.jpg",
            "train",
            {"weather": "clear", "scene": "city street", "timeofday": "daytime"},
        )
    ]
    out = tmp_path / "attributes.csv"
    write_attributes(rows, out)
    text = out.read_text()
    assert text.splitlines()[0] == "name,split,weather,scene,timeofday"
    assert "a.jpg,train,clear,city street,daytime" in text


def test_stratified_train_val_is_proportional_and_seeded():
    pairs = []
    for i in range(5):
        pairs.append(
            (Path(f"c{i}.jpg"), FrameLabels(f"c{i}.jpg", ["0 0.5 0.5 0.1 0.1"]))
        )
    for i in range(5):
        pairs.append(
            (Path(f"p{i}.jpg"), FrameLabels(f"p{i}.jpg", ["1 0.5 0.5 0.1 0.1"]))
        )
    train, val = stratified_train_val(pairs, val_ratio=0.4, seed=42)
    assert len(train) == 6 and len(val) == 4
    # deterministic under a fixed seed
    train2, val2 = stratified_train_val(pairs, val_ratio=0.4, seed=42)
    assert [p.name for p, _ in val] == [p.name for p, _ in val2]
