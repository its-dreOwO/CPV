from pathlib import Path

import cv2
import numpy as np
import pytest

from src.utils.data_validation import (
    _parse_yolo_line,
    validate_dataset,
    validate_videos,
)


def _write_image(path: Path, w: int = 32, h: int = 32) -> None:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_parse_yolo_line_valid():
    assert _parse_yolo_line("0 0.5 0.5 0.2 0.3", num_classes=3) is None


def test_parse_yolo_line_field_count():
    assert "5 fields" in _parse_yolo_line("0 0.5 0.5 0.2", num_classes=None)


def test_parse_yolo_line_class_range():
    err = _parse_yolo_line("9 0.5 0.5 0.2 0.3", num_classes=3)
    assert "out of range" in err


def test_parse_yolo_line_coord_out_of_range():
    err = _parse_yolo_line("0 1.5 0.5 0.2 0.3", num_classes=None)
    assert "x_center" in err


def test_parse_yolo_line_zero_dim():
    err = _parse_yolo_line("0 0.5 0.5 0.0 0.3", num_classes=None)
    assert "width" in err


def test_validate_dataset_happy_path(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()

    _write_image(images / "a.jpg")
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.3\n")

    report = validate_dataset(images, labels, num_classes=1)
    assert report.ok
    assert report.total_images == 1
    assert report.total_labels == 1
    assert report.class_counts[0] == 1


def test_validate_dataset_missing_label(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    _write_image(images / "a.jpg")

    report = validate_dataset(images, labels)
    assert not report.ok
    assert len(report.missing_labels) == 1


def test_validate_videos_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_videos(tmp_path / "nope")


def test_validate_videos_empty_dir(tmp_path):
    report = validate_videos(tmp_path)
    assert report.ok
    assert report.total_videos == 0
