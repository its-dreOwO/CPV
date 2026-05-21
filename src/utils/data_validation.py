"""Pre-training dataset integrity checks.

Verifies image readability, YOLO-format label validity, image/label pairing,
class-id ranges, and optionally MD5-based duplicate detection. Use directly
or via ``scripts/validate_data.py``.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass
class ValidationReport:
    total_images: int = 0
    total_labels: int = 0
    total_videos: int = 0
    unreadable_images: List[Path] = field(default_factory=list)
    missing_labels: List[Path] = field(default_factory=list)
    orphan_labels: List[Path] = field(default_factory=list)
    invalid_labels: List[Tuple[Path, int, str]] = field(default_factory=list)
    duplicates: List[Tuple[Path, Path]] = field(default_factory=list)
    class_counts: Counter = field(default_factory=Counter)
    image_sizes: List[Tuple[int, int]] = field(default_factory=list)
    unreadable_videos: List[Path] = field(default_factory=list)
    videos: List[VideoInfo] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.unreadable_images
            or self.missing_labels
            or self.invalid_labels
            or self.unreadable_videos
        )


def _list_images(images_dir: Path) -> List[Path]:
    return sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _list_videos(videos_dir: Path) -> List[Path]:
    return sorted(p for p in videos_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def _check_image(path: Path) -> Optional[Tuple[int, int]]:
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    if w <= 0 or h <= 0:
        return None
    return w, h


def _check_video(path: Path) -> Optional[VideoInfo]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
            return None
        ok, _ = cap.read()
        if not ok:
            return None
        return VideoInfo(
            path=path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
        )
    finally:
        cap.release()


def validate_videos(videos_dir: Path) -> ValidationReport:
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"videos_dir not found: {videos_dir}")
    report = ValidationReport()
    for vid_path in _list_videos(videos_dir):
        report.total_videos += 1
        info = _check_video(vid_path)
        if info is None:
            report.unreadable_videos.append(vid_path)
        else:
            report.videos.append(info)
    return report


def _parse_yolo_line(line: str, num_classes: Optional[int]) -> Optional[str]:
    parts = line.strip().split()
    if not parts:
        return "empty line"
    if len(parts) != 5:
        return f"expected 5 fields, got {len(parts)}"
    try:
        cls = int(parts[0])
        x, y, w, h = (float(p) for p in parts[1:])
    except ValueError as e:
        return f"non-numeric value: {e}"
    if num_classes is not None and not (0 <= cls < num_classes):
        return f"class_id {cls} out of range [0, {num_classes})"
    for name, v in (("x_center", x), ("y_center", y)):
        if not (0.0 <= v <= 1.0):
            return f"{name}={v} out of [0, 1]"
    for name, v in (("width", w), ("height", h)):
        if not (0.0 < v <= 1.0):
            return f"{name}={v} not in (0, 1]"
    return None


def _hash_image(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def validate_dataset(
    images_dir: Path,
    labels_dir: Path,
    num_classes: Optional[int] = None,
    check_duplicates: bool = False,
) -> ValidationReport:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"labels_dir not found: {labels_dir}")

    report = ValidationReport()
    images = _list_images(images_dir)
    report.total_images = len(images)

    label_stems = {p.stem for p in labels_dir.rglob("*.txt")}
    image_stems = {p.stem for p in images}
    report.orphan_labels = sorted(
        labels_dir / f"{s}.txt" for s in (label_stems - image_stems)
    )

    hashes: dict = {}
    for img_path in images:
        size = _check_image(img_path)
        if size is None:
            report.unreadable_images.append(img_path)
            continue
        report.image_sizes.append(size)

        if check_duplicates:
            h = _hash_image(img_path)
            if h in hashes:
                report.duplicates.append((hashes[h], img_path))
            else:
                hashes[h] = img_path

        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            report.missing_labels.append(img_path)
            continue
        report.total_labels += 1

        for lineno, line in enumerate(label_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            err = _parse_yolo_line(line, num_classes)
            if err:
                report.invalid_labels.append((label_path, lineno, err))
            else:
                report.class_counts[int(line.split()[0])] += 1

    return report


def print_report(report: ValidationReport) -> None:
    print(f"Images scanned: {report.total_images}")
    print(f"Labels found:   {report.total_labels}")
    print(f"Videos scanned: {report.total_videos}")

    def _section(title: str, items: Iterable) -> None:
        items = list(items)
        marker = "OK" if not items else f"FAIL ({len(items)})"
        print(f"[{marker}] {title}")
        for item in items[:10]:
            print(f"  - {item}")
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more")

    _section("Unreadable images", report.unreadable_images)
    _section("Missing labels", report.missing_labels)
    _section("Orphan labels", report.orphan_labels)
    _section(
        "Invalid label lines",
        (f"{p}:{ln} {msg}" for p, ln, msg in report.invalid_labels),
    )
    _section("Duplicate images", report.duplicates)
    _section("Unreadable videos", report.unreadable_videos)

    if report.class_counts:
        print("\nClass distribution:")
        for cls, n in sorted(report.class_counts.items()):
            print(f"  class {cls}: {n}")

    if report.image_sizes:
        widths = [w for w, _ in report.image_sizes]
        heights = [h for _, h in report.image_sizes]
        print("\nImage size stats:")
        print(
            f"  width:  min={min(widths)}, max={max(widths)}, "
            f"mean={np.mean(widths):.0f}"
        )
        print(
            f"  height: min={min(heights)}, max={max(heights)}, "
            f"mean={np.mean(heights):.0f}"
        )

    if report.videos:
        widths = [v.width for v in report.videos]
        heights = [v.height for v in report.videos]
        fps_list = [v.fps for v in report.videos]
        frames = [v.frame_count for v in report.videos]
        print("\nVideo stats:")
        print(
            f"  resolution: {min(widths)}x{min(heights)} - {max(widths)}x{max(heights)}"
        )
        print(f"  fps:        min={min(fps_list):.1f}, max={max(fps_list):.1f}")
        print(
            f"  frames:     min={min(frames)}, max={max(frames)}, total={sum(frames)}"
        )

    print(f"\nResult: {'PASS' if report.ok else 'FAIL'}")
