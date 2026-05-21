"""Validate dataset integrity before training.

Examples
--------
Validate an image dataset with YOLO labels::

    python scripts/validate_data.py --images data/processed/train/images \\
        --labels data/processed/train/labels --num-classes 10

Validate a video directory::

    python scripts/validate_data.py --videos data/raw/visdrone-mot/sequences

Both together (merged report)::

    python scripts/validate_data.py --images ... --labels ... --videos ...
"""

import argparse
import sys
from pathlib import Path

from src.utils.data_validation import (
    ValidationReport,
    print_report,
    validate_dataset,
    validate_videos,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate dataset before training")
    parser.add_argument("--images", type=Path, help="Path to images dir")
    parser.add_argument("--labels", type=Path, help="Path to YOLO labels dir")
    parser.add_argument("--videos", type=Path, help="Path to videos dir")
    parser.add_argument(
        "--num-classes", type=int, default=None, help="Expected class count"
    )
    parser.add_argument(
        "--check-duplicates",
        action="store_true",
        help="MD5-hash images to detect duplicates (slow on large datasets)",
    )
    return parser.parse_args()


def _merge(into: ValidationReport, other: ValidationReport) -> ValidationReport:
    into.total_images += other.total_images
    into.total_labels += other.total_labels
    into.total_videos += other.total_videos
    into.unreadable_images.extend(other.unreadable_images)
    into.missing_labels.extend(other.missing_labels)
    into.orphan_labels.extend(other.orphan_labels)
    into.invalid_labels.extend(other.invalid_labels)
    into.duplicates.extend(other.duplicates)
    into.class_counts.update(other.class_counts)
    into.image_sizes.extend(other.image_sizes)
    into.unreadable_videos.extend(other.unreadable_videos)
    into.videos.extend(other.videos)
    return into


def main():
    args = parse_args()

    if not any([args.images, args.videos]):
        sys.exit("error: provide --images (with --labels) and/or --videos")

    report = ValidationReport()

    if args.images:
        if not args.labels:
            sys.exit("error: --labels is required when --images is provided")
        report = _merge(
            report,
            validate_dataset(
                images_dir=args.images,
                labels_dir=args.labels,
                num_classes=args.num_classes,
                check_duplicates=args.check_duplicates,
            ),
        )

    if args.videos:
        report = _merge(report, validate_videos(videos_dir=args.videos))

    print_report(report)
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
