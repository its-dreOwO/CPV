"""Demo pipeline: run detection + tracking + avoidance on a video source."""

import argparse
import random
from pathlib import Path

import cv2
import torch

from src.detection.yolo_detector import YoloDetector
from src.tracking.kalman_tracker import KalmanTracker
from src.avoidance.geometric_planner import GeometricPlanner
from src.utils.visualizer import draw_tracks

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Drone obstacle avoidance demo")
    parser.add_argument(
        "--source", type=str, default="0", help="Video file, camera index, or image directory"
    )
    parser.add_argument(
        "--weights", type=str, default="yolov8m.pt", help="Model weights path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda/cpu)",
    )
    parser.add_argument(
        "--delay", type=int, default=100, help="Ms between frames in image-folder mode"
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Directory to save output frames"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Stop after N frames"
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle images in directory mode"
    )
    return parser.parse_args()


def iter_frames_from_dir(folder: Path, shuffle: bool = False, max_frames: int = None):
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")
    if shuffle:
        random.shuffle(paths)
    if max_frames:
        paths = paths[:max_frames]
    for p in paths:
        frame = cv2.imread(str(p))
        if frame is not None:
            yield p.name, frame


def run_pipeline(
    frames,
    detector,
    tracker,
    planner,
    delay_ms: int,
    reset_per_frame: bool = False,
    save_dir: Path = None,
    video_out: cv2.VideoWriter = None,
):
    print("Starting pipeline. Press 'q' to quit, any key to advance.")
    for name, frame in frames:
        if reset_per_frame:
            tracker.reset()
        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        avoidance_cmd = planner.plan(tracks)
        vis = draw_tracks(frame, tracks, avoidance_cmd)

        if video_out:
            video_out.write(vis)
        elif save_dir:
            out_path = save_dir / name
            cv2.imwrite(str(out_path), vis)
            print(f"Saved {out_path}")

        cv2.imshow("Drone Obstacle Avoidance", vis)
        if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    source_path = Path(args.source)
    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    if source_path.is_dir():
        all_frames = list(
            iter_frames_from_dir(
                source_path, shuffle=args.shuffle, max_frames=args.max_frames
            )
        )
        _, first = all_frames[0]
        h, w = first.shape[:2]
        frames = iter(all_frames)
    else:
        source = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open source {source}")
            return
        ret, first = cap.read()
        if not ret:
            print("Error: Could not read first frame")
            cap.release()
            return
        h, w = first.shape[:2]
        count = [0]

        def _cap_frames():
            yield "frame_0000.jpg", first
            while True:
                if args.max_frames and count[0] >= args.max_frames:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                count[0] += 1
                yield f"frame_{count[0]:04d}.jpg", frame

        frames = _cap_frames()

    print(f"Initializing Detector with {args.weights} on {args.device}...")
    detector = YoloDetector(model_path=args.weights)
    tracker = KalmanTracker()
    planner = GeometricPlanner(frame_width=w, frame_height=h)

    video_out = None
    if save_dir and not source_path.is_dir():
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        out_name = source_path.stem + "_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_out = cv2.VideoWriter(str(save_dir / out_name), fourcc, fps, (w, h))
        print(f"Saving annotated video to {save_dir / out_name}")

    try:
        run_pipeline(
            frames,
            detector,
            tracker,
            planner,
            delay_ms=args.delay,
            reset_per_frame=source_path.is_dir(),
            save_dir=save_dir,
            video_out=video_out,
        )
    finally:
        if not source_path.is_dir():
            cap.release()
        if video_out:
            video_out.release()


if __name__ == "__main__":
    main()
