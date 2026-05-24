"""Demo pipeline: run detection + tracking + avoidance on a video source."""

import argparse
import cv2
import torch

from src.detection.yolo_detector import YoloDetector
from src.tracking.kalman_tracker import KalmanTracker
from src.avoidance.geometric_planner import GeometricPlanner
from src.utils.visualizer import draw_tracks


def parse_args():
    parser = argparse.ArgumentParser(description="Drone obstacle avoidance demo")
    parser.add_argument(
        "--source", type=str, default="0", help="Video file or camera index"
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
    return parser.parse_args()


def main():
    args = parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open source {source}")
        return

    # Read first frame to get dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        cap.release()
        return
    h, w = frame.shape[:2]

    # Initialize components
    print(f"Initializing Detector with {args.weights} on {args.device}...")
    detector = YoloDetector(model_path=args.weights)
    tracker = KalmanTracker()
    planner = GeometricPlanner(frame_width=w, frame_height=h)

    print("Starting pipeline. Press 'q' to quit.")

    # We already read the first frame, so process it then continue loop
    while True:
        # 1. Detection
        detections = detector.detect(frame)

        # 2. Tracking
        tracks = tracker.update(detections)

        # 3. Avoidance Planning
        avoidance_cmd = planner.plan(tracks)

        # 4. Visualization
        vis = draw_tracks(frame, tracks, avoidance_cmd)

        cv2.imshow("Drone Obstacle Avoidance", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        ret, frame = cap.read()
        if not ret:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
