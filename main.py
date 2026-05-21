"""Demo pipeline: run detection + tracking on a video source."""
import argparse

import cv2

from src.utils.visualizer import draw_tracks


def parse_args():
    parser = argparse.ArgumentParser(description="Drone obstacle avoidance demo")
    parser.add_argument("--source", type=str, default="0", help="Video file or camera index")
    parser.add_argument("--weights", type=str, default="", help="Model weights path")
    return parser.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # TODO: plug in detector and tracker here
        tracks = []
        vis = draw_tracks(frame, tracks)
        cv2.imshow("Drone Obstacle Avoidance", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
