"""Entry point for model evaluation."""
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate obstacle detection model")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--data", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Evaluating weights={args.weights} on data={args.data}, device={args.device}")


if __name__ == "__main__":
    main()
