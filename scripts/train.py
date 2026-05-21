"""Entry point for model training."""
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train obstacle detection model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Training with config={args.config}, epochs={args.epochs}, device={args.device}")


if __name__ == "__main__":
    main()
