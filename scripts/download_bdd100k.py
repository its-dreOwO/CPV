"""Download BDD100K (images + detection labels) from a Kaggle mirror.

Prerequisite: Kaggle API auth configured (``kaggle config view`` shows your
username). Create a token at https://www.kaggle.com/settings if needed.

Default mirror: solesensei/solesensei_bdd100k — ships
``images/100k/{train,val,test}/`` and
``labels/bdd100k_labels_images_{train,val}.json`` (with weather/scene/timeofday
attributes). The Kaggle CLI unzips into --dest.

Usage::

    python scripts/download_bdd100k.py
    python scripts/download_bdd100k.py --dataset owner/other-bdd100k
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Download BDD100K from a Kaggle mirror")
    p.add_argument(
        "--dataset",
        default="solesensei/solesensei_bdd100k",
        help="Kaggle dataset slug (default: the canonical solesensei mirror)",
    )
    p.add_argument("--dest", type=Path, default=Path("data/raw"))
    return p.parse_args()


def main():
    args = parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        args.dataset,
        "-p",
        str(args.dest),
        "--unzip",
    ]
    print("Running:", " ".join(cmd))
    # No pre-flight credential check: auth may be a kaggle.json, an env var,
    # or an ACCESS_TOKEN under ~/.kaggle. The kaggle CLI emits its own clear
    # error if credentials are missing or invalid.
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
