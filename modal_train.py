"""Modal training script — CPV vehicle-perception (dashcam obstacle detection).

Quick start
-----------
1. Install Modal and authenticate (one-time):
       pip install modal
       modal setup

2. Upload the dataset — tar first so it's one file, not 8000 (one-time):
       tar czf processed.tar.gz -C data processed
       modal volume create cpv-bdd100k
       modal volume put cpv-bdd100k processed.tar.gz /processed.tar.gz
       modal run modal_train.py::extract_dataset
       rm processed.tar.gz   # optional cleanup
   After extraction the dataset is at /vol/processed/bdd100k.

3. Sanity check — 5 epochs on YOLOv8n (~15 min, ~$0.15 on L4):
       modal run modal_train.py::main --model yolov8n --epochs 5

4. Full training runs (run independently, ~$1–2 each on L4):
       modal run modal_train.py::main --model yolov8n --epochs 50
       modal run modal_train.py::main --model yolov8m --epochs 50
       modal run modal_train.py::main --model rtdetr  --epochs 50

5. Download a trained model:
       modal run modal_train.py::fetch --model yolov8m
       # saves to models/yolov8m-best.pt

GPU cost guide (Modal pay-as-you-go):
    L4   $0.80/hr  24 GB VRAM  — default, best value for these models
    A10G $1.10/hr  24 GB VRAM  — fallback if L4 unavailable
    T4   $0.59/hr  16 GB VRAM  -- budget option; RT-DETR may OOM at batch 8

Storage: Modal volumes cost ~$0.20/GB/month (~$0.40/mo for this 1.9 GB dataset).
"""

import subprocess
from pathlib import Path

import modal

APP_NAME = "cpv-vehicle-perception"
VOLUME_NAME = "cpv-bdd100k"
VOLUME_PATH = Path("/vol")
# Dataset is extracted to /vol/processed/bdd100k (the dir holding train/ val/ test/).
DATASET_SUBDIR = "processed/bdd100k"


def build_train_cmd(model, epochs, dataset_path, run_dir, resume):
    """Argv for the in-container scripts/train.py call. Pure + unit-testable."""
    cmd = [
        "python",
        "/app/scripts/train.py",
        "--config",
        f"/app/configs/{model}.yaml",
        "--epochs",
        str(epochs),
        "--device",
        "0",
        "--data-root",
        str(dataset_path),
        "--project",
        str(run_dir),
    ]
    if resume:
        cmd.append("--resume")
    return cmd


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Pretrained weights are cached in the image so cold starts don't re-download them.
# configs/ and scripts/ are baked in so they're available without a separate mount.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("ultralytics>=8.2.0", "pyyaml>=6.0.1")
    .env({"WANDB_DISABLED": "true", "YOLO_VERBOSE": "True", "PYTHONUNBUFFERED": "1"})
    .run_commands(
        'python -c "'
        "from ultralytics import YOLO; "
        "[YOLO(w) for w in ['yolov8n.pt','yolov8m.pt','rtdetr-l.pt']]\""
    )
    .add_local_dir("configs", remote_path="/app/configs")
    .add_local_dir("scripts", remote_path="/app/scripts")
)

_VALID_MODELS = ("yolov8n", "yolov8m", "rtdetr")


@app.function(
    image=image,
    gpu="L4",
    volumes={VOLUME_PATH: volume},
    timeout=24 * 60 * 60,
    retries=2,
)
def train(model: str, epochs: int, fresh: bool = False) -> None:
    import os
    import shutil

    dataset_path = VOLUME_PATH / DATASET_SUBDIR
    if not dataset_path.exists():
        raise RuntimeError(
            "Dataset not found in volume. Run:\n"
            "  tar czf processed.tar.gz -C data processed\n"
            f"  modal volume put {VOLUME_NAME} processed.tar.gz /processed.tar.gz\n"
            "  modal run modal_train.py::extract_dataset"
        )

    run_dir = VOLUME_PATH / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    model_run_dir = run_dir / model
    last_pt = model_run_dir / "weights/last.pt"

    if fresh and model_run_dir.exists():
        shutil.rmtree(model_run_dir)
        print(f"--fresh: cleared {model_run_dir}")

    if not fresh and last_pt.exists():
        print(f"Found checkpoint at {last_pt} — resuming.")
    else:
        if model_run_dir.exists():
            shutil.rmtree(model_run_dir)
            print(f"Cleared previous run at {model_run_dir}")

    resume = (not fresh) and last_pt.exists()
    cmd = build_train_cmd(model, epochs, dataset_path, run_dir, resume)
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd="/app", env={**os.environ}, bufsize=1)
    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Training exited with code {result.returncode}")

    best = run_dir / model / "weights/best.pt"
    print(f"Done. Best weights at {best} (in volume)")


@app.function(image=image, gpu="L4", timeout=60)
def gpu_info() -> None:
    """Print GPU name, VRAM, and CUDA version."""
    import subprocess

    import torch

    print(f"GPU:        {torch.cuda.get_device_name(0)}")
    print(
        f"VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )
    print(f"CUDA:       {torch.version.cuda}")
    subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        check=True,
    )


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={VOLUME_PATH: volume},
    timeout=10 * 60,
)
def extract_dataset() -> None:
    """Extract /vol/processed.tar.gz → /vol/processed/ inside the volume."""
    import tarfile

    archive = VOLUME_PATH / "processed.tar.gz"
    if not archive.exists():
        raise RuntimeError(
            f"{archive} not found. Upload it first:\n"
            "  tar czf processed.tar.gz -C data processed\n"
            f"  modal volume put {VOLUME_NAME} processed.tar.gz /processed.tar.gz"
        )
    print(f"Extracting {archive} …")
    with tarfile.open(archive) as tar:
        tar.extractall(path=VOLUME_PATH)
    n = sum(1 for p in (VOLUME_PATH / "processed").rglob("*") if p.is_file())
    print(f"Done — {n} files extracted to {VOLUME_PATH / 'processed'}")
    volume.commit()


@app.local_entrypoint()
def main(model: str = "yolov8m", epochs: int = 50, fresh: bool = False) -> None:
    if model not in _VALID_MODELS:
        raise SystemExit(f"Unknown model '{model}'. Choose from: {_VALID_MODELS}")
    print(f"Launching {model} for {epochs} epochs on Modal L4 …")
    train.remote(model=model, epochs=epochs, fresh=fresh)


@app.local_entrypoint()
def fetch(model: str = "yolov8m") -> None:
    """Download best.pt for a trained model to models/<model>-best.pt."""
    if model not in _VALID_MODELS:
        raise SystemExit(f"Unknown model '{model}'. Choose from: {_VALID_MODELS}")
    out = f"models/{model}-best.pt"
    Path("models").mkdir(exist_ok=True)
    subprocess.run(
        ["modal", "volume", "get", VOLUME_NAME, f"/runs/{model}/weights/best.pt", out],
        check=True,
    )
    print(f"Saved to {out}")
