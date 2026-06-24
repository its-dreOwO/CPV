import importlib.util
from pathlib import Path


def _load_modal_train():
    """Import modal_train.py without requiring the `modal` package at import time."""
    import sys
    import types

    if "modal" not in sys.modules:
        stub = types.ModuleType("modal")
        stub.App = lambda *a, **k: types.SimpleNamespace(
            function=lambda *fa, **fk: (lambda f: f),
            local_entrypoint=lambda *fa, **fk: (lambda f: f),
        )

        class _Img:
            def __getattr__(self, _):
                return lambda *a, **k: self

        stub.Image = types.SimpleNamespace(debian_slim=lambda *a, **k: _Img())
        stub.Volume = types.SimpleNamespace(from_name=lambda *a, **k: object())
        sys.modules["modal"] = stub

    path = Path(__file__).resolve().parents[1] / "modal_train.py"
    spec = importlib.util.spec_from_file_location("modal_train", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_train_cmd_points_data_root_at_dataset_subdir():
    mt = _load_modal_train()
    cmd = mt.build_train_cmd(
        model="yolov8n",
        epochs=50,
        dataset_path="/vol/processed/bdd100k",
        run_dir="/vol/runs",
        resume=False,
    )
    assert "--data-root" in cmd
    assert cmd[cmd.index("--data-root") + 1] == "/vol/processed/bdd100k"
    assert "--config" in cmd
    assert cmd[cmd.index("--config") + 1] == "/app/configs/yolov8n.yaml"
    assert "--resume" not in cmd


def test_build_train_cmd_appends_resume_flag():
    mt = _load_modal_train()
    cmd = mt.build_train_cmd("yolov8m", 50, "/vol/processed/bdd100k", "/vol/runs", True)
    assert cmd[-1] == "--resume"
