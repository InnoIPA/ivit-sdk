"""Missing Zoo export dependencies produce an actionable message (no torch needed to run)."""

import sys
from types import SimpleNamespace

import pytest

from ivit.zoo import registry


def test_missing_torch_names_every_zoo_dependency(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError
    with pytest.raises(ImportError) as error:
        registry._export_torchvision_onnx(
            "resnet18", tmp_path / "resnet18.onnx", registry.get_model_info("resnet18")
        )
    message = str(error.value)
    # `pip install torch torchvision` alone left the export failing on onnxscript.
    for package in ("torch", "torchvision", "onnx", "onnxscript", ".[zoo]"):
        assert package in message


@pytest.mark.parametrize("missing", ["onnxscript", "onnx"])
def test_export_missing_onnx_package_is_explained(tmp_path, missing):
    def export(*args, **kwargs):
        raise ModuleNotFoundError(f"No module named '{missing}'", name=missing)

    fake_torch = SimpleNamespace(onnx=SimpleNamespace(export=export))
    with pytest.raises(ImportError, match=f"'{missing}' package.*\\.\\[zoo\\]"):
        registry._export_with_new_exporter(fake_torch, object(), object(), tmp_path / "m.onnx")


def test_other_missing_modules_are_not_rewritten(tmp_path):
    def export(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'somethingelse'", name="somethingelse")

    fake_torch = SimpleNamespace(onnx=SimpleNamespace(export=export))
    with pytest.raises(ModuleNotFoundError):
        registry._export_with_new_exporter(fake_torch, object(), object(), tmp_path / "m.onnx")


def test_hint_recommends_the_small_cpu_build_of_pytorch():
    # the default pip wheel pulls CUDA (several GB) although the export runs once on the CPU
    assert "whl/cpu" in registry._ZOO_EXTRAS_HINT


@pytest.mark.parametrize(
    "example, model",
    [("02_classification.py", "resnet18"), ("02_segmentation.py", "deeplabv3-mobilenetv3")],
)
def test_torchvision_examples_explain_a_missing_pytorch_without_a_traceback(tmp_path, example, model):
    import subprocess

    pytest.importorskip("ivit._ivit_core")
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    runner = "\n".join([
        "import runpy, sys",
        "sys.modules['torch'] = None  # as if PyTorch were not installed",
        "script = sys.argv[1]; sys.argv = sys.argv[1:]",
        "runpy.run_path(script, run_name='__main__')",
    ])
    env = {**__import__("os").environ, "IVIT_CACHE_DIR": str(tmp_path), "HOME": str(tmp_path)}
    done = subprocess.run(
        [sys.executable, "-c", runner, str(root / "examples/python" / example),
         "-m", model, "-i", str(root / "examples/data/images/bus.jpg"), "-d", "cpu"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert done.returncode == 1, done.stdout[-300:] + done.stderr[-300:]
    assert "Traceback" not in done.stderr
    assert "Error:" in done.stderr and "torchvision" in done.stderr and ".[zoo]" in done.stderr
