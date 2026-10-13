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
