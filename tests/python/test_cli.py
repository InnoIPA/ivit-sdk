"""The `ivit` command line runs end to end (needs the built bindings)."""

import json
import sys

import numpy as np
import pytest

import ivit
from ivit import cli

pytestmark = pytest.mark.skipif(
    not ivit.is_cpp_available(), reason="C++ bindings are not built"
)


def run(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["ivit", *argv])
    code = 0
    try:
        cli.main()
    except SystemExit as exit:
        code = exit.code or 0
    out = capsys.readouterr()
    return code, out.out, out.err


@pytest.fixture
def classifier_file(tmp_path):
    from test_model_contract import model_file

    return str(model_file(tmp_path / "cls.onnx", [1, 3, 8, 8], [[0.1, 0.7, 0.2]]))


@pytest.fixture
def image_file(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "image.png"
    cv2.imwrite(str(path), np.full((8, 8, 3), 128, np.uint8))
    return str(path)


def test_devices_table_and_json(monkeypatch, capsys):
    code, out, _ = run(monkeypatch, capsys, "devices")
    assert code == 0 and "Total:" in out and "Status" in out
    code, out, _ = run(monkeypatch, capsys, "devices", "--json")
    assert code == 0
    data = json.loads(out)
    assert data and all(isinstance(d["available"], bool) for d in data)


def test_benchmark_runs_a_model_file(monkeypatch, capsys, classifier_file):
    code, out, _ = run(
        monkeypatch, capsys, "benchmark", classifier_file, "-d", "cpu", "-n", "2", "-w", "1",
        "--task", "classify",
    )
    assert code == 0 and "FPS:" in out and "Input shape: [1, 3, 8, 8]" in out


def test_predict_decodes_with_task_and_saves_the_drawing(
    monkeypatch, capsys, tmp_path, classifier_file, image_file
):
    monkeypatch.chdir(tmp_path)
    code, out, _ = run(
        monkeypatch, capsys, "predict", classifier_file, image_file, "-d", "cpu",
        "--task", "classify", "--save",
    )
    assert code == 0 and "top-1" in out
    assert (tmp_path / "image_result.png").is_file()


def test_predict_without_task_says_nothing_is_decoded(
    monkeypatch, capsys, classifier_file, image_file
):
    code, out, _ = run(monkeypatch, capsys, "predict", classifier_file, image_file, "-d", "cpu")
    assert code == 0 and "no decoded results" in out


def test_export_writes_the_requested_output_path(monkeypatch, capsys, tmp_path, classifier_file):
    target = tmp_path / "out" / "renamed.xml"
    code, _, _ = run(monkeypatch, capsys, "export", classifier_file, "-o", str(target),
                     "-t", "intel_cpu")
    assert code == 0 and target.is_file()


def test_errors_exit_non_zero_without_a_traceback(monkeypatch, capsys, image_file):
    code, _, err = run(monkeypatch, capsys, "predict", "missing.onnx", image_file)
    assert code == 1 and "Model not found" in err and "Traceback" not in err
    code, _, err = run(monkeypatch, capsys, "zoo", "info", "no-such-model")
    assert code == 1 and "no-such-model" in err
    code, _, _ = run(monkeypatch, capsys, "zoo")
    assert code == 1


@pytest.fixture
def torch_files(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
    model.eval()
    files = {
        "script": tmp_path / "script.pt",
        "pickle": tmp_path / "pickle.pt",
        "weights": tmp_path / "weights.pth",
        "junk": tmp_path / "junk.pt",
    }
    torch.jit.script(model).save(str(files["script"]))
    torch.save(model, str(files["pickle"]))
    torch.save(model.state_dict(), str(files["weights"]))
    files["junk"].write_bytes(b"not a model")
    return model, files


def test_convert_exports_torchscript_to_onnx_with_the_requested_size(
    monkeypatch, capsys, tmp_path, torch_files
):
    import torch

    model, files = torch_files
    out_dir = tmp_path / "out"
    code, out, err = run(
        monkeypatch, capsys, "convert", str(files["script"]), "-f", "onnx", "-o", str(out_dir),
        "--input-size", "24,32",
    )
    assert code == 0, err
    onnx_path = out_dir / "script.onnx"
    assert onnx_path.exists() and "1x3x24x32" in out

    ov = pytest.importorskip("openvino")
    x = torch.rand(1, 3, 24, 32)
    got = ov.Core().compile_model(str(onnx_path), "CPU")(x.numpy())[0]
    assert abs(got - model(x).detach().numpy()).max() < 1e-4


def test_convert_pickled_model_needs_trust_pickle(monkeypatch, capsys, tmp_path, torch_files):
    _, files = torch_files
    out_dir = tmp_path / "out"
    code, _, err = run(monkeypatch, capsys, "convert", str(files["pickle"]), "-f", "onnx", "-o", str(out_dir))
    assert code == 1 and "--trust-pickle" in err and "Traceback" not in err
    assert not (out_dir / "pickle.onnx").exists()
    code, _, err = run(
        monkeypatch, capsys, "convert", str(files["pickle"]), "-f", "onnx", "-o", str(out_dir),
        "--trust-pickle", "--input-size", "16",
    )
    assert code == 0, err
    assert (out_dir / "pickle.onnx").exists()


def test_convert_pt_to_openvino_goes_through_onnx(monkeypatch, capsys, tmp_path, torch_files):
    _, files = torch_files
    out_dir = tmp_path / "ov"
    code, _, err = run(
        monkeypatch, capsys, "convert", str(files["script"]), "-f", "openvino", "-o", str(out_dir),
        "--input-size", "16",
    )
    assert code == 0, err
    assert (out_dir / "script.onnx").exists() and (out_dir / "script.xml").exists()


@pytest.mark.parametrize(
    "kind, args, message",
    [
        ("weights", [], "state dict"),
        ("junk", [], "not a TorchScript archive"),
        ("script", ["--input-size", "0"], "--input-size"),
    ],
)
def test_convert_rejects_unusable_pt_input_cleanly(
    monkeypatch, capsys, tmp_path, torch_files, kind, args, message
):
    _, files = torch_files
    code, _, err = run(
        monkeypatch, capsys, "convert", str(files[kind]), "-f", "onnx", "-o", str(tmp_path / "out"), *args
    )
    assert code == 1 and message in err and "Traceback" not in err


def test_convert_pt_to_tensorrt_builds_an_engine_that_matches_torch(
    monkeypatch, capsys, tmp_path, torch_files
):
    import torch

    if not any(device.backend == "tensorrt" for device in ivit.list_devices()):
        pytest.skip("TensorRT GPU required")
    model, files = torch_files
    out_dir = tmp_path / "trt"
    code, _, err = run(
        monkeypatch, capsys, "convert", str(files["script"]), "-f", "tensorrt", "-o", str(out_dir),
        "--input-size", "24,32",
    )
    assert code == 0, err
    engine = out_dir / "script.engine"
    assert (out_dir / "script.onnx").exists() and engine.exists()

    loaded = ivit.load(
        str(engine), backend="tensorrt", device="cuda:0", task="classification", use_cache=False
    )
    assert list(loaded.input_info[0].shape) == [1, 3, 24, 32]
    options = ivit.InferConfig()
    options.conf_threshold = 0
    actual = loaded.predict(np.zeros((24, 32, 3), np.uint8), options)
    with torch.no_grad():
        expected = torch.softmax(model(torch.zeros(1, 3, 24, 32)), dim=1)[0].numpy()
    scores = {item.class_id: item.score for item in actual.classifications}
    np.testing.assert_allclose([scores[i] for i in range(4)], expected, atol=1e-4)


def test_info_reports_the_backends_the_sdk_loaded_not_pip_packages(monkeypatch, capsys):
    from ivit import _ivit_core

    versions = _ivit_core._backend_versions()
    assert versions  # the CI builds always have at least OpenVINO CPU
    # A pip "tensorrt"/"openvino" package must not decide what is shown.
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    monkeypatch.setitem(sys.modules, "openvino", None)
    code, out, _ = run(monkeypatch, capsys, "info")
    assert code == 0
    for key, label in (("openvino", "OpenVINO"), ("tensorrt", "TensorRT")):
        line = next(line for line in out.splitlines() if line.strip().startswith(label + ":"))
        if key in versions:
            assert versions[key] in line
        else:
            assert "Not available" in line
