"""
Regression tests for the P0 correctness fixes.

Two groups:

* ``TestDeviceRanking`` is pure Python and runs everywhere. It feeds
  ``DeviceManager`` a fake C++ module that reports devices the way the real
  bindings do.
* Everything else needs the compiled bindings (``ivit._ivit_core``). Tests that
  run real inference also need the ``onnx`` package and the OpenVINO backend,
  and are skipped when those are missing.
"""

import gc
import sys
import threading
import types
from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ivit_cpp():
    """The ivit package, skipping when the C++ bindings are not built."""
    ivit = pytest.importorskip("ivit")
    if not ivit.is_cpp_available():
        pytest.skip("C++ bindings are not built")
    return ivit


@pytest.fixture(scope="module")
def openvino_ivit(ivit_cpp):
    """ivit with a working OpenVINO backend (needed to run real inference)."""
    pytest.importorskip("numpy")
    pytest.importorskip("onnx")
    if not any(d.backend == "openvino" for d in ivit_cpp.list_devices()):
        pytest.skip("OpenVINO backend is not available")
    return ivit_cpp


def _build_classifier(path, dynamic_batch):
    """Write a tiny Conv -> GlobalAveragePool -> Flatten ONNX classifier (4 classes)."""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper
    import onnx

    rng = np.random.RandomState(0)
    weight = numpy_helper.from_array(rng.randn(4, 3, 3, 3).astype(np.float32), "W")
    bias = numpy_helper.from_array(rng.randn(4).astype(np.float32), "B")

    batch = "N" if dynamic_batch else 1
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["input", "W", "B"], ["c"], pads=[1, 1, 1, 1]),
            helper.make_node("GlobalAveragePool", ["c"], ["g"]),
            helper.make_node("Flatten", ["g"], ["logits"], axis=1),
        ],
        "tiny_classifier",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch, 3, 8, 8])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [batch, 4])],
        initializer=[weight, bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture(scope="module")
def static_model_path(openvino_ivit, tmp_path_factory):
    return _build_classifier(tmp_path_factory.mktemp("models") / "static.onnx", False)


@pytest.fixture(scope="module")
def dynamic_model_path(openvino_ivit, tmp_path_factory):
    return _build_classifier(tmp_path_factory.mktemp("models") / "dynamic.onnx", True)


@pytest.fixture
def images():
    import numpy as np

    rng = np.random.RandomState(1)
    return [rng.randint(0, 255, (8, 8, 3), dtype=np.uint8) for _ in range(8)]


def _load(ivit, path, **kwargs):
    kwargs.setdefault("device", "cpu")
    kwargs.setdefault("task", "classification")
    kwargs.setdefault("use_cache", False)
    return ivit.load(path, **kwargs)


def _top1(result):
    top = result.top1()
    return top.class_id, top.score


def _assert_same(a, b):
    assert a[0] == b[0]
    assert a[1] == pytest.approx(b[1], abs=1e-4)


def test_npu_rejects_unbounded_shape_without_crashing(
    openvino_ivit, dynamic_model_path, static_model_path
):
    import subprocess

    if not any(
        d.id == "npu" and d.backend == "openvino" for d in openvino_ivit.list_devices()
    ):
        pytest.skip("OpenVINO NPU is not available")
    # Isolate the native compiler: a regression must fail, not kill pytest.
    code = """
import sys
import numpy as np
import ivit
try:
    ivit.load(sys.argv[1], device="npu", backend="openvino",
              task="classification", use_cache=False)
except ivit.IVITError as error:
    assert "requires bounded input shapes" in str(error), str(error)
else:
    raise AssertionError("Unbounded NPU model was not rejected")
model = ivit.load(sys.argv[2], device="npu", backend="openvino",
                  task="classification", use_cache=False)
model.predict(np.zeros((8, 8, 3), dtype=np.uint8))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, dynamic_model_path, static_model_path],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------------------
# Pure Python: device ranking
# ---------------------------------------------------------------------------


def _device(id, type, backend, name="", available=True):
    return SimpleNamespace(
        id=id,
        type=type,
        backend=backend,
        name=name,
        is_available=available,
        memory_total=0,
    )


@pytest.fixture
def make_manager(monkeypatch):
    """Build a DeviceManager that discovers the given fake C++ devices."""

    def _make(raw_devices):
        import importlib

        devices_module = importlib.import_module("ivit.devices")

        fake = types.ModuleType("ivit._ivit_core")
        fake.list_devices = lambda: raw_devices
        monkeypatch.setitem(sys.modules, "ivit._ivit_core", fake)
        monkeypatch.setattr(sys.modules["ivit"], "_ivit_core", fake, raising=False)
        return devices_module.DeviceManager()

    return _make


class TestDeviceRanking:
    # The C++ side reports CUDA GPUs and Intel iGPUs alike as type "gpu".
    DEVICES = [
        _device("cpu", "cpu", "openvino", "Intel Core"),
        _device("gpu:0", "gpu", "openvino", "Intel(R) UHD Graphics"),
        _device("npu", "npu", "openvino", "Intel(R) AI Boost"),
        _device("cuda:0", "gpu", "tensorrt", "NVIDIA GeForce RTX 4090"),
    ]

    @pytest.mark.parametrize("reverse", [False, True])
    def test_nvidia_gpu_beats_intel_igpu_in_any_order(self, make_manager, reverse):
        devices = list(reversed(self.DEVICES)) if reverse else list(self.DEVICES)
        manager = make_manager(devices)
        assert manager.best().id == "cuda:0"
        assert manager.best("throughput").id == "cuda:0"

    def test_efficiency_and_balanced_prefer_npu(self, make_manager):
        manager = make_manager(self.DEVICES)
        assert manager.best("efficiency").id == "npu"
        assert manager.best("balanced").id == "npu"

    def test_intel_discrete_gpu_beats_igpu(self, make_manager):
        manager = make_manager(
            [
                _device("cpu", "cpu", "openvino"),
                _device("gpu:0", "gpu", "openvino", "Intel(R) UHD Graphics"),
                _device("gpu:1", "gpu", "openvino", "Intel(R) Arc(TM) A770 Graphics"),
            ]
        )
        assert manager.best().id == "gpu:1"

    def test_unavailable_devices_are_skipped(self, make_manager):
        manager = make_manager(
            [
                _device("cuda:0", "gpu", "tensorrt", "NVIDIA RTX", available=False),
                _device("cpu", "cpu", "openvino"),
            ]
        )
        assert manager.best().id == "cpu"

    def test_vendor_and_discrete_inference(self):
        from ivit.devices import _vendor_and_discrete

        assert _vendor_and_discrete("gpu", "tensorrt", "NVIDIA RTX") == ("nvidia", True)
        assert _vendor_and_discrete("gpu", "openvino", "Intel(R) Arc(TM) A380") == (
            "intel",
            True,
        )
        assert _vendor_and_discrete("gpu", "openvino", "Intel UHD") == ("intel", False)
        assert _vendor_and_discrete("npu", "openvino", "AI Boost") == ("intel", False)
        assert _vendor_and_discrete("cpu", "openvino", "AMD Ryzen") == (
            "unknown",
            False,
        )


# ---------------------------------------------------------------------------
# C++ bindings (no model needed)
# ---------------------------------------------------------------------------


class TestBindings:
    def test_load_config_exposes_task_and_backend(self, ivit_cpp):
        config = ivit_cpp.LoadConfig()
        assert config.backend == "auto"
        config.task = "detection"
        config.backend = "openvino"
        assert (config.task, config.backend) == ("detection", "openvino")

    def test_enums_are_exported(self, ivit_cpp):
        assert ivit_cpp.TaskType.Classification != ivit_cpp.TaskType.Detection
        assert ivit_cpp.BackendType.OpenVINO != ivit_cpp.BackendType.TensorRT

    def test_unknown_backend_name_raises_ivit_error(self, ivit_cpp):
        with pytest.raises(ivit_cpp.IVITError, match="Unknown backend"):
            ivit_cpp.load("whatever.onnx", device="cpu", backend="bogus")

    def test_unknown_load_kwarg_raises_type_error(self, ivit_cpp):
        with pytest.raises(TypeError, match="unexpected keyword argument 'bogus'"):
            ivit_cpp.load("whatever.onnx", device="cpu", bogus=1)

    def test_missing_model_is_caught_by_base_class(self, ivit_cpp, tmp_path):
        with pytest.raises(ivit_cpp.IVITError):
            ivit_cpp.load(str(tmp_path / "missing.onnx"), device="cpu")

    def test_cpp_exception_tree_is_connected(self, ivit_cpp):
        assert issubclass(ivit_cpp.ModelLoadError, ivit_cpp.IVITError)
        assert issubclass(ivit_cpp.InferenceError, ivit_cpp.IVITError)
        assert issubclass(ivit_cpp.DeviceNotFoundError, ivit_cpp.IVITError)
        assert issubclass(ivit_cpp.UnsupportedFormatError, ivit_cpp.IVITError)

    def test_python_errors_are_caught_by_cpp_base_classes(self, ivit_cpp):
        from ivit.core.exceptions import ConfigurationError, ModelLoadError

        assert issubclass(ConfigurationError, ivit_cpp.IVITError)
        assert issubclass(ModelLoadError, ivit_cpp.ModelLoadError)


# ---------------------------------------------------------------------------
# Real inference (OpenVINO + onnx)
# ---------------------------------------------------------------------------


class TestInference:
    def test_task_aliases_are_normalised(self, openvino_ivit, static_model_path):
        model = _load(openvino_ivit, static_model_path, task="classify")
        assert model.task == openvino_ivit.TaskType.Classification

    def test_cache_key_includes_task(self, openvino_ivit, static_model_path):
        # use_cache=True on purpose: the cache is what used to mix tasks up
        as_classifier = _load(openvino_ivit, static_model_path, use_cache=True)
        as_detector = _load(
            openvino_ivit, static_model_path, task="detection", use_cache=True
        )
        assert as_classifier.task == openvino_ivit.TaskType.Classification
        assert as_detector.task == openvino_ivit.TaskType.Detection
        assert as_classifier is not as_detector

    def test_explicit_backend_is_honoured(self, openvino_ivit, static_model_path):
        model = _load(openvino_ivit, static_model_path, backend="openvino")
        assert model.backend == openvino_ivit.BackendType.OpenVINO

    def test_unavailable_backend_raises(self, openvino_ivit, static_model_path):
        if any(d.backend == "tensorrt" for d in openvino_ivit.list_devices()):
            pytest.skip("TensorRT is available here")
        with pytest.raises(openvino_ivit.IVITError, match="not available"):
            _load(openvino_ivit, static_model_path, backend="tensorrt")

    def test_call_matches_predict(self, openvino_ivit, static_model_path, images):
        model = _load(openvino_ivit, static_model_path)
        _assert_same(_top1(model(images[0])), _top1(model.predict(images[0])))

    def test_non_contiguous_array_is_read_correctly(
        self, openvino_ivit, static_model_path, images
    ):
        import numpy as np

        model = _load(openvino_ivit, static_model_path)
        view = images[0][:, :, ::-1]  # negative stride: not C-contiguous
        assert not view.flags["C_CONTIGUOUS"]
        _assert_same(_top1(model(view)), _top1(model(np.ascontiguousarray(view))))

    def test_dynamic_batch_model_works(self, openvino_ivit, dynamic_model_path, images):
        model = _load(openvino_ivit, dynamic_model_path)
        model.warmup(1)
        singles = [_top1(model.predict(img)) for img in images[:4]]
        batched = [_top1(r) for r in model.predict_batch(images[:4])]
        assert len(batched) == 4
        for single, batch in zip(singles, batched):
            _assert_same(single, batch)

    def test_concurrent_predict_matches_sequential(
        self, openvino_ivit, static_model_path, images
    ):
        model = _load(openvino_ivit, static_model_path)
        expected = [_top1(model.predict(img)) for img in images]
        got = [_top1(r) for r in model.predict_concurrent(images, max_concurrent=4)]
        for e, g in zip(expected, got):
            _assert_same(e, g)

    @pytest.mark.parametrize("limit", [0, -1])
    def test_concurrent_rejects_invalid_limit(
        self, openvino_ivit, static_model_path, limit
    ):
        model = _load(openvino_ivit, static_model_path)
        with pytest.raises(openvino_ivit.IVITError, match="must be positive"):
            model.predict_concurrent([], max_concurrent=limit)

    def test_concurrent_error_allows_reuse(
        self, openvino_ivit, static_model_path, images
    ):
        import numpy as np

        model = _load(openvino_ivit, static_model_path)
        empty = np.empty((0, 0, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError):
            model.predict_concurrent([empty] * 8, max_concurrent=2)
        expected = [_top1(model.predict(image)) for image in images]
        actual = [
            _top1(result)
            for result in model.predict_concurrent(images, max_concurrent=2)
        ]
        for single, concurrent in zip(expected, actual):
            _assert_same(single, concurrent)

    def test_python_threads_share_one_model(
        self, openvino_ivit, static_model_path, images
    ):
        model = _load(openvino_ivit, static_model_path)
        expected = [_top1(model.predict(img)) for img in images]
        errors = []

        def worker(index):
            try:
                for _ in range(20):
                    _assert_same(_top1(model.predict(images[index])), expected[index])
            except BaseException as exc:  # noqa: BLE001 - report from the thread
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(len(images))
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert not errors, errors[0]

    def test_submit_inference_survives_releasing_the_model(
        self, openvino_ivit, static_model_path, images
    ):
        model = _load(openvino_ivit, static_model_path)
        done = threading.Event()
        received = []

        def on_result(result):
            received.append(result)
            done.set()

        model.submit_inference(images[0], on_result)
        del model  # the worker thread now holds the last reference
        gc.collect()

        assert done.wait(timeout=30), "callback was never invoked"
        assert len(received) == 1
