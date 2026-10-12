#!/usr/bin/env python3
"""Verify a specific OpenVINO device through native runtime and SDK bindings."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import openvino as ov
import ivit


def make_model(path, dynamic_batch=False):
    rng = np.random.default_rng(42)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv", ["input", "W", "B"], ["features"], pads=[1, 1, 1, 1]
            ),
            helper.make_node("GlobalAveragePool", ["features"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["logits"], axis=1),
        ],
        "device_verification",
        [
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, ["N" if dynamic_batch else 1, 3, 32, 32]
            )
        ],
        [
            helper.make_tensor_value_info(
                "logits", TensorProto.FLOAT, ["N" if dynamic_batch else 1, 4]
            )
        ],
        initializer=[
            numpy_helper.from_array(
                rng.normal(0, 0.2, (4, 3, 3, 3)).astype(np.float32), "W"
            ),
            numpy_helper.from_array(np.array([0.1, 0.2, 0.3, 0.4], np.float32), "B"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)


def scores(result):
    return np.array(
        [
            item.score
            for item in sorted(result.classifications, key=lambda item: item.class_id)
        ]
    )


def verify(device, tolerance, dynamic_batch=False):
    if device == "npu" and dynamic_batch:
        raise RuntimeError(
            "This test model has an unbounded batch dimension, which NPU does not "
            "support; use the static test. Bounded dynamic shapes additionally "
            "require a supported NPU platform and compiler."
        )
    core = ov.Core()
    target = (
        "GPU." + device.split(":")[1] if device.startswith("gpu:") else device.upper()
    )
    available = core.available_devices
    print("OpenVINO devices:", available, flush=True)
    if not any(name.split(".")[0] == target.split(".")[0] for name in available):
        raise RuntimeError(
            f"Requested {target} is not enumerated; available: {available}"
        )
    # get_property also handles GPU.0 when enumeration returns the alias GPU.
    full_name = core.get_property(target, "FULL_DEVICE_NAME")
    if device.startswith("gpu:") and "INTEL" not in full_name.upper():
        raise RuntimeError(
            f"Requested Intel GPU resolves to {full_name}; refusing to count another vendor as validated"
        )
    if not ivit.is_cpp_available():
        raise RuntimeError(f"Bindings unavailable: {getattr(ivit, '_CPP_ERROR', None)}")
    sdk_devices = ivit.list_devices()
    if not any(
        item.id == device and item.backend == "openvino" for item in sdk_devices
    ):
        raise RuntimeError(
            f"Device {device} is absent from SDK enumeration: {sdk_devices}"
        )

    with tempfile.TemporaryDirectory(prefix="ivit-openvino-device-") as temporary:
        path = Path(temporary) / "classifier.onnx"
        make_model(path, dynamic_batch)
        compiled = core.compile_model(core.read_model(path), target)
        execution_devices = compiled.get_property("EXECUTION_DEVICES")
        if isinstance(execution_devices, str):
            execution_devices = [execution_devices]
        else:
            execution_devices = list(execution_devices)
        if not execution_devices or any(
            not name.startswith(target.split(".")[0]) for name in execution_devices
        ):
            raise RuntimeError(f"Unexpected execution devices: {execution_devices}")
        cpu = ivit.load(
            str(path),
            device="cpu",
            backend="openvino",
            task="classification",
            use_cache=False,
        )
        model = ivit.load(
            str(path),
            device=device,
            backend="openvino",
            task="classification",
            use_cache=False,
        )
        config = ivit.InferConfig()
        config.conf_threshold = 0.0
        predict = lambda image: model.predict(image, config)
        model.warmup(2)
        images = [
            np.random.default_rng(i).integers(0, 256, (32, 32, 3), dtype=np.uint8)
            for i in range(8)
        ]
        expected = [scores(cpu.predict(image, config)) for image in images]
        started = time.perf_counter()
        actual = [scores(predict(image)) for image in images]
        elapsed = time.perf_counter() - started
        for image, reference, received in zip(images, expected, actual):
            np.testing.assert_allclose(
                received, reference, rtol=tolerance, atol=tolerance
            )
            tensor = np.transpose(
                image[:, :, ::-1].astype(np.float32) / 255, (2, 0, 1)
            )[None]
            logits = np.asarray(
                compiled([np.ascontiguousarray(tensor)])[compiled.output(0)]
            ).reshape(-1)
            probability = np.exp(logits - logits.max())
            probability /= probability.sum()
            np.testing.assert_allclose(
                received, probability, rtol=tolerance, atol=tolerance
            )
        for method in (
            lambda values: model.predict_batch(values, config),
            lambda values: model.predict_concurrent(
                values, max_concurrent=2, config=config
            ),
        ):
            method_results = method(images)
            assert len(method_results) == len(images)
            for reference, result in zip(expected, method_results):
                np.testing.assert_allclose(
                    scores(result), reference, rtol=tolerance, atol=tolerance
                )
        view = images[0][:, :, ::-1]
        np.testing.assert_allclose(
            scores(predict(view)),
            scores(predict(np.ascontiguousarray(view))),
            rtol=tolerance,
            atol=tolerance,
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(predict, images))
        for reference, result in zip(expected, results):
            np.testing.assert_allclose(
                scores(result), reference, rtol=tolerance, atol=tolerance
            )
        return {
            "status": "passed",
            "device": device,
            "full_name": full_name,
            "execution_devices": execution_devices,
            "openvino": ov.__version__,
            "sdk": ivit.__version__,
            "tolerance": tolerance,
            "checks": [
                "native compile/infer",
                "SDK warmup/single",
                "CPU reference",
                "native output reference",
                "batch",
                "concurrent",
                "noncontiguous input",
                "Python threads",
            ],
            "dynamic_batch": dynamic_batch,
            "single_image_count": len(images),
            "single_elapsed_seconds": elapsed,
            "limits": "Small classifier only; not a production-model accuracy or throughput benchmark.",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", choices=["cpu", "gpu:0", "gpu:1", "npu"], required=True
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()
    if not 0 < args.tolerance <= 0.05:
        parser.error("tolerance must be positive and at most 0.05")
    try:
        report = verify(args.device, args.tolerance, args.dynamic_batch)
    except Exception as error:
        report = {"status": "failed", "device": args.device, "error": str(error)}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered, flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
