#!/usr/bin/env python3
"""Validate TensorRT bindings against an OpenVINO CPU reference."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import ivit
import numpy as np

sys.dont_write_bytecode = True  # importing the sibling script must not leave scripts/__pycache__
from verify_openvino import make_model, scores  # noqa: E402


def verify(device, precision, dynamic_batch, tolerance):
    if not ivit.is_cpp_available():
        raise RuntimeError("C++ bindings are unavailable")
    devices = ivit.list_devices()
    selected = next(
        (d for d in devices if d.id == device and d.backend == "tensorrt"), None
    )
    if selected is None:
        raise RuntimeError(f"TensorRT device {device} is not enumerated: {devices}")
    config = ivit.InferConfig()
    config.conf_threshold = 0.0
    with tempfile.TemporaryDirectory(prefix="ivit-tensorrt-") as temporary:
        path = Path(temporary) / "classifier.onnx"
        make_model(path, dynamic_batch)
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
            backend="tensorrt",
            precision=precision,
            task="classification",
            use_cache=False,
        )
        assert model.backend == ivit.BackendType.TensorRT, model.backend
        assert model.device == device, model.device
        assert not list(
            Path(temporary).glob("*.engine")
        ), "use_cache=False wrote an engine"
        images = [
            np.random.default_rng(i).integers(0, 256, (32, 32, 3), dtype=np.uint8)
            for i in range(8)
        ]
        expected = [scores(cpu.predict(image, config)) for image in images]

        def compare(results, references=expected):
            assert len(results) == len(references)
            for result, reference in zip(results, references):
                np.testing.assert_allclose(
                    scores(result), reference, atol=tolerance, rtol=tolerance
                )

        model.warmup(2)
        compare([model.predict(image, config) for image in images])
        compare(model.predict_batch(images, config))
        compare(model.predict_concurrent(images, max_concurrent=2, config=config))
        view = images[0][:, :, ::-1]
        compare(
            [model.predict(view, config)],
            [scores(model.predict(np.ascontiguousarray(view), config))],
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            compare(
                list(executor.map(lambda image: model.predict(image, config), images))
            )
        # Vary shapes after batch=8 to catch stale output sizes/buffer misuse.
        for size in (1, 4, 8, 2, 1):
            compare(model.predict_batch(images[:size], config), expected[:size])
        if dynamic_batch:
            try:
                model.predict_batch(images + images[:1], config)
            except ivit.IVITError as error:
                assert "optimization profile" in str(error), str(error)
            else:
                raise AssertionError("batch=9 exceeded the profile without an error")
            compare(model.predict_batch(images[:1], config), expected[:1])
        del model
        gc.collect()
        # Explicit unload/reload must work in the same process.
        for _ in range(3):
            model = ivit.load(
                str(path),
                device=device,
                backend="tensorrt",
                precision=precision,
                task="classification",
                use_cache=False,
            )
            compare([model.predict(images[0], config)], expected[:1])
            del model
            gc.collect()
        # The child leaves a cached model alive at interpreter shutdown.
        code = """
import sys
import ivit
import numpy as np
model = ivit.load(sys.argv[1], device=sys.argv[2], backend="tensorrt",
                  precision=sys.argv[3], task="classification", use_cache=True)
model.predict(np.zeros((32, 32, 3), dtype=np.uint8))
"""
        child = subprocess.run(
            [sys.executable, "-c", code, str(path), device, precision],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert child.returncode == 0, child.stdout + child.stderr
        engines = list(Path(temporary).glob("*.engine"))
        assert len(engines) == 1, engines
        model = ivit.load(
            str(engines[0]),
            device=device,
            backend="tensorrt",
            task="classification",
            use_cache=False,
        )
        compare(model.predict_batch(images, config))
        del model
        gc.collect()
        return {
            "status": "passed",
            "device": device,
            "full_name": selected.name,
            "backend": "tensorrt",
            "precision": precision,
            "dynamic_batch": dynamic_batch,
            "tolerance": tolerance,
            "checks": (["profile overflow/recovery"] if dynamic_batch else [])
            + [
                "bindings",
                "explicit backend/device",
                "CPU reference",
                "warmup/single",
                "batch",
                "concurrent",
                "noncontiguous input",
                "Python threads",
                "changing batch sizes",
                "unload/reload",
                "cached model process shutdown",
                "serialized engine roundtrip",
            ],
            "limits": "Small classifier only; no INT8, production models or multi-GPU validation.",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not 0 < args.tolerance <= 0.05:
        parser.error("tolerance must be positive and at most 0.05")
    try:
        report = verify(args.device, args.precision, args.dynamic_batch, args.tolerance)
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
