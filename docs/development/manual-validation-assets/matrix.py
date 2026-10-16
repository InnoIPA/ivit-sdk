"""Manual acceptance helper: run Zoo contracts through an explicit backend."""
import argparse
from pathlib import Path

import cv2
import ivit
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--backend", required=True, choices=["openvino", "tensorrt"])
parser.add_argument("--device", required=True)
parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
parser.add_argument("--image", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
assert ivit.is_cpp_available(), ivit.get_runtime_info()
assert any(d.id == args.device and d.backend == args.backend and d.is_available
           for d in ivit.list_devices()), "Requested backend/device unavailable"
image = cv2.imread(args.image)
assert image is not None, args.image
args.output.mkdir(parents=True, exist_ok=True)
for name in ("resnet18", "yolox-s", "deeplabv3-mobilenetv3"):
    model = ivit.zoo.load(name, device=args.device, backend=args.backend,
                          precision=args.precision, use_cache=False)
    assert model.device == args.device, model.device
    assert str(model.backend).lower().endswith(args.backend), model.backend
    model.warmup(3)
    result = model.predict(image)
    assert result.device_used == args.device, result.device_used
    if name == "resnet18":
        assert len(result.classifications) == 1000
        assert all(np.isfinite(item.score) for item in result.classifications)
    elif name == "yolox-s":
        assert result.detections, "Expected detections on bus.jpg"
    else:
        assert result.segmentation_mask.shape == image.shape[:2]
        assert np.all((result.segmentation_mask >= 0) & (result.segmentation_mask < 21))
        np.save(args.output / f"{name}-mask.npy", result.segmentation_mask)
    result.save(str(args.output / f"{name}.json"))
    cv2.imwrite(str(args.output / f"{name}.jpg"), result.visualize(image))
    print(name, "backend=", model.backend, "device=", result.device_used,
          "precision=", args.precision, "ms=", result.inference_time_ms)
