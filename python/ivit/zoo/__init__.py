"""
iVIT Model Zoo - Pre-optimized models for edge deployment.

All models use commercial-friendly licenses (Apache-2.0, BSD-3-Clause).
No AGPL/GPL dependencies - safe for commercial use.

Model families. "ready" models decode out of the box: zoo.load() applies the verified task,
decoder, color order, normalization and labels. "raw" models can be downloaded, but the SDK has
no decoder for their outputs.

- YOLOX (Megvii, Apache-2.0), ready: yolox-nano, yolox-tiny, yolox-s, yolox-m, yolox-l
- ResNet/MobileNet/EfficientNet (torchvision, BSD-3), ready: classification
- DeepLabV3 (torchvision, BSD-3), ready: semantic segmentation
- Faster R-CNN/RetinaNet/SSDLite/FCOS (torchvision), raw: boxes/scores/labels outputs

Examples:
    >>> import cv2, ivit
    >>> ivit.zoo.list_models()
    >>> model = ivit.zoo.load("yolox-s")              # detection, COCO labels
    >>> results = model.predict(cv2.imread("image.jpg"))
    >>> model = ivit.zoo.load("resnet18", device="gpu:0")
    >>> path = ivit.zoo.download("fasterrcnn-resnet50")  # raw: run it in your own runtime
"""

from .registry import (
    list_models,
    search,
    get_model_info,
    download,
    load,
    ModelInfo,
)

__all__ = [
    "list_models",
    "search",
    "get_model_info",
    "download",
    "load",
    "ModelInfo",
]
