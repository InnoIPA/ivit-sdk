"""
iVIT Model Zoo - Pre-optimized models for edge deployment.

All models use commercial-friendly licenses (Apache-2.0, BSD-3-Clause).
No AGPL/GPL dependencies - safe for commercial use.

Available model families:
- YOLOX (Megvii, Apache-2.0): yolox-nano, yolox-tiny, yolox-s, yolox-m, yolox-l
- RT-DETR (PaddleDetection, Apache-2.0): rtdetr-l, rtdetr-x
- ResNet/MobileNet/EfficientNet (torchvision, BSD-3): Classification models
- DeepLabV3 (torchvision, BSD-3): Semantic segmentation
- RTMPose (MMPose, Apache-2.0): Human pose estimation

Examples:
    >>> import ivit
    >>>
    >>> # List available models
    >>> ivit.zoo.list_models()
    >>> ivit.zoo.search("yolox")
    >>>
    >>> # Load detection model
    >>> model = ivit.zoo.load("yolox-s")
    >>> results = model("image.jpg")
    >>>
    >>> # Load classification model
    >>> model = ivit.zoo.load("resnet50")
    >>> results = model("image.jpg")
    >>>
    >>> # Load pose estimation model
    >>> model = ivit.zoo.load("rtmpose-s", device="npu")
    >>> results = model("image.jpg")
"""

from .registry import (
    list_models,
    search,
    get_model_info,
    download,
    load,
    ModelInfo,
    is_trainable,
    get_training_backend,
    list_trainable_models,
)

__all__ = [
    "list_models",
    "search",
    "get_model_info",
    "download",
    "load",
    "ModelInfo",
    "is_trainable",
    "get_training_backend",
    "list_trainable_models",
]
