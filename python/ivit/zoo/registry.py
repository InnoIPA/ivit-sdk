"""
Model Zoo Registry - Model catalog and download management.

All models in the registry use commercial-friendly licenses (Apache-2.0, BSD-3-Clause, MIT).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Model information in the zoo."""
    name: str
    task: str  # detect, classify, segment, pose
    description: str
    input_size: tuple  # (H, W)
    num_classes: int
    formats: List[str]  # Available formats: onnx, openvino, tensorrt
    url: Optional[str] = None        # 原始框架權重 (.pth, .pdparams)
    onnx_url: Optional[str] = None   # 預轉換 ONNX 下載 URL
    source: str = "torchvision"  # torchvision, megvii, paddlepaddle, mmpose, custom
    license: str = "Apache-2.0"
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


# Model Zoo Registry
# All models use commercial-friendly licenses (Apache-2.0, BSD-3-Clause, MIT)
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    # =========================================================================
    # YOLOX Detection Models (Megvii, Apache-2.0)
    # =========================================================================
    "yolox-nano": ModelInfo(
        name="yolox-nano",
        task="detect",
        description="YOLOX Nano - Ultra-lightweight model for edge devices",
        input_size=(416, 416),
        num_classes=80,
        formats=["onnx", "openvino", "tensorrt"],
        url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth",
        onnx_url="https://github.com/innodisk-mannywang/ivit-sdk/releases/download/models-v1.0/yolox-nano.onnx",
        source="megvii",
        license="Apache-2.0",
        metrics={"mAP50-95": 25.8, "params_m": 0.91, "flops_g": 1.08},
        tags=["yolox", "detection", "fast", "edge", "commercial-friendly"],
    ),
    "yolox-tiny": ModelInfo(
        name="yolox-tiny",
        task="detect",
        description="YOLOX Tiny - Lightweight model for edge devices",
        input_size=(416, 416),
        num_classes=80,
        formats=["onnx", "openvino", "tensorrt"],
        url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth",
        onnx_url="https://github.com/innodisk-mannywang/ivit-sdk/releases/download/models-v1.0/yolox-tiny.onnx",
        source="megvii",
        license="Apache-2.0",
        metrics={"mAP50-95": 32.8, "params_m": 5.06, "flops_g": 6.45},
        tags=["yolox", "detection", "fast", "edge", "commercial-friendly"],
    ),
    "yolox-s": ModelInfo(
        name="yolox-s",
        task="detect",
        description="YOLOX Small - Balance of speed and accuracy",
        input_size=(640, 640),
        num_classes=80,
        formats=["onnx", "openvino", "tensorrt"],
        url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth",
        onnx_url="https://github.com/innodisk-mannywang/ivit-sdk/releases/download/models-v1.0/yolox-s.onnx",
        source="megvii",
        license="Apache-2.0",
        metrics={"mAP50-95": 40.5, "params_m": 9.0, "flops_g": 26.8},
        tags=["yolox", "detection", "balanced", "commercial-friendly"],
    ),
    "yolox-m": ModelInfo(
        name="yolox-m",
        task="detect",
        description="YOLOX Medium - Higher accuracy",
        input_size=(640, 640),
        num_classes=80,
        formats=["onnx", "openvino", "tensorrt"],
        url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth",
        onnx_url="https://github.com/innodisk-mannywang/ivit-sdk/releases/download/models-v1.0/yolox-m.onnx",
        source="megvii",
        license="Apache-2.0",
        metrics={"mAP50-95": 46.9, "params_m": 25.3, "flops_g": 73.8},
        tags=["yolox", "detection", "accurate", "commercial-friendly"],
    ),
    "yolox-l": ModelInfo(
        name="yolox-l",
        task="detect",
        description="YOLOX Large - High accuracy for server deployment",
        input_size=(640, 640),
        num_classes=80,
        formats=["onnx", "openvino", "tensorrt"],
        url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth",
        onnx_url="https://github.com/innodisk-mannywang/ivit-sdk/releases/download/models-v1.0/yolox-l.onnx",
        source="megvii",
        license="Apache-2.0",
        metrics={"mAP50-95": 49.7, "params_m": 54.2, "flops_g": 155.6},
        tags=["yolox", "detection", "accurate", "server", "commercial-friendly"],
    ),

    # =========================================================================
    # RT-DETR Detection Models (PaddleDetection, Apache-2.0)
    # =========================================================================
    "rtdetr-l": ModelInfo(
        name="rtdetr-l",
        task="detect",
        description="RT-DETR Large - Real-time DETR, no NMS needed",
        input_size=(640, 640),
        num_classes=80,
        formats=["onnx", "openvino", "tensorrt"],
        url="https://bj.bcebos.com/v1/paddledet/models/rtdetr_r50vd_6x_coco.pdparams",
        onnx_url="https://github.com/innodisk-mannywang/ivit-sdk/releases/download/models-v1.0/rtdetr-l.onnx",
        source="paddlepaddle",
        license="Apache-2.0",
        metrics={"mAP50-95": 53.0, "params_m": 32.0},
        tags=["rtdetr", "detr", "detection", "transformer", "accurate", "commercial-friendly"],
    ),
    "rtdetr-x": ModelInfo(
        name="rtdetr-x",
        task="detect",
        description="RT-DETR Extra Large - Maximum accuracy DETR",
        input_size=(640, 640),
        num_classes=80,
        formats=["onnx", "openvino", "tensorrt"],
        url="https://bj.bcebos.com/v1/paddledet/models/rtdetr_r101vd_6x_coco.pdparams",
        onnx_url="https://github.com/innodisk-mannywang/ivit-sdk/releases/download/models-v1.0/rtdetr-x.onnx",
        source="paddlepaddle",
        license="Apache-2.0",
        metrics={"mAP50-95": 54.8, "params_m": 67.0},
        tags=["rtdetr", "detr", "detection", "transformer", "accurate", "server", "commercial-friendly"],
    ),

    # =========================================================================
    # Detection Models (torchvision, BSD-3-Clause) - Ready to use!
    # =========================================================================
    "fasterrcnn-mobilenet": ModelInfo(
        name="fasterrcnn-mobilenet",
        task="detect",
        description="Faster R-CNN MobileNetV3 - Fast detection for edge",
        input_size=(640, 640),
        num_classes=91,  # COCO classes
        formats=["onnx", "openvino"],
        source="torchvision",
        license="BSD-3-Clause",
        metrics={"mAP50-95": 32.8, "params_m": 19.4},
        tags=["detection", "fasterrcnn", "mobilenet", "fast", "edge", "commercial-friendly", "ready-to-use"],
    ),
    "fasterrcnn-resnet50": ModelInfo(
        name="fasterrcnn-resnet50",
        task="detect",
        description="Faster R-CNN ResNet-50 FPN v2 - Accurate detection",
        input_size=(800, 800),
        num_classes=91,
        formats=["onnx", "openvino"],
        source="torchvision",
        license="BSD-3-Clause",
        metrics={"mAP50-95": 46.7, "params_m": 43.7},
        tags=["detection", "fasterrcnn", "resnet", "accurate", "commercial-friendly", "ready-to-use"],
    ),
    "retinanet-resnet50": ModelInfo(
        name="retinanet-resnet50",
        task="detect",
        description="RetinaNet ResNet-50 FPN v2 - One-stage detector",
        input_size=(800, 800),
        num_classes=91,
        formats=["onnx", "openvino"],
        source="torchvision",
        license="BSD-3-Clause",
        metrics={"mAP50-95": 41.5, "params_m": 38.2},
        tags=["detection", "retinanet", "resnet", "one-stage", "commercial-friendly", "ready-to-use"],
    ),
    "ssdlite-mobilenet": ModelInfo(
        name="ssdlite-mobilenet",
        task="detect",
        description="SSDLite MobileNetV3 - Ultra-fast edge detection",
        input_size=(320, 320),
        num_classes=91,
        formats=["onnx", "openvino"],
        source="torchvision",
        license="BSD-3-Clause",
        metrics={"mAP50-95": 21.3, "params_m": 3.4},
        tags=["detection", "ssd", "mobilenet", "fast", "edge", "commercial-friendly", "ready-to-use"],
    ),
    "fcos-resnet50": ModelInfo(
        name="fcos-resnet50",
        task="detect",
        description="FCOS ResNet-50 FPN - Anchor-free detector",
        input_size=(800, 800),
        num_classes=91,
        formats=["onnx", "openvino"],
        source="torchvision",
        license="BSD-3-Clause",
        metrics={"mAP50-95": 39.2, "params_m": 32.3},
        tags=["detection", "fcos", "anchor-free", "resnet", "commercial-friendly", "ready-to-use"],
    ),

    # =========================================================================
    # Classification Models (torchvision, BSD-3-Clause)
    # =========================================================================
    "resnet18": ModelInfo(
        name="resnet18",
        task="classify",
        description="ResNet-18 ImageNet Classification - Lightweight",
        input_size=(224, 224),
        num_classes=1000,
        formats=["onnx", "openvino"],
        source="torchvision",
        license="BSD-3-Clause",
        metrics={"top1": 69.8, "top5": 89.1, "params_m": 11.7},
        tags=["classification", "imagenet", "classic", "fast", "commercial-friendly", "ready-to-use"],
    ),
    "resnet50": ModelInfo(
        name="resnet50",
        task="classify",
        description="ResNet-50 ImageNet Classification",
        input_size=(224, 224),
        num_classes=1000,
        formats=["onnx", "openvino"],
        source="torchvision",
        license="BSD-3-Clause",
        metrics={"top1": 76.1, "top5": 92.9, "params_m": 25.6},
        tags=["classification", "imagenet", "classic", "commercial-friendly", "ready-to-use"],
    ),
    "mobilenetv3": ModelInfo(
        name="mobilenetv3",
        task="classify",
        description="MobileNetV3 Large - Efficient mobile classification",
        input_size=(224, 224),
        num_classes=1000,
        formats=["onnx", "openvino"],
        source="torchvision",
        license="BSD-3-Clause",
        metrics={"top1": 74.0, "top5": 91.3, "params_m": 5.5},
        tags=["classification", "mobile", "efficient", "edge", "commercial-friendly", "ready-to-use"],
    ),
    "efficientnet-b0": ModelInfo(
        name="efficientnet-b0",
        task="classify",
        description="EfficientNet-B0 - Efficient scaling",
        input_size=(224, 224),
        num_classes=1000,
        formats=["onnx", "openvino"],
        source="torchvision",
        license="Apache-2.0",
        metrics={"top1": 77.1, "top5": 93.3, "params_m": 5.3},
        tags=["classification", "efficient", "commercial-friendly", "ready-to-use"],
    ),

    # =========================================================================
    # MambaVision Classification Models (NVIDIA Research, Apache-2.0)
    # Hybrid Mamba + Transformer architecture for image classification.
    # Export: requires `pip install mambavision` and selective_scan patching.
    # See: ivit.train.exporter.load_mambavision_for_export()
    # =========================================================================
    "mambavision-t": ModelInfo(
        name="mambavision-t",
        task="classify",
        description="MambaVision-Tiny - Hybrid Mamba+Transformer, fast edge inference",
        input_size=(224, 224),
        num_classes=1000,
        formats=["onnx", "openvino", "tensorrt"],
        url="https://huggingface.co/nvidia/MambaVision-T-1K",
        onnx_url="https://github.com/innodisk-mannywang/ivit-sdk/releases/download/models-v1.0/mambavision-t.onnx",
        source="mambavision",
        license="Apache-2.0",
        metrics={"top1": 82.3, "params_m": 31.8, "macs_g": 4.4},
        tags=["classification", "imagenet", "mamba", "ssm", "transformer", "hybrid", "fast", "commercial-friendly"],
    ),
    "mambavision-s": ModelInfo(
        name="mambavision-s",
        task="classify",
        description="MambaVision-Small - Balanced speed/accuracy",
        input_size=(224, 224),
        num_classes=1000,
        formats=["onnx", "openvino", "tensorrt"],
        url="https://huggingface.co/nvidia/MambaVision-S-1K",
        onnx_url="https://github.com/innodisk-mannywang/ivit-sdk/releases/download/models-v1.0/mambavision-s.onnx",
        source="mambavision",
        license="Apache-2.0",
        metrics={"top1": 83.3, "params_m": 50.1, "macs_g": 7.5},
        tags=["classification", "imagenet", "mamba", "ssm", "transformer", "hybrid", "balanced", "commercial-friendly"],
    ),
    "mambavision-b": ModelInfo(
        name="mambavision-b",
        task="classify",
        description="MambaVision-Base - High accuracy",
        input_size=(224, 224),
        num_classes=1000,
        formats=["onnx", "openvino", "tensorrt"],
        url="https://huggingface.co/nvidia/MambaVision-B-1K",
        onnx_url="https://github.com/innodisk-mannywang/ivit-sdk/releases/download/models-v1.0/mambavision-b.onnx",
        source="mambavision",
        license="Apache-2.0",
        metrics={"top1": 84.2, "params_m": 97.7, "macs_g": 15.0},
        tags=["classification", "imagenet", "mamba", "ssm", "transformer", "hybrid", "accurate", "commercial-friendly"],
    ),
    "mambavision-l": ModelInfo(
        name="mambavision-l",
        task="classify",
        description="MambaVision-Large - Maximum accuracy",
        input_size=(224, 224),
        num_classes=1000,
        formats=["onnx", "openvino", "tensorrt"],
        url="https://huggingface.co/nvidia/MambaVision-L-1K",
        onnx_url="https://github.com/innodisk-mannywang/ivit-sdk/releases/download/models-v1.0/mambavision-l.onnx",
        source="mambavision",
        license="Apache-2.0",
        metrics={"top1": 85.0, "params_m": 227.9, "macs_g": 34.9},
        tags=["classification", "imagenet", "mamba", "ssm", "transformer", "hybrid", "accurate", "server", "commercial-friendly"],
    ),
    "mambavision-l2": ModelInfo(
        name="mambavision-l2",
        task="classify",
        description="MambaVision-Large2 - State-of-the-art accuracy",
        input_size=(224, 224),
        num_classes=1000,
        formats=["onnx", "openvino", "tensorrt"],
        url="https://huggingface.co/nvidia/MambaVision-L2-1K",
        onnx_url="https://github.com/innodisk-mannywang/ivit-sdk/releases/download/models-v1.0/mambavision-l2.onnx",
        source="mambavision",
        license="Apache-2.0",
        metrics={"top1": 85.3, "params_m": 241.5, "macs_g": 37.5},
        tags=["classification", "imagenet", "mamba", "ssm", "transformer", "hybrid", "accurate", "server", "commercial-friendly"],
    ),

    # =========================================================================
    # Semantic Segmentation Models (torchvision, BSD-3-Clause)
    # =========================================================================
    "deeplabv3-resnet50": ModelInfo(
        name="deeplabv3-resnet50",
        task="segment",
        description="DeepLabV3 ResNet-50 - Semantic segmentation",
        input_size=(520, 520),
        num_classes=21,
        formats=["onnx", "openvino"],
        source="torchvision",
        license="BSD-3-Clause",
        metrics={"mIoU": 66.4},
        tags=["segmentation", "semantic", "voc", "classic", "commercial-friendly", "ready-to-use"],
    ),
    "deeplabv3-mobilenetv3": ModelInfo(
        name="deeplabv3-mobilenetv3",
        task="segment",
        description="DeepLabV3 MobileNetV3 - Lightweight semantic segmentation",
        input_size=(520, 520),
        num_classes=21,
        formats=["onnx", "openvino"],
        source="torchvision",
        license="BSD-3-Clause",
        metrics={"mIoU": 60.3},
        tags=["segmentation", "semantic", "voc", "mobile", "efficient", "commercial-friendly", "ready-to-use"],
    ),

    # =========================================================================
    # Pose Estimation Models (MMPose, Apache-2.0)
    # =========================================================================
    "rtmpose-s": ModelInfo(
        name="rtmpose-s",
        task="pose",
        description="RTMPose Small - Fast human pose estimation",
        input_size=(256, 192),
        num_classes=17,  # COCO keypoints
        formats=["onnx", "openvino", "tensorrt"],
        url="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth",
        onnx_url="https://github.com/innodisk-mannywang/ivit-sdk/releases/download/models-v1.0/rtmpose-s.onnx",
        source="mmpose",
        license="Apache-2.0",
        metrics={"AP": 72.2, "params_m": 5.5},
        tags=["pose", "keypoint", "human", "fast", "commercial-friendly"],
    ),
    "rtmpose-m": ModelInfo(
        name="rtmpose-m",
        task="pose",
        description="RTMPose Medium - Balanced pose estimation",
        input_size=(256, 192),
        num_classes=17,  # COCO keypoints
        formats=["onnx", "openvino", "tensorrt"],
        url="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth",
        onnx_url="https://github.com/innodisk-mannywang/ivit-sdk/releases/download/models-v1.0/rtmpose-m.onnx",
        source="mmpose",
        license="Apache-2.0",
        metrics={"AP": 75.3, "params_m": 13.6},
        tags=["pose", "keypoint", "human", "balanced", "commercial-friendly"],
    ),
}


def _get_cache_dir() -> Path:
    """Get model cache directory."""
    cache_dir = os.environ.get("IVIT_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir)

    # Default: ~/.cache/ivit/models
    return Path.home() / ".cache" / "ivit" / "models"


def list_models(task: Optional[str] = None) -> List[str]:
    """
    List available models in the zoo.

    Args:
        task: Filter by task type (detect, classify, segment, pose)

    Returns:
        List of model names

    Examples:
        >>> ivit.zoo.list_models()
        ['yolox-s', 'yolox-m', 'resnet50', ...]
        >>> ivit.zoo.list_models(task="detect")
        ['yolox-s', 'yolox-m', ...]
    """
    models = []
    for name, info in MODEL_REGISTRY.items():
        if task is None or info.task == task:
            models.append(name)
    return sorted(models)


def search(query: str) -> List[str]:
    """
    Search models by name or tag.

    Args:
        query: Search query

    Returns:
        List of matching model names

    Examples:
        >>> ivit.zoo.search("yolo")
        ['yolox-s', 'yolox-m', 'yolox-l', ...]
        >>> ivit.zoo.search("edge")
        ['yolox-nano', 'yolox-tiny', 'mobilenet_v3_small', ...]
    """
    query = query.lower()
    results = []

    for name, info in MODEL_REGISTRY.items():
        # Match name
        if query in name.lower():
            results.append(name)
            continue

        # Match tags
        if any(query in tag.lower() for tag in info.tags):
            results.append(name)
            continue

        # Match description
        if query in info.description.lower():
            results.append(name)

    return sorted(set(results))


def get_model_info(name: str) -> ModelInfo:
    """
    Get detailed model information.

    Args:
        name: Model name

    Returns:
        ModelInfo object

    Raises:
        KeyError: If model not found

    Examples:
        >>> info = ivit.zoo.get_model_info("yolox-s")
        >>> print(f"Task: {info.task}, Input: {info.input_size}")
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Model not found: {name}. Use ivit.zoo.list_models() to see available models.")
    return MODEL_REGISTRY[name]


# ---------------------------------------------------------------------------
# Training backend dispatch
# ---------------------------------------------------------------------------

# Registry 中可訓練的偵測模型（MODEL_REGISTRY 內 source==megvii）
_YOLOX_TRAINABLE = {"yolox-nano", "yolox-tiny", "yolox-s", "yolox-m", "yolox-l"}

# Registry 中可訓練的分類模型（torchvision）
_TORCHVISION_CLS_TRAINABLE = {
    "resnet18", "resnet34", "resnet50", "resnet101",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
    "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
    "vgg16", "vgg19", "densenet121",
}

# Registry 中可訓練的語意分割模型
_TORCHVISION_SEG_TRAINABLE = {"deeplabv3-resnet50", "deeplabv3-mobilenetv3"}


def is_trainable(name: str) -> bool:
    """模型名是否能透過 ivit.train 訓練。"""
    n = name.lower()
    return (
        n in _YOLOX_TRAINABLE
        or n in _TORCHVISION_CLS_TRAINABLE
        or n in _TORCHVISION_SEG_TRAINABLE
    )


def get_training_backend(name: str) -> str:
    """
    回傳訓練後端：
        - "yolox"
        - "torchvision-classify"
        - "torchvision-segment"
    """
    n = name.lower()
    if n in _YOLOX_TRAINABLE:
        return "yolox"
    if n in _TORCHVISION_CLS_TRAINABLE:
        return "torchvision-classify"
    if n in _TORCHVISION_SEG_TRAINABLE:
        return "torchvision-segment"
    raise ValueError(f"Model '{name}' is not registered as trainable.")


def list_trainable_models() -> Dict[str, List[str]]:
    """依 backend 分組列出所有可訓練模型名稱。"""
    return {
        "yolox": sorted(_YOLOX_TRAINABLE),
        "torchvision-classify": sorted(_TORCHVISION_CLS_TRAINABLE),
        "torchvision-segment": sorted(_TORCHVISION_SEG_TRAINABLE),
    }


def download(
    name: str,
    format: str = "onnx",
    force: bool = False,
    from_source: bool = False,
) -> Path:
    """
    Download model from zoo.

    Args:
        name: Model name
        format: Model format (onnx, openvino, tensorrt)
        force: Force re-download
        from_source: If True, download original weights and convert locally
            instead of using pre-converted ONNX files

    Returns:
        Path to downloaded model

    Examples:
        >>> path = ivit.zoo.download("yolox-s")
        >>> print(path)
        /home/user/.cache/ivit/models/yolox-s.onnx
    """
    info = get_model_info(name)

    if format not in info.formats:
        raise ValueError(f"Format '{format}' not available for {name}. Available: {info.formats}")

    cache_dir = _get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine file path
    if format == "onnx":
        model_path = cache_dir / f"{name}.onnx"
    elif format == "openvino":
        model_path = cache_dir / f"{name}_openvino" / f"{name}.xml"
    elif format == "tensorrt":
        model_path = cache_dir / f"{name}.engine"
    else:
        model_path = cache_dir / f"{name}.{format}"

    # Check if already exists
    if model_path.exists() and not force:
        logger.info(f"Model already cached: {model_path}")
        return model_path

    # Direct ONNX download: if format is onnx, onnx_url exists, and not from_source
    if format == "onnx" and info.onnx_url and not from_source:
        return _download_onnx_direct(name, info.onnx_url, model_path, force)

    # Route to appropriate exporter based on source
    if info.source == "megvii":
        # YOLOX models
        if format == "onnx":
            model_path = _export_yolox_onnx(name, model_path, info, force)
    elif info.source == "paddlepaddle":
        # RT-DETR models
        if format == "onnx":
            model_path = _export_rtdetr_onnx(name, model_path, info, force)
    elif info.source == "mmpose":
        # RTMPose models
        if format == "onnx":
            model_path = _export_mmpose_onnx(name, model_path, info, force)
    elif info.source == "torchvision":
        if format == "onnx":
            model_path = _export_torchvision_onnx(name, model_path, info)

    # Guard against silent failure: never return a path to a file that no
    # downloader/exporter actually produced.
    if not Path(model_path).exists():
        if format != "onnx":
            raise NotImplementedError(
                f"zoo.download does not build the '{format}' format for '{name}'. "
                f"Download ONNX (the default) and convert it, e.g. "
                f"`ivit convert {name}.onnx -f {format}`."
            )
        raise RuntimeError(
            f"Failed to produce '{name}' (source='{info.source}'): no exporter "
            f"handles this model. See the model's documentation for how to obtain it."
        )

    return model_path


_TORCHVISION_MODEL_MAP = {
    # Classification models
    "resnet18": ("resnet18", "ResNet18_Weights.DEFAULT"),
    "resnet50": ("resnet50", "ResNet50_Weights.DEFAULT"),
    "mobilenetv3": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights.DEFAULT"),
    "efficientnet-b0": ("efficientnet_b0", "EfficientNet_B0_Weights.DEFAULT"),
    # Segmentation models
    "deeplabv3-resnet50": ("deeplabv3_resnet50", "DeepLabV3_ResNet50_Weights.DEFAULT"),
    "deeplabv3-mobilenetv3": ("deeplabv3_mobilenet_v3_large", "DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT"),
    # Detection models
    "fasterrcnn-mobilenet": ("fasterrcnn_mobilenet_v3_large_fpn", "FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT"),
    "fasterrcnn-resnet50": ("fasterrcnn_resnet50_fpn_v2", "FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT"),
    "retinanet-resnet50": ("retinanet_resnet50_fpn_v2", "RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT"),
    "ssdlite-mobilenet": ("ssdlite320_mobilenet_v3_large", "SSDLite320_MobileNet_V3_Large_Weights.DEFAULT"),
    "fcos-resnet50": ("fcos_resnet50_fpn", "FCOS_ResNet50_FPN_Weights.DEFAULT"),
}


def _export_torchvision_onnx(name: str, onnx_path: Path, info: ModelInfo) -> Path:
    """Export a torchvision model to ONNX."""
    try:
        import torch
        import torchvision.models as models
        import torchvision.models.segmentation as seg_models
        import torchvision.models.detection as det_models
    except ImportError:
        raise ImportError(
            "PyTorch and torchvision required for torchvision model export. "
            "Install with: pip install torch torchvision"
        )

    if name not in _TORCHVISION_MODEL_MAP:
        raise ValueError(f"No torchvision mapping for model: {name}")

    func_name, weights_name = _TORCHVISION_MODEL_MAP[name]

    logger.info(f"Loading torchvision model: {func_name}")

    # Determine which module the model belongs to
    if hasattr(det_models, func_name):
        module = det_models
    elif hasattr(seg_models, func_name):
        module = seg_models
    else:
        module = models

    weights_enum = None
    for part in weights_name.split("."):
        weights_enum = getattr(module if weights_enum is None else weights_enum, part)

    raw_model = getattr(module, func_name)(weights=weights_enum)
    raw_model.eval()

    h, w = info.input_size
    dummy = torch.randn(1, 3, h, w)

    # Handle different model types
    if info.task == "segment":
        # Segmentation models return OrderedDict with "out" and "aux" keys
        class _SegWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                return self.m(x)["out"]
        model = _SegWrapper(raw_model)
        output_names = ["output"]

    elif info.task == "detect":
        # Detection models are complex - use pre-exported ONNX from torchvision hub
        # torchvision detection models (Faster R-CNN, RetinaNet, SSD, FCOS) have
        # complex control flow that doesn't export well with torch.onnx.export
        # We'll try scripting first, then tracing
        logger.info("Detection model export requires special handling...")

        # Put model in eval mode and use scripting
        raw_model.eval()

        # For detection, we need to trace the model differently
        # Use the model's transform to get proper input format
        class _DetWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.m.eval()

            def forward(self, images):
                # Detection models expect list of tensors
                img_list = [images[i] for i in range(images.shape[0])]
                outputs = self.m(img_list)
                # For single batch, return first result
                # Output format: boxes [N, 4], scores [N], labels [N]
                if len(outputs) > 0:
                    boxes = outputs[0]["boxes"]
                    scores = outputs[0]["scores"]
                    labels = outputs[0]["labels"].float()  # Convert to float for ONNX
                    return boxes, scores, labels
                else:
                    # Return empty tensors if no detections
                    return torch.zeros(0, 4), torch.zeros(0), torch.zeros(0)

        model = _DetWrapper(raw_model)
        model.eval()
        output_names = ["boxes", "scores", "labels"]

        # Test forward pass
        with torch.no_grad():
            try:
                test_out = model(dummy)
                logger.info(f"Forward test passed. Output shapes: boxes={test_out[0].shape}, scores={test_out[1].shape}")
            except Exception as e:
                logger.warning(f"Detection model forward test failed: {e}")

    else:
        model = raw_model
        output_names = ["output"]

    logger.info(f"Exporting to ONNX: {onnx_path}")

    # Detection models need special handling with legacy exporter
    if info.task == "detect":
        dynamic_axes = {
            "input": {0: "batch"},
            "boxes": {0: "num_detections"},
            "scores": {0: "num_detections"},
            "labels": {0: "num_detections"},
        }
        # Use legacy exporter (dynamo=False) for detection models
        # Detection models have complex control flow that dynamo doesn't handle well
        try:
            torch.onnx.export(
                model,
                dummy,
                str(onnx_path),
                opset_version=11,  # Use opset 11 for better compatibility
                input_names=["input"],
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                export_params=True,
                dynamo=False,  # Force legacy exporter
            )
        except TypeError:
            # Older PyTorch versions don't have dynamo parameter
            torch.onnx.export(
                model,
                dummy,
                str(onnx_path),
                opset_version=11,
                input_names=["input"],
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                export_params=True,
            )
    else:
        try:
            torch.onnx.export(
                model,
                dummy,
                str(onnx_path),
                opset_version=11,
                input_names=["input"],
                output_names=output_names,
                dynamo=False,  # Force legacy exporter (no onnxscript needed)
            )
        except TypeError:
            # Older PyTorch versions don't have dynamo parameter
            torch.onnx.export(
                model,
                dummy,
                str(onnx_path),
                opset_version=11,
                input_names=["input"],
                output_names=output_names,
            )

    logger.info(f"Exported: {onnx_path}")

    # Save labels from weights metadata
    try:
        categories = weights_enum.meta.get("categories", [])
        if categories:
            labels_path = onnx_path.with_suffix(".txt")
            with open(labels_path, "w") as f:
                for cat in categories:
                    f.write(cat + "\n")
            logger.info(f"Saved {len(categories)} labels: {labels_path}")
    except Exception:
        pass

    return onnx_path


def _download_file(url: str, dest_path: Path, desc: str = None) -> Path:
    """Download a file with progress bar."""
    import requests
    from tqdm import tqdm

    if desc is None:
        desc = dest_path.name

    logger.info(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    logger.info(f"Downloaded to: {dest_path}")
    return dest_path


# Minimum expected ONNX file size (1 KB) to catch corrupted/empty downloads
_MIN_ONNX_SIZE = 1024


def _download_onnx_direct(name: str, onnx_url: str, onnx_path: Path, force: bool = False) -> Path:
    """Download a pre-converted ONNX file directly from a URL.

    Args:
        name: Model name (for display purposes)
        onnx_url: URL to the pre-converted ONNX file
        onnx_path: Local path to save the ONNX file
        force: Force re-download even if cached

    Returns:
        Path to the downloaded ONNX file

    Raises:
        RuntimeError: If the download fails or the file is too small
    """
    if onnx_path.exists() and not force:
        logger.info(f"Model already cached: {onnx_path}")
        return onnx_path

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading pre-converted ONNX for {name}...")
    _download_file(onnx_url, onnx_path, desc=f"{name}.onnx")

    # Basic validation: check file size
    file_size = onnx_path.stat().st_size
    if file_size < _MIN_ONNX_SIZE:
        onnx_path.unlink()
        raise RuntimeError(
            f"Downloaded ONNX file is too small ({file_size} bytes), "
            f"possibly corrupted or unavailable. URL: {onnx_url}"
        )

    logger.info(f"Downloaded pre-converted ONNX: {onnx_path} ({file_size / 1024 / 1024:.1f} MB)")
    return onnx_path


# YOLOX model name to config mapping
_YOLOX_MODEL_MAP = {
    "yolox-nano": ("yolox_nano", 0.33, 0.25),  # depth, width
    "yolox-tiny": ("yolox_tiny", 0.33, 0.375),
    "yolox-s": ("yolox_s", 0.33, 0.50),
    "yolox-m": ("yolox_m", 0.67, 0.75),
    "yolox-l": ("yolox_l", 1.0, 1.0),
}


def _export_yolox_onnx(name: str, onnx_path: Path, info: ModelInfo, force: bool = False) -> Path:
    """Export YOLOX model to ONNX format."""
    cache_dir = _get_cache_dir()
    pth_path = cache_dir / f"{name}.pth"

    # Download weights if needed
    if not pth_path.exists() or force:
        _download_file(info.url, pth_path, name)

    # Check if ONNX already exists
    if onnx_path.exists() and not force:
        logger.info(f"Model already cached: {onnx_path}")
        return onnx_path

    # Try to export using yolox package
    try:
        import torch

        logger.info(f"Exporting YOLOX to ONNX: {onnx_path}")

        # Try importing yolox
        try:
            from yolox.exp import get_exp
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

            # Get model config
            exp_name = name.replace("-", "_")
            exp = get_exp(None, exp_name)

            # Build model
            model = exp.get_model()
            model.eval()

            # Load weights
            ckpt = torch.load(str(pth_path), map_location="cpu")
            model.load_state_dict(ckpt["model"])

        except ImportError:
            # Fallback: Build model manually without yolox package
            logger.warning("yolox package not found, using manual model construction")

            if name not in _YOLOX_MODEL_MAP:
                raise ValueError(f"Unknown YOLOX model: {name}")

            _, depth, width = _YOLOX_MODEL_MAP[name]

            # Import PyTorch and build a simple wrapper
            # This requires the user to have exported the model already
            # or we provide pre-converted ONNX files
            raise ImportError(
                f"YOLOX package required for first-time model export. "
                f"Install with: pip install yolox\n"
                f"Or download pre-converted ONNX from: "
                f"https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime"
            )

        # Export to ONNX
        h, w = info.input_size
        dummy_input = torch.randn(1, 3, h, w)

        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            opset_version=11,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes={
                "images": {0: "batch"},
                "output": {0: "batch"},
            },
        )

        logger.info(f"Exported: {onnx_path}")
        return onnx_path

    except ImportError as e:
        raise ImportError(str(e))


# RT-DETR model configurations
_RTDETR_MODEL_MAP = {
    "rtdetr-l": "rtdetr_r50vd_6x_coco",
    "rtdetr-x": "rtdetr_r101vd_6x_coco",
}


def _export_rtdetr_onnx(name: str, onnx_path: Path, info: ModelInfo, force: bool = False) -> Path:
    """Export RT-DETR model to ONNX format."""
    cache_dir = _get_cache_dir()

    # Check if ONNX already exists
    if onnx_path.exists() and not force:
        logger.info(f"Model already cached: {onnx_path}")
        return onnx_path

    logger.info(f"Exporting RT-DETR to ONNX: {onnx_path}")

    try:
        import torch

        # Try to use paddle2onnx or provide pre-converted URL
        try:
            import paddle
            from ppdet.core.workspace import load_config, create

            # This requires paddlepaddle and ppdet
            config_name = _RTDETR_MODEL_MAP.get(name)
            if not config_name:
                raise ValueError(f"Unknown RT-DETR model: {name}")

            # Download and convert
            logger.info("RT-DETR conversion requires PaddlePaddle and PaddleDetection")
            raise ImportError(
                f"PaddlePaddle required for RT-DETR model export. "
                f"Install with: pip install paddlepaddle paddle2onnx\n"
                f"Or download pre-converted ONNX from PaddleDetection model zoo."
            )

        except ImportError:
            # RT-DETR requires PaddlePaddle for conversion
            # Provide clear instructions
            raise ImportError(
                f"RT-DETR model '{name}' requires PaddlePaddle for conversion.\n\n"
                f"Option 1: Install PaddlePaddle and convert:\n"
                f"  pip install paddlepaddle paddledet paddle2onnx\n"
                f"  paddle2onnx --model_dir ... --save_file {onnx_path}\n\n"
                f"Option 2: Download pre-converted ONNX from PaddleDetection:\n"
                f"  https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.7/configs/rtdetr\n\n"
                f"Option 3: Use PyTorch RT-DETR from torchvision (v0.16+):\n"
                f"  import torchvision.models.detection as det\n"
                f"  model = det.rtdetr_resnet50(weights='DEFAULT')"
            )

    except ImportError as e:
        raise ImportError(str(e))


# RTMPose model configurations
_RTMPOSE_MODEL_MAP = {
    "rtmpose-s": {
        "config": "rtmpose-s_8xb256-420e_coco-256x192",
    },
    "rtmpose-m": {
        "config": "rtmpose-m_8xb256-420e_coco-256x192",
    },
}


def _export_mmpose_onnx(name: str, onnx_path: Path, info: ModelInfo, force: bool = False) -> Path:
    """Export MMPose model (RTMPose) to ONNX format."""
    cache_dir = _get_cache_dir()

    # Check if ONNX already exists
    if onnx_path.exists() and not force:
        logger.info(f"Model already cached: {onnx_path}")
        return onnx_path

    logger.info(f"Exporting RTMPose to ONNX: {onnx_path}")

    if name not in _RTMPOSE_MODEL_MAP:
        raise ValueError(f"Unknown RTMPose model: {name}")

    model_config = _RTMPOSE_MODEL_MAP[name]

    # RTMPose requires mmpose/mmdeploy for conversion
    # Provide clear instructions
    raise ImportError(
        f"RTMPose model '{name}' requires conversion from PyTorch.\n\n"
        f"Option 1: Use mmdeploy to convert:\n"
        f"  pip install mmdeploy mmpose\n"
        f"  python -m mmdeploy.backend.onnxruntime ...\n\n"
        f"Option 2: Download pre-converted ONNX from:\n"
        f"  https://github.com/open-mmlab/mmdeploy/tree/main/configs/mmpose\n\n"
        f"Option 3: Use the PyTorch weights directly with mmpose:\n"
        f"  pip install mmpose\n"
        f"  Weight URL: {info.url}"
    )


def load(
    name: str,
    device: str = "auto",
    format: str = "onnx",
    **kwargs
):
    """
    Load model from zoo.

    Args:
        name: Model name or path
        device: Target device
        format: Model format
        **kwargs: Additional arguments for ivit.load()

    Returns:
        Loaded model ready for inference

    Examples:
        >>> model = ivit.zoo.load("yolox-s")
        >>> results = model("image.jpg")
        >>>
        >>> model = ivit.zoo.load("yolox-s", device="cuda:0")
        >>> model = ivit.zoo.load("yolox-s", device="npu")
    """
    import ivit

    # Check if it's a zoo model or a path
    if name in MODEL_REGISTRY:
        # Download if needed
        model_path = download(name, format=format)
    else:
        # Treat as path
        model_path = Path(name)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {name}")

    # Load with ivit
    return ivit.load(str(model_path), device=device, **kwargs)


def print_models(task: Optional[str] = None):
    """
    Print formatted model list.

    Args:
        task: Filter by task type
    """
    models = list_models(task)

    print()
    print("=" * 88)
    print("iVIT Model Zoo")
    print("=" * 88)
    print()
    print(f"{'Model':<28} {'Task':<12} {'Input':<12} {'Description':<30}")
    print("-" * 88)

    for name in models:
        info = MODEL_REGISTRY[name]
        input_str = f"{info.input_size[0]}x{info.input_size[1]}"
        desc = info.description[:28] + ".." if len(info.description) > 30 else info.description
        print(f"{name:<28} {info.task:<12} {input_str:<12} {desc:<30}")

    print("-" * 88)
    print(f"Total: {len(models)} models")
    print()
    print("All models use commercial-friendly licenses (Apache-2.0, BSD-3-Clause)")
    print()
    print("Usage: model = ivit.zoo.load('yolox-s')")
    print()
