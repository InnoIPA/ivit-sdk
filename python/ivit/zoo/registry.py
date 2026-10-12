"""
Model Zoo Registry - Model catalog and download management.

All models in the registry use commercial-friendly licenses (Apache-2.0, BSD-3-Clause, MIT).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
import hashlib
import json
import os
import re
import shutil
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Model information in the zoo."""
    name: str
    task: str  # detect, classify, segment
    description: str
    input_size: tuple  # (H, W)
    num_classes: int
    url: Optional[str] = None        # 原始框架權重 (.pth, .pdparams)
    onnx_url: Optional[str] = None   # 預轉換 ONNX 下載 URL
    sha256: Optional[str] = None     # onnx_url 檔案的 SHA256；未設定時查 checksums.json
    source: str = "torchvision"  # torchvision, megvii, custom
    license: str = "Apache-2.0"
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    # ivit.load() settings that decode this model's ONNX correctly (applied by zoo.load()). Empty
    # means the SDK has no decoder for the model's outputs: download() works, zoo.load() refuses.
    load_options: Dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        """True when zoo.load() can return a model that produces decoded results."""
        return bool(self.load_options)



# COCO-80 class names in the order YOLOX predicts them.
COCO_LABELS = [name.replace("-", " ") for name in """person bicycle car motorcycle airplane bus train truck boat traffic-light fire-hydrant stop-sign parking-meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard sports-ball kite baseball-bat baseball-glove skateboard surfboard tennis-racket bottle wine-glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot hot-dog pizza donut cake chair couch potted-plant bed dining-table toilet tv laptop mouse remote keyboard cell-phone microwave oven toaster sink refrigerator book clock vase scissors teddy-bear hair-drier toothbrush""".split()]
assert len(COCO_LABELS) == 80

# How each exported ONNX must be fed and decoded. Verified end to end against the original
# framework on examples/data/images/bus.jpg (YOLOX on OpenVINO CPU/iGPU and TensorRT; torchvision
# classification and segmentation on OpenVINO CPU).
_YOLOX_LOAD = {
    "task": "detection", "decoder": "yolox_decoded", "input_color": "bgr", "input_scale": 1.0,
    "resize_mode": "letterbox_top_left", "labels": COCO_LABELS,
}
_IMAGENET = {"input_color": "rgb", "input_scale": 1.0 / 255.0,
             "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
# Class names come from the .labels.txt sidecar written at export time (torchvision weights meta).
_TORCHVISION_CLASSIFY_LOAD = {"task": "classification", "decoder": "logits", **_IMAGENET}
_TORCHVISION_SEGMENT_LOAD = {"task": "segmentation", "decoder": "logits", **_IMAGENET}

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
        url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth",
        onnx_url="https://drive.google.com/file/d/1q4v6wUYRTQI6u7bwX7naRdGfC2HSKHLp/view",
        source="megvii",
        license="Apache-2.0",
        metrics={"mAP50-95": 25.8, "params_m": 0.91, "flops_g": 1.08},
        tags=["yolox", "detection", "fast", "edge", "commercial-friendly"],
        load_options=_YOLOX_LOAD,
    ),
    "yolox-tiny": ModelInfo(
        name="yolox-tiny",
        task="detect",
        description="YOLOX Tiny - Lightweight model for edge devices",
        input_size=(416, 416),
        num_classes=80,
        url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth",
        onnx_url="https://drive.google.com/file/d/1qWtUuSGoGyRVmvR40wu5atzrVs2SB6yG/view",
        source="megvii",
        license="Apache-2.0",
        metrics={"mAP50-95": 32.8, "params_m": 5.06, "flops_g": 6.45},
        tags=["yolox", "detection", "fast", "edge", "commercial-friendly"],
        load_options=_YOLOX_LOAD,
    ),
    "yolox-s": ModelInfo(
        name="yolox-s",
        task="detect",
        description="YOLOX Small - Balance of speed and accuracy",
        input_size=(640, 640),
        num_classes=80,
        url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth",
        onnx_url="https://drive.google.com/file/d/14tYG_2l_Wjp_z2aniRXuMxWhXTW-VXCm/view",
        source="megvii",
        license="Apache-2.0",
        metrics={"mAP50-95": 40.5, "params_m": 9.0, "flops_g": 26.8},
        tags=["yolox", "detection", "balanced", "commercial-friendly"],
        load_options=_YOLOX_LOAD,
    ),
    "yolox-m": ModelInfo(
        name="yolox-m",
        task="detect",
        description="YOLOX Medium - Higher accuracy",
        input_size=(640, 640),
        num_classes=80,
        url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth",
        onnx_url="https://drive.google.com/file/d/1Ykp-gsYbynDmOAEn1dWdyX4vAn4o-X07/view",
        source="megvii",
        license="Apache-2.0",
        metrics={"mAP50-95": 46.9, "params_m": 25.3, "flops_g": 73.8},
        tags=["yolox", "detection", "accurate", "commercial-friendly"],
        load_options=_YOLOX_LOAD,
    ),
    "yolox-l": ModelInfo(
        name="yolox-l",
        task="detect",
        description="YOLOX Large - High accuracy for server deployment",
        input_size=(640, 640),
        num_classes=80,
        url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth",
        onnx_url="https://drive.google.com/file/d/1nyI9G0D281vFJk0imLqm6N1m8zMDgNj3/view",
        source="megvii",
        license="Apache-2.0",
        metrics={"mAP50-95": 49.7, "params_m": 54.2, "flops_g": 155.6},
        tags=["yolox", "detection", "accurate", "server", "commercial-friendly"],
        load_options=_YOLOX_LOAD,
    ),

    # =========================================================================
    # Detection Models (torchvision, BSD-3-Clause) - raw: the SDK has no decoder for their
    # boxes/scores/labels outputs. Their output shapes depend on the data (NonZero inside If), so
    # TensorRT cannot build them; on OpenVINO they run through C++ Model::infer_raw.
    # =========================================================================
    "fasterrcnn-mobilenet": ModelInfo(
        name="fasterrcnn-mobilenet",
        task="detect",
        description="Faster R-CNN MobileNetV3 - Fast detection for edge",
        input_size=(640, 640),
        num_classes=91,  # COCO classes
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
        source="torchvision",
        license="BSD-3-Clause",
        metrics={"top1": 69.8, "top5": 89.1, "params_m": 11.7},
        tags=["classification", "imagenet", "classic", "fast", "commercial-friendly", "ready-to-use"],
        load_options=_TORCHVISION_CLASSIFY_LOAD,
    ),
    "resnet50": ModelInfo(
        name="resnet50",
        task="classify",
        description="ResNet-50 ImageNet Classification",
        input_size=(224, 224),
        num_classes=1000,
        source="torchvision",
        license="BSD-3-Clause",
        metrics={"top1": 76.1, "top5": 92.9, "params_m": 25.6},
        tags=["classification", "imagenet", "classic", "commercial-friendly", "ready-to-use"],
        load_options=_TORCHVISION_CLASSIFY_LOAD,
    ),
    "mobilenetv3": ModelInfo(
        name="mobilenetv3",
        task="classify",
        description="MobileNetV3 Large - Efficient mobile classification",
        input_size=(224, 224),
        num_classes=1000,
        source="torchvision",
        license="BSD-3-Clause",
        metrics={"top1": 74.0, "top5": 91.3, "params_m": 5.5},
        tags=["classification", "mobile", "efficient", "edge", "commercial-friendly", "ready-to-use"],
        load_options=_TORCHVISION_CLASSIFY_LOAD,
    ),
    "efficientnet-b0": ModelInfo(
        name="efficientnet-b0",
        task="classify",
        description="EfficientNet-B0 - Efficient scaling",
        input_size=(224, 224),
        num_classes=1000,
        source="torchvision",
        license="Apache-2.0",
        metrics={"top1": 77.1, "top5": 93.3, "params_m": 5.3},
        tags=["classification", "efficient", "commercial-friendly", "ready-to-use"],
        load_options=_TORCHVISION_CLASSIFY_LOAD,
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
        source="torchvision",
        license="BSD-3-Clause",
        metrics={"mIoU": 66.4},
        tags=["segmentation", "semantic", "voc", "classic", "commercial-friendly", "ready-to-use"],
        load_options=_TORCHVISION_SEGMENT_LOAD,
    ),
    "deeplabv3-mobilenetv3": ModelInfo(
        name="deeplabv3-mobilenetv3",
        task="segment",
        description="DeepLabV3 MobileNetV3 - Lightweight semantic segmentation",
        input_size=(520, 520),
        num_classes=21,
        source="torchvision",
        license="BSD-3-Clause",
        metrics={"mIoU": 60.3},
        tags=["segmentation", "semantic", "voc", "mobile", "efficient", "commercial-friendly", "ready-to-use"],
        load_options=_TORCHVISION_SEGMENT_LOAD,
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
        task: Filter by task type: "detect", "classify" or "segment"

    Returns:
        List of model names

    Raises:
        ValueError: task is not one of the zoo's task names

    Examples:
        >>> ivit.zoo.list_models()
        ['deeplabv3-mobilenetv3', 'deeplabv3-resnet50', 'efficientnet-b0', ...]
        >>> ivit.zoo.list_models(task="detect")
        ['fasterrcnn-mobilenet', 'fasterrcnn-resnet50', ...]
    """
    tasks = sorted({info.task for info in MODEL_REGISTRY.values()})
    if task is not None and task not in tasks:
        raise ValueError(f"Unknown zoo task {task!r}; use one of {', '.join(tasks)}")
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
        ['yolov8n', 'yolov8s', 'yolov8m', ...]
        >>> ivit.zoo.search("edge")
        ['yolov8n', 'yolov8n-seg', 'yolov8n-cls', ...]
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
        >>> info = ivit.zoo.get_model_info("yolov8n")
        >>> print(f"Task: {info.task}, Input: {info.input_size}")
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Model not found: {name}. Use ivit.zoo.list_models() to see available models.")
    return MODEL_REGISTRY[name]


def download(
    name: str,
    force: bool = False,
    from_source: bool = False,
    sha256: Optional[str] = None,
) -> Path:
    """
    Download a zoo model as ONNX. ivit.load() converts it for the target
    backend (and caches the result) when it is loaded.

    Args:
        name: Model name
        force: Force re-download
        from_source: If True, download original weights and convert locally
            instead of using pre-converted ONNX files
        sha256: Expected SHA256 of a pre-converted ONNX download. Defaults to the
            value in ivit/zoo/checksums.json. Set IVIT_ZOO_REQUIRE_CHECKSUM=1 to
            refuse downloads that have no known checksum.

    Returns:
        Path to downloaded model

    Examples:
        >>> path = ivit.zoo.download("yolox-s")
        >>> print(path)
        /home/user/.cache/ivit/models/yolox-s.onnx
    """
    info = get_model_info(name)

    cache_dir = _get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / f"{name}.onnx"

    # Direct ONNX download when the model has an onnx_url and from_source is not set.
    # It handles its own cache check so cached files are integrity-verified too.
    if info.onnx_url and not from_source:
        path = _download_onnx_direct(name, info.onnx_url, model_path, force, sha256=sha256)
        _write_labels_sidecar(path, info)
        return path

    if sha256 is not None:
        raise ValueError(
            "sha256 only applies to pre-converted ONNX downloads "
            "(from_source=False, model with an onnx_url)"
        )

    # Check if already exists
    if model_path.exists() and not force:
        logger.info(f"Model already cached: {model_path}")
        return model_path

    # Route to appropriate exporter based on source
    if info.source == "megvii":
        model_path = _export_yolox_onnx(name, model_path, info, force)
    elif info.source == "torchvision":
        model_path = _export_torchvision_onnx(name, model_path, info)

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
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            opset_version=18,
            input_names=["input"],
            output_names=output_names,
        )

    _inline_external_data(onnx_path)
    categories = getattr(weights_enum, "meta", {}).get("categories")
    if categories and info.task in ("classify", "segment"):
        # ivit.load() picks up <model>.labels.txt automatically.
        onnx_path.with_suffix(".labels.txt").write_text(
            "\n".join(c.replace("\n", " ") for c in categories) + "\n", encoding="utf-8")
    logger.info(f"Exported: {onnx_path}")
    return onnx_path


def _write_labels_sidecar(onnx_path: Path, info: ModelInfo) -> None:
    """Write <model>.labels.txt from the registry so C/C++ users of the bare file see class names.

    The SDK reads the sidecar automatically. An existing sidecar is left alone.
    """
    labels = info.load_options.get("labels")
    sidecar = onnx_path.with_suffix(".labels.txt")
    if labels and not sidecar.exists():
        sidecar.write_text("\n".join(labels) + "\n", encoding="utf-8")


def _inline_external_data(onnx_path: Path) -> Path:
    """Fold weights stored in a ``<name>.onnx.data`` sidecar back into the .onnx file.

    Recent torch versions export large initializers to a separate file. The zoo caches,
    checksums and hands out a single path, so a sidecar would leave a graph that cannot
    load by itself and weights that no checksum covers.
    """
    sidecar = onnx_path.with_name(onnx_path.name + ".data")
    if not sidecar.exists():
        return onnx_path
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=True)
    if model.ByteSize() >= 2 * 1024**3:
        raise RuntimeError(
            f"{onnx_path.name} is larger than the 2 GB ONNX single-file limit; "
            "cannot embed its weights"
        )
    merged = onnx_path.with_name(onnx_path.name + ".merged")
    try:
        onnx.save_model(model, str(merged), save_as_external_data=False)
        os.replace(merged, onnx_path)
    finally:
        merged.unlink(missing_ok=True)
    sidecar.unlink()
    return onnx_path


_GOOGLE_DRIVE_HOSTS = ("drive.google.com", "docs.google.com", "drive.usercontent.google.com")
_GOOGLE_DRIVE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{10,}$")


def _google_drive_file_id(url: str) -> Optional[str]:
    """File ID of a Google Drive share or download link, or None for any other URL."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in _GOOGLE_DRIVE_HOSTS:
        return None
    parts = parsed.path.split("/")
    if "d" in parts and parts.index("d") + 1 < len(parts):  # /file/d/<id>/view
        candidate = parts[parts.index("d") + 1]
    else:  # /open?id=<id>, /uc?id=<id>, /download?id=<id>
        candidate = parse_qs(parsed.query).get("id", [""])[0]
    return candidate if _GOOGLE_DRIVE_ID_RE.match(candidate) else None


def _direct_download_url(url: str) -> str:
    """Turn a Google Drive share link into its direct download URL; other URLs are unchanged.

    ``confirm=t`` skips the "can't scan this file for viruses" page Drive shows for files over
    about 100 MB. Google may change this behaviour; the HTML check in _download_file() and the
    SHA256 check catch it if they do.
    """
    file_id = _google_drive_file_id(url)
    if file_id is None:
        return url
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def _download_file(url: str, dest_path: Path, desc: str = None) -> Path:
    """Download a file with progress bar."""
    import requests
    from tqdm import tqdm

    if desc is None:
        desc = dest_path.name

    url = _direct_download_url(url)
    logger.info(f"Downloading from {url}...")
    # (connect, read) timeouts: a stalled server must not hang the application.
    response = requests.get(url, stream=True, timeout=(10, 60))
    drive_hint = (
        " Check that the Google Drive file is shared as 'Anyone with the link' and the link is "
        "right; a popular file can also hit Drive's download quota (try again later or set "
        "IVIT_ZOO_MIRROR)."
        if _google_drive_file_id(url) else ""
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        if not drive_hint:
            raise
        raise RuntimeError(f"{error}.{drive_hint}") from error
    if response.headers.get("content-type", "").lower().startswith("text/html"):
        response.close()
        raise RuntimeError(f"Expected a file but got a web page from {url}.{drive_hint}")

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

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHECKSUMS_PATH = Path(__file__).with_name("checksums.json")


def _normalize_sha256(value: str, source: str) -> str:
    digest = str(value).strip().lower()
    if not _SHA256_RE.match(digest):
        raise ValueError(f"Invalid SHA256 from {source}: expected 64 hex characters")
    return digest


def _load_checksums() -> Dict[str, str]:
    """Load name -> SHA256 for pre-converted ONNX files (empty if absent)."""
    try:
        data = json.loads(_CHECKSUMS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    entries = data.get("sha256", {}) if isinstance(data, dict) else {}
    return {
        name: _normalize_sha256(value, f"{_CHECKSUMS_PATH.name}:{name}")
        for name, value in entries.items()
    }


def _expected_sha256(name: str, explicit: Optional[str] = None) -> Optional[str]:
    """Resolve the expected digest: explicit argument > ModelInfo > checksums.json."""
    if explicit:
        return _normalize_sha256(explicit, "sha256 argument")
    info = MODEL_REGISTRY.get(name)
    if info is not None and info.sha256:
        return _normalize_sha256(info.sha256, f"registry:{name}")
    return _load_checksums().get(name)


def _require_checksum() -> bool:
    return os.environ.get("IVIT_ZOO_REQUIRE_CHECKSUM", "").strip().lower() in ("1", "true", "yes")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mirror_source(filename: str) -> Optional[str]:
    """Mirror location of ``filename`` (e.g. "yolox-s.onnx") from IVIT_ZOO_MIRROR, or None.

    IVIT_ZOO_MIRROR is an http(s) base URL or a local directory (``file://`` or a plain
    path, for air-gapped sites). Files are named after the model, not taken from ``onnx_url``,
    because share links such as Google Drive's carry no file name.
    """
    base = os.environ.get("IVIT_ZOO_MIRROR", "").strip()
    if not base or not filename:
        return None
    if base.startswith("file://"):
        base = base[len("file://"):]
    return base.rstrip("/") + "/" + filename


def _fetch_into(source: str, partial: Path, desc: str) -> None:
    if source.startswith(("http://", "https://")):
        _download_file(source, partial, desc=desc)
    else:
        shutil.copyfile(source, partial)


def _download_onnx_direct(
    name: str,
    onnx_url: str,
    onnx_path: Path,
    force: bool = False,
    sha256: Optional[str] = None,
) -> Path:
    """Download a pre-converted ONNX file directly from a URL.

    The file is written to a temporary name, checked, and only then moved into
    place, so a failed or tampered download never leaves a usable-looking file.

    If IVIT_ZOO_MIRROR is set, the mirror is tried first and ``onnx_url`` is the fallback
    when the mirror is unreachable or serves an unusable file. A checksum mismatch is
    never retried elsewhere: it means the copy was corrupted or tampered with.

    Args:
        name: Model name (for display purposes)
        onnx_url: URL to the pre-converted ONNX file
        onnx_path: Local path to save the ONNX file
        force: Force re-download even if cached
        sha256: Expected SHA256 (overrides the registry / checksums.json value)

    Returns:
        Path to the downloaded ONNX file

    Raises:
        RuntimeError: If the download fails, is too small, or fails verification
            (or no checksum is known while IVIT_ZOO_REQUIRE_CHECKSUM is set)
    """
    expected = _expected_sha256(name, sha256)
    if expected is None and _require_checksum():
        raise RuntimeError(
            f"No SHA256 is registered for '{name}' and IVIT_ZOO_REQUIRE_CHECKSUM is set; "
            "pass sha256=... or add it to ivit/zoo/checksums.json"
        )

    if onnx_path.exists() and not force:
        if expected is None:
            logger.warning(f"Using cached model without integrity check: {onnx_path}")
            return onnx_path
        if _sha256_file(onnx_path) == expected:
            logger.info(f"Model already cached and verified: {onnx_path}")
            return onnx_path
        logger.warning(f"Cached model does not match the expected SHA256, re-downloading: {onnx_path}")
        onnx_path.unlink()

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    partial = onnx_path.with_name(onnx_path.name + ".part")

    sources = [src for src in (_mirror_source(f"{name}.onnx"), onnx_url) if src]
    used = onnx_url
    logger.info(f"Downloading pre-converted ONNX for {name}...")
    try:
        for index, source in enumerate(sources):
            try:
                _fetch_into(source, partial, f"{name}.onnx")
                # Basic validation: check file size (catches HTML error pages)
                file_size = partial.stat().st_size
                if file_size < _MIN_ONNX_SIZE:
                    raise RuntimeError(
                        f"Downloaded ONNX file is too small ({file_size} bytes), "
                        f"possibly corrupted or unavailable. URL: {source}"
                    )
                used = source
                break
            except Exception as error:
                partial.unlink(missing_ok=True)
                if index == len(sources) - 1:
                    raise
                logger.warning(f"Mirror unavailable ({error}); falling back to {onnx_url}")

        if expected is not None:
            actual = _sha256_file(partial)
            if actual != expected:
                raise RuntimeError(
                    f"SHA256 mismatch for '{name}': expected {expected}, got {actual}. "
                    f"The file was discarded. Source: {used}"
                )
        else:
            logger.warning(
                f"No SHA256 registered for '{name}'; downloaded file is NOT integrity-checked"
            )
        os.replace(partial, onnx_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    logger.info(f"Downloaded pre-converted ONNX: {onnx_path} ({file_size / 1024 / 1024:.1f} MB)")
    return onnx_path


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
            raise ImportError(
                "Exporting YOLOX from its weights needs the yolox package. Its PyPI release "
                "(0.3.0) does not build with current setuptools; put the YOLOX source tree on "
                "PYTHONPATH instead (e.g. unpack `pip download --no-deps --no-binary :all: yolox`). "
                "Simpler: download the verified pre-converted ONNX (from_source=False)."
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

        _inline_external_data(onnx_path)
        logger.info(f"Exported: {onnx_path}")
        return onnx_path

    except ImportError as e:
        raise ImportError(str(e))


def load(
    name: str,
    device: str = "auto",
    **kwargs
):
    """
    Load a zoo model ready for inference.

    The model's verified load settings (task, decoder, color order, normalization, labels) are
    applied automatically; any keyword argument overrides them.

    Args:
        name: Model name, or a path to a model file (loaded as-is)
        device: Target device
        **kwargs: Additional arguments for ivit.load()

    Returns:
        Loaded model ready for inference

    Raises:
        ValueError: The model has no SDK decoder (see ModelInfo.ready) and no task was given.

    Examples:
        >>> model = ivit.zoo.load("yolox-s")
        >>> results = model.predict(cv2.imread("image.jpg"))
        >>> model = ivit.zoo.load("resnet18", device="gpu:0")
    """
    import ivit

    if name in MODEL_REGISTRY:
        info = MODEL_REGISTRY[name]
        if not info.ready and "task" not in kwargs:
            raise ValueError(
                f"'{name}' has no SDK decoder for its outputs, so zoo.load() cannot return decoded "
                f"results. Download it with ivit.zoo.download('{name}') and run it in your own "
                "runtime (Python has no raw-output API; C++ has Model::infer_raw), or pass task=/decoder= "
                "explicitly if you know a matching contract."
            )
        model_path = download(name)
        options = {**info.load_options, **kwargs}
    else:
        model_path = Path(name)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {name}")
        options = kwargs

    return ivit.load(str(model_path), device=device, **options)


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
    print(f"{'Model':<28} {'Task':<12} {'Input':<12} {'SDK':<6} {'Description':<26}")
    print("-" * 88)

    for name in models:
        info = MODEL_REGISTRY[name]
        input_str = f"{info.input_size[0]}x{info.input_size[1]}"
        desc = info.description[:24] + ".." if len(info.description) > 26 else info.description
        status = "ready" if info.ready else "raw"
        print(f"{name:<28} {info.task:<12} {input_str:<12} {status:<6} {desc:<26}")

    print("-" * 88)
    print(f"Total: {len(models)} models")
    print()
    print("All models use commercial-friendly licenses (Apache-2.0, BSD-3-Clause)")
    print()
    print("SDK: ready = zoo.load() returns decoded results; raw = download only, decode it yourself")
    print("Usage: model = ivit.zoo.load('yolox-s')")
    print()
