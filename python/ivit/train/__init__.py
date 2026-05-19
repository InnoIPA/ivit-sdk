"""
Training module for iVIT-SDK.

Provides transfer learning for classification, object detection (YOLO 系列),
and semantic segmentation。
"""

from .dataset import (
    BaseDataset,
    ImageFolderDataset,
    COCODataset,
    YOLODataset,
    SegmentationDataset,
    split_dataset,
)

from .base import BaseTrainer
from .classification import ClassificationTrainer
from .segmentation import SegmentationTrainer

# DetectionTrainer 依賴 optional 套件（yolox），lazy import 失敗時降級
try:
    from .detection import DetectionTrainer  # noqa: F401
    _DETECTION_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    DetectionTrainer = None  # type: ignore[assignment]
    _DETECTION_AVAILABLE = False

# 向後相容
from .classification import ClassificationTrainer as Trainer

from .callbacks import (
    TrainingCallback,
    EarlyStopping,
    ModelCheckpoint,
    ProgressLogger,
    LRScheduler,
    TensorBoardLogger,
)

from .augmentation import (
    Compose,
    Resize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    ColorJitter,
    Normalize,
    ToTensor,
    get_default_augmentation,
    get_train_augmentation,
    get_val_augmentation,
)

from .exporter import ModelExporter

__all__ = [
    # Datasets
    "BaseDataset",
    "ImageFolderDataset",
    "COCODataset",
    "YOLODataset",
    "SegmentationDataset",
    "split_dataset",
    # Trainers
    "BaseTrainer",
    "ClassificationTrainer",
    "SegmentationTrainer",
    "DetectionTrainer",
    "Trainer",  # alias = ClassificationTrainer
    # Callbacks
    "TrainingCallback",
    "EarlyStopping",
    "ModelCheckpoint",
    "ProgressLogger",
    "LRScheduler",
    "TensorBoardLogger",
    # Augmentation
    "Compose",
    "Resize",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomRotation",
    "ColorJitter",
    "Normalize",
    "ToTensor",
    "get_default_augmentation",
    "get_train_augmentation",
    "get_val_augmentation",
    # Exporter
    "ModelExporter",
]
