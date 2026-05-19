"""
ClassificationTrainer：影像分類 transfer learning。

從 torchvision 載入預訓練分類模型（ResNet / EfficientNet / MobileNet / VGG / DenseNet），
更換最後一層 classifier head 對齊 dataset 的類別數，並用 CrossEntropyLoss 訓練。
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging

import numpy as np

from .base import BaseTrainer
from .callbacks import TrainingCallback
from .dataset import BaseDataset
from .augmentation import get_train_augmentation, get_val_augmentation

logger = logging.getLogger(__name__)


class ClassificationTrainer(BaseTrainer):
    """
    分類專用 Trainer。

    Examples:
        >>> from ivit.train import ClassificationTrainer, ImageFolderDataset
        >>> dataset = ImageFolderDataset("./data", train_split=0.8)
        >>> trainer = ClassificationTrainer(
        ...     model="resnet50",
        ...     dataset=dataset,
        ...     epochs=20,
        ...     learning_rate=0.001,
        ...     device="cuda:0",
        ... )
        >>> trainer.fit()
        >>> metrics = trainer.evaluate()
        >>> trainer.export("model.onnx")
    """

    def __init__(
        self,
        model: Union[str, Any],
        dataset: BaseDataset,
        val_dataset: Optional[BaseDataset] = None,
        epochs: int = 10,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        optimizer: str = "adam",
        device: str = "cuda:0",
        freeze_backbone: bool = True,
        num_workers: int = 4,
        image_size: int = 224,
        **kwargs,
    ):
        self.image_size = image_size
        super().__init__(
            model=model,
            dataset=dataset,
            val_dataset=val_dataset,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            optimizer=optimizer,
            device=device,
            num_workers=num_workers,
            freeze_backbone=freeze_backbone,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # 模型建構
    # ------------------------------------------------------------------

    def _build_model(self, spec: Union[str, Any]) -> Any:
        if isinstance(spec, str):
            model = self._load_pretrained_model(spec)
        else:
            model = spec

        num_classes = self.dataset.num_classes
        self._modify_classifier(model, num_classes)

        if self.freeze_backbone:
            self._freeze_backbone_layers(model)

        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model: {total_params:,} total params, {trainable:,} trainable")
        return model

    def _load_pretrained_model(self, model_name: str) -> Any:
        import torchvision.models as models

        name = model_name.lower()
        model_map = {
            "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
            "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
            "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
            "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT),
            "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
            "efficientnet_b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
            "efficientnet_b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
            "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
            "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT),
            "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT),
            "vgg16": (models.vgg16, models.VGG16_Weights.DEFAULT),
            "vgg19": (models.vgg19, models.VGG19_Weights.DEFAULT),
            "densenet121": (models.densenet121, models.DenseNet121_Weights.DEFAULT),
        }
        if name not in model_map:
            available = ", ".join(model_map.keys())
            raise ValueError(f"Unknown model: {model_name}. Available: {available}")

        model_fn, weights = model_map[name]
        logger.info(f"Loading pre-trained {name}")
        return model_fn(weights=weights)

    def _modify_classifier(self, model: Any, num_classes: int) -> None:
        import torch.nn as nn

        if hasattr(model, "fc"):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            logger.debug(f"Modified fc layer: {in_features} -> {num_classes}")
        elif hasattr(model, "classifier"):
            classifier = model.classifier
            if isinstance(classifier, nn.Sequential):
                last = classifier[-1]
                if isinstance(last, nn.Linear):
                    in_features = last.in_features
                    classifier[-1] = nn.Linear(in_features, num_classes)
            elif isinstance(classifier, nn.Linear):
                in_features = classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
            logger.debug(f"Modified classifier layer for {num_classes} classes")
        elif hasattr(model, "heads"):
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)
            logger.debug(f"Modified ViT head: {in_features} -> {num_classes}")
        else:
            logger.warning("Could not automatically modify classifier - unknown architecture")

    def _freeze_backbone_layers(self, model: Any) -> None:
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, "fc"):
            for p in model.fc.parameters():
                p.requires_grad = True
        elif hasattr(model, "classifier"):
            for p in model.classifier.parameters():
                p.requires_grad = True
        elif hasattr(model, "heads"):
            for p in model.heads.parameters():
                p.requires_grad = True
        logger.info("Froze backbone layers, only training classifier")

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------

    def _build_dataloaders(self) -> Tuple[Any, Optional[Any]]:
        from torch.utils.data import DataLoader

        train_ds = _ClassifyWrapper(self.dataset, transform=get_train_augmentation(self.image_size))
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.device_str.startswith("cuda"),
        )

        val_loader = None
        if self.val_dataset is not None:
            val_ds = _ClassifyWrapper(self.val_dataset, transform=get_val_augmentation(self.image_size))
            val_loader = DataLoader(
                val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        logger.info(f"Train batches: {len(train_loader)}")
        return train_loader, val_loader

    def _make_eval_loader(self, dataset: BaseDataset) -> Any:
        from torch.utils.data import DataLoader

        eval_ds = _ClassifyWrapper(dataset, transform=get_val_augmentation(self.image_size))
        return DataLoader(eval_ds, batch_size=self.batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # 訓練 / 評估
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int, callbacks: List[TrainingCallback]) -> Dict[str, float]:
        import torch.nn as nn

        criterion = nn.CrossEntropyLoss()
        self.model.train()

        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            for cb in callbacks:
                cb.on_batch_start(self, batch_idx)

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for cb in callbacks:
                cb.on_batch_end(self, batch_idx, batch_loss)

        return {
            "loss": epoch_loss / max(1, len(self.train_loader)),
            "accuracy": correct / max(1, total),
        }

    def _evaluate(self, loader: Any) -> Dict[str, float]:
        import torch
        import torch.nn as nn

        criterion = nn.CrossEntropyLoss()
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return {
            "loss": total_loss / max(1, len(loader)),
            "accuracy": correct / max(1, total),
        }

    # ------------------------------------------------------------------
    # Export 簽章
    # ------------------------------------------------------------------

    def _export_signature(self) -> Dict[str, Any]:
        return {
            "task": "classify",
            "input_shape": (1, 3, self.image_size, self.image_size),
            "class_names": list(self.dataset.class_names),
        }


class _ClassifyWrapper:
    """把 ImageFolderDataset 之類純量 label 的資料集包成 PyTorch DataLoader 相容介面。"""

    def __init__(self, dataset: BaseDataset, transform: Optional[Callable] = None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        import torch

        image, label = self.dataset[idx]
        if self.transform is not None:
            image = self.transform(image)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if image.dim() == 4:
            image = image.squeeze(0)
        return image, label
