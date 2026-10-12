"""
SegmentationTrainer：torchvision DeepLabV3 fine-tuning（semantic segmentation）。

Loss：CrossEntropy + ignore_index=255，可選 aux loss（torchvision 預設多輸出）。
Metric：confusion-matrix 累積後算 per-class IoU、mIoU、pixel accuracy。
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging

import numpy as np

from .base import BaseTrainer
from .callbacks import TrainingCallback
from .dataset import BaseDataset
from .augmentation import (
    Compose,
    Resize,
    RandomHorizontalFlip,
    Normalize,
    ToTensor,
)

logger = logging.getLogger(__name__)


_TORCHVISION_SEG_MODELS = {
    "deeplabv3-resnet50": "deeplabv3_resnet50",
    "deeplabv3_resnet50": "deeplabv3_resnet50",
    "deeplabv3-mobilenetv3": "deeplabv3_mobilenet_v3_large",
    "deeplabv3_mobilenet_v3_large": "deeplabv3_mobilenet_v3_large",
}


class SegmentationTrainer(BaseTrainer):
    """
    Semantic segmentation Trainer。

    Args:
        model: 模型名稱（"deeplabv3-resnet50" / "deeplabv3-mobilenetv3"）或 nn.Module
        dataset: SegmentationDataset 等回傳 (image, {"mask": ...}) 的資料集
        image_size: 訓練解析度（會 letterbox resize）
        aux_loss: 是否使用 torchvision 的 aux classifier（loss 加權 0.4）
        ignore_index: 訓練/評估忽略的 mask 像素值

    Examples:
        >>> from ivit.train import SegmentationTrainer, SegmentationDataset
        >>> ds = SegmentationDataset("./voc/JPEGImages", "./voc/SegmentationClass",
        ...                          class_names=[...])
        >>> tr = SegmentationTrainer(model="deeplabv3-mobilenetv3", dataset=ds,
        ...                          epochs=20, learning_rate=1e-4, device="cuda:0")
        >>> tr.fit()
        >>> tr.export("seg.onnx")
    """

    def __init__(
        self,
        model: Union[str, Any],
        dataset: BaseDataset,
        val_dataset: Optional[BaseDataset] = None,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        optimizer: str = "adamw",
        device: str = "cuda:0",
        freeze_backbone: bool = False,
        num_workers: int = 4,
        image_size: int = 520,
        aux_loss: bool = True,
        ignore_index: int = 255,
        **kwargs,
    ):
        self.image_size = image_size
        self.aux_loss = aux_loss
        self.ignore_index = ignore_index
        # 由 _build_model 在進 super().__init__ 中被呼叫前無法存取，
        # 但 num_classes 由 dataset 決定 — 在 build 中讀。
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
        import torch.nn as nn

        if not isinstance(spec, str):
            return spec

        from torchvision.models import segmentation as tv_seg

        key = spec.lower()
        if key not in _TORCHVISION_SEG_MODELS:
            available = ", ".join(sorted(set(_TORCHVISION_SEG_MODELS.keys())))
            raise ValueError(
                f"Unknown segmentation model: {spec}. Available: {available}"
            )

        builder = getattr(tv_seg, _TORCHVISION_SEG_MODELS[key])
        # 預訓練權重（COCO）含 aux_classifier，因此 weights="DEFAULT" 必須搭配 aux_loss=True；
        # 是否在 loss 計入 aux 改由 self.aux_loss 在訓練迴圈控制。
        model = builder(weights="DEFAULT", aux_loss=True)

        num_classes = self.dataset.num_classes
        # 換 head：classifier[-1] 是 (in, num_classes) 的 1x1 conv
        in_features = model.classifier[-1].in_channels
        model.classifier[-1] = nn.Conv2d(in_features, num_classes, kernel_size=1)
        if getattr(model, "aux_classifier", None) is not None:
            aux_in = model.aux_classifier[-1].in_channels
            model.aux_classifier[-1] = nn.Conv2d(aux_in, num_classes, kernel_size=1)

        if self.freeze_backbone and hasattr(model, "backbone"):
            for p in model.backbone.parameters():
                p.requires_grad = False
            logger.info("Froze backbone, only training classifier head")

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Segmentation model: {total:,} params, {trainable:,} trainable")
        return model

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------

    def _build_dataloaders(self) -> Tuple[Any, Optional[Any]]:
        from torch.utils.data import DataLoader

        train_tf = Compose(
            [
                Resize((self.image_size, self.image_size), keep_ratio=True),
                RandomHorizontalFlip(p=0.5),
                Normalize(),
                ToTensor(),
            ]
        )
        val_tf = Compose(
            [
                Resize((self.image_size, self.image_size), keep_ratio=True),
                Normalize(),
                ToTensor(),
            ]
        )

        # 把資料集的 transforms 換成我們這套（不破壞原本傳入的 transforms 屬性）
        train_loader = DataLoader(
            _SegWrapper(self.dataset, transform=train_tf),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.device_str.startswith("cuda"),
        )

        val_loader = None
        if self.val_dataset is not None:
            val_loader = DataLoader(
                _SegWrapper(self.val_dataset, transform=val_tf),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        logger.info(f"Train batches: {len(train_loader)}")
        return train_loader, val_loader

    def _make_eval_loader(self, dataset: BaseDataset) -> Any:
        from torch.utils.data import DataLoader

        val_tf = Compose(
            [
                Resize((self.image_size, self.image_size), keep_ratio=True),
                Normalize(),
                ToTensor(),
            ]
        )
        return DataLoader(
            _SegWrapper(dataset, transform=val_tf),
            batch_size=self.batch_size,
            shuffle=False,
        )

    # ------------------------------------------------------------------
    # 訓練 / 評估
    # ------------------------------------------------------------------

    def _train_one_epoch(
        self, epoch: int, callbacks: List[TrainingCallback]
    ) -> Dict[str, float]:
        import torch.nn as nn

        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.model.train()

        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            for cb in callbacks:
                cb.on_batch_start(self, batch_idx)

            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(images)
            loss = criterion(out["out"], masks)
            if self.aux_loss and "aux" in out:
                loss = loss + 0.4 * criterion(out["aux"], masks)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            for cb in callbacks:
                cb.on_batch_end(self, batch_idx, loss.item())

        return {"loss": epoch_loss / max(1, n_batches)}

    def _evaluate(self, loader: Any) -> Dict[str, float]:
        import torch
        import torch.nn as nn

        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.model.eval()

        num_classes = self.dataset.num_classes
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for images, masks in loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                out = self.model(images)
                logits = out["out"]
                loss = criterion(logits, masks)
                total_loss += loss.item()
                n_batches += 1

                preds = logits.argmax(dim=1).cpu().numpy()
                gt = masks.cpu().numpy()
                _accumulate_confusion(
                    confusion, gt, preds, num_classes, self.ignore_index
                )

        ious = _per_class_iou(confusion)
        miou = float(np.nanmean(ious)) if ious.size else 0.0
        pix_acc = float(np.diag(confusion).sum() / max(1, confusion.sum()))
        return {
            "loss": total_loss / max(1, n_batches),
            "miou": miou,
            "pixel_acc": pix_acc,
        }

    # ------------------------------------------------------------------
    # Export 簽章
    # ------------------------------------------------------------------

    def _export_signature(self) -> Dict[str, Any]:
        return {
            "task": "segment",
            "input_shape": (1, 3, self.image_size, self.image_size),
            "class_names": list(self.dataset.class_names),
        }

    def export(self, path: str, format: str = "onnx", **kwargs) -> str:
        """
        torchvision 的 segmentation 模型 forward 回傳 OrderedDict({"out", "aux"})，
        torch.onnx.export 不支援 dict 輸出 — 包一層 nn.Module 取出 "out"。
        """

        wrapped = _SegOutOnly(self.model)
        from .exporter import ModelExporter

        sig = self._export_signature()
        exporter = ModelExporter(wrapped, self.device)
        return exporter.export(
            path=path,
            format=format,
            input_shape=sig["input_shape"],
            class_names=sig["class_names"],
            task="segment",
            **kwargs,
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _accumulate_confusion(
    confusion: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    ignore_index: int,
) -> None:
    """以 ignore mask 過濾後累加 confusion matrix。"""
    valid = (gt != ignore_index) & (gt < num_classes)
    g = gt[valid].astype(np.int64)
    p = pred[valid].astype(np.int64)
    p = np.clip(p, 0, num_classes - 1)
    idx = g * num_classes + p
    bins = np.bincount(idx, minlength=num_classes * num_classes)
    confusion += bins.reshape(num_classes, num_classes)


def _per_class_iou(confusion: np.ndarray) -> np.ndarray:
    tp = np.diag(confusion).astype(np.float64)
    fp_fn = confusion.sum(axis=0) + confusion.sum(axis=1) - 2 * tp
    denom = tp + fp_fn
    iou = np.where(denom > 0, tp / denom, np.nan)
    return iou


class _SegWrapper:
    """把 SegmentationDataset 的 (image, {"mask": ...}) 轉成 DataLoader 友善格式。"""

    def __init__(self, dataset: BaseDataset, transform: Optional[Callable] = None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        import torch

        image, target = self.dataset[idx]
        if self.transform is not None:
            image, target = self.transform(image, target)

        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).float()
        else:
            image_tensor = image
        # ToTensor 加了 batch dim，這裡 squeeze 掉
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)

        mask = target["mask"]
        if isinstance(mask, np.ndarray):
            mask_tensor = torch.from_numpy(mask.astype(np.int64))
        else:
            mask_tensor = mask.long()
        return image_tensor, mask_tensor


class _SegOutOnly:
    """Wrapper：把 torchvision 的 OrderedDict 輸出收斂成單一 tensor，方便 ONNX export。"""

    def __new__(cls, model):
        import torch.nn as nn

        class _Wrap(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m(x)["out"]

        return _Wrap(model).eval()
