"""
BaseTrainer ABC：共用訓練生命週期、callback、optimizer/device 初始化與 history。

各任務子類（ClassificationTrainer / DetectionTrainer / SegmentationTrainer）
覆寫抽象方法以提供任務專屬的 model 建構、dataloader、單 epoch 訓練、評估與 export 簽章。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time

from .dataset import BaseDataset
from .callbacks import TrainingCallback, ProgressLogger

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    所有 task-specific Trainer 的共用基底。

    Args:
        model: 模型名稱字串或預先建構好的模型物件
        dataset: 訓練資料集（BaseDataset 子類）
        val_dataset: 驗證資料集（可選）
        epochs: 訓練 epoch 數
        learning_rate: 學習率
        batch_size: batch 大小
        optimizer: "adam" / "adamw" / "sgd"
        device: "cuda:0" / "cpu" / ...
        num_workers: DataLoader worker 數
        freeze_backbone: transfer learning 時是否凍結 backbone
        **kwargs: 任務專屬參數，存到 self.config 供子類取用
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
        num_workers: int = 4,
        freeze_backbone: bool = True,
        **kwargs,
    ):
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer_type = optimizer
        self.device_str = device
        self.num_workers = num_workers
        self.freeze_backbone = freeze_backbone
        self.config = kwargs

        self._stop_training = False
        self._history: List[Dict[str, float]] = []
        self._current_epoch = 0
        self._batches_per_epoch = 0

        self._init_torch()
        self.model = self._build_model(model)
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        self._init_optimizer()
        self.train_loader, self.val_loader = self._build_dataloaders()
        if self.train_loader is not None:
            self._batches_per_epoch = len(self.train_loader)

    # ------------------------------------------------------------------
    # 共用初始化
    # ------------------------------------------------------------------

    def _init_torch(self) -> None:
        """初始化 PyTorch、檢查 CUDA。"""
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for training. "
                "Install with: pip install 'ivit-sdk[train]'"
            ) from e

        self._torch = torch
        if self.device_str.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device_str = "cpu"
        self.device = torch.device(self.device_str)
        logger.info(f"Using device: {self.device}")

    def _init_optimizer(self) -> None:
        """依 optimizer_type 建立 optimizer。子類可覆寫以自訂 param groups。"""
        import torch.optim as optim

        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            logger.warning("No trainable parameters found")

        opt = self.optimizer_type.lower()
        if opt == "adam":
            self.optimizer = optim.Adam(params, lr=self.learning_rate)
        elif opt == "adamw":
            self.optimizer = optim.AdamW(params, lr=self.learning_rate)
        elif opt == "sgd":
            self.optimizer = optim.SGD(params, lr=self.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
        logger.info(f"Optimizer: {opt}, lr={self.learning_rate}")

    # ------------------------------------------------------------------
    # 抽象方法 — 子類必須實作
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_model(self, spec: Union[str, Any]) -> Any:
        """根據 spec（模型名或物件）建構並回傳 PyTorch model（尚未 .to(device)）。"""

    @abstractmethod
    def _build_dataloaders(self) -> Tuple[Any, Optional[Any]]:
        """回傳 (train_loader, val_loader)。val_loader 可為 None。"""

    @abstractmethod
    def _train_one_epoch(
        self, epoch: int, callbacks: List[TrainingCallback]
    ) -> Dict[str, float]:
        """跑完一個 epoch，回傳該 epoch 的指標 dict（至少含 'loss'）。"""

    @abstractmethod
    def _evaluate(self, loader: Any) -> Dict[str, float]:
        """在 loader 上跑驗證/評估，回傳指標 dict。"""

    @abstractmethod
    def _export_signature(self) -> Dict[str, Any]:
        """
        回傳給 ModelExporter 用的簽章 dict：
            {
                "task": "classify" | "detect" | "segment",
                "input_shape": (1, 3, H, W),
                "class_names": Optional[List[str]],
            }
        """

    # ------------------------------------------------------------------
    # 公開介面
    # ------------------------------------------------------------------

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @property
    def batches_per_epoch(self) -> int:
        return self._batches_per_epoch

    @property
    def history(self) -> List[Dict[str, float]]:
        return self._history

    def fit(
        self,
        callbacks: Optional[List[TrainingCallback]] = None,
    ) -> List[Dict[str, float]]:
        """執行訓練主迴圈。"""
        callbacks = callbacks or [ProgressLogger()]

        for cb in callbacks:
            cb.on_train_start(self)

        logger.info(f"Starting training for {self.epochs} epochs")
        start_time = time.time()

        for epoch in range(self.epochs):
            if self._stop_training:
                logger.info("Training stopped early")
                break

            self._current_epoch = epoch
            for cb in callbacks:
                cb.on_epoch_start(self, epoch)

            metrics = self._train_one_epoch(epoch, callbacks)

            if self.val_loader is not None:
                val_metrics = self._evaluate(self.val_loader)
                metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

            self._history.append(metrics)

            for cb in callbacks:
                cb.on_epoch_end(self, epoch, metrics)

        for cb in callbacks:
            cb.on_train_end(self)

        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.1f}s")
        return self._history

    def evaluate(self, dataset: Optional[BaseDataset] = None) -> Dict[str, float]:
        """在指定資料集（或既有 val 資料集）上評估。"""
        if dataset is not None:
            loader = self._make_eval_loader(dataset)
        elif self.val_loader is not None:
            loader = self.val_loader
        else:
            raise ValueError("No validation dataset available")
        return self._evaluate(loader)

    def _make_eval_loader(self, dataset: BaseDataset) -> Any:
        """子類可覆寫以提供任務專屬的 eval loader 建構方式。預設沿用 val_loader 規格。"""
        raise NotImplementedError(
            "Subclass must override _make_eval_loader to support evaluate(dataset=...)."
        )

    def export(
        self,
        path: str,
        format: str = "onnx",
        optimize_for: Optional[str] = None,
        quantize: Optional[str] = None,
        **kwargs,
    ) -> str:
        """匯出訓練好的模型。子類可覆寫以走任務專屬 export 流程。"""
        from .exporter import ModelExporter

        sig = self._export_signature()
        exporter = ModelExporter(self.model, self.device)
        return exporter.export(
            path=path,
            format=format,
            optimize_for=optimize_for,
            quantize=quantize,
            input_shape=sig.get("input_shape", (1, 3, 224, 224)),
            class_names=sig.get("class_names"),
            task=sig.get("task", "classify"),
            **kwargs,
        )
