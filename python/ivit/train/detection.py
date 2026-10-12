"""
DetectionTrainer：將 fit() / evaluate() / export() 委派給 YOLOX backend。

由於 YOLOX 有完整的訓練流程（Exp.train），本 Trainer 不繼承 BaseTrainer 的標準 epoch
loop，而是把 fit() / evaluate() / export() 委派給對應 backend。這保留了與
ClassificationTrainer / SegmentationTrainer 一致的對外介面。
"""

from typing import Any, Dict, List, Optional
import logging

from .callbacks import TrainingCallback

logger = logging.getLogger(__name__)


_YOLOX_NAMES = {"yolox-nano", "yolox-tiny", "yolox-s", "yolox-m", "yolox-l"}


def _resolve_backend(model_name: str) -> str:
    name = model_name.lower()
    if name in _YOLOX_NAMES:
        return "yolox"
    raise ValueError(
        f"Unknown detection model: {model_name}. "
        f"Supported: YOLOX ({sorted(_YOLOX_NAMES)})."
    )


class DetectionTrainer:
    """
    YOLOX 偵測 Trainer。

    Args:
        model: 模型名稱 "yolox-{nano,tiny,s,m,l}"
        epochs / batch_size / device: 通用訓練參數
        image_size: 訓練輸入解析度
        data_dir: COCO root（含 images/、annotations/）
        train_ann / val_ann: annotation 檔名
        num_classes: 類別數（不指定則用 80）
        pretrained_weights: 預訓練 .pth
        output_dir: 輸出根目錄

    Examples:
        >>> tr = DetectionTrainer(model="yolox-s",
        ...                       data_dir="./datasets/coco128",
        ...                       num_classes=80,
        ...                       epochs=10, image_size=640, device="cuda:0")
        >>> tr.fit(); tr.export("yolox-s.onnx")
    """

    def __init__(
        self,
        model: str,
        epochs: int = 100,
        batch_size: int = 16,
        device: str = "cuda:0",
        image_size: int = 640,
        output_dir: str = "runs/train",
        data_dir: Optional[str] = None,
        train_ann: str = "instances_train2017.json",
        val_ann: str = "instances_val2017.json",
        num_classes: int = 80,
        pretrained_weights: Optional[str] = None,
        **kwargs,
    ):
        self.model_name = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.output_dir = output_dir
        self.config = kwargs

        self._backend_name = _resolve_backend(model)
        self._history: List[Dict[str, Any]] = []

        if data_dir is None:
            raise ValueError("YOLOX training requires `data_dir` (COCO root).")
        from ._yolox_backend import YoloxBackend

        self._backend = YoloxBackend(
            model=model,
            data_dir=data_dir,
            train_ann=train_ann,
            val_ann=val_ann,
            num_classes=num_classes,
            epochs=epochs,
            imgsz=image_size,
            batch_size=batch_size,
            device=device,
            output_dir=output_dir,
            pretrained_weights=pretrained_weights,
            **kwargs,
        )

    @property
    def backend_name(self) -> str:
        """回傳 'yolox'。"""
        return self._backend_name

    @property
    def history(self) -> List[Dict[str, Any]]:
        return self._history

    def fit(
        self, callbacks: Optional[List[TrainingCallback]] = None
    ) -> List[Dict[str, Any]]:
        """執行訓練。callbacks 目前僅作為觀察用（YOLOX 自帶完整 logger）。"""
        if callbacks:
            for cb in callbacks:
                cb.on_train_start(self)
        result = self._backend.fit()
        self._history = [result] if isinstance(result, dict) else list(result)
        if callbacks:
            for cb in callbacks:
                cb.on_train_end(self)
        return self._history

    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """跑底層 backend 的驗證流程（YOLOX: COCOEvaluator；Ultralytics: model.val()）。

        Kwargs 會直接 forward 到 backend：
            - YOLOX: ckpt, conf_thres, nms_thres, verbose
            - Ultralytics: 無（走自身 model.val 的 data yaml）
        """
        if hasattr(self._backend, "evaluate"):
            return self._backend.evaluate(**kwargs)
        return {}

    def export(self, path: str, format: str = "onnx", **kwargs) -> str:
        """匯出模型。"""
        return self._backend.export(path=path, format=format, **kwargs)
