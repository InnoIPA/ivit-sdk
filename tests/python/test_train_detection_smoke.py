"""
DetectionTrainer smoke test：
- backend 選擇分派正確（不需安裝 yolox 也能驗證）
- yolox 實際訓練測試在缺套件時自動 skip
"""

import pytest
import numpy as np
from pathlib import Path


def test_detection_backend_dispatch():
    """無需任何 optional 套件，純驗證模型名解析。"""
    from ivit.train.detection import _resolve_backend

    assert _resolve_backend("yolox-s") == "yolox"
    assert _resolve_backend("yolox-nano") == "yolox"

    with pytest.raises(ValueError):
        _resolve_backend("unknown-model")


def test_zoo_training_helpers():
    """ivit.zoo 公開的訓練輔助函式。"""
    from ivit.zoo import is_trainable, get_training_backend, list_trainable_models

    assert is_trainable("yolox-s") is True
    assert is_trainable("resnet50") is True
    assert is_trainable("deeplabv3-mobilenetv3") is True
    assert is_trainable("totally-fake-model") is False

    assert get_training_backend("yolox-s") == "yolox"
    assert get_training_backend("resnet50") == "torchvision-classify"
    assert get_training_backend("deeplabv3-resnet50") == "torchvision-segment"

    backends = list_trainable_models()
    assert {"yolox", "torchvision-classify", "torchvision-segment"} <= set(backends)


def test_yolox_backend_requires_data_dir():
    """缺 data_dir 時應該 raise 而不是進到 yolox import。"""
    from ivit.train import DetectionTrainer

    with pytest.raises(ValueError, match="data_dir"):
        DetectionTrainer(model="yolox-s")  # 沒給 data_dir


@pytest.mark.slow
def test_yolox_constructor_smoke():
    """需安裝 yolox 才執行。"""
    pytest.importorskip("yolox")
    from ivit.train._yolox_backend import YoloxBackend

    # 不真的 fit，只驗證 Exp 子類能被建立
    backend = YoloxBackend(
        model="yolox-nano",
        data_dir="/tmp/fake",
        num_classes=2,
        epochs=1,
        imgsz=416,
        batch_size=1,
        device="cpu",
    )
    assert backend.model_name == "yolox-nano"
