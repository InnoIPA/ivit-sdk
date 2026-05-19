"""
ClassificationTrainer smoke test：
- 確認 `Trainer` alias 仍可從 `ivit.train` 匯入
- 用 6 張假圖跑 1 epoch，回傳 history dict 含 loss / accuracy
"""

import pytest
import numpy as np
from pathlib import Path

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")


def _make_fake_imagefolder(root: Path, n_per_class: int = 3, classes=("cat", "dog")) -> None:
    import cv2

    for cls in classes:
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
            cv2.imwrite(str(d / f"{i}.jpg"), img)


def test_trainer_alias_available():
    from ivit.train import Trainer, ClassificationTrainer
    assert Trainer is ClassificationTrainer


def test_classification_trainer_smoke(tmp_path):
    from ivit.train import ClassificationTrainer, ImageFolderDataset

    _make_fake_imagefolder(tmp_path)

    train_ds = ImageFolderDataset(tmp_path, train_split=0.66, split="train")
    val_ds = ImageFolderDataset(tmp_path, train_split=0.66, split="val")

    trainer = ClassificationTrainer(
        model="mobilenet_v2",
        dataset=train_ds,
        val_dataset=val_ds,
        epochs=1,
        learning_rate=1e-3,
        batch_size=2,
        device="cpu",
        num_workers=0,
        image_size=64,
        freeze_backbone=True,
    )
    history = trainer.fit()

    assert isinstance(history, list) and history, "fit() should return non-empty history"
    last = history[-1]
    assert "loss" in last
    assert "accuracy" in last
