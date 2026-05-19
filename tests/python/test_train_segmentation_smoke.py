"""
SegmentationTrainer smoke test：
- 假 4 張 64×64 圖、3 類，1 epoch，CPU
- 驗證 fit() 回 history、evaluate() 回 mIoU
- ONNX export 經 onnxruntime 跑 inference 對拍 shape
"""

import pytest
import numpy as np
from pathlib import Path

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")


def _make_fake_seg_dataset(root: Path, n: int = 4, num_classes: int = 3, size: int = 64) -> tuple:
    import cv2

    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        img = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
        cv2.imwrite(str(images_dir / f"{i}.png"), img)

        mask = np.random.randint(0, num_classes, size=(size, size), dtype=np.uint8)
        cv2.imwrite(str(masks_dir / f"{i}.png"), mask)
    return images_dir, masks_dir


def test_segmentation_trainer_smoke(tmp_path):
    from ivit.train import SegmentationTrainer, SegmentationDataset

    images_dir, masks_dir = _make_fake_seg_dataset(tmp_path)

    train_ds = SegmentationDataset(
        images_dir, masks_dir,
        class_names=["bg", "a", "b"],
        train_split=0.5, split="train",
    )
    val_ds = SegmentationDataset(
        images_dir, masks_dir,
        class_names=["bg", "a", "b"],
        train_split=0.5, split="val",
    )

    trainer = SegmentationTrainer(
        model="deeplabv3-mobilenetv3",
        dataset=train_ds,
        val_dataset=val_ds,
        epochs=1,
        learning_rate=1e-4,
        batch_size=2,
        device="cpu",
        num_workers=0,
        image_size=64,
        aux_loss=False,        # 加快 smoke test
        freeze_backbone=True,
    )
    history = trainer.fit()
    assert history, "fit() should return non-empty history"
    last = history[-1]
    assert "loss" in last
    assert "val_miou" in last

    metrics = trainer.evaluate()
    assert "miou" in metrics


def test_segmentation_export_onnx(tmp_path):
    """Verify ONNX export wraps OrderedDict output into a single tensor."""
    pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")

    from ivit.train import SegmentationTrainer, SegmentationDataset

    images_dir, masks_dir = _make_fake_seg_dataset(tmp_path)
    ds = SegmentationDataset(
        images_dir, masks_dir,
        class_names=["bg", "a", "b"],
        train_split=1.0, split="train",
    )
    trainer = SegmentationTrainer(
        model="deeplabv3-mobilenetv3",
        dataset=ds,
        epochs=1, learning_rate=1e-4, batch_size=2,
        device="cpu", num_workers=0, image_size=64,
        aux_loss=False, freeze_backbone=True,
    )

    onnx_path = tmp_path / "seg.onnx"
    trainer.export(str(onnx_path), format="onnx")
    assert onnx_path.exists()

    sess = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = np.random.rand(1, 3, 64, 64).astype(np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: inp})[0]
    assert out.ndim == 4
    assert out.shape[0] == 1
    assert out.shape[1] == 3  # num_classes
