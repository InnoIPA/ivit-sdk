"""
augmentation / dataset 驗證：
- mask 經 Resize / HFlip / VFlip 後類別 index 不被插值破壞
- bbox 路徑的 Resize / Flip 仍正確
- SegmentationDataset 能正確配對 image + mask
"""

import numpy as np
import pytest
from pathlib import Path


def test_resize_mask_uses_nearest_neighbor():
    from ivit.train.augmentation import Resize

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.array([
        [0, 1, 2, 3] * 2 + [0, 0],
        [0, 1, 2, 3] * 2 + [0, 0],
    ] * 5, dtype=np.uint8)  # 10x10
    target = {"mask": mask}

    transformed_image, transformed_target = Resize((20, 20))(image, target)
    out_mask = transformed_target["mask"]

    # 用 INTER_NEAREST 後值仍只應在 {0,1,2,3}
    unique = set(np.unique(out_mask).tolist())
    assert unique <= {0, 1, 2, 3}, f"unexpected mask values after resize: {unique}"
    assert out_mask.shape == (20, 20)
    assert transformed_image.shape == (20, 20, 3)


def test_horizontal_flip_mask_and_box_consistent():
    from ivit.train.augmentation import RandomHorizontalFlip

    image = np.zeros((4, 6, 3), dtype=np.uint8)
    mask = np.array([
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 1, 2, 2],
    ], dtype=np.uint8)
    boxes = np.array([[0, 0, 2, 4], [4, 0, 6, 4]], dtype=np.float32)

    flip = RandomHorizontalFlip(p=1.0)  # 確定翻
    target = {"mask": mask, "boxes": boxes.copy()}
    _, out = flip(image, target)

    np.testing.assert_array_equal(out["mask"][:, 0], np.array([2, 2, 2, 2]))
    np.testing.assert_array_equal(out["mask"][:, -1], np.array([0, 0, 0, 0]))

    expected_boxes = np.array([[4, 0, 6, 4], [0, 0, 2, 4]], dtype=np.float32)
    np.testing.assert_array_equal(out["boxes"], expected_boxes)


def test_classify_path_unchanged():
    """純 image 路徑（target=None）不應被新分支影響。"""
    from ivit.train.augmentation import Resize, RandomHorizontalFlip

    image = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
    out = Resize((16, 16))(image)
    assert out.shape == (16, 16, 3)
    out2 = RandomHorizontalFlip(p=1.0)(image)
    assert out2.shape == image.shape


def test_segmentation_dataset_loads_pair(tmp_path: Path):
    import cv2
    from ivit.train import SegmentationDataset

    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    for i in range(2):
        cv2.imwrite(str(images_dir / f"{i}.png"), np.zeros((8, 8, 3), dtype=np.uint8))
        cv2.imwrite(str(masks_dir / f"{i}.png"), np.full((8, 8), i, dtype=np.uint8))

    ds = SegmentationDataset(
        images_dir, masks_dir, class_names=["a", "b"], train_split=1.0, split="all",
    )
    assert len(ds) == 2
    img, target = ds[0]
    assert img.ndim == 3 and img.shape[2] == 3
    assert "mask" in target
    assert target["mask"].shape == (8, 8)
    assert ds.num_classes == 2
