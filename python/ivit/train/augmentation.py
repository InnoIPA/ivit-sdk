"""
Data augmentation transforms for training.

Provides composable transforms for images.
"""

from typing import List, Tuple, Union, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Compose:
    """
    Compose multiple transforms together.

    Args:
        transforms: List of transform objects

    Examples:
        >>> transform = Compose([
        ...     Resize(256),
        ...     RandomHorizontalFlip(),
        ...     Normalize(),
        ...     ToTensor(),
        ... ])
        >>> image = transform(image)
    """

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, image: np.ndarray, target: Any = None) -> Any:
        for t in self.transforms:
            if target is not None:
                result = t(image, target)
                if isinstance(result, tuple):
                    image, target = result
                else:
                    image = result
            else:
                image = t(image)

        if target is not None:
            return image, target
        return image

    def __repr__(self) -> str:
        transforms_str = ", ".join([str(t) for t in self.transforms])
        return f"Compose([{transforms_str}])"


class Resize:
    """
    Resize image to target size.

    Args:
        size: Target size (int for square, tuple for (h, w))
        keep_ratio: If True, resize with aspect ratio preservation
        pad_value: Padding value when keep_ratio=True

    Examples:
        >>> resize = Resize(224)
        >>> image = resize(image)  # 224x224
        >>>
        >>> resize = Resize((480, 640), keep_ratio=True)
        >>> image = resize(image)  # Letterbox resize
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        keep_ratio: bool = False,
        pad_value: int = 114,
    ):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.keep_ratio = keep_ratio
        self.pad_value = pad_value

    def __call__(self, image: np.ndarray, target: Any = None) -> Any:
        import cv2

        h, w = self.size

        if self.keep_ratio:
            # Letterbox resize
            orig_h, orig_w = image.shape[:2]
            scale = min(w / orig_w, h / orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            pad_w, pad_h = (w - new_w) // 2, (h - new_h) // 2

            resized = cv2.resize(image, (new_w, new_h))
            output = np.full((h, w, 3), self.pad_value, dtype=np.uint8)
            output[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

            # Update target boxes if present
            if target is not None and "boxes" in target:
                boxes = target["boxes"].copy()
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_w
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_h
                target = {**target, "boxes": boxes}
            # Update segmentation mask if present
            if target is not None and "mask" in target:
                mask = target["mask"]
                resized_mask = cv2.resize(
                    mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST
                )
                # 255 = ignore，當作 padding 值
                out_mask = np.full((h, w), 255, dtype=mask.dtype)
                out_mask[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized_mask
                target = {**target, "mask": out_mask}
        else:
            output = cv2.resize(image, (w, h))

            # Scale target boxes
            if target is not None and "boxes" in target:
                orig_h, orig_w = image.shape[:2]
                boxes = target["boxes"].copy()
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * w / orig_w
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * h / orig_h
                target = {**target, "boxes": boxes}
            # Resize segmentation mask with nearest-neighbor to preserve class indices
            if target is not None and "mask" in target:
                mask = target["mask"]
                target = {
                    **target,
                    "mask": cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST),
                }

        if target is not None:
            return output, target
        return output

    def __repr__(self) -> str:
        return f"Resize(size={self.size}, keep_ratio={self.keep_ratio})"


class RandomHorizontalFlip:
    """
    Randomly flip image horizontally.

    Args:
        p: Probability of flip (default: 0.5)

    Examples:
        >>> flip = RandomHorizontalFlip(p=0.5)
        >>> image = flip(image)
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray, target: Any = None) -> Any:
        import cv2

        if np.random.random() < self.p:
            image = cv2.flip(image, 1)  # Horizontal flip

            if target is not None and "boxes" in target:
                w = image.shape[1]
                boxes = target["boxes"].copy()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target = {**target, "boxes": boxes}
            if target is not None and "mask" in target:
                target = {
                    **target,
                    "mask": np.ascontiguousarray(target["mask"][:, ::-1]),
                }

        if target is not None:
            return image, target
        return image

    def __repr__(self) -> str:
        return f"RandomHorizontalFlip(p={self.p})"


class RandomVerticalFlip:
    """
    Randomly flip image vertically.

    Args:
        p: Probability of flip (default: 0.5)
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray, target: Any = None) -> Any:
        import cv2

        if np.random.random() < self.p:
            image = cv2.flip(image, 0)  # Vertical flip

            if target is not None and "boxes" in target:
                h = image.shape[0]
                boxes = target["boxes"].copy()
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
                target = {**target, "boxes": boxes}
            if target is not None and "mask" in target:
                target = {
                    **target,
                    "mask": np.ascontiguousarray(target["mask"][::-1, :]),
                }

        if target is not None:
            return image, target
        return image

    def __repr__(self) -> str:
        return f"RandomVerticalFlip(p={self.p})"


class RandomRotation:
    """
    Randomly rotate image.

    Args:
        degrees: Range of rotation in degrees (-degrees, +degrees)
        p: Probability of rotation

    Examples:
        >>> rotate = RandomRotation(30)  # Rotate between -30 and +30 degrees
    """

    def __init__(self, degrees: float, p: float = 0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, image: np.ndarray, target: Any = None) -> Any:
        import cv2

        if np.random.random() < self.p:
            angle = np.random.uniform(-self.degrees, self.degrees)
            h, w = image.shape[:2]
            center = (w / 2, h / 2)

            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderValue=(114, 114, 114))

            # Note: Box rotation is complex, skip for simplicity
            # In production, use proper affine transform for boxes

        if target is not None:
            return image, target
        return image

    def __repr__(self) -> str:
        return f"RandomRotation(degrees={self.degrees}, p={self.p})"


class ColorJitter:
    """
    Randomly adjust brightness, contrast, saturation, and hue.

    Args:
        brightness: Brightness adjustment range (0-1)
        contrast: Contrast adjustment range (0-1)
        saturation: Saturation adjustment range (0-1)
        hue: Hue adjustment range (0-0.5)

    Examples:
        >>> jitter = ColorJitter(brightness=0.2, contrast=0.2)
        >>> image = jitter(image)
    """

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image: np.ndarray, target: Any = None) -> Any:
        pass

        # Random order of adjustments
        adjustments = [
            (self._adjust_brightness, self.brightness),
            (self._adjust_contrast, self.contrast),
            (self._adjust_saturation, self.saturation),
            (self._adjust_hue, self.hue),
        ]
        np.random.shuffle(adjustments)

        for adjust_fn, param in adjustments:
            if param > 0:
                image = adjust_fn(image, param)

        if target is not None:
            return image, target
        return image

    def _adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        adjustment = np.random.uniform(-factor, factor)
        image = image.astype(np.float32)
        image = image + adjustment * 255
        return np.clip(image, 0, 255).astype(np.uint8)

    def _adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        adjustment = np.random.uniform(1 - factor, 1 + factor)
        image = image.astype(np.float32)
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        image = (image - mean) * adjustment + mean
        return np.clip(image, 0, 255).astype(np.uint8)

    def _adjust_saturation(self, image: np.ndarray, factor: float) -> np.ndarray:
        import cv2

        adjustment = np.random.uniform(1 - factor, 1 + factor)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * adjustment
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def _adjust_hue(self, image: np.ndarray, factor: float) -> np.ndarray:
        import cv2

        adjustment = np.random.uniform(-factor, factor) * 180
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + adjustment) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def __repr__(self) -> str:
        return (
            f"ColorJitter(brightness={self.brightness}, contrast={self.contrast}, "
            f"saturation={self.saturation}, hue={self.hue})"
        )


class Normalize:
    """
    Normalize image with mean and std.

    Args:
        mean: Per-channel mean (default: ImageNet)
        std: Per-channel std (default: ImageNet)

    Examples:
        >>> normalize = Normalize()  # ImageNet normalization
        >>> image = normalize(image)  # Returns float32 array
    """

    def __init__(
        self,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, image: np.ndarray, target: Any = None) -> Any:
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std

        if target is not None:
            return image, target
        return image

    def __repr__(self) -> str:
        return f"Normalize(mean={self.mean.flatten().tolist()}, std={self.std.flatten().tolist()})"


class ToTensor:
    """
    Convert numpy array to NCHW tensor format.

    Examples:
        >>> to_tensor = ToTensor()
        >>> tensor = to_tensor(image)  # Shape: (1, C, H, W)
    """

    def __call__(self, image: np.ndarray, target: Any = None) -> Any:
        # Ensure float32
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0

        # HWC -> CHW
        if len(image.shape) == 3:
            image = image.transpose(2, 0, 1)

        # Add batch dimension
        image = np.expand_dims(image, 0)

        if target is not None:
            # Mark mask for downstream wrappers to cast to torch.long.
            # 我們不在這裡 import torch（augmentation 應與 torch 解耦），
            # 由各 task 的 DataLoader wrapper 處理 dtype 轉換。
            return image, target
        return image

    def __repr__(self) -> str:
        return "ToTensor()"


def get_default_augmentation(size: int = 224) -> Compose:
    """
    Get default augmentation pipeline.

    Args:
        size: Target image size

    Returns:
        Compose transform
    """
    return Compose(
        [
            Resize(size),
            Normalize(),
            ToTensor(),
        ]
    )


def get_train_augmentation(
    size: int = 224,
    flip_p: float = 0.5,
    color_jitter: bool = True,
) -> Compose:
    """
    Get training augmentation pipeline.

    Args:
        size: Target image size
        flip_p: Horizontal flip probability
        color_jitter: Enable color jittering

    Returns:
        Compose transform
    """
    transforms = [Resize(size)]

    if flip_p > 0:
        transforms.append(RandomHorizontalFlip(p=flip_p))

    if color_jitter:
        transforms.append(
            ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            )
        )

    transforms.extend(
        [
            Normalize(),
            ToTensor(),
        ]
    )

    return Compose(transforms)


def get_val_augmentation(size: int = 224) -> Compose:
    """
    Get validation augmentation pipeline (no random transforms).

    Args:
        size: Target image size

    Returns:
        Compose transform
    """
    return Compose(
        [
            Resize(size),
            Normalize(),
            ToTensor(),
        ]
    )
