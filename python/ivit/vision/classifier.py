"""
Image classification module.

Uses C++ bindings for inference.
"""

from typing import Union, List, Tuple
import numpy as np

from .._ivit_core import (
    Classifier as _CppClassifier,
    LoadConfig as _CppLoadConfig,
    Results,
)


class Classifier:
    """
    Image classifier.

    Examples:
        >>> classifier = Classifier("resnet50.onnx", device="cuda:0")
        >>> results = classifier.predict("cat.jpg")
        >>> print(f"Top-1: {results.top1.label} ({results.top1.score:.2%})")
    """

    def __init__(self, model: str, device: str = "auto", **kwargs):
        """
        Create classifier.

        Args:
            model: Model path (.onnx, .engine, .xml)
            device: Target device ("auto", "cpu", "cuda:0", etc.)
            **kwargs: Additional configuration options
        """
        config = _CppLoadConfig()
        config.device = device
        for key, value in kwargs.items():
            if not hasattr(config, key):
                raise TypeError(f"Unknown configuration option: {key}")
            setattr(config, key, value)

        self._cpp_classifier = _CppClassifier(model, device, config)
        self._device = device

    def predict(self, image: Union[str, np.ndarray], top_k: int = 5) -> Results:
        """
        Classify image.

        Args:
            image: Input image (file path or numpy array)
            top_k: Number of top predictions to return

        Returns:
            Results object containing classifications
        """
        if isinstance(image, str):
            return self._cpp_classifier.predict(image, top_k)
        return self._cpp_classifier.predict(image, top_k)

    def __call__(self, image: Union[str, np.ndarray], top_k: int = 5) -> Results:
        """Shorthand for predict()."""
        return self.predict(image, top_k=top_k)

    def predict_batch(
        self, images: List[Union[str, np.ndarray]], top_k: int = 5
    ) -> List[Results]:
        """
        Batch classification.

        Args:
            images: List of images (paths or arrays)
            top_k: Number of top predictions

        Returns:
            List of Results objects
        """
        import cv2

        np_images = []
        for img in images:
            if isinstance(img, str):
                np_img = cv2.imread(img)
                if np_img is None:
                    raise ValueError(f"Failed to load image: {img}")
                np_images.append(np_img)
            else:
                np_images.append(img)

        return self._cpp_classifier.predict_batch(np_images, top_k)

    @property
    def classes(self) -> List[str]:
        """Get class labels."""
        return list(self._cpp_classifier.classes)

    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return self._cpp_classifier.num_classes

    @property
    def input_size(self) -> Tuple[int, int]:
        """Get input size (width, height)."""
        return self._cpp_classifier.input_size

    @property
    def device(self) -> str:
        """Get device being used."""
        return self._device
