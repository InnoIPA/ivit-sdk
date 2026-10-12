"""
Semantic segmentation module.

Uses C++ bindings for inference.
"""

from typing import Union, List, Tuple
import numpy as np

from .._ivit_core import (
    Segmentor as _CppSegmentor,
    LoadConfig as _CppLoadConfig,
    Results,
)


class Segmentor:
    """
    Semantic segmentor.

    Examples:
        >>> segmentor = Segmentor("deeplabv3.onnx", device="cuda:0")
        >>> results = segmentor.predict("street.jpg")
        >>> mask = results.segmentation_mask
    """

    def __init__(self, model: str, device: str = "auto", **kwargs):
        """
        Create segmentor.

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

        self._cpp_segmentor = _CppSegmentor(model, device, config)
        self._device = device

    def predict(
        self,
        image: Union[str, np.ndarray],
    ) -> Results:
        """
        Segment image.

        Args:
            image: Input image (file path or numpy array)

        Returns:
            Results object containing segmentation mask
        """
        if isinstance(image, str):
            return self._cpp_segmentor.predict(image)
        return self._cpp_segmentor.predict(image)

    def __call__(
        self,
        image: Union[str, np.ndarray],
    ) -> Results:
        """Shorthand for predict()."""
        return self.predict(image)

    @property
    def classes(self) -> List[str]:
        """Get class labels."""
        return list(self._cpp_segmentor.classes)

    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return self._cpp_segmentor.num_classes

    @property
    def input_size(self) -> Tuple[int, int]:
        """Get input size (width, height)."""
        return self._cpp_segmentor.input_size

    @property
    def device(self) -> str:
        """Get device being used."""
        return self._device
