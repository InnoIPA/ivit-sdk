"""
Object detection module.

Uses C++ bindings for inference.
"""

from typing import Union, List, Optional, Callable, Tuple
import numpy as np

from .._ivit_core import (
    Detector as _CppDetector,
    LoadConfig as _CppLoadConfig,
    InferConfig as _CppInferConfig,
    Results,
)


class Detector:
    """
    Object detector.

    High-level API for object detection tasks.
    Supports YOLO, SSD, and other detection models.

    Examples:
        >>> detector = Detector("yolox-s.onnx", device="cuda:0")
        >>> results = detector.predict("image.jpg")
        >>> for det in results.detections:
        ...     print(f"{det.label}: {det.confidence:.2%}")
    """

    def __init__(
        self,
        model: str,
        device: str = "auto",
        **kwargs
    ):
        """
        Create detector.

        Args:
            model: Model path (.onnx, .engine, .xml)
            device: Target device ("auto", "cpu", "cuda:0", etc.)
            **kwargs: Additional configuration options
        """
        config = _CppLoadConfig()
        config.device = device
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self._cpp_detector = _CppDetector(model, device, config)
        self._device = device

    def predict(
        self,
        image: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        classes: Optional[List[int]] = None,
        max_detections: int = 300,
    ) -> Results:
        """
        Detect objects in image.

        Args:
            image: Input image (file path or numpy array)
            conf_threshold: Confidence threshold (0.0 - 1.0)
            iou_threshold: NMS IoU threshold (0.0 - 1.0)
            classes: Filter by class IDs (None = all classes)
            max_detections: Maximum number of detections

        Returns:
            Results object containing detections
        """
        if isinstance(image, str):
            return self._cpp_detector.predict(image, conf_threshold, iou_threshold)
        return self._cpp_detector.predict(image, conf_threshold, iou_threshold)

    def __call__(
        self,
        image: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        **kwargs
    ) -> Results:
        """Shorthand for predict()."""
        return self.predict(image, conf_threshold=conf_threshold,
                            iou_threshold=iou_threshold, **kwargs)

    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        **kwargs
    ) -> List[Results]:
        """
        Batch detection on multiple images.

        Args:
            images: List of images (paths or arrays)
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold

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

        config = _CppInferConfig()
        config.conf_threshold = conf_threshold
        config.iou_threshold = iou_threshold

        return self._cpp_detector.predict_batch(np_images, config)

    def predict_video(
        self,
        source: Union[str, int],
        callback: Callable[[Results, np.ndarray], None],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        **kwargs
    ) -> None:
        """
        Detect objects in video stream.

        Args:
            source: Video file path or camera ID
            callback: Function called for each frame: callback(results, frame)
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold
        """
        import cv2

        cap = cv2.VideoCapture(source)
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = self.predict(frame, conf_threshold=conf_threshold,
                                       iou_threshold=iou_threshold, **kwargs)
                callback(results, frame)
        finally:
            cap.release()

    @property
    def classes(self) -> List[str]:
        """Get class labels."""
        return list(self._cpp_detector.classes)

    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return self._cpp_detector.num_classes

    @property
    def input_size(self) -> Tuple[int, int]:
        """Get input size (width, height)."""
        return self._cpp_detector.input_size

    @property
    def device(self) -> str:
        """Get device being used."""
        return self._device
