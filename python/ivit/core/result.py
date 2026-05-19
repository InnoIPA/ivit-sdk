"""
Results container for iVIT-SDK.
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import json
from pathlib import Path

from .types import (
    Detection,
    ClassificationResult,
    Keypoint,
    Pose,
    BBox,
)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class Results:
    """
    Unified results container for all task types.

    Attributes:
        classifications: List of classification results
        detections: List of detection results
        segmentation_mask: Segmentation mask array
        poses: List of pose results
        raw_outputs: Raw model outputs
        inference_time_ms: Inference time in milliseconds
        device_used: Device used for inference
        image_size: Original image size (H, W)
    """

    def __init__(self):
        # Classification
        self.classifications: List[ClassificationResult] = []

        # Detection
        self.detections: List[Detection] = []

        # Segmentation
        self.segmentation_mask: Optional[np.ndarray] = None

        # Pose
        self.poses: List[Pose] = []

        # Raw outputs
        self.raw_outputs: Dict[str, np.ndarray] = {}

        # Metadata
        self.inference_time_ms: float = 0.0
        self.device_used: str = ""
        self.image_size: Tuple[int, int] = (0, 0)
        self._original_image: Optional[np.ndarray] = None

        # Stream metadata
        self.frame_idx: int = -1
        self.source_fps: float = 0.0

    # =========================================================================
    # Classification
    # =========================================================================

    @property
    def top1(self) -> ClassificationResult:
        """Get top-1 classification result."""
        if not self.classifications:
            raise ValueError("No classification results")
        return self.classifications[0]

    @property
    def top5(self) -> List[ClassificationResult]:
        """Get top-5 classification results."""
        return self.classifications[:5]

    def topk(self, k: int) -> List[ClassificationResult]:
        """Get top-K classification results."""
        return self.classifications[:k]

    # =========================================================================
    # Detection
    # =========================================================================

    def filter_by_class(
        self,
        classes: List[str]
    ) -> List[Detection]:
        """Filter detections by class labels."""
        return [d for d in self.detections if d.label in classes]

    def filter_by_confidence(
        self,
        min_conf: float
    ) -> List[Detection]:
        """Filter detections by minimum confidence."""
        return [d for d in self.detections if d.confidence >= min_conf]

    def filter_by_area(
        self,
        min_area: float,
        max_area: Optional[float] = None
    ) -> List[Detection]:
        """Filter detections by bounding box area."""
        result = [d for d in self.detections if d.bbox.area >= min_area]
        if max_area is not None:
            result = [d for d in result if d.bbox.area <= max_area]
        return result

    def filter(
        self,
        confidence: Optional[float] = None,
        classes: Optional[List[str]] = None,
        min_area: Optional[float] = None,
        max_area: Optional[float] = None,
    ) -> 'Results':
        """
        Filter results with multiple criteria.

        Unified filter method that can combine multiple filtering conditions.
        Returns a new Results object with filtered detections.

        Args:
            confidence: Minimum confidence threshold
            classes: List of class labels to keep
            min_area: Minimum bounding box area
            max_area: Maximum bounding box area

        Returns:
            New Results object with filtered detections

        Examples:
            >>> # Filter by confidence
            >>> filtered = results.filter(confidence=0.9)
            >>>
            >>> # Filter by class
            >>> filtered = results.filter(classes=["person", "car"])
            >>>
            >>> # Combine filters
            >>> filtered = results.filter(confidence=0.8, classes=["person"])
        """
        # Start with all detections
        filtered_detections = self.detections.copy()

        # Apply confidence filter
        if confidence is not None:
            filtered_detections = [
                d for d in filtered_detections
                if d.confidence >= confidence
            ]

        # Apply class filter
        if classes is not None:
            filtered_detections = [
                d for d in filtered_detections
                if d.label in classes
            ]

        # Apply area filter
        if min_area is not None:
            filtered_detections = [
                d for d in filtered_detections
                if d.bbox.area >= min_area
            ]

        if max_area is not None:
            filtered_detections = [
                d for d in filtered_detections
                if d.bbox.area <= max_area
            ]

        # Create new Results object with filtered detections
        new_results = Results()
        new_results.detections = filtered_detections
        new_results.classifications = self.classifications.copy()
        new_results.segmentation_mask = self.segmentation_mask
        new_results.poses = self.poses.copy()
        new_results.raw_outputs = self.raw_outputs.copy()
        new_results.inference_time_ms = self.inference_time_ms
        new_results.device_used = self.device_used
        new_results.image_size = self.image_size
        new_results._original_image = self._original_image
        new_results.frame_idx = self.frame_idx
        new_results.source_fps = self.source_fps

        return new_results

    # =========================================================================
    # Segmentation
    # =========================================================================

    def colorize_mask(
        self,
        colormap: Optional[Dict[int, Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """
        Colorize segmentation mask.

        Args:
            colormap: Custom color mapping {class_id: (R, G, B)}

        Returns:
            RGB colorized mask
        """
        if self.segmentation_mask is None:
            raise ValueError("No segmentation mask")

        mask = self.segmentation_mask
        h, w = mask.shape[:2]
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        # Generate default colormap if not provided
        if colormap is None:
            colormap = self._generate_colormap(int(mask.max()) + 1)

        for class_id, color in colormap.items():
            colored[mask == class_id] = color

        return colored

    def overlay_mask(
        self,
        image: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay segmentation mask on image.

        Args:
            image: Original image (BGR)
            alpha: Transparency (0-1)

        Returns:
            Image with overlaid mask
        """
        colored = self.colorize_mask()
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            overlay = image.copy()
            mask_rgb = colored[:, :, ::-1]  # RGB to BGR
            cv2_available = True
            try:
                import cv2
                overlay = cv2.addWeighted(overlay, 1 - alpha, mask_rgb, alpha, 0)
            except ImportError:
                overlay = (overlay * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)
            return overlay
        return image

    def get_contours(
        self,
        class_id: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Extract contours from segmentation mask.

        Args:
            class_id: Specific class (None for all)

        Returns:
            List of contour arrays
        """
        if self.segmentation_mask is None:
            raise ValueError("No segmentation mask")

        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV required for contour extraction")

        contours = []
        mask = self.segmentation_mask

        if class_id is not None:
            binary = (mask == class_id).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.extend(cnts)
        else:
            for cid in np.unique(mask):
                if cid == 0:  # Skip background
                    continue
                binary = (mask == cid).astype(np.uint8) * 255
                cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours.extend(cnts)

        return contours

    # =========================================================================
    # Quick Visualization
    # =========================================================================

    def show(
        self,
        wait: bool = True,
        window_name: str = "iVIT Results",
    ) -> None:
        """
        Display results in a window.

        Quick visualization method.

        Args:
            wait: Wait for key press to close window
            window_name: Window title

        Examples:
            >>> results = model("image.jpg")
            >>> results.show()  # Display and wait for key press
            >>> results.show(wait=False)  # Display without blocking
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV required for show(). Install: pip install opencv-python")

        vis = self.visualize()

        cv2.imshow(window_name, vis)
        if wait:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.waitKey(1)

    def plot(
        self,
        show_labels: bool = True,
        show_confidence: bool = True,
        line_width: int = 2,
        font_size: float = 0.5,
    ) -> np.ndarray:
        """
        Plot results on image and return.

        Render annotated image.

        Args:
            show_labels: Show class labels
            show_confidence: Show confidence scores
            line_width: Bounding box line width
            font_size: Label font size

        Returns:
            Image with plotted results

        Examples:
            >>> results = model("image.jpg")
            >>> plotted = results.plot()
            >>> cv2.imwrite("output.jpg", plotted)
        """
        return self.visualize(
            show_labels=show_labels,
            show_confidence=show_confidence,
        )

    # =========================================================================
    # Visualization
    # =========================================================================

    def visualize(
        self,
        image: Optional[np.ndarray] = None,
        show_labels: bool = True,
        show_confidence: bool = True,
        show_boxes: bool = True,
        show_masks: bool = True,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Visualize results on image.

        Args:
            image: Input image (uses cached if None)
            show_labels: Show class labels
            show_confidence: Show confidence scores
            show_boxes: Show bounding boxes
            show_masks: Show segmentation masks

        Returns:
            Visualized image
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV required for visualization")

        if image is None:
            image = self._original_image
        if image is None:
            raise ValueError("No image provided")

        vis = image.copy()

        # Draw segmentation mask
        if show_masks and self.segmentation_mask is not None:
            vis = self.overlay_mask(vis, alpha=0.5)

        # Draw detections
        if show_boxes and self.detections:
            for det in self.detections:
                color = self._get_color(det.class_id)
                x1, y1 = int(det.bbox.x1), int(det.bbox.y1)
                x2, y2 = int(det.bbox.x2), int(det.bbox.y2)

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

                if show_labels or show_confidence:
                    label_parts = []
                    if show_labels:
                        label_parts.append(det.label)
                    if show_confidence:
                        label_parts.append(f"{det.confidence:.2f}")
                    label = " ".join(label_parts)

                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                    cv2.putText(vis, label, (x1, y1 - 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw poses
        if self.poses:
            vis = self._draw_poses(vis, self.poses)

        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, vis)

        return vis

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, cls=NumpyEncoder)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "inference_time_ms": self.inference_time_ms,
            "device_used": self.device_used,
            "image_size": list(self.image_size),
        }

        if self.classifications:
            result["classifications"] = [
                {
                    "class_id": c.class_id,
                    "label": c.label,
                    "score": c.score,
                }
                for c in self.classifications
            ]

        if self.detections:
            result["detections"] = [
                {
                    "class_id": d.class_id,
                    "label": d.label,
                    "confidence": d.confidence,
                    "bbox": {
                        "x1": d.bbox.x1,
                        "y1": d.bbox.y1,
                        "x2": d.bbox.x2,
                        "y2": d.bbox.y2,
                    }
                }
                for d in self.detections
            ]

        if self.poses:
            result["poses"] = [
                {
                    "keypoints": [
                        {"x": kp.x, "y": kp.y, "confidence": kp.confidence, "name": kp.name}
                        for kp in p.keypoints
                    ],
                    "confidence": p.confidence,
                }
                for p in self.poses
            ]

        return result

    def save(
        self,
        path: str = None,
        format: str = "auto"
    ) -> str:
        """
        Save results to file.

        Supports saving visualized images or result data.

        Args:
            path: Output path (auto-generates if None)
            format: Format
                - "auto": Detect from extension
                - Image formats: jpg, png, bmp (saves visualization)
                - Data formats: json, txt (saves result data)

        Returns:
            Path to saved file

        Examples:
            >>> results.save("output.jpg")   # Save visualization
            >>> results.save("output.json")  # Save result data
            >>> results.save()               # Auto-save with timestamp
        """
        # Auto-generate path if not provided
        if path is None:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = f"ivit_result_{timestamp}.jpg"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "auto":
            format = path.suffix.lstrip(".").lower()

        # Image formats - save visualization
        if format in ("jpg", "jpeg", "png", "bmp", "tiff"):
            try:
                import cv2
            except ImportError:
                raise ImportError("OpenCV required for image save")

            vis = self.visualize()
            cv2.imwrite(str(path), vis)

        # JSON format
        elif format == "json":
            with open(path, "w") as f:
                f.write(self.to_json())

        # YOLO TXT format
        elif format == "txt":
            with open(path, "w") as f:
                for det in self.detections:
                    cx, cy, w, h = det.bbox.to_cxcywh()
                    ih, iw = self.image_size
                    f.write(f"{det.class_id} {cx/iw:.6f} {cy/ih:.6f} {w/iw:.6f} {h/ih:.6f}\n")
        else:
            raise ValueError(f"Unsupported format: {format}")

        return str(path)

    # =========================================================================
    # Iteration
    # =========================================================================

    def __len__(self) -> int:
        """Get total number of results."""
        return max(
            len(self.classifications),
            len(self.detections),
            len(self.poses),
        )

    def __iter__(self):
        """Iterate over detections (most common use case)."""
        return iter(self.detections)

    def __getitem__(self, idx: int) -> Detection:
        """Get detection by index."""
        return self.detections[idx]

    @property
    def empty(self) -> bool:
        """Check if results are empty (no detections, classifications, or poses)."""
        return len(self) == 0

    # =========================================================================
    # Private methods
    # =========================================================================

    @staticmethod
    def _get_color(class_id: int) -> Tuple[int, int, int]:
        """Get color for class ID."""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 0), (128, 0, 128), (0, 128, 128),
        ]
        return colors[class_id % len(colors)]

    @staticmethod
    def _generate_colormap(num_classes: int) -> Dict[int, Tuple[int, int, int]]:
        """Generate default colormap."""
        import colorsys
        colormap = {}
        for i in range(num_classes):
            hue = i / num_classes
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
            colormap[i] = tuple(int(c * 255) for c in rgb)
        return colormap

    @staticmethod
    def _draw_poses(
        image: np.ndarray,
        poses: List[Pose]
    ) -> np.ndarray:
        """Draw poses on image."""
        try:
            import cv2
        except ImportError:
            return image

        # COCO skeleton
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
        ]

        for pose in poses:
            # Draw keypoints
            for kp in pose.keypoints:
                if kp.confidence > 0.5:
                    cv2.circle(image, (int(kp.x), int(kp.y)), 3, (0, 255, 0), -1)

            # Draw skeleton
            for i, j in skeleton:
                if i < len(pose.keypoints) and j < len(pose.keypoints):
                    kp1, kp2 = pose.keypoints[i], pose.keypoints[j]
                    if kp1.confidence > 0.5 and kp2.confidence > 0.5:
                        cv2.line(
                            image,
                            (int(kp1.x), int(kp1.y)),
                            (int(kp2.x), int(kp2.y)),
                            (255, 0, 0), 2
                        )

        return image
