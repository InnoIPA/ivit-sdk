"""
Model exporter for converting trained models to deployment formats.

Supports ONNX, TorchScript, and backend-specific formats.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ModelExporter:
    """
    Export trained models to various deployment formats.

    Supports:
    - ONNX (cross-platform)
    - TorchScript (PyTorch native)
    - OpenVINO IR (Intel optimization)
    - TensorRT Engine (NVIDIA optimization)

    Args:
        model: PyTorch model to export
        device: Device the model is on

    Examples:
        >>> exporter = ModelExporter(model, device)
        >>> exporter.export("model.onnx", format="onnx")
        >>> exporter.export("model.xml", format="openvino", quantize="int8", calibration_data=images)
    """

    def __init__(self, model: Any, device: Any):
        self.model = model
        self.device = device

    def export(
        self,
        path: str,
        format: str = "onnx",
        optimize_for: Optional[str] = None,
        quantize: Optional[str] = None,
        input_shape: tuple = (1, 3, 224, 224),
        opset_version: int = 17,
        class_names: Optional[List[str]] = None,
        calibration_data: Optional[List[np.ndarray]] = None,
        task: str = "classify",
        **kwargs,
    ) -> str:
        """
        Export model to specified format.

        Args:
            path: Output path
            format: Export format ("onnx", "torchscript", "openvino", "tensorrt")
            optimize_for: Target hardware optimization
            quantize: Quantization mode ("fp16", "int8")
            input_shape: Model input shape
            opset_version: ONNX opset version
            class_names: Class names for metadata
            calibration_data: Data for INT8 calibration
            task: 任務型態 — "classify" / "detect" / "segment"，
                  影響 ONNX 輸出 names 與 dynamic_axes

        Returns:
            Path to exported model
        """
        format = format.lower()
        if quantize not in (None, "fp16", "int8"):
            raise ValueError("quantize must be fp16, int8 or None")
        if format in ("onnx", "tensorrt", "torchscript") and quantize == "int8":
            raise ValueError(
                "ONNX INT8 export requires calibration and is not implemented"
            )
        if format == "openvino" and quantize == "int8" and not calibration_data:
            raise ValueError("OpenVINO INT8 requires nonempty calibration_data")

        # Create output directory
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        if format == "onnx":
            return self._export_onnx(
                path, input_shape, opset_version, quantize, class_names, task
            )
        elif format == "torchscript":
            return self._export_torchscript(path, input_shape)
        elif format == "openvino":
            exported = self._export_openvino(
                path, input_shape, quantize, calibration_data, task
            )
            if class_names:
                Path(exported).with_suffix(".labels.txt").write_text(
                    "\n".join(class_names), encoding="utf-8"
                )
            return exported
        elif format == "tensorrt":
            exported = self._export_tensorrt(path, input_shape, quantize, task)
            if class_names:
                Path(exported).with_suffix(".labels.txt").write_text(
                    "\n".join(class_names), encoding="utf-8"
                )
            return exported
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _onnx_io_spec(
        self, task: str
    ) -> Tuple[List[str], List[str], Dict[str, Dict[int, str]]]:
        """依 task 回傳 (input_names, output_names, dynamic_axes)。"""
        task = (task or "classify").lower()
        if task == "detect":
            # YOLOX raw output：(batch, num_dets, 5+num_classes) 或 backbone-style multi-head；
            # 這裡先以單張量輸出為準（YOLOX 標準 export 形式），dynamic 包含 batch 與 anchors 數
            return (
                ["input"],
                ["output"],
                {"input": {0: "batch"}, "output": {0: "batch", 1: "num_dets"}},
            )
        if task == "segment":
            return (
                ["input"],
                ["logits"],
                {
                    "input": {0: "batch", 2: "height", 3: "width"},
                    "logits": {0: "batch", 2: "height", 3: "width"},
                },
            )
        # classify
        return (
            ["input"],
            ["output"],
            {"input": {0: "batch"}, "output": {0: "batch"}},
        )

    def _export_onnx(
        self,
        path: str,
        input_shape: tuple,
        opset_version: int,
        quantize: Optional[str],
        class_names: Optional[List[str]],
        task: str = "classify",
    ) -> str:
        """Export to ONNX format."""
        import torch

        self.model.eval()

        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)

        input_names, output_names, dynamic_axes = self._onnx_io_spec(task)

        logger.info(f"Exporting to ONNX (task={task}): {path}")
        torch.onnx.export(
            self.model,
            dummy_input,
            path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
        )

        # Apply quantization if requested
        if quantize == "fp16":
            path = self._quantize_onnx_fp16(path)
        elif quantize == "int8":
            raise ValueError(
                "ONNX INT8 export requires calibration and is not implemented; no INT8 model was produced"
            )

        # Save class names
        if class_names:
            labels_path = Path(path).with_suffix(".labels.txt")
            with open(labels_path, "w", encoding="utf-8") as f:
                f.write("\n".join(class_names))
            logger.info(f"Saved labels to {labels_path}")

        logger.info(f"ONNX export complete: {path}")
        return path

    def _quantize_onnx_fp16(self, path: str) -> str:
        """Convert ONNX model to FP16."""
        try:
            from onnxconverter_common import float16
            import onnx

            model = onnx.load(path)
            model_fp16 = float16.convert_float_to_float16(model)

            fp16_path = str(Path(path).with_suffix("")) + "_fp16.onnx"
            onnx.save(model_fp16, fp16_path)

            logger.info(f"FP16 quantization complete: {fp16_path}")
            return fp16_path

        except ImportError as error:
            raise ImportError(
                "FP16 ONNX conversion requires onnxconverter-common"
            ) from error

    def _export_torchscript(
        self,
        path: str,
        input_shape: tuple,
    ) -> str:
        """Export to TorchScript format."""
        import torch

        self.model.eval()

        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)

        # Trace model
        logger.info(f"Exporting to TorchScript: {path}")
        traced = torch.jit.trace(self.model, dummy_input)
        traced.save(path)

        logger.info(f"TorchScript export complete: {path}")
        return path

    def _export_openvino(
        self,
        path: str,
        input_shape: tuple,
        quantize: Optional[str],
        calibration_data: Optional[List[np.ndarray]],
        task: str = "classify",
    ) -> str:
        """Export to OpenVINO IR format."""
        # First export to ONNX
        onnx_path = str(Path(path).with_suffix(".onnx"))
        self._export_onnx(onnx_path, input_shape, 17, None, None, task)

        try:
            import openvino as ov
            from openvino import Core
            from openvino.runtime import serialize

            core = Core()
            logger.info(f"Converting to OpenVINO IR: {path}")

            # Read ONNX model
            model = core.read_model(onnx_path)

            # Apply quantization if requested
            if quantize == "int8" and calibration_data is not None:
                model = self._quantize_openvino_int8(model, calibration_data)
            elif quantize == "fp16":
                # Compress weights to FP16 during serialization
                xml_path = str(Path(path).with_suffix(".xml"))
                ov.save_model(model, xml_path, compress_to_fp16=True)
                logger.info(f"OpenVINO IR export complete (FP16): {xml_path}")
                return xml_path

            # Serialize to IR format
            xml_path = str(Path(path).with_suffix(".xml"))
            serialize(model, xml_path)

            logger.info(f"OpenVINO IR export complete: {xml_path}")
            return xml_path

        except ImportError as error:
            raise ImportError("OpenVINO export requires openvino") from error

    def _quantize_openvino_int8(self, model, calibration_data: List[np.ndarray]):
        """Apply INT8 quantization using NNCF."""
        try:
            import nncf

            # Create calibration dataset
            def transform_fn(data_item):
                return np.expand_dims(data_item, 0)

            calibration_dataset = nncf.Dataset(calibration_data, transform_fn)

            # Quantize model
            quantized_model = nncf.quantize(
                model,
                calibration_dataset,
                preset=nncf.QuantizationPreset.MIXED,
            )

            logger.info("INT8 quantization complete")
            return quantized_model

        except ImportError as error:
            raise ImportError("OpenVINO INT8 calibration requires nncf") from error

    def _export_tensorrt(
        self,
        path: str,
        input_shape: tuple,
        quantize: Optional[str],
        task: str = "classify",
    ) -> str:
        """Export to TensorRT engine format."""
        # First export to ONNX
        onnx_path = str(Path(path).with_suffix(".onnx"))
        self._export_onnx(onnx_path, input_shape, 17, None, None, task)

        import ivit

        precision = "fp16" if quantize == "fp16" else "fp32"
        engine_path = str(Path(path).with_suffix(".engine"))
        ivit.convert_model(onnx_path, engine_path, device="cuda:0", precision=precision)
        return engine_path
