"""
iVIT-SDK Custom Exceptions with Friendly Error Messages.

Provides detailed error messages with:
- Clear problem description
- Possible causes
- Suggested solutions
"""

from typing import List, Optional, Dict, Any
from pathlib import Path


class IVITError(Exception):
    """
    Base exception for all iVIT-SDK errors.

    Provides structured error messages with causes and suggestions.
    """

    def __init__(
        self,
        message: str,
        causes: Optional[List[str]] = None,
        suggestions: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.causes = causes or []
        self.suggestions = suggestions or []
        self.details = details or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with causes and suggestions."""
        lines = [f"\n{self.message}"]

        if self.causes:
            lines.append("\n可能原因：")
            for i, cause in enumerate(self.causes, 1):
                lines.append(f"  {i}. {cause}")

        if self.suggestions:
            lines.append("\n解決建議：")
            for suggestion in self.suggestions:
                lines.append(f"  - {suggestion}")

        if self.details:
            lines.append("\n詳細資訊：")
            for key, value in self.details.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)


class ModelLoadError(IVITError):
    """Error raised when model loading fails."""

    def __init__(
        self,
        model_path: str,
        reason: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        path = Path(model_path)
        causes = []
        suggestions = []
        details = {"model_path": model_path}

        if backend:
            details["backend"] = backend

        # Check if file exists
        if not path.exists():
            causes.append(f"檔案不存在：請確認路徑 '{model_path}' 是否正確")
            suggestions.append("檢查檔案路徑是否有拼寫錯誤")
            suggestions.append("使用絕對路徑確保檔案位置正確")
        else:
            # Check file extension
            supported_formats = [".onnx", ".xml", ".engine", ".trt", ".dlc", ".bin"]
            if path.suffix.lower() not in supported_formats:
                causes.append(f"格式不支援：目前支援 {', '.join(supported_formats)}")
                suggestions.append("使用 `ivit convert` 轉換模型格式")
            else:
                causes.append("模型檔案可能損壞或格式不相容")
                causes.append("推論後端可能未正確安裝")

        # Backend-specific suggestions
        if backend == "tensorrt" or (path.suffix.lower() in [".engine", ".trt"]):
            causes.append("TensorRT 需要 NVIDIA GPU")
            suggestions.append("執行 `ivit devices` 確認 CUDA 裝置可用")
            suggestions.append("確認已安裝 TensorRT 和 CUDA")
        elif backend == "openvino" or path.suffix.lower() == ".xml":
            causes.append("OpenVINO IR 模型需要對應的 .bin 權重檔")
            suggestions.append("確認 .xml 和 .bin 檔案在同一目錄")
            suggestions.append("確認已安裝 OpenVINO Runtime")
        elif backend == "snpe" or path.suffix.lower() == ".dlc":
            causes.append("SNPE 僅支援 Qualcomm 裝置")
            suggestions.append("確認在支援的 Qualcomm 硬體上執行")

        if reason:
            causes.append(reason)

        suggestions.append("執行 `ivit info` 查看系統資訊和可用後端")

        super().__init__(
            message=f"無法載入模型 '{path.name}'",
            causes=causes,
            suggestions=suggestions,
            details=details,
        )


class DeviceNotFoundError(IVITError):
    """Error raised when specified device is not available."""

    def __init__(self, device: str, available_devices: Optional[List[str]] = None):
        causes = [
            f"指定的裝置 '{device}' 不存在或不可用",
            "裝置驅動程式可能未安裝",
            "裝置可能被其他程序佔用",
        ]

        suggestions = ["執行 `ivit devices` 查看可用裝置"]

        if available_devices:
            suggestions.append(f"可用裝置：{', '.join(available_devices)}")

        if "cuda" in device.lower() or "gpu" in device.lower():
            causes.append("NVIDIA GPU 驅動程式可能未安裝")
            suggestions.append("執行 `nvidia-smi` 確認 GPU 狀態")
            suggestions.append("確認已安裝 CUDA Toolkit")
        elif "npu" in device.lower():
            causes.append("NPU 驅動程式可能未安裝")
            suggestions.append("確認已安裝 Intel NPU 驅動程式")

        suggestions.append("使用 `device='auto'` 讓系統自動選擇最佳裝置")

        super().__init__(
            message=f"找不到裝置 '{device}'",
            causes=causes,
            suggestions=suggestions,
            details={"requested_device": device},
        )


class BackendNotAvailableError(IVITError):
    """Error raised when a backend is not installed or available."""

    BACKEND_INSTALL_HINTS = {
        "openvino": "pip install -e .  # 建置 C++ binding（提供 OpenVINO 推論）",
        "tensorrt": "請參考 NVIDIA TensorRT 安裝指南",
        "snpe": "請參考 Qualcomm SNPE SDK 安裝指南",
    }

    def __init__(self, backend: str, reason: Optional[str] = None):
        causes = [
            f"後端 '{backend}' 未安裝",
            "相關依賴套件可能遺失",
        ]

        if reason:
            causes.append(reason)

        suggestions = []
        install_hint = self.BACKEND_INSTALL_HINTS.get(backend.lower())
        if install_hint:
            suggestions.append(f"安裝指令：{install_hint}")

        suggestions.append("執行 `ivit info` 查看已安裝的後端")
        suggestions.append("查看文件了解後端安裝需求")

        super().__init__(
            message=f"推論後端 '{backend}' 不可用",
            causes=causes,
            suggestions=suggestions,
            details={"backend": backend},
        )


class InferenceError(IVITError):
    """Error raised when inference fails."""

    def __init__(
        self,
        reason: str,
        model_name: Optional[str] = None,
        input_shape: Optional[tuple] = None,
        expected_shape: Optional[tuple] = None,
    ):
        causes = [reason]
        suggestions = []
        details = {}

        if model_name:
            details["model"] = model_name

        if input_shape and expected_shape:
            if input_shape != expected_shape:
                causes.append(f"輸入形狀不符：收到 {input_shape}，預期 {expected_shape}")
                suggestions.append(f"調整輸入影像大小為 {expected_shape}")
            details["input_shape"] = str(input_shape)
            details["expected_shape"] = str(expected_shape)

        suggestions.append("確認輸入資料格式正確（NumPy array 或影像路徑）")
        suggestions.append("使用 `model.input_info` 查看模型輸入規格")

        super().__init__(
            message="推論過程發生錯誤",
            causes=causes,
            suggestions=suggestions,
            details=details,
        )


class InvalidInputError(IVITError):
    """Error raised when input data is invalid."""

    def __init__(
        self,
        expected_type: str,
        received_type: str,
        param_name: Optional[str] = None,
    ):
        param_info = f"參數 '{param_name}'" if param_name else "輸入"

        causes = [
            f"{param_info}類型錯誤：收到 {received_type}，預期 {expected_type}",
        ]

        suggestions = []

        if "ndarray" in expected_type.lower():
            suggestions.append("使用 NumPy array 格式：`import numpy as np; img = np.array(...)`")
        if "str" in expected_type.lower() or "path" in expected_type.lower():
            suggestions.append("傳入檔案路徑字串：`model.predict('image.jpg')`")
        if "list" in expected_type.lower():
            suggestions.append("傳入列表格式：`model.predict([img1, img2])`")

        suggestions.append("查看 API 文件了解支援的輸入格式")

        super().__init__(
            message=f"{param_info}格式無效",
            causes=causes,
            suggestions=suggestions,
            details={
                "expected_type": expected_type,
                "received_type": received_type,
            },
        )


class ModelNotLoadedError(IVITError):
    """Error raised when trying to use a model that hasn't been loaded."""

    def __init__(self):
        super().__init__(
            message="模型尚未載入",
            causes=[
                "模型載入失敗或尚未呼叫載入函數",
                "模型可能已被釋放",
            ],
            suggestions=[
                "使用 `ivit.load('model.onnx')` 載入模型",
                "確認模型路徑正確",
            ],
        )


class ConfigurationError(IVITError):
    """Error raised when configuration is invalid."""

    def __init__(
        self,
        param_name: str,
        param_value: Any,
        valid_values: Optional[List[Any]] = None,
        reason: Optional[str] = None,
    ):
        causes = []
        if reason:
            causes.append(reason)
        else:
            causes.append(f"參數 '{param_name}' 的值 '{param_value}' 無效")

        suggestions = []
        if valid_values:
            suggestions.append(f"有效值：{', '.join(str(v) for v in valid_values)}")

        suggestions.append("查看 API 文件了解參數設定")

        super().__init__(
            message=f"設定錯誤：{param_name}",
            causes=causes,
            suggestions=suggestions,
            details={
                "parameter": param_name,
                "value": str(param_value),
            },
        )


class ModelConversionError(IVITError):
    """Error raised when model conversion fails."""

    def __init__(
        self,
        source_format: str,
        target_format: str,
        reason: Optional[str] = None,
    ):
        causes = [
            f"無法將模型從 {source_format} 轉換為 {target_format}",
        ]

        if reason:
            causes.append(reason)

        suggestions = []

        if target_format.lower() == "tensorrt":
            suggestions.append("確認已安裝 TensorRT 和 CUDA")
            suggestions.append("確認 NVIDIA GPU 可用")
        elif target_format.lower() == "openvino":
            suggestions.append("確認已建置 C++ binding（pip install -e .）以取得 convert_model")
            suggestions.append("INT8 量化改用 NNCF ONNX backend：pip install -e '.[quantize]'")

        suggestions.append("確認來源模型格式正確")
        suggestions.append("查看轉換日誌了解詳細錯誤")

        super().__init__(
            message=f"模型轉換失敗：{source_format} -> {target_format}",
            causes=causes,
            suggestions=suggestions,
            details={
                "source_format": source_format,
                "target_format": target_format,
            },
        )


class ResourceExhaustedError(IVITError):
    """Error raised when system resources are exhausted."""

    def __init__(self, resource_type: str, details_info: Optional[str] = None):
        causes = [f"{resource_type} 資源不足"]

        if details_info:
            causes.append(details_info)

        suggestions = []

        if "memory" in resource_type.lower() or "記憶體" in resource_type:
            suggestions.append("減少批次大小（batch_size）")
            suggestions.append("使用較小的模型")
            suggestions.append("釋放其他佔用記憶體的程序")
        elif "gpu" in resource_type.lower():
            suggestions.append("執行 `nvidia-smi` 查看 GPU 記憶體使用情況")
            suggestions.append("使用 `model.configure_tensorrt(workspace_size=...)` 調整工作區大小")

        suggestions.append("使用串流模式處理大量資料")

        super().__init__(
            message=f"{resource_type}資源耗盡",
            causes=causes,
            suggestions=suggestions,
        )


class UnsupportedOperationError(IVITError):
    """Error raised when an operation is not supported."""

    def __init__(
        self,
        operation: str,
        reason: Optional[str] = None,
        alternatives: Optional[List[str]] = None,
    ):
        causes = [f"操作 '{operation}' 不支援"]

        if reason:
            causes.append(reason)

        suggestions = alternatives or []
        suggestions.append("查看 API 文件了解支援的操作")

        super().__init__(
            message=f"不支援的操作：{operation}",
            causes=causes,
            suggestions=suggestions,
        )


# Helper function to wrap common errors
def wrap_error(error: Exception, context: str = "") -> IVITError:
    """
    Wrap a generic exception into an IVITError with helpful context.

    Args:
        error: The original exception
        context: Additional context about what operation was being performed

    Returns:
        An IVITError with helpful debugging information
    """
    error_type = type(error).__name__
    error_msg = str(error)

    causes = [f"發生 {error_type}: {error_msg}"]
    suggestions = [
        "檢查輸入資料和參數是否正確",
        "確認所有依賴套件已正確安裝",
    ]

    if context:
        causes.insert(0, f"在 {context} 時發生錯誤")

    # Add specific suggestions based on error type
    if "CUDA" in error_msg or "cuda" in error_msg:
        suggestions.append("執行 `nvidia-smi` 確認 GPU 狀態")
        suggestions.append("確認 CUDA 驅動程式版本相容")
    elif "memory" in error_msg.lower():
        suggestions.append("減少批次大小或使用較小的模型")
    elif "shape" in error_msg.lower() or "dimension" in error_msg.lower():
        suggestions.append("檢查輸入資料的形狀是否符合模型要求")

    return IVITError(
        message=f"操作失敗：{context}" if context else "操作失敗",
        causes=causes,
        suggestions=suggestions,
        details={"original_error": f"{error_type}: {error_msg}"},
    )
