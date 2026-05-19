"""
Model loading and management.

This module provides the load_model function which delegates to C++ bindings.
"""

from typing import Optional

from .._ivit_core import (
    LoadConfig,
    load_model as _cpp_load_model,
)


def load_model(
    source: str,
    device: str = "auto",
    task: str = None,
    **kwargs
):
    """
    Load a model for inference.

    Args:
        source: Model path (.onnx, .xml, .engine) or Model Zoo name
        device: Target device ("auto", "cpu", "cuda:0", "npu")
        task: Task type hint ("classification", "detection", "segmentation")
        **kwargs: Additional LoadConfig options

    Returns:
        Model: C++ Model object ready for inference

    Examples:
        >>> model = load_model("yolox-s.onnx", device="cuda:0", task="detection")
        >>> results = model.predict("image.jpg")
    """
    config = LoadConfig()
    config.device = device
    if task:
        config.task = task
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return _cpp_load_model(source, config)
