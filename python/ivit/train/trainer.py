"""
向後相容 shim：`Trainer` 是 `ClassificationTrainer` 的別名。

此檔保留只是為了維持 `from ivit.train.trainer import Trainer` 等既有 import 不破。
新程式碼請直接使用 `from ivit.train import ClassificationTrainer`（或對應任務的子類別）。
"""

from .classification import ClassificationTrainer
from .classification import ClassificationTrainer as Trainer

__all__ = ["Trainer", "ClassificationTrainer"]
