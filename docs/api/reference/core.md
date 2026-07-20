# Core

核心型別與設定。

!!! note "資料結構的權威文件在 C++ 端"
    `Model`、`Results`、`BBox`、`Detection`、`ClassificationResult`、`Keypoint`、`Pose` 以及 `DataType` / `Precision` / `Layout` 等 enum 是由 C++ 綁定（`_ivit_core`）匯出的原生型別，在 Python 端只有精簡說明。
    完整欄位與方法說明請見 **[C++ API Reference](../cpp/html/index.html)**（`ivit::Model`、`ivit::Results`、`ivit::BBox` …）。

## 設定管理

::: ivit.core.config

## 前處理 / 後處理器

::: ivit.core.processors

## Callbacks

::: ivit.core.callbacks

## 例外

::: ivit.core.exceptions
