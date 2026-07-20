# Python API Reference

本區由 [mkdocstrings](https://mkdocstrings.github.io/) 直接從 `python/ivit/` 的原始碼 docstring 自動產生，內容永遠與程式碼同步。

## 模組總覽

| 模組 | 內容 |
|------|------|
| [`ivit`](#top-level) | 頂層進入點：`load` / `load_model` |
| [Vision](vision.md) | `Detector` / `Classifier` / `Segmentor` |
| [Core](core.md) | `Model`、`Results`、config、callbacks、processors |
| [Model Zoo](zoo.md) | 模型清單、搜尋、下載 |
| [Training](train.md) | Trainer、Dataset、augmentation、`ModelExporter` |
| [Utils](utils.md) | `Visualizer`、`Profiler`、video、metrics |
| [Devices & CLI](cli.md) | `DeviceManager`、`ivit` 命令列工具 |

!!! tip "查詢視覺任務類別請看 [Vision](vision.md)"
    `ivit.vision.Detector` 等包裝類別帶有完整的參數說明與範例；頂層 `ivit.Detector` 是 C++ 綁定的原生類別，說明較精簡。

## 頂層進入點 { #top-level }

推薦使用高階入口 `ivit.load()`（自動偵測任務型別與裝置，回傳對應的 vision 類別）。
低階的 `ivit.load_model()` 是 C++ 綁定函式，回傳 [`Model`](../cpp/html/index.html)，其完整說明見 C++ API Reference。

::: ivit.load
