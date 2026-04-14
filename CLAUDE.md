# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概述

iVIT-SDK (Innodisk Vision Intelligence Toolkit) 是宜鼎國際為其 AI 運算平台開發的統一電腦視覺推論與訓練 SDK。提供跨硬體平台的統一 API，目前支援 Intel (OpenVINO)、NVIDIA (TensorRT)，並採用可擴展架構以便新增更多加速器平台（如 Qualcomm QNN）。

> **iVIT 命名規則**：開頭的 "i" 代表 Innodisk，VIT 代表 Vision Intelligence Toolkit。

本專案採用 AI 虛擬團隊協作開發模式，透過多角色分工提升開發效率與品質。

## 虛擬開發團隊

| 角色 | 專長 | 技能檔案 |
|------|------|---------|
| AI 硬體產品經理 | GPU/NPU/ASIC、AI 晶片選型 | `.claude/skills/ai-hardware-product-manager/SKILL.md` |
| AI 軟體產品經理 | LLM 應用、AI 框架、產品策略 | `.claude/skills/ai-software-product-manager/SKILL.md` |
| 系統架構師 | 系統設計、技術選型、架構文件 | `.claude/skills/system-architect/SKILL.md` |
| 軟體工程師 | 程式開發、程式碼品質、測試 | `.claude/skills/software-engineer/SKILL.md` |
| QA 工程師 | 測試規劃、自動化測試、品質保證 | `.claude/skills/qa-engineer/SKILL.md` |
| UX 設計師 | 使用者體驗、介面設計、原型 | `.claude/skills/ux-designer/SKILL.md` |

## 開發指令

```bash
# 建置 C++ 核心（必須在安裝 Python 套件前完成）
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DIVIT_USE_OPENVINO=ON \   # 啟用 OpenVINO（預設）
         -DIVIT_USE_TENSORRT=ON \   # 啟用 TensorRT（預設）
         -DIVIT_USE_QNN=OFF          # 啟用 Qualcomm QNN（選用）
make -j$(nproc)

# 建置並安裝 Python 套件（scikit-build 驅動，會自動觸發 cmake）
pip install -e .                         # 基本安裝
pip install -e ".[openvino]"             # 含 OpenVINO Python 套件
pip install -e ".[tensorrt]"             # 含 TensorRT Python 套件
pip install -e ".[zoo,train]"            # 含模型下載與訓練功能
pip install -e ".[dev]"                  # 含開發工具（pytest、black 等）

# 執行所有 Python 測試
pytest tests/python/ -v

# 執行特定測試
pytest tests/python/test_zoo_download.py -v
pytest tests/python/test_api.py -v

# 執行整合測試（需要 C++ 綁定 + 硬體）
pytest tests/integration/test_user_scenarios.py -v

# 執行 C++ 測試
ctest --test-dir build

# Lint
flake8 python/
black python/
clang-format -i src/**/*.cpp include/**/*.hpp

# CLI（安裝後可用）
ivit info
ivit devices
ivit benchmark models/yolox-s.onnx --device cuda:0
ivit predict models/yolox-s.onnx image.jpg --conf 0.5
ivit convert models/yolox-s.onnx -f openvino -o models/
ivit zoo list
ivit zoo download yolox-s
```

## 架構概覽

### 分層架構

```
Python API (ivit.*) ──────────────────────────────
  pybind11 bindings (_ivit_core.so)  ────────────
    C++ Vision Layer (Classifier/Detector/Segmentor)
      C++ Core (Model / ModelManager)  ──────────
        Runtime Abstraction (IRuntime)  ────────
          OpenVINORuntime │ TensorRTRuntime │ QNNRuntime
```

### 關鍵設計點

**1. Runtime 插件系統（`src/runtime/` + `include/ivit/runtime/`）**

所有後端實作 `IRuntime` 純虛擬介面（`runtime.hpp`）。`RuntimeFactory` 是 singleton，在程式庫載入時由 `RuntimeAutoRegister`（`runtime_init.cpp` 中的全域靜態物件）自動呼叫 `register_runtime()`。新增後端只需：實作 `IRuntime`，在 `runtime_init.cpp` 加入 `#ifdef IVIT_HAS_XXX` 區塊，在 CMakeLists.txt 新增對應 option。

**2. Python 優雅降級（`python/ivit/__init__.py`）**

C++ 綁定（`_ivit_core.so`）為可選：若 import 失敗，`_CPP_AVAILABLE = False`，Model Zoo CLI 和純 Python 功能仍可用。需要 C++ 的函式呼叫 `_require_cpp()` 產生明確錯誤訊息。

**3. Vision Layer（`src/vision/` + `python/ivit/vision/`）**

`Classifier`、`Detector`、`Segmentor` 封裝了各自的前處理（resize、normalize）和後處理（softmax、NMS、mask decode）邏輯，透過 `ModelManager` 單例管理載入的 model handle。Python 端對應同名類別作為 C++ 類別的輕薄封裝。

**4. Model Zoo（`python/ivit/zoo/registry.py`）**

純 Python，不依賴 C++ 綁定。`MODEL_REGISTRY` 是以模型名稱為 key 的 dict，值為 `ModelInfo` dataclass（包含 `onnx_url` 直接下載欄位）。模型 ONNX 從 GitHub Releases 下載，存放於本地 cache 目錄。

**5. Training Module（`python/ivit/train/`）**

PyTorch 依賴，lazy import（不影響無 torch 環境）。`ModelExporter` 支援匯出 ONNX、TorchScript、OpenVINO IR、TensorRT engine，含 INT8 量化（需要 calibration data）。

**6. pybind11 綁定（`python/ivit/_binding.cpp`）**

`cv::Mat` ↔ NumPy 的轉換在此處理。C++ `Results` 物件（含 `BBox`、`Detection` 等）也在此轉換為 Python 友善形式。

### CMake 關鍵 Flag

| Flag | 預設 | 說明 |
|------|------|------|
| `IVIT_USE_OPENVINO` | ON | 連結 OpenVINO Runtime |
| `IVIT_USE_TENSORRT` | ON | 連結 TensorRT |
| `IVIT_USE_QNN` | OFF | 連結 Qualcomm QNN |
| `IVIT_BUILD_PYTHON` | ON | 建置 pybind11 `.so` |
| `IVIT_BUNDLE_DEPS` | OFF | 打包依賴庫進 SDK（用於發行） |

若系統找不到 OpenVINO，CMake 會自動偵測 pip 安裝的 OpenVINO（透過 `python3 -c "import openvino"` 取得路徑）。

**ABI 注意**：pip 安裝的 OpenVINO 使用舊 ABI（`_GLIBCXX_USE_CXX11_ABI=0`），但系統 OpenCV 通常使用新 ABI。CMakeLists.txt 強制設定 `_GLIBCXX_USE_CXX11_ABI=1` 以與 OpenCV 相容。

## 硬體支援

| 廠商 | 硬體類型 | 推論引擎 | x86 | ARM |
|------|----------|----------|-----|-----|
| Intel | CPU/iGPU/NPU/VPU | OpenVINO | ✅ | ✅ |
| NVIDIA | dGPU/Jetson | TensorRT | ✅ | ✅ |
| Qualcomm | HTP/Adreno GPU (IQ8/IQ9) | QNN (AI Engine Direct) | ⬜ | ✅ |

## 開發規範

- **語言**：C++ 17（Google C++ Style Guide）、Python 3.9+（PEP 8、Black）
- **Commit 格式**：Conventional Commits
- **測試路徑**：`tests/python/`（pytest）、`tests/integration/`（需要硬體）、`tests/cpp/`（ctest）

## 相關文件

- [PRD-001: iVIT-SDK 產品需求文件](docs/development/prd.md)
- [ADR-001: 系統架構設計](docs/architecture/adr-001-system.md)
- [API-SPEC-001: API 規格文件](docs/api/api-spec.md)
