# iVIT-SDK 發布自動化指南

> 本文件描述既有平台交付工具。2026-10-07 清理驗證涵蓋來源封裝與本機 OpenVINO wheel；deb／Jetson CI 未於本輪重建，正式交付須另行驗收。來源封裝使用 `python3 scripts/package_source.py --output /tmp/ivit-source-release`。

本文件說明 iVIT-SDK 的自動化打包與發布流程。

## 概述

iVIT-SDK 提供兩種交付方案（tarball、deb），支援三個目標平台：

| 平台 ID | 架構 | 後端 | 適用場景 |
|---------|------|------|----------|
| `x86_64-nvidia` | x86_64 | TensorRT | 桌機/伺服器 + NVIDIA GPU |
| `x86_64-intel` | x86_64 | OpenVINO | 桌機/伺服器 + Intel CPU/iGPU/NPU |
| `aarch64-jetson` | aarch64 | TensorRT | NVIDIA Jetson 嵌入式平台 |

| 交付方案 | 說明 | 依賴處理 |
|----------|------|----------|
| **Tarball** | 預編譯壓縮包（`.tar.xz`） | 內含後端函式庫（`deps/lib`：OpenVINO 與其 `openvino-<版本>/` 裝置外掛目錄，或 TensorRT／CUDA runtime）；OpenCV、`libtbb12` 與 GPU／NPU 驅動由目標系統提供 |
| **Deb** | Debian 套件 | 依賴系統套件（透過 apt） |

---

## 自動化流程

### 觸發方式

1. **推送 Git Tag**（推薦用於正式發布）
   ```bash
   git tag v1.1.0
   git push origin v1.1.0
   ```

2. **手動觸發**（適合測試或 hotfix）
   - 前往 GitHub Actions → Release workflow
   - 點擊「Run workflow」
   - 填入版本號與目標平台

### 發佈前的檢查

- **版號必須一致**：tag（或手動輸入的版號）去掉 `v` 與 `-rc1` 之類的後綴後，必須等於 `CMakeLists.txt`、`pyproject.toml`、`python/ivit/__init__.py`、`include/ivit/ivit.hpp` 宣告的版號，否則 `prepare` 直接失敗。本機可用 `scripts/ci/check_version.sh v1.1.0` 預先確認。
- **CI 必須先通過**：Release 會先呼叫 `ci.yml`，tarball 與 deb 都等它全部成功才開始打包。

### 流程圖

```
┌─────────────┐
│  Push Tag   │  或  手動觸發
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Prepare   │  檢查版號、建立平台矩陣  (同時等待 CI 通過)
└──────┬──────┘
       │
       ├──────────────┐
       ▼              ▼
┌─────────────┐ ┌─────────────┐
│ Tarball x3  │ │   Deb x3    │  平行建置
└──────┬──────┘ └──────┬──────┘
       │              │
       └──────┬───────┘
              ▼
      ┌─────────────┐
      │   Publish   │  上傳至 GitHub Release（僅 tag）
      └──────┬──────┘
             ▼
      ┌─────────────┐
      │   Notify    │  回報結果
      └─────────────┘
```

---

## 手動建置

如需在本機手動建置，可使用以下腳本：

### 一鍵建置所有方案

```bash
# 建置單一平台的所有方案
./scripts/release.sh x86_64-nvidia

# 建置所有平台
./scripts/release.sh all

# 跳過特定方案
./scripts/release.sh x86_64-intel --skip-deb
```

### 個別建置

```bash
# Tarball
./scripts/build_tarball.sh x86_64-nvidia
# 產出: build-tarball-x86_64-nvidia/ivit-sdk-1.1.0-x86_64-nvidia.tar.xz

# Deb
./scripts/build_deb.sh x86_64-intel
# 產出: build-deb-x86_64-intel/ivit-sdk_1.1.0_amd64-intel.deb
```

### 驗證產出

```bash
# 驗證 tarball
./scripts/verify_package.sh tarball build-tarball-x86_64-nvidia/ivit-sdk-1.1.0-x86_64-nvidia.tar.xz

# 驗證 deb
./scripts/verify_package.sh deb build-deb-x86_64-intel/ivit-sdk_1.1.0_amd64-intel.deb
```

---

## 產出物說明

### Tarball 結構

```
ivit-sdk-1.1.0-x86_64-nvidia/
├── lib/
│   ├── libivit.so              # 核心函式庫
│   └── cmake/ivit/             # CMake 設定檔
├── include/ivit/               # C++ 標頭檔
├── deps/lib/                   # 打包的依賴庫 (TensorRT/OpenVINO)，每個函式庫一份實體檔，其餘名稱為 symlink
├── python/
│   └── ivit_sdk-1.1.0-*.whl    # Python wheel
├── bin/
│   └── setup_env.sh            # 環境設定腳本
├── examples/                   # 範例程式
└── INSTALL.md                  # 安裝說明
```

- 客戶專案可直接 `find_package(ivit 1.1)`（`CMAKE_PREFIX_PATH` 指向解壓目錄）：`ivitConfig.cmake`
  偵測到 `deps/lib` 內的 OpenVINO／TensorRT 時直接使用，不需在客戶機器上另裝後端或 CUDA toolkit。
- Tarball 以 xz 壓縮（`.tar.xz`）。TensorRT 11.4 的 NVIDIA 套件解開後 3.5 GB，其中依 GPU 架構拆分的
  `libnvinfer_builder_resource_sm*` 佔大半，SDK 現場建 TensorRT engine 需要它們。gzip 壓縮為 2.35 GB，
  超過 GitHub Release 單檔 2 GiB 上限；xz 為 1.41 GB。升級 TensorRT 時須重新確認大小。
  xz 壓縮在 20 核機器約 2 分鐘（總 CPU 約 38 分鐘），可用 `XZ_OPT` 調整。

### Deb 套件

- 安裝路徑：`/usr/lib`、`/usr/include`
- 自動宣告依賴：依建置時連結的版本產生，例如
  - Intel：`libopencv-dev (>= 4.0), libopenvino-2026.4.1, libopenvino-intel-cpu-plugin-2026.4.1, libopenvino-auto-plugin-2026.4.1, libopenvino-onnx-frontend-2026.4.1, libopenvino-ir-frontend-2026.4.1`；另以 Recommends 帶入 GPU／NPU 外掛（apt 預設會安裝）。OpenVINO 核心套件本身不含任何裝置外掛與模型讀取器，只依賴核心時無法推論。
  - NVIDIA：`libopencv-dev (>= 4.0), libnvinfer11, libnvonnxparsers11, cuda-cudart-12-9`
- NVIDIA deb 需先在目標機器加入 NVIDIA CUDA apt 來源（`cuda-keyring`），apt 才找得到上述套件。
- 檔名格式：`ivit-sdk_<version>_<arch>-<backend>.deb`

---

## CI/CD 設定

### 必要的 Secrets

| Secret | 說明 |
|--------|------|
| `GITHUB_TOKEN` | 自動提供，用於建立 GitHub Release |

### Self-hosted Runner（aarch64）

x86 平台使用 GitHub 提供的 `ubuntu-24.04` runner：

- Intel：Intel apt 來源 `https://apt.repos.intel.com/openvino ubuntu24`（不分年份；`openvino/2025` 來源只到 2025.4.1），OpenVINO 2026.4.1。此來源沒有整合套件，`release.yml` 逐一安裝 `libopenvino-dev-2026.4.1` 與 CPU／GPU／NPU／AUTO／HETERO／AUTO-BATCH 外掛。
- NVIDIA：NVIDIA apt 來源 `cuda/repos/ubuntu2404`（`cuda-keyring`），CUDA 12.9 與 TensorRT
  `11.4.0.106-1+cuda12.9`。**TensorRT 版本必須鎖定**：不指定版本時 `apt install libnvinfer-dev`
  會裝到來源中最新的版本，大版本改變時 `libnvinfer` 的 soname 也隨之改變。升級時請同時修改
  `release.yml` 兩個工作的 `CUDA_VERSION`／`TRT_VERSION`，以及 CI 用的
  `scripts/ci/build_and_test.sh` 中的 `TRT_VERSION`／`TRT_TAG`／`ONNX_PARSER_BRANCH`。

Jetson 平台需要 ARM64 runner，**預設不會建置**：推送 tag 或以 `platforms=all` 手動觸發時，只有設定了 repository variable `IVIT_JETSON_RUNNER`（任意非空值）才會包含 `aarch64-jetson`。沒有 runner 時該工作會一直排隊直到逾時，而發布需要所有建置工作都成功，會連 x86 的發布一起擋住。手動觸發時也可以直接指定 `platforms=aarch64-jetson`。發布說明只列出實際建出的平台，檔名使用 CMake 的版本（預發行 tag 如 `v1.1.0-rc1` 的檔名仍是 `1.1.0`）。

1. 在 Jetson 設備上安裝 GitHub Actions Runner
2. 加入 label：`self-hosted-arm64`，並設定 repository variable `IVIT_JETSON_RUNNER`
3. 確保已安裝 JetPack（CUDA 於 `/usr/local/cuda`、TensorRT）。工作開始時檢查 `nvcc` 與 TensorRT 標頭，缺少即失敗；OpenCV、CMake 由工作以 `sudo apt-get` 安裝，runner 帳號需可免密碼 sudo。

---

## 版本管理

### 版本號規則

遵循 [Semantic Versioning](https://semver.org/)：

- `MAJOR.MINOR.PATCH` (e.g., `1.1.0`)
- 預發布版：`1.1.0-rc1`、`1.1.0-beta`

### 發布檢查清單

1. [ ] 更新版本號：`CMakeLists.txt`、`pyproject.toml`、`python/ivit/__init__.py`（`__version__`）、`include/ivit/ivit.hpp`（`IVIT_VERSION_STRING`）
2. [ ] 執行 `scripts/ci/check_version.sh vX.Y.Z` 確認四處一致
3. [ ] 更新 CHANGELOG
4. [ ] 確保所有測試通過
5. [ ] 建立並推送 tag

```bash
# 更新版本後
git add CMakeLists.txt pyproject.toml python/ivit/__init__.py include/ivit/ivit.hpp CHANGELOG.md
git commit -m "chore: bump version to 1.1.0"
git tag v1.1.0
git push origin main --tags
```

---

## 故障排除

### Tarball 建置失敗

**症狀**：CMake 找不到 OpenVINO/TensorRT

**解決**：
```bash
# Intel 平台：apt 安裝時確認已裝 libopenvino-dev-2026.4.1；使用 OpenVINO archive 時
source <openvino 安裝目錄>/setupvars.sh

# NVIDIA 平台（apt 安裝的 CUDA 不在 PATH 上）
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDAToolkit_ROOT=/usr/local/cuda-12.9
# TensorRT 若不是 apt 安裝：
export TENSORRT_ROOT=/usr/local/tensorrt
```

### Deb 安裝後找不到 library

**症狀**：`error while loading shared libraries: libivit.so`

**解決**：
```bash
sudo ldconfig
```

---

## 相關檔案

| 檔案 | 用途 |
|------|------|
| `.github/workflows/release.yml` | CI/CD 工作流程定義 |
| `scripts/build_tarball.sh` | Tarball 建置腳本 |
| `scripts/build_deb.sh` | Deb 建置腳本 |
| `scripts/release.sh` | 整合建置腳本 |
| `scripts/verify_package.sh` | 驗證腳本 |
| `packaging/INSTALL.md` | 客戶安裝說明 |
