# iVIT-SDK

Innodisk Vision Intelligence Toolkit：以 C++17 為核心的電腦視覺 SDK，提供 Python、C++、C ABI 與 C# .NET 8 整合方式。影像推論支援分類、物件偵測與語意分割；Python 另提供模型下載及訓練工具。

## 安裝與確認

這是來源專案，未包含已安裝的環境或預先建置的 SDK。請在專案根目錄執行；Ubuntu／Debian 的 OpenVINO 版本使用：

```bash
bash scripts/install_sdk.sh
./.venv/bin/python3 -c 'import ivit; assert ivit.is_cpp_available(); print(ivit.list_devices())'
```

安裝腳本會建立 `.venv/`、安裝系統開發套件及建置 Python bindings；需要網路與系統套件安裝權限。之後可用 `source .venv/bin/activate` 啟用環境。驗證命令只檢查 bindings 與裝置，不需要模型。

- 純 C++ 建置：`bash scripts/install_sdk.sh --mode cpp`。
- 只查看選項：`bash scripts/install_sdk.sh --help`。
- 已準備好依賴時，可用 `--check` 檢查；此選項不會安裝缺少的套件。
- 此腳本只安裝 OpenVINO 版；TensorRT 版的建置見手冊 3.4／3.5。Intel GPU／NPU 與 NVIDIA 驅動另行準備。

推論須自行提供相容模型、影像及模型設定；本專案不附 `classifier.onnx`。實際推論、C/C++ 及 C# 整合請依 [SDK 使用手冊](docs/sdk-manual.zh-TW.md) 操作。二進位 tarball 的安裝方式見 [INSTALL](packaging/INSTALL.md)。

## 支援與驗證範圍

OpenVINO CPU／GPU 與 TensorRT FP32／FP16 有本機驗證紀錄；OpenVINO NPU 有獨立設備紀錄。支援的後端只有 OpenVINO 與 TensorRT。完整 YOLOX 訓練及其他平台的交付狀態見手冊的[目前限制](docs/sdk-manual.zh-TW.md#10-目前限制與原始碼索引)與 [CHANGELOG](CHANGELOG.md)。

## 專案第一層結構

```text
.
├── include/        # C/C++ 公開標頭
├── src/            # C++ 核心、後端與共同前後處理
├── python/         # Python 套件與 pybind11 入口
├── bindings/       # C# 等額外語言 wrapper
├── examples/       # 各語言範例與樣本影像
├── tests/          # Python、C++ 與外部整合測試
├── docs/           # 完整手冊、擴充說明與驗證紀錄
├── cmake/          # CMake package 設定模板
├── scripts/        # 安裝、建置、驗證及來源封裝命令
├── packaging/      # 二進位套件安裝說明
└── .github/        # GitHub Actions workflow
```

| 根目錄檔案 | 職責 |
|---|---|
| `CMakeLists.txt` | 原生 SDK、bindings、範例及測試建置 |
| `pyproject.toml` | Python 依賴、wheel／sdist 與開發工具設定 |
| `README.md` | 專案入口與無模型的安裝驗證 |
| `AGENTS.md` | Contributor guide |
| `LICENSE` | Apache-2.0 授權 |
| `.gitignore` | 排除本機環境與生成產物 |

`.venv/`、`build/` 是安裝／開發時產生的目錄；`models/` 是自行放置模型的慣例目錄（Zoo 快取預設在 `~/.cache/ivit/models`）。這些都不包含在乾淨來源中。`python/` 與 `bindings/` 的位置由目前建置流程使用；兩者角色不同。

## 文件與上版

完整 API 與架構統一維護於 [SDK 使用手冊](docs/sdk-manual.zh-TW.md)；專用文件由 [文件索引](docs/README.md) 導覽。

```bash
python3 scripts/package_source.py --output /tmp/ivit-source-release
```

輸出來源壓縮檔與 SHA256 manifest，目標目錄必須在來源樹外，且不能含同名產物。平台二進位交付流程見 [release automation](docs/deployment/release-automation.md)。
