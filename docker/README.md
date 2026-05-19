# iVIT-SDK Docker 驗證環境

本目錄包含用於驗證 SDK 完整性的 Docker 配置，讓您可以在乾淨環境中測試安裝、編譯和執行。

## 快速開始

### 方法一：使用 Docker Compose（推薦）

```bash
cd ivit-sdk

# 完整驗證（自動執行所有步驟）
docker compose -f docker/docker-compose.yml up verify

# 進入互動模式（手動操作）
docker compose -f docker/docker-compose.yml run --rm dev bash

# GPU 環境驗證（需要 NVIDIA Docker）
docker compose -f docker/docker-compose.yml up verify-gpu

# 只驗證 Python
docker compose -f docker/docker-compose.yml up verify-python

# 只驗證 C++
docker compose -f docker/docker-compose.yml up verify-cpp
```

### 方法二：直接使用 Docker

```bash
cd ivit-sdk

# 1. 建置映像
docker build -t ivit-verify -f docker/Dockerfile.verify .

# 2. 執行完整驗證
docker run --rm -v $(pwd):/workspace/ivit-sdk ivit-verify verify-sdk.sh

# 3. 進入互動模式
docker run -it --rm -v $(pwd):/workspace/ivit-sdk ivit-verify bash
```

## 可用的 Docker 映像

| Dockerfile | 用途 | 基礎映像 | GPU |
|------------|------|----------|-----|
| `Dockerfile.verify` | **完整驗證環境** | Ubuntu 22.04 | 否 |
| `Dockerfile.build-test` | 基本編譯測試 | Ubuntu 22.04 | 否 |
| `Dockerfile.full` | 全後端環境 | TensorRT 24.01 | 是 |

## 驗證腳本選項

```bash
# 完整驗證
verify-sdk.sh

# 只驗證 Python
verify-sdk.sh --python-only

# 只驗證 C++
verify-sdk.sh --cpp-only

# 跳過測試
verify-sdk.sh --skip-tests

# 顯示幫助
verify-sdk.sh --help
```

## 驗證流程說明

驗證腳本會執行以下步驟：

### Python 驗證
1. **安裝 Python 套件** - `pip install .`
2. **執行 Python 測試** - `pytest tests/integration/`
3. **執行 Python 範例** - 物件偵測範例

### C++ 驗證
4. **下載依賴** - `./scripts/download_deps.sh`
5. **CMake 配置與編譯** - `cmake .. && make`
6. **執行 C++ 測試** - `ctest`
7. **執行 C++ 範例** - simple_inference

## 互動模式操作指南

進入容器後，可以手動執行以下操作：

### Python 套件安裝與測試

```bash
# 安裝 Python 套件
cd /workspace/ivit-sdk
pip install .

# 驗證安裝
python3 -c "import ivit; print(ivit.__version__); ivit.list_devices()"

# 執行測試
pytest tests/integration/ -v

# 執行範例
python3 examples/python/01_quickstart.py
```

### C++ 編譯與測試

```bash
# 複製到可寫目錄
cp -r /workspace/ivit-sdk /tmp/ivit-sdk
cd /tmp/ivit-sdk

# 下載依賴
./scripts/download_deps.sh

# 編譯
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DIVIT_USE_OPENVINO=ON \
    -DIVIT_USE_TENSORRT=ON \
    -DIVIT_BUILD_EXAMPLES=ON \
    -DIVIT_BUNDLE_DEPS=ON

make -j$(nproc)

# 執行範例
./bin/simple_inference devices
./bin/simple_inference detect \
    /opt/test-data/models/yolox-s.onnx \
    /opt/test-data/images/bus.jpg \
    cpu \
    output.jpg
```

## 測試資料

容器內預裝了測試資料：

```
/opt/test-data/
├── images/
│   ├── bus.jpg        # 公車圖片（物件偵測）
│   └── zidane.jpg     # 人物圖片（物件偵測）
└── models/
    └── yolox-s.onnx   # YOLOX-S 模型
```

## GPU 環境驗證

需要安裝 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)。

```bash
# 建置 GPU 映像
docker build -t ivit-full -f docker/Dockerfile.full .

# 執行驗證
docker run --rm --gpus all \
    -v $(pwd):/workspace/ivit-sdk \
    ivit-full verify-sdk.sh

# 互動模式
docker run -it --rm --gpus all \
    -v $(pwd):/workspace/ivit-sdk \
    ivit-full bash
```

## 常見問題

### Q: Python 測試失敗？

確認 SDK 原始碼已正確掛載：

```bash
docker run -v $(pwd):/workspace/ivit-sdk ...
```

### Q: 如何查看輸出檔案？

驗證輸出儲存在 `/tmp/ivit-output/`：

```bash
ls -la /tmp/ivit-output/
```

使用 Docker Compose 時，可以用 volume 保存：

```bash
docker compose -f docker/docker-compose.yml run --rm \
    -v $(pwd)/output:/tmp/ivit-output \
    dev verify-sdk.sh
```

### Q: 如何使用其他後端？

在 Dockerfile.full 環境中可以啟用所有後端：

```bash
cmake .. \
    -DIVIT_USE_OPENVINO=ON \
    -DIVIT_USE_TENSORRT=ON \
    ...
```

## 相關文件

- [Getting Started](../docs/getting-started.md)
- [User Guide](../docs/user-guide.md)
- [API Specification](../docs/api/api-spec.md)
