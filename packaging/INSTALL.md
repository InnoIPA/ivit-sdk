# Binary SDK Installation

此文件只說明已建置的 platform tarball。原始碼安裝、完整 API 與硬體限制請參閱 repository 的 `docs/sdk-manual.zh-TW.md`。

## 環境需求

使用與產物相符的 Linux／CPU 架構，並安裝 OpenCV 開發套件（Intel 版的 OpenVINO 另需 `libtbb12`，安裝 OpenCV 時會一併帶入）。後端函式庫（OpenVINO，或 TensorRT／CUDA runtime）已放在 `deps/lib`；NVIDIA 版另需 GPU 驅動，Intel GPU／NPU 需相應驅動。套件存在不代表目標硬體已完成驗收。

## 安裝

```bash
tar -xf ivit-sdk-<version>-<platform>.tar.xz
cd ivit-sdk-<version>-<platform>
source bin/setup_env.sh
```

## C／C++

套件提供 `include/ivit/`、`lib/libivit.so` 及 `lib/cmake/ivit/`。C++ 客戶專案使用：

```cmake
find_package(ivit 1.1 CONFIG REQUIRED)
target_link_libraries(app PRIVATE ivit::ivit)
```

Configure 時加入 `-DCMAKE_PREFIX_PATH=/absolute/path/to/sdk`。只需另外安裝 OpenCV 開發套件；後端函式庫使用 `deps/lib` 內的版本。C 整合使用 `ivit/c_api.h`。

## Python

先確認 `python/` 內的 wheel 與你的 Python 版本和平台相符（打包時 wheel 建置失敗會讓整個打包失敗，tarball 一定含有 wheel；它是用打包機器的 Python 建置的，版本不同時須自行建置）。

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install python/ivit_sdk-*.whl
python3 -c "import ivit; assert ivit.is_cpp_available(); print(ivit.list_devices())"
```

原生 extension 不等於完整 Python 套件；單獨設定 PYTHONPATH 無法取代 wheel 安裝。推論模型與 labels 另行交付，TensorRT engine 應在目標平台產生及驗證。

## 函式庫載入

找不到函式庫時，確認已執行 `setup_env.sh`，並檢查 `lib/`、`deps/lib/` 與 vendor runtime。GPU／NPU 未出現時，檢查裝置權限及相符驅動。
