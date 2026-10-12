# TensorRT 裝置驗證紀錄

日期：2026-10-05。先完成 OpenVINO Intel GPU／NPU，再執行本機 TensorRT 驗證。

## 環境與結果

- NVIDIA GeForce RTX 2070 SUPER，8 GiB，compute capability 7.5。
- NVIDIA driver 610.57.04、TensorRT 10.12.0.36、CUDA runtime／headers 12.9.79、GCC 13.3。
- Python 3.12.3；venv `/tmp/ivit-manual-check-20261002`；OpenVINO 2026.4.1 作 CPU 參考。
- NVIDIA 官方 Ubuntu 24.04 套件下載後核對套件索引 SHA256，僅解壓至 `/tmp/ivit-trt-deps-20261005/root`，沒有安裝系統 CUDA／TensorRT 或更換驅動。

| 模型 | FP32 | FP16 |
|---|---|---|
| 靜態 batch=1 | 通過 | 通過 |
| 動態 batch，profile 1–8 | 通過 | 通過 |

這是小型 Conv／GlobalAveragePool／Flatten 分類模型的功能驗證，不是客戶模型準確度驗收或效能 benchmark。FP16 指 builder precision 設定，不宣稱所有 layer 均使用 FP16。未涵蓋 INT8 校正、偵測／分割、多 GPU、其他 TensorRT 版本或長時間壓力測試。

## 實測內容與修正

`scripts/verify_tensorrt.py` 明確指定 TensorRT／cuda:0，檢查 bindings、模型 backend／device，並以 OpenVINO CPU 的完整四類 softmax 分數作參考；rtol／atol 均為 0.01。

四組均通過暖機、單張、batch、concurrent、負 stride NumPy 輸入、兩個 Python worker 共用模型、不同 batch 長度、三次 unload／reload、保留快取模型時的子程序退出、序列化 engine 重新載入。動態模型另驗證 batch=9 被拒絕後可繼續 batch=1 推論。

實測發現 `LoadConfig.use_cache=False` 未傳至 RuntimeConfig，仍產生 engine 檔；已補上傳遞，腳本檢查停用快取時沒有 engine 輸出。TensorRT 版本資訊改用官方 major／minor／patch 常數，避免錯誤解析新版 NV_TENSORRT_VERSION。

Python 回歸：74 passed、5 skipped（本機無 NPU、TensorRT 可用所以略過「後端不存在」測試，以及三個既有 VideoSource skip）。pip check 通過。含 OpenVINO／TensorRT 的 C++ SDK 建置成功，CTest 2/2 passed。

## 重現

原生建置需 TensorRT headers、nvinfer／nvonnxparser libraries 與 CUDA toolkit。一般安裝參考 [NVIDIA tar 安裝說明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/installing-tensorrt/install-tar.html)；版本相容性參考 [官方支援矩陣](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/support-matrix.html)。僅安裝 Python TensorRT wheel 不足以保證 SDK 的 C++ 後端可建置。

本次解壓環境須明確指定 CMake 的 CUDAToolkit_ROOT、TENSORRT_INCLUDE_DIR、TENSORRT_LIBRARY、NVONNXPARSER_LIBRARY，並啟用 IVIT_USE_TENSORRT／IVIT_USE_OPENVINO。執行時：

```bash
source /tmp/ivit-manual-check-20261002/bin/activate
export LD_LIBRARY_PATH=/tmp/ivit-trt-deps-20261005/root/usr/lib/x86_64-linux-gnu:/tmp/ivit-trt-deps-20261005/root/usr/local/cuda-12.9/lib64
python3 scripts/verify_tensorrt.py --report build/verification/tensorrt-fp32-static.json
python3 scripts/verify_tensorrt.py --dynamic-batch --report build/verification/tensorrt-fp32-dynamic.json
python3 scripts/verify_tensorrt.py --precision fp16 --report build/verification/tensorrt-fp16-static.json
python3 scripts/verify_tensorrt.py --precision fp16 --dynamic-batch --report build/verification/tensorrt-fp16-dynamic.json
```

JSON 與詳細日誌位於忽略的 build/verification。反覆釋放及程序退出測試通過，不代表既有 shutdown 特殊處理、跨 GPU 資源釋放與 engine 快取失效策略均已修正；這些仍是後續整合工作。
