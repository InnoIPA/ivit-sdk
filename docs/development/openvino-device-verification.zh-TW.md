# OpenVINO GPU／NPU 驗證紀錄

日期：2026-10-04。使用獨立 venv `/tmp/ivit-manual-check-20261002`，Python 3.12.3、OpenVINO 2026.4.1、SDK 1.0.0。

## 狀態

| 裝置 | 結果 | 實際執行硬體 |
|---|---|---|
| CPU 基準 | 通過 | Intel Core i5-13500 |
| GPU 靜態 batch=1 模型 | 通過 | Intel UHD Graphics 770，OpenVINO GPU.0 |
| GPU 動態 batch 模型 | 通過 | 同上；實際 batch=8 推論並比對逐張結果 |
| 遠端 GPU 靜態模型 | 通過 | Core Ultra 7 255U Intel Graphics，OpenVINO GPU.0 |
| 遠端 NPU 靜態模型 | 通過 | Intel AI Boost，架構 3720，OpenVINO NPU |
| 遠端 NPU 無上限動態 batch | 不支援；SDK 安全拒絕測試通過 | 原生編譯曾 exit 139，修正後 SDK 回報 ModelLoadError |
| TensorRT | FP32／FP16 靜態／動態分類通過 | RTX 2070 SUPER；見 [TensorRT 紀錄](tensorrt-device-verification.zh-TW.md) |

本機沒有 NPU，NPU 驗證於 2026-10-05 改用遠端主機完成。不將 NVIDIA OpenCL 裝置當作 Intel GPU 驗證。

## GPU 環境修復

原先 `/etc/OpenCL/vendors` 只有 NVIDIA ICD，OpenVINO 的 GPU 指向 RTX 2070 SUPER；SDK 會過濾該裝置，因此只列 CPU。

經使用者授權，安裝 Ubuntu 的 intel-opencl-icd 及其相依 runtime，並授予目前帳號 Intel `/dev/dri/renderD128` 的讀寫 ACL。之後 OpenVINO 列出：

- GPU.0：Intel UHD Graphics 770。
- GPU.1：NVIDIA RTX 2070 SUPER。
- SDK：cpu、gpu:0（OpenVINO）。

未更改 NVIDIA 驅動，未安裝 NPU 驅動，未啟動 TensorRT 驗證。ACL 用於本次測試，裝置節點重新建立後可能需要再次設定；正式部署應由管理員設定 render 群組／udev 規則。

Intel GPU 使用 OpenCL runtime，詳見 [OpenVINO GPU 設定](https://docs.openvino.ai/nightly/get-started/install-openvino/configurations/configurations-intel-gpu.html)。NPU 的支援範圍及驅動要求見 [OpenVINO 系統需求](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/system-requirements.html)。

## 測試內容

`scripts/verify_openvino.py` 產生可重現的小型 Conv／GlobalAveragePool／Flatten ONNX 分類模型，明確指定後端與裝置，沒有 AUTO fallback。分類 confidence 門檻設為 0，確保比對完整的四類 softmax 分數。

每組通過以下檢查：

1. OpenVINO native compile/infer，檢查 EXECUTION_DEVICES。
2. SDK bindings 載入、暖機、八張影像單張推論。
3. GPU SDK 結果與 CPU SDK 及 native OpenVINO softmax 結果比對。
4. SDK predict_batch、predict_concurrent；驗證回傳數量及順序。
5. 負 stride NumPy 切片與連續副本結果一致。
6. 兩個 Python worker 共用同一 SDK 模型，結果一致。

容許誤差 rtol=0.01、atol=0.01；允許不同後端預設精度差異。這是功能及一致性驗證，不是特定客戶模型的準確度驗收，也不是效能 benchmark。尚未涵蓋偵測／分割模型、INT8 校正或大量長時間壓力測試。

## 重現命令

```bash
source /tmp/ivit-manual-check-20261002/bin/activate
python3 scripts/verify_openvino.py --device cpu --report build/verification/openvino-cpu.json
python3 scripts/verify_openvino.py --device gpu:0 --report build/verification/openvino-gpu.json
python3 scripts/verify_openvino.py --device gpu:0 --dynamic-batch --report build/verification/openvino-gpu-dynamic.json
python3 scripts/verify_openvino.py --device npu --report build/verification/openvino-npu.json
```

本機 NPU 命令以非零 exit code 返回「Requested NPU is not enumerated」；遠端靜態模型通過。JSON 在忽略的 build/ 內，不作交付資產。

GPU runtime 安裝後重跑 Python P0 回歸測試：26 passed，沒有 skip。

## 遠端 NPU 主機登入檢查（2026-10-05）

使用者提供第一台遠端測試主機與帳號（主機位址與帳號不記錄於文件）。SSH 連接埠 22 可達，金鑰交換成功，伺服器提供 publickey／password 驗證。

互動登入、遠端命令與停用 publickey 的 password-only 診斷，共三次嘗試皆在送出密碼後被伺服器關閉連線（`Connection closed by <host> port 22`），exit code 255。診斷輸出未出現登入驗證成功訊息；尚無法判定是帳號、密碼或伺服器登入政策問題。

未執行遠端命令，未安裝套件，也尚未確認該主機的 NPU 硬體與驅動。需確認帳號可經 SSH 登入，或由主機管理員檢查 sshd／驗證日誌後再繼續。密碼不記錄於文件或腳本。TensorRT 驗證仍等待 OpenVINO NPU 驗證完成。

## 遠端 NPU 驗證完成（2026-10-05）

改用另一台遠端測試主機後登入成功。Ubuntu 24.04.5、Core Ultra 7 255U、Intel NPU PCI `8086:7d1d`、架構 3720；`/dev/accel/accel0` 存在，帳號具有 render 權限。NPU driver/compiler 套件為 `1.38.0.20260910-34487311128~ubuntu24.04`。

於 `/tmp/ivit-npu-check-20261005/venv` 建立 Python 3.12.3／OpenVINO 2026.4.1 隔離環境；補裝 pkg-config、libopencv-dev 後建置 editable SDK。bindings 載入成功、pip check 通過。

靜態 NPU 與遠端 Intel GPU 均通過上述八項檢查；NPU 的 EXECUTION_DEVICES 確認為 NPU。靜態 batch=1 模型的 predict_batch 驗證八張影像回傳一致性，不代表 NPU 支援動態 batch=8。

發現並修正：

- NPU 的 EXECUTION_DEVICES 屬性在此版本回傳字串；腳本需正規化為單項清單。
- 無上限動態 batch 在原生 NPU 編譯階段發生 segmentation fault（exit 139）。SDK 改在 compile 前拒絕動態 rank／無上限維度，回報可處理的 ModelLoadError；驗證腳本也拒絕此測試設定。
- 新增隔離子程序回歸測試，確認無上限 NPU 模型被拒絕後，仍可載入並推論靜態模型。遠端 P0：27 passed；API／core：49 passed、3 個既有 VideoSource skip。本機修正後 CTest：2/2 passed。

[OpenVINO NPU 文件](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html#dynamic-shapes)將動態 shape 列為預覽功能，需 NPU40XX 以上、受支援驅動與 Compiler-In-Plugin，且維度必須有上限。本次 3720 主機未驗證 bounded dynamic shape，客戶模型應使用靜態輸入並另做準確度驗收。
