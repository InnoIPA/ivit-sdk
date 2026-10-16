# 人工 Checklist 完整實跑結果（2026-10-15～16）

> 後續更新：第 9／12 步的原問題已修正並通過正常 GPU 驗收，但無可見 GPU 的補測發現初始化崩潰，詳見 [修正後複驗](manual-validation-followup-20261016.zh-TW.md)。以下保留原次實跑結果。

> **後續修正（2026-10-16）**：`tests/cpp/test_inference.cpp` 的 `DeviceManagerListDevices` 已依編入的後端判斷（OpenVINO 建置要求 CPU；純 TensorRT 建置要求沒有 CPU 且所有裝置皆屬 TensorRT）；`python/ivit/zoo/registry.py` 的 `_SegWrapper` 已補 `.eval()`；清單補上 .NET 前置條件並填寫簽核表。重建後 OpenVINO／TensorRT-only 兩份 CTest 均 3/3 通過。下文各節保留修正前的原始觀察。分割匯出 warning 的修正尚未重跑真實匯出確認。

依據：[人工驗證執行清單](manual-validation-checklist.zh-TW.md)。安裝於 10 月 15 日完成，服務中斷後於 10 月 16 日確認既有日誌並接續其餘步驟。本次沒有修改 SDK 實作或失敗測試來取得通過結果。

## 結論

四個語言／後端組合的三種任務均能執行，TensorRT 另完成 FP32／FP16；OpenVINO CPU／Intel GPU、外部 CMake、兩後端 C ABI／C# 整合均有實際結果。

照原清單操作，確認兩個會卡住的地方：

1. **第 9 步：純 TensorRT 的 CTest 失敗。** 測試寫死必須有 CPU，但這份建置只啟用 TensorRT；不是本次三任務推論失敗。
2. **第 12 步：`dotnet` 不在 PATH。** 字面命令回傳 127；改用現有 .NET 8 SDK 完整路徑後，OpenVINO／TensorRT C# 檢查均通過。

未執行全新的 NVIDIA 系統套件安裝；第 4 步選用清單的路徑 B。不能把以下結果當成 `--install-nvidia` 全新安裝認證。

## 環境與隔離

- Repository：`/home/ipa-sw2-d1/Workspace/ivit-sdk`，含未提交修改的工作樹。commit、初始差異與狀態存於本次 log；不是固定 commit 的發行認證。
- 獨立產物：`/tmp/ivit-checklist-20261015/`，未覆寫使用者的 `/tmp/ivit-manual-20261014-230219/`。
- Python 3.12；OpenVINO 2026.4.1；系統 CUDA 12.9；**實際載入 `/usr/lib/x86_64-linux-gnu/libnvinfer.so.11.4.0`**。
- i5-13500、Intel UHD 770、NVIDIA GeForce RTX 2070 SUPER。
- PyTorch 2.14.1+cpu、torchvision 0.29.1+cpu；.NET SDK 8.0.425。
- Python：全新 `py-ov` 與 `py-trt`；C++：獨立 OpenVINO-only／TensorRT-only build、install prefix 與 customer build。
- 模型 cache 與 `TORCH_HOME` 均指向本次目錄；真實下載 YOLOX-S 與 torchvision 權重，首次匯出 ResNet18／DeepLabV3-MobileNetV3。沒有用舊 ONNX mirror 取代第 5 步網路驗證。

## 逐步結果

| 步驟 | 結果 | 實際執行與限制 |
| --- | --- | --- |
| 1. 記錄驗證對象 | 完成 | 保存 commit、工作樹狀態與 tracked diff；依舊是工作樹驗證 |
| 2. 主機／依賴／驅動 | 通過前置檢查 | 系統套件已具備，以 dpkg、工具版本、nvidia-smi 與 `--check` 核對；未再次執行 sudo apt 安裝 |
| 3. Python × OpenVINO 安裝 | 通過 | 全新 venv；原生綁定、CPU／Intel GPU、沒有 TensorRT 裝置；pip check 通過 |
| 4. Python × TensorRT 安裝 | 路徑 B 通過 | 使用現有系統 CUDA／TensorRT，兩個後端可用；核對真正載入 11.4.0；未執行路徑 A 的 sudo 自動安裝 |
| 5. 模型下載／首次匯出 | 通過，有 warning | 下載三模型所需資產，輸出 ONNX、labels 與 options；分割 wrapper 匯出時有 training-mode warning，見下文 |
| 6. Python 現有範例 | 9/9 通過 | CPU、Intel GPU、TensorRT 各跑分類／偵測／分割；不是僅用合成模型 |
| 7. Python 明確後端矩陣 | 通過 | OpenVINO CPU／Intel GPU、TensorRT FP32／FP16，三任務均通過 helper 斷言並輸出結果 |
| 8. 後端參考比對 | 7/7 通過 | OpenVINO CPU、GPU static／dynamic；TensorRT FP32／FP16 × static／dynamic 四組全部通過 |
| 9. C++ 建置與 CTest | **部分失敗** | 兩份 build／install／裝置列舉成功；OpenVINO CTest 3/3 通過；TensorRT-only 2/3 通過，`test_inference` 失敗 |
| 10. C++ 範例／外部 CMake | 通過 | CPU、Intel GPU、TensorRT FP32／FP16 三任務均執行；原有分類／偵測 demo 在 CPU／CUDA 執行成功 |
| 11. 結果比對 | 完成量測 | 分類 top-1 均一致、偵測均五個物件；FP16 有小幅數值／mask 差異，詳見量測表，非資料集精度認證 |
| 12. C／C# 整合 | **原命令有環境缺口；調整後通過** | `dotnet --version` 找不到程式；完整 SDK 路徑可 build/run；OpenVINO／TensorRT 的 C ABI 與 C# 均通過 |
| 13. 測試／靜態檢查 | 通過 | OpenVINO Python 359 passed／10 skipped；TensorRT Python 366 passed／3 skipped；Flake8、shell syntax、diff check 通過 |

第 13 步另驗證空測試目錄的 `ctest --no-tests=error` 回傳 8，這是防護成功的預期非零，不應與第 9 步的真實測試失敗混為一談。

## 第 9 步的確定失敗：測試不支援 TensorRT-only 建置

失敗命令：

```bash
ctest --test-dir /tmp/ivit-checklist-20261015/cpp-trt \
  --output-on-failure --no-tests=error
```

退出碼：`8`。`test_inference` 的八個 GoogleTest case 中，七個通過，失敗項為 `InferenceTest.DeviceManagerListDevices`：

```text
tests/cpp/test_inference.cpp:49
Value of: has_cpu
  Actual: false
Expected: true
```

原因：`tests/cpp/test_inference.cpp` 的 `DeviceManagerListDevices` 無條件要求 CPU，但清單明確設定：

```text
IVIT_USE_OPENVINO=OFF
IVIT_USE_TENSORRT=ON
```

這份建置的裝置列舉為可用的 `cuda:0`，沒有 CPU 是符合後端選擇的結果。後續 TensorRT-only 三任務、C consumer 及 C# 推論均成功，支持此失敗來自測試假設。

**建議修正**：測試依已編入的後端及可用硬體驗證裝置；只有提供 CPU runtime 時要求 CPU。沒有 GPU 的 TensorRT-only 環境也要明確定義「可編譯、裝置不可用」的測試預期。不能只刪除斷言，也不應為了讓此測試過而悄悄把 OpenVINO 打開，掩蓋 TensorRT-only 覆蓋缺口。

本次保留原測試，未修正後重跑，因此第 9 步不能整體標示通過。

## 第 12 步的環境缺口：.NET SDK 未在 PATH

原清單的字面命令 `dotnet --version` 在執行環境找不到程式，退出碼 `127`。SDK 本機存在於：

```text
/tmp/ivit-dotnet-20261005/dotnet
```

以該完整路徑取得版本 8.0.425、build 成功，兩個後端各自使用正確的 `LD_LIBRARY_PATH` 後均顯示：

```text
C# lifecycle/buffer checks passed
```

建議清單在第 2 或第 12 步先列 .NET 8 SDK 的安裝／PATH 前置條件；有自訂 SDK 路徑時提供以下擇一方式：

```bash
# 將實際 SDK 目錄加入 PATH（此路徑只是本機例子）。
export PATH="/tmp/ivit-dotnet-20261005:$PATH"
dotnet --version

# 或所有 dotnet 命令使用完整路徑。
/tmp/ivit-dotnet-20261005/dotnet --version
```

此為整合環境與清單引導的缺口；本次沒有觀察到 C# wrapper 功能失敗。

## 非阻塞警告與人工操作注意

- 第 5 步分割 ONNX 匯出有 `Exporting a model while it is in training mode` warning。原始模型已呼叫 `raw_model.eval()`，但 `_SegWrapper(raw_model)` 未另外呼叫 `.eval()`，wrapper 自身仍是 training 狀態。可改善為對 wrapper 也設定 eval；不能僅依這個 warning 認定本次分割輸出錯誤。原始碼位置為 `python/ivit/zoo/registry.py` 的 torchvision segmentation export 分支。
- 使用 venv 的 CMake 4 時外部專案 configure 有 developer warning，但本次 configure／build 均成功。驗收時區分 warning 與非零失敗。
- 原 C++ 分類／偵測 demo 在桌面環境可能等待 GUI 按鍵。本次使用 `env -u DISPLAY -u WAYLAND_DISPLAY` 執行其無視窗流程，避免把等待按鍵誤判為卡死。
- 本機已有 NVIDIA 套件，且前置依賴已安裝。全新的 apt 安裝、全新 NVIDIA 自動安裝仍沒有被本次執行覆蓋。

## 四組合與跨後端結果

測試資產：同一份 ResNet18、YOLOX-S、DeepLabV3-MobileNetV3 ONNX 與 `examples/data/images/bus.jpg`，明確配置 ImageNet／YOLOX 前處理契約。C++ 與 Python 的輸出採用獨立 engine build，不假設 FP16 engine 逐位元一致。

| 比對 | 分類 top-1 | 最大分類分數差 | 偵測數 | 最小配對 box IoU | 分割 class mask 一致率 |
| --- | --- | --- | --- | --- | --- |
| Python／C++ OpenVINO CPU | 相同 | 0 | 5／5 | 1 | 100% |
| Python／C++ OpenVINO Intel GPU | 相同 | 0 | 5／5 | 1 | 100% |
| Python／C++ TensorRT FP32 | 相同 | 5.9e-8 | 5／5 | 0.9999993 | 100% |
| Python／C++ TensorRT FP16 | 相同 | 0 | 5／5 | 1 | 99.97485% |
| Python CPU／Intel GPU | 相同 | 2.98e-7 | 5／5 | 0.9999986 | 100% |
| Python CPU／TensorRT FP32 | 相同 | 8.34e-7 | 5／5 | 0.9999994 | 100% |
| Python CPU／TensorRT FP16 | 相同 | 0.000384926 | 5／5 | 0.9962403 | 99.96536% |
| Python TensorRT FP32／FP16 | 相同 | 0.000384092 | 5／5 | 0.9962401 | 99.96536% |

分類 top-1 為 `minibus`，偵測為一台 bus 與四個 person。另直接檢視 TensorRT FP16 偵測及分割視覺化，框與區域有對應原圖中的車輛與人物。這是單張影像的功能與一致性量測，沒有真實標註 accuracy／mAP／IoU；團隊仍需依產品需求訂定驗收門檻。

## Python 測試與跳過

- OpenVINO-only：**359 通過、10 跳過、24 warnings**。跳過包括沒有 NPU，以及此建置沒有 TensorRT，相關 GPU 測試不適用。
- OpenVINO＋TensorRT：**366 通過、3 跳過、30 warnings**。跳過為沒有 NPU、以及兩個只適用於沒有可用 TensorRT 的情境。
- `IVIT_C_LIBRARY` 指向本次 OpenVINO native prefix，與原清單安排相同；TensorRT C ABI 證據另外來自 C consumer，不能將此 Python C ABI suite 說成全套 TensorRT 覆蓋。
- warnings 包含既有 TorchScript／ONNX exporter deprecation，沒有將「測試成功」描述成「沒有警告」。

## 日誌與重現

全部產物在 `/tmp/ivit-checklist-20261015/`，為本機暫存，不隨 Git 發布：

- `log/step03-install.log`、`log/step04-install.log`、`log/step04-loaded.log`：兩份全新安裝與 11.4 實際載入。
- `log/step05-models.log`：真正下載、首次匯出與警告。
- `log/step09-ctest-trt.log`：第 9 步完整失敗訊息。
- `log/step12-literal-dotnet.log`：原命令找不到 dotnet。
- `log/step12-dotnet-ov.log`、`log/step12-dotnet-trt.log`：指定 SDK 路徑後成功結果。
- `results/comparison.json`：完整結果比對數據。
- `results/pytest-ov.xml`、`results/pytest-trt.xml`：JUnit 報告。
- `results/commands.tsv`：每個命令的退出碼與執行秒數。空 CTest 的退出碼 8 是預期防護；TensorRT-only CTest 的 8 是真實失敗。
- `results/python-*`、`results/cpp-*`：JSON、影像及原始 class mask。

執行腳本為 `/tmp/ivit-checklist-20261015.sh` 與 `/tmp/ivit-checklist-runtime-20261015.sh`。後者的 `tests` 分支最後刻意執行空 CTest，因此整個分支退出碼可能為 8，應以個別日誌與 TSV 判讀，不能把它說成 Python suite 失敗。

原清單外的可選 CI 全新重建、deb／tarball／wheel 發行形式、多 GPU、INT8、NPU、其他平台、長時間串流與完整訓練仍不在本次認證範圍。
