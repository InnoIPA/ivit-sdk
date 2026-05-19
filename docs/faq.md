# iVIT-SDK 常見問題 FAQ

## 安裝與環境

### Q: 如何安裝 iVIT-SDK？

```bash
pip install ivit-sdk
```

如需特定後端支援：
```bash
# OpenVINO 支援
pip install ivit-sdk[openvino]

# TensorRT 支援
pip install ivit-sdk[tensorrt]

# 全部支援
pip install ivit-sdk[all]
```

### Q: 如何確認安裝是否成功？

```bash
ivit info
```

這會顯示版本、可用後端和裝置列表。

### Q: Python 版本需求？

iVIT-SDK 支援 Python 3.9 及以上版本。

### Q: 支援哪些作業系統？

- Linux (Ubuntu 20.04+, CentOS 8+)
- Windows 10/11
- macOS（部分功能）

---

## 模型載入

### Q: 支援哪些模型格式？

| 格式 | 副檔名 | 後端 |
|------|--------|------|
| ONNX | .onnx | 所有 |
| OpenVINO IR | .xml + .bin | OpenVINO |
| TensorRT Engine | .engine, .trt | TensorRT |
| Qualcomm DLC | .dlc | QNN (規劃中) |

### Q: 如何載入模型？

```python
import ivit

# 從檔案載入
model = ivit.load("model.onnx")

# 指定裝置
model = ivit.load("model.onnx", device="cuda:0")

# 從 Model Zoo 載入
model = ivit.zoo.load("yolox-s")
```

### Q: 模型載入失敗怎麼辦？

常見原因和解決方案：

1. **檔案不存在**：確認路徑正確
2. **格式不支援**：使用 `ivit convert` 轉換
3. **後端不可用**：執行 `ivit info` 確認後端安裝
4. **版本不相容**：重新轉換模型

```python
# 查看詳細錯誤
try:
    model = ivit.load("model.onnx")
except ivit.ModelLoadError as e:
    print(e)  # 顯示詳細錯誤訊息和建議
```

### Q: 如何自動偵測模型任務類型？

iVIT-SDK 會根據模型輸出自動偵測：
- 1 個輸出 + softmax → 分類
- 多個輸出 + boxes → 偵測
- 輸出含 mask → 分割

也可以手動指定：
```python
model = ivit.load("model.onnx", task="detect")
```

---

## 裝置與硬體

### Q: 如何列出可用裝置？

```python
import ivit

# 列出所有裝置
for device in ivit.devices():
    print(f"{device.id}: {device.name}")

# 或使用 CLI
# ivit devices
```

### Q: 如何選擇最佳裝置？

```python
# 自動選擇
device = ivit.devices.best()

# 按優先級選擇（效能優先）
device = ivit.devices.best("performance")

# 按優先級選擇（效率優先）
device = ivit.devices.best("efficiency")
```

### Q: 如何確認 GPU 是否可用？

```python
import ivit

# 檢查 CUDA 裝置
cuda_devices = ivit.devices.cuda()
if cuda_devices:
    print(f"Found GPU: {cuda_devices.name}")
else:
    print("No CUDA GPU available")
```

### Q: 多 GPU 如何選擇？

```python
# 使用特定 GPU
model = ivit.load("model.onnx", device="cuda:0")
model = ivit.load("model.onnx", device="cuda:1")

# 列出所有 GPU
for i in range(ivit.devices.cuda_count()):
    print(f"GPU {i}: {ivit.devices.cuda(i).name}")
```

---

## 推論

### Q: 如何執行推論？

```python
import ivit

model = ivit.load("model.onnx")

# 單張影像
results = model("image.jpg")

# NumPy array
import cv2
image = cv2.imread("image.jpg")
results = model(image)

# 批次推論
results = model.predict_batch(["img1.jpg", "img2.jpg"])
```

### Q: 如何處理影片？

```python
# 串流處理
for results in model.stream("video.mp4"):
    frame = results.plot()
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) == ord('q'):
        break
```

### Q: 如何調整信心度閾值？

```python
# 偵測時設定
results = model.predict(
    "image.jpg",
    conf=0.5,    # 信心度閾值
    iou=0.45,    # NMS IOU 閾值
)
```

### Q: 如何只偵測特定類別？

```python
results = model.predict(
    "image.jpg",
    classes=[0, 1, 2],  # 只偵測類別 0, 1, 2
)
```

---

## 結果處理

### Q: 如何取得偵測結果？

```python
results = model("image.jpg")

# 遍歷偵測結果
for det in results.detections:
    print(f"Class: {det.class_name}")
    print(f"Confidence: {det.confidence:.2f}")
    print(f"Box: {det.box}")  # (x1, y1, x2, y2)
```

### Q: 如何視覺化結果？

```python
# 顯示
results.show()

# 儲存
results.save("output.jpg")

# 取得繪製後的影像
image = results.plot()
```

### Q: 如何取得原始輸出？

```python
# 取得原始推論輸出
raw_outputs = model.infer_raw({"input": preprocessed_image})
```

---

## 效能優化

### Q: 如何提升推論速度？

1. **使用預熱**
   ```python
   model.warmup(10)
   ```

2. **使用 FP16**
   ```python
   model.configure_tensorrt(enable_fp16=True)
   model.configure_openvino(inference_precision="FP16")
   ```

3. **使用批次推論**
   ```python
   results = model.predict_batch(images, batch_size=8)
   ```

4. **選擇適當的模型**
   - nano (n) 版本最快
   - large (l), extra (x) 版本最準

### Q: 如何減少記憶體使用？

1. 使用 FP16/INT8 精度
2. 減少批次大小
3. 使用串流模式處理影片
4. 選擇較小的模型

### Q: 如何監控效能？

```python
from ivit.core.callbacks import FPSCounter

fps = FPSCounter()
model.on("infer_end", fps)

# 推論後
print(f"FPS: {fps.fps}")
```

---

## 模型轉換

### Q: 如何轉換模型格式？

`ivit convert` 優先使用 C++ binding 進行轉換（與 APT 安裝的 OpenVINO 相容），不需要額外安裝 Python openvino 模組。

```bash
# ONNX → OpenVINO IR（FP32）
ivit convert model.onnx -f openvino

# ONNX → OpenVINO IR（FP16 壓縮）
ivit convert model.onnx -f openvino --precision fp16

# ONNX → TensorRT
ivit convert model.onnx -f tensorrt -o ./output/

# 指定精度
ivit convert model.onnx -f tensorrt --precision fp16
```

### Q: 轉換前需要先編譯嗎？

是的，`ivit convert` 依賴 C++ binding。首次使用或更新 SDK 後需要：

```bash
# 編譯
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DIVIT_BUILD_PYTHON=ON
make -j$(nproc)
cd ..

# 安裝系統 library 與 Python 套件
sudo cp build/lib/libivit.so /usr/local/lib/
sudo ldconfig
pip install -e .
```

### Q: 可以轉換 INT8 精度嗎？

OpenVINO 的 INT8 量化**無法**透過 `ivit convert` 直接完成，因為 INT8 量化需要校準資料集（calibration dataset）來計算每一層的 scale 和 zero-point。

如需 INT8 量化，請使用 [NNCF toolkit](https://github.com/openvinotoolkit/nncf)：

```python
import nncf
import openvino as ov

# 準備校準資料集
calibration_dataset = nncf.Dataset(data_loader)

# 量化模型
model = ov.Core().read_model("model.xml")
quantized_model = nncf.quantize(model, calibration_dataset)
ov.save_model(quantized_model, "model_int8.xml")
```

TensorRT 的 FP16/INT8 可透過 `ivit convert --precision fp16` 或 `--precision int8` 直接轉換（INT8 精度取決於硬體支援）。

### Q: FP16 轉換會提升推論速度嗎？

**不會**。FP16 壓縮的唯一好處是模型檔案大小減半（例如 YOLOX-Nano 從 13MB 縮小至 6MB），不影響推論速度：

- **NPU**：硬體原生以 FP16 計算，不論模型是 FP32 或 FP16，最終執行相同的編譯結果。FP16 IR 因額外的 Convert 節點反而可能略慢。
- **CPU**：x86 CPU 以 FP32 計算，FP16 權重會在推論時解壓回 FP32，無效能增益。

**建議**：直接使用 ONNX 或 FP32 IR 即可獲得最佳效能。FP16 適合儲存空間有限的部署場景。

### Q: 轉換失敗怎麼辦？

1. 確認已完成編譯並安裝（見上方「轉換前需要先編譯嗎？」）
2. 確認來源模型為有效的 ONNX 格式
3. 確認目標後端已安裝（OpenVINO APT 套件或 TensorRT）
4. 檢查運算元支援——嘗試簡化模型

```bash
# 簡化 ONNX 模型
pip install onnxsim
python -m onnxsim model.onnx model_sim.onnx
```

---

## Model Zoo

### Q: 如何使用 Model Zoo？

```python
import ivit

# 列出可用模型
ivit.zoo.list_models()

# 載入模型（自動下載）
model = ivit.zoo.load("yolox-s")

# 搜尋模型
ivit.zoo.search("yolo")
```

### Q: Model Zoo 有哪些模型？

| 任務 | 模型 |
|------|------|
| 偵測 | yolox-{nano,tiny,s,m,l} |
| 分類 | resnet50, mobilenet_v3_small, efficientnet_b0 |
| 分割 | deeplabv3-resnet50, deeplabv3-mobilenetv3 |
| 姿態 | rtmpose-s |

### Q: 如何下載模型到本地？

```python
# 下載到指定目錄
ivit.zoo.download("yolox-s", path="./models/")
```

---

## Callback 系統

### Q: 如何使用 Callback？

```python
# 使用裝飾器
@model.on("infer_end")
def log_result(ctx):
    print(f"Latency: {ctx.latency_ms}ms")

# 使用 lambda
model.on("infer_end", lambda ctx: print(ctx.latency_ms))

# 使用內建 callback
from ivit.core.callbacks import FPSCounter
model.on("infer_end", FPSCounter())
```

### Q: 有哪些 Callback 事件？

| 事件 | 說明 |
|------|------|
| pre_process | 前處理前 |
| post_process | 後處理後 |
| infer_start | 推論開始 |
| infer_end | 推論結束 |
| batch_start | 批次開始 |
| batch_end | 批次結束 |
| stream_start | 串流開始 |
| stream_frame | 串流每幀 |
| stream_end | 串流結束 |

### Q: 如何移除 Callback？

```python
# 移除特定 callback
model.remove_callback("infer_end", my_callback)

# 移除某事件的所有 callback
model.remove_callback("infer_end")
```

---

## CLI 工具

### Q: 有哪些 CLI 指令？

```bash
ivit info          # 系統資訊
ivit devices       # 列出裝置
ivit predict       # 執行推論
ivit benchmark     # 效能測試
ivit convert       # 模型轉換
ivit zoo           # Model Zoo 操作
```

### Q: 如何使用 CLI 執行推論？

```bash
# 基本推論
ivit predict model.onnx image.jpg

# 指定裝置
ivit predict model.onnx image.jpg --device cuda:0

# 批次推論
ivit predict model.onnx images/ --batch-size 4

# 處理影片
ivit predict model.onnx video.mp4 --stream
```

---

## 錯誤處理

### Q: 如何處理錯誤？

```python
import ivit

try:
    model = ivit.load("model.onnx")
except ivit.ModelLoadError as e:
    print(f"載入失敗: {e}")
except ivit.DeviceNotFoundError as e:
    print(f"裝置不可用: {e}")
except ivit.BackendNotAvailableError as e:
    print(f"後端未安裝: {e}")
```

### Q: 錯誤訊息看不懂怎麼辦？

iVIT-SDK 提供友善的錯誤訊息，包含：
- 問題描述
- 可能原因
- 解決建議

```
ivit.ModelLoadError: 無法載入模型 'model.onnx'

可能原因：
  1. 檔案不存在：請確認路徑 'model.onnx' 是否正確
  2. 格式不支援：目前支援 .onnx, .xml, .engine

解決建議：
  - 執行 `ivit info` 查看系統資訊和可用後端
  - 使用 `ivit convert` 轉換模型格式
```

---

## 其他

### Q: 如何取得技術支援？

1. 查看文件：https://github.com/innodisk-mannywang/ivit-sdk
2. 提交 Issue：https://github.com/innodisk-mannywang/ivit-sdk/issues
3. 聯繫 Innodisk 技術支援

### Q: 如何貢獻程式碼？

1. Fork 專案
2. 建立功能分支
3. 提交 Pull Request

### Q: 授權方式？

iVIT-SDK 採用 Apache 2.0 授權。
