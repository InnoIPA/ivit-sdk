# iVIT-SDK 效能調優指南

本指南提供在不同平台上優化推論效能的最佳實踐。

## 效能優化層級

```
┌─────────────────────────────────────────────────────────────┐
│  Level 5: 系統層級優化（OS、驅動程式、硬體設定）              │
├─────────────────────────────────────────────────────────────┤
│  Level 4: 模型層級優化（量化、剪枝、蒸餾）                    │
├─────────────────────────────────────────────────────────────┤
│  Level 3: 後端層級優化（TensorRT/OpenVINO/QNN 參數）         │
├─────────────────────────────────────────────────────────────┤
│  Level 2: SDK 層級優化（批次、串流、預熱）                    │
├─────────────────────────────────────────────────────────────┤
│  Level 1: 應用層級優化（資料載入、前後處理）                  │
└─────────────────────────────────────────────────────────────┘
```

## Level 1: 應用層級優化

### 1.1 影像載入優化

```python
import ivit
import cv2
import numpy as np

# ❌ 每次都從檔案載入
for i in range(100):
    results = model("image.jpg")  # 每次讀取檔案

# ✅ 預載入到記憶體
image = cv2.imread("image.jpg")
for i in range(100):
    results = model(image)  # 使用記憶體中的影像
```

### 1.2 避免重複初始化

```python
# ❌ 每次建立新模型
for image in images:
    model = ivit.load("model.onnx")
    results = model(image)

# ✅ 重複使用同一模型
model = ivit.load("model.onnx")
for image in images:
    results = model(image)
```

### 1.3 使用適當的輸入尺寸

```python
# 根據需求選擇輸入尺寸
# 較小尺寸 = 較快速度，較低準確度
model = ivit.load("model.onnx")

# 偵測大物件：較小尺寸即可
results = model(image, imgsz=320)

# 偵測小物件：需要較大尺寸
results = model(image, imgsz=640)
```

## Level 2: SDK 層級優化

### 2.1 模型預熱

```python
import ivit

model = ivit.load("model.onnx", device="cuda:0")

# 預熱（首次推論較慢，因為需要初始化）
model.warmup(iterations=10)

# 實際推論
results = model("image.jpg")
```

### 2.2 批次推論

```python
import ivit

model = ivit.load("model.onnx", device="cuda:0")

# ❌ 逐張處理
for image in images:
    result = model(image)

# ✅ 批次處理（GPU 並行）
results = model.predict_batch(images, batch_size=8)
```

### 2.3 串流模式

```python
import ivit

model = ivit.load("model.onnx", device="cuda:0")

# 使用 generator 模式減少記憶體使用
for results in model.stream("video.mp4"):
    # 處理每一幀
    process(results)
```

### 2.4 非同步推論

```python
import ivit
import asyncio

model = ivit.load("model.onnx", device="cuda:0")

async def async_inference(images):
    tasks = [model.predict_async(img) for img in images]
    results = await asyncio.gather(*tasks)
    return results
```

## Level 3: 後端層級優化

### 3.1 OpenVINO 優化

```python
model = ivit.load("model.onnx", device="cpu")

model.configure_openvino(
    # 效能模式
    performance_mode="LATENCY",      # 單張快速
    # performance_mode="THROUGHPUT", # 批次處理

    # CPU 優化
    num_threads=0,                   # 自動（或指定核心數）
    enable_cpu_pinning=True,         # 綁定 CPU 核心

    # 精度
    inference_precision="FP16",      # 比 FP32 快 ~2x

    # 並行流
    num_streams=4,                   # 多流並行

    # 快取
    cache_dir="./cache/",            # 模型快取
)
```

### 3.2 TensorRT 優化

```python
model = ivit.load("model.onnx", device="cuda:0")

model.configure_tensorrt(
    # 精度（效能提升排序：INT8 > FP16 > FP32）
    enable_fp16=True,                # ~2x 速度
    # enable_int8=True,              # ~4x 速度（需校準）

    # 優化等級
    builder_optimization_level=5,    # 最高優化（建置較慢）

    # 工作區
    workspace_size=1 << 30,          # 1GB

    # Ampere+ GPU
    enable_sparsity=True,            # 稀疏優化

    # 快取
    timing_cache_path="./timing.cache",
)
```

### 3.3 QNN 優化 (Qualcomm IQ Series)（規劃中）

> **注意**：QNN 後端支援目前為規劃中功能，以下配置尚未提供。

```python
model = ivit.load("model.onnx", device="iq9")  # IQ9/IQ8/IQ6

model.configure_qnn(
    # HTP 後端（Hexagon Tensor Processor）
    backend="htp",

    # 效能模式
    performance_profile="HIGH_PERFORMANCE",

    # HTP 精度
    htp_precision="fp16",  # fp16 or int8

    # 零複製
    use_native_memory=True,
)
```

## Level 4: 模型層級優化

### 4.1 精度量化

| 精度 | 記憶體 | 速度 | 準確度損失 |
|------|--------|------|-----------|
| FP32 | 100% | 1x | 0% |
| FP16 | 50% | ~2x | <1% |
| INT8 | 25% | ~4x | 1-3% |

```bash
# ONNX 轉 INT8（TensorRT）
ivit convert model.onnx -f tensorrt -o ./output/ \
    --precision INT8 \
    --calibration-data ./calibration/
```

### 4.2 選擇適當的模型

| 模型 | 參數量 | FPS (GPU) | mAP |
|------|--------|-----------|-----|
| YOLOX-Nano | 3.2M | 450 | 37.3 |
| YOLOX-S | 11.2M | 280 | 44.9 |
| YOLOX-M | 25.9M | 140 | 50.2 |
| YOLOX-L | 43.7M | 85 | 52.9 |
| YOLOX-L | 68.2M | 55 | 53.9 |

```python
# 邊緣裝置選擇輕量模型
model = ivit.zoo.load("yolox-nano")  # nano 版本

# 伺服器選擇大模型
model = ivit.zoo.load("yolox-l")  # large 版本
```

## Level 5: 系統層級優化

### 5.1 Linux 系統

```bash
# 設定 CPU 效能模式
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 關閉 Turbo Boost（一致性）
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# 設定 NUMA 親和性
numactl --cpunodebind=0 --membind=0 python inference.py
```

### 5.2 NVIDIA GPU

```bash
# 設定持續模式
sudo nvidia-smi -pm 1

# 設定最大時脈
sudo nvidia-smi -lgc 2100

# 查看 GPU 狀態
nvidia-smi -q -d PERFORMANCE
```

### 5.3 Jetson 平台

```bash
# 設定最大效能模式
sudo nvpmodel -m 0

# 開啟所有時脈
sudo jetson_clocks

# 查看狀態
sudo jetson_clocks --show
```

## 效能測試

### 使用 ivit benchmark

```bash
# 基本測試
ivit benchmark model.onnx

# 指定裝置和迭代次數
ivit benchmark model.onnx --device cuda:0 --iterations 1000

# 測試不同批次大小
ivit benchmark model.onnx --batch-size 1 4 8 16
```

### 使用 Python API

```python
import ivit
from ivit.core.callbacks import FPSCounter, LatencyLogger

model = ivit.load("model.onnx", device="cuda:0")

# 設定回調
fps = FPSCounter(window_size=100)
model.on("infer_end", fps)

@model.on("infer_end")
def log_timing(ctx):
    print(f"Pre: {ctx.preprocess_ms:.1f}ms, "
          f"Inf: {ctx.inference_ms:.1f}ms, "
          f"Post: {ctx.postprocess_ms:.1f}ms")

# 預熱
model.warmup(50)

# 測試
for i in range(1000):
    results = model("image.jpg")

print(f"\n=== Results ===")
print(f"FPS: {fps.fps:.1f}")
print(f"Latency: {1000/fps.fps:.1f}ms")
```

## 常見效能瓶頸

### 1. 資料傳輸

**症狀**：GPU 使用率低，CPU 使用率高

**解決方案**：
- 使用 GPU 前處理
- 減少 CPU-GPU 資料傳輸
- 使用 pinned memory

### 2. 記憶體不足

**症狀**：OOM 錯誤，推論突然變慢

**解決方案**：
- 減少批次大小
- 使用 FP16/INT8
- 串流處理大型輸入

### 3. 首次推論慢

**症狀**：第一次推論比後續慢很多

**解決方案**：
- 使用預熱
- 使用模型快取
- 預編譯 TensorRT Engine

### 4. 效能不穩定

**症狀**：延遲波動大

**解決方案**：
- 設定固定時脈頻率
- 關閉 CPU 節能
- 使用即時調度器

## 效能優化檢查清單

```
□ 預熱模型（warmup）
□ 使用適當的批次大小
□ 開啟 FP16/INT8
□ 配置後端特定參數
□ 使用模型快取
□ 避免重複載入模型
□ 預載入影像到記憶體
□ 使用串流模式處理影片
□ 設定系統效能模式
□ 選擇適當大小的模型
```

## 平台效能參考

### Intel Core i7-1365U (CPU)

| 模型 | FP32 FPS | FP16 FPS |
|------|----------|----------|
| YOLOX-Nano | 25 | 45 |
| YOLOX-S | 12 | 22 |
| ResNet50 | 80 | 120 |

### NVIDIA RTX 4090 (GPU)

| 模型 | FP32 FPS | FP16 FPS | INT8 FPS |
|------|----------|----------|----------|
| YOLOX-Nano | 450 | 680 | 920 |
| YOLOX-S | 280 | 420 | 580 |
| YOLOX-L | 55 | 90 | 140 |

### Qualcomm Snapdragon 8 Gen 2 (DSP)（規劃中）

| 模型 | FP16 FPS | INT8 FPS |
|------|----------|----------|
| YOLOX-Nano | 45 | 85 |
| MobileNetV3 | 120 | 200 |
