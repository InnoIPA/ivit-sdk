# NVIDIA TensorRT 部署指南

本指南說明如何在 NVIDIA 平台上使用 iVIT-SDK 進行模型部署。

## 支援硬體

| 硬體類型 | 架構 | 說明 |
|----------|------|------|
| NVIDIA dGPU | x86_64 | GeForce RTX, Quadro, Tesla |
| NVIDIA Jetson | ARM64 | Orin, Xavier, Nano |

## 環境準備

### 1. 安裝 CUDA 和 TensorRT

**桌面/伺服器 GPU：**
```bash
# 安裝 CUDA Toolkit（以 CUDA 12.2 為例）
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-12-2

# 安裝 TensorRT
sudo apt-get install tensorrt
```

**Jetson 平台：**
```bash
# JetPack 已包含 CUDA 和 TensorRT
# 確認版本
dpkg -l | grep tensorrt
```

### 2. 安裝 iVIT-SDK

```bash
pip install ivit-sdk[tensorrt]
```

### 3. 確認安裝

```bash
# 檢查 GPU 狀態
nvidia-smi

# 檢查 iVIT 環境
ivit info
```

預期輸出：
```
iVIT-SDK v1.0.0
============================================================
System: Linux (Ubuntu 22.04)
Python: 3.10.12

Available Backends:
  - TensorRT 8.6.1 ✓

Devices:
  - cuda:0: NVIDIA GeForce RTX 4090 (TensorRT)
  - cuda:1: NVIDIA GeForce RTX 4090 (TensorRT)
```

## 快速開始

### 基本推論

```python
import ivit

# 自動使用 GPU
model = ivit.load("yolox-s.onnx", device="cuda:0")

# 執行推論
results = model("image.jpg")
results.show()
```

### 使用 TensorRT Engine

```python
# 直接載入 TensorRT Engine（最快）
model = ivit.load("yolox-s.engine", device="cuda:0")

# 或從 ONNX 自動轉換
model = ivit.load("yolox-s.onnx", device="cuda:0")
```

## 模型轉換

### ONNX 轉 TensorRT Engine

```bash
# 使用 CLI
ivit convert model.onnx -f tensorrt -o ./output/

# 指定精度
ivit convert model.onnx -f tensorrt -o ./output/ --precision FP16

# INT8 量化（需要校準資料）
ivit convert model.onnx -f tensorrt -o ./output/ --precision INT8 \
    --calibration-data ./calibration_images/
```

```python
# 使用 Python API
from ivit.convert import convert_to_tensorrt

convert_to_tensorrt(
    "model.onnx",
    output_path="model.engine",
    precision="FP16",
    workspace_size=1 << 30,  # 1GB
)
```

## 效能優化

### 配置 TensorRT 參數

```python
import ivit

model = ivit.load("model.onnx", device="cuda:0")

# 配置 TensorRT 特定參數
model.configure_tensorrt(
    workspace_size=1 << 30,         # 1GB 工作區
    enable_fp16=True,               # 開啟 FP16
    builder_optimization_level=5,   # 最高優化等級
    enable_sparsity=True,           # 稀疏性優化（Ampere+）
)

# 預熱
model.warmup(10)

# 執行推論
results = model("image.jpg")
```

### DLA 加速（Jetson）

```python
# 使用 DLA 核心（Jetson Orin/Xavier）
model = ivit.load("model.onnx", device="cuda:0")

model.configure_tensorrt(
    dla_core=0,                     # 使用 DLA 核心 0
    allow_gpu_fallback=True,        # 不支援的層回退到 GPU
    enable_fp16=True,               # DLA 需要 FP16
)
```

### 精度選擇

| 精度 | 速度 | 準確度 | 記憶體 | 適用場景 |
|------|------|--------|--------|----------|
| FP32 | 基準 | 最高 | 最大 | 高精度需求 |
| FP16 | 2x | 極小損失 | 減半 | 一般應用（推薦）|
| INT8 | 4x | 些許損失 | 1/4 | 邊緣裝置 |

## 多 GPU 推論

```python
import ivit

# 載入模型到不同 GPU
model_0 = ivit.load("model.onnx", device="cuda:0")
model_1 = ivit.load("model.onnx", device="cuda:1")

# 或使用裝置列表
for gpu_id in range(ivit.devices.cuda_count()):
    device = f"cuda:{gpu_id}"
    model = ivit.load("model.onnx", device=device)
    print(f"Loaded on {device}")
```

## 串流處理（CUDA Streams）

```python
import ivit

model = ivit.load("model.engine", device="cuda:0")

# 處理影片串流
for results in model.stream("rtsp://camera/stream"):
    frame = results.plot()
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

## 批次推論

```python
import ivit

model = ivit.load("model.onnx", device="cuda:0")

# TensorRT 批次推論
images = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
results = model.predict_batch(images, batch_size=4)

for i, r in enumerate(results):
    print(f"Image {i}: {len(r)} objects detected")
```

## 效能監控

```python
import ivit
from ivit.core.callbacks import FPSCounter, LatencyLogger

model = ivit.load("model.engine", device="cuda:0")

# 開啟 TensorRT profiling
model.configure_tensorrt(enable_profiling=True)

# 新增效能監控
fps = FPSCounter(window_size=30)
model.on("infer_end", fps)

@model.on("infer_end")
def log_timing(ctx):
    print(f"Preprocess: {ctx.preprocess_ms:.1f}ms, "
          f"Inference: {ctx.inference_ms:.1f}ms, "
          f"Postprocess: {ctx.postprocess_ms:.1f}ms")

# 執行推論
for i in range(100):
    results = model("image.jpg")

print(f"Average FPS: {fps.fps:.1f}")
```

## Jetson 專用優化

### 效能模式設定

```bash
# 設定最大效能模式（需要 root）
sudo nvpmodel -m 0
sudo jetson_clocks
```

### 記憶體優化

```python
model = ivit.load("model.onnx", device="cuda:0")

model.configure_tensorrt(
    workspace_size=512 << 20,  # 512MB（Jetson 記憶體較少）
    enable_fp16=True,          # 減少記憶體使用
)
```

### DLA + GPU 混合推論

```python
# Jetson Orin 有 2 個 DLA 核心
model.configure_tensorrt(
    dla_core=0,
    allow_gpu_fallback=True,
)
```

## 常見問題

### Q: 如何確認 CUDA 是否可用？

```bash
# 使用 nvidia-smi
nvidia-smi

# 使用 iVIT
ivit devices
```

### Q: TensorRT Engine 載入失敗？

TensorRT Engine 與 GPU 和 TensorRT 版本綁定。需要重新轉換：
```bash
ivit convert model.onnx -f tensorrt -o ./output/
```

### Q: 記憶體不足 (OOM)？

1. 減少 batch_size
2. 使用 FP16 精度
3. 減少 workspace_size

```python
model.configure_tensorrt(
    workspace_size=256 << 20,  # 256MB
    enable_fp16=True,
)
```

### Q: 如何提升推論速度？

1. 使用 TensorRT Engine（而非 ONNX）
2. 開啟 FP16/INT8
3. 使用 CUDA Graph
4. 增加 builder_optimization_level

```python
model.configure_tensorrt(
    enable_fp16=True,
    builder_optimization_level=5,
)
```

## 範例程式

完整範例程式位於：`examples/python/02_detection.py` 及 `examples/python/advanced/embedded_optimization.py`

```python
#!/usr/bin/env python3
"""NVIDIA TensorRT 部署範例"""

import ivit
from ivit.core.callbacks import FPSCounter

def main():
    # 確認 GPU 可用
    print("Available CUDA devices:")
    for d in ivit.devices():
        if "cuda" in d.id:
            print(f"  - {d.id}: {d.name}")

    # 載入模型
    model = ivit.load("yolox-s.onnx", device="cuda:0")

    # 配置 TensorRT
    model.configure_tensorrt(
        enable_fp16=True,
        builder_optimization_level=3,
    )

    # 新增 FPS 計數器
    fps = FPSCounter()
    model.on("infer_end", fps)

    # 預熱
    model.warmup(10)

    # 執行效能測試
    print("\nRunning benchmark...")
    for i in range(100):
        results = model("image.jpg")

    print(f"\nAverage FPS: {fps.fps:.1f}")
    print(f"Latency: {1000/fps.fps:.1f}ms")

if __name__ == "__main__":
    main()
```
