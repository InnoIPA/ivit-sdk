# 裝置選擇機制

iVIT-SDK 提供智慧型裝置自動選擇功能，根據當前平台自動選擇最佳的推論裝置。

## 設計原則

1. **平台適應**：根據當前執行平台選擇最佳裝置
2. **廠商感知**：區分不同廠商的硬體特性（NVIDIA dGPU vs Intel iGPU）
3. **優雅降級**：當首選裝置不可用時自動回退並發出警告

## 裝置優先順序

預設情況下（`latency` 策略），裝置選擇優先順序為：

```
NVIDIA 獨立顯卡 > Intel 獨立顯卡 > Intel 內建顯卡 > NPU > CPU
```

### 優先順序說明

| 優先級 | 裝置類型 | 說明 |
|--------|----------|------|
| 1 | NVIDIA dGPU | 獨立顯卡，最高運算效能 |
| 2 | Intel dGPU | Intel Arc 獨立顯卡 |
| 3 | Intel iGPU | Intel 內建顯卡 |
| 4 | NPU | 神經處理單元，低功耗 |
| 5 | CPU | 中央處理器，最終回退選項 |

## 選擇策略

SDK 提供四種選擇策略，適用於不同的應用場景：

### latency（預設）

**用途**：即時應用、低延遲需求

```python
import ivit

# 預設使用 latency 策略
model = ivit.load("model.onnx", device=ivit.devices.best())
model = ivit.load("model.onnx", device=ivit.devices.best(strategy="latency"))
```

**優先順序**：dGPU > iGPU > NPU > CPU

### throughput

**用途**：批次處理、高吞吐量需求

```python
model = ivit.load("model.onnx", device=ivit.devices.best(strategy="throughput"))
```

**優先順序**：dGPU > iGPU > NPU > CPU（與 latency 相同）

### efficiency

**用途**：邊緣裝置、低功耗需求

```python
model = ivit.load("model.onnx", device=ivit.devices.best(strategy="efficiency"))
```

**優先順序**：NPU > iGPU > CPU > dGPU

### balanced

**用途**：一般應用、平衡效能與功耗

```python
model = ivit.load("model.onnx", device=ivit.devices.best(strategy="balanced"))
```

**優先順序**：dGPU ≈ NPU > iGPU > CPU

## 評分機制

裝置選擇採用加權評分機制：

### 評分因素

| 因素 | 權重 | 說明 |
|------|------|------|
| 廠商 | 40% | NVIDIA (0.40) > Intel (0.30) > AMD (0.25) |
| 裝置類型 | 30% | 根據策略調整優先順序 |
| 記憶體 | 15% | 可用記憶體越大分數越高 |
| 任務適配 | 15% | 特定任務的加分項 |

### 廠商分數

```
NVIDIA:   0.40
Intel:    0.30
AMD:      0.25
其他:     0.10
```

### 裝置類型分數（latency 策略）

```
dGPU:     0.30
iGPU:     0.20
NPU:      0.15
CPU:      0.05
```

### 裝置類型分數（efficiency 策略）

```
NPU:      0.30
iGPU:     0.20
CPU:      0.15
dGPU:     0.10
```

## 使用方式

### 自動選擇（推薦）

```python
import ivit

# 自動選擇最佳裝置
model = ivit.load("yolox-s.onnx")  # device="auto" 是預設值
```

### 策略選擇

```python
# 最低延遲（預設）
model = ivit.load("model.onnx", device=ivit.devices.best())

# 最高能效
model = ivit.load("model.onnx", device=ivit.devices.best(strategy="efficiency"))

# 平衡模式
model = ivit.load("model.onnx", device=ivit.devices.best(strategy="balanced"))
```

### 指定裝置

```python
# 指定 CUDA 裝置
model = ivit.load("model.onnx", device="cuda:0")
model = ivit.load("model.onnx", device=ivit.devices.cuda())

# 指定 CPU
model = ivit.load("model.onnx", device="cpu")
model = ivit.load("model.onnx", device=ivit.devices.cpu())

# 指定 NPU
model = ivit.load("model.onnx", device="npu")
model = ivit.load("model.onnx", device=ivit.devices.npu())
```

### 列出可用裝置

```python
import ivit

# 列出所有裝置
for device in ivit.devices():
    print(f"{device.id}: {device.name}")

# 查看裝置摘要
print(ivit.devices.summary())
```

## 環境變數覆蓋

可透過環境變數 `IVIT_DEVICE` 強制指定裝置：

```bash
# 強制使用 CUDA 裝置 0
export IVIT_DEVICE=cuda:0
python my_app.py

# 強制使用 CPU
export IVIT_DEVICE=cpu
python my_app.py
```

## 後端選擇

裝置選擇會自動決定使用的推論後端：

| 裝置 | 後端 |
|------|------|
| `cuda:*` | TensorRT |
| `cpu`, `gpu:*`, `npu`, `vpu` | OpenVINO |

### TensorRT 優先

當 NVIDIA CUDA 裝置可用時，SDK 會優先使用 TensorRT 後端（效能提升 2-3 倍）：

```
TensorRT 可用 → 使用 TensorRT
TensorRT 不可用 → 回退到 OpenVINO
CUDA 不可用 → 回退到 CPU (OpenVINO)
```

## 常見問題

### Q: 為什麼選擇了非預期的裝置？

可能原因：
1. 首選裝置驅動程式未安裝
2. 首選裝置記憶體不足
3. 環境變數 `IVIT_DEVICE` 被設定

### Q: 如何確認使用的是哪個裝置？

```python
model = ivit.load("model.onnx")
print(f"Device: {model.device}")
print(f"Backend: {model.backend}")
```

### Q: 如何強制使用特定裝置？

1. 直接指定裝置字串：
```python
model = ivit.load("model.onnx", device="cuda:0")
```

2. 使用環境變數：
```bash
export IVIT_DEVICE=cuda:0
```

## 最佳實踐

1. **開發階段**：使用 `device="auto"` 讓 SDK 自動選擇
2. **部署階段**：明確指定裝置以確保一致性
3. **邊緣裝置**：使用 `strategy="efficiency"` 優化功耗
4. **即時應用**：使用 `strategy="latency"` 確保最低延遲
