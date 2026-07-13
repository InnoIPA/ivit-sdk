# iVIT-SDK Model Zoo 模型清單

iVIT-SDK 內建 Model Zoo（`python/ivit/zoo/registry.py` 的 `MODEL_REGISTRY`）目前支援以下 **25 個模型**，全部採用商用友善授權（Apache-2.0、BSD-3-Clause）。

## 任務分類統計

| 任務 | 數量 |
|------|------|
| 物件偵測 Detection | 12 |
| 影像分類 Classification | 9 |
| 語意分割 Segmentation | 2 |
| 姿態估計 Pose Estimation | 2 |
| **總計** | **25** |

---

## 🎯 物件偵測 Detection（12 個）

| 模型 | 來源 | 輸入尺寸 | 類別數 | mAP50-95 | 授權 | 格式 |
|------|------|---------|--------|----------|------|------|
| `yolox-nano` | Megvii | 416×416 | 80 | 25.8 | Apache-2.0 | onnx / openvino / tensorrt |
| `yolox-tiny` | Megvii | 416×416 | 80 | 32.8 | Apache-2.0 | onnx / openvino / tensorrt |
| `yolox-s` | Megvii | 640×640 | 80 | 40.5 | Apache-2.0 | onnx / openvino / tensorrt |
| `yolox-m` | Megvii | 640×640 | 80 | 46.9 | Apache-2.0 | onnx / openvino / tensorrt |
| `yolox-l` | Megvii | 640×640 | 80 | 49.7 | Apache-2.0 | onnx / openvino / tensorrt |
| `rtdetr-l` | PaddlePaddle | 640×640 | 80 | 53.0 | Apache-2.0 | onnx / openvino / tensorrt |
| `rtdetr-x` | PaddlePaddle | 640×640 | 80 | 54.8 | Apache-2.0 | onnx / openvino / tensorrt |
| `fasterrcnn-mobilenet` | torchvision | 640×640 | 91 | 32.8 | BSD-3-Clause | onnx / openvino |
| `fasterrcnn-resnet50` | torchvision | 800×800 | 91 | 46.7 | BSD-3-Clause | onnx / openvino |
| `retinanet-resnet50` | torchvision | 800×800 | 91 | 41.5 | BSD-3-Clause | onnx / openvino |
| `ssdlite-mobilenet` | torchvision | 320×320 | 91 | 21.3 | BSD-3-Clause | onnx / openvino |
| `fcos-resnet50` | torchvision | 800×800 | 91 | 39.2 | BSD-3-Clause | onnx / openvino |

## 🏷️ 影像分類 Classification（9 個）

| 模型 | 來源 | 輸入尺寸 | 類別數 | Top-1 | 授權 | 格式 |
|------|------|---------|--------|-------|------|------|
| `resnet18` | torchvision | 224×224 | 1000 | 69.8 | BSD-3-Clause | onnx / openvino |
| `resnet50` | torchvision | 224×224 | 1000 | 76.1 | BSD-3-Clause | onnx / openvino |
| `mobilenetv3` | torchvision | 224×224 | 1000 | 74.0 | BSD-3-Clause | onnx / openvino |
| `efficientnet-b0` | torchvision | 224×224 | 1000 | 77.1 | Apache-2.0 | onnx / openvino |
| `mambavision-t` | NVIDIA | 224×224 | 1000 | 82.3 | Apache-2.0 | onnx / openvino / tensorrt |
| `mambavision-s` | NVIDIA | 224×224 | 1000 | 83.3 | Apache-2.0 | onnx / openvino / tensorrt |
| `mambavision-b` | NVIDIA | 224×224 | 1000 | 84.2 | Apache-2.0 | onnx / openvino / tensorrt |
| `mambavision-l` | NVIDIA | 224×224 | 1000 | 85.0 | Apache-2.0 | onnx / openvino / tensorrt |
| `mambavision-l2` | NVIDIA | 224×224 | 1000 | 85.3 | Apache-2.0 | onnx / openvino / tensorrt |

## 🖼️ 語意分割 Segmentation（2 個）

| 模型 | 來源 | 輸入尺寸 | 類別數 | mIoU | 授權 | 格式 |
|------|------|---------|--------|------|------|------|
| `deeplabv3-resnet50` | torchvision | 520×520 | 21 | 66.4 | BSD-3-Clause | onnx / openvino |
| `deeplabv3-mobilenetv3` | torchvision | 520×520 | 21 | 60.3 | BSD-3-Clause | onnx / openvino |

## 🧍 姿態估計 Pose Estimation（2 個）

| 模型 | 來源 | 輸入尺寸 | 關鍵點 | AP | 授權 | 格式 |
|------|------|---------|--------|-----|------|------|
| `rtmpose-s` | MMPose | 256×192 | 17 | 72.2 | Apache-2.0 | onnx / openvino / tensorrt |
| `rtmpose-m` | MMPose | 256×192 | 17 | 75.3 | Apache-2.0 | onnx / openvino / tensorrt |

---

## 使用方式

### CLI

```bash
ivit zoo list                  # 列出所有模型
ivit zoo download yolox-s      # 下載模型
```

### Python API

```python
import ivit

ivit.zoo.list_models()                 # 列出全部模型名稱
ivit.zoo.list_models(task="detect")    # 依任務過濾（detect/classify/segment/pose）
ivit.zoo.print_models()                # 格式化表格輸出
ivit.zoo.search("yolo")                # 依名稱／標籤／描述搜尋
ivit.zoo.get_model_info("yolox-s")     # 取得單一模型詳細資訊

model = ivit.zoo.load("yolox-s")       # 下載並載入模型
results = model("image.jpg")           # 推論
```

---

## 可訓練模型清單（`ivit.train`）

除了上述用於下載／推論的 `MODEL_REGISTRY`，訓練模組另以後端分組支援以下可訓練模型
（部分名稱尚未列入下載用 registry，可透過 `ivit.zoo.list_trainable_models()` 查詢）：

| 訓練後端 | 模型 |
|----------|------|
| `yolox` | `yolox-nano`, `yolox-tiny`, `yolox-s`, `yolox-m`, `yolox-l` |
| `torchvision-classify` | `resnet18`, `resnet34`, `resnet50`, `resnet101`, `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`, `mobilenet_v2`, `mobilenet_v3_small`, `mobilenet_v3_large`, `vgg16`, `vgg19`, `densenet121` |
| `torchvision-segment` | `deeplabv3-resnet50`, `deeplabv3-mobilenetv3` |

```python
import ivit
ivit.zoo.list_trainable_models()       # 依後端分組列出可訓練模型
ivit.zoo.is_trainable("yolox-s")       # 判斷模型是否可訓練
```
