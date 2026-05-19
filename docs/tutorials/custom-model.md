# 自定義模型整合指南

> **適用對象**：需要使用非 Model Zoo 模型的開發者
> **難度**：中級
> **預估時間**：30-60 分鐘

---

## 目錄

1. [前言](#前言)
2. [常見問題與診斷](#常見問題與診斷)
3. [範例一：前後處理不匹配](#範例一前後處理不匹配)
4. [範例二：不支援的運算子](#範例二不支援的運算子)
5. [範例三：自定義輸出格式](#範例三自定義輸出格式)
6. [偵錯技巧](#偵錯技巧)
7. [最佳實務](#最佳實務)

---

## 前言

### 為什麼需要這份指南？

iVIT Model Zoo 中的模型已經過測試，前後處理器都已正確配置。但在實際專案中，您可能需要使用：

- 自己訓練的模型
- 從其他框架轉換的模型（TensorFlow、PaddlePaddle 等）
- 修改過網路結構的模型
- 第三方提供的模型

這些模型可能會遇到各種相容性問題。

### 支援的模型格式

| 格式 | 副檔名 | 推論引擎 | 備註 |
|------|--------|----------|------|
| ONNX | `.onnx` | OpenVINO / TensorRT | 推薦格式，相容性最高 |
| OpenVINO IR | `.xml` + `.bin` | OpenVINO | Intel 硬體最佳化 |
| TensorRT Engine | `.engine` / `.trt` | TensorRT | NVIDIA 硬體最佳化，需相同 GPU 架構 |

---

## 常見問題與診斷

### 問題診斷流程

```
載入模型失敗？
    ├─ 是 → 檢查格式和運算子支援
    └─ 否 → 推論結果正確嗎？
              ├─ 是 → 完成！
              └─ 否 → 檢查前後處理
```

### 第一步：檢查模型資訊

```python
import ivit

# 載入模型
model = ivit.load("your_model.onnx")

# 檢查輸入資訊
print("=== 輸入資訊 ===")
for info in model.input_info:
    print(f"  名稱: {info['name']}")
    print(f"  形狀: {info['shape']}")
    print(f"  類型: {info['dtype']}")

# 檢查輸出資訊
print("\n=== 輸出資訊 ===")
for info in model.output_info:
    print(f"  名稱: {info['name']}")
    print(f"  形狀: {info['shape']}")
    print(f"  類型: {info['dtype']}")
```

### 第二步：執行原始推論

```python
import numpy as np

# 建立測試輸入（根據模型輸入形狀）
input_shape = model.input_info[0]['shape']  # 例如 [1, 3, 640, 640]
dummy_input = np.random.rand(*input_shape).astype(np.float32)

# 執行原始推論（無前後處理）
input_name = model.input_info[0]['name']
raw_outputs = model.infer_raw({input_name: dummy_input})

# 檢查輸出
for name, output in raw_outputs.items():
    print(f"{name}: shape={output.shape}, dtype={output.dtype}")
    print(f"  min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
```

---

## 範例一：前後處理不匹配

### 問題描述

您訓練了一個自定義 YOLO 模型，但匯出時使用了不同於 iVIT-SDK 預設的正規化設定。

**症狀**：
- 模型可以載入
- 推論可以執行
- 但偵測結果全錯或沒有偵測結果

### 問題模型：custom_yolo.onnx

假設您的模型有以下特性：
- 輸入：BGR 格式（而非 RGB）
- 正規化：ImageNet 標準（mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]）
- 輸入尺寸：416×416（而非 640×640）

### 診斷過程

```python
import ivit
import numpy as np
import cv2

# 載入模型
model = ivit.load("custom_yolo.onnx")

# 檢查輸入
print(model.input_info)
# 輸出: [{'name': 'images', 'shape': [1, 3, 416, 416], 'dtype': 'float32'}]

# 使用預設推論
image = cv2.imread("test.jpg")
results = model(image)

print(f"偵測數量: {len(results)}")  # 可能是 0 或結果不正確
```

### 解決方案：自定義前處理器

```python
import ivit
from ivit.core.processors import BasePreProcessor, register_preprocessor
import numpy as np
import cv2


class CustomYOLOPreProcessor(BasePreProcessor):
    """
    自定義 YOLO 前處理器

    處理流程：
    1. Letterbox 縮放到 416x416
    2. BGR 格式（不轉換）
    3. ImageNet 正規化
    4. HWC → NCHW
    """

    def __init__(self, target_size=416):
        self.target_size = target_size
        # ImageNet 標準（BGR 順序）
        self.mean = np.array([0.406, 0.456, 0.485], dtype=np.float32)
        self.std = np.array([0.225, 0.224, 0.229], dtype=np.float32)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # 1. Letterbox 縮放
        image, self.scale, self.pad = self._letterbox(image, self.target_size)

        # 2. 保持 BGR 格式（不做轉換）

        # 3. 正規化
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std

        # 4. HWC → NCHW
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)

        return image.astype(np.float32)

    def _letterbox(self, image, target_size):
        """Letterbox 縮放，保持比例"""
        h, w = image.shape[:2]
        scale = min(target_size / h, target_size / w)

        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h))

        # 填充到目標尺寸
        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2

        padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = image

        return padded, scale, (pad_w, pad_h)


# 註冊並使用
register_preprocessor("custom_yolo", CustomYOLOPreProcessor)

model = ivit.load("custom_yolo.onnx")
model.set_preprocessor(CustomYOLOPreProcessor(target_size=416))

# 現在應該能正確推論
results = model("test.jpg")
print(f"偵測數量: {len(results)}")
```

---

## 範例二：不支援的運算子

### 問題描述

您從 PaddlePaddle 匯出了一個模型，轉換為 ONNX 後載入失敗。

**症狀**：
- 模型載入時拋出錯誤
- 錯誤訊息包含 "Unsupported operator" 或 "Unknown op type"

### 錯誤訊息範例

```
ivit.ModelLoadError: 無法載入模型 'paddle_model.onnx'

可能原因：
  1. 模型包含不支援的運算子: CustomOp, DeformConv

解決建議：
  - 使用 ONNX Simplifier 簡化模型
  - 檢查 ONNX opset 版本（建議 opset 11-17）
  - 將自定義運算子替換為標準運算子組合
```

### 診斷過程

```python
# 使用 onnx 檢查模型
import onnx

model = onnx.load("paddle_model.onnx")
print(f"ONNX opset 版本: {model.opset_import[0].version}")

# 列出所有運算子
ops = set()
for node in model.graph.node:
    ops.add(node.op_type)

print(f"使用的運算子: {sorted(ops)}")

# 檢查是否有自定義運算子
standard_ops = {
    'Conv', 'BatchNormalization', 'Relu', 'MaxPool', 'Add',
    'Concat', 'Reshape', 'Transpose', 'Sigmoid', 'Mul',
    # ... 等等
}

custom_ops = ops - standard_ops
if custom_ops:
    print(f"⚠️ 可能不支援的運算子: {custom_ops}")
```

### 解決方案一：使用 ONNX Simplifier

```bash
# 安裝 onnx-simplifier
pip install onnx-simplifier

# 簡化模型
python -m onnxsim paddle_model.onnx paddle_model_simplified.onnx
```

```python
# 驗證簡化後的模型
import ivit

model = ivit.load("paddle_model_simplified.onnx")
print("模型載入成功！")
```

### 解決方案二：調整 ONNX opset 版本

```python
import onnx
from onnx import version_converter

# 載入原始模型
model = onnx.load("paddle_model.onnx")

# 轉換到 opset 17（較新版本支援更多運算子）
converted = version_converter.convert_version(model, 17)

# 儲存
onnx.save(converted, "paddle_model_opset17.onnx")
```

### 解決方案三：重新匯出模型

如果上述方法都無效，建議重新匯出模型：

```python
# PaddlePaddle 匯出範例
import paddle
from paddle.static import InputSpec

# 載入模型
model = paddle.jit.load("your_model")

# 定義輸入規格
input_spec = [InputSpec(shape=[1, 3, 640, 640], dtype='float32', name='image')]

# 匯出為 ONNX（使用較新的 opset）
paddle.onnx.export(
    model,
    "model_fixed.onnx",
    input_spec=input_spec,
    opset_version=13,  # 使用支援良好的版本
    enable_onnx_checker=True,
)
```

---

## 範例三：自定義輸出格式

### 問題描述

您的模型輸出格式與標準 YOLO 不同，需要自定義後處理器。

**常見情況**：
- 輸出 shape 不同（例如 `[1, 25200, 85]` vs `[1, 84, 8400]`）
- 座標格式不同（`xyxy` vs `xywh` vs `cxcywh`）
- 信心度計算方式不同

### 診斷過程

```python
import ivit
import numpy as np

model = ivit.load("custom_detector.onnx")

# 檢查輸出格式
dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
outputs = model.infer_raw({"images": dummy_input})

for name, output in outputs.items():
    print(f"{name}:")
    print(f"  shape: {output.shape}")
    print(f"  範例值: {output[0, :5, :5]}")  # 查看部分數據
```

### 解決方案：自定義後處理器

```python
import ivit
from ivit.core.processors import BasePostProcessor, register_postprocessor
from ivit.core.result import Results
from ivit.core.types import Detection, BBox
import numpy as np


class CustomDetectorPostProcessor(BasePostProcessor):
    """
    自定義偵測後處理器

    假設模型輸出格式：
    - shape: [1, num_boxes, 6]
    - 每個 box: [cx, cy, w, h, confidence, class_id]
    """

    def __init__(
        self,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        class_names: list = None,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or [f"class_{i}" for i in range(80)]

    def __call__(self, outputs: dict, original_shape: tuple) -> Results:
        results = Results()
        results.image_size = original_shape[:2]

        # 取得模型輸出
        output = list(outputs.values())[0]  # [1, num_boxes, 6]
        predictions = output[0]  # [num_boxes, 6]

        # 過濾低信心度
        mask = predictions[:, 4] >= self.conf_threshold
        predictions = predictions[mask]

        if len(predictions) == 0:
            return results

        # 轉換座標格式 (cxcywh → xyxy)
        boxes = self._cxcywh_to_xyxy(predictions[:, :4])
        scores = predictions[:, 4]
        class_ids = predictions[:, 5].astype(int)

        # 縮放到原始圖像尺寸
        h, w = original_shape[:2]
        scale_x = w / 640  # 假設模型輸入是 640x640
        scale_y = h / 640

        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        # NMS
        keep_indices = self._nms(boxes, scores, self.iou_threshold)

        # 建立 Detection 物件
        for idx in keep_indices:
            det = Detection(
                bbox=BBox(
                    x1=float(boxes[idx, 0]),
                    y1=float(boxes[idx, 1]),
                    x2=float(boxes[idx, 2]),
                    y2=float(boxes[idx, 3]),
                ),
                class_id=int(class_ids[idx]),
                label=self.class_names[int(class_ids[idx])],
                confidence=float(scores[idx]),
            )
            results.detections.append(det)

        return results

    def _cxcywh_to_xyxy(self, boxes):
        """中心點格式轉角點格式"""
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.stack([x1, y1, x2, y2], axis=1)

    def _nms(self, boxes, scores, iou_threshold):
        """非極大值抑制"""
        # 簡化版 NMS
        indices = np.argsort(scores)[::-1]
        keep = []

        while len(indices) > 0:
            current = indices[0]
            keep.append(current)

            if len(indices) == 1:
                break

            # 計算 IoU
            ious = self._compute_iou(boxes[current], boxes[indices[1:]])

            # 保留 IoU 小於閾值的
            mask = ious < iou_threshold
            indices = indices[1:][mask]

        return keep

    def _compute_iou(self, box, boxes):
        """計算 IoU"""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union = area1 + area2 - inter

        return inter / (union + 1e-6)


# 使用自定義後處理器
register_postprocessor("custom_detector", CustomDetectorPostProcessor)

model = ivit.load("custom_detector.onnx")
model.set_postprocessor(CustomDetectorPostProcessor(
    conf_threshold=0.5,
    iou_threshold=0.45,
    class_names=["person", "car", "bike", "dog", "cat"],  # 您的類別
))

# 推論
results = model("test.jpg")
for det in results:
    print(f"{det.label}: {det.confidence:.2%}")
```

---

## 偵錯技巧

### 1. 比較輸出值範圍

```python
import numpy as np

# 正確的 YOLO 輸出通常：
# - 座標值在 0-640 範圍（或 0-1 正規化）
# - 信心度在 0-1 範圍
# - 類別機率在 0-1 範圍

output = model.infer_raw(inputs)["output"]

print(f"座標範圍: {output[..., :4].min():.2f} ~ {output[..., :4].max():.2f}")
print(f"信心度範圍: {output[..., 4].min():.2f} ~ {output[..., 4].max():.2f}")
```

### 2. 視覺化中間結果

```python
import cv2
import numpy as np

def visualize_raw_detections(image, boxes, scores, threshold=0.5):
    """視覺化原始偵測框（偵錯用）"""
    vis = image.copy()

    for i, (box, score) in enumerate(zip(boxes, scores)):
        if score < threshold:
            continue

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{score:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Debug", vis)
    cv2.waitKey(0)
```

### 3. 逐步驗證流程

```python
import ivit
import numpy as np
import cv2

# 步驟 1：載入圖像
image = cv2.imread("test.jpg")
print(f"原始圖像: {image.shape}, dtype={image.dtype}")

# 步驟 2：手動前處理
preprocessor = model._preprocessor
processed = preprocessor(image)
print(f"前處理後: {processed.shape}, dtype={processed.dtype}")
print(f"值範圍: {processed.min():.4f} ~ {processed.max():.4f}")

# 步驟 3：原始推論
input_name = model.input_info[0]['name']
outputs = model.infer_raw({input_name: processed})
for name, out in outputs.items():
    print(f"輸出 {name}: {out.shape}")

# 步驟 4：手動後處理
postprocessor = model._postprocessor
results = postprocessor(outputs, image.shape)
print(f"偵測數量: {len(results)}")
```

### 4. 與參考實作比較

```python
# 如果有原始框架的推論程式碼，可以比較輸出

# 原始框架推論
# reference_output = original_model.infer(image)

# iVIT 推論
ivit_output = model.infer_raw(inputs)

# 比較差異
# diff = np.abs(reference_output - ivit_output).max()
# print(f"最大差異: {diff}")
```

---

## 最佳實務

### 1. 模型匯出建議

```python
# PyTorch 匯出 ONNX 建議設定
import torch

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=17,           # 使用較新的 opset
    input_names=["images"],     # 明確命名
    output_names=["output"],
    dynamic_axes={              # 支援動態 batch
        "images": {0: "batch"},
        "output": {0: "batch"},
    },
    do_constant_folding=True,   # 優化常數
)
```

### 2. 驗證匯出的模型

```bash
# 使用 onnx 檢查模型
python -c "import onnx; onnx.checker.check_model('model.onnx')"
```

### 3. 建立測試案例

```python
import numpy as np
import json

def create_test_case(model_path, image_path, expected_detections):
    """建立測試案例供日後驗證"""
    test_case = {
        "model": model_path,
        "image": image_path,
        "expected": expected_detections,
    }

    with open("test_case.json", "w") as f:
        json.dump(test_case, f, indent=2)

def verify_test_case(test_case_path):
    """驗證測試案例"""
    import ivit

    with open(test_case_path) as f:
        test_case = json.load(f)

    model = ivit.load(test_case["model"])
    results = model(test_case["image"])

    # 比較結果
    actual = len(results)
    expected = test_case["expected"]["count"]

    print(f"預期偵測數: {expected}, 實際偵測數: {actual}")
    assert abs(actual - expected) <= 2, "偵測數量差異過大"
```

### 4. 文件化您的處理器

```python
class MyCustomPreProcessor(BasePreProcessor):
    """
    自定義前處理器 for [模型名稱]

    模型來源: [來源說明]
    訓練框架: [PyTorch/TensorFlow/PaddlePaddle]

    預期輸入:
        - 格式: BGR
        - 尺寸: 任意（會縮放到 640x640）
        - 類型: uint8

    輸出:
        - 格式: NCHW
        - 尺寸: [1, 3, 640, 640]
        - 類型: float32
        - 正規化: /255.0

    注意事項:
        - 此模型使用 Letterbox 縮放
        - 填充值為 114
    """
    pass
```

---

## 總結

| 問題類型 | 診斷方法 | 解決方案 |
|----------|----------|----------|
| 載入失敗 | 檢查錯誤訊息中的運算子 | ONNX Simplifier / 重新匯出 |
| 結果全錯 | 比較前處理輸出值範圍 | 自定義前處理器 |
| 無偵測結果 | 檢查原始輸出值 | 自定義後處理器 / 調整閾值 |
| 座標偏移 | 視覺化原始框 | 檢查座標格式轉換 |

**需要協助？**

如果您遇到無法解決的問題，請在 [GitHub Issues](https://github.com/innodisk-mannywang/ivit-sdk/issues) 提交問題，並附上：

1. 模型來源和訓練方式
2. 錯誤訊息完整內容
3. `model.input_info` 和 `model.output_info` 輸出
4. 原始推論輸出的形狀和值範圍

---

> **返回** [User Guide](../user-guide.md)
