# HailoRT Detection 推論流程與 NMS 解碼說明

> 對應範例：`examples/python/02_detection.py`
> 對應後端：`src/runtime/hailort_runtime.cpp`、`src/vision/detector.cpp`

---

## 1. 執行入口：`02_detection.py`

[examples/python/02_detection.py](../../examples/python/02_detection.py)

```
Python script
  ↓ argparse (model, device, image, conf, iou)
  ↓ Detector(model_path, device="hailo:0")
  ↓ detector.predict(image, conf_threshold, iou_threshold)
  ↓ Results → visualize / save JSON
```

### 關鍵參數

| 參數 | 預設 | 說明 |
|------|------|------|
| `--model` / `-m` | 必填 | `.hef` 模型路徑 |
| `--device` / `-d` | `auto` | `hailo`, `hailo:0` |
| `--conf` | `0.5` | Confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold（Hailo on-chip NMS 已完成，此值為 CPU 二次過濾用） |
| `--benchmark` / `-b` | false | 開啟 benchmark 模式（10 次 warm-up + N 次計時） |

---

## 2. 分層呼叫路徑

```
02_detection.py
  └─ Detector.predict(image, conf, iou)          [detector.cpp:166]
       ├─ preprocess(image)                       [detector.cpp:258]
       │    ├─ letterbox resize (scale + padding)
       │    ├─ RGB normalization [0,1]
       │    └─ HWC → NCHW float32
       ├─ model_->infer_raw(inputs)               [model.cpp → IRuntime]
       │    └─ HailoRTRuntime::infer()            [hailort_runtime.cpp:182]
       └─ postprocess(outputs, ...)               [detector.cpp:321]
            ├─ 判斷 tensor.layout() == HailoNMS?
            │    YES → postprocess_hailo_nms()    [detector.cpp:528]
            │    NO  → postprocess_yolo()         [detector.cpp:350]
            └─ Results
```

---

## 3. HailoRT Runtime 推論

### 3.1 模型載入（`load_model`）

[src/runtime/hailort_runtime.cpp:67](../../src/runtime/hailort_runtime.cpp#L67)

```
VDevice::create()
  └─ vdevice->create_infer_model(path)       # 讀取 .hef
       └─ infer_model->configure()           # → ConfiguredInferModel
```

**Output tensor 識別（`make_tensor_info`）**：[hailort_runtime.cpp:354](../../src/runtime/hailort_runtime.cpp#L354)

```cpp
if (stream.format().order == HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS) {
    // On-chip NMS 輸出
    // shape = {num_classes, 1 + max_bboxes_per_class * 5}
    info.layout = Layout::HailoNMS;
    info.dtype  = DataType::Float32;
} else {
    // 一般 feature map 輸出
    info.layout = Layout::NHWC;
}
```

### 3.2 推論執行（`infer`）

[src/runtime/hailort_runtime.cpp:182](../../src/runtime/hailort_runtime.cpp#L182)

#### 輸入轉換（NCHW float32 → NHWC uint8）

Vision Layer 送進來的是 `NCHW float32 [0,1]`，但 Hailo-8 晶片需要 `NHWC uint8 [0,255]`：

```cpp
// hailort_runtime.cpp:215-229
if (hailo_dtype == DataType::UInt8 && tensor.dtype() == DataType::Float32) {
    // NCHW float32 → NHWC uint8
    for h, w, c:
        dst[h*W*C + w*C + c] = uint8(src[c*H*W + h*W + w] * 255)
}
```

#### 輸出收集

```cpp
// hailort_runtime.cpp:277-296
if (info.layout == Layout::HailoNMS) {
    t = Tensor(info.shape, DataType::Float32);
    t.set_layout(Layout::HailoNMS);   // ← 標記給 postprocess 識別
} else {
    t = Tensor({frame_size}, DataType::UInt8);
}
memcpy(t.data(), buf.data(), buf.size());
```

---

## 4. NMS 解碼：`postprocess_hailo_nms`

[src/vision/detector.cpp:528](../../src/vision/detector.cpp#L528)

### 4.1 Hailo NMS 輸出格式

HailoRT 使用 `HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS`，輸出記憶體佈局：

```
shape: [num_classes, 1 + max_bboxes_per_class * 5]

每個 class 的資料：
┌─────────────┬──────────────────────────────────────────────────┐
│ bbox_count  │ y1, x1, y2, x2, score │ y1, x1, y2, x2, score │ …│
│  (float32)  │     bbox #0 (5 floats)│     bbox #1 (5 floats) │  │
└─────────────┴──────────────────────────────────────────────────┘
```

- **NMS 已在晶片上完成**，CPU 端不再做第二次 NMS
- 座標為**正規化值** `[0, 1]`，相對於 model input size（例如 640×640）
- 順序為 `y1, x1, y2, x2`（注意 yx 順序）

### 4.2 解碼邏輯

[src/vision/detector.cpp:552-603](../../src/vision/detector.cpp#L552)

```cpp
for cls in range(num_classes):
    cls_ptr = fdata + cls * floats_per_class
    count = int(cls_ptr[0])                 // bbox 數量

    for bi in range(count):
        bbox = cls_ptr + 1 + bi * 5
        y1, x1, y2, x2 = bbox[0..3]        // 正規化座標
        score           = bbox[4]

        if score < conf_threshold: skip

        // 正規化 → 像素 → 去除 letterbox → 原圖座標
        ox1 = clamp((x1 * input_w - pad_w) / scale, 0, orig_w)
        oy1 = clamp((y1 * input_h - pad_h) / scale, 0, orig_h)
        ox2 = clamp((x2 * input_w - pad_w) / scale, 0, orig_w)
        oy2 = clamp((y2 * input_h - pad_h) / scale, 0, orig_h)

        Detection(BBox(ox1, oy1, ox2, oy2), cls, score, label)

// 依 confidence 排序，裁切至 max_detections
// ← 不做 CPU NMS（on-chip 已完成）
```

### 4.3 座標轉換公式

```
pixel_coord = normalized_coord × input_size
orig_coord  = (pixel_coord − pad) / scale
```

| 變數 | 說明 |
|------|------|
| `scale` | letterbox 縮放比例（`min(W/orig_w, H/orig_h)`） |
| `pad_w` / `pad_h` | letterbox 左側/上側填充像素數 |
| `input_w` / `input_h` | 模型輸入尺寸（通常 640×640） |

---

## 5. 一般 YOLO NMS（非 Hailo 路徑）

對比：[src/vision/detector.cpp:606](../../src/vision/detector.cpp#L606)

Hailo 路徑 **不走** 此函式。一般後端（OpenVINO / TensorRT）的後處理包含：

1. 信心度過濾（`obj_conf × class_score < conf_threshold`）
2. 格式解碼（YOLOX grid decode）
3. Center format → Corner format
4. **CPU NMS**：Greedy IoU suppression，per-class

---

## 6. 關鍵檔案總覽

| 檔案 | 功能 |
|------|------|
| [examples/python/02_detection.py](../../examples/python/02_detection.py) | Python 範例入口 |
| [include/ivit/vision/detector.hpp](../../include/ivit/vision/detector.hpp) | Detector class 宣告 |
| [src/vision/detector.cpp](../../src/vision/detector.cpp) | preprocess、postprocess、NMS |
| [include/ivit/runtime/hailort_runtime.hpp](../../include/ivit/runtime/hailort_runtime.hpp) | `HailoRTModel` struct、HailoRT API 宣告 |
| [src/runtime/hailort_runtime.cpp](../../src/runtime/hailort_runtime.cpp) | 推論實作、format 轉換、NMS tensor 識別 |

---

## 7. Hailo vs 其他後端差異摘要

| 項目 | Hailo-8（HailoRT） | OpenVINO / TensorRT |
|------|-------------------|---------------------|
| 輸入格式 | NHWC uint8 | NCHW float32 |
| 輸入轉換 | Runtime 自動 NCHW→NHWC + float→uint8 | 無需轉換 |
| NMS | **On-chip（晶片內完成）** | CPU 端 Greedy NMS |
| 輸出座標 | 正規化 [0,1]，yx 順序 | 像素絕對值，xy 順序 |
| Layout tag | `Layout::HailoNMS` | N/A |
| 模型格式 | `.hef`（Hailo Executable Format） | `.xml/.bin` / `.engine` |
