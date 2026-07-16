# YOLOX 訓練接 iVIT 推論 Bug 紀錄

記錄把 YOLOX 訓練（`ivit.train.DetectionTrainer` 走 yolox 後端）一路接到 iVIT C++ 推論層
（`ivit predict` / `ivit.load(...).predict()`）過程中踩到的 5 個獨立 bug 與修法。

問題的初始症狀很簡單：

> **訓練 100 epoch 完成，best AP = 0.01，`ivit predict` 在 bus.jpg 上 0 detections。**

但實際上這個症狀同時混雜了 5 層獨立缺陷（從訓練配置、export 流程、到 C++ 後處理路徑），
本文件按發現順序記錄，方便日後遇到類似問題時可以快速對照。

相關檔案：

- `python/ivit/train/_yolox_backend.py` — YOLOX 訓練 backend
- `python/ivit/train/detection.py` — DetectionTrainer 入口
- `python/ivit/vision/yolox_predictor.py` — YOLOX-aware Python wrapper（最後新增）
- `src/core/model.cpp` — C++ Model::predict 與其 model-agnostic 後處理
- `src/vision/detector.cpp` — yolox-aware 後處理（**目前沒被 Model::predict dispatch 到**）

---

## Bug #1 — 預設從 random init 訓練，永遠 converge 不了

### 症狀

```
2026-05-08 10:32:25.169 | INFO  | yolox.core.trainer:after_train:200
  - Training of experiment is done and the best AP is 0.01
```

100 epoch / 128 張圖 / 從零訓 → AP 0.01，class-wise AP 全部 0.000 或 NaN。

### 根因

`YoloxBackend.fit()` 預設沒帶 `args.ckpt`：

```python
args = _make_yolox_args(
    batch_size=self.batch_size,
    devices=self._device_count(),
    ckpt=self.pretrained_weights,   # 預設 None → 從零訓
    experiment_name=self.model_name,
)
```

YOLOX-nano 的設計就是要在 COCO（118k 張）跑 300 epoch 才會到 AP 25.8。從零開始訓 128 張 ×
100 epoch 算術上**永遠到不了能用的程度**——這不是 bug，是物理限制。

### 修法

`_yolox_backend.py:118-156` 新增 `_auto_pretrained()`：

- 自動從 `ivit.zoo.registry.MODEL_REGISTRY[model_name].url` 取 Megvii 官方 .pth URL
- 下載到 `~/.cache/ivit/pretrained/{model}_coco.pth` 後快取
- 傳給 yolox `Trainer.args.ckpt` 作為 starting point

```python
ckpt_path = self.pretrained_weights or self._auto_pretrained()
```

`fit()` 預設啟用，除非顯式傳 `pretrained_weights=...` 或 user 走 `--no-pretrained` 路徑。

---

## Bug #2 — 模型 layer 名稱與官方 .pth 對不上（depthwise 配置缺失）

### 症狀

接續 Bug #1 修完後，重新訓練 100 epoch 仍然 AP 0.01。檢查 train log 開頭：

```
ckpt='~/.cache/ivit/pretrained/yolox-nano_coco.pth'
INFO     | yolox.core.trainer:resume_train:338 - loading checkpoint for fine tuning
WARNING  | yolox.utils.checkpoint:load_ckpt:16 -
  backbone.backbone.dark2.0.conv.weight is not in the ckpt
WARNING  | yolox.utils.checkpoint:load_ckpt:16 -
  backbone.backbone.dark2.0.bn.weight is not in the ckpt
... (約 200+ 條 WARNING)
```

Validation：

```
matched layers: 0/642  (load 完幾乎全部 random init)
```

### 根因

YOLOX-nano 的官方 Exp **覆寫了 `get_model()`** 加上 `depthwise=True`，這會把 backbone 的
conv 換成 **depthwise separable conv**，layer 名變成 `.dconv.dconv.weight`、`.dconv.bn.weight`
等，與一般 conv 不同。

我原本的實作繼承 base `yolox.exp.Exp` 然後手動覆寫 `depth` / `width`，但**沒有覆寫
`get_model()`**：

```python
# ❌ 錯誤實作
class _IvitExp(YoloxExp):
    def __init__(self):
        super().__init__()
        self.depth = 0.33
        self.width = 0.25
        # ...
        # 缺：self.depthwise = True，且 base Exp 不認這個欄位
```

→ 建出來的是 **non-depthwise nano-size** 模型，layer 名跟官方 depthwise nano 完全對不上 →
`load_ckpt` 把 642 個 layer 的 weight 全部標記為 missing → 等同從零訓 + head 被污染。

### 修法

`_yolox_backend.py:80-117` 改成繼承官方對應變體 Exp 子類：

```python
_YOLOX_OFFICIAL_EXP_MODULES = {
    "yolox-nano": "yolox.exp.default.yolox_nano",
    "yolox-tiny": "yolox.exp.default.yolox_tiny",
    "yolox-s":    "yolox.exp.default.yolox_s",
    "yolox-m":    "yolox.exp.default.yolox_m",
    "yolox-l":    "yolox.exp.default.yolox_l",
}

def _build_exp(self) -> Any:
    module = importlib.import_module(_YOLOX_OFFICIAL_EXP_MODULES[self.model_name])
    official = module.Exp

    class _IvitExp(official):       # ← 繼承官方變體，自帶 get_model() 與 depthwise / aug 設定
        def __init__(self):
            super().__init__()
            # 只覆寫 dataset / 訓練超參，不動 architecture 欄位
            self.num_classes = backend.num_classes
            self.data_dir    = backend.data_dir
            # ...
    return _IvitExp()
```

驗證：

```
matched layers: 642/642  (100.0%)
first 5 missing in ckpt: []
first 5 unused in ckpt: []
```

修完後 30 epoch 直接到 **AP = 52.32**。

---

## Bug #3 — ONNX export 時 GPU 模型 + CPU dummy → device mismatch

### 症狀

訓練成功後跑 `ivit train --export-onnx`：

```
RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)
should be the same or input should be a MKLDNN tensor and weight is a dense tensor
```

trace 在 `yolox/models/network_blocks.py:51`，conv 第一層就掛。

### 根因

`YoloxBackend.export()` 用 `self._exp.get_model()` 拿模型：

| 物件                                     | Device                                                                                  |
| ---------------------------------------- | --------------------------------------------------------------------------------------- |
| `model = self._exp.get_model()`        | **CUDA**（訓練時 yolox `Trainer` 已搬上 GPU，`self._exp.model` 屬性被 cache） |
| `dummy = torch.zeros(1, 3, ...)`       | **CPU**                                                                           |
| `torch.onnx.export(model, dummy, ...)` | trace 觸發 `model.forward(dummy)` → conv 看到 weight=cuda、input=cpu → 爆           |

訓練前手動測試時是用一個剛建構的 `YoloxBackend`（沒呼叫過 `fit()`），`get_model()`
第一次呼叫建在 CPU 上，所以剛好沒事。一旦走完整 `fit() → export()` 流程，model 就被卡在 GPU 上。

### 修法

`_yolox_backend.py:165-167`：export 前強制把 model 搬回 CPU：

```python
ckpt = torch.load(self._best_ckpt, map_location="cpu", weights_only=False)
model.load_state_dict(...)
model = model.cpu().eval()    # ← 確保 trace 走純 CPU 路徑
```

ONNX export 用 CPU trace 是常見做法，比較不會 trace 到 cuda-only 的 op，輸出也比較通用。

---

## Bug #4 — `decode_in_inference` 與 C++ 後處理對不齊

### 症狀

ONNX export 成功，`onnxruntime` 自己跑 obj_score top 5 都是 0.99，但**走 ivit C++ runtime
跑出 0 detections**。

### 根因

YOLOX head 的 forward 結尾：

```python
# yolox/models/yolo_head.py
if self.decode_in_inference:
    return self.decode_outputs(outputs, dtype=...)   # 在 graph 內 grid decode
else:
    return outputs                                   # 純 raw grid-relative 輸出
```

兩種輸出的數值範圍完全不同：

| 模式                                    | cx 範圍          | cy 範圍          | w 範圍            | h 範圍            |
| --------------------------------------- | ---------------- | ---------------- | ----------------- | ----------------- |
| `decode_in_inference=False`（raw）    | -2 ~ 3           | -2 ~ 3           | -0.7 ~ 3.2（log） | -0.7 ~ 3.2（log） |
| `decode_in_inference=True`（decoded） | 0 ~ 416（pixel） | 0 ~ 416（pixel） | 0 ~ 416           | 0 ~ 416           |

iVIT 的 `Detector::postprocess_yolo`（src/vision/detector.cpp:391-446）寫死了 grid decode：

```cpp
cx = (cx + gx) * stride;
cy = (cy + gy) * stride;
w = std::exp(w) * stride;
h = std::exp(h) * stride;
```

→ 它預期 ONNX 是 **raw grid-relative**（與 zoo 預建 yolox-nano.onnx 一致）。
我原本 export 時設 `decode_in_inference=True`（誤以為是「給下游 ready-to-use 的 decoded box」），
結果 C++ 對已 decode 過的座標再 decode 一次：

```
cx_out = (200 + 10) * 8 = 1680   # 飛掉
w_out  = exp(50) * 8             # 直接溢位
```

→ 全部 box 飛到圖外，被 clip 壓回邊界（變 0,0,0,0 之類），加 NMS 後變 0 detections。

### 修法

`_yolox_backend.py:170`：預設改為 `decode_in_inference=False`，與 Megvii 官方
`tools/export_onnx.py` 與 zoo 預建 ONNX 一致：

```python
if hasattr(model, "head"):
    model.head.decode_in_inference = bool(kwargs.pop("decode_in_inference", False))
```

---

## Bug #5 — `Model::predict` 走 model-agnostic 後處理，不認 YOLOX

### 症狀

把 Bug #4 修完後重 export ONNX，**`ivit predict` 仍然 0 detections**。同樣的 ONNX 用
`onnxruntime` 直接跑 + 我們 Python 寫的 yolox decode，會跑出正確的 4 個物件
（1 bus + 3 person）。

更詭異：**zoo 預建的 yolox-nano.onnx**（Megvii 在 COCO 完整訓 300 epoch、AP 25.8）
丟進 `ivit predict` 也是 0 detections。

### 根因

iVIT C++ 層其實有兩條獨立 code path：

```
A. ivit.load(...).predict(image, config)      ← Python 呼叫的就是這條
   └─ Model::predict (src/core/model.cpp:77)
      ├─ Model::preprocess           (model.cpp:228)
      └─ Model::postprocess_detection(model.cpp:360)   ← 「simplified YOLO postprocessor」

B. Detector class                              ← 沒被 A 用到
   └─ Detector::predict
      ├─ Detector::preprocess        (detector.cpp:258)  ← yolox-aware: BGR + no normalize
      └─ Detector::postprocess_yolo  (detector.cpp:391)  ← yolox grid decode
```

`Model::predict`（A 路徑）的 preprocess 是 model-agnostic 通用版，**對 YOLOX 全部錯**：

```cpp
// src/core/model.cpp:228-283 — Model::preprocess
cv::resize(image, resized, target_size);                  // ❌ 沒做 letterbox（直接拉伸）
resized.convertTo(float_img, CV_32F, 1.0 / 255.0);        // ❌ /255 normalize（YOLOX 要 0~255）
cv::cvtColor(float_img, float_img, cv::COLOR_BGR2RGB);    // ❌ 轉 RGB（YOLOX 要 BGR）
```

postprocess 同樣是 model-agnostic 簡化版（`model.cpp:360 postprocess_detection`），假設
cx/cy 是 normalized [0, 1] 直接 `* image_size.width`，沒 grid decode、沒考慮 letterbox padding。

而 `Detector` 那條 yolox-aware 的 code path（`detector.cpp`）——雖然寫好了——
**根本沒被 `Model::predict` dispatch 到**。註解寫著：

```cpp
// model.cpp:368
// This is a simplified YOLO-style postprocessor
// Real implementation would need model-specific handling
```

——也就是這條路徑本來就**沒實作 YOLOX**。

### 修法（Python wrapper，繞過 C++ 缺陷）

完整修法是讓 `Model::predict` 在 task=Detection 時 dispatch 到 `Detector`，需要重 build C++。
作為過渡，`python/ivit/vision/yolox_predictor.py` 新增 `YoloxPredictor`：

```python
class YoloxPredictor:
    def __init__(self, model_path, device="cuda:0", ...):
        self._model = ivit.load(model_path, device=device)
        # 從 ONNX I/O 抓正確 input name 與 input H/W

    def predict(self, image, conf=0.3, iou=0.45):
        x, scale, pad_w, pad_h = self._preprocess(image)   # letterbox + BGR + no normalize
        out = self._model.infer({self._input_name: x})     # 走 C++ TRT/OpenVINO runtime 拿 raw tensor
        return self._postprocess(out, ..., conf, iou)      # YOLOX grid decode + per-class NMS
```

底層仍然走 **C++ TensorRT runtime**（保留硬體加速：FP32 ~3.5 ms / 張），只有 pre/post 在 Python。

驗證：

```
=== 我們訓練的 yolox-nano（30 epochs from COCO pretrained）===
  [0] bus        0.936  [13,221,807,755]
  [1] person     0.887  [52,393,212,910]
  [2] person     0.788  [215,392,346,880]
  [3] person     0.729  [685,478,810,881]
```

### TODO（C++ 端正解）

`Model::predict` 應該根據 `task_` + `model_type_` dispatch 到對應的 `Classifier` /
`Detector` / `Segmentor`，而不是用 `Model::postprocess_detection` 這個 simplified 版本。
這需要 Model class 持有對應的 vision 物件 instance（或 strategy pattern）並在 ctor
依模型名建立。同時 `Model::preprocess` 也要走 task-specific 路徑（YOLOX = BGR + no
normalize；YOLOv5/v8 = RGB + /255）。

---

## 五個 Bug 的相依圖

```
Bug #1（從 random init）─────┐
                              │
Bug #2（depthwise 沒設定）────┼──→ AP=0 → 沒物件
                              │
Bug #3（GPU/CPU mismatch）────┴──→ export 直接掛
                                    ↓
                              成功 export ONNX
                                    ↓
Bug #4（decode 兩次）──────────────→ box 飛到天文數字 → 0 detections
                                    ↓
                              改 decode_in_inference=False
                                    ↓
Bug #5（Model::predict 後處理錯）──→ 仍 0 detections
                                    ↓
                              Python YoloxPredictor 繞過
                                    ↓
                              ✅ 正確偵測
```

每個 bug 都會獨立讓「最終偵測數 = 0」，所以解一個還會繼續看到 0，需要全部解完才會成功。
這是這次除錯比較吃力的原因。

---

## 驗證清單

未來要新增其他 yolox 變體（tiny / s / m / l）或其他 detector 訓練 backend 時，建議跑這套
sanity check：

1. **預訓練 layer match**：

   ```python
   matched / total = 100%；missing/unused 都是空 list
   ```
2. **ONNX export 與 zoo 對齊**：

   ```python
   onnx_input_name == "images"
   onnx_input_shape == [1, 3, H, W]   # static
   onnx_output_shape == [1, num_anchors, 5+num_classes]   # static
   onnx.opset_version == 11
   ```
3. **Raw output 數值範圍**（驗證沒誤 decode）：

   ```python
   cx, cy ∈ ~[-3, 3]      # grid-relative
   w, h  ∈ ~[-1, 4]       # log-scale
   obj_score ∈ [0, 1]     # 已 sigmoid
   ```
4. **PyTorch 原生路徑與 ONNX 推論一致**：

   ```python
   pytorch_pth_inference(bus.jpg, conf=0.3) ≈ yolox_predictor(onnx, bus.jpg, conf=0.3)
   ```
   兩條路徑應該都看到 1 bus + 3-4 person，confidence 在 ±2% 內一致。
5. **C++ runtime raw output 與 onnxruntime 一致**：

   ```python
   ivit.load(onnx).infer(x) ≈ onnxruntime.run(onnx, x)
   ```
