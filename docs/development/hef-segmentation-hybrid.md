# HEF Segmentation — Hailo + CPU Hybrid Pipeline

**日期**：2026-04-24
**狀態**：已驗證（DeepLabV3-MobileNetV3 on Hailo-8 + OpenVINO CPU head, 15.2 FPS @ 520×520）
**相關 bug**：[`hailo-compile-bug-log.md`](./hailo-compile-bug-log.md) Bug #10、Bug #11

---

## 1. 背景 — 為什麼需要 hybrid

目標：把 torchvision 的 `deeplabv3_mobilenet_v3_large`（520×520, COCO → PASCAL VOC 21 class）部署到 Hailo-8。

直接編 `deeplabv3-mobilenetv3.onnx` 會連續撞兩個問題：

| 問題 | 原因 | 詳見 |
|------|------|------|
| `Shift delta ... larger than 2` | MobileNetV3 SE block 的 9 個 GlobalAveragePool spatial 太大（65×65 / 33×33），int8 accumulator shift 爆掉 | Bug #10 |
| `Compilation failed with unexpected crash` @ `AddShortcut` assertion | Hailo compiler 對 MobileNetV3 密集 inverted-residual shortcut + DeepLab ASPP 多分支的 multi-context 分配爆自家 C++ assertion | Bug #11 |

**關鍵洞察**：Bug #11 是 compiler 內部 bug，不是使用者能改 model script 避開的。唯一穩定的路徑是**砍掉問題區域**，讓 Hailo 只跑 backbone，其餘交給 CPU。

> 這是 YOLO 「拆 concat」（Bug #3）的自然延伸 — 不只尾端肥 concat 要切，任何讓 compiler 分區爆炸的子圖都用 `end_node_names` 砍出去。

---

## 2. 架構全景

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Host (C++ Segmentor)                                                      │
│                                                                           │
│  cv::Mat image (BGR)                                                      │
│       │                                                                   │
│       ▼                                                                   │
│  preprocess()                                                             │
│    resize(520×520) + BGR→RGB + /255                                       │
│    (Hailo backend: skip ImageNet mean/std — baked into HAR, Bug #8)       │
│       │                                                                   │
│       ▼                                                                   │
│  float32 NCHW [1, 3, 520, 520], range [0, 1]                              │
│       │                                                                   │
└───────┼───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ HailoRT runtime → Hailo-8 chip                                            │
│                                                                           │
│  hailort_runtime.cpp :: infer()                                           │
│    float32 [0,1] * 255 → uint8 [0,255]  (Bug #8 round-trip OK)            │
│       │                                                                   │
│       ▼                                                                   │
│  Chip input:  UINT8 NHWC [520, 520, 3]                                    │
│  Baked normalize: (x − ImageNet mean)/std  in [0,1] scale  (Bug #9)       │
│       │                                                                   │
│       ▼                                                                   │
│  deeplabv3_mbv3.hef  (4 contexts, 8 clusters)                             │
│    backbone.0 → … → backbone.16                                           │
│    with TSE-free SE blocks pre-reduced via global_avgpool_reduction       │
│       │                                                                   │
│       ▼                                                                   │
│  Output: FLOAT32 NHWC [33, 33, 960]   ← truncated at backbone.16.2.Mul    │
└───────┼───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Host (C++ Segmentor::postprocess_hailo)                                   │
│                                                                           │
│  NHWC [33, 33, 960] → transpose → NCHW [1, 960, 33, 33]                   │
│       │                                                                   │
│       ▼                                                                   │
│  cpu_head_->infer_raw()    ← ModelManager 載入 deeplabv3_mbv3_head.onnx   │
│                              via OpenVINORuntime (device=cpu)             │
│                                                                           │
│    deeplabv3_mbv3_head.onnx (37 nodes) =                                  │
│       ASPP (4 ASPPConv + 1 ASPPPooling)                                   │
│       → project Conv(1280→256) + ReLU + Dropout                           │
│       → Conv(256→256, 3×3) + BN + ReLU                                    │
│       → Conv(256→21, 1×1)                                                 │
│       → bilinear Resize(520×520)                                          │
│       │                                                                   │
│       ▼                                                                   │
│  FLOAT32 NCHW [1, 21, 520, 520]                                           │
│       │                                                                   │
│       ▼                                                                   │
│  argmax over channel axis → CV_32S mask [520, 520]                        │
│       │                                                                   │
│       ▼                                                                   │
│  cv::resize(INTER_NEAREST) → mask at orig_size                            │
│       │                                                                   │
│       ▼                                                                   │
│  Results { segmentation_mask }                                            │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 檔案命名約定

| 檔案 | 作用 | 產生方式 |
|------|------|----------|
| `deeplabv3-mobilenetv3.onnx` | 完整 ONNX（torchvision export） | 預先 export |
| `deeplabv3_mbv3.hef` | Hailo 的 **backbone-only HEF** | `compile_deeplabv3.py --end-backbone-only` |
| `deeplabv3_mbv3_head.onnx` | CPU 的 **classifier head** | `export_deeplab_classifier.py` |
| `deeplabv3_mbv3.har` | Hailo 中間物（translated + quantized） | compile 過程產生 |

**約定**：CPU head 的檔名必須是 `<主 model 檔名 stem>_head.onnx`，放在主 model 同目錄。`Segmentor` 建構時會自動偵測載入。

---

## 4. 階段一：Hailo HEF 編譯

### 指令

```bash
python compile_deeplabv3.py \
  --onnx deeplabv3-mobilenetv3.onnx \
  --hef deeplabv3_mbv3.hef \
  --har deeplabv3_mbv3.har \
  --model-name deeplabv3_mbv3 \
  --mbv3-gap-fix \
  --end-backbone-only
```

### 三個 flag 的關鍵作用

| Flag | 產生的 model script | 解決的問題 |
|------|--------------------|-----------|
| (預設 bake normalize) | `normalization1 = normalization([0.485, ...], [0.229, ...])` | Bug #9 — normalize 在晶片內用 [0,1] scale 做 |
| `--mbv3-gap-fix` | 兩條 `pre_quantization_optimization(global_avgpool_reduction, ...)`：avgpool1-3 用 `[5, 5]`（65 = 5×13）、avgpool4-8 用 `[3, 3]`（33 = 3×11） | Bug #10 — GAP shift delta > 2 |
| `--end-backbone-only` | `translate_onnx_model(..., end_node_names=["/m/backbone/backbone.16/backbone.16.2/Mul"])` | Bug #11 — 砍 classifier 避開 compiler AddShortcut assertion |

### 產物檢查

```
Successful Mapping (allocation time: 20m 43s)
Compiling kernels of deeplabv3_mbv3_context_0..3...
Successful Compilation (compilation time: 12s)

Wrote deeplabv3_mbv3.hef (8.1 MB)
```

- 4 contexts，8 clusters，`cluster_4` memory 92.2% 為最緊
- 砍掉 classifier 後 `Building optimization options` 從 9 分鐘 → 3 秒（複雜度下降的實證）

---

## 5. 階段二：CPU Head ONNX 匯出

### 指令

```bash
python export_deeplab_classifier.py \
  --onnx deeplabv3-mobilenetv3.onnx \
  --out deeplabv3_mbv3_head.onnx
```

### 核心邏輯

`export_deeplab_classifier.py` 用 `onnx.utils.extract_model` 從原 ONNX **切子圖**，不重建 model、不下載 torchvision pretrained — 這保證 CPU head 的權重跟 Hailo backbone **同源**（沒有版本漂移）。

```python
onnx.utils.extract_model(
    "deeplabv3-mobilenetv3.onnx",
    "deeplabv3_mbv3_head.onnx",
    input_names=["/m/backbone/backbone.16/backbone.16.2/Mul_output_0"],
    output_names=["output"],
)
```

### 產物接口

```
Input : /m/backbone/backbone.16/backbone.16.2/Mul_output_0   [1, 960, 33, 33]   ← 對接 Hailo HEF 輸出
Output: output                                                [1, 21, 520, 520]  ← 直接出 mask 尺寸
37 nodes：ASPP (5 branches + concat + project) + 2 conv + final 1×1 + Resize
```

---

## 6. 階段三：Segmentor Runtime 流程

### 載入

```cpp
Segmentor::Segmentor(const std::string& model_path, const std::string& device, const LoadConfig& config) {
    model_ = ModelManager::instance().load(model_path, load_config);
    // ... input_size_, labels_, colormap_ ...

    // 自動偵測 CPU head
    if (model_->backend() == BackendType::HailoRT) {
        fs::path head_path = fs::path(model_path).parent_path()
                           / (fs::path(model_path).stem().string() + "_head.onnx");
        if (fs::exists(head_path)) {
            LoadConfig head_cfg{.device = "cpu", .task = "segmentation"};
            cpu_head_ = ModelManager::instance().load(head_path.string(), head_cfg);
        }
    }
}
```

### Inference 分流

```cpp
Results Segmentor::postprocess(...) {
    if (model_->backend() == BackendType::HailoRT) {
        return postprocess_hailo(outputs, orig_size);   // ← Hailo 專屬分支
    }
    // ... 舊的 NCHW argmax path ...
}
```

### `postprocess_hailo` 三種狀態

| Hailo 輸出 shape | `cpu_head_` | 行為 |
|-------------------|-------------|------|
| `[H, W, num_classes]` | — | 直接 NHWC argmax（完整 segmentation HEF 的情境）|
| `[H, W, C]` with `C != num_classes` | 存在 | NHWC → NCHW transpose → `cpu_head_->infer_raw()` → NCHW argmax → resize |
| `[H, W, C]` with `C != num_classes` | 不存在 | `throw InferenceError` — 指向 `export_deeplab_classifier.py` 和 Bug #11 |

### NHWC → NCHW Transpose

```cpp
// Hailo emits [H, W, C]; CPU head (ONNX) wants NCHW [1, C, H, W]
float* dst = head_input.data_ptr<float>();
for (int c = 0; c < C; c++) {
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            dst[c * H * W + y * W + x] = data[(y * W + x) * C + c];
        }
    }
}
```

---

## 7. 資料流形狀 / Layout / 型別 cheat sheet

| 位置 | Shape | Layout | dtype | Range |
|------|-------|--------|-------|-------|
| 原圖（OpenCV 讀入） | `[H_orig, W_orig, 3]` | HWC | uint8 | [0, 255] BGR |
| preprocess 後 | `[3, 520, 520]` | NCHW | float32 | [0, 1] RGB, 無 ImageNet normalize |
| Hailo runtime 餵給晶片 | `[520, 520, 3]` | NHWC | uint8 | [0, 255] |
| 晶片 baked normalize 後 | `[520, 520, 3]` | NHWC | int8 (quantized) | ~[-2.1, 2.6] 的量化值 |
| Hailo 輸出（backbone.16.2.Mul） | `[33, 33, 960]` | NHWC | float32（`set_format_type`）| — |
| 轉 head input | `[1, 960, 33, 33]` | NCHW | float32 | 同上 |
| head 輸出（logits） | `[1, 21, 520, 520]` | NCHW | float32 | logits |
| argmax 後 mask | `[520, 520]` | HW | int32 (CV_32S) | [0, 20] class ID |
| resize 回原圖 | `[H_orig, W_orig]` | HW | int32 | [0, 20] |

---

## 8. 檔案改動清單

| 檔案 | 狀態 | 摘要 |
|------|------|------|
| `docs/development/hailo-compile-bug-log.md` | 修改 | 新增 Bug #10（GAP shift delta + `global_avgpool_reduction` 解法）、Bug #11（AddShortcut assertion + end_node 截斷解法） |
| `docs/development/hef-segmentation-hybrid.md` | **新增** | 本文件 — 完整 hybrid pipeline 說明 |
| `compile_deeplabv3.py` | 修改 | 新增 `--mbv3-gap-fix`、`--end-backbone-only`、`--perf-max` flag |
| `export_deeplab_classifier.py` | **新增** | 用 `onnx.utils.extract_model` 切 classifier 子圖 |
| `include/ivit/vision/segmentor.hpp` | 修改 | 加 `cpu_head_` member、`postprocess_hailo()` declaration |
| `src/vision/segmentor.cpp` | 修改 | 加 `<sstream>`、`<limits>` include；Hailo backend 偵測；`cpu_head_` auto-load；NHWC→NCHW transpose；head inference；三狀態分流 |
| `build/CMakeCache.txt` | 修改 | `IVIT_USE_HAILORT=ON` |

`src/runtime/hailort_runtime.cpp` 這次**沒動** — Bug #7、#8、#9 的修法已涵蓋 normalize/dtype 需求，segmentor 直接復用。

---

## 9. 執行範例

```bash
conda run -n forhailo --no-capture-output \
  python examples/python/02_segmentation.py \
  -m deeplabv3_mbv3.hef \
  -i examples/data/images/bus.jpg \
  -d hailo:0
```

預期輸出（已驗證）：

```
Using device: hailo:0
Loading model: deeplabv3_mbv3.hef
Model loaded in 83.4 ms
  Input size: (520, 520)
  Classes: 21

Image: bus.jpg (810x1080)
Running inference...

Results:
  Inference time: 65.72 ms (15.2 FPS)
  Device: hailo:0
  Mask size: 810x1080
  Classes found: 3
    - background: 57.7%
    - bus: 31.3%
    - person: 11.0%
```

FPS 包含 Hailo backbone + CPU head 全部流程。CPU head 的 ASPP 4 條 dilation conv 是 520×520 尺度的 6.6M params 主瓶頸。

---

## 10. 建置注意事項

### ABI / Python 版本一致

`_ivit_core.cpython-<X>-*.so` 必須跟 `libivit.so` **在同一輪 build 產生**，否則 class layout 不符（例如我們剛新增的 `cpu_head_` member），執行時會 `malloc_consolidate(): invalid chunk size` heap corruption。

切 env 時：

```bash
cd build
cmake .. -DPython_EXECUTABLE=$(conda run -n <env> which python | tail -1)
cmake --build . -j$(nproc)
```

跨多個 Python 版本需要分開 build dir（cmake 一次只能綁一個 `Python_EXECUTABLE`）。

### CMake flag

```bash
cmake .. \
  -DIVIT_USE_OPENVINO=ON \       # CPU head 走這個 runtime
  -DIVIT_USE_HAILORT=ON \        # 本次新增
  -DCMAKE_BUILD_TYPE=Release
```

`IVIT_USE_HAILORT` 預設 OFF — CMakeCache.txt 要手動打開。

### ABI flag

`_GLIBCXX_USE_CXX11_ABI=1` 由 `CMakeLists.txt` 強制設定（pip 裝的 OpenVINO 是 ABI=0，但系統 OpenCV 是 ABI=1，以 OpenCV 為準）。

---

## 11. 已知限制與後續改進

| 項目 | 現況 | 改進方向 |
|------|------|---------|
| Calibration 品質 | 128 張 COCO val → `optimization_level=0`（無 GPU） | 1024+ 張 domain-matched 資料 + 有 GPU 開 `opt_level=1+`（見 Bug #13）|
| Input size 固定 | 520×520 baked 進 HEF 和 head ONNX | 如需改尺寸兩者都要重新 export / compile |
| Classifier head `task="segmentation"` 繞 label loading | `Segmentor` 建構時會再嘗試幫 head 載 labels（多餘） | 給 head 一個 hint 跳過 labels / colormap 的後續初始化 |
| `postprocess_hailo` 的 NHWC→NCHW transpose 是 naive triple loop | 960 × 33 × 33 ≈ 1M float copy，現況可忽略 | 可用 `cv::transpose` + `cv::merge` 或 SIMD |
| CPU head 用 OpenVINO | 走 `ModelManager` default openvino runtime | 如需 GPU 可改用 TensorRT runtime 載同一 ONNX |
| Batch size | 僅支援 batch=1（HEF 編譯時固定） | HEF 重編 + head ONNX 用 dynamic batch |
| 第二個 `Segmentor` constructor（接現成 `shared_ptr<Model>`）不做 auto-load | 使用者自帶 model 時只能得到 backbone feature map | 若需 hybrid 支援，可加 overload 接受 `cpu_head` 參數 |

---

## 12. 除錯 / 診斷指令

### 確認 Hailo backend 有編進去

```bash
grep IVIT_USE_HAILORT build/CMakeCache.txt
# 預期: IVIT_USE_HAILORT:BOOL=ON
```

### 列出 HEF 裡的 avgpool（編譯前診斷 SE block 數量）

```python
from hailo_sdk_client import ClientRunner
r = ClientRunner(hw_arch="hailo8"); r.load_har("deeplabv3_mbv3.har")
hn = r.get_hn()
import json; hn_dict = json.loads(hn) if isinstance(hn, str) else hn
for name, cfg in hn_dict.get("layers", {}).items():
    if cfg.get("type") == "avgpool":
        print(name, cfg.get("input_shapes"))
```

### head ONNX 接口確認

```bash
python -c "
import onnx; m = onnx.load('deeplabv3_mbv3_head.onnx')
for x in m.graph.input:  print('in :', x.name, [d.dim_value for d in x.type.tensor_type.shape.dim])
for x in m.graph.output: print('out:', x.name, [d.dim_value for d in x.type.tensor_type.shape.dim])
"
```

### 背景追 compile log 而不阻塞

```bash
conda run -n forhailo --no-capture-output python compile_deeplabv3.py ... > /tmp/compile.log 2>&1 &
until grep -qE "(Wrote .*\.hef|AccelerasNumerizationError|Traceback|unexpected crash)" /tmp/compile.log; do
    sleep 10
done
tail -40 /tmp/compile.log | tr '\r' '\n'
```

每輪 compile 失敗要 10+ 分鐘才看得到結果，不要盯 terminal。

---

## 13. 教訓

1. **Hailo compiler 的 multi-context 分配器對 inverted-residual + 多分支 head 組合會爆 C++ assertion**。遇到 `Compilation failed with unexpected crash` → 第一反應是精簡拓撲（`end_node_names` 切尾巴），不是調 model script。
2. **「量化失敗」和「compile 失敗」是兩層問題**。MobileNetV3 兩層都撞到；`global_avgpool_reduction` 只解量化，`end_node_names` 只解 compile。兩者正交，必須同時處理。
3. **從 ONNX 切子圖當 CPU head 優於重建 PyTorch model**。`onnx.utils.extract_model` 能保證 head 跟 backbone 的權重完全同源，不會有 torchvision 版本 / 訓練 script 差異造成的漂移。
4. **檔名約定勝過 API 膨脹**。Segmentor 自動找 `<stem>_head.onnx` 而不是加 constructor 參數，舊的 OpenVINO/TensorRT 使用者一行都不用改。
5. **Python ABI mismatch 的 heap corruption 非常容易誤判為邏輯 bug**。切 env 時永遠先 reconfigure cmake 的 `Python_EXECUTABLE`，再跑 demo。
