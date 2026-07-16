# HEF Classification — ResNet50 / EfficientNet-B0 / MambaVision-T on Hailo-8

**日期**：2026-04-29
**狀態**：三個 model 都已部署上 Hailo-8 並透過 `Classifier` SDK 端到端驗證
**相關 bug**：[`hailo-compile-bug-log.md`](./hailo-compile-bug-log.md) Bug #7-#15

---

## 1. 三個 ImageNet classifier 的對照表

| Model                     | 部署模式                                        | HEF 大小 | FPS @ Hailo-8 | 編譯需 GPU？                      | 推論需 GPU？ |
| ------------------------- | ----------------------------------------------- | -------- | ------------- | --------------------------------- | ------------ |
| **ResNet50**        | 純 Hailo                                        | ~25 MB   | ~30           | 否                                | **否** |
| **EfficientNet-B0** | 純 Hailo                                        | 9.7 MB   | **~80** | **必要**（QAT 救 collapse） | 否           |
| **MambaVision-T**   | Hybrid (conv on Hailo + SSM/Transformer on CPU) | 6.2 MB   | ~30           | 必要（QAT）                       | 否           |

> 三個 model 都在 PASCAL VOC 21-class 之外的 ImageNet 1000-class domain 跑（標準 Top-K 分類）。

---

## 2. 為什麼三個 model 走不同路線

### ResNet50 — baseline（最 Hailo-friendly）

純 conv + ReLU + BN，無 SE / 無 swish / 無 dynamic shape。直接編、直接跑。詳細編譯流程在 `compile_resnet50.py`。

### EfficientNet-B0 — 純 Hailo 但需要 GPU 編譯

Swish + 16 個 SE block 對 int8 量化敏感。沒 GPU 時 Hailo 強制 `opt_level=0`，把 Bias Correction / Adaround / Finetune Encoding / **Quantization-Aware Fine-Tuning** 全 skip → activation collapse，輸出退化到「網格紋理」cluster。

**唯一的解**：機器要有 NVIDIA GPU，且要 bypass Hailo 內部「GPU memory < 5%」的硬閾值（Bug #14）。

### MambaVision-T — Hybrid

Mamba **selective scan (S6) + Einsum** Hailo Dataflow Compiler 不支援，translate 階段就 reject（`cannot reshape array of size 160 into shape (1,160,196)`，Bug #15）。

官方建議：把純 conv 部分丟 Hailo，SSM/Transformer 留 CPU。我們切在 `levels.1/downsample/reduction/reduction.0/Conv`（最後一個純 conv），Hailo HEF 跑 backbone（1.5 ms），CPU 跑 head（36-50 ms）。

---

## 3. ResNet50

### 編譯

```bash
conda run -n forhailo --no-capture-output \
  python compile_resnet50.py \
  --calib-dir dogs-vs-cats/test1 \
  --calib-size 2048
```

> 不需要 `CUDA_VISIBLE_DEVICES=0` —— ResNet50 對量化 robust，opt_level=0 也能用。

### 推論

```bash
conda run -n forhailo --no-capture-output \
  python examples/python/02_classification.py \
  -m resnet50.hef \
  -i examples/data/images/bus.jpg \
  -d hailo:0
```

預期：top-1 minibus（高信心）。

---

## 4. EfficientNet-B0

### 編譯（必須 GPU）

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n forhailo --no-capture-output \
  python compile_efficientnet_b0.py \
  --calib-dir dogs-vs-cats/test1 \
  --calib-size 2048 \
  --gap-fix
```

`CUDA_VISIBLE_DEVICES=0` 必要 — 繞過 Hailo 內部 5% GPU memory 閾值。`--gap-fix` 啟用 `global_avgpool_reduction` 救 SE block 的 GAP shift delta（Bug #10）。

### 編譯期看到的關鍵 milestone

```
[info] Using default optimization level of 2
[info] Starting Quantization-Aware Fine-Tuning
[info] Model Optimization Algorithm Quantization-Aware Fine-Tuning is done (3m 55s)
[info] Starting Layer Noise Analysis  (57s)
[info] Successful Mapping (allocation time: 8m 47s)
Wrote efficientnet_b0.hef (9.7 MB)
```

QAT 是真正救 collapse 的 step（Bug #14 詳述）。

### Labels 配對

```bash
cp ~/.cache/ivit/models/efficientnet-b0.txt \
   efficientnet_b0.txt
```

`Classifier` 自動偵測同名 `.txt`（連字號 vs 底線無所謂，依 HEF stem name）。

### 推論

```bash
conda run -n forhailo --no-capture-output \
  python examples/python/02_classification.py \
  -m efficientnet_b0.hef \
  -i examples/data/images/bus.jpg \
  -d hailo:0
```

預期：top-1 streetcar / minibus / trolleybus 等車輛 cluster；FPS ~80。

---

## 5. MambaVision-T (Hybrid)

### 編譯流程（兩個 artifact）

#### Step 1: Hailo backbone HEF

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n forhailo --no-capture-output \
  python compile_mambavision_t.py
```

Script 預設：

- `--end-node "/m/model/levels.1/downsample/reduction/reduction.0/Conv"` —— SSM 開始前的最後一個 conv
- `--calib-dir dogs-vs-cats/test1`, `--calib-size 2048`
- 輸出 `mambavision_t.hef` (6.2 MB)、`mambavision_t.har`

關鍵 log：

```
[info] End nodes mapped from original model:
       '/m/model/levels.1/downsample/reduction/reduction.0/Conv'
[info] Quantization-Aware Fine-Tuning  (1m 53s)
[info] Successful Mapping (allocation time: 13s)   ← backbone 很小，分區快
```

#### Step 2: 抽出 CPU head ONNX

```bash
conda run -n forhailo --no-capture-output \
  python extract_mambavision_head.py \
  --onnx ~/.cache/ivit/models/mambavision-t.onnx \
  --out  mambavision_t_head.onnx
```

Script (`extract_mambavision_head.py`) 內部用 `onnx.utils.extract_model` 切 subgraph，input tensor 接 Hailo HEF 的輸出邊界（`/m/model/levels.1/downsample/reduction/reduction.0/Conv_output_0`），output tensor 是原 model 的 `logits`。

Head subgraph：8017 nodes（整個 SSM + Transformer + classifier），input `[1, 320, 14, 14]` NCHW, output `[1, 1000]`。

#### Step 3: Labels（跟 efficientnet 同款）

```bash
cp ~/.cache/ivit/models/mambavision-t.txt \
   mambavision_t.txt
```

### 部署 artifact 配對

```
mambavision_t.hef           6.2 MB    Hailo backbone
mambavision_t_head.onnx     ~120 MB   CPU SSM/Transformer (auto-loaded)
mambavision_t.txt           ImageNet 1000 labels (auto-loaded)
```

### 推論（一行 SDK demo）

```bash
conda run -n forhailo --no-capture-output \
  python examples/python/02_classification.py \
  -m mambavision_t.hef \
  -i examples/data/images/bus.jpg \
  -d hailo:0
```

預期：

```
Model loaded in 3769.7 ms
Classes: 1000
Inference: 32.68 ms / 30.6 FPS
Top-1: minibus 52.48%
```

`Classifier` 在 constructor 偵測 backend == HailoRT + 同目錄有 `<stem>_head.onnx` → 自動載入 cpu_head 模型 → `postprocess_hailo` 自動 transpose + 跑 head + softmax + top-k。

---

## 6. SDK 整合架構

### Classifier / Segmentor 共用同款 hybrid pattern

```
Classifier::Classifier(model_path, ...)
    load main model via ModelManager (HailoRT or OpenVINO/TRT)
    if backend == HailoRT:
        head_path = <model_dir> / "<stem>_head.onnx"
        if exists(head_path):
            cpu_head_ = ModelManager.load(head_path, device="cpu")
                         # OpenVINO runtime, runs Einsum/MatMul/SSM on CPU

Classifier::postprocess(outputs, top_k)
    if backend == HailoRT:
        return postprocess_hailo(...)
    # ... 舊路徑（OpenVINO/TensorRT）完全不變

Classifier::postprocess_hailo(outputs, top_k)
    if shape == [N] or [1, N] with N == num_classes:
        # logits（ResNet50 / EfficientNet-B0）-> 直接 softmax + top-k
    elif shape == [H, W, C] or [1, H, W, C]:
        # backbone feature map（mambavision-t）
        if cpu_head_ is None:  throw clear error
        NHWC -> NCHW transpose
        head_logits = cpu_head_->infer_raw({input: nchw_tensor})
        # softmax + top-k on head_logits
    else:
        throw "unsupported Hailo backbone output rank"
```

### 三種 Hailo HEF 都走進同一個 entry point

`Classifier` 不需要知道是哪一種，看輸出 shape 就分流：

| HEF 類型                       | 輸出 shape        | 走哪條                 |
| ------------------------------ | ----------------- | ---------------------- |
| 完整分類器（ResNet50）         | `[1000]`        | 直接 softmax           |
| 完整分類器（EfficientNet-B0）  | `[1, 1000]`     | 直接 softmax           |
| Backbone-only（mambavision-t） | `[14, 14, 320]` | NHWC→NCHW + cpu_head_ |

---

## 7. 推論時的設備需求對照

### 編譯機（dev workstation）

| 元件                                  | ResNet50           | EfficientNet-B0              | MambaVision-T         |
| ------------------------------------- | ------------------ | ---------------------------- | --------------------- |
| NVIDIA GPU                            | 不必               | **必要**（QAT）        | **必要**（QAT） |
| CUDA 13.0+, TF 2.x with CUDA          | —                 | 必要                         | 必要                  |
| Hailo SDK (`hailo_sdk_client`, DFC) | 必要               | 必要                         | 必要                  |
| Hailo-8 卡                            | 不必（編譯不接卡） | 不必                         | 不必                  |
| RAM                                   | 8 GB+              | 16 GB+ (QAT activation 統計) | 16 GB+                |

### 部署機（edge）

| 元件                                   | ResNet50                                   | EfficientNet-B0 | MambaVision-T                                         |
| -------------------------------------- | ------------------------------------------ | --------------- | ----------------------------------------------------- |
| NVIDIA GPU                             | **不必**                             | **不必**  | **不必**                                        |
| CUDA / TF                              | 不必                                       | 不必            | 不必                                                  |
| Host CPU                               | 任意 x86 / ARM (Pi 5, Jetson Nano, RK3588) | 同左            | 同左，但 SSM head 吃 CPU 算力（i5-13500 上 36-50 ms） |
| Hailo-8 PCIe / M.2                     | **必要**                             | 必要            | 必要                                                  |
| HailoRT runtime (`libhailort.so`)    | 必要                                       | 必要            | 必要                                                  |
| OpenVINO Runtime（CPU 端跑 head ONNX） | 不必                                       | 不必            | **必要**                                        |
| RAM                                    | ~512 MB                                    | ~512 MB         | ~1 GB（head ONNX 載入 + 工作 buffer）                 |

> 關鍵點：**推論時 NVIDIA GPU 完全不參與**。實測 `CUDA_VISIBLE_DEVICES=""` 跑 200 張 ResNet50 / EfficientNet 維持 98.3 FPS、結果一致。GPU 只在編譯期跑 TF 算量化參數。

### 範例部署架構

```
[開發機 / dev workstation]
i5-13500 + RTX 3060 + Hailo-8 PCIe
  │
  ├─ ONNX → HEF compile (GPU + QAT)
  ├─ extract_model → head.onnx
  └─ 產出 deployment artifacts
        │
        ▼
[部署機 / edge device]
Pi 5 (4 GB) + Hailo-8 M.2 (USB-C 介面 SD-Bus)
  │
  ├─ HailoRT runtime
  ├─ OpenVINO CPU Runtime (給 hybrid head 用)
  ├─ ivit-sdk libivit.so
  └─ 推論 @ 30-80 FPS / < 10W 整機功耗
```

整顆 RTX 3060 不需要跟著到部署機。

---

## 8. 編譯踩坑速查（按 model 對應 bug）

| Model           | Bug | 症狀                                                   | 解                                                  |
| --------------- | --- | ------------------------------------------------------ | --------------------------------------------------- |
| ResNet50        | #7  | 推論 NaN（output buffer dtype 錯）                     | `set_format_type(FLOAT32)` before `configure()` |
| ResNet50        | #8  | 分類全錯（host normalize 毀掉 input）                  | classifier.cpp 對 HailoRT skip mean/std             |
| ResNet50        | #9  | 分類全錯（baked normalize 用 *255 變體）               | 改用 [0,1] scale `[0.485,...]`                    |
| ResNet50        | #13 | top-1 信心很低（COCO 校準 ImageNet 模型）              | dogs-vs-cats 2048 calib                             |
| EfficientNet-B0 | #10 | `Shift delta > 2` on SE GAP                          | `global_avgpool_reduction` 分組 fix               |
| EfficientNet-B0 | #14 | activation collapse → 固定 cluster                    | **GPU + `CUDA_VISIBLE_DEVICES=0` + QAT**    |
| MambaVision-T   | #15 | translate `cannot reshape size 160 into (1,160,196)` | end_node 切 levels.1 + CPU head ONNX                |

---

## 9. 程式檔案清單（這次相關）

| 檔案                                              | 角色                                                                            |
| ------------------------------------------------- | ------------------------------------------------------------------------------- |
| `compile_resnet50.py`                           | ResNet50 編譯（已存在）                                                         |
| `compile_efficientnet_b0.py`                    | EfficientNet-B0 編譯（含 `--gap-fix`）                                        |
| `compile_mambavision_t.py`                      | MambaVision-T backbone 編譯（含 end_node）                                      |
| `extract_mambavision_head.py`                   | 從 mambavision-t.onnx 抽 SSM/Transformer head subgraph 給 CPU 跑                |
| `strip_se_blocks.py`                            | （exploration only）EfficientNet SE 拔除嘗試，已驗證不可行                      |
| `include/ivit/vision/classifier.hpp`            | 加 `cpu_head_` member + `postprocess_hailo` 宣告                            |
| `src/vision/classifier.cpp`                     | 加 Hailo backend 自動偵測 + auto-load `<stem>_head.onnx` + hybrid postprocess |
| `docs/development/hailo-compile-bug-log.md`     | Bug #14, #15                                                                    |
| `docs/development/hef-classification-hybrid.md` | 本文件                                                                          |
| `docs/development/hef-segmentation-hybrid.md`   | DeepLab MBv3 同款架構（更早做）                                                 |

`src/runtime/hailort_runtime.cpp` 沒動 — Bug #7/#8/#9 的修法已支撐三個 model 的 input/output dtype 與 layout。

---

## 10. 快速驗證指令

完整檢查整條 hybrid pipeline：

```bash
# Direct pyhailort 跑 backbone HEF
conda run -n forhailo --no-capture-output python -c "
from hailo_platform import HEF
hef = HEF('mambavision_t.hef')
print('input :', [(x.name, x.shape) for x in hef.get_input_vstream_infos()])
print('output:', [(x.name, x.shape) for x in hef.get_output_vstream_infos()])
"
# 預期: input (224, 224, 3) UINT8, output (14, 14, 320) FLOAT32

# Head ONNX 接口檢查
python -c "
import onnx; m = onnx.load('mambavision_t_head.onnx')
for x in m.graph.input:  print('in :', x.name, [d.dim_value for d in x.type.tensor_type.shape.dim])
for x in m.graph.output: print('out:', x.name, [d.dim_value for d in x.type.tensor_type.shape.dim])
"
# 預期: in [1, 320, 14, 14] / out [1, 1000]

# SDK 端到端
conda run -n forhailo --no-capture-output \
  python examples/python/02_classification.py \
  -m mambavision_t.hef -i examples/data/images/bus.jpg -d hailo:0
# 預期: top-1 minibus, ~30 FPS
```

---

## 11. 已知限制 / 後續

| 項目                                                                        | 現況                                      | 改進方向                                                                |
| --------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------- |
| MambaVision FPS 30                                                          | CPU head 36-50 ms 是瓶頸                  | 試 onnxruntime + OpenVINO Execution Provider；或更小的 mambavision-nano |
| EfficientNet 沒 GPU 不能編                                                  | Hailo 強制 opt_level=0                    | 開發機需要 NVIDIA GPU；或者直接用 ResNet50                              |
| Calibration 用 dogs-vs-cats                                                 | 不是 ImageNet val 但 domain-相近          | 換 ImageNet val 子集（2048 張）可能再提精度                             |
| MambaVision 切點固定在 levels.1                                             | 切更深可能讓 Hailo 多做幾步、CPU 少做幾步 | levels.2 之後是 SSM，無法切；levels.1 已是 conv 最後一站                |
| Hailo SDK 沒 mid-network feedback                                           | EfficientNet SE 切 CPU 不可行             | 等 SDK 支援、或重訓 EfficientNet 移除 SE                                |
| 第二個 `Classifier` constructor 接受 `shared_ptr<Model>` 不做 auto-load | 使用者自己塞 model 時要手動處理 head      | 加 overload 接受 cpu_head 參數                                          |
