# MambaVision 整合流程紀錄

> 日期：2026-04-07  
> 環境：Anaconda `ivitsdk` (Python 3.10)、Ubuntu 22.04、RTX 3060 + Intel i5-13500

---

## 1. 什麼是 MambaVision

MambaVision 是 NVIDIA Research 發布的混合架構影像分類模型，結合 **Mamba (SSM)** 與 **Transformer** 兩種結構：

- 前幾層用 Mamba selective scan，低延遲提取局部特徵
- 後幾層用 Multi-head Self-Attention，全域語意理解
- ImageNet-1K 預訓練，1000 類分類

| 變體 | Top-1 | Params | MACs |
|------|-------|--------|------|
| mambavision-t | 82.3% | 31.8M | 4.4G |
| mambavision-s | 83.3% | 50.1M | 7.3G |
| mambavision-b | 84.2% | 97.7M | 15.0G |
| mambavision-l | 85.0% | 227.9M | 34.9G |
| mambavision-l2 | 85.3% | 241.5M | 36.9G |

授權：Apache-2.0（商業友善）

---

## 2. 新增到 Model Zoo

### 2.1 修改 `python/ivit/zoo/registry.py`

在 Classification 區段的 `efficientnet-b0` 之後新增 5 個 MambaVision 變體：

```python
"mambavision-t": ModelInfo(
    name="mambavision-t",
    task="classify",
    description="MambaVision-Tiny - Hybrid Mamba+Transformer, fast edge inference",
    input_size=(224, 224),
    num_classes=1000,
    formats=["onnx", "openvino", "tensorrt"],
    url="https://huggingface.co/nvidia/MambaVision-T-1K",
    source="mambavision",
    license="Apache-2.0",
    metrics={"top1": 82.3, "params_m": 31.8, "macs_g": 4.4},
    tags=["classification", "imagenet", "mamba", "ssm", "transformer",
          "hybrid", "fast", "commercial-friendly"],
),
# 同樣模式新增 mambavision-s / -b / -l / -l2
```

驗證：

```bash
python3 -c "
import sys; sys.path.insert(0, 'python')
from ivit.zoo import list_models
models = [m for m in list_models() if 'mamba' in m]
print(models)
"
# ['mambavision-t', 'mambavision-s', 'mambavision-b', 'mambavision-l', 'mambavision-l2']
```

### 2.2 新增 `load_mambavision_for_export()` 至 `python/ivit/train/exporter.py`

MambaVision 使用 `mamba_ssm` 的 CUDA-only selective scan kernel，無法直接 `torch.onnx.export`。需要三個 patch：

```python
def load_mambavision_for_export(variant: str = "t"):
    # Patch A: torch.linspace 在 PyTorch ≥ 2.8 + accelerate 下
    #          會進入 meta tensor context，.item() 拋錯
    _orig_linspace = torch.linspace
    def _cpu_linspace(*args, **kwargs):
        kwargs.pop('device', None)
        return _orig_linspace(*args, device='cpu', **kwargs)
    torch.linspace = _cpu_linspace

    # Patch B: transformers 5.5 將 _tied_weights_keys 改名
    if not hasattr(PreTrainedModel, 'all_tied_weights_keys'):
        PreTrainedModel.all_tied_weights_keys = property(
            lambda self: getattr(self, '_tied_weights_keys', None) or {}
        )

    model = AutoModelForImageClassification.from_pretrained(
        "nvidia/MambaVision-T-1K", trust_remote_code=True, low_cpu_mem_usage=False
    )
    torch.linspace = _orig_linspace  # 恢復

    # Patch C: 修補 modeling_mambavision 模組層級的 selective_scan_fn
    # 必須透過 sys.modules 修補，因為 from ... import 會建立本地綁定
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
    mod_key = next((k for k in sys.modules if 'modeling_mambavision' in k), None)
    if mod_key:
        sys.modules[mod_key].selective_scan_fn = selective_scan_ref

    model.eval()
    return model
```

---

## 3. 安裝依賴

```bash
# mamba_ssm 需要編譯，使用 --no-build-isolation 才能成功
pip install mamba-ssm --no-build-isolation

# ONNX 相關
pip install onnx onnxruntime
```

---

## 4. 匯出 ONNX 模型

```python
import sys, os, torch
sys.path.insert(0, 'python')
from ivit.train.exporter import load_mambavision_for_export

model = load_mambavision_for_export('t')

# HuggingFace 模型回傳 dict，需包裝為只輸出 logits 的 Module
class ExportWrapper(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x)['logits']

os.makedirs(os.path.expanduser('~/.cache/ivit/models'), exist_ok=True)
path = os.path.expanduser('~/.cache/ivit/models/mambavision-t.onnx')

torch.onnx.export(
    ExportWrapper(model.cpu().eval()),
    torch.randn(1, 3, 224, 224),
    path,
    input_names=['input'],
    output_names=['logits'],
    opset_version=17,
)
print('Exported:', path)  # ~123 MB
```

---

## 5. 純 Python 推論驗證（onnxruntime）

使用獨立 demo 腳本 `examples/python/02_mambavision.py`，不依賴 C++ bindings：

```bash
python examples/python/02_mambavision.py \
    -m ~/.cache/ivit/models/mambavision-t.onnx \
    -i examples/data/images/bus.jpg
```

預期輸出：

```
Top-5 Predictions:
  1. minibus               69.53%  █████████████
```

GPU (CUDA) 模式：

```bash
python examples/python/02_mambavision.py \
    -m ~/.cache/ivit/models/mambavision-t.onnx \
    -i examples/data/images/bus.jpg \
    -d cuda:0
```

---

## 6. 透過 C++ Bindings 推論

### 6.1 問題：GLIBCXX 版本不符

使用 `conda activate ivitsdk` 後執行 `ivit.Classifier` 出現：

```
ImportError: GLIBCXX_3.4.30 not found
```

原因：Anaconda base 環境的 `libstdc++.so.6` 最高版本為 3.4.26，但 `_ivit_core.so` 是用系統 GCC 11 編譯（需要 3.4.30）。

### 6.2 修復

```bash
conda install -c conda-forge libstdcxx-ng
```

安裝後 Anaconda 的 libstdc++ 升級至 GCC 13.2（GLIBCXX_3.4.32），問題解決。

### 6.3 標籤檔缺失

C++ Classifier 按以下順序尋找標籤：
1. 模型內建 labels
2. `{model_stem}.txt`（如 `mambavision-t.txt`）
3. `labels.txt`
4. fallback：只有 10 筆的 hardcoded IMAGENET_LABELS

由於沒有 `mambavision-t.txt`，`num_classes` 顯示 10，標籤顯示 `class_654`。

修復：

```bash
cp ~/.cache/ivit/models/resnet50.txt \
   ~/.cache/ivit/models/mambavision-t.txt
```

### 6.4 執行

```bash
conda activate ivitsdk

# CPU (OpenVINO)
python examples/python/02_classification.py \
    -m ~/.cache/ivit/models/mambavision-t.onnx \
    -i examples/data/images/bus.jpg \
    -d cpu

# GPU (TensorRT — 首次會建置 engine，約 90 秒)
python examples/python/02_classification.py \
    -m ~/.cache/ivit/models/mambavision-t.onnx \
    -i examples/data/images/bus.jpg \
    -d cuda:0
```

TensorRT engine 會快取至：

```
~/.cache/ivit/models/mambavision-t_trt_fp32_sm86.engine
```

第二次啟動直接載入，無需重建。

---

## 7. 最終結果

| 後端 | 裝置 | 推論時間 | FPS | Top-1 |
|------|------|----------|-----|-------|
| onnxruntime | CPU | ~33 ms | 30 | minibus 69.53% |
| onnxruntime | CUDA | — | — | minibus 69.53% |
| OpenVINO (C++) | CPU | 30.8 ms | 32.5 | minibus 69.53% |
| TensorRT (C++) | RTX 3060 | 12.9 ms | **77.7** | minibus 69.47% |

---

## 8. 遇到的坑與解法摘要

| 問題 | 根因 | 解法 |
|------|------|------|
| `mamba-ssm` pip 安裝失敗 | 需要編譯但 build isolation 阻擋 | `pip install mamba-ssm --no-build-isolation` |
| `Tensor.item() on meta tensor` | PyTorch 2.8 + accelerate 讓 `torch.linspace` 進入 meta context | 匯出前 monkey-patch `torch.linspace` 強制 `device='cpu'` |
| `has no attribute 'all_tied_weights_keys'` | transformers 5.5 API 改名 | 在 `PreTrainedModel` 上加 property alias |
| `selective_scan_fn` patch 無效 | `from ... import` 建立本地綁定，patch 原始模組無效 | 透過 `sys.modules['...modeling_mambavision']` 直接替換 |
| `Expected u.is_cuda() to be true` | CUDA-only kernel 在 CPU 上執行 | 上述 `selective_scan_fn` patch |
| `dict has no attribute 'logits'` | HF 模型回傳 dict 而非物件 | `ExportWrapper` 包裝 `model(x)['logits']` |
| `GLIBCXX_3.4.30 not found` | Anaconda libstdc++ 版本過舊（3.4.26） | `conda install -c conda-forge libstdcxx-ng` |
| `num_classes: 10`、`class_654` | 找不到標籤檔，fallback 只有 10 筆 | 複製 `resnet50.txt` 為 `mambavision-t.txt` |
