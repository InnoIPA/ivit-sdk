# 移除 Python 版 OpenVINO 依賴 — 變更說明

## 這件事在幹嘛（白話）

軟體要跑 AI 模型,需要一個引擎叫 **OpenVINO**(Intel 出的)。原本專案裡 OpenVINO 被裝了**兩份**:

- **C++ 版**(核心、正式在用)→ **完全不動**。
- **Python 版**(額外安裝的,平常只用來「縮小模型」)→ **本次全部拿掉**。

「縮小模型」(技術上叫**量化 / INT8**,讓檔案更小、跑更快)改用另一個工具 **NNCF** 完成,NNCF 不需要 Python 版 OpenVINO。

**結果:整個 Python 程式裡,已經沒有任何一處用到 Python 版 OpenVINO。**

---

## 替代方案總覽

| 原本用途            | 原本作法                                                        | 改成                                                        |
| ------------------- | --------------------------------------------------------------- | ----------------------------------------------------------- |
| INT8 量化(縮小模型) | Python OpenVINO + NNCF(OpenVINO backend),輸出 `.xml`/`.bin` | **NNCF ONNX backend**,輸出量化 `.onnx`              |
| FP16/FP32 格式轉換  | Python OpenVINO API                                             | 委派**C++ binding**(`convert_model`)或 `ovc` 工具 |
| 讀模型輸入尺寸      | onnx 套件 or Python OpenVINO                                    | 只用**onnx 套件**                                     |
| 硬體裝置偵測        | Python OpenVINO `Core`                                        | **C++ binding** `list_devices()`                    |
| 純 Python 推論後端  | `openvino_runtime.py`(已 deprecated)                          | **刪除**,改用 C++ backend                             |

---

## 逐檔變更（位置 + 改動前/後）

### 1. `python/ivit/cli.py` — `cmd_info`(約 L52-63)

印出版本資訊。

**前**

```python
import openvino
print(f"  OpenVINO:       {openvino.__version__}")
```

**後**

```python
import ivit
if ivit.is_cpp_available():
    print("  OpenVINO:       via C++ backend (_ivit_core)")
# 另新增印出 onnxruntime 版本
```

### 2. `python/ivit/cli.py` — `_get_onnx_input_shape`(L344)

讀模型輸入尺寸。

**前**：Strategy 1(onnx 套件)+ Strategy 2(`import openvino as ov; core.read_model`)
**後**：刪除 Strategy 2,只留 onnx 套件。

### 3. `python/ivit/cli.py` — `_convert_to_openvino_int8`(L423)

INT8 量化。

**前**

```python
import openvino as ov
ov_model = ov.convert_model(str(input_path))
quantized_model = nncf.quantize(ov_model, ...)
ov.save_model(quantized_model, str(output_xml))   # 輸出 .xml/.bin
```

**後**

```python
import onnx                                        # 不再 import openvino
onnx_model = onnx.load(str(input_path))
quantized_model = nncf.quantize(onnx_model, ...)   # NNCF ONNX backend
onnx.save(quantized_model, str(output_onnx))       # 輸出 _int8.onnx
```

### 4. `python/ivit/cli.py` — `_convert_to_openvino`(L492,錯誤處理約 L546)

FP16/FP32 轉換,原本 3 個 fallback 策略。

**前**：Strategy 1(C++ binding)+ Strategy 2(ovc)+ Strategy 3(`import openvino as ov`)
**後**：刪除 Strategy 3,只留 C++ binding + ovc,皆無則報錯。

### 5. `python/ivit/train/exporter.py` — `_export_openvino`(L206)

訓練後輸出模型。

**前**

```python
import openvino as ov
from openvino import Core
from openvino.runtime import serialize
model = core.read_model(onnx_path)
serialize(model, xml_path)
```

**後**

```python
# INT8 → nncf ONNX backend,輸出 _int8.onnx
# FP16/FP32 → from ivit._ivit_core import convert_model（委派 C++）
```

### 6. `python/ivit/train/exporter.py` — `_quantize_openvino_int8`(L250)

**前**：吃 openvino model 物件 → `nncf.quantize(ov_model)`
**後**：吃 onnx 檔路徑 → `onnx.load` → `nncf.quantize(onnx_model)`,回傳量化 onnx

### 7. `python/ivit/devices.py` — `_discover_devices`(L118)

硬體偵測。

**前**

```python
except ImportError:
    self._discover_openvino_devices()   # 內部 from openvino import Core
    self._discover_cuda_devices()
    self._add_cpu_fallback()
# 以及整個 _discover_openvino_devices 方法（約 40 行）
```

**後**

```python
except ImportError:
    self._discover_cuda_devices()       # openvino 那段整個移除
    self._add_cpu_fallback()
```

### 8. `python/ivit/runtime/__init__.py` — `_create_openvino_runtime`(L69) + `list_available_backends`(L105)

**前**

```python
from .openvino_runtime import OpenVINORuntime   # L72
return OpenVINORuntime()
...
from openvino import Core                        # L108 偵測探針
available.append("openvino")
```

**後**

```python
return None                                      # 純 Python runtime 已移除
# list_available_backends 裡的 openvino 偵測整段刪除
```

### 9. `python/ivit/runtime/openvino_runtime.py` — **整檔刪除**

檔頭本來就標記 `deprecated. Use C++ bindings instead`。

### 10. `python/ivit/core/exceptions.py` — `BACKEND_INSTALL_HINTS`(L150) + 轉換建議(L324)

**前**

```python
"openvino": "pip install openvino",
...
suggestions.append("pip install openvino-dev")
```

**後**

```python
"openvino": "pip install -e .  # 建置 C++ binding（提供 OpenVINO 推論）",
...
suggestions.append("INT8 量化改用 NNCF ONNX backend：pip install -e '.[quantize]'")
```

### 11. `pyproject.toml` — `quantize`(L89) + `all`(L104)

**前**

```toml
quantize = ["nncf>=2.7", "openvino>=2024.0"]
all = ["ivit-sdk[openvino,tensorrt,zoo,train,training-all,quantize,dev]"]
```

**後**

```toml
quantize = ["nncf>=2.7", "onnx>=1.14"]
all = ["ivit-sdk[tensorrt,zoo,train,training-all,quantize,dev]"]
```

---

## 驗證結果（全程封鎖 Python openvino import）

用「強制讓 Python openvino 無法被使用」的方式自我檢查,再把軟體跑一遍:

- ✅ `import ivit` 正常,C++ backend 可用
- ✅ `ivit info` → `OpenVINO: via C++ backend` / `ONNX Runtime: 1.23.2`
- ✅ INT8 量化(`ivit convert ... -p int8 -c calib/`)成功:resnet50 **98MB → 25MB**,走 NNCF ONNX backend
- ✅ 量化後 ONNX 由 **OpenVINO CPU C++ backend 載入 + 推論成功**
- ✅ FP16 轉換走 C++ binding 產出 `.xml`/`.bin`
- ✅ 所有改過模組語法 / import 正常

---

## 注意事項

1. **要重裝才生效**:目前系統的 `ivit` 是 site-packages 複製版(非 editable)。修改在原始碼 `python/ivit/`,需重裝才會套用:

   ```bash
   pip install -e .
   ```
2. **量化 ONNX 不能跑 TensorRT**:NNCF 產出的是**非對稱量化**,TensorRT 會報錯
   「Non-zero zero point is not supported」。**OpenVINO 沒問題**。推論時指定 `device='cpu'`(或 Intel 裝置)走 OpenVINO。
3. **`ivit devices` 既有 bug(與本次無關)**:`cli.py` cmd_devices 用了 `d.available`,
   但 C++ `DeviceInfo` 屬性叫 `is_available`。此為原有問題,本次未更動。
