# SDK 人工驗證執行清單

> 2026-10-16 實跑更新：已完成四組合、真實模型下載／匯出及整合驗證；第 9 步純 TensorRT 的測試假設與第 12 步 .NET PATH 有確定問題，詳見 [完整實跑結果](manual-validation-results-20261016.zh-TW.md)。其後已修正 `tests/cpp/test_inference.cpp` 的 CPU 假設（OpenVINO 建置要求 CPU；純 TensorRT 建置要求沒有 CPU 且所列裝置皆為 TensorRT），並讓分割匯出的 `_SegWrapper` 呼叫 `.eval()`；兩份建置的 CTest 重跑均為 3/3 通過。第 12 步已補上 .NET 前置條件。

本清單涵蓋從原始碼安裝、執行範例，到外部專案與 C／C# 整合。每一步附上來源文件、章節及目前行號；行號可能隨後續修改變動，章節名稱可用來定位。

以下命令供人工逐步執行，本清單建立時沒有執行系統安裝或整套驗證。適用 Ubuntu 24.04 x86_64、Bash、有可用 NVIDIA GPU／驅動的主機；其他平台不能直接沿用 NVIDIA 安裝指令。

## 驗證矩陣與通過規則

| 組合 | 安裝／建置 | 明確指定後端 | 裝置 | 任務 |
| --- | --- | --- | --- | --- |
| Python × OpenVINO | 獨立 OpenVINO-only venv | `openvino` | `cpu`；有 Intel GPU 再跑 `gpu:0` | 分類、偵測、分割 |
| Python × TensorRT | 另一個含 OpenVINO＋TensorRT 的 venv | `tensorrt` | `cuda:0` | 分類、偵測、分割；FP32／FP16 |
| C++ × OpenVINO | 獨立 OpenVINO-only build／prefix | `openvino` | `cpu`；有 Intel GPU 再跑 `gpu:0` | 分類、偵測、分割 |
| C++ × TensorRT | 獨立 TensorRT-only build／prefix | `tensorrt` | `cuda:0` | 分類、偵測、分割；FP32／FP16 |

Python TensorRT 環境保留 OpenVINO，是為了執行 TensorRT 驗證工具的 CPU 參考比對；推論時仍明確指定 TensorRT。C++ 使用純 TensorRT 建置，另驗證它不依賴 OpenVINO 後端。

「所有組合」在此指上述四個語言／後端組合，不代表所有 SDK 模型、所有硬體、多 GPU 或 INT8。沒有 Intel GPU／NPU 時記為未驗證；沒有 NVIDIA GPU 時，不能將 TensorRT 跳過標示為通過。

每項需記錄命令、退出碼、實際後端／裝置、版本及輸出。退出碼 0 只是必要條件，還需確認結果內容。FP16 與不同後端不要求逐位元相同。

## 1. 固定驗證對象與工作目錄

來源：[複驗報告「環境與重現證據」](sdk-improvement-revalidation-20261014.zh-TW.md)、[手冊 §3.0，L106](../sdk-manual.zh-TW.md)。保留工作樹差異與分開目錄是本清單的驗證安排。

在 repository 根目錄執行，另開乾淨 Bash，避免沿用其他 SDK 的 `PYTHONPATH`／`LD_LIBRARY_PATH`：

```bash
export IVIT_ROOT="$PWD"
export IVIT_AUDIT="/tmp/ivit-manual-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$IVIT_AUDIT/log" "$IVIT_AUDIT/models" "$IVIT_AUDIT/results"
set -o pipefail
unset PYTHONPATH
unset LD_LIBRARY_PATH
git rev-parse HEAD > "$IVIT_AUDIT/log/commit.txt"
git status --short > "$IVIT_AUDIT/log/worktree.txt"
git diff --binary > "$IVIT_AUDIT/log/worktree.patch"
```

- [ ] 記錄 commit、未提交修改及未追蹤檔案。Patch 不含未追蹤檔案；若測試對象含這些檔案，另行保存它們。
- [ ] 若需可重現的發行認證，使用所有待驗收內容已提交的 commit。工作樹驗證只能宣稱對應當次工作樹。

## 2. 確認主機、系統依賴與 NVIDIA 驅動

來源：[手冊 §3.1，L141](../sdk-manual.zh-TW.md)、[§3.4，L236](../sdk-manual.zh-TW.md)、[§3.0 空間說明，L133](../sdk-manual.zh-TW.md)。

```bash
cat /etc/os-release
uname -m
python3 --version
c++ --version
df -h "$IVIT_AUDIT" /usr /tmp
nvidia-smi
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-venv python3-dev \
  libopencv-dev pkg-config libgtest-dev curl
pkg-config --modversion opencv4
bash scripts/install_sdk.sh --check
```

- [ ] Python 3.10+、C++17、CMake 3.18+、原生 OpenCV 及 GoogleTest 可用。
- [ ] `nvidia-smi` 顯示可用 GPU／驅動；腳本不安裝 NVIDIA 驅動。
- [ ] 預檢空間足夠；模型、PyTorch、engine、下載及建置仍會額外佔空間。
- [ ] 第 12 步需要 .NET 8 SDK：先確認 `dotnet --version` 為 8.x；若 SDK 不在 PATH，執行 `export PATH="<SDK 目錄>:$PATH"`，或第 12 步所有 `dotnet` 命令改用完整路徑。

## 3. 全新安裝 Python × OpenVINO

來源：[手冊 §3.0，L106](../sdk-manual.zh-TW.md)、[§3.2，L166](../sdk-manual.zh-TW.md)。

```bash
bash scripts/install_sdk.sh --skip-system --dev \
  --venv "$IVIT_AUDIT/py-ov" --jobs 2 \
  2>&1 | tee "$IVIT_AUDIT/log/install-python-ov.log"
source "$IVIT_AUDIT/py-ov/bin/activate"
python3 -m pip check
python3 -c 'import ivit; print(ivit.__file__); print(ivit.get_runtime_info()); assert ivit.is_cpp_available(); print(ivit.list_devices())'
```

- [ ] 安裝成功、依賴檢查通過、原生綁定可用。
- [ ] `cpu` 的 backend 為 OpenVINO；若有 Intel GPU，確認 `gpu:0`。
- [ ] 此環境不應出現可用的 TensorRT 裝置。

## 4. 全新安裝 Python × TensorRT，核對真正使用的版本

來源：[手冊 §3.0，L106](../sdk-manual.zh-TW.md)、[Python TensorRT 安裝，L196](../sdk-manual.zh-TW.md)、[§3.4，L236](../sdk-manual.zh-TW.md)、[§9.4，L742](../sdk-manual.zh-TW.md)。

若要驗證 Ubuntu 24.04 的 NVIDIA 自動安裝流程，執行第一種；若 NVIDIA 開發套件已準備好，執行第二種。兩種路徑擇一，不要把已安裝套件時的跳過當成全新自動安裝通過。

```bash
deactivate
# 路徑 A：實際執行 NVIDIA 套件安裝；需 sudo，且不安裝 NVIDIA 驅動。
bash scripts/install_sdk.sh --dev --tensorrt --install-nvidia \
  --venv "$IVIT_AUDIT/py-trt" --jobs 2 \
  2>&1 | tee "$IVIT_AUDIT/log/install-python-trt.log"

# 路徑 B：已具備 CUDA／TensorRT 開發套件時，改跑這條。
# bash scripts/install_sdk.sh --skip-system --dev --tensorrt \
#   --venv "$IVIT_AUDIT/py-trt" --jobs 2 \
#   2>&1 | tee "$IVIT_AUDIT/log/install-python-trt.log"

source "$IVIT_AUDIT/py-trt/bin/activate"
python3 -m pip check
ivit info
python3 - <<'PY'
import ivit
from ivit import _ivit_core
assert ivit.is_cpp_available(), ivit.get_runtime_info()
print(ivit.__file__)
print(_ivit_core.__file__)
print([(d.id, d.backend, d.is_available) for d in ivit.list_devices()])
print('\n'.join(sorted({line.split()[-1] for line in open('/proc/self/maps')
                       if any(s in line for s in ('libivit', 'libnvinfer.so', 'libopenvino.so'))})))
assert any(d.id == 'cuda:0' and d.backend == 'tensorrt' and d.is_available
           for d in ivit.list_devices())
PY
python3 -m pip freeze > "$IVIT_AUDIT/log/python-trt-packages.txt"
dpkg-query -W 'libnvinfer*' 'libnvonnxparsers*' 'cuda-cudart*' \
  > "$IVIT_AUDIT/log/nvidia-packages.txt"
```

- [ ] 實際載入的 `libnvinfer.so.*` 版本與目標建置一致，沒有載入其他 checkout／暫存 SDK。
- [ ] 要宣稱 TensorRT 11.4 通過，實際載入必須是 11.4；套件安裝記錄不能取代載入診斷。
- [ ] `cuda:0` 是 TensorRT，且有 OpenVINO CPU 可作參考比對。

## 5. 準備模型，驗證下載與首次匯出

來源：[手冊 §6.4，L497](../sdk-manual.zh-TW.md)、[§4.2，L307](../sdk-manual.zh-TW.md)、[Zoo 的 load_options，L60](../../python/ivit/zoo/registry.py)。

仍在 `py-trt` 環境執行。直接安裝匯出依賴，避免重跑不帶後端選項的 editable SDK 安裝而改變建置：

```bash
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install onnx onnxscript
export IVIT_CACHE_DIR="$IVIT_AUDIT/model-cache"
unset IVIT_ZOO_MIRROR
python3 - <<'PY'
import json, os, shutil
from pathlib import Path
import ivit
dest = Path(os.environ['IVIT_AUDIT']) / 'models'
for name in ('resnet18', 'yolox-s', 'deeplabv3-mobilenetv3'):
    src = Path(ivit.zoo.download(name))
    shutil.copy2(src, dest / f'{name}.onnx')
    labels = src.with_suffix('.labels.txt')
    if labels.exists():
        shutil.copy2(labels, dest / f'{name}.labels.txt')
    (dest / f'{name}-options.json').write_text(
        json.dumps(ivit.zoo.get_model_info(name).load_options, indent=2))
    print(name, src)
PY
sha256sum "$IVIT_AUDIT"/models/*.onnx > "$IVIT_AUDIT/log/model-sha256.txt"
```

- [ ] YOLOX-S 真正下載且 checksum 驗證成功；若改用 mirror，記錄為離線／鏡像驗證。
- [ ] ResNet18、DeepLabV3 首次下載權重與本機 ONNX 匯出成功。
- [ ] 同時保存 ONNX、labels 與 load_options；C++／Python 共用這批模型。
- [ ] 新 SDK cache 不代表 torchvision 權重 cache 也是空的；要驗證權重首次下載，另在全新使用者／乾淨 torchvision cache 環境執行，不需刪除既有 cache。

## 6. 執行 Python 的現有範例

來源：[手冊 §6，L425](../sdk-manual.zh-TW.md)、[分類範例，L57](../../examples/python/02_classification.py)、[偵測範例](../../examples/python/02_detection.py)、[分割範例](../../examples/python/02_segmentation.py)。

以下一組命令依矩陣重複：`py-ov + cpu`、`py-ov + gpu:0`（有 Intel GPU）、`py-trt + cuda:0`。`IVIT_RUN` 自訂成不同名稱，例如 `python-ov-cpu`。

```bash
# 每輪先 source 對應的 venv；例如：
source "$IVIT_AUDIT/py-ov/bin/activate"
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install onnx onnxscript
export IVIT_DEVICE=cpu
export IVIT_RUN=python-ov-cpu
mkdir -p "$IVIT_AUDIT/results/$IVIT_RUN"
export IVIT_ZOO_MIRROR="$IVIT_AUDIT/models"

python3 examples/python/02_classification.py --model resnet18 --device "$IVIT_DEVICE" \
  2>&1 | tee "$IVIT_AUDIT/log/$IVIT_RUN-classification.log"
python3 examples/python/02_detection.py --model yolox-s --device "$IVIT_DEVICE" \
  --output "$IVIT_AUDIT/results/$IVIT_RUN/detection.jpg" \
  2>&1 | tee "$IVIT_AUDIT/log/$IVIT_RUN-detection.log"
python3 examples/python/02_segmentation.py --model deeplabv3-mobilenetv3 --device "$IVIT_DEVICE" \
  --output "$IVIT_AUDIT/results/$IVIT_RUN/segmentation.jpg" \
  2>&1 | tee "$IVIT_AUDIT/log/$IVIT_RUN-segmentation.log"
```

`py-ov` 也需 `torch`／`torchvision`／`onnxscript` 才能首次匯出分類／分割，依第 5 步安裝相同依賴。其 torchvision ONNX cache 可沿用第 5 步；完整首次匯出只需另行明確記錄，不必每輪重做。

- [ ] 看到預期 device；類別數不是錯誤的零，未知時顯示 `unknown`。
- [ ] 偵測與分割影像／JSON／mask 有生成，人工打開檢視。
- [ ] Quickstart 可另跑 `python3 examples/python/01_quickstart.py`，但它會自動選裝置，不能取代指定組合的驗證。

## 7. 明確指定 Python 後端，跑三任務與 FP32／FP16

來源：[手冊 §5.2，L340](../sdk-manual.zh-TW.md)、[§6.1，L427](../sdk-manual.zh-TW.md)、[§6.4，L497](../sdk-manual.zh-TW.md)。[matrix.py](manual-validation-assets/matrix.py) 是本清單新增的驗收輔助程式，依 Zoo 契約明確指定後端，不是原手冊的逐字命令。

```bash
source "$IVIT_AUDIT/py-ov/bin/activate"
python3 docs/development/manual-validation-assets/matrix.py \
  --backend openvino --device cpu --precision fp32 \
  --image examples/data/images/bus.jpg --output "$IVIT_AUDIT/results/python-ov-cpu"
# 有 Intel GPU 時，以 --device gpu:0、不同 output 目錄再跑一輪。

source "$IVIT_AUDIT/py-trt/bin/activate"
for precision in fp32 fp16; do
  python3 docs/development/manual-validation-assets/matrix.py \
    --backend tensorrt --device cuda:0 --precision "$precision" \
    --image examples/data/images/bus.jpg --output "$IVIT_AUDIT/results/python-trt-$precision"
done
```

- [ ] 三種任務均通過斷言，輸出 JSON、影像及原始分割 class mask。
- [ ] 明確記錄 `backend`、`device`、請求精度；精度實際行為依手冊 §5.2 與模型張量型別。

## 8. 執行後端參考比對與動態 batch

來源：[手冊 §3.6，L288](../sdk-manual.zh-TW.md)、[§3.7，L294](../sdk-manual.zh-TW.md)。

```bash
source "$IVIT_AUDIT/py-ov/bin/activate"
python3 scripts/verify_openvino.py --device cpu --report "$IVIT_AUDIT/results/openvino-cpu.json"
# 有 Intel GPU 時執行以下兩條；有 NPU 時另跑 --device npu（靜態）。
python3 scripts/verify_openvino.py --device gpu:0 --report "$IVIT_AUDIT/results/openvino-gpu.json"
python3 scripts/verify_openvino.py --device gpu:0 --dynamic-batch \
  --report "$IVIT_AUDIT/results/openvino-gpu-dynamic.json"

source "$IVIT_AUDIT/py-trt/bin/activate"
for precision in fp32 fp16; do
  python3 scripts/verify_tensorrt.py --precision "$precision" \
    --report "$IVIT_AUDIT/results/tensorrt-$precision.json"
  python3 scripts/verify_tensorrt.py --precision "$precision" --dynamic-batch \
    --report "$IVIT_AUDIT/results/tensorrt-$precision-dynamic.json"
done
```

- [ ] 所有已具備裝置的組合退出碼為 0，報告 `status` 為 `passed`。
- [ ] TensorRT 四種組合都跑到，尤其不能漏掉 FP32 dynamic。
- [ ] 此步是小型分類器參考比對，不替代第 6／7 步的真實模型結果檢視。

## 9. 分別建置 C++ × OpenVINO 與 C++ × TensorRT

來源：[手冊 §3.3，L218](../sdk-manual.zh-TW.md)、[§3.4，L236](../sdk-manual.zh-TW.md)、[§9.2，L716](../sdk-manual.zh-TW.md)。將手冊的測試 OFF 改成 ON 是為了本次驗收；使用獨立 build 與安裝 prefix。

```bash
source "$IVIT_AUDIT/py-ov/bin/activate"
export IVIT_OV_CMAKE="$(python3 -c 'from pathlib import Path; import openvino; print(Path(openvino.__file__).parent / "cmake")')"
export IVIT_OV_LIB="$(python3 -c 'from pathlib import Path; import openvino; print(Path(openvino.__file__).parent / "libs")')"
# 若 native CUDA 在其他位置，改成實際的 lib64 路徑。
export IVIT_CUDA_LIB="${CUDAToolkit_ROOT:-/usr/local/cuda}/lib64"
export LD_LIBRARY_PATH="$IVIT_OV_LIB:$IVIT_CUDA_LIB"

cmake -S . -B "$IVIT_AUDIT/cpp-ov" -DCMAKE_BUILD_TYPE=Release \
  -DIVIT_BUILD_PYTHON=OFF -DIVIT_BUILD_TESTS=ON -DIVIT_BUILD_EXAMPLES=ON \
  -DIVIT_USE_OPENVINO=ON -DIVIT_USE_TENSORRT=OFF -DIVIT_REQUIRE_BACKENDS=ON \
  -DOpenVINO_DIR="$IVIT_OV_CMAKE"
cmake --build "$IVIT_AUDIT/cpp-ov" -j2
ctest --test-dir "$IVIT_AUDIT/cpp-ov" --output-on-failure --no-tests=error
cmake --install "$IVIT_AUDIT/cpp-ov" --prefix "$IVIT_AUDIT/prefix-ov"
"$IVIT_AUDIT/cpp-ov/bin/simple_inference" devices

cmake -S . -B "$IVIT_AUDIT/cpp-trt" -DCMAKE_BUILD_TYPE=Release \
  -DIVIT_BUILD_PYTHON=OFF -DIVIT_BUILD_TESTS=ON -DIVIT_BUILD_EXAMPLES=ON \
  -DIVIT_USE_OPENVINO=OFF -DIVIT_USE_TENSORRT=ON -DIVIT_REQUIRE_BACKENDS=ON
cmake --build "$IVIT_AUDIT/cpp-trt" -j2
ctest --test-dir "$IVIT_AUDIT/cpp-trt" --output-on-failure --no-tests=error
cmake --install "$IVIT_AUDIT/cpp-trt" --prefix "$IVIT_AUDIT/prefix-trt"
"$IVIT_AUDIT/cpp-trt/bin/simple_inference" devices
```

CUDA／TensorRT 非系統安裝時，SDK configure 與後續 consumer configure 都需加 `-DTENSORRT_ROOT=... -DCUDAToolkit_ROOT=...`。

- [ ] 兩份 CMake Summary 各自啟用正確後端，不能接受自動停用。
- [ ] OpenVINO build 列出 `cpu`；TensorRT-only build 列出 `cuda:0`。
- [ ] CTest 有實際測試，不是空目錄成功；檢查是否有測試在內部跳過，記錄其原因。

## 10. 執行 C++ 現有範例，並驗證外部 CMake 三任務整合

來源：[手冊 §7.1，L547](../sdk-manual.zh-TW.md)、[§7.2，L569](../sdk-manual.zh-TW.md)、[§7.3，L617](../sdk-manual.zh-TW.md)、[classification_demo 的 usage，L27](../../examples/cpp/classification_demo.cpp)、[detection_demo](../../examples/cpp/detection_demo.cpp)。

先跑原有分類／偵測範例；有視窗時按鍵結束。執行 TensorRT 時將 bin 改成 `cpp-trt/bin`，device 改成 `cuda:0`。每輪在不同結果目錄執行，避免覆寫影像：

```bash
mkdir -p "$IVIT_AUDIT/results/cpp-ov-examples"
cd "$IVIT_AUDIT/results/cpp-ov-examples"
"$IVIT_AUDIT/cpp-ov/bin/classification_demo" "$IVIT_AUDIT/models/resnet18.onnx" \
  "$IVIT_ROOT/examples/data/images/bus.jpg" cpu imagenet
"$IVIT_AUDIT/cpp-ov/bin/detection_demo" "$IVIT_AUDIT/models/yolox-s.onnx" \
  "$IVIT_ROOT/examples/data/images/bus.jpg" cpu 0.5 yolox
cd "$IVIT_ROOT"
```

`simple_inference segment` 不帶 ImageNet mean/std，不適合直接作 Zoo DeepLab 精度驗收。本清單提供 [main.cpp](manual-validation-assets/main.cpp)，依 §4.2、§7.2／7.3 配置正確契約，也可明確選 FP16；它是新增的客戶整合示範，不是既有正式範例。

```bash
cmake -S docs/development/manual-validation-assets -B "$IVIT_AUDIT/customer-ov" \
  -DCMAKE_PREFIX_PATH="$IVIT_AUDIT/prefix-ov" -DOpenVINO_DIR="$IVIT_OV_CMAKE"
cmake --build "$IVIT_AUDIT/customer-ov" -j2
cmake -S docs/development/manual-validation-assets -B "$IVIT_AUDIT/customer-trt" \
  -DCMAKE_PREFIX_PATH="$IVIT_AUDIT/prefix-trt"
cmake --build "$IVIT_AUDIT/customer-trt" -j2

# 此區塊每輪修改 IVIT_CPP_APP／BACKEND／DEVICE／PRECISION／RUN：
# customer-ov + openvino + cpu + fp32；有 Intel GPU 再跑 gpu:0。
# customer-trt + tensorrt + cuda:0 + fp32；再跑 fp16。
export IVIT_CPP_APP="$IVIT_AUDIT/customer-ov/matrix_app"
export IVIT_BACKEND=openvino IVIT_DEVICE=cpu IVIT_PRECISION=fp32 IVIT_RUN=cpp-ov-cpu
mkdir -p "$IVIT_AUDIT/results/$IVIT_RUN"
"$IVIT_CPP_APP" "$IVIT_AUDIT/models/resnet18.onnx" "$IVIT_ROOT/examples/data/images/bus.jpg" \
  classification "$IVIT_BACKEND" "$IVIT_DEVICE" "$IVIT_PRECISION" "$IVIT_AUDIT/results/$IVIT_RUN/resnet18"
"$IVIT_CPP_APP" "$IVIT_AUDIT/models/yolox-s.onnx" "$IVIT_ROOT/examples/data/images/bus.jpg" \
  detection "$IVIT_BACKEND" "$IVIT_DEVICE" "$IVIT_PRECISION" "$IVIT_AUDIT/results/$IVIT_RUN/yolox-s"
"$IVIT_CPP_APP" "$IVIT_AUDIT/models/deeplabv3-mobilenetv3.onnx" "$IVIT_ROOT/examples/data/images/bus.jpg" \
  segmentation "$IVIT_BACKEND" "$IVIT_DEVICE" "$IVIT_PRECISION" "$IVIT_AUDIT/results/$IVIT_RUN/deeplabv3-mobilenetv3"
```

- [ ] 外部專案透過 `find_package(ivit 1.1)` 成功 configure／build／run。
- [ ] C++ 三任務 × OpenVINO／TensorRT 都執行；TensorRT 另跑 FP16。
- [ ] 結果明確顯示 device，JSON／視覺化／分割 class PNG 生成。

## 11. 跨語言、跨後端檢視結果

來源：[手冊 §4.2，L307](../sdk-manual.zh-TW.md)、[§3.7，L294](../sdk-manual.zh-TW.md)、[原驗證報告的精度限制](developer-validation-20261014.zh-TW.md)。此步為本清單增加的人工判讀，不是既有基準測試。

- [ ] 相同模型、影像、精度與後端下，比較 Python／C++ 的分類 top-1、分數及偵測 class ID／confidence／bbox。
- [ ] 比較 OpenVINO CPU／Intel GPU／TensorRT FP32；另比較 TensorRT FP32／FP16。
- [ ] Python 的 `*-mask.npy` 與 C++ 的 `*-classes.png` 是可比對的原始分割 class ID；不要拿彩色 overlay 判定逐像素一致。
- [ ] 記錄分數最大差、box IoU／偵測數、mask 一致率；有差異先核對 mean/std、decoder、letterbox、信心值門檻，再調查後端。
- [ ] 預先訂定可接受的精度差異。手冊中的歷史數值是參考，不能直接當所有模型的保證。

## 12. 驗證 C ABI 與 C#，兩個後端都跑

來源：[手冊 §8.1，L653](../sdk-manual.zh-TW.md)、[C# README，L1](../../bindings/csharp/README.md)、[C consumer 原始範例，L1](../../tests/integration/cmake_consumer/main.c)。

既有 C consumer 寫死 CPU／OpenVINO；本清單的 [c_matrix.c](manual-validation-assets/c_matrix.c) 保留其 stride、buffer 與生命週期檢查，將 backend／device 改為命令列參數，因此可驗證 TensorRT。

先以現有驗證工具產生小型四類 ONNX，避免 C# 的零影像 smoke test 被當成 ImageNet 精度驗證：

```bash
# 前置：dotnet 必須在 PATH（見第 2 步）；否則本區塊的 dotnet 命令會回傳 127。
source "$IVIT_AUDIT/py-trt/bin/activate"
python3 - <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from verify_openvino import make_model
make_model(Path(os.environ['IVIT_AUDIT']) / 'models' / 'integration-classifier.onnx')
PY
"$IVIT_AUDIT/customer-ov/c_matrix" "$IVIT_AUDIT/models/integration-classifier.onnx" openvino cpu
"$IVIT_AUDIT/customer-trt/c_matrix" "$IVIT_AUDIT/models/integration-classifier.onnx" tensorrt cuda:0

dotnet --version
dotnet build examples/csharp/Quickstart/Quickstart.csproj
export LD_LIBRARY_PATH="$IVIT_AUDIT/prefix-ov/lib:$IVIT_OV_LIB:$IVIT_CUDA_LIB"
dotnet run --project examples/csharp/Quickstart -- "$IVIT_AUDIT/models/integration-classifier.onnx" cpu
export LD_LIBRARY_PATH="$IVIT_AUDIT/prefix-trt/lib:$IVIT_CUDA_LIB"
dotnet run --project examples/csharp/Quickstart -- "$IVIT_AUDIT/models/integration-classifier.onnx" cuda:0
# 後續 Python 測試必須移除 native SDK 路徑，避免 libivit 版本遮蔽。
export LD_LIBRARY_PATH="$IVIT_OV_LIB:$IVIT_CUDA_LIB"
```

- [ ] C consumer 兩次皆成功退出，結果 JSON 有分類；短 buffer、double release、model 釋放後仍可讀結果皆通過。
- [ ] C# 使用 .NET 8；兩次都顯示 `C# lifecycle/buffer checks passed`，確認 JSON 的實際 device。
- [ ] TensorRT runtime 若在自訂路徑，將其 lib 目錄加入以上 LD_LIBRARY_PATH；不要加入另一個 SDK 的 lib。

## 13. 跑公開測試，保存跳過原因與總結

來源：[手冊 §9.2，L716](../sdk-manual.zh-TW.md)、[§3.5，L267](../sdk-manual.zh-TW.md)、[安裝／修正複驗紀錄](sdk-improvement-revalidation-20261014.zh-TW.md)。

```bash
export LD_LIBRARY_PATH="$IVIT_OV_LIB:$IVIT_CUDA_LIB"
export IVIT_C_LIBRARY="$IVIT_AUDIT/prefix-ov/lib/libivit.so"
source "$IVIT_AUDIT/py-ov/bin/activate"
python3 -m pytest tests/python -q -rs --junitxml="$IVIT_AUDIT/results/pytest-ov.xml" \
  2>&1 | tee "$IVIT_AUDIT/log/pytest-ov.log"
source "$IVIT_AUDIT/py-trt/bin/activate"
python3 -m pytest tests/python -q -rs --junitxml="$IVIT_AUDIT/results/pytest-trt.xml" \
  2>&1 | tee "$IVIT_AUDIT/log/pytest-trt.log"
python3 -m flake8 python scripts tests --select=E9,F63,F7
bash -n scripts/install_sdk.sh scripts/ci/build_and_test.sh
git diff --check
```

`IVIT_C_LIBRARY` 明確指向 OpenVINO native SDK，因為既有 C ABI Python suite 主要使用 CPU／OpenVINO；不能拿它宣稱 TensorRT C ABI 全套已測完，TensorRT 整合證據來自第 12 步。兩個 venv 若未安裝可選訓練依賴，相關跳過需明列。

- [ ] 安裝腳本／類別數／release contract 回歸通過。
- [ ] OpenVINO-only 的 TensorRT 測試跳過屬預期；TensorRT 環境不應因 backend 不可用而大量跳過。
- [ ] 不把歷史的 366 當固定通過數；以本次收集到的測試、版本與跳過原因為準。
- [ ] 留下四組安裝／建置、三任務結果、TensorRT 四份參考比對及兩後端 C／C# 結果。

如另需完整 CI 重現，可依 §3.5 執行 `bash scripts/ci/build_and_test.sh --work "$IVIT_AUDIT/ci-ov" --jobs 2` 與 `--tensorrt --work "$IVIT_AUDIT/ci-trt" --jobs 2`。這條路徑使用 pip 的 NVIDIA runtime，不替代第 4 步的真實系統安裝；腳本也會清理 source tree 中的 bindings／bytecode，宜在獨立 checkout 執行。

## 最終簽核表

| 項目 | 狀態（通過／失敗／未驗證） | 版本／裝置／日誌 |
| --- | --- | --- |
| Python OpenVINO：安裝＋三任務 | 通過 | OpenVINO 2026.4.1；cpu、gpu:0；`log/step03-install.log` |
| Python TensorRT：安裝＋三任務 FP32／FP16 | 通過（安裝路徑 B） | 實際載入 libnvinfer 11.4.0；cuda:0；`log/step04-loaded.log` |
| C++ OpenVINO：建置＋三任務＋外部專案 | 通過 | cpu、gpu:0；CTest 3/3 |
| C++ TensorRT：建置＋三任務 FP32／FP16＋外部專案 | 通過 | cuda:0；原先 `test_inference` 失敗，修正後 CTest 3/3 |
| OpenVINO CPU／GPU 參考比對 | 通過 | cpu、gpu:0 static／dynamic；NPU 未驗證（無硬體） |
| TensorRT 四種參考比對 | 通過 | FP32／FP16 × static／dynamic |
| C ABI：OpenVINO／TensorRT | 通過 | `c_matrix` |
| C#：OpenVINO／TensorRT | 通過 | .NET 8.0.425，使用完整 SDK 路徑 |
| 回歸測試、空間防護、CI 空測試防護 | 通過 | OpenVINO 359 通過／10 跳過；TensorRT 366 通過／3 跳過；空 CTest 回傳 8 為預期 |
| 真實 NVIDIA 自動安裝 | 未驗證 | 只跑路徑 B，未執行 `--install-nvidia` |
| 網路首次下載／torchvision 首次匯出 | 通過 | `log/step05-models.log`（分割匯出 training-mode warning 已修正，尚未重跑確認） |

仍需獨立安排的範圍：二進位 deb／tarball／wheel 安裝、sdist 重建、多 GPU、INT8、其他平台與 Python 版本、真實攝影機／長時間壓力、完整訓練與標註資料集精度。原始碼四組合通過，不等於這些項目都完成。
