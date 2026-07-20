# iVIT-SDK 文件

**Innodisk Vision Intelligence Toolkit** — 統一的電腦視覺推論與訓練 SDK。

提供跨硬體平台的一致性 API：一套程式碼即可在 Intel（OpenVINO）、NVIDIA（TensorRT）等平台上執行分類、物件偵測、語意分割與姿態估計，並支援遷移式學習與模型匯出。

## 快速導覽

- **[快速上手](getting-started.md)** — 安裝環境、跑第一個推論。
- **[使用手冊](user-guide.md)** — 常見任務與工作流程。
- **[Model Zoo](model-zoo.md)** — 內建模型清單與下載。
- **[Python API Reference](api/reference/index.md)** — 從程式碼自動產生，可查詢每個類別 / 函式的簽名、參數與範例。
- **[C++ API Reference](api/cpp/html/index.html)** — 由 Doxygen 從 `include/ivit/` 產生的 C++ 公開 API。

## API 文件怎麼看

| 你在寫的是… | 看這裡 |
|-------------|--------|
| Python（`import ivit`） | [Python API Reference](api/reference/index.md) |
| C++（`#include <ivit/ivit.hpp>`） | [C++ API Reference](api/cpp/html/index.html) |

!!! tip "Python 使用者請優先查 `ivit.vision.*`"
    高階視覺任務類別的完整文件（含參數說明與範例）位於 `ivit.vision.Detector` / `Classifier` / `Segmentor`。
    頂層的 `ivit.Detector` 等是 C++ 綁定的原生類別，說明較精簡。

## 硬體支援

| 廠商 | 硬體類型 | 後端 |
|------|---------|------|
| Intel | CPU、iGPU、NPU | OpenVINO |
| NVIDIA | dGPU、Jetson | TensorRT |
| Qualcomm | IQ9/IQ8/IQ6 | QNN[規劃中] |
