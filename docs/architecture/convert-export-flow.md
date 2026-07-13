# iVIT-SDK 模型轉檔／匯出流程圖

- **Convert（轉檔）**：檔案轉檔案（ONNX / IR → 目標格式），C++ core 優先，Python fallback。
- **Export（匯出）**：訓練好的 PyTorch `nn.Module` → 各格式，純 Python，一定先過 ONNX 中間檔。

---

## 一、Convert 流程（檔案轉檔案）

```mermaid
flowchart TD
    START([ivit convert]):::cli --> DISP{格式?}:::dec

    DISP -->|openvino| OV{int8?}:::dec
    OV -->|"是 (+calib)"| NNCF["NNCF 量化<br/>純 Python"]:::py
    OV -->|否| CORE
    DISP -->|tensorrt| CORE
    DISP -->|onnx| COPY["複製 / torch 匯出"]:::py

    CORE{"C++ core 可用?"}:::dec
    CORE -->|是| CPP["C++ convert_model"]:::cpp
    CORE -->|否 fallback| PYTOOL["ovc / openvino / tensorrt<br/>純 Python"]:::py

    CPP --> BE{device 決定 backend}:::dec
    BE -->|cpu/gpu/npu| OVRT["OpenVINO<br/>read_model + save_model"]:::cpp
    BE -->|cuda| TRTRT["TensorRT<br/>parse ONNX + build engine"]:::cpp

    NNCF --> OUTX(["MODEL.xml + .bin"]):::out
    OVRT --> OUTX
    PYTOOL --> OUTX
    TRTRT --> OUTE(["MODEL.engine"]):::out
    COPY --> OUTO(["MODEL.onnx"]):::out

    classDef cli fill:#1e3a5f,stroke:#4a90d9,color:#fff
    classDef py fill:#2d4a22,stroke:#6aa84f,color:#fff
    classDef cpp fill:#5f1e3a,stroke:#d94a90,color:#fff
    classDef dec fill:#333333,stroke:#f0c040,color:#f0c040
    classDef out fill:#1e4a3a,stroke:#4ad9a9,color:#fff
```

---

## 二、Export 流程（PyTorch nn.Module 匯出）

```mermaid
flowchart TD
    START([ivit train --export]):::cli --> TTYPE{任務?}:::dec

    TTYPE -->|Detection| YOLOX["YOLOX backend<br/>只出 ONNX"]:::py
    YOLOX --> OUTO(["MODEL.onnx"]):::out

    TTYPE -->|Classify / Segment| DISP{format?}:::dec

    DISP -->|torchscript| TS["torch.jit.trace"]:::api
    TS --> OUTTS(["MODEL (TorchScript)"]):::out

    DISP -->|onnx| ONNX["torch.onnx.export"]:::api
    ONNX --> OUTO

    DISP -->|openvino| OV["先 torch→ONNX<br/>再 read_model + serialize"]:::api
    OV --> OUTX(["MODEL.xml + .bin"]):::out

    DISP -->|tensorrt| TRT["先 torch→ONNX<br/>再 build engine"]:::api
    TRT --> OUTE(["MODEL.engine"]):::out

    classDef cli fill:#1e3a5f,stroke:#4a90d9,color:#fff
    classDef py fill:#2d4a22,stroke:#6aa84f,color:#fff
    classDef api fill:#3a3a3a,stroke:#999999,color:#fff
    classDef dec fill:#333333,stroke:#f0c040,color:#f0c040
    classDef out fill:#1e4a3a,stroke:#4ad9a9,color:#fff
```

---

## 三、關鍵差異

| 面向 | Convert | Export |
|------|---------|--------|
| 源頭 | 檔案（ONNX / IR） | PyTorch `nn.Module` |
| 執行位置 | C++ core 優先，Python fallback | 純 Python |
| 中間 ONNX | 不需要 | 一定先產生 |
| INT8 真校正 | 僅 OpenVINO（NNCF） | 僅 OpenVINO（NNCF） |
