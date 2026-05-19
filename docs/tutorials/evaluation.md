# 模型評估 — Detection (COCO mAP)

iVIT-SDK 透過 `YoloxPredictor.evaluate()` 與 `ivit eval` CLI 對訓練後的 YOLOX 模型做標準
COCO mAP 評估（含 mAP@0.5:0.95、mAP@0.5、mAP@0.75、AR、per-class AP）。

底層用 `pycocotools.cocoeval`（業界 reference implementation），跑出來的數字與
YOLOX / Ultralytics / MMDetection 完全可比。

> 目前評估 API **只支援 detection**（COCO format）。Classification 與 segmentation 的
> 評估邏輯內建在各自 Trainer 的 `_evaluate()` 內，會在 `Trainer.fit()` 的 epoch 結尾自動跑。

## 安裝

```bash
pip install pycocotools   # YOLOX 訓練 extras 已包含
```

## 資料格式

需要兩樣：

1. **COCO annotation JSON**（標準 `instances_xxx.json` 格式）
2. **影像資料夾**（與 annotation `images[].file_name` 對應）

```
my_dataset/
├── annotations/
│   └── instances_val2017.json
└── val2017/
    ├── 000000000139.jpg
    ├── 000000000285.jpg
    └── ...
```

若你只有 YOLO format 標註，可以參考 `tests/python/` 內的 `yolo_to_coco.py` 做格式轉換。

## Python API

```python
from ivit.vision.yolox_predictor import YoloxPredictor

predictor = YoloxPredictor(
    "runs/yolox-nano/yolox-nano.onnx",
    device="cuda:0",
)

report = predictor.evaluate(
    annotation_file="my_dataset/annotations/instances_val2017.json",
    images_dir="my_dataset/val2017",
    conf=0.001,   # 評估用：盡量低，讓 COCO eval 自己 sweep PR curve
    iou=0.65,     # NMS IoU（YOLOX eval 慣例）
    verbose=True,
)

print(f"mAP @ 0.50:0.95 : {report.mAP:.4f}")
print(f"mAP @ 0.50      : {report.mAP_50:.4f}")
print(f"mAP @ 0.75      : {report.mAP_75:.4f}")
print(f"AR  @ 100 dets  : {report.AR_100:.4f}")
print(f"per-class AP    : {report.per_class_AP}")
```

`EvalReport.to_dict()` 可序列化成 JSON 存檔，供後續分析或 dashboard。

## CLI

```bash
ivit eval runs/yolox-nano/yolox-nano.onnx \
    --ann   my_dataset/annotations/instances_val2017.json \
    --images my_dataset/val2017 \
    --conf  0.001 \
    --iou   0.65 \
    --device cuda:0 \
    --output runs/yolox-nano/eval_report.json
```

輸出（節錄）：

```
[ivit eval] model: runs/yolox-nano/yolox-nano.onnx
[ivit eval] device: cuda:0  conf: 0.001  iou: 0.65

... pycocotools summary ...

==================================================
  mAP @ IoU=0.50:0.95  : 0.5159
  mAP @ IoU=0.50       : 0.7354
  mAP @ IoU=0.75       : 0.5695
  AR  @ maxDets=100    : 0.5774
  images: 128  detections: 14214
==================================================
Per-class AP (IoU=0.50:0.95):
  zebra                0.8886
  airplane             0.8842
  motorcycle           0.8827
  ... (70 more classes)
```

## 評估時的常見參數

| 參數 | 預設 | 說明 |
|------|------|------|
| `conf` | `0.001` | 越低越好；COCO eval 會自己 sweep precision-recall curve，過早門檻會壓低 mAP |
| `iou` | `0.65` | NMS IoU，與 YOLOX 訓練評估慣例一致 |
| `max_detections` | `300` | 每張圖保留多少偵測進 COCO eval；標準 cap=100，YOLOX 慣用 300 |
| `category_id_map` | `None` | model class_id（0~num_classes-1）→ COCO category_id。預設依 annotation 內 categories 出現順序 1:1 對應；當你用真實 COCO（category_id 是離散 1, 2, 3, ..., 90）就要自己提供對應表 |

## 結果如何解讀

| 指標 | 嚴格度 | 適合場景 |
|------|--------|---------|
| **mAP @ 0.50:0.95** | 嚴 | 主要對比指標（學術論文、模型 leaderboard） |
| **mAP @ 0.50** | 寬 | 比較與 PASCAL VOC、舊 YOLO 版本 |
| **mAP @ 0.75** | 嚴 | 看模型定位精度（box 對齊好不好） |
| **AR @ 100 dets** | — | 看 recall 上限（如果太低，代表很多物件根本沒被偵測到，不是 conf 門檻問題） |

per-class AP 拿來找弱點類別。如果某類 AP 顯著低於平均（例如 person 0.85、handbag 0.10），
通常是該類訓練樣本不足或物件偏小，應該補資料或調 anchor 配置。

## 範例：與訓練 log 自評對拍

YOLOX 訓練過程中 `train_log.txt` 結尾會有 yolox 自己跑的 COCO eval 結果。我們的
`YoloxPredictor.evaluate()` 應該跟它非常接近（差異 ±1% 內，來自 NMS conf 門檻、TRT FP32
與 PyTorch FP32 數值小幅差異）。

```bash
# yolox 自評（train log 末尾）
grep "Average Precision  (AP)" runs/yolox-nano/yolox-nano/train_log.txt | tail -3

# iVIT 評估（用 ONNX → TRT → 我們的 post-processor）
ivit eval runs/yolox-nano/yolox-nano.onnx \
    --ann   /path/to/instances_val2017.json \
    --images /path/to/val2017
```

兩個數字若差距 > 2%，通常代表 ONNX export 階段有問題（例如 `decode_in_inference` 設錯、
opset 版本與 C++ runtime 不相容、static vs dynamic shape 引發某層 fallback 等）；可參考
`docs/development/yolox-training-bug-log.md` 排查。
