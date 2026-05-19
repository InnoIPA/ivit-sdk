# 物件偵測訓練 (YOLOX)

iVIT-SDK 的 `DetectionTrainer` 封裝 **YOLOX** 訓練流程。YOLOX 為 Apache-2.0 授權，可放心用於商用部署。

| 後端 | 模型 | License | 推薦情境 |
|------|------|---------|---------|
| **YOLOX** | `yolox-{nano,tiny,s,m,l}` | Apache-2.0 | 通用 / 商用部署 |

## 安裝

YOLOX 的官方套件 `setup.py` 在 build 時就要 `import torch`，但 pip 的 PEP 517
build isolation 會建立沒有 torch 的乾淨 build 環境，導致一般 `pip install yolox` 直接失敗：

```
AssertionError: torch is required for pre-compiling ops, please install it first.
```

**正確安裝方式**（先裝 torch 與其他相依，再用 `--no-build-isolation` 裝 yolox 本身）：

```bash
pip install -e ".[training-yolox-deps]"   # torch / torchvision / pycocotools / loguru ...
pip install --no-build-isolation \
    'git+https://github.com/Megvii-BaseDetection/YOLOX.git'
```

> PyPI 上的 `yolox==0.3.0` 已多年沒更新且打包不全（連 `requirements.txt` 都讀不到），
> 請直接從 GitHub 主分支安裝。

## 訓練

### 資料準備

YOLOX 走 COCO format，目錄如下：

```
datasets/coco128/
├── images/
│   ├── train2017/   *.jpg
│   └── val2017/     *.jpg
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

### Python API

```python
from ivit.train import DetectionTrainer

trainer = DetectionTrainer(
    model="yolox-s",
    data_dir="./datasets/coco128",
    train_ann="instances_train2017.json",
    val_ann="instances_val2017.json",
    num_classes=80,
    epochs=100,
    batch_size=8,
    image_size=640,
    device="cuda:0",
    output_dir="runs/yolox-s",
)
trainer.fit()
trainer.export("runs/yolox-s/yolox-s.onnx")
```

### CLI

```bash
ivit train --task detect --model yolox-s \
           --data ./datasets/coco128 \
           --ann instances_train2017.json \
           --val-ann instances_val2017.json \
           --epochs 100 --batch-size 8 --img-size 640 \
           --device cuda:0 --output runs/yolox-s --export-onnx
```

匯出的 ONNX 直接可用 `ivit predict` 跑推論：

```bash
ivit predict runs/yolox-s/yolox-s.onnx demo.jpg --conf 0.5
```

## 常見問題

- **`pip install yolox` 在 Windows 失敗**：YOLOX 對 `pycocotools` 與 `loguru` 在 Windows 上的相容性較差；建議在 Linux 環境訓練，匯出 ONNX 後再到目標平台部署。
- **訓練後拿不到 mAP**：YOLOX 會在自己的 log 裡印；iVIT 的 `history` 只回最終 metric dict。

## 參考

- [iVIT-SDK 自訂模型教學](custom-model.md)
- [語意分割訓練教學](training-segmentation.md)
- [YOLOX 官方倉庫](https://github.com/Megvii-BaseDetection/YOLOX)
