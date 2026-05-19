# 語意分割訓練 (Semantic Segmentation)

iVIT-SDK 的 `SegmentationTrainer` 以 torchvision 的 DeepLabV3 為基礎做 fine-tune，支援：

| 模型 | Backbone | License |
|------|----------|---------|
| `deeplabv3-resnet50` | ResNet-50 | BSD-3-Clause |
| `deeplabv3-mobilenetv3` | MobileNetV3-Large | BSD-3-Clause |

兩者在 Model Zoo 中皆已登錄（COCO 預訓練），可直接套用到自有資料集做 transfer learning。

## 安裝

```bash
pip install -e ".[train]"
```

## 資料規格

`SegmentationDataset` 需要兩個資料夾：影像與遮罩。

```
voc_mini/
├── JPEGImages/
│   ├── 000001.jpg
│   ├── 000002.jpg
└── SegmentationClass/
    ├── 000001.png
    └── 000002.png
```

**遮罩規格**：

- 灰階 PNG（`uint8`）
- 像素值 = 類別索引（0..num_classes-1）
- 像素值 `255` 視為 ignore（訓練/評估時跳過）
- 也支援 PIL palette PNG（讀進來會被當作灰階）

## Python API

```python
from ivit.train import SegmentationTrainer, SegmentationDataset

train_ds = SegmentationDataset(
    images_dir="./voc_mini/JPEGImages",
    masks_dir="./voc_mini/SegmentationClass",
    class_names=["bg", "person", "car", "tree"],
    train_split=0.8, split="train",
)
val_ds = SegmentationDataset(
    images_dir="./voc_mini/JPEGImages",
    masks_dir="./voc_mini/SegmentationClass",
    class_names=["bg", "person", "car", "tree"],
    train_split=0.8, split="val",
)

trainer = SegmentationTrainer(
    model="deeplabv3-mobilenetv3",
    dataset=train_ds,
    val_dataset=val_ds,
    epochs=20,
    learning_rate=1e-4,
    batch_size=8,
    device="cuda:0",
    image_size=520,
    aux_loss=True,
    ignore_index=255,
    freeze_backbone=False,    # 全模型 fine-tune；要凍結 backbone 設 True
)
history = trainer.fit()
metrics = trainer.evaluate()
print(f"mIoU: {metrics['val_miou']:.4f}, pixel acc: {metrics['val_pixel_acc']:.4f}")

trainer.export("seg.onnx", format="onnx")     # ONNX
trainer.export("seg.xml",  format="openvino") # OpenVINO IR
```

> Trainer 的 `export()` 會自動把 torchvision 的 `OrderedDict({"out", "aux"})` 包成單一 tensor 輸出
> （`logits`），並設好動態的 batch / height / width 軸，下游可任意大小推論。

## CLI

```bash
ivit train --task segment \
           --model deeplabv3-mobilenetv3 \
           --data ./voc_mini/JPEGImages \
           --masks ./voc_mini/SegmentationClass \
           --epochs 20 --batch-size 8 --img-size 520 \
           --device cuda:0 --output runs/seg --export-onnx
```

## 評估指標

`evaluate()` / 訓練 epoch 結尾的 `val_*` 指標：

| 名稱 | 意義 |
|------|------|
| `val_loss` | CrossEntropy（含 aux loss × 0.4） |
| `val_miou` | 全類別平均 IoU；類別 IoU 為 NaN 時自動跳過（`np.nanmean`） |
| `val_pixel_acc` | 全像素正確率（不含 ignore） |

要監控 mIoU 早停：

```python
from ivit.train import EarlyStopping, ModelCheckpoint
trainer.fit(callbacks=[
    EarlyStopping(monitor="val_miou", mode="max", patience=5),
    ModelCheckpoint("best_seg.pt", monitor="val_miou", mode="max"),
])
```

## 參考

- [iVIT-SDK 自訂模型教學](custom-model.md)
- [物件偵測訓練教學](training-detection.md)
- [torchvision DeepLabV3](https://pytorch.org/vision/stable/models/deeplabv3.html)
