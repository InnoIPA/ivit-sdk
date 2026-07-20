# Vision

高階視覺任務類別。每個類別封裝了各自的前處理（resize、normalize）與後處理（softmax、NMS、mask decode），透過統一的 `predict()` 介面回傳 `Results`（其欄位說明見 [C++ API Reference](../cpp/html/classivit_1_1_results.html)）。

!!! note
    這裡指向具體模組路徑（`ivit.vision.detector` 等）以取得完整方法文件；實際使用時直接寫 `ivit.vision.Detector` 或 `ivit.Detector` 即可。

## Detector

::: ivit.vision.detector.Detector

## Classifier

::: ivit.vision.classifier.Classifier

## Segmentor

::: ivit.vision.segmentor.Segmentor
