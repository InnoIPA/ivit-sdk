# Devices & CLI

## 裝置探索

`ivit.devices` 是 `DeviceManager` 的單例，用於列舉與挑選硬體裝置。

::: ivit.devices.DeviceManager

::: ivit.devices.Device

## 命令列工具

安裝後可使用 `ivit` 命令（進入點 `ivit.cli:main`）：

```bash
ivit info                                   # 環境與版本資訊
ivit devices                                # 列出可用裝置
ivit benchmark models/yolox-s.onnx --device cuda:0
ivit predict models/yolox-s.onnx image.jpg --conf 0.5
ivit convert models/yolox-s.onnx -f openvino -o models/
ivit zoo list
ivit zoo download yolox-s
```

::: ivit.cli.main
