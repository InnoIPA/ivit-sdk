from hailo_sdk_client import ClientRunner

model_name = "yolox-s"
onnx_path = "models/onnx/yolox-s.onnx"

# 初始化 Runner 並解析模型
runner = ClientRunner()
hn, npz = runner.translate_onnx_model(onnx_path, model_name)
runner.save_har("models/hailo/yolox_s.har")