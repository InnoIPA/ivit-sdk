# ADR-001: iVIT-SDK 系統架構設計

| 文件編號 | ADR-001 |
|----------|---------|
| 版本 | 1.0 |
| 狀態 | Proposed |
| 建立日期 | 2026-01-24 |
| 作者 | 系統架構師 |

---

## 1. 執行摘要

本文件定義 iVIT-SDK 的整體系統架構，包含核心組件設計、層次結構、後端適配器模式、以及關鍵技術決策的理由與權衡。

---

## 2. 架構目標

### 2.1 設計原則

| 原則 | 說明 |
|------|------|
| **統一抽象** | 對使用者隱藏不同硬體後端的差異 |
| **可擴展性** | 易於新增硬體後端和任務類型 |
| **效能優先** | 最小化抽象層開銷 |
| **雙語支援** | Python 和 C++ API 功能對等 |
| **向後相容** | API 穩定，版本演進不破壞 |

### 2.2 品質屬性需求

| 屬性 | 目標 | 測量方式 |
|------|------|----------|
| **效能** | 抽象層開銷 < 5% | Benchmark 比較 |
| **可用性** | API 學習曲線 < 1 小時 | 使用者測試 |
| **可維護性** | 程式碼覆蓋率 > 80% | CI 報告 |
| **可擴展性** | 新增後端 < 1 人週 | 開發時程 |
| **相容性** | 支援 3 個主要 Python 版本 | CI 測試 |

---

## 3. 系統架構概覽

### 3.1 高階架構圖

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           User Application Layer                             │
│                                                                              │
│    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│    │ Python App   │    │   C++ App    │    │  REST API    │                 │
│    └──────────────┘    └──────────────┘    └──────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              iVIT-SDK API Layer                              │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         Public API (ivit)                              │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │ │
│  │  │load_model│ │ predict  │ │  train   │ │ export   │ │ devices  │     │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     Task-Specific API (ivit.vision)                    │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │ │
│  │  │Classifier│ │ Detector │ │Segmentor │ │PoseEstim │ │FaceAnalyz│     │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      Training API (ivit.train)                         │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │ │
│  │  │ Trainer  │ │ Dataset  │ │Augmentat │ │ Callback │ │ Exporter │     │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Core Engine Layer                                 │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  Model Manager  │  │ Inference Engine│  │  Result Manager │             │
│  │  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │             │
│  │  │  Loader   │  │  │  │ Executor  │  │  │  │  Parser   │  │             │
│  │  │ Converter │  │  │  │ Scheduler │  │  │  │Visualizer │  │             │
│  │  │ Optimizer │  │  │  │ Memory Mgr│  │  │  │ Exporter  │  │             │
│  │  │   Cache   │  │  │  │ Profiler  │  │  │  │           │  │             │
│  │  └───────────┘  │  │  └───────────┘  │  │  └───────────┘  │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │ Training Engine │  │   Model Zoo     │  │  Device Manager │             │
│  │  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │             │
│  │  │  Trainer  │  │  │  │  Registry │  │  │  │ Discovery │  │             │
│  │  │ Optimizer │  │  │  │ Downloader│  │  │  │ Selector  │  │             │
│  │  │Data Loader│  │  │  │   Cache   │  │  │  │  Monitor  │  │             │
│  │  │Checkpoint │  │  │  │ Validator │  │  │  │           │  │             │
│  │  └───────────┘  │  │  └───────────┘  │  │  └───────────┘  │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Runtime Abstraction Layer                            │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      Unified Runtime Interface                         │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │  │  interface IRuntime {                                            │  │ │
│  │  │    load(model_path, config) -> Model                             │  │ │
│  │  │    infer(model, inputs) -> Outputs                               │  │ │
│  │  │    get_device_info() -> DeviceInfo                               │  │ │
│  │  │    optimize(model, config) -> OptimizedModel                     │  │ │
│  │  │  }                                                               │  │ │
│  │  └──────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Backend Adapter Layer                               │
│                                                                              │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                  │
│  │   OpenVINO    │  │   TensorRT    │  │ SNPE/QNN      │                  │
│  │    Adapter    │  │    Adapter    │  │   Adapter     │                  │
│  │               │  │               │  │  (Planned)    │                  │
│  │ ┌───────────┐ │  │ ┌───────────┐ │  │ ┌───────────┐ │                  │
│  │ │   Intel   │ │  │ │  NVIDIA   │ │  │ │ Qualcomm  │ │                  │
│  │ │CPU/GPU/NPU│ │  │ │ GPU/Jetson│ │  │ │  Hexagon  │ │                  │
│  │ └───────────┘ │  │ └───────────┘ │  │ └───────────┘ │                  │
│  └───────────────┘  └───────────────┘  └───────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Hardware Layer                                    │
│                                                                              │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐        │
│  │       Intel       │  │      NVIDIA       │  │     Qualcomm      │        │
│  │                   │  │                   │  │                   │        │
│  │  ┌─────┐ ┌─────┐  │  │  ┌─────┐ ┌─────┐  │  │  ┌─────┐ ┌─────┐  │        │
│  │  │ CPU │ │ GPU │  │  │  │dGPU │ │Jetson│  │  │  │ SoC │ │ IoT │  │        │
│  │  └─────┘ └─────┘  │  │  └─────┘ └─────┘  │  │  └─────┘ └─────┘  │        │
│  │  ┌─────┐ ┌─────┐  │  │  ┌─────┐          │  │  ┌─────┐          │        │
│  │  │ NPU │           │  │  │ MIG │          │  │  │Hexagon│         │        │
│  │  └─────┘           │  │  └─────┘          │  │  └─────┘          │        │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 層次職責

| 層次 | 職責 | 關鍵組件 |
|------|------|----------|
| **API Layer** | 使用者介面，提供易用的 API | load_model, predict, train |
| **Core Engine** | 核心業務邏輯，任務無關 | Model Manager, Inference Engine |
| **Runtime Abstraction** | 統一運行時介面 | IRuntime interface |
| **Backend Adapter** | 硬體後端適配 | OpenVINO, TensorRT, SNPE/QNN (規劃中) adapters |
| **Hardware** | 實際硬體設備 | Intel/NVIDIA/Qualcomm 硬體 |

---

## 4. 核心組件設計

### 4.1 Model Manager

負責模型的載入、轉換、快取和版本管理。

```
┌─────────────────────────────────────────────────────────────┐
│                       Model Manager                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   Loader    │  │  Converter  │  │  Optimizer  │          │
│  │             │  │             │  │             │          │
│  │ - ONNX      │  │ - PyTorch   │  │ - Quantize  │          │
│  │ - OpenVINO  │  │   → ONNX    │  │ - Prune     │          │
│  │ - TensorRT  │  │ - TF → ONNX │  │ - Fuse ops  │          │
│  │ - SNPE      │  │ - ONNX →    │  │             │          │
│  │             │  │   Backend   │  │             │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐                           │
│  │    Cache    │  │  Registry   │                           │
│  │             │  │             │                           │
│  │ - LRU cache │  │ - Model Zoo │                           │
│  │ - Disk cache│  │ - Versions  │                           │
│  │ - Memory    │  │ - Metadata  │                           │
│  └─────────────┘  └─────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

#### 類別設計

```cpp
// C++ Interface
namespace ivit {

class ModelManager {
public:
    // 載入模型
    std::shared_ptr<Model> load(
        const std::string& path,
        const LoadConfig& config = LoadConfig::default_()
    );

    // 從 Model Zoo 載入
    std::shared_ptr<Model> load_from_zoo(
        const std::string& model_name,
        const std::string& version = "latest"
    );

    // 轉換模型格式
    void convert(
        const std::string& src_path,
        const std::string& dst_path,
        const ConvertConfig& config
    );

    // 優化模型
    std::shared_ptr<Model> optimize(
        std::shared_ptr<Model> model,
        const OptimizeConfig& config
    );

private:
    std::unique_ptr<ModelCache> cache_;
    std::unique_ptr<ModelRegistry> registry_;
    std::vector<std::unique_ptr<ModelLoader>> loaders_;
    std::vector<std::unique_ptr<ModelConverter>> converters_;
};

struct LoadConfig {
    std::string device = "auto";      // auto, cpu, gpu:0, npu
    std::string backend = "auto";     // auto, openvino, tensorrt, snpe
    std::string task = "";            // classification, detection, etc.
    bool use_cache = true;
    int batch_size = 1;

    static LoadConfig default_();
};

} // namespace ivit
```

```python
# Python Interface
class ModelManager:
    def load(
        self,
        path: Union[str, Path],
        device: str = "auto",
        backend: str = "auto",
        task: Optional[str] = None,
        use_cache: bool = True,
        batch_size: int = 1,
    ) -> Model:
        """載入模型"""
        ...

    def load_from_zoo(
        self,
        model_name: str,
        version: str = "latest",
        **kwargs
    ) -> Model:
        """從 Model Zoo 載入"""
        ...

    def convert(
        self,
        src_path: Union[str, Path],
        dst_path: Union[str, Path],
        format: str,
        **kwargs
    ) -> None:
        """轉換模型格式"""
        ...

    def optimize(
        self,
        model: Model,
        quantize: Optional[str] = None,  # int8, fp16
        target_device: Optional[str] = None,
        calibration_data: Optional[Dataset] = None,
    ) -> Model:
        """優化模型"""
        ...
```

### 4.2 Inference Engine

負責模型推論執行，包含排程、記憶體管理和效能分析。

```
┌─────────────────────────────────────────────────────────────┐
│                      Inference Engine                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  Executor   │  │  Scheduler  │  │ Memory Mgr  │          │
│  │             │  │             │  │             │          │
│  │ - Sync exec │  │ - Batch     │  │ - Allocate  │          │
│  │ - Async exec│  │ - Priority  │  │ - Pool      │          │
│  │ - Stream    │  │ - Pipeline  │  │ - Release   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐                           │
│  │  Profiler   │  │ Pre/Post    │                           │
│  │             │  │ Processor   │                           │
│  │ - Latency   │  │             │                           │
│  │ - Throughput│  │ - Resize    │                           │
│  │ - Memory    │  │ - Normalize │                           │
│  │ - Ops       │  │ - NMS       │                           │
│  └─────────────┘  └─────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

#### 類別設計

```cpp
namespace ivit {

class InferenceEngine {
public:
    // 同步推論
    Results infer(
        std::shared_ptr<Model> model,
        const InputData& inputs,
        const InferConfig& config = InferConfig::default_()
    );

    // 非同步推論
    std::future<Results> infer_async(
        std::shared_ptr<Model> model,
        const InputData& inputs,
        const InferConfig& config = InferConfig::default_()
    );

    // 批次推論
    std::vector<Results> infer_batch(
        std::shared_ptr<Model> model,
        const std::vector<InputData>& inputs,
        const InferConfig& config = InferConfig::default_()
    );

    // 串流推論
    void infer_stream(
        std::shared_ptr<Model> model,
        std::shared_ptr<VideoStream> stream,
        std::function<void(Results)> callback,
        const InferConfig& config = InferConfig::default_()
    );

    // 效能分析
    ProfileReport profile(
        std::shared_ptr<Model> model,
        const InputData& sample_input,
        int iterations = 100
    );

private:
    std::unique_ptr<Scheduler> scheduler_;
    std::unique_ptr<MemoryManager> memory_manager_;
    std::unique_ptr<Profiler> profiler_;
    std::unique_ptr<PreProcessor> pre_processor_;
    std::unique_ptr<PostProcessor> post_processor_;
};

struct InferConfig {
    float conf_threshold = 0.5f;
    float iou_threshold = 0.45f;
    bool enable_profiling = false;
    int max_detections = 100;

    static InferConfig default_();
};

} // namespace ivit
```

### 4.3 Backend Adapter 設計

使用 **Adapter Pattern** 統一不同推論後端的介面。

```
┌─────────────────────────────────────────────────────────────┐
│                    IRuntime (Interface)                      │
├─────────────────────────────────────────────────────────────┤
│  + load(path, config) -> IModel                             │
│  + infer(model, inputs) -> Outputs                          │
│  + get_device_info() -> DeviceInfo                          │
│  + is_available() -> bool                                    │
│  + supported_formats() -> vector<string>                     │
│  + optimize(model, config) -> IModel                         │
└─────────────────────────────────────────────────────────────┘
                            △
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ OpenVINORuntime │ │ TensorRTRuntime │ │  QNNRuntime     │
│                 │ │                 │ │   (Planned)     │
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│ - Core          │ │ - Builder       │ │ - QNN Context   │
│ - CompiledModel │ │ - Runtime       │ │ - QNN Graph     │
│ - InferRequest  │ │ - Context       │ │ - QNN Tensor    │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

#### 介面定義

```cpp
namespace ivit {

// 統一運行時介面
class IRuntime {
public:
    virtual ~IRuntime() = default;

    // 載入模型
    virtual std::shared_ptr<IModel> load(
        const std::string& path,
        const RuntimeConfig& config
    ) = 0;

    // 推論
    virtual Outputs infer(
        std::shared_ptr<IModel> model,
        const Inputs& inputs
    ) = 0;

    // 設備資訊
    virtual DeviceInfo get_device_info() const = 0;

    // 可用性檢查
    virtual bool is_available() const = 0;

    // 支援的格式
    virtual std::vector<std::string> supported_formats() const = 0;

    // 優化模型
    virtual std::shared_ptr<IModel> optimize(
        std::shared_ptr<IModel> model,
        const OptimizeConfig& config
    ) = 0;
};

// 統一模型介面
class IModel {
public:
    virtual ~IModel() = default;

    virtual std::string name() const = 0;
    virtual std::vector<TensorInfo> input_info() const = 0;
    virtual std::vector<TensorInfo> output_info() const = 0;
    virtual size_t memory_usage() const = 0;
};

} // namespace ivit
```

#### OpenVINO Adapter 實作範例

```cpp
namespace ivit {

class OpenVINORuntime : public IRuntime {
public:
    OpenVINORuntime() {
        core_ = std::make_unique<ov::Core>();
    }

    std::shared_ptr<IModel> load(
        const std::string& path,
        const RuntimeConfig& config
    ) override {
        // 載入模型
        auto model = core_->read_model(path);

        // 編譯到目標設備
        std::string device = config.device;
        if (device == "auto") {
            device = select_best_device();
        }

        auto compiled = core_->compile_model(model, device);

        return std::make_shared<OpenVINOModel>(compiled);
    }

    Outputs infer(
        std::shared_ptr<IModel> model,
        const Inputs& inputs
    ) override {
        auto ov_model = std::static_pointer_cast<OpenVINOModel>(model);
        auto request = ov_model->create_infer_request();

        // 設定輸入
        for (const auto& [name, tensor] : inputs) {
            request.set_input_tensor(name, to_ov_tensor(tensor));
        }

        // 執行推論
        request.infer();

        // 取得輸出
        Outputs outputs;
        for (const auto& info : model->output_info()) {
            outputs[info.name] = from_ov_tensor(
                request.get_output_tensor(info.name)
            );
        }

        return outputs;
    }

    DeviceInfo get_device_info() const override {
        DeviceInfo info;
        info.backend = "openvino";
        info.devices = core_->get_available_devices();
        return info;
    }

    bool is_available() const override {
        return core_ != nullptr;
    }

    std::vector<std::string> supported_formats() const override {
        return {"onnx", "xml", "pdmodel"};
    }

private:
    std::unique_ptr<ov::Core> core_;

    std::string select_best_device() {
        // 優先順序: GPU > NPU > CPU
        auto devices = core_->get_available_devices();
        if (std::find(devices.begin(), devices.end(), "GPU") != devices.end()) {
            return "GPU";
        }
        if (std::find(devices.begin(), devices.end(), "NPU") != devices.end()) {
            return "NPU";
        }
        return "CPU";
    }
};

} // namespace ivit
```

#### TensorRT Adapter 實作範例

```cpp
namespace ivit {

class TensorRTRuntime : public IRuntime {
public:
    TensorRTRuntime() {
        logger_ = std::make_unique<TRTLogger>();
    }

    std::shared_ptr<IModel> load(
        const std::string& path,
        const RuntimeConfig& config
    ) override {
        // 檢查是否已有序列化 engine
        std::string engine_path = get_engine_path(path, config);

        if (std::filesystem::exists(engine_path)) {
            // 載入現有 engine
            return load_engine(engine_path);
        }

        // 從 ONNX 建立 engine
        auto builder = nvinfer1::createInferBuilder(*logger_);
        auto network = builder->createNetworkV2(
            1U << static_cast<uint32_t>(
                nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH
            )
        );

        auto parser = nvonnxparser::createParser(*network, *logger_);
        parser->parseFromFile(path.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

        // 設定建置選項
        auto build_config = builder->createBuilderConfig();
        if (config.enable_fp16) {
            build_config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        if (config.enable_int8) {
            build_config->setFlag(nvinfer1::BuilderFlag::kINT8);
            // 設定校正器...
        }

        // 建置 engine
        auto engine = builder->buildSerializedNetwork(*network, *build_config);

        // 序列化儲存
        save_engine(engine_path, engine);

        // 建立 runtime
        auto runtime = nvinfer1::createInferRuntime(*logger_);
        auto deserialized = runtime->deserializeCudaEngine(
            engine->data(), engine->size()
        );

        return std::make_shared<TensorRTModel>(deserialized);
    }

    // ... 其他方法實作

private:
    std::unique_ptr<TRTLogger> logger_;
};

} // namespace ivit
```

### 4.4 Device Manager

負責硬體設備的探索、選擇和監控。

```cpp
namespace ivit {

class DeviceManager {
public:
    // 列出所有可用設備
    std::vector<DeviceInfo> list_devices();

    // 取得最佳設備
    DeviceInfo get_best_device(
        const std::string& task = "",
        const ModelInfo& model_info = ModelInfo()
    );

    // 取得特定設備資訊
    DeviceInfo get_device(const std::string& device_id);

    // 監控設備狀態
    DeviceStatus get_device_status(const std::string& device_id);

    // 設備能力查詢
    bool supports_precision(const std::string& device_id, Precision precision);
    bool supports_format(const std::string& device_id, const std::string& format);

private:
    std::vector<std::unique_ptr<IRuntime>> runtimes_;
    mutable std::mutex mutex_;
    std::map<std::string, DeviceInfo> device_cache_;
};

struct DeviceInfo {
    std::string id;           // "cpu", "gpu:0", "npu", "cuda:0"
    std::string name;         // "Intel Core Ultra 7 165H"
    std::string backend;      // "openvino", "tensorrt", "snpe"
    std::string type;         // "cpu", "gpu", "npu", "vpu"
    size_t memory_total;      // bytes
    size_t memory_available;  // bytes
    std::vector<Precision> supported_precisions;
};

struct DeviceStatus {
    float utilization;        // 0.0 - 1.0
    size_t memory_used;       // bytes
    float temperature;        // Celsius
    float power_usage;        // Watts
};

} // namespace ivit
```

---

## 5. 資料流設計

### 5.1 推論資料流

```
                    ┌─────────────────────────────────────────┐
                    │              User Input                  │
                    │  (image path / numpy array / URL)        │
                    └─────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │             Input Validator              │
                    │  - Check file exists                     │
                    │  - Validate image format                 │
                    │  - URL download if needed                │
                    └─────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │             Pre-Processor                │
                    │  - Decode image (JPEG, PNG, etc.)        │
                    │  - Resize to model input size            │
                    │  - Color space conversion (BGR→RGB)      │
                    │  - Normalize (mean, std)                 │
                    │  - Convert to tensor (NCHW/NHWC)         │
                    └─────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │           Inference Engine               │
                    │  - Select runtime based on device        │
                    │  - Execute model inference               │
                    │  - Handle batching if needed             │
                    └─────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │            Post-Processor                │
                    │  - Decode model outputs                  │
                    │  - Apply NMS (for detection)             │
                    │  - Filter by confidence                  │
                    │  - Convert coordinates                   │
                    └─────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │            Result Object                 │
                    │  - Structured result data                │
                    │  - Visualization methods                 │
                    │  - Export methods (JSON, CSV)            │
                    └─────────────────────────────────────────┘
```

### 5.2 模型載入資料流

```
                    ┌─────────────────────────────────────────┐
                    │           Model Path/Name                │
                    │  (local path / URL / zoo name)           │
                    └─────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │            Path Resolver                 │
                    │  - Local file check                      │
                    │  - Model Zoo lookup                      │
                    │  - URL download                          │
                    └─────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │           Format Detector                │
                    │  - Detect model format                   │
                    │  - Read model metadata                   │
                    └─────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │           Cache Checker                  │
                    │  - Check if converted model cached       │
                    │  - Validate cache freshness              │
                    └─────────────────────────────────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         │ Cache Hit                   │ Cache Miss
                         ▼                             ▼
          ┌─────────────────────────┐  ┌─────────────────────────┐
          │    Load from Cache      │  │    Model Converter      │
          └─────────────────────────┘  │  - ONNX → Backend fmt   │
                         │             │  - Optimize for device  │
                         │             │  - Save to cache        │
                         │             └─────────────────────────┘
                         │                             │
                         └──────────────┬──────────────┘
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │          Runtime Loader                  │
                    │  - Load to selected backend              │
                    │  - Allocate device memory                │
                    │  - Return Model object                   │
                    └─────────────────────────────────────────┘
```

---

## 6. 專案結構

```
ivit-sdk/
├── CMakeLists.txt                    # 頂層 CMake 設定
├── pyproject.toml                    # Python 專案設定
├── README.md                         # 專案說明
├── LICENSE                           # 授權
│
├── include/                          # C++ 公開標頭
│   └── ivit/
│       ├── ivit.hpp                  # 主標頭
│       ├── core/
│       │   ├── common.hpp
│       │   ├── model.hpp
│       │   ├── device.hpp
│       │   └── runtime_config.hpp
│       ├── runtime/
│       │   ├── tensorrt_runtime.hpp
│       │   └── ...
│       └── utils/
│           └── ...
│
├── src/                              # C++ 實作
│   ├── core/
│   │   ├── model.cpp
│   │   └── device_manager.cpp
│   ├── runtime/
│   │   ├── runtime_factory.cpp
│   │   ├── runtime_init.cpp
│   │   ├── openvino_runtime.cpp
│   │   └── tensorrt_runtime.cpp
│   └── utils/
│       └── ...
│
├── python/                           # Python 綁定
│   └── ivit/
│       ├── __init__.py
│       ├── _binding.cpp              # pybind11 綁定
│       ├── core/
│       │   ├── __init__.py
│       │   ├── device.py
│       │   ├── exceptions.py
│       │   ├── runtime_config.py
│       │   └── types.py
│       ├── runtime/
│       │   ├── __init__.py
│       │   └── base.py
│       ├── devices.py
│       └── cli.py
│
├── models/                           # Model Zoo
│   └── ...
│
├── tests/                            # 測試
│   └── integration/
│       └── test_user_scenarios.py
│
├── examples/                         # 範例
│   ├── cpp/
│   │   └── ...
│   └── python/
│       ├── 01_quickstart.py
│       ├── 02_detection.py
│       ├── 02_classification.py
│       └── advanced/
│           ├── si_quickstart.py
│           ├── embedded_optimization.py
│           └── ...
│
├── docs/                             # 文件
│   ├── development/                  # 開發文件 (PRD, roadmap)
│   ├── architecture/                 # 架構文件
│   ├── api/                          # API 文件
│   ├── deployment/                   # 部署指南
│   └── tutorials/                    # 教學
│
├── scripts/                          # 工具腳本
│   └── download_deps.sh
│
└── docker/                           # Docker
    ├── Dockerfile.verify
    ├── Dockerfile.full
    └── docker-compose.yml
```

---

## 7. 技術決策記錄

### 7.1 ADR-001-01: 採用 ONNX 作為中間模型格式

**狀態**: Accepted

**背景**:
需要在不同框架 (PyTorch, TensorFlow) 和不同後端 (OpenVINO, TensorRT, SNPE) 之間轉換模型。

**決策**:
採用 ONNX 作為統一的中間模型格式。

**理由**:
- ONNX 是業界標準，主流框架都支援匯出
- OpenVINO、TensorRT、SNPE 都支援從 ONNX 轉換
- OpenVINO 可作為 fallback 後端 (CPU)
- 社群活躍，工具鏈完善

**後果**:
- ✅ 簡化模型轉換流程
- ✅ 使用者只需提供一種格式
- ⚠️ 可能有些運算子轉換不完美
- ⚠️ 需要維護 ONNX 版本相容性

### 7.2 ADR-001-02: 採用 pybind11 做 Python 綁定

**狀態**: Accepted

**背景**:
需要將 C++ 核心暴露給 Python 使用者。

**決策**:
採用 pybind11 作為 C++/Python 綁定工具。

**理由**:
- Header-only，易於整合
- 語法簡潔，學習曲線低
- 自動型別轉換（numpy, STL）
- 活躍維護，廣泛使用

**替代方案考慮**:
- Cython: 需要額外語法，適合純 Python 專案
- SWIG: 設定複雜，生成程式碼不直觀
- Python C API: 太底層，開發效率低

**後果**:
- ✅ 開發效率高
- ✅ 與 numpy 整合良好
- ⚠️ 編譯時間較長
- ⚠️ 需要與 Python 版本匹配

### 7.3 ADR-001-03: 後端選擇策略

**狀態**: Accepted

**背景**:
需要決定在 `device="auto"` 時如何選擇最佳後端。

**決策**:
採用以下優先順序：
1. 檢測可用硬體
2. 根據任務類型和模型大小選擇設備
3. 預設優先順序: GPU > NPU > CPU

**選擇邏輯**:
```
if NVIDIA GPU available:
    backend = TensorRT
    device = cuda:0
elif Intel GPU available:
    backend = OpenVINO
    device = GPU
elif Intel NPU available:
    backend = OpenVINO
    device = NPU
elif Qualcomm NPU available:  # (Planned)
    backend = SNPE/QNN
    device = DSP
else:
    backend = OpenVINO
    device = CPU
```

**後果**:
- ✅ 使用者無需了解底層細節
- ✅ 自動獲得最佳效能
- ⚠️ 可能選擇非最優設備
- ⚠️ 使用者需要能覆寫選擇

### 7.4 ADR-001-04: 模型快取策略

**狀態**: Accepted

**背景**:
TensorRT 需要建置 engine，SNPE 需要轉換 DLC，這些操作耗時。

**決策**:
實作兩層快取:
1. 記憶體快取: 已載入的模型實例
2. 磁碟快取: 轉換後的模型檔案

**快取路徑**:
```
~/.cache/ivit/
├── models/                     # 下載的原始模型
├── engines/                    # TensorRT engines
│   └── yolox-s_fp16_rtx4090.engine
├── dlc/                        # SNPE DLC 檔案
│   └── yolox-s_int8_qcs8550.dlc
└── openvino/                   # OpenVINO 快取
    └── yolox-s_int8_npu/
```

**快取 Key 包含**:
- 模型 hash
- 目標設備
- 精度設定
- 後端版本

**後果**:
- ✅ 首次載入後大幅加速
- ✅ 減少重複轉換
- ⚠️ 需要管理快取清理
- ⚠️ 磁碟空間佔用

---

## 8. 介面規格

### 8.1 核心資料結構

```cpp
namespace ivit {

// 張量資訊
struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    DataType dtype;
    Layout layout;  // NCHW, NHWC
};

// 資料型別
enum class DataType {
    Float32,
    Float16,
    Int8,
    UInt8,
    Int32,
    Int64,
};

// 張量佈局
enum class Layout {
    NCHW,
    NHWC,
    NC,
    CHW,
    HWC,
};

// 精度
enum class Precision {
    FP32,
    FP16,
    INT8,
    INT4,
};

// 邊界框
struct BBox {
    float x1, y1, x2, y2;

    float width() const { return x2 - x1; }
    float height() const { return y2 - y1; }
    float area() const { return width() * height(); }
    float iou(const BBox& other) const;
};

// 偵測結果
struct Detection {
    BBox bbox;
    int class_id;
    std::string label;
    float confidence;
    std::optional<cv::Mat> mask;  // for instance segmentation
};

// 分類結果
struct ClassificationResult {
    int class_id;
    std::string label;
    float score;
};

// 關鍵點
struct Keypoint {
    float x, y;
    float confidence;
    std::string name;
};

// 推論結果（統一容器）
class Results {
public:
    // 分類結果
    std::vector<ClassificationResult> classifications;

    // 偵測結果
    std::vector<Detection> detections;

    // 分割遮罩
    cv::Mat segmentation_mask;

    // 關鍵點
    std::vector<std::vector<Keypoint>> keypoints;

    // 原始輸出
    std::map<std::string, Tensor> raw_outputs;

    // 元資料
    float inference_time_ms;
    std::string device_used;

    // 方法
    cv::Mat visualize(const cv::Mat& image) const;
    std::string to_json() const;
    void save(const std::string& path) const;
};

} // namespace ivit
```

### 8.2 Python 型別對應

```python
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np

@dataclass
class TensorInfo:
    name: str
    shape: tuple
    dtype: str
    layout: str

@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: 'BBox') -> float:
        ...

@dataclass
class Detection:
    bbox: BBox
    class_id: int
    label: str
    confidence: float
    mask: Optional[np.ndarray] = None

@dataclass
class ClassificationResult:
    class_id: int
    label: str
    score: float

@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float
    name: str

class Results:
    classifications: List[ClassificationResult]
    detections: List[Detection]
    segmentation_mask: Optional[np.ndarray]
    keypoints: List[List[Keypoint]]
    raw_outputs: Dict[str, np.ndarray]
    inference_time_ms: float
    device_used: str

    def visualize(self, image: np.ndarray) -> np.ndarray:
        ...

    def to_json(self) -> str:
        ...

    def save(self, path: str) -> None:
        ...
```

---

## 9. 效能設計

### 9.1 效能目標

| 指標 | 目標 | 測量條件 |
|------|------|----------|
| 模型載入 | < 3 秒 | 首次載入（含轉換） |
| 模型載入（快取）| < 500 ms | 從快取載入 |
| 推論延遲 (YOLOX-Nano) | < 5 ms | RTX 4090, FP16, 640x640 |
| 推論延遲 (YOLOX-Nano) | < 10 ms | Jetson Orin, FP16, 640x640 |
| 推論延遲 (YOLOX-Nano) | < 15 ms | Intel Core Ultra NPU, INT8, 640x640 |
| 記憶體佔用 | < 500 MB | 典型模型 |
| API 開銷 | < 1 ms | Python wrapper 開銷 |

### 9.2 效能優化策略

#### 預處理優化
- 使用 SIMD 指令（SSE, AVX, NEON）
- 影像 resize 使用 GPU 加速
- 批次處理多張影像

#### 推論優化
- 啟用後端特定優化（TensorRT 層融合、OpenVINO 圖優化）
- 使用混合精度（FP16, INT8）
- 記憶體預分配，避免動態分配

#### 後處理優化
- NMS 使用 batched NMS
- 使用 numpy vectorization
- 避免不必要的資料複製

### 9.3 Benchmark 工具

```python
from ivit.utils import Profiler

# 建立 profiler
profiler = Profiler()

# 載入模型
model = ivit.load_model("yolox-s.onnx", device="cuda:0")

# 執行 benchmark
report = profiler.benchmark(
    model=model,
    input_shape=(1, 3, 640, 640),
    iterations=100,
    warmup=10,
)

print(report)
# Output:
# ┌─────────────────────────────────────────────────────────────┐
# │                    Benchmark Report                         │
# ├─────────────────────────────────────────────────────────────┤
# │ Model: yolox-s.onnx                                         │
# │ Device: NVIDIA GeForce RTX 4090 (cuda:0)                    │
# │ Backend: TensorRT 9.2                                       │
# │ Precision: FP16                                             │
# ├─────────────────────────────────────────────────────────────┤
# │ Latency:                                                    │
# │   Mean:     3.45 ms                                         │
# │   Median:   3.42 ms                                         │
# │   Std:      0.12 ms                                         │
# │   P95:      3.67 ms                                         │
# │   P99:      3.89 ms                                         │
# ├─────────────────────────────────────────────────────────────┤
# │ Throughput: 289.8 FPS                                       │
# │ Memory: 245 MB                                              │
# └─────────────────────────────────────────────────────────────┘
```

---

## 10. 安全性考量

### 10.1 輸入驗證

- 驗證模型檔案完整性（checksum）
- 檢查模型來源（Model Zoo 簽章）
- 限制輸入影像大小
- 防止路徑遍歷攻擊

### 10.2 資源管理

- 限制記憶體使用上限
- 設定推論超時
- 處理 OOM 錯誤
- 避免資源洩漏

### 10.3 模型安全

- 不執行不受信任的模型
- 模型檔案沙箱化
- 記錄模型來源

---

## 11. 監控與日誌

### 11.1 日誌等級

| 等級 | 用途 |
|------|------|
| DEBUG | 詳細除錯資訊 |
| INFO | 一般操作資訊 |
| WARNING | 警告訊息 |
| ERROR | 錯誤訊息 |
| CRITICAL | 嚴重錯誤 |

### 11.2 日誌設定

```python
import ivit

# 設定日誌等級
ivit.set_log_level("INFO")

# 設定日誌輸出
ivit.set_log_file("/var/log/ivit/sdk.log")

# 程式化設定
ivit.configure_logging(
    level="DEBUG",
    file="ivit.log",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    rotation="10MB",
    retention=7,
)
```

### 11.3 效能監控

```python
from ivit.utils import Monitor

monitor = Monitor()
monitor.start()

# 執行推論
for image in images:
    results = model.predict(image)

# 取得統計
stats = monitor.stop()
print(f"Total inferences: {stats.total_inferences}")
print(f"Average latency: {stats.avg_latency_ms:.2f} ms")
print(f"Peak memory: {stats.peak_memory_mb:.2f} MB")
```

---

## 12. 附錄

### A. 術語表

| 術語 | 定義 |
|------|------|
| Backend | 底層推論引擎（OpenVINO, TensorRT, SNPE/QNN（規劃中）） |
| Runtime | 模型執行環境 |
| Engine | TensorRT 編譯後的模型 |
| IR | OpenVINO 中間表示格式 |
| DLC | SNPE 模型格式 |
| NMS | 非極大值抑制 |

### B. 參考架構

- OpenVINO: https://docs.openvino.ai/
- TensorRT: https://developer.nvidia.com/tensorrt
- SNPE: https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk

---

**文件結束**
