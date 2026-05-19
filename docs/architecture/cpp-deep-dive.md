# iVIT-SDK C++ 深度分析

> 版本：1.2.0 | 標準：C++17 | 最後更新：2026-04-13

---

## 目錄

1. [模組架構圖](#1-模組架構圖)
2. [各模組詳解](#2-各模組詳解)
3. [完整呼叫流程](#3-完整呼叫流程)
4. [多後端擴展機制](#4-多後端擴展機制)
5. [資料流](#5-資料流)
6. [關鍵設計模式](#6-關鍵設計模式)
7. [各後端差異比較](#7-各後端差異比較)
8. [編譯與建置系統](#8-編譯與建置系統)
9. [執行流程總結](#9-執行流程總結)
10. [實作細節深潛](#10-實作細節深潛)

---

## 1. 模組架構圖

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                        │
│              (使用者應用程式 - ivit::vision::*)                 │
└──────────────────┬──────────────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────────────┐
│                    Vision Module Layer                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────────┐            │
│  │Classifier  │  │ Detector   │  │ Segmentor      │            │
│  │(高階 API)  │  │(高階 API)  │  │ (高階 API)     │            │
│  └────────────┘  └────────────┘  └────────────────┘            │
└──────────────────┬──────────────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────────────┐
│                     Core Model Layer                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Model (抽象)  -> InferenceModel (具體實現)               │   │
│  │ - predict()   - predict_batch()  - predict_async()       │   │
│  │ - infer_raw() - warmup()         - predict_tta()         │   │
│  │ - stream()    - predict_concurrent()                     │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────┬──────────────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────────────┐
│            Runtime Abstraction Layer (IRuntime)                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │RuntimeFactory (工廠模式) + RuntimeAutoRegister (自動註冊) │   │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────┬──────────────────────────────────────────────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
┌───────▼──┐ ┌────▼────┐ ┌──▼──────────┐
│OpenVINO  │ │TensorRT │ │QNN Runtime  │
│Runtime   │ │Runtime  │ │ (Qualcomm)  │
│(Intel)   │ │(NVIDIA) │ │             │
└──────────┘ └─────────┘ └─────────────┘
      │            │             │
┌─────┴────────────┴─────────────┴──────┐
│              Device Layer             │
├──────────────┬──────────┬─────────────┤
│ CPU / iGPU   │ GPU      │ HTP/Adreno  │
│ NPU / VPU    │ (CUDA)   │ DSP / CPU   │
│ (Intel)      │ (NVIDIA) │ (Qualcomm)  │
└──────────────┴──────────┴─────────────┘
```

**支撐服務層：**
```
┌────────────────────────────────────────────────────────────┐
│                    Support Services                        │
├────────────────────────────────────────────────────────────┤
│ DeviceManager    - 設備發現與管理                           │
│ CallbackManager  - 事件回調系統                            │
│ VideoSource      - 視頻源抽象層                            │
│ Tensor           - 張量容器 (CPU 側)                       │
│ Results          - 推論結果統一容器                         │
│ RuntimeConfig    - 各後端運行時配置                        │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│                    Utility Services                        │
├────────────────────────────────────────────────────────────┤
│ Profiler         - 性能基準測試與統計                       │
│ Visualizer       - 推論結果可視化                           │
└────────────────────────────────────────────────────────────┘
```

**檔案對應：**
```
include/ivit/
├── ivit.hpp                    # 全域入口 API
├── core/
│   ├── common.hpp              # 枚舉、基礎資料結構、異常
│   ├── tensor.hpp              # Tensor / TensorInfo
│   ├── model.hpp               # Model 抽象 + InferenceModel
│   ├── result.hpp              # BBox / Detection / Results
│   ├── device.hpp              # DeviceInfo / DeviceManager
│   ├── runtime_config.hpp      # OpenVINOConfig / TensorRTConfig / QNNConfig
│   ├── callback.hpp            # CallbackEvent / CallbackManager
│   └── video_source.hpp        # VideoSource
├── runtime/
│   ├── runtime.hpp             # IRuntime / RuntimeFactory
│   ├── openvino_runtime.hpp    # OpenVINORuntime
│   ├── tensorrt_runtime.hpp    # TensorRTRuntime
│   └── qnn_runtime.hpp         # QNNRuntime
├── vision/
│   ├── classifier.hpp          # Classifier
│   ├── detector.hpp            # Detector
│   └── segmentor.hpp           # Segmentor
└── utils/
    ├── profiler.hpp            # Profiler / ScopedTimer
    └── visualizer.hpp          # Visualizer
```

---

## 2. 各模組詳解

### 2.1 Core 模組

#### `common.hpp` — 全域定義

所有模組共享的基礎枚舉與資料結構。

```cpp
enum class DataType  { Float32, Float16, Int8, UInt8, Int32, Int64, Bool, Unknown };
enum class Layout    { NCHW, NHWC, NC, CHW, HWC, Unknown };
enum class TaskType  { Classification, Detection, Segmentation, PoseEstimation, ... };
enum class BackendType { OpenVINO, TensorRT, QNN, Auto, Unknown };

struct LoadConfig {
  std::string device;    // "cpu", "cuda:0", "htp"
  std::string backend;   // "openvino", "tensorrt", "qnn"
  std::string precision; // "fp32", "fp16", "int8"
};
```

**異常層次：**
```
IVITError
├── ModelError      - 模型加載/格式錯誤
├── DeviceError     - 設備不可用
└── InferenceError  - 推論執行失敗
```

---

#### `tensor.hpp` — 張量容器

CPU 側輕量張量，所有後端的輸入/輸出交換格式。

| 類別 | 關鍵方法 | 說明 |
|------|----------|------|
| `TensorInfo` | `numel()`, `byte_size()` | 形狀描述，無實際資料 |
| `Tensor` | `data_ptr<T>()`, `reshape()`, `clone()` | 含資料的張量容器 |

```cpp
// 內部儲存
std::vector<uint8_t> data_;   // 原始字節

// 類型安全存取
float* ptr = tensor.data_ptr<float>();
```

---

#### `model.hpp` — 模型推論核心

```
Model (抽象基類)
└── InferenceModel (具體實現，持有 IRuntime + void* handle)
```

**Model 關鍵 API：**

| 類別 | 方法 | 說明 |
|------|------|------|
| 屬性 | `name()`, `task()`, `device()`, `backend()` | 模型基本信息 |
| 資訊 | `input_info()`, `output_info()`, `memory_usage()` | 張量規格 |
| 推論 | `predict(cv::Mat)`, `predict_batch(vector<Mat>)` | 同步推論 |
| 原始 | `infer_raw(map<string, Tensor>)` | 繞過前後處理 |
| 異步 | `predict_async()`, `predict_batch_async()` | 非阻塞推論 |
| 並發 | `predict_concurrent()` | 帶限流並發 |
| 回調 | `submit_inference()` | Fire-and-forget 風格 |
| 流式 | `stream(source)` → `StreamIterator` | 逐幀迭代 |
| 測試 | `predict_tta()` | Test-Time Augmentation |
| 硬體 | `configure_openvino/tensorrt/qnn()` | 後端特定調優 |

**InferenceModel 內部：**
- `shared_ptr<IRuntime> runtime_` — 後端實例
- `void* handle_` — 後端模型句柄
- 非同步執行通過 `std::thread + std::queue`
- 生命週期：`preprocess → infer → postprocess`

---

#### `result.hpp` — 推論結果

```cpp
struct BBox {
  float x1, y1, x2, y2;
  float iou(const BBox& other) const;
  BBox  to_xywh() const;
  cv::Rect to_rect() const;
};

struct Detection {
  BBox   bbox;
  float  confidence;
  int    class_id;
  std::string label;
  cv::Mat mask;   // 實例分割用
};

class Results {
  std::vector<ClassificationResult> classifications;
  std::vector<Detection>            detections;
  std::vector<Pose>                 poses;
  cv::Mat                           segmentation_mask;
  std::map<std::string, Tensor>     raw_outputs;

  // 過濾
  Results filter_by_class(int class_id) const;
  Results filter_by_confidence(float threshold) const;

  // 輸出
  cv::Mat visualize(const cv::Mat& image) const;
  std::string to_json() const;
  void save(const std::string& path) const;
};
```

---

#### `device.hpp` — 設備管理

```cpp
struct DeviceInfo {
  std::string id;          // "cpu", "cuda:0", "htp"
  std::string name;        // 顯示名稱
  BackendType backend;
  std::string type;        // "cpu", "gpu", "npu"
  size_t      memory_mb;
  std::vector<std::string> supported_precisions;
};

class DeviceManager {          // 單例
  std::vector<DeviceInfo> list_devices();
  DeviceInfo get_best_device(const std::string& task,
                              const std::string& priority); // "performance"|"efficiency"|"memory"
  // 後端特定發現
  void discover_openvino_devices();
  void discover_tensorrt_devices();
  void discover_qnn_devices();    // dlopen probe libQnnHtp/Gpu/Cpu.so
};

// 輔助函數
std::pair<BackendType, std::string> parse_device_string(const std::string& s);
bool cuda_is_available();
bool openvino_is_available();
bool qnn_is_available();
```

**設備字符串對應表：**
```
"cpu"               → OpenVINO CPU
"gpu:0"             → OpenVINO GPU.0 (Intel iGPU)
"npu"               → OpenVINO NPU (Intel AI Boost)
"vpu"               → OpenVINO VPU (Movidius)
"cuda:0"            → TensorRT CUDA:0
"htp" / "iq6"~"iq9" → QNN HTP (Hexagon NPU)
"hexagon" / "dsp"   → QNN HTP
"qnn:gpu"           → QNN Adreno GPU
"qnn:cpu"           → QNN CPU
```

---

#### `runtime_config.hpp` — 後端配置

```cpp
struct OpenVINOConfig {
  std::string performance_mode = "LATENCY";  // LATENCY | THROUGHPUT
  int         num_streams      = 1;
  std::string precision        = "auto";
  std::string cache_dir        = "";
  std::map<std::string,std::string> to_ov_config() const;
};

struct TensorRTConfig {
  bool   enable_fp16        = false;
  bool   enable_int8        = false;
  size_t workspace_size_mb  = 1024;
  int    dla_core           = -1;    // -1 = 不使用 DLA
  int    max_batch_size     = 1;
  bool   enable_sparsity    = false;
  std::string calibration_cache = "";
};

struct QNNConfig {
  std::string backend          = "HTP";  // HTP | GPU | CPU
  std::string performance_mode = "high_performance";
  std::string precision        = "fp16";
  bool        enable_htp_fp16 = true;
};
```

---

#### `callback.hpp` — 事件回調

```cpp
enum class CallbackEvent {
  PreProcess, PostProcess,
  InferStart, InferEnd,
  BatchStart, BatchEnd,
  StreamStart, StreamFrame, StreamEnd
};

struct CallbackContext {
  CallbackEvent event;
  std::string   model_name;
  std::string   device;
  double        latency_ms;
  int           batch_size;
  int           frame_index;
  const Results* results;    // InferEnd 後有效
  std::map<std::string,std::string> metadata;
};

// 使用方式
model->on("infer_end", [](const CallbackContext& ctx) {
  std::cout << "Latency: " << ctx.latency_ms << "ms\n";
}, /*priority=*/10);
```

---

### 2.2 Runtime 模組

#### `runtime.hpp` — IRuntime 介面

```cpp
class IRuntime {
public:
  // 標識
  virtual BackendType type()         const = 0;
  virtual std::string name()         const = 0;
  virtual bool        is_available() const = 0;

  // 發現
  virtual std::vector<std::string> supported_formats() const = 0;
  virtual std::vector<DeviceInfo>  get_devices()       const = 0;

  // 生命週期
  virtual void* load_model  (const std::string& path, const LoadConfig& cfg) = 0;
  virtual void  unload_model(void* handle) = 0;

  // 張量規格
  virtual std::vector<TensorInfo> get_input_info (void* handle) const = 0;
  virtual std::vector<TensorInfo> get_output_info(void* handle) const = 0;

  // 推論
  virtual std::map<std::string, Tensor> infer(
      void* handle,
      const std::map<std::string, Tensor>& inputs) = 0;

  // 格式轉換
  virtual void convert_model(const std::string& src,
                              const std::string& dst,
                              const LoadConfig& cfg) = 0;
};
```

**RuntimeFactory（工廠 + 單例）：**
```cpp
class RuntimeFactory {
  shared_ptr<IRuntime> get_runtime(BackendType type);
  shared_ptr<IRuntime> get_runtime_for_device(const std::string& device);
  shared_ptr<IRuntime> get_best_runtime(const std::string& model_path,
                                         const std::string& device);
  void register_runtime(shared_ptr<IRuntime> rt);
  std::vector<shared_ptr<IRuntime>> list_available_runtimes();
};
```

**後端選擇優先級（`get_best_runtime`）：**
```
副檔名 .engine/.trt  → TensorRT
副檔名 .xml/.bin     → OpenVINO
副檔名 .bin/.serialized → QNN
副檔名 .onnx:
  ├── device=cpu/npu     → OpenVINO
  ├── device=cuda:*      → TensorRT
  └── 自動偵測           → TensorRT > OpenVINO (優先選已可用的)
```

---

#### `openvino_runtime.hpp` — Intel OpenVINO

```cpp
struct OpenVINOModel {
  ov::CompiledModel   compiled_model;
  ov::InferRequest    infer_request;
  std::vector<TensorInfo> input_infos;
  std::vector<TensorInfo> output_infos;
  std::string         device;
};

class OpenVINORuntime : public IRuntime {
  // 設備名稱映射
  // "cpu"   → "CPU"
  // "gpu:0" → "GPU.0"
  // "npu"   → "NPU"
  std::string map_device_name(const std::string& device) const;

  // 功能
  void set_cache_dir(const std::string& dir);
  void set_async(bool enable);
};
```

| 屬性 | 值 |
|------|----|
| 支持格式 | `.onnx`, `.xml`, `.bin`, `.pdmodel` |
| 支持設備 | CPU, iGPU (Intel Iris/Arc), NPU (AI Boost), VPU (Movidius) |
| 精度 | FP32, FP16 (GPU), INT8 (CPU/GPU) |

---

#### `tensorrt_runtime.hpp` — NVIDIA TensorRT

```cpp
struct TensorRTEngine {
  shared_ptr<nvinfer1::ICudaEngine>    engine;
  unique_ptr<nvinfer1::IExecutionContext> context;
  std::vector<void*>  device_buffers;   // CUDA malloc
  std::vector<float>  host_buffers;     // host 端
  cudaStream_t        stream;
  // 靜態析構防護（避免 atexit segfault）
};

class TensorRTRuntime : public IRuntime {
  void build_engine_from_onnx(const std::string& path,
                               const TensorRTConfig& cfg);
  void load_engine (const std::string& path);
  void save_engine (const std::string& path);
  void set_device  (int cuda_device_id);
};
```

| 屬性 | 值 |
|------|----|
| 支持格式 | `.onnx`, `.engine`, `.trt` |
| 支持設備 | NVIDIA GPU (CUDA), Jetson |
| 精度 | FP32, FP16, INT8, INT4 |
| 引擎緩存 | 自動儲存 `.engine` 避免重複編譯 |

---

#### `qnn_runtime.hpp` — Qualcomm QNN

```cpp
struct QNNModel {
  Qnn_ContextHandle_t               context  = nullptr;
  Qnn_GraphHandle_t                 graph    = nullptr;
  std::string                       graph_name;
  std::string                       backend_name;  // "HTP"|"GPU"|"CPU"
  std::vector<Qnn_Tensor_t>         input_tensors;
  std::vector<Qnn_Tensor_t>         output_tensors;
  std::vector<std::vector<uint8_t>> output_buffers;
  std::vector<uint8_t>              binary_data;   // 保活供 context 生命週期
};

class QNNRuntime : public IRuntime {
  // 動態加載 libQnnHtp.so / libQnnGpu.so / libQnnCpu.so
  bool load_backend_library(const std::string& backend, const std::string& skel_dir);
  void initialize();

private:
  void*                 backend_lib_handle_ = nullptr;
  const QnnInterface_t* qnn_interface_      = nullptr;
  Qnn_BackendHandle_t   shared_backend_     = nullptr;
  const QnnSystemInterface_t* sys_interface_ = nullptr;
  std::string backend_name_ = "HTP";
  std::mutex  mutex_;
};
```

| 屬性 | 值 |
|------|----|
| 支持格式 | `.bin`, `.serialized` (QNN context binary) |
| 支持設備 | Hexagon HTP (NPU), Adreno GPU, CPU fallback |
| 精度 | FP32, FP16, INT8 |
| 加載方式 | dlopen 動態探測 (無硬連結) |

---

### 2.3 Vision 高階 API

#### `classifier.hpp`
```cpp
class Classifier {
  explicit Classifier(const std::string& model,
                      const std::string& device = "cpu",
                      const LoadConfig& cfg = {});

  ClassificationResult predict(const cv::Mat& image, int top_k = 5);
  std::vector<ClassificationResult> predict_batch(const std::vector<cv::Mat>&, int top_k = 5);

  std::vector<std::string> classes()     const;
  int                      num_classes() const;
  std::pair<int,int>       input_size()  const;
};
```

內部流程：BGR→RGB → resize → ImageNet normalize → softmax → top-K partial_sort

---

#### `detector.hpp`
```cpp
class Detector {
  // 自動偵測模型類型: YOLO v5/v7/v8/X, SSD, Faster R-CNN, FCOS, RetinaNet
  explicit Detector(const std::string& model,
                    const std::string& device = "cpu");

  Results predict(const cv::Mat& image,
                  float conf = 0.5f, float iou = 0.45f);
  Results predict(const std::string& image_path, float conf, float iou);
  std::vector<Results> predict_batch(...);

  // 視頻流（回調風格）
  void predict_video(const std::string& source,
                     std::function<void(cv::Mat, Results)> callback,
                     float conf = 0.5f);
};
```

後處理：
- Letterbox resize（保持寬高比，灰色填充 114,114,114）
- YOLOX grid decode（strides 8/16/32）
- 自訂 greedy NMS（per class，按 confidence 排序）
- 座標恢復：`(cx ± w/2 - pad) / scale`，clamp 到圖像邊界

---

#### `segmentor.hpp`
```cpp
class Segmentor {
  // 自動偵測標籤集: VOC(21) / Cityscapes(19) / ADE20K(150)
  explicit Segmentor(const std::string& model,
                     const std::string& device = "cpu");

  Results predict(const cv::Mat& image);
  std::vector<Results> predict_batch(const std::vector<cv::Mat>&);

  std::vector<std::string>   classes()  const;
  std::vector<cv::Vec3b>     colormap() const;
  void set_colormap(const std::vector<cv::Vec3b>&);
};
```

後處理：argmax → segmentation_mask → INTER_NEAREST resize 回原尺寸

---

### 2.4 Utils 模組

#### `profiler.hpp`
```cpp
struct ProfileReport {
  // 延遲統計 (ms)
  double mean, median, std_dev, min, max, p95, p99;
  double throughput_fps;
  double memory_mb;

  std::string to_string() const;
  std::string to_json()   const;
  void        save(const std::string& path) const;
};

class Profiler {
  ProfileReport benchmark(std::shared_ptr<Model> model,
                           const std::vector<int>& input_shape,
                           int iterations = 100,
                           int warmup     = 10);
  void   start();
  void   stop();
  double elapsed_ms() const;
};

class ScopedTimer {            // RAII
  ScopedTimer(Profiler& p) { p.start(); }
  ~ScopedTimer()            { profiler_.stop(); }
};
```

#### `visualizer.hpp`
```cpp
class Visualizer {
  static cv::Mat draw_detections (const cv::Mat& image,
                                   const std::vector<Detection>& dets,
                                   const std::vector<std::string>& labels,
                                   const VisConfig& cfg = {});
  static cv::Mat draw_segmentation(const cv::Mat& image,
                                   const cv::Mat& mask,
                                   const std::vector<cv::Vec3b>& colormap,
                                   float alpha = 0.5f);
  static cv::Mat draw_poses       (const cv::Mat& image,
                                   const std::vector<Pose>& poses,
                                   const std::vector<std::pair<int,int>>& skeleton,
                                   const VisConfig& cfg = {});
  static cv::Mat create_comparison(const std::vector<cv::Mat>& images,
                                   const std::vector<std::string>& titles,
                                   int cols = 2);
  static cv::Vec3b   get_color    (int class_id);
  static std::vector<cv::Vec3b> generate_colormap(int num_classes);
};
```

---

## 3. 完整呼叫流程

以 YOLOX 目標偵測推論為例：

```
用戶代碼
│
├── ivit::load_model("yolox-s.onnx", {device="cuda:0"})
│   │
│   ├── ModelManager::instance().load()                     [model.cpp]
│   │   │
│   │   ├── parse_device_string("cuda:0")                  [device_manager.cpp]
│   │   │   └── → (BackendType::TensorRT, device_id="0")
│   │   │
│   │   ├── RuntimeFactory::get_best_runtime("*.onnx", "cuda:0")  [runtime_factory.cpp]
│   │   │   └── → TensorRTRuntime
│   │   │
│   │   └── TensorRTRuntime::load_model("yolox-s.onnx", cfg)      [tensorrt_runtime.cpp]
│   │       ├── get_cache_path() → "yolox-s_trt_fp16_sm89.engine"
│   │       ├── 快取存在且較新 → load_engine()（直接反序列化）
│   │       ├── 否則 build_engine_from_onnx()
│   │       │   ├── NvOnnxParser: ONNX → network definition
│   │       │   ├── dynamic profile: min=1/opt=1/max=32
│   │       │   ├── builder->buildSerializedNetwork()
│   │       │   └── 儲存 engine 至 .engine 緩存
│   │       ├── create_engine_wrapper()
│   │       │   ├── allocate device buffers (cudaMalloc, alloc_dims)
│   │       │   └── context->setTensorAddress() [TRT 10 API]
│   │       ├── create CUDA stream
│   │       └── return void* handle → TensorRTEngine*
│   │
│   └── InferenceModel(runtime, handle, "yolox-s", Detection, "cuda:0")
│       ├── query input_info: [{name:"images", shape:[1,3,640,640]}]
│       ├── query output_info
│       └── load labels from yolox-s.txt
│
│   return shared_ptr<Model>
│
├── model->predict(cv::Mat image)
│   │
│   ├── InferenceModel::predict()                           [model.cpp]
│   │   │
│   │   ├── trigger CallbackEvent::PreProcess
│   │   │
│   │   ├── Detector::preprocess(image)                    [detector.cpp]
│   │   │   ├── letterbox resize → 640×640
│   │   │   │   scale = min(640/orig_w, 640/orig_h)
│   │   │   │   pad 灰色 (114,114,114)
│   │   │   ├── BGR → RGB（非 YOLOX）
│   │   │   ├── normalize ÷255
│   │   │   └── return {"images": Tensor[1,3,640,640]}
│   │   │
│   │   ├── trigger CallbackEvent::InferStart
│   │   │
│   │   ├── TensorRTRuntime::infer(handle, inputs)          [tensorrt_runtime.cpp]
│   │   │   ├── context->setInputShape("images", [1,3,640,640])
│   │   │   ├── cudaMemcpyAsync(host→device, stream)
│   │   │   ├── context->enqueueV3(stream)   [TRT 10 API]
│   │   │   ├── context->getTensorShape("output0") → 實際輸出形狀
│   │   │   ├── cudaMemcpyAsync(device→host, stream)
│   │   │   ├── cudaStreamSynchronize(stream)
│   │   │   └── return {"output0": Tensor[1, 8400, 84]}
│   │   │
│   │   ├── trigger CallbackEvent::InferEnd
│   │   │
│   │   ├── Detector::postprocess(outputs, original_size, cfg) [detector.cpp]
│   │   │   ├── 解析 [cx, cy, w, h, obj_conf, cls0..cls79]
│   │   │   ├── YOLOX grid decode（per-stride [8,16,32]，cx/cy 加 grid 偏移 × stride）
│   │   │   ├── 過濾 confidence < threshold
│   │   │   ├── 自訂 greedy NMS (per class, sorted by conf)
│   │   │   ├── 座標恢復: (cx±w/2 - pad_w) / scale，clamp
│   │   │   └── 填充 Detection{bbox, confidence, class_id, label}
│   │   │
│   │   ├── trigger CallbackEvent::PostProcess
│   │   │
│   │   └── return Results{detections=[...], inference_time_ms=X}
│   │
│   └── 回到用戶代碼
│
└── 使用結果
    ├── for (const auto& det : results.detections) { ... }
    └── cv::Mat vis = results.visualize(image)     [visualizer.cpp]
        └── Visualizer::draw_detections()
```

**各步驟對應原始檔案：**

| 步驟 | 主要檔案 |
|------|----------|
| API 入口 | `include/ivit/ivit.hpp` |
| 模型管理 | `src/core/model.cpp` |
| 設備解析 | `src/core/device_manager.cpp` |
| 後端選擇 | `src/runtime/runtime_factory.cpp` |
| 自動註冊 | `src/runtime/runtime_init.cpp` |
| OpenVINO | `src/runtime/openvino_runtime.cpp` |
| TensorRT | `src/runtime/tensorrt_runtime.cpp` |
| QNN | `src/runtime/qnn_runtime.cpp` |
| 結果處理 | `src/core/result.cpp` |
| 高階 API | `src/vision/detector.cpp` |

---

## 4. 多後端擴展機制

### 4.1 自動註冊（Static Initializer）

```cpp
// src/runtime/runtime_init.cpp
class RuntimeAutoRegister {
public:
  RuntimeAutoRegister() {
    auto& factory = RuntimeFactory::instance();

#ifdef IVIT_HAS_OPENVINO
    try {
      auto ov = std::make_shared<OpenVINORuntime>();
      if (ov->is_available()) factory.register_runtime(ov);
    } catch (...) {}
#endif

#ifdef IVIT_HAS_TENSORRT
    try {
      auto trt = std::make_shared<TensorRTRuntime>();
      if (trt->is_available()) factory.register_runtime(trt);
    } catch (...) {}
#endif

#ifdef IVIT_HAS_QNN
    try {
      auto qnn = std::make_shared<QNNRuntime>();
      if (qnn->is_available()) factory.register_runtime(qnn);
    } catch (...) {}
#endif
  }
};

// 靜態實例 → 庫加載時自動執行
static RuntimeAutoRegister g_runtime_auto_register;
```

### 4.2 新增後端的完整步驟

```
1. 建立介面實現
   include/ivit/runtime/mybackend_runtime.hpp
   src/runtime/mybackend_runtime.cpp

2. 繼承 IRuntime
   class MyBackendRuntime : public IRuntime {
     BackendType type() const override { return BackendType::MyBackend; }
     bool is_available() const override { /* probe library */ }
     void* load_model(...) override { ... }
     std::map<string, Tensor> infer(...) override { ... }
     // ...
   };

3. 新增枚舉值 (common.hpp)
   enum class BackendType { OpenVINO, TensorRT, QNN, MyBackend, Auto, Unknown };

4. 在 runtime_init.cpp 註冊
   #ifdef IVIT_HAS_MYBACKEND
   auto mb = make_shared<MyBackendRuntime>();
   if (mb->is_available()) factory.register_runtime(mb);
   #endif

5. 在 device_manager.cpp 新增設備發現
   void DeviceManager::discover_mybackend_devices() { ... }

6. 更新 CMakeLists.txt
   option(IVIT_USE_MYBACKEND "Enable MyBackend" OFF)
   if(IVIT_USE_MYBACKEND)
     find_path(MB_INCLUDE_DIR MyBackend.h ...)
     if(MB_INCLUDE_DIR)
       add_compile_definitions(IVIT_HAS_MYBACKEND)
       list(APPEND IVIT_SOURCES src/runtime/mybackend_runtime.cpp)
       target_include_directories(ivit PRIVATE ${MB_INCLUDE_DIR})
       target_link_libraries(ivit PRIVATE ${CMAKE_DL_LIBS})
     endif()
   endif()
```

---

## 5. 資料流

### 5.1 完整資料流（從輸入到結果）

```
cv::Mat (BGR, HWC, uint8)
      ↓
┌─────────────────────────────────┐
│  Preprocessing                  │
│  BGR→RGB / letterbox / normalize│
│  → HWC uint8 → NCHW float32    │
└──────────────┬──────────────────┘
               ↓
Tensor{shape=[1,3,H,W], dtype=Float32}
     data_: vector<uint8_t> (CPU 主機記憶體)
               ↓
   ┌───────────┴────────────┐
   ▼           ▼            ▼
[TRT]        [OV]         [QNN]
cudaMemcpy  set_tensor   set ptr
host→device  直接使用     零複製輸入
   ↓           ↓            ↓
Device-side inference kernel
   ↓           ↓            ↓
cudaMemcpy  get_tensor   memcpy from
device→host  複製輸出     output_buffer
   └───────────┴────────────┘
               ↓
map<string, Tensor>{outputs}
               ↓
┌──────────────────────────────────┐
│  Postprocessing                  │
│  NMS / argmax / scale / decode   │
└──────────────┬───────────────────┘
               ↓
Results{detections/classifications/...}
```

### 5.2 DataType 對應

| DataType | C++ 類型 | 字節 |
|----------|----------|------|
| Float32 | `float` | 4 |
| Float16 | `uint16_t` (half) | 2 |
| Int32 | `int32_t` | 4 |
| Int64 | `int64_t` | 8 |
| Int8 | `int8_t` | 1 |
| UInt8 | `uint8_t` | 1 |
| Bool | `bool` | 1 |

### 5.3 Layout 記憶體排列

```
NCHW (PyTorch/ONNX 預設):
  offset = b×C×H×W + c×H×W + h×W + w

NHWC (TF/QNN 偏好):
  offset = b×H×W×C + h×W×C + w×C + c

TensorInfo::byte_size():
  = numel() × sizeof(dtype)
  = (N×C×H×W) × 4  (Float32)
```

### 5.4 跨後端 Buffer 策略

| 後端 | Host Buffer | Device Buffer | 策略 |
|------|-------------|---------------|------|
| TensorRT | `vector<float>` | `cudaMalloc` | 明確雙向 cudaMemcpy |
| OpenVINO | 直接使用 caller ptr | 透明 (框架管理) | CPU: 無複製；GPU: 自動 |
| QNN | `vector<uint8_t>` | DSP/GPU | 設置指針（輸入），memcpy（輸出） |

---

## 6. 關鍵設計模式

### 6.1 工廠模式（Factory）
**RuntimeFactory** — 集中管理後端實例，按需返回正確後端。
```cpp
auto runtime = RuntimeFactory::instance().get_best_runtime("model.onnx", "cuda:0");
```

### 6.2 策略模式（Strategy）
**IRuntime** — 後端實現可在運行時互換，上層代碼無需更改。
```cpp
void predict(shared_ptr<IRuntime> runtime, ...) {
  runtime->infer(...);  // TRT 或 OV 或 QNN，呼叫方不感知
}
```

### 6.3 模板方法模式（Template Method）
**Model::predict** — 固定推論骨架，細節由子類實現。
```cpp
Results Model::predict(const cv::Mat& image) {
  auto inputs  = preprocess(image);    // 子類覆寫（letterbox/normalize 各異）
  auto outputs = runtime_->infer(...);
  Results res;
  postprocess(outputs, image.size(), res);  // 子類覆寫（NMS/argmax 各異）
  return res;
}
```

### 6.4 單例模式（Singleton）
**RuntimeFactory、ModelManager、DeviceManager** — 全局唯一實例。
```cpp
static RuntimeFactory& instance() {
  static RuntimeFactory inst;  // 懶初始化，線程安全（C++11）
  return inst;
}
```

### 6.5 觀察者模式（Observer）
**CallbackManager** — 解耦推論監控與業務邏輯。
```cpp
model->on("infer_end", handler, priority=10);
// 推論完成後自動觸發
```

### 6.6 RAII 模式
**ScopedTimer** — 自動資源管理。
```cpp
{ ScopedTimer t(profiler); model->predict(image); }  // 自動停止計時
```

**TensorRTEngine 析構** — CUDA buffer 正常釋放；TRT 物件在 `g_in_shutdown=true` 時刻意洩漏（防 sm_120+ segfault）。

### 6.7 靜態初始化（Startup Registration）
`g_runtime_auto_register` — 庫加載時自動探測並註冊可用後端，不需要用戶調用任何初始化函數。

---

## 7. 各後端差異比較

| 特性 | OpenVINO | TensorRT | QNN |
|------|----------|----------|-----|
| **目標廠商** | Intel | NVIDIA | Qualcomm |
| **設備** | CPU/iGPU/NPU/VPU | GPU (CUDA)/Jetson | HTP(NPU)/Adreno/CPU |
| **輸入格式** | ONNX, IR (.xml) | ONNX, .engine | .bin (預編譯) |
| **初始化方式** | `ov::Core().compile_model()` | ONNX Parser + build | `contextCreateFromBinary` |
| **Linking** | 靜態連結 libopenvino.so | 靜態連結 libnvinfer.so | **dlopen 動態加載** |
| **編譯緩存** | 自動（compile cache） | 手動（engine 序列化） | 無需（binary 即成品） |
| **動態 Shape** | 部分支持 | 有限支持（need profile） | 不支持 |
| **精度** | FP32/FP16/INT8 | FP32/FP16/INT8/INT4 | FP32/FP16/INT8 |
| **初始化延遲** | 中等 | 高（第一次 build） | 低 |
| **推論延遲** | 中（針對 Intel 優化） | 最低（針對 NVIDIA） | 低（針對 Qualcomm） |
| **功耗效率** | 中 | 高耗電 | 最省電（DSP） |
| **`convert_model()`** | 支持 ONNX→IR, FP16 壓縮 | 支持 ONNX→engine | 拋出異常，引導使用離線工具 |

### 模型加載代碼對比

```cpp
// ── OpenVINO ──────────────────────────────────────────────────
ov::Core core;
// 注意：不設置 ov::log::level 全域（GPU plugin 相容問題）
core.set_property(device, ov::hint::performance_mode(LATENCY));
auto model    = core.read_model("model.onnx");
auto compiled = core.compile_model(model, "CPU");
auto request  = compiled.create_infer_request();

// ── TensorRT ──────────────────────────────────────────────────
IBuilder* builder  = createInferBuilder(logger);
auto network = builder->createNetworkV2(...);
auto parser  = nvonnxparser::createParser(*network, logger);
parser->parseFromFile("model.onnx", 0);
auto config  = builder->createBuilderConfig();
config->setMemoryPoolLimit(kWORKSPACE, 1ULL<<30);  // 1GB
// 動態 profile: min/opt/max batch
auto serialized = builder->buildSerializedNetwork(*network, *config);
// 儲存 .engine 緩存（鍵：{stem}_trt_{prec}_sm{CC}.engine）

// ── QNN ───────────────────────────────────────────────────────
// 1. dlopen libQnnHtp.so（設置 ADSP_LIBRARY_PATH for HTP skel）
void* lib = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_GLOBAL);
auto get_providers = dlsym(lib, "QnnInterface_getProviders");
const QnnInterface_t* iface = providers[0];

// 2. backendCreate（null log/config = 預設值）
iface->backendCreate(nullptr, nullptr, &backend_handle);

// 3. 加載 binary（保活 binary_data，ctx 需持續指向）
std::vector<uint8_t> binary = read_file("model.bin");
iface->contextCreateFromBinary(..., binary.data(), binary.size(), ..., &ctx);

// 4. 透過 QnnSystemContext 讀取元資料（圖名、tensor 形狀）
// 5. graphRetrieve(ctx, graph_name, &graph)
```

### 推論代碼對比

```cpp
// ── OpenVINO ──────────────────────────────────────────────────
// 零複製：直接以 caller 指針建立 ov::Tensor
ov::Tensor ov_input(ov::element::f32, shape, input_ptr);
request.set_input_tensor(ov_input);
request.infer();  // 或 start_async() + wait()
auto output = request.get_output_tensor();

// ── TensorRT ──────────────────────────────────────────────────
context->setInputShape("images", actual_dims);   // 動態 batch
cudaMemcpyAsync(device_buf[0], host_input, size,
                cudaMemcpyHostToDevice, stream);
context->enqueueV3(stream);                      // TRT 10 API
cudaStreamSynchronize(stream);
auto out_shape = context->getTensorShape("output0");  // 動態輸出大小
cudaMemcpyAsync(host_output, device_buf[1], out_size,
                cudaMemcpyDeviceToHost, stream);

// ── QNN ───────────────────────────────────────────────────────
// 零複製輸入：設置 clientBuf 指針
input_tensors[0].v1.clientBuf.data = input_ptr;
input_tensors[0].v1.memType = QNN_TENSORMEMTYPE_RAW;
iface->graphExecute(graph,
                    input_tensors.data(),  num_inputs,
                    output_tensors.data(), num_outputs,
                    nullptr, nullptr);
// 從 pre-alloc output_buffers memcpy 出結果
memcpy(host_output, output_buffers[0].data(), size);
```

---

## 8. 編譯與建置系統

### 8.1 選項架構

```cmake
# 後端開關
option(IVIT_USE_OPENVINO "Enable OpenVINO backend" ON)
option(IVIT_USE_TENSORRT "Enable TensorRT backend" ON)
option(IVIT_USE_QNN      "Enable Qualcomm QNN"    OFF)

# Python 綁定
option(IVIT_BUILD_PYTHON "Build pybind11 bindings" ON)

# 依賴打包（自包含 SDK）
option(IVIT_BUNDLE_DEPS  "Bundle runtime libraries" OFF)
```

### 8.2 源文件組織

```cmake
set(IVIT_SOURCES
  # Core
  src/core/model.cpp
  src/core/device_manager.cpp
  src/core/result.cpp
  src/core/runtime_config.cpp
  src/core/callback.cpp
  src/core/video_source.cpp
  # Runtime 固定部分
  src/runtime/runtime_factory.cpp
  src/runtime/runtime_init.cpp
  # Vision
  src/vision/classifier.cpp
  src/vision/detector.cpp
  src/vision/segmentor.cpp
  # Utils
  src/utils/profiler.cpp
  src/utils/visualizer.cpp
)

# 條件編譯各後端
if(IVIT_USE_OPENVINO) list(APPEND IVIT_SOURCES src/runtime/openvino_runtime.cpp) endif()
if(IVIT_USE_TENSORRT) list(APPEND IVIT_SOURCES src/runtime/tensorrt_runtime.cpp) endif()
if(IVIT_USE_QNN)      list(APPEND IVIT_SOURCES src/runtime/qnn_runtime.cpp)      endif()

add_library(ivit SHARED ${IVIT_SOURCES})
```

### 8.3 依賴打包（`IVIT_BUNDLE_DEPS`）

```cmake
if(IVIT_BUNDLE_DEPS)
  # RPATH 設定：讓 libivit.so 優先搜尋旁邊的 deps/lib
  set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib;$ORIGIN/../deps/lib")

  if(IVIT_USE_OPENVINO)
    file(GLOB OV_LIBS "${OV_LIB_DIR}/*.so*")
    list(APPEND IVIT_BUNDLE_LIBRARIES ${OV_LIBS})
  endif()

  install(FILES ${IVIT_BUNDLE_LIBRARIES} DESTINATION deps/lib)
endif()
```

### 8.4 QNN SDK 發現

```cmake
if(IVIT_USE_QNN)
  if(NOT DEFINED QNN_SDK_ROOT AND DEFINED ENV{QNN_SDK_ROOT})
    set(QNN_SDK_ROOT "$ENV{QNN_SDK_ROOT}")
  endif()

  find_path(QNN_INCLUDE_DIR QnnInterface.h
    HINTS ${QNN_SDK_ROOT} /opt/qnn /usr/local/qnn
    PATH_SUFFIXES include/QNN include QNN)

  if(QNN_INCLUDE_DIR)
    add_compile_definitions(IVIT_HAS_QNN)
    list(APPEND IVIT_SOURCES src/runtime/qnn_runtime.cpp)
    target_include_directories(ivit PRIVATE ${QNN_INCLUDE_DIR})
    target_link_libraries(ivit PRIVATE ${CMAKE_DL_LIBS})  # dlopen
  else()
    message(WARNING "QNN SDK not found — disabling QNN backend.")
    set(IVIT_USE_QNN OFF)
  endif()
endif()
```

---

## 9. 執行流程總結

```
程式啟動
    │
    ▼
┌──────────────────────────────────────┐
│ 1. 靜態初始化 (庫加載時自動執行)      │
│    RuntimeAutoRegister::ctor()        │
│    → probe + register:                │
│      TensorRTRuntime (if CUDA 可用)   │
│      OpenVINORuntime (if OV 可用)     │
│      QNNRuntime      (if QNN 庫存在)  │
└─────────────────┬────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│ 2. 模型加載                           │
│    ivit::load_model(path, cfg)        │
│    → ModelManager (緩存+生命週期)     │
│      cache_key = path+"_"+dev+"_"+be  │
│      double-check locking             │
│    → RuntimeFactory (後端選擇)        │
│    → IRuntime::load_model()           │
│    → InferenceModel 封裝              │
└─────────────────┬────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│ 3. 推論迴圈                           │
│    model->predict(image)             │
│    ├── 觸發 PreProcess 回調           │
│    ├── preprocess (letterbox/norm)    │
│    ├── 觸發 InferStart 回調           │
│    ├── IRuntime::infer() (後端執行)   │
│    ├── 觸發 InferEnd 回調             │
│    └── postprocess (NMS/argmax/scale) │
│                                       │
│    → return Results                   │
└─────────────────┬────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│ 4. 結果使用                           │
│    results.detections                 │
│    results.visualize(image)           │
│    results.to_json()                  │
└─────────────────┬────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│ 5. 資源釋放 (RAII)                    │
│    ~InferenceModel()                  │
│    → IRuntime::unload_model()         │
│    → 釋放 GPU/NPU 記憶體              │
│    (TRT: g_in_shutdown=true 時洩漏    │
│     TRT 物件，防 sm_120+ segfault)    │
└──────────────────────────────────────┘
```

---

## 10. 實作細節深潛

本節逐一追蹤各 `.cpp` 的關鍵實作路徑，補充架構圖未能呈現的細節。

---

### 10.1 `src/core/model.cpp` — InferenceModel 核心

#### `preprocess()`（InferenceModel 基礎版）

InferenceModel 的基礎前處理**直接 resize，不做 letterbox**。各 Vision 子類（Classifier、Detector、Segmentor）會覆寫此函式提供特定邏輯。

```cpp
// InferenceModel::preprocess() — model.cpp
// 1. 取目標尺寸；若輸入 shape[2/3] == -1（動態 dim），fallback 為 224
int th = (input_info.shape[2] != -1) ? input_info.shape[2] : 224;
int tw = (input_info.shape[3] != -1) ? input_info.shape[3] : 224;

// 2. 直接縮放（不保持長寬比）
cv::Mat resized;
cv::resize(image, resized, cv::Size(tw, th));

// 3. 轉為 float32 並歸一化至 [0, 1]
cv::Mat float_img;
resized.convertTo(float_img, CV_32F, 1.0 / 255.0);

// 4. HWC → NCHW（利用 cv::split + memcpy）
std::vector<cv::Mat> channels;
cv::split(float_img, channels);
float* dst = tensor.data_ptr<float>();
for (int c = 0; c < 3; ++c)
    std::memcpy(dst + c * th * tw, channels[c].ptr<float>(), th * tw * sizeof(float));
```

> **與 Detector 的差異**：Detector::preprocess() 使用 letterbox（保持寬高比 + 灰色填充），InferenceModel 直接 resize。兩者輸出的座標回推邏輯因此完全不同。

---

#### `postprocess_classification()`

```cpp
// softmax — 對整個輸出向量做歸一化
float max_val = *std::max_element(scores, scores + n);
float sum = 0.f;
for (int i = 0; i < n; ++i) { scores[i] = std::exp(scores[i] - max_val); sum += scores[i]; }
for (int i = 0; i < n; ++i) scores[i] /= sum;

// top-k：最多取 min(top_k, 10)；使用 std::partial_sort（不全排）
int k = std::min(top_k, std::min(n, 10));
std::partial_sort(indexed.begin(), indexed.begin() + k, indexed.end(),
                  [&](auto& a, auto& b){ return scores[a] > scores[b]; });
```

- 輸出最多返回 **10** 筆結果（top-10 cap）
- `std::partial_sort` O(n log k)，比全排 O(n log n) 快

---

#### `postprocess_detection()`（InferenceModel 基礎版）

```
輸出 Tensor 格式：[batch, num_anchors, pred_size]
  pred_size = 4 (bbox) + 1 (obj_conf) + num_classes

每個 anchor:
  [cx, cy, w, h, obj_conf, cls0, cls1, ..., clsN]

過濾：conf = obj_conf * max(cls_scores) < threshold → 丟棄
NMS：cv::dnn::NMSBoxes()（per class）
```

> **注意**：Detector 子類完全覆寫此函式，使用自訂 greedy NMS 並加入 YOLOX grid decode。

---

#### `postprocess_segmentation()`

```cpp
// 輸出 shape [1, C, H, W] → pixel-by-pixel argmax
for (int h = 0; h < H; ++h)
    for (int w = 0; w < W; ++w) {
        int best = 0; float best_val = -FLT_MAX;
        for (int c = 0; c < C; ++c) {
            float v = output_ptr[c * H * W + h * W + w];
            if (v > best_val) { best_val = v; best = c; }
        }
        mask.at<int>(h, w) = best;
    }

// 最近鄰插值縮放回原圖尺寸（保持標籤整數不模糊）
cv::resize(mask, mask, original_size, 0, 0, cv::INTER_NEAREST);
```

---

#### `predict_batch()`

```cpp
// 動態 batch 偵測：shape[0] == -1 表示支援真正批次
bool has_dynamic_batch = (input_info[0].shape[0] == -1);

if (has_dynamic_batch) {
    // 合併 N 張圖的 Tensor → 一次 infer → 切割輸出
    auto merged = merge_tensors(preprocessed_inputs, N);
    auto outputs = runtime_->infer(handle_, {{"images", merged}});
    return split_and_postprocess(outputs, N);
} else {
    // sequential fallback
    std::vector<Results> all;
    for (auto& inp : preprocessed_inputs)
        all.push_back(postprocess(runtime_->infer(handle_, inp)));
    return all;
}
```

---

#### `predict_concurrent()`

```cpp
// Semaphore via mutex + condition_variable
std::mutex mtx_;
std::condition_variable cv_;
int in_flight_ = 0;

// max_concurrent 限制同時在飛的推論數
{
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait(lock, [&]{ return in_flight_ < max_concurrent; });
    ++in_flight_;
}
auto future = std::async(std::launch::async, [&, img]{
    auto r = predict(img);
    { std::lock_guard<std::mutex> g(mtx_); --in_flight_; }
    cv_.notify_one();
    return r;
});
```

---

#### `predict_tta()`（Test-Time Augmentation）

支援 **8 種增強**：

| 索引 | 變換 |
|------|------|
| 0 | original（原圖） |
| 1 | hflip（水平翻轉） |
| 2 | vflip（垂直翻轉） |
| 3 | rotate90 |
| 4 | rotate180 |
| 5 | rotate270 |
| 6 | scale_up（放大 1.2×） |
| 7 | scale_down（縮小 0.8×） |

- **分類任務**：對 8 個 softmax 向量取平均 → argmax
- **偵測任務**：各增強預測框回投至原圖座標 → 合併 → 重跑 NMS

---

#### `ensure_executor()` 與異步執行

```cpp
// 持久 worker thread + task queue
void InferenceModel::ensure_executor() {
    if (executor_thread_.joinable()) return;
    executor_running_ = true;
    executor_thread_ = std::thread([this]{
        while (executor_running_) {
            std::function<void()> task;
            { std::unique_lock<std::mutex> lock(queue_mtx_);
              queue_cv_.wait(lock, [&]{ return !task_queue_.empty() || !executor_running_; });
              if (!executor_running_) break;
              task = std::move(task_queue_.front()); task_queue_.pop(); }
            task();
        }
    });
}
```

---

#### `ModelManager` — 模型快取

```cpp
// cache key = path + "_" + device + "_" + backend
std::string key = path + "_" + cfg.device + "_" + cfg.backend;

// Double-check locking
{
    std::shared_lock<std::shared_mutex> rlock(cache_mtx_);
    auto it = cache_.find(key);
    if (it != cache_.end()) return it->second;
}
{
    std::unique_lock<std::shared_mutex> wlock(cache_mtx_);
    auto it = cache_.find(key);
    if (it != cache_.end()) return it->second;  // 第二次檢查
    auto model = create_model(path, cfg);
    cache_[key] = model;
    return model;
}
```

---

### 10.2 `src/runtime/openvino_runtime.cpp` — OpenVINO 後端

#### 不設置全域 log level 的原因

```cpp
// ❌ 刻意省略：
// ov::log::Level::WARNING → 導致 GPU plugin dlopen 在部分 driver 版本下失敗

// ✅ 改為環境變數（不影響 plugin 載入）
setenv("OPENVINO_LOG_LEVEL", std::to_string(level).c_str(), 1);
```

---

#### `load_model()` 細節

```cpp
std::lock_guard<std::mutex> lock(mutex_);  // 串列化載入

ov::Core core;
if (cfg.precision == "fp16")
    core.set_property("GPU", ov::hint::inference_precision(ov::element::f16));
core.set_property(ov_device, ov::hint::performance_mode(LATENCY));
if (ov_device == "CPU" && cfg.num_threads > 0)
    core.set_property("CPU", ov::inference_num_threads(cfg.num_threads));
if (!cfg.cache_dir.empty())
    core.set_property(ov::cache_dir(cfg.cache_dir));

auto model    = core.read_model(path);
auto compiled = core.compile_model(model, ov_device);
auto request  = compiled.create_infer_request();
```

---

#### `infer()` — 零複製輸入

```cpp
// 直接使用 caller 的 data pointer，零複製
ov::Tensor ov_input(ov::element::f32, input_shape,
                    const_cast<float*>(tensor.data_ptr<float>()));
request.set_input_tensor(ov_input);

if (use_async_) { request.start_async(); request.wait(); }
else            { request.infer(); }
```

---

#### `convert_model()` 細節

```cpp
if (cfg.precision == "int8") {
    throw ModelError("INT8 量化需要 NNCF 校準，使用：optimum-cli export openvino --int8");
}
// FP16 壓縮
auto core = ov::Core();
auto model = core.read_model(src);
ov::save_model(model, dst, /*compress_fp16=*/true);
```

---

#### `map_device_name()` 完整映射

| 輸入 | OV 設備名 |
|------|-----------|
| `"auto"` | `"AUTO"` |
| `"cpu"` | `"CPU"` |
| `"gpu"` / `"gpu:0"` | `"GPU.0"` |
| `"npu"` | `"NPU"` |
| `"vpu"` | `"MYRIAD"` |

---

#### `extract_tensor_info()` 細節

```cpp
// 動態維度 → -1
for (auto& d : shape)
    info.shape.push_back(d.is_dynamic() ? -1 : (int)d.get_length());

// layout 從 rank 推斷
switch (rank) {
    case 4: info.layout = Layout::NCHW; break;
    case 2: info.layout = Layout::NC;   break;
    default: info.layout = Layout::Unknown;
}
```

---

### 10.3 `src/runtime/tensorrt_runtime.cpp` — TensorRT 後端

#### 刻意洩漏（Intentional Leak Pattern）

sm_120+（Hopper/Ada Lovelace）驅動下，`atexit` 順序問題導致 TRT 物件在 CUDA driver 卸載後才析構，觸發 segfault。

```cpp
static bool g_in_shutdown = false;
struct ShutdownRegistrar {
    ShutdownRegistrar() { std::atexit([]{ g_in_shutdown = true; }); }
};
static ShutdownRegistrar g_shutdown_registrar;

TensorRTEngine::~TensorRTEngine() {
    // CUDA 資源：正常釋放
    for (void* buf : device_buffers_) cudaFree(buf);
    cudaStreamDestroy(stream_);

    if (g_in_shutdown) {
        // ⚠️ 刻意洩漏 TRT 物件，避免 shutdown 時 segfault
        engine_.release(); context_.release(); trt_runtime_.release();
    }
}
```

---

#### 引擎緩存鍵格式

```
{model_stem}_trt_{precision}_sm{major}{minor}.engine
例：yolox-s_trt_fp16_sm89.engine（RTX 4090 = sm_89）
```

不同 GPU 算力（sm86/sm89/sm90）產生不同緩存，不可共用。

---

#### `build_engine_from_onnx()` 細節

```cpp
config->setMemoryPoolLimit(kWORKSPACE, 1ULL << 30);  // 1 GB

// 硬體能力檢查後才設置精度
if (cfg.enable_fp16 && builder->platformHasFastFp16()) config->setFlag(kFP16);
if (cfg.enable_int8 && builder->platformHasFastInt8()) config->setFlag(kINT8);

// 動態 shape profile
profile->setDimensions("images", kMIN, Dims4{1,3,H,W});
profile->setDimensions("images", kOPT, Dims4{1,3,H,W});
profile->setDimensions("images", kMAX, Dims4{32,3,H,W});
```

---

#### `infer()` — TRT 10 API

```cpp
context->setInputShape("images", actual_input_dims);  // 動態 batch

cudaMemcpyAsync(device_buf[in], host_input, size, cudaMemcpyHostToDevice, stream_);
context->enqueueV3(stream_);         // TRT 10（取代舊版 executeV2）
cudaStreamSynchronize(stream_);

// 動態輸出大小在執行後才確定
auto out_shape = context->getTensorShape("output0");
size_t actual_size = volume(out_shape) * sizeof(float);
cudaMemcpyAsync(host_output, device_buf[out], actual_size, cudaMemcpyDeviceToHost, stream_);
```

---

### 10.4 `src/runtime/qnn_runtime.cpp` — QNN 後端

#### `load_backend_library()` — ADSP 路徑設置

```cpp
// HTP backend 需要 Hexagon skel 庫
if (backend == "HTP" && !skel_dir.empty()) {
    std::string existing = getenv_or_empty("ADSP_LIBRARY_PATH");
    setenv("ADSP_LIBRARY_PATH", (skel_dir + ":" + existing).c_str(), 1);
}

void* lib = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_GLOBAL);
auto get_providers = (QnnInterfaceGetProvidersFn_t)dlsym(lib, "QnnInterface_getProviders");
qnn_interface_ = providers[0];

// 可選：libQnnSystem.so（用於 QnnSystemContext 元資料 API）
sys_lib_handle_ = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_LOCAL);
```

---

#### `initialize()` — backend 切換安全

```cpp
// 切換 backend 時先釋放舊的
if (shared_backend_ && backend_name_ != new_backend) {
    qnn_interface_->backendFree(shared_backend_);
    shared_backend_ = nullptr;
    dlclose(backend_lib_handle_);
}
// null log/config → 使用 QNN 預設
qnn_interface_->backendCreate(nullptr, nullptr, &shared_backend_);
```

---

#### `extract_tensor_infos()` — 元資料讀取

```cpp
QnnSystemContext_Handle_t sys_ctx;
sys_interface_->systemContextCreate(&sys_ctx);

const QnnSystemContext_BinaryInfo_t* bin_info;
sys_interface_->systemContextGetBinaryInfo(sys_ctx, binary.data(), size, &bin_info);

auto& gv1 = bin_info->contextBlobInfo.graphs[0].graphInfoV1;
model->graph_name = gv1.graphName;

for (uint32_t i = 0; i < gv1.numInputTensors; ++i) {
    auto& qt = gv1.inputTensors[i];
    TensorInfo info;
    info.name = qt.v1.name;
    // 4-D QNN tensor → 預設 NHWC（與 TRT/OV 的 NCHW 相反！）
    info.layout = (qt.v1.rank == 4) ? Layout::NHWC : Layout::Unknown;
    for (uint32_t d = 0; d < qt.v1.rank; ++d)
        info.shape.push_back(qt.v1.dimensions[d]);
}
```

---

#### `infer()` — 零複製輸入

```cpp
std::lock_guard<std::mutex> lock(mutex_);  // QNN context 非線程安全

// 輸入：直接設置 client buffer 指針，不複製
model->input_tensors[0].v1.clientBuf.data = const_cast<void*>(input_ptr);
model->input_tensors[0].v1.memType = QNN_TENSORMEMTYPE_RAW;

qnn_interface_->graphExecute(graph,
    input_tensors.data(),  num_inputs,
    output_tensors.data(), num_outputs,
    nullptr, nullptr);

// 輸出：從 pre-alloc buffer memcpy
for (size_t i = 0; i < output_tensors.size(); ++i) {
    std::memcpy(out_tensor.data_ptr<void>(),
                model->output_buffers[i].data(),
                model->output_buffers[i].size());
}
```

---

#### `convert_model()` — 引導使用外部工具

```cpp
throw ModelError(
    "QNN 不支援裝置上轉換。使用 QNN SDK 離線工具：\n"
    "  qnn-onnx-converter -i " + src + " -o model.cpp\n"
    "  qnn-context-binary-generator --backend libQnnHtp.so \\\n"
    "    --model model.so --output_dir . --binary_file " + dst
);
```

---

### 10.5 `src/vision/classifier.cpp` — 分類器

#### 標籤載入優先級

```
1. 模型內建標籤（model metadata）
2. {model_stem}.txt（與模型同目錄同名）
3. labels.txt（模型所在目錄）
4. IMAGENET_LABELS（內建 10 筆 demo：tench, goldfish, great_white_shark...）
   → 顯示 "class_XXX" 表示找不到任何標籤檔案
```

---

#### `preprocess()` — ImageNet 正規化

```cpp
// 覆寫 InferenceModel 基礎版
cv::resize(image, resized, cv::Size(input_w_, input_h_));
cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
rgb.convertTo(float_img, CV_32F, 1.0 / 255.0);

// ImageNet mean/std（RGB 順序）
// mean = {0.485, 0.456, 0.406}
// std  = {0.229, 0.224, 0.225}
for (int c = 0; c < 3; ++c)
    for (int h = 0; h < input_h_; ++h)
        for (int w = 0; w < input_w_; ++w) {
            float v = float_img.at<cv::Vec3f>(h, w)[c];
            dst[c * H * W + h * W + w] = (v - MEAN[c]) / STD[c];
        }
```

---

### 10.6 `src/vision/detector.cpp` — 偵測器

#### 模型類型推斷

```cpp
// 從檔名關鍵字推斷（預設 YOLOX）
if (name.find("yolox") != npos)  model_type_ = YOLOX;
else if (name.find("yolov5") != npos)  model_type_ = YOLOv5;
// ...
```

#### `preprocess()` — Letterbox + YOLOX 例外

```cpp
float scale = std::min((float)tw / ow, (float)th / oh);
int new_w = (int)(ow * scale), new_h = (int)(oh * scale);
cv::resize(image, resized, {new_w, new_h});

cv::Mat padded(th, tw, CV_8UC3, cv::Scalar(114, 114, 114));
resized.copyTo(padded(cv::Rect(pad_w, pad_h, new_w, new_h)));

if (model_type_ == YOLOX) {
    padded.convertTo(float_img, CV_32F);  // [0,255]，保留 BGR
} else {
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(float_img, CV_32F, 1.0 / 255.0);
}
```

#### YOLOX Grid Decode

```cpp
static const int STRIDES[] = {8, 16, 32};
int offset = 0;
for (int s : STRIDES) {
    for (int gy = 0; gy < H/s; ++gy)
        for (int gx = 0; gx < W/s; ++gx) {
            float cx = (pred[0] + gx) * s;   // 相對偏移 + grid → 乘 stride
            float cy = (pred[1] + gy) * s;
            float w  = std::exp(pred[2]) * s;
            float h  = std::exp(pred[3]) * s;
            ++offset;
        }
}
```

#### 自訂 Greedy NMS（非 `cv::dnn::NMSBoxes`）

```cpp
// 依 confidence 降序排列，per class
std::sort(dets.begin(), dets.end(),
          [](auto& a, auto& b){ return a.confidence > b.confidence; });

for (size_t i = 0; i < dets.size(); ++i) {
    if (suppressed[i]) continue;
    for (size_t j = i+1; j < dets.size(); ++j)
        if (dets[i].class_id == dets[j].class_id &&
            dets[i].bbox.iou(dets[j].bbox) > iou_thresh)
            suppressed[j] = true;
}
```

---

### 10.7 `src/vision/segmentor.cpp` — 分割器

#### 標籤集自動偵測

```cpp
auto name = to_lower(fs::path(model_path).stem().string());
if (name.find("cityscapes") != npos) label_set_ = Cityscapes;  // 19 類
else if (name.find("ade") != npos)   label_set_ = ADE20K;      // 150 類
else                                  label_set_ = VOC;          // 21 類（預設）
```

#### `postprocess()` — 雙格式支援

```cpp
if (shape.size() == 4) {
    // [1, C, H, W] → pixel argmax
    for (int h = 0; h < H; ++h) for (int w = 0; w < W; ++w) {
        int best = 0; float best_v = -FLT_MAX;
        for (int c = 0; c < C; ++c) { ... }
        mask.at<int>(h, w) = best;
    }
} else {
    // [1, H, W] → 已是 argmax 結果（TFLite 格式）
    mask = Tensor_to_Mat_int(output);
}
cv::resize(mask, mask, original_size, 0, 0, cv::INTER_NEAREST);
```

#### `init_default_colormap()` — HSV + VOC 特殊調色板

```cpp
colormap_[0] = {0, 0, 0};  // 背景 → 黑色

if (label_set_ == VOC) {
    // VOC 21 色固定調色板（非 HSV 生成）
    static const cv::Vec3b VOC_COLORS[] = {
        {0,0,0}, {128,0,0}, {0,128,0}, {128,128,0}, ... };
    return;
}

// 其他：HSV 均勻分佈
for (int i = 1; i < num_classes_; ++i) {
    float h = (float)(i * 360) / num_classes_;
    cv::Mat hsv(1,1,CV_8UC3, cv::Scalar(h/2, 255, 255));
    cv::Mat bgr; cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    colormap_[i] = bgr.at<cv::Vec3b>(0,0);
}
```

---

### 10.8 `src/core/device_manager.cpp` — 設備管理

#### `discover_openvino_devices()` — stderr 抑制 + NVIDIA 過濾

```cpp
// 抑制 OV 初始化噪音
int saved = dup(STDERR_FILENO);
int devnull = open("/dev/null", O_WRONLY);
dup2(devnull, STDERR_FILENO);

ov::Core core;
auto device_list = core.get_available_devices();

dup2(saved, STDERR_FILENO); close(devnull); close(saved);

for (const auto& d : device_list) {
    if (is_nvidia_gpu(d)) continue;  // ⚠️ 過濾 NVIDIA，由 TRT 負責
    register_device(d, BackendType::OpenVINO);
}
```

#### 日誌系統

```cpp
void set_log_level(int level) {
    g_parsed_log_level = level;  // TRT logger 讀此全域變數
    // 0=Error, 1=Warning, 2=Info, 3=Verbose
    setenv("OPENVINO_LOG_LEVEL", std::to_string(level).c_str(), 1);
}
```

#### `clear_cache()` — 只刪 engine，保留 ONNX

```cpp
for (auto& entry : fs::recursive_directory_iterator(dir)) {
    auto ext = entry.path().extension().string();
    if (ext == ".engine" || ext == ".trt" || ext == ".blob")
        fs::remove(entry.path());
    // ✅ 保留 .onnx, .xml, .bin
}
```

---

### 10.9 前處理差異對照表

| | InferenceModel（基礎） | Classifier | Detector（非YOLOX） | Detector（YOLOX） | Segmentor |
|---|---|---|---|---|---|
| **Resize** | 直接 resize | 直接 resize | Letterbox | Letterbox | 直接 resize |
| **顏色空間** | BGR（不轉） | BGR→RGB | BGR→RGB | BGR（保留） | BGR→RGB |
| **數值範圍** | [0, 1] | [0, 1] | [0, 1] | [0, 255] | [0, 1] |
| **均值/標準差** | 無 | ImageNet | 無 | 無 | ImageNet |
| **CHW 轉換** | cv::split+memcpy | pixel loop | pixel/memcpy | pixel/memcpy | pixel loop |
| **填充顏色** | 無 | 無 | (114,114,114) | (114,114,114) | 無 |

---

### 10.10 後處理差異對照表

| | Classifier | Detector | Segmentor |
|---|---|---|---|
| **NMS** | 無 | 自訂 greedy NMS（per class） | 無 |
| **softmax** | 有（全分布） | 無 | 無 |
| **座標恢復** | 無 | letterbox 反算（scale/pad）| 無 |
| **格式解碼** | argmax+partial_sort | YOLO/YOLOX 格式解析 | argmax |
| **resize 插值** | 無 | 無 | INTER_NEAREST |
| **批次支援** | 真批次（動態 batch） | 真批次（動態 batch） | 循序 only |

---

*本文件由原始碼逐行分析生成，涵蓋 `include/` 與 `src/` 下全部 C++ 檔案。*
