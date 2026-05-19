# iVIT-SDK API 規格文件

| 文件編號 | API-SPEC-001 |
|----------|--------------|
| 版本 | 1.1 |
| 狀態 | Updated |
| 建立日期 | 2026-01-24 |
| 更新日期 | 2026-01-26 |
| 作者 | 系統架構師 |

> **v1.1 更新內容**：新增 `ivit.load()` 快速 API、`devices` 模組、`zoo` 模組、`stream()` 方法、CLI 工具文件。

---

## 1. API 概覽

iVIT-SDK 提供 Python 和 C++ 雙語 API，功能對等。本文件定義完整的 API 規格。

### 1.1 模組結構

```
ivit
├── core                    # 核心 API
│   ├── load_model()
│   ├── list_devices()
│   ├── get_device()
│   └── set_log_level()
├── devices                 # 裝置管理 (NEW)
│   ├── devices()           # 列出所有裝置
│   ├── devices.cuda()      # 取得 CUDA 裝置
│   ├── devices.npu()       # 取得 NPU 裝置
│   └── devices.best()      # 取得最佳裝置
├── zoo                     # Model Zoo (NEW)
│   ├── list_models()
│   ├── search()
│   ├── download()
│   └── load()
├── vision                  # 電腦視覺任務
│   ├── Classifier
│   ├── Detector
│   ├── Segmentor
│   ├── PoseEstimator
│   └── FaceAnalyzer
├── train                   # 訓練功能 (Future)
│   ├── Trainer
│   ├── Dataset
│   └── Augmentation
└── utils                   # 工具函式
    ├── Visualizer
    ├── Profiler
    └── VideoStream
```

### 1.2 快速 API

```python
import ivit

# 一行載入並推論
model = ivit.load("yolox-s.onnx")
results = model("image.jpg")
results.show()

# 裝置選擇
ivit.devices()                    # 列出所有裝置
model = ivit.load("model.onnx", device=ivit.devices.best())

# Model Zoo
ivit.zoo.list_models()            # 列出所有模型
model = ivit.zoo.load("yolox-s")  # 自動下載並載入
```

---

## 2. 核心 API (ivit)

### 2.1 load_model

載入模型並返回可用於推論的模型物件。

#### Python

```python
def load_model(
    path: Union[str, Path],
    device: str = "auto",
    backend: str = "auto",
    task: Optional[str] = None,
    batch_size: int = 1,
    precision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs
) -> Model:
    """
    載入模型。

    Args:
        path: 模型路徑。支援:
            - 本地檔案路徑 (例如 "yolox-s.onnx")
            - Model Zoo 名稱 (例如 "yolox-s")
            - URL (例如 "https://...")
        device: 目標設備。選項:
            - "auto": 自動選擇最佳設備
            - "cpu": CPU
            - "gpu" 或 "gpu:0": Intel GPU
            - "npu": Intel NPU
            - "cuda" 或 "cuda:0": NVIDIA GPU
            - "jetson": NVIDIA Jetson
            - "hexagon": Qualcomm Hexagon DSP/NPU
        backend: 推論後端。選項:
            - "auto": 根據設備自動選擇
            - "openvino": Intel OpenVINO
            - "tensorrt": NVIDIA TensorRT
            - "qnn": Qualcomm QNN (規劃中)
            - "onnx": ONNX format (loaded via OpenVINO or TensorRT)
        task: 任務類型提示。選項:
            - "classification"
            - "detection"
            - "segmentation"
            - "pose"
            - "face"
            - None: 自動推斷
        batch_size: 批次大小
        precision: 推論精度。選項:
            - None: 使用模型原始精度
            - "fp32": 32位元浮點
            - "fp16": 16位元浮點
            - "int8": 8位元整數
        cache_dir: 模型快取目錄

    Returns:
        Model: 載入的模型物件

    Raises:
        ModelLoadError: 模型載入失敗
        DeviceNotFoundError: 指定設備不可用
        UnsupportedFormatError: 不支援的模型格式

    Examples:
        >>> # 基本用法
        >>> model = ivit.load_model("yolox-s.onnx")

        >>> # 指定設備和後端
        >>> model = ivit.load_model(
        ...     "yolox-s.onnx",
        ...     device="cuda:0",
        ...     backend="tensorrt",
        ...     precision="fp16"
        ... )

        >>> # 從 Model Zoo 載入
        >>> model = ivit.load_model("yolox-s", task="detection")
    """
```

#### C++

```cpp
namespace ivit {

/// 載入模型
/// @param path 模型路徑或 Model Zoo 名稱
/// @param config 載入設定
/// @return 載入的模型
/// @throws ModelLoadError 模型載入失敗
/// @throws DeviceNotFoundError 指定設備不可用
std::shared_ptr<Model> load_model(
    const std::string& path,
    const LoadConfig& config = LoadConfig::default_()
);

struct LoadConfig {
    std::string device = "auto";      ///< 目標設備
    std::string backend = "auto";     ///< 推論後端
    std::string task = "";            ///< 任務類型
    int batch_size = 1;               ///< 批次大小
    std::string precision = "";       ///< 推論精度
    std::string cache_dir = "";       ///< 快取目錄

    /// 建立預設設定
    static LoadConfig default_();
};

} // namespace ivit
```

### 2.2 list_devices

列出所有可用的推論設備。

#### Python

```python
def list_devices() -> List[DeviceInfo]:
    """
    列出所有可用設備。

    Returns:
        List[DeviceInfo]: 可用設備列表

    Examples:
        >>> devices = ivit.list_devices()
        >>> for dev in devices:
        ...     print(f"{dev.id}: {dev.name} ({dev.backend})")
        cpu: Intel Core Ultra 7 165H (openvino)
        gpu:0: Intel Arc Graphics (openvino)
        npu: Intel AI Boost (openvino)
        cuda:0: NVIDIA GeForce RTX 4090 (tensorrt)
    """

@dataclass
class DeviceInfo:
    id: str                           # 設備 ID
    name: str                         # 設備名稱
    backend: str                      # 對應後端
    type: str                         # 設備類型 (cpu, gpu, npu)
    memory_total: int                 # 總記憶體 (bytes)
    memory_available: int             # 可用記憶體 (bytes)
    supported_precisions: List[str]   # 支援的精度
```

#### C++

```cpp
namespace ivit {

/// 列出可用設備
std::vector<DeviceInfo> list_devices();

struct DeviceInfo {
    std::string id;
    std::string name;
    std::string backend;
    std::string type;
    size_t memory_total;
    size_t memory_available;
    std::vector<std::string> supported_precisions;
};

} // namespace ivit
```

### 2.3 get_best_device

取得最適合特定任務的設備。

#### Python

```python
def get_best_device(
    task: Optional[str] = None,
    model_size: Optional[str] = None,
    priority: str = "performance"
) -> DeviceInfo:
    """
    取得最佳設備。

    Args:
        task: 任務類型 (classification, detection, etc.)
        model_size: 模型大小 (small, medium, large)
        priority: 優先考量:
            - "performance": 效能優先
            - "efficiency": 能效優先
            - "memory": 記憶體優先

    Returns:
        DeviceInfo: 推薦的設備

    Examples:
        >>> best = ivit.get_best_device(task="detection")
        >>> print(f"推薦設備: {best.name}")
    """
```

### 2.4 set_log_level

設定日誌等級。

#### Python

```python
def set_log_level(level: str) -> None:
    """
    設定日誌等級。

    Args:
        level: 日誌等級。選項:
            - "debug"
            - "info"
            - "warning"
            - "error"
            - "critical"

    Examples:
        >>> ivit.set_log_level("debug")
    """
```

---

## 2.5 ivit.load (快速 API)

簡化的模型載入函式，提供 簡潔的單行 API。

#### Python

```python
def load(
    source: str,
    device: str = "auto",
    task: str = None,
    **kwargs
) -> Model:
    """
    載入模型（簡化 API）。

    這是 iVIT-SDK 的主要入口點，提供單行載入、簡潔 API。

    Args:
        source: 模型來源
            - 本地路徑 (例如 "yolox-s.onnx")
            - Model Zoo 名稱 (例如 "yolox-s")
        device: 目標設備
            - "auto": 自動選擇最佳設備
            - "cpu": CPU
            - "cuda:0": NVIDIA GPU
            - "npu": Intel NPU
            - Device 物件 (來自 ivit.devices)
        task: 任務類型提示 (auto-detected if None)
            - "detect": 物件偵測
            - "classify": 圖像分類
            - "segment": 語意分割
            - "pose": 姿態估計

    Returns:
        Model: 載入的模型，可直接呼叫進行推論

    Examples:
        >>> import ivit
        >>>
        >>> # 最簡單用法
        >>> model = ivit.load("yolox-s.onnx")
        >>> results = model("image.jpg")
        >>> results.show()
        >>>
        >>> # 指定設備
        >>> model = ivit.load("model.onnx", device="cuda:0")
        >>> model = ivit.load("model.onnx", device=ivit.devices.best())
        >>>
        >>> # 從 Model Zoo 載入
        >>> model = ivit.load("yolox-s")  # 自動下載
    """
```

---

## 2.6 devices 模組

設備發現與選擇 API。

#### Python

```python
# 列出所有設備
def devices() -> List[Device]:
    """
    列出所有可用設備。

    Returns:
        List[Device]: 設備列表

    Examples:
        >>> for d in ivit.devices():
        ...     print(f"{d.id}: {d.name}")
        cpu: Intel Core i7-12700
        cuda:0: NVIDIA RTX 3080
        npu: Intel AI Boost
    """

# 取得特定類型設備
devices.cpu() -> Device      # CPU 設備
devices.cuda(index=0) -> Device  # CUDA GPU
devices.npu() -> Device      # Intel NPU
devices.gpu(index=0) -> Device   # Intel GPU

# 取得最佳設備
def devices.best(priority: str = "performance") -> Device:
    """
    取得最佳設備。

    Args:
        priority: 優先考量
            - "performance": 效能優先（預設）
            - "efficiency": 能效優先

    Returns:
        Device: 推薦的設備

    Examples:
        >>> best = ivit.devices.best()
        >>> model = ivit.load("model.onnx", device=best)
    """

@dataclass
class Device:
    id: str          # 設備 ID (例如 "cuda:0")
    name: str        # 設備名稱
    type: str        # 設備類型 (cpu, gpu, npu)
    backend: str     # 推論後端
    available: bool  # 是否可用

# 快速存取別名
D = devices  # 簡短別名

# 使用範例
model = ivit.load("model.onnx", device=ivit.D.cuda())
```

---

## 2.7 zoo 模組 (Model Zoo)

預訓練模型庫管理。

#### Python

```python
# 列出模型
def zoo.list_models(task: Optional[str] = None) -> List[str]:
    """
    列出 Model Zoo 中的模型。

    Args:
        task: 按任務過濾 (detect, classify, segment, pose)

    Returns:
        List[str]: 模型名稱列表

    Examples:
        >>> ivit.zoo.list_models()
        ['yolox-s', 'yolox-m', 'resnet50', ...]
        >>> ivit.zoo.list_models(task="detect")
        ['yolox-nano', 'yolox-tiny', 'yolox-s', 'yolox-m', 'yolox-l']
    """

# 搜尋模型
def zoo.search(query: str) -> List[str]:
    """
    搜尋模型。

    Args:
        query: 搜尋關鍵字

    Returns:
        List[str]: 符合的模型名稱

    Examples:
        >>> ivit.zoo.search("yolo")
        ['yolox-nano', 'yolox-tiny', 'yolox-s', ...]
        >>> ivit.zoo.search("edge")  # 找邊緣裝置模型
        ['yolox-nano', 'mobilenet_v3_small', ...]
    """

# 取得模型資訊
def zoo.get_model_info(name: str) -> ModelInfo:
    """
    取得模型詳細資訊。

    Args:
        name: 模型名稱

    Returns:
        ModelInfo: 模型資訊

    Examples:
        >>> info = ivit.zoo.get_model_info("yolox-s")
        >>> print(f"Task: {info.task}, Input: {info.input_size}")
        Task: detect, Input: (640, 640)
    """

# 下載模型
def zoo.download(
    name: str,
    format: str = "onnx",
    force: bool = False,
) -> Path:
    """
    下載模型。

    Args:
        name: 模型名稱
        format: 模型格式 (onnx, openvino, tensorrt)
        force: 強制重新下載

    Returns:
        Path: 下載的模型路徑

    Examples:
        >>> path = ivit.zoo.download("yolox-s")
        >>> print(path)
        /home/user/.cache/ivit/models/yolox-s.onnx
    """

# 載入模型（下載 + 載入）
def zoo.load(
    name: str,
    device: str = "auto",
    format: str = "onnx",
    **kwargs
) -> Model:
    """
    從 Model Zoo 載入模型。

    Args:
        name: 模型名稱或本地路徑
        device: 目標設備
        format: 模型格式

    Returns:
        Model: 載入的模型

    Examples:
        >>> model = ivit.zoo.load("yolox-s", device="cuda:0")
        >>> results = model("image.jpg")
    """

@dataclass
class ModelInfo:
    name: str                 # 模型名稱
    task: str                 # 任務類型
    description: str          # 描述
    input_size: tuple         # 輸入尺寸 (H, W)
    num_classes: int          # 類別數
    formats: List[str]        # 可用格式
    metrics: Dict[str, float] # 效能指標
    tags: List[str]           # 標籤
```

---

## 3. Model 類別

載入的模型物件，用於執行推論。

### 3.1 屬性

#### Python

```python
class Model:
    @property
    def name(self) -> str:
        """模型名稱"""

    @property
    def task(self) -> str:
        """任務類型"""

    @property
    def device(self) -> str:
        """執行設備"""

    @property
    def backend(self) -> str:
        """推論後端"""

    @property
    def input_info(self) -> List[TensorInfo]:
        """輸入張量資訊"""

    @property
    def output_info(self) -> List[TensorInfo]:
        """輸出張量資訊"""

    @property
    def memory_usage(self) -> int:
        """記憶體使用量 (bytes)"""
```

### 3.2 predict 方法

執行推論。

#### Python

```python
def predict(
    self,
    source: Union[str, Path, np.ndarray, List],
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.45,
    max_detections: int = 100,
    classes: Optional[List[int]] = None,
    **kwargs
) -> Results:
    """
    執行推論。

    Args:
        source: 輸入來源。支援:
            - 影像路徑 (str, Path)
            - NumPy 陣列 (np.ndarray)
            - 影像列表 (批次推論)
            - URL
        conf_threshold: 信心分數閾值
        iou_threshold: NMS IoU 閾值
        max_detections: 最大偵測數量
        classes: 只保留指定類別 (None 保留全部)

    Returns:
        Results: 推論結果

    Examples:
        >>> results = model.predict("image.jpg")
        >>> results = model.predict(np.array(...))
        >>> results = model.predict(["img1.jpg", "img2.jpg"])
    """

def stream(
    self,
    source: Union[str, int],
    conf: float = 0.5,
    iou: float = 0.45,
    classes: Optional[List[int]] = None,
    max_det: int = 300,
    show: bool = False,
    save: bool = False,
    save_dir: str = "runs/detect",
    loop: bool = False,
    **kwargs
) -> Generator[Results, None, None]:
    """
    串流推論（影片或攝影機）。

    使用 generator 模式逐幀處理，適合即時應用。

    Args:
        source: 輸入來源
            - 影片路徑 (str)
            - 攝影機索引 (int, 例如 0)
            - RTSP URL (str)
        conf: 信心閾值
        iou: NMS IoU 閾值
        classes: 只保留指定類別
        max_det: 最大偵測數量
        show: 即時顯示結果
        save: 儲存結果影片
        save_dir: 儲存目錄
        loop: 是否循環播放

    Yields:
        Results: 每幀的推論結果

    Examples:
        >>> # 處理影片
        >>> for results in model.stream("video.mp4"):
        ...     print(f"Frame: {results.frame_idx}, Detections: {len(results)}")
        ...     if results.detections:
        ...         results.show()
        >>>
        >>> # 處理攝影機
        >>> for results in model.stream(0, show=True):
        ...     if len(results) > 10:
        ...         print("Too many objects!")
        >>>
        >>> # 儲存結果
        >>> for results in model.stream("video.mp4", save=True):
        ...     pass  # 自動儲存到 runs/detect/
    """

def predict_batch(
    self,
    sources: List[Union[str, np.ndarray]],
    conf: float = 0.5,
    iou: float = 0.45,
    **kwargs
) -> List[Results]:
    """
    批次推論多張圖片。

    Args:
        sources: 圖片列表（路徑或 numpy 陣列）
        conf: 信心閾值
        iou: NMS IoU 閾值

    Returns:
        List[Results]: 每張圖片的結果

    Examples:
        >>> images = ["img1.jpg", "img2.jpg", "img3.jpg"]
        >>> results_list = model.predict_batch(images)
        >>> for path, results in zip(images, results_list):
        ...     print(f"{path}: {len(results)} detections")
    """

def __call__(
    self,
    source: Union[str, np.ndarray, List],
    **kwargs
) -> Results:
    """
    呼叫模型進行推論。

    允許直接呼叫模型物件：model(image)

    Args:
        source: 輸入來源
        **kwargs: 傳遞給 predict() 的參數

    Returns:
        Results: 推論結果

    Examples:
        >>> model = ivit.load("yolox-s.onnx")
        >>> results = model("image.jpg")  # 等同於 model.predict("image.jpg")
        >>> results.show()
    """
```

#### C++

```cpp
namespace ivit {

class Model {
public:
    /// 執行推論
    Results predict(
        const cv::Mat& image,
        const InferConfig& config = InferConfig::default_()
    );

    /// 批次推論
    std::vector<Results> predict_batch(
        const std::vector<cv::Mat>& images,
        const InferConfig& config = InferConfig::default_()
    );

    /// 屬性
    std::string name() const;
    std::string task() const;
    std::string device() const;
    std::string backend() const;
    std::vector<TensorInfo> input_info() const;
    std::vector<TensorInfo> output_info() const;
    size_t memory_usage() const;
};

struct InferConfig {
    float conf_threshold = 0.5f;
    float iou_threshold = 0.45f;
    int max_detections = 100;
    std::vector<int> classes = {};

    static InferConfig default_();
};

} // namespace ivit
```

---

## 4. Results 類別

推論結果容器。

### 4.1 屬性

#### Python

```python
class Results:
    # 分類結果
    @property
    def classifications(self) -> List[ClassificationResult]:
        """分類結果列表"""

    @property
    def top1(self) -> ClassificationResult:
        """Top-1 分類結果"""

    @property
    def top5(self) -> List[ClassificationResult]:
        """Top-5 分類結果"""

    # 偵測結果
    @property
    def detections(self) -> List[Detection]:
        """偵測結果列表"""

    # 分割結果
    @property
    def segmentation_mask(self) -> Optional[np.ndarray]:
        """分割遮罩 (H, W) 或 (H, W, C)"""

    # 姿態估計結果
    @property
    def keypoints(self) -> List[List[Keypoint]]:
        """關鍵點列表 (每個人一組)"""

    # 原始輸出
    @property
    def raw_outputs(self) -> Dict[str, np.ndarray]:
        """原始模型輸出"""

    # 元資料
    @property
    def inference_time_ms(self) -> float:
        """推論時間 (毫秒)"""

    @property
    def device_used(self) -> str:
        """使用的設備"""

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        """輸入影像形狀 (H, W, C)"""

    # 串流相關屬性 (NEW)
    @property
    def frame_idx(self) -> Optional[int]:
        """影格索引（串流模式）"""

    @property
    def source_fps(self) -> Optional[float]:
        """來源 FPS（串流模式）"""
```

### 4.2 方法

#### 快速方法 (NEW)

```python
class Results:
    def show(self, wait: bool = True) -> None:
        """
        快速顯示結果。

        在新視窗中顯示視覺化結果。

        Args:
            wait: 是否等待按鍵關閉

        Examples:
            >>> results = model("image.jpg")
            >>> results.show()  # 開啟視窗顯示
        """

    def save(
        self,
        path: Union[str, Path] = None,
        format: str = "auto"
    ) -> str:
        """
        快速儲存結果。

        支援儲存為圖片或結構化資料。

        Args:
            path: 儲存路徑
                - None: 自動產生路徑 (runs/detect/result_N.jpg)
                - 圖片路徑: 儲存視覺化圖片
                - JSON 路徑: 儲存結構化資料
            format: 格式
                - "auto": 根據副檔名自動判斷
                - "json": JSON 格式
                - "image": 視覺化圖片

        Returns:
            str: 儲存的路徑

        Examples:
            >>> results = model("image.jpg")
            >>> results.save("output.jpg")      # 儲存視覺化圖片
            >>> results.save("results.json")    # 儲存 JSON
            >>> path = results.save()           # 自動路徑
        """

    def plot(
        self,
        show_labels: bool = True,
        show_conf: bool = True,
        line_width: int = 2,
        font_size: float = 0.5,
    ) -> np.ndarray:
        """
        繪製結果到原始圖片上。

        Args:
            show_labels: 顯示類別標籤
            show_conf: 顯示信心分數
            line_width: 線條寬度
            font_size: 字體大小

        Returns:
            np.ndarray: 繪製後的圖片

        Examples:
            >>> results = model("image.jpg")
            >>> annotated = results.plot()
            >>> cv2.imwrite("annotated.jpg", annotated)
        """

    def __len__(self) -> int:
        """返回偵測數量"""

    def __iter__(self):
        """迭代偵測結果"""
```

#### 進階方法

#### Python

```python
class Results:
    def visualize(
        self,
        image: Optional[np.ndarray] = None,
        show_labels: bool = True,
        show_confidence: bool = True,
        show_boxes: bool = True,
        show_masks: bool = True,
        colors: Optional[Dict[str, Tuple]] = None,
        font_scale: float = 0.5,
        thickness: int = 2,
    ) -> np.ndarray:
        """
        視覺化結果。

        Args:
            image: 原始影像 (如果不提供，使用推論時的影像)
            show_labels: 顯示類別標籤
            show_confidence: 顯示信心分數
            show_boxes: 顯示邊界框
            show_masks: 顯示分割遮罩
            colors: 自訂類別顏色
            font_scale: 字體大小
            thickness: 線條粗細

        Returns:
            np.ndarray: 視覺化後的影像
        """

    def save(
        self,
        path: Union[str, Path],
        format: str = "auto"
    ) -> None:
        """
        儲存結果。

        Args:
            path: 儲存路徑
            format: 格式。選項:
                - "auto": 根據副檔名自動選擇
                - "json": JSON 格式
                - "csv": CSV 格式
                - "xml": Pascal VOC XML 格式
                - "txt": YOLO 格式
        """

    def to_json(self) -> str:
        """轉換為 JSON 字串"""

    def to_dict(self) -> Dict:
        """轉換為字典"""

    def filter(
        self,
        classes: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        min_area: Optional[float] = None,
    ) -> 'Results':
        """
        過濾結果。

        Args:
            classes: 保留的類別
            min_confidence: 最小信心分數
            min_area: 最小邊界框面積

        Returns:
            Results: 過濾後的結果
        """

    def __len__(self) -> int:
        """返回結果數量"""

    def __iter__(self):
        """迭代結果"""

    def __getitem__(self, idx: int):
        """索引存取"""
```

---

## 5. 視覺任務 API (ivit.vision)

### 5.1 Classifier

圖像分類器。

#### Python

```python
class Classifier:
    def __init__(
        self,
        model: Union[str, Model],
        device: str = "auto",
        **kwargs
    ):
        """
        建立分類器。

        Args:
            model: 模型名稱或已載入的 Model 物件
            device: 目標設備

        Examples:
            >>> classifier = Classifier("efficientnet_b0")
            >>> classifier = Classifier("resnet50", device="cuda:0")
        """

    def predict(
        self,
        image: Union[str, np.ndarray],
        top_k: int = 5
    ) -> ClassificationResults:
        """
        分類圖像。

        Args:
            image: 輸入影像
            top_k: 返回前 K 個結果

        Returns:
            ClassificationResults: 分類結果
        """

    @property
    def classes(self) -> List[str]:
        """類別列表"""

    @property
    def num_classes(self) -> int:
        """類別數量"""

class ClassificationResults(Results):
    @property
    def top1(self) -> ClassificationResult:
        """Top-1 結果"""

    @property
    def top5(self) -> List[ClassificationResult]:
        """Top-5 結果"""

    @property
    def probabilities(self) -> np.ndarray:
        """所有類別的機率"""

@dataclass
class ClassificationResult:
    class_id: int
    label: str
    score: float
```

#### C++

```cpp
namespace ivit::vision {

class Classifier {
public:
    explicit Classifier(
        const std::string& model,
        const std::string& device = "auto"
    );

    ClassificationResults predict(
        const cv::Mat& image,
        int top_k = 5
    );

    std::vector<std::string> classes() const;
    int num_classes() const;
};

} // namespace ivit::vision
```

### 5.2 Detector

物件偵測器。

#### Python

```python
class Detector:
    def __init__(
        self,
        model: Union[str, Model],
        device: str = "auto",
        **kwargs
    ):
        """
        建立偵測器。

        Args:
            model: 模型名稱或 Model 物件
            device: 目標設備

        Examples:
            >>> detector = Detector("yolox-s")
            >>> detector = Detector("yolox-m", device="cuda:0")
        """

    def predict(
        self,
        image: Union[str, np.ndarray],
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        classes: Optional[List[int]] = None,
        max_detections: int = 100,
    ) -> DetectionResults:
        """
        偵測物件。

        Args:
            image: 輸入影像
            conf_threshold: 信心閾值
            iou_threshold: NMS IoU 閾值
            classes: 只偵測指定類別
            max_detections: 最大偵測數量

        Returns:
            DetectionResults: 偵測結果
        """

    def predict_video(
        self,
        source: Union[str, int],
        callback: Callable[[DetectionResults, np.ndarray], None],
        **kwargs
    ) -> None:
        """
        串流影片偵測。

        Args:
            source: 影片路徑或攝影機 ID
            callback: 每幀回調函式
        """

    @property
    def classes(self) -> List[str]:
        """類別列表"""

class DetectionResults(Results):
    @property
    def detections(self) -> List[Detection]:
        """偵測列表"""

    def filter_by_class(self, classes: List[str]) -> 'DetectionResults':
        """按類別過濾"""

    def filter_by_confidence(self, min_conf: float) -> 'DetectionResults':
        """按信心分數過濾"""

    def filter_by_area(self, min_area: float, max_area: float = None) -> 'DetectionResults':
        """按面積過濾"""

@dataclass
class Detection:
    bbox: BBox                        # 邊界框
    class_id: int                     # 類別 ID
    label: str                        # 類別標籤
    confidence: float                 # 信心分數
    mask: Optional[np.ndarray] = None # 實例遮罩 (如果有)

@dataclass
class BBox:
    x1: float                         # 左上角 x
    y1: float                         # 左上角 y
    x2: float                         # 右下角 x
    y2: float                         # 右下角 y

    @property
    def width(self) -> float: ...
    @property
    def height(self) -> float: ...
    @property
    def area(self) -> float: ...
    @property
    def center(self) -> Tuple[float, float]: ...

    def iou(self, other: 'BBox') -> float: ...
    def to_xywh(self) -> Tuple[float, float, float, float]: ...
    def to_cxcywh(self) -> Tuple[float, float, float, float]: ...
```

### 5.3 Segmentor

語意分割器。

#### Python

```python
class Segmentor:
    def __init__(
        self,
        model: Union[str, Model],
        device: str = "auto",
        **kwargs
    ):
        """
        建立分割器。

        Args:
            model: 模型名稱或 Model 物件
            device: 目標設備

        Examples:
            >>> segmentor = Segmentor("deeplabv3_resnet50")
        """

    def predict(
        self,
        image: Union[str, np.ndarray],
    ) -> SegmentationResults:
        """
        語意分割。

        Args:
            image: 輸入影像

        Returns:
            SegmentationResults: 分割結果
        """

    @property
    def classes(self) -> List[str]:
        """類別列表"""

    @property
    def num_classes(self) -> int:
        """類別數量"""

class SegmentationResults(Results):
    @property
    def mask(self) -> np.ndarray:
        """分割遮罩 (H, W)，值為類別 ID"""

    @property
    def class_masks(self) -> Dict[str, np.ndarray]:
        """每個類別的二值遮罩"""

    def colorize(
        self,
        colormap: Optional[Dict[int, Tuple]] = None
    ) -> np.ndarray:
        """
        彩色化遮罩。

        Args:
            colormap: 自訂色彩映射

        Returns:
            np.ndarray: RGB 彩色遮罩
        """

    def overlay(
        self,
        image: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        疊加到原圖。

        Args:
            image: 原始影像
            alpha: 透明度

        Returns:
            np.ndarray: 疊加後的影像
        """

    def get_contours(
        self,
        class_id: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        提取輪廓。

        Args:
            class_id: 指定類別 (None 為全部)

        Returns:
            List[np.ndarray]: 輪廓列表
        """
```

### 5.4 PoseEstimator

姿態估計器。

#### Python

```python
class PoseEstimator:
    def __init__(
        self,
        model: Union[str, Model] = "rtmpose-s",
        device: str = "auto",
        **kwargs
    ):
        """
        建立姿態估計器。

        Args:
            model: 模型名稱或 Model 物件
            device: 目標設備
        """

    def predict(
        self,
        image: Union[str, np.ndarray],
        conf_threshold: float = 0.5,
    ) -> PoseResults:
        """
        估計人體姿態。

        Args:
            image: 輸入影像
            conf_threshold: 信心閾值

        Returns:
            PoseResults: 姿態結果
        """

    @property
    def keypoint_names(self) -> List[str]:
        """關鍵點名稱列表"""

    @property
    def skeleton(self) -> List[Tuple[int, int]]:
        """骨架連接關係"""

class PoseResults(Results):
    @property
    def poses(self) -> List[Pose]:
        """姿態列表（每人一個）"""

    def draw_skeleton(
        self,
        image: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """繪製骨架"""

@dataclass
class Pose:
    keypoints: List[Keypoint]         # 關鍵點列表
    bbox: Optional[BBox] = None       # 邊界框
    confidence: float = 0.0           # 整體信心

    def get_keypoint(self, name: str) -> Optional[Keypoint]:
        """根據名稱取得關鍵點"""

@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float
    name: str
```

### 5.5 FaceAnalyzer

人臉分析器。

#### Python

```python
class FaceAnalyzer:
    def __init__(
        self,
        detect_model: str = "retinaface",
        recognize_model: Optional[str] = "arcface",
        device: str = "auto",
        **kwargs
    ):
        """
        建立人臉分析器。

        Args:
            detect_model: 人臉偵測模型
            recognize_model: 人臉識別模型 (None 只做偵測)
            device: 目標設備
        """

    def detect(
        self,
        image: Union[str, np.ndarray],
        conf_threshold: float = 0.5,
    ) -> FaceDetectionResults:
        """偵測人臉"""

    def extract_embedding(
        self,
        image: Union[str, np.ndarray],
        face: Optional[Face] = None,
    ) -> np.ndarray:
        """
        提取人臉特徵向量。

        Args:
            image: 輸入影像
            face: 指定人臉 (None 使用最大的)

        Returns:
            np.ndarray: 512 維特徵向量
        """

    def compare(
        self,
        face1: Union[np.ndarray, str],
        face2: Union[np.ndarray, str],
    ) -> float:
        """
        比較兩張人臉相似度。

        Returns:
            float: 餘弦相似度 (0-1)
        """

@dataclass
class Face:
    bbox: BBox
    confidence: float
    landmarks: Optional[np.ndarray] = None  # (5, 2) 五點地標
    embedding: Optional[np.ndarray] = None  # 512-d 特徵
```

---

## 6. 訓練 API (ivit.train)

### 6.1 Dataset

資料集類別。

#### Python

```python
class ImageFolderDataset:
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        train_split: float = 0.8,
        seed: int = 42,
    ):
        """
        從資料夾載入分類資料集。

        Args:
            root: 資料夾根目錄，結構:
                root/
                    class_a/
                        img1.jpg
                        img2.jpg
                    class_b/
                        img3.jpg
            transform: 資料轉換
            train_split: 訓練集比例
            seed: 隨機種子

        Examples:
            >>> dataset = ImageFolderDataset("./my_dataset")
            >>> print(f"類別: {dataset.classes}")
            >>> print(f"訓練集: {len(dataset.train_set)}")
        """

    @property
    def classes(self) -> List[str]:
        """類別列表"""

    @property
    def num_classes(self) -> int:
        """類別數量"""

    @property
    def train_set(self) -> 'Subset':
        """訓練集"""

    @property
    def val_set(self) -> 'Subset':
        """驗證集"""

    @property
    def calibration_set(self) -> 'Subset':
        """校正集 (用於量化)"""

class DetectionDataset:
    def __init__(
        self,
        root: Union[str, Path],
        format: str = "coco",
        transform: Optional[Callable] = None,
        train_split: float = 0.8,
    ):
        """
        物件偵測資料集。

        Args:
            root: 資料夾根目錄
            format: 標註格式:
                - "coco": COCO JSON 格式
                - "yolo": YOLO txt 格式
                - "voc": Pascal VOC XML 格式
            transform: 資料轉換
            train_split: 訓練集比例
        """
```

### 6.2 Augmentation

資料增強。

#### Python

```python
class Augmentation:
    """資料增強管線"""

    @staticmethod
    def default_train() -> 'Augmentation':
        """預設訓練增強"""

    @staticmethod
    def default_val() -> 'Augmentation':
        """預設驗證增強 (只有 resize + normalize)"""

    def __init__(self, transforms: List[Dict]):
        """
        自訂增強管線。

        Args:
            transforms: 轉換列表

        Examples:
            >>> aug = Augmentation([
            ...     {"type": "resize", "size": (224, 224)},
            ...     {"type": "horizontal_flip", "p": 0.5},
            ...     {"type": "color_jitter", "brightness": 0.2},
            ...     {"type": "normalize", "mean": [0.485, 0.456, 0.406],
            ...                           "std": [0.229, 0.224, 0.225]},
            ... ])
        """

# 支援的增強類型
AUGMENTATION_TYPES = {
    "resize": {"size": (224, 224)},
    "center_crop": {"size": 224},
    "random_crop": {"size": 224, "scale": (0.8, 1.0)},
    "horizontal_flip": {"p": 0.5},
    "vertical_flip": {"p": 0.5},
    "rotation": {"degrees": 15},
    "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2},
    "normalize": {"mean": [...], "std": [...]},
    "random_erasing": {"p": 0.5, "scale": (0.02, 0.33)},
    "mixup": {"alpha": 0.2},
    "cutout": {"n_holes": 1, "length": 16},
}
```

### 6.3 Trainer

訓練器。

#### Python

```python
class Trainer:
    def __init__(
        self,
        model: Union[str, Model],
        dataset: Dataset,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        optimizer: str = "adam",
        scheduler: Optional[str] = "cosine",
        device: str = "auto",
        output_dir: Optional[str] = None,
        **kwargs
    ):
        """
        建立訓練器。

        Args:
            model: 模型名稱或 Model 物件
            dataset: 資料集
            epochs: 訓練輪數
            batch_size: 批次大小
            learning_rate: 學習率
            optimizer: 優化器:
                - "adam"
                - "sgd"
                - "adamw"
            scheduler: 學習率排程:
                - "cosine"
                - "step"
                - "plateau"
                - None
            device: 訓練設備
            output_dir: 輸出目錄

        Examples:
            >>> trainer = Trainer(
            ...     model="resnet50",
            ...     dataset=dataset,
            ...     epochs=20,
            ...     batch_size=32,
            ... )
        """

    def fit(
        self,
        callbacks: Optional[List['Callback']] = None,
        resume_from: Optional[str] = None,
    ) -> 'TrainingHistory':
        """
        開始訓練。

        Args:
            callbacks: 回調函式列表
            resume_from: 從 checkpoint 恢復

        Returns:
            TrainingHistory: 訓練歷史
        """

    def evaluate(
        self,
        dataset: Optional[Dataset] = None,
    ) -> Dict[str, float]:
        """
        評估模型。

        Args:
            dataset: 評估資料集 (None 使用驗證集)

        Returns:
            Dict: 指標字典 (accuracy, loss, etc.)
        """

    def export(
        self,
        path: Union[str, Path],
        format: str = "onnx",
        optimize_for: Optional[str] = None,
        quantize: Optional[str] = None,
        calibration_data: Optional[Dataset] = None,
    ) -> None:
        """
        匯出模型。

        Args:
            path: 輸出路徑
            format: 格式:
                - "onnx"
                - "openvino"
                - "tensorrt"
                - "snpe"
            optimize_for: 目標優化:
                - "intel_cpu"
                - "intel_gpu"
                - "intel_npu"
                - "nvidia_gpu"
                - "qualcomm_npu"
            quantize: 量化:
                - "fp16"
                - "int8"
            calibration_data: INT8 校正資料集
        """

    # 回調類別
    class callbacks:
        @staticmethod
        def EarlyStopping(
            monitor: str = "val_loss",
            patience: int = 5,
            min_delta: float = 0.001,
        ) -> 'Callback':
            """早停回調"""

        @staticmethod
        def ModelCheckpoint(
            path: str,
            monitor: str = "val_loss",
            save_best_only: bool = True,
        ) -> 'Callback':
            """模型檢查點回調"""

        @staticmethod
        def LearningRateMonitor() -> 'Callback':
            """學習率監控"""

        @staticmethod
        def ProgressBar() -> 'Callback':
            """進度條"""

@dataclass
class TrainingHistory:
    epochs: List[int]
    train_loss: List[float]
    val_loss: List[float]
    train_accuracy: List[float]
    val_accuracy: List[float]
    learning_rates: List[float]
    best_epoch: int
    best_val_loss: float
    best_val_accuracy: float

    def plot(self, save_path: Optional[str] = None) -> None:
        """繪製訓練曲線"""
```

---

## 7. 工具 API (ivit.utils)

### 7.1 Visualizer

視覺化工具。

#### Python

```python
class Visualizer:
    @staticmethod
    def draw_detections(
        image: np.ndarray,
        detections: List[Detection],
        colors: Optional[Dict[str, Tuple]] = None,
        show_labels: bool = True,
        show_confidence: bool = True,
        thickness: int = 2,
        font_scale: float = 0.5,
    ) -> np.ndarray:
        """繪製偵測結果"""

    @staticmethod
    def draw_segmentation(
        image: np.ndarray,
        mask: np.ndarray,
        colormap: Optional[Dict[int, Tuple]] = None,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """繪製分割結果"""

    @staticmethod
    def draw_poses(
        image: np.ndarray,
        poses: List[Pose],
        skeleton: List[Tuple[int, int]],
        colors: Optional[Dict[str, Tuple]] = None,
        thickness: int = 2,
    ) -> np.ndarray:
        """繪製姿態骨架"""

    @staticmethod
    def draw_faces(
        image: np.ndarray,
        faces: List[Face],
        show_landmarks: bool = True,
    ) -> np.ndarray:
        """繪製人臉"""

    @staticmethod
    def create_comparison(
        images: List[np.ndarray],
        titles: Optional[List[str]] = None,
        cols: int = 2,
    ) -> np.ndarray:
        """建立比較圖"""
```

### 7.2 Profiler

效能分析工具。

#### Python

```python
class Profiler:
    def benchmark(
        self,
        model: Model,
        input_shape: Tuple[int, ...],
        iterations: int = 100,
        warmup: int = 10,
    ) -> 'ProfileReport':
        """
        執行效能基準測試。

        Args:
            model: 模型
            input_shape: 輸入形狀 (N, C, H, W)
            iterations: 測試次數
            warmup: 預熱次數

        Returns:
            ProfileReport: 效能報告
        """

    def profile_layer(
        self,
        model: Model,
        input_shape: Tuple[int, ...],
    ) -> List['LayerProfile']:
        """逐層分析"""

@dataclass
class ProfileReport:
    model_name: str
    device: str
    backend: str
    precision: str
    input_shape: Tuple[int, ...]
    iterations: int

    # 延遲統計 (毫秒)
    latency_mean: float
    latency_median: float
    latency_std: float
    latency_p95: float
    latency_p99: float
    latency_min: float
    latency_max: float

    # 吞吐量
    throughput_fps: float

    # 記憶體
    memory_mb: float

    def __str__(self) -> str:
        """格式化輸出"""

    def to_dict(self) -> Dict:
        """轉為字典"""

    def save(self, path: str) -> None:
        """儲存報告"""
```

### 7.3 VideoStream

影片串流處理。

#### Python

```python
class VideoStream:
    def __init__(
        self,
        source: Union[str, int],
        fps: Optional[int] = None,
        resolution: Optional[Tuple[int, int]] = None,
    ):
        """
        開啟影片串流。

        Args:
            source: 來源:
                - 影片檔案路徑
                - 攝影機 ID (0, 1, ...)
                - RTSP URL
            fps: 目標 FPS (None 使用原始)
            resolution: 目標解析度 (None 使用原始)
        """

    def __iter__(self):
        """迭代每一幀"""

    def __next__(self) -> np.ndarray:
        """取得下一幀"""

    def read(self) -> Optional[np.ndarray]:
        """讀取一幀"""

    def release(self) -> None:
        """釋放資源"""

    @property
    def fps(self) -> float:
        """FPS"""

    @property
    def width(self) -> int:
        """寬度"""

    @property
    def height(self) -> int:
        """高度"""

    @property
    def frame_count(self) -> int:
        """總幀數 (影片檔案)"""

class VideoWriter:
    def __init__(
        self,
        path: str,
        fps: float,
        resolution: Tuple[int, int],
        codec: str = "mp4v",
    ):
        """建立影片寫入器"""

    def write(self, frame: np.ndarray) -> None:
        """寫入一幀"""

    def release(self) -> None:
        """釋放資源"""
```

---

## 8. 例外處理

### 8.1 例外類別

```python
# 基礎例外
class IVITError(Exception):
    """iVIT-SDK 基礎例外"""

# 模型相關
class ModelError(IVITError):
    """模型相關錯誤"""

class ModelLoadError(ModelError):
    """模型載入失敗"""

class ModelConvertError(ModelError):
    """模型轉換失敗"""

class UnsupportedFormatError(ModelError):
    """不支援的模型格式"""

# 設備相關
class DeviceError(IVITError):
    """設備相關錯誤"""

class DeviceNotFoundError(DeviceError):
    """設備不存在"""

class DeviceMemoryError(DeviceError):
    """設備記憶體不足"""

# 推論相關
class InferenceError(IVITError):
    """推論相關錯誤"""

class InputError(InferenceError):
    """輸入錯誤"""

class TimeoutError(InferenceError):
    """推論超時"""

# 訓練相關
class TrainingError(IVITError):
    """訓練相關錯誤"""

class DatasetError(TrainingError):
    """資料集錯誤"""
```

### 8.2 錯誤處理範例

```python
import ivit
from ivit.exceptions import ModelLoadError, DeviceNotFoundError

try:
    model = ivit.load_model("yolox-s.onnx", device="cuda:0")
except DeviceNotFoundError:
    print("CUDA 設備不可用，改用 CPU")
    model = ivit.load_model("yolox-s.onnx", device="cpu")
except ModelLoadError as e:
    print(f"模型載入失敗: {e}")
    raise
```

---

## 9. 設定管理

### 9.1 全域設定

```python
import ivit

# 設定日誌
ivit.set_log_level("info")
ivit.set_log_file("/var/log/ivit/sdk.log")

# 設定快取
ivit.set_cache_dir("~/.cache/ivit")
ivit.set_max_cache_size("10GB")

# 設定預設設備
ivit.set_default_device("cuda:0")

# 設定預設精度
ivit.set_default_precision("fp16")

# 取得設定
config = ivit.get_config()
print(config)
```

### 9.2 設定檔

```yaml
# ~/.ivit/config.yaml

# 日誌設定
logging:
  level: info
  file: ~/.ivit/logs/sdk.log
  rotation: 10MB
  retention: 7

# 快取設定
cache:
  dir: ~/.cache/ivit
  max_size: 10GB
  ttl: 30d

# 預設設定
defaults:
  device: auto
  precision: fp16
  backend: auto

# Model Zoo
model_zoo:
  registry_url: https://models.ivit.ai/registry.json
  download_dir: ~/.ivit/models

# 效能設定
performance:
  num_threads: 4
  enable_profiling: false
```

---

## 10. 版本相容性

### 10.1 API 版本

```python
import ivit

print(ivit.__version__)      # "1.0.0"
print(ivit.__api_version__)  # "1.0"
```

### 10.2 相容性保證

| API 版本 | 保證 |
|----------|------|
| 1.x | 向後相容，不破壞現有 API |
| 2.x | 可能有不相容變更 |

### 10.3 棄用警告

```python
import warnings

# 棄用的 API 會發出警告
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    model.old_method()  # 已棄用
    assert len(w) == 1
    assert "deprecated" in str(w[0].message)
```

---

## 11. 命令列工具 (CLI) (NEW)

iVIT-SDK 提供命令列工具，方便快速測試和部署。

### 11.1 安裝

```bash
pip install ivit-sdk
# 或
pip install -e .
```

### 11.2 命令列用法

```bash
# 顯示系統資訊
ivit info

# 列出可用設備
ivit devices
ivit devices --json

# 效能測試
ivit benchmark <model> [options]
ivit benchmark yolox-s.onnx -d cuda:0 -n 100

# 執行推論
ivit predict <model> <source> [options]
ivit predict yolox-s.onnx image.jpg
ivit predict yolox-s.onnx video.mp4 --save --show
```

### 11.3 命令詳細說明

#### ivit info

顯示系統和 SDK 資訊。

```bash
$ ivit info
============================================================
iVIT-SDK System Information
============================================================

SDK:
  Version:        1.0.0
  C++ Bindings:   Available

System:
  Platform:       Linux 5.15.0
  Architecture:   x86_64
  Python:         3.10.12

Backends:
  OpenVINO:       2024.0.0
  TensorRT:       8.6.1

Devices:
  [cpu] Intel Core i7-12700 (openvino)
  [cuda:0] NVIDIA GeForce RTX 3080 (tensorrt)

============================================================
```

#### ivit devices

列出可用的推論設備。

```bash
$ ivit devices

Available Devices:
------------------------------------------------------------
ID           Name                           Backend      Type
------------------------------------------------------------
cpu          Intel Core i7-12700            openvino     cpu
gpu:0        Intel UHD Graphics 770         openvino     gpu
cuda:0       NVIDIA GeForce RTX 3080        tensorrt     gpu
------------------------------------------------------------
Total: 3 device(s)

Best device (performance): cuda:0
Best device (efficiency):  cpu

$ ivit devices --json
[
  {"id": "cpu", "name": "Intel Core i7-12700", "type": "cpu", "backend": "openvino"},
  {"id": "cuda:0", "name": "NVIDIA GeForce RTX 3080", "type": "gpu", "backend": "tensorrt"}
]
```

#### ivit benchmark

執行效能基準測試。

```bash
$ ivit benchmark yolox-s.onnx -d cuda:0 -n 100 -w 10

============================================================
iVIT-SDK Benchmark
============================================================

Model:      yolox-s.onnx
Device:     cuda:0
Iterations: 100
Warmup:     10

Loading model... Done (0.45s)

Input shape: [1, 3, 640, 640]

Warming up (10 iterations)... Done

Benchmarking (100 iterations)...
  10/100 completed
  20/100 completed
  ...
  100/100 completed

----------------------------------------
Results:
----------------------------------------
  Mean:     5.23 ms
  Std:      0.42 ms
  Min:      4.85 ms
  Max:      6.21 ms
  P50:      5.18 ms
  P90:      5.65 ms
  P99:      6.15 ms
  FPS:      191.2
----------------------------------------
```

**選項**：

| 選項 | 說明 | 預設值 |
|------|------|--------|
| `-d, --device` | 目標設備 | auto |
| `-n, --iterations` | 測試次數 | 100 |
| `-w, --warmup` | 預熱次數 | 10 |

#### ivit predict

執行推論。

```bash
# 單張圖片
$ ivit predict yolox-s.onnx image.jpg

Model:  yolox-s.onnx
Source: image.jpg
Device: cuda:0

Loading model... Done

Running inference...

Inference time: 5.42 ms
Detections: 3

  [0] person: 95.23% @ [120, 50, 380, 450]
  [1] car: 87.65% @ [450, 200, 620, 350]
  [2] dog: 76.32% @ [50, 300, 200, 450]

# 影片處理
$ ivit predict yolox-s.onnx video.mp4 --save --show

Streaming inference...
----------------------------------------
Frame 1: 5 detections, 5.4ms
Frame 2: 4 detections, 5.2ms
...
Frame 120: 3 detections, 5.3ms
----------------------------------------
Total frames: 120
```

**選項**：

| 選項 | 說明 | 預設值 |
|------|------|--------|
| `-d, --device` | 目標設備 | auto |
| `--conf` | 信心閾值 | 0.5 |
| `--iou` | NMS IoU 閾值 | 0.45 |
| `--save` | 儲存結果 | False |
| `--show` | 顯示結果 | False |

### 11.4 程式化使用

CLI 命令也可以在 Python 中使用：

```python
from ivit.cli import cmd_info, cmd_devices, cmd_benchmark

# 以程式方式執行命令
import argparse
args = argparse.Namespace(json=False)
cmd_devices(args)
```

---

**文件結束**
