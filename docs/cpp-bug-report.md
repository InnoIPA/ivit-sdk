# iVIT-SDK C++ 異常錯誤分析報告

**分析日期**：2026-04-13  
**分析範圍**：`src/core/model.cpp`、`src/vision/classifier.cpp`、`src/vision/detector.cpp`、`src/vision/segmentor.cpp`

---

## 彙整總表

| 編號 | 嚴重度 | 位置 | 問題摘要 |
|------|--------|------|----------|
| BUG-1 | 🔴 CRITICAL | `model.cpp:616` | `submit_inference()` detached thread use-after-free |
| BUG-2 | 🟠 MEDIUM | `model.cpp:692` | `ensure_executor()` TOCTOU race condition |
| BUG-3 | 🟠 MEDIUM | `detector.cpp:187` / `segmentor.cpp:159` | 動態維度 -1 未替換，memcpy 大小錯誤 |
| BUG-4 | 🟡 LOW | `segmentor.cpp:24` | ADE20K 標籤不完整（20/150） |

---

## BUG-1 — CRITICAL：`submit_inference()` use-after-free

**位置**：`src/core/model.cpp:616`

### 問題描述

`submit_inference()` 建立一個 detached thread，並在 lambda 中捕獲 `this`。  
`Model::~Model()` 析構時只停止 executor queue thread，**不等待**這個 detached thread 完成。  
若 `Model` 在推論完成前被 destroy，`this->predict()` 會 dereference 已釋放的記憶體，造成 undefined behavior。

### 問題程式碼

```cpp
// model.cpp:616
std::thread([this, image_copy, callback, config]() {
    try {
        Results results = this->predict(image_copy, config);  // ← this 可能已釋放
        if (callback) callback(std::move(results));
    } catch (...) { ... }
}).detach();  // ← detach 後無法追蹤壽命
```

### 修正方案

改用已有的 `enqueue_inference()` executor queue 機制，或在 `Model` 持有所有 detached futures 並在析構函式中等待：

```cpp
// 方案 A：改用 executor queue（推薦）
void Model::submit_inference(
    const cv::Mat& image,
    std::function<void(Results)> callback,
    const InferConfig& config
) {
    ensure_executor();
    cv::Mat image_copy = image.clone();
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.push([this, image_copy, callback, config]() {
            try {
                Results results = this->predict(image_copy, config);
                if (callback) callback(std::move(results));
            } catch (const std::exception& e) {
                std::cerr << "[iVIT] Async inference error: " << e.what() << std::endl;
            }
        });
    }
    queue_cv_.notify_one();
}
```

---

## BUG-2 — MEDIUM：`ensure_executor()` TOCTOU race condition

**位置**：`src/core/model.cpp:692`

### 問題描述

`load()` 和 `store()` 不是原子的 compare-and-exchange 操作。  
兩個執行緒同時呼叫 `ensure_executor()` 時，都可能讀到 `false` 並同時建立 executor thread。  
第一個建立的 `executor_thread_` 被第二個覆蓋，造成 thread 洩漏，task queue 行為未定義。

### 問題程式碼

```cpp
// model.cpp:692
void Model::ensure_executor() {
    if (executor_running_.load()) return;   // Thread A 讀到 false
    executor_running_.store(true);           // Thread B 也讀到 false（還沒看到 A 的 store）
    executor_thread_ = std::make_unique<std::thread>(...);  // 兩個 thread 都建立
}
```

### 修正方案

使用 `compare_exchange_strong` 確保原子性：

```cpp
void Model::ensure_executor() {
    bool expected = false;
    if (!executor_running_.compare_exchange_strong(expected, true)) return;
    // 只有 CAS 成功的執行緒才能繼續
    executor_thread_ = std::make_unique<std::thread>([this]() {
        while (executor_running_.load()) {
            // ... task loop
        }
    });
}
```

---

## BUG-3 — MEDIUM：動態維度 (-1) 未替換

**位置**：`src/vision/detector.cpp:187`、`src/vision/segmentor.cpp:159`

### 問題描述

當模型的 input shape 含有動態維度（`-1`，例如動態 batch `[-1, 3, 640, 640]`），  
`Tensor` 建構子以 `-1` 計算 `byte_size()` 會得到錯誤結果（可能為 0 或負數溢出），  
後續 `std::memcpy` 複製大小錯誤，造成記憶體損壞或程式 crash。

`classifier.cpp` 已正確處理此問題，但 `detector.cpp` 和 `segmentor.cpp` 遺漏了相同修正。

### 問題程式碼

```cpp
// detector.cpp:187 / segmentor.cpp:159（兩者相同問題）
const auto& info = input_info[0];
Tensor input(info.shape, DataType::Float32);          // ❌ info.shape 可能含 -1
std::memcpy(input.data(), processed.data, input.byte_size());  // ❌ 大小計算錯誤
```

### 正確做法（參考 classifier.cpp:147）

```cpp
// classifier.cpp 的正確實作
const auto& info = input_info[0];
Shape input_shape = info.shape;
for (auto& d : input_shape) { if (d < 0) d = 1; }   // ✅ 替換動態維度為 1
Tensor input(input_shape, DataType::Float32);
input.set_name(info.name);
std::memcpy(input.data(), processed.data, input.byte_size());  // ✅ 大小正確
```

**detector.cpp 和 segmentor.cpp 需補上同樣的三行修正。**

---

## BUG-4 — LOW：ADE20K 標籤不完整

**位置**：`src/vision/segmentor.cpp:24`

### 問題描述

`ADE20K_LABELS` 靜態陣列只定義了 20 個標籤，但 ADE20K 資料集有 150 個類別。  
使用 ADE20K 語意分割模型時，class 20 以後的所有類別都回退到 `"class_XXX"` 字串。  
不影響分割精度，但標籤輸出不正確。

### 問題程式碼

```cpp
// segmentor.cpp:24
static const std::vector<std::string> ADE20K_LABELS = {
    "wall", "building", "sky", "floor", "tree",
    "ceiling", "road", "bed", "windowpane", "grass",
    "cabinet", "sidewalk", "person", "earth", "door",
    "table", "mountain", "plant", "curtain", "chair"
    // ... 150 classes total  ← 只有 comment，實際只有 20 個
};
```

### 修正方案

補齊完整的 150 個 ADE20K 標籤。可從官方資料集的 `objectInfo150.txt` 取得完整清單，  
或改為在執行時從 `.txt` 標籤檔讀取（更靈活，避免 hardcode）。

---

## 無問題確認項目

| 項目 | 位置 | 結論 |
|------|------|------|
| `predict_concurrent()` 引用捕獲 | `model.cpp:641` | ✅ 安全（所有 future 在 local vars 超出 scope 前都已 `.get()`）|
| `RuntimeConfig` vs `LoadConfig` 型別 | `model.cpp:1019` | ✅ 正確（有意轉換，`IRuntime::load_model` 接受 `RuntimeConfig`）|
| Classifier 動態維度處理 | `classifier.cpp:147` | ✅ 正確（有 `-1 → 1` 替換）|
| TRT intentional leak | `tensorrt_runtime.cpp` | ✅ 有意設計（避免 sm_120+ 析構 segfault）|
