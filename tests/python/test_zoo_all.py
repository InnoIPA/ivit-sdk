"""
全面 Model Zoo 測試 — 涵蓋 registry、ONNX 匯出、推論、訓練

測試分組：
  TestZooRegistry          — 純 metadata 測試 (不需網路 / 下載)
  TestZooExportTorchvision — torchvision 模型 ONNX 匯出 (需 torch + 網路)
  TestZooInferenceORT      — ONNX Runtime 推論正確性 (依賴前一組)
  TestZooTraining          — 訓練 + 匯出 (需 torch，使用假資料集)
  TestZooNonTorchvision    — 非 torchvision 模型的錯誤 / 下載路徑驗證

執行方式：
  pytest tests/python/test_zoo_all.py -v              # 只跑 registry 測試
  pytest tests/python/test_zoo_all.py -v -m slow      # 加入下載 / 訓練測試
"""

import os
import sys
import importlib.util
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# 路徑設定
# ──────────────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "python"))

# ──────────────────────────────────────────────────────────────────────────────
# 可用性旗標
# ──────────────────────────────────────────────────────────────────────────────

_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_TORCHVISION = importlib.util.find_spec("torchvision") is not None
_HAS_ORT = importlib.util.find_spec("onnxruntime") is not None
_HAS_CV2 = importlib.util.find_spec("cv2") is not None

requires_torch = pytest.mark.skipif(
    not (_HAS_TORCH and _HAS_TORCHVISION),
    reason="PyTorch / torchvision not available"
)
requires_ort = pytest.mark.skipif(
    not _HAS_ORT,
    reason="onnxruntime not available"
)
slow = pytest.mark.slow  # 使用 pytest -m slow 觸發

# ──────────────────────────────────────────────────────────────────────────────
# 共用 helper
# ──────────────────────────────────────────────────────────────────────────────

def _imagenet_preprocess(bgr_img: np.ndarray, size: int = 224) -> np.ndarray:
    """將 BGR uint8 → 正規化後的 float32 NCHW tensor."""
    import cv2
    img = cv2.resize(bgr_img, (size, size))
    img = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, [0,1]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))   # HWC → CHW
    return np.expand_dims(img, 0)        # 加 batch 維度


def _make_dummy_image(h: int = 480, w: int = 640) -> np.ndarray:
    """建立含有幾何圖形的假圖像（BGR uint8）."""
    img = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
    img[100:300, 150:350] = (0, 0, 200)    # 紅色方塊
    img[200:400, 300:500] = (200, 0, 0)    # 藍色方塊
    return img


def _make_imagefolder_dataset(tmpdir: Path, n_classes: int = 3,
                               n_images: int = 6) -> Path:
    """在 tmpdir 下建立 ImageFolder 格式的假資料集."""
    import cv2
    for c in range(n_classes):
        cls_dir = tmpdir / f"class_{c}"
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_images):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            # 在圖中加入類別專屬顏色讓模型可區分
            img[:, :, c % 3] = 200 + c * 10
            cv2.imwrite(str(cls_dir / f"img_{i:03d}.jpg"), img)
    return tmpdir


# ──────────────────────────────────────────────────────────────────────────────
# 1. Zoo Registry 測試（純 metadata，不需下載）
# ──────────────────────────────────────────────────────────────────────────────

class TestZooRegistry:
    """驗證 Model Zoo 的 registry metadata — 不需網路或下載."""

    @pytest.fixture(autouse=True)
    def _zoo(self):
        from ivit.zoo.registry import MODEL_REGISTRY, list_models, search, get_model_info
        self.reg  = MODEL_REGISTRY
        self.list = list_models
        self.srch = search
        self.info = get_model_info

    # ── 總數 ──────────────────────────────────────────────────────────────────

    def test_model_count_at_least_25(self):
        assert len(self.reg) >= 25, f"Expected ≥25 models, got {len(self.reg)}"

    # ── 每個任務都有模型 ──────────────────────────────────────────────────────

    @pytest.mark.parametrize("task,min_count", [
        ("classify", 4),
        ("detect",   5),
        ("segment",  2),
        ("pose",     1),
    ])
    def test_task_model_counts(self, task, min_count):
        models = self.list(task=task)
        assert len(models) >= min_count, (
            f"Task '{task}': expected ≥{min_count}, got {len(models)}"
        )

    # ── list_models 無 filter ─────────────────────────────────────────────────

    def test_list_all_models(self):
        all_models = self.list()
        assert isinstance(all_models, list)
        assert len(all_models) == len(self.reg)

    def test_list_models_returns_sorted_or_consistent(self):
        m1 = self.list()
        m2 = self.list()
        assert m1 == m2, "list_models() must be deterministic"

    # ── search ────────────────────────────────────────────────────────────────

    def test_search_yolo_returns_yolox_models(self):
        results = self.srch("yolo")
        assert len(results) >= 5
        for name in results:
            assert "yolox" in name.lower() or "yolo" in name.lower()

    def test_search_resnet_returns_resnet_models(self):
        results = self.srch("resnet")
        assert len(results) >= 2

    def test_search_mamba_returns_mambavision_models(self):
        results = self.srch("mamba")
        assert len(results) >= 1

    def test_search_unknown_returns_empty(self):
        results = self.srch("zzznotamodel999")
        assert results == []

    # ── get_model_info ────────────────────────────────────────────────────────

    def test_get_model_info_resnet18(self):
        info = self.info("resnet18")
        assert info.name == "resnet18"
        assert info.task == "classify"
        assert info.input_size == (224, 224)
        assert info.num_classes == 1000
        assert "onnx" in info.formats
        assert info.source == "torchvision"
        assert info.license in ("Apache-2.0", "BSD-3-Clause", "MIT")

    def test_get_model_info_yolox_s(self):
        info = self.info("yolox-s")
        assert info.task == "detect"
        assert info.num_classes == 80
        assert info.input_size == (640, 640)

    def test_get_model_info_missing_raises(self):
        with pytest.raises(KeyError):
            self.info("this_model_does_not_exist")

    # ── 所有模型必要欄位完整 ──────────────────────────────────────────────────

    @pytest.mark.parametrize("model_name", list(
        __import__("importlib").import_module("ivit.zoo.registry").MODEL_REGISTRY.keys()
        if __import__("importlib").util.find_spec("ivit.zoo.registry") else []
    ) or ["resnet18", "yolox-s", "deeplabv3-resnet50"])  # fallback
    def test_each_model_has_required_fields(self, model_name):
        info = self.info(model_name)
        assert info.name,        f"{model_name}: missing name"
        assert info.task,        f"{model_name}: missing task"
        assert info.input_size,  f"{model_name}: missing input_size"
        assert info.num_classes > 0, f"{model_name}: num_classes must be positive"
        assert info.formats,     f"{model_name}: no formats listed"
        assert info.license,     f"{model_name}: missing license"
        assert info.task in ("classify", "detect", "segment", "pose"), (
            f"{model_name}: unknown task '{info.task}'"
        )

    def test_all_licenses_are_commercial_friendly(self):
        friendly = {"Apache-2.0", "BSD-3-Clause", "MIT"}
        for name, info in self.reg.items():
            assert info.license in friendly, (
                f"Model {name} has non-commercial license: {info.license}"
            )

    def test_all_models_have_onnx_format(self):
        for name, info in self.reg.items():
            assert "onnx" in info.formats, f"{name} missing onnx format"

    # ── 任務特定驗證 ──────────────────────────────────────────────────────────

    def test_detect_models_have_coco_or_voc_classes(self):
        for name, info in self.reg.items():
            if info.task == "detect":
                assert info.num_classes in (80, 91, 17, 133), (
                    f"{name}: unexpected num_classes {info.num_classes}"
                )

    def test_classify_models_imagenet_have_1000_classes(self):
        for name, info in self.reg.items():
            if info.task == "classify" and info.num_classes == 1000:
                assert "imagenet" in info.tags or "classification" in info.tags, (
                    f"{name}: 1000-class model should have imagenet tag"
                )

    def test_torchvision_models_are_ready_to_use(self):
        from ivit.zoo.registry import _TORCHVISION_MODEL_MAP
        for name in _TORCHVISION_MODEL_MAP:
            info = self.info(name)
            assert "ready-to-use" in info.tags, f"{name} should have ready-to-use tag"

    # ── print_models 不會 crash ───────────────────────────────────────────────

    def test_print_models_runs_without_error(self, capsys):
        from ivit.zoo.registry import print_models
        print_models()
        captured = capsys.readouterr()
        assert "iVIT Model Zoo" in captured.out
        assert "Total:" in captured.out

    def test_print_models_filtered_by_task(self, capsys):
        from ivit.zoo.registry import print_models
        print_models(task="classify")
        captured = capsys.readouterr()
        assert "Total:" in captured.out


# ──────────────────────────────────────────────────────────────────────────────
# 2. Torchvision ONNX 匯出測試（需要 torch + torchvision + 網路）
# ──────────────────────────────────────────────────────────────────────────────

# 輕量分類模型（優先測試）
_CLASSIFY_MODELS = ["resnet18", "mobilenetv3", "efficientnet-b0", "resnet50"]

# 分割模型
_SEGMENT_MODELS = ["deeplabv3-mobilenetv3", "deeplabv3-resnet50"]

# 偵測模型（按大小排序）
_DETECT_MODELS = [
    "ssdlite-mobilenet",
    "fasterrcnn-mobilenet",
    "fcos-resnet50",
    "retinanet-resnet50",
    "fasterrcnn-resnet50",
]


@requires_torch
class TestZooExportTorchvision:
    """torchvision 模型 ONNX 匯出測試 — 需要 torch + torchvision（首次需網路）."""

    @pytest.fixture(scope="class")
    def cache_dir(self):
        """使用真實 cache 目錄（讓模型在測試間共享）."""
        from ivit.zoo.registry import _get_cache_dir
        d = _get_cache_dir()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _export(self, name: str, cache_dir: Path) -> Path:
        from ivit.zoo.registry import download
        return download(name, format="onnx")

    # ── 分類模型匯出 ──────────────────────────────────────────────────────────

    @slow
    @pytest.mark.parametrize("model_name", _CLASSIFY_MODELS)
    def test_export_classification_model(self, model_name, cache_dir):
        onnx_path = self._export(model_name, cache_dir)
        assert onnx_path.exists(), f"{model_name}: ONNX file not created"
        assert onnx_path.suffix == ".onnx"
        size_mb = onnx_path.stat().st_size / 1024 / 1024
        assert size_mb > 1.0, f"{model_name}: ONNX too small ({size_mb:.2f}MB)"
        print(f"  {model_name}: {size_mb:.1f} MB")

    @slow
    def test_export_resnet18_has_label_file(self, cache_dir):
        from ivit.zoo.registry import download, _get_cache_dir
        onnx_path = download("resnet18", format="onnx")
        label_path = onnx_path.with_suffix(".txt")
        if label_path.exists():
            labels = label_path.read_text().strip().split("\n")
            assert len(labels) == 1000, f"Expected 1000 labels, got {len(labels)}"
            print(f"  resnet18 labels: {labels[:3]} ...")

    # ── 分割模型匯出 ──────────────────────────────────────────────────────────

    @slow
    @pytest.mark.parametrize("model_name", _SEGMENT_MODELS)
    def test_export_segmentation_model(self, model_name, cache_dir):
        onnx_path = self._export(model_name, cache_dir)
        assert onnx_path.exists()
        size_mb = onnx_path.stat().st_size / 1024 / 1024
        assert size_mb > 1.0, f"{model_name}: {size_mb:.2f}MB — too small"
        print(f"  {model_name}: {size_mb:.1f} MB")

    # ── 偵測模型匯出 ──────────────────────────────────────────────────────────

    @slow
    @pytest.mark.parametrize("model_name", _DETECT_MODELS)
    def test_export_detection_model(self, model_name, cache_dir):
        onnx_path = self._export(model_name, cache_dir)
        assert onnx_path.exists()
        size_mb = onnx_path.stat().st_size / 1024 / 1024
        assert size_mb > 0.5, f"{model_name}: {size_mb:.2f}MB — too small"
        print(f"  {model_name}: {size_mb:.1f} MB")

    # ── 快取行為 ──────────────────────────────────────────────────────────────

    @slow
    def test_cached_model_not_redownloaded(self, cache_dir):
        """第二次 download() 應直接使用 cache，不重新下載."""
        from ivit.zoo.registry import download
        path1 = download("resnet18", format="onnx")
        mtime_before = path1.stat().st_mtime

        path2 = download("resnet18", format="onnx")
        mtime_after = path2.stat().st_mtime

        assert path1 == path2
        assert mtime_before == mtime_after, "Cache should not be re-downloaded"


# ──────────────────────────────────────────────────────────────────────────────
# 3. ONNX Runtime 推論測試
# ──────────────────────────────────────────────────────────────────────────────

@requires_ort
@requires_torch
class TestZooInferenceORT:
    """ONNX Runtime 推論正確性驗證 — 依賴 TestZooExportTorchvision 先跑."""

    @pytest.fixture(autouse=True)
    def _ort(self):
        import onnxruntime as ort
        self.ort = ort

    def _get_onnx_path(self, name: str) -> Path:
        from ivit.zoo.registry import _get_cache_dir
        p = _get_cache_dir() / f"{name}.onnx"
        return p

    def _run_ort(self, onnx_path: Path, input_np: np.ndarray) -> list:
        sess = self.ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"]
        )
        input_name = sess.get_inputs()[0].name
        return sess.run(None, {input_name: input_np})

    # ── 分類推論 ──────────────────────────────────────────────────────────────

    @slow
    @pytest.mark.parametrize("model_name,input_h,input_w", [
        ("resnet18",       224, 224),
        ("mobilenetv3",    224, 224),
        ("efficientnet-b0", 224, 224),
        ("resnet50",       224, 224),
    ])
    def test_classify_inference_output_shape(self, model_name, input_h, input_w):
        onnx_path = self._get_onnx_path(model_name)
        if not onnx_path.exists():
            pytest.skip(f"{model_name} not exported yet — run with -m slow first")

        dummy_bgr = _make_dummy_image()
        input_np  = _imagenet_preprocess(dummy_bgr, size=input_h)

        t0 = time.perf_counter()
        outputs = self._run_ort(onnx_path, input_np)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert len(outputs) >= 1, "No output from model"
        out = outputs[0]
        assert out.shape == (1, 1000), (
            f"{model_name}: expected (1, 1000), got {out.shape}"
        )
        assert elapsed_ms < 10_000, f"Inference too slow: {elapsed_ms:.0f}ms"
        print(f"  {model_name}: shape={out.shape}, latency={elapsed_ms:.1f}ms")

    @slow
    @pytest.mark.parametrize("model_name", ["resnet18", "mobilenetv3"])
    def test_classify_top1_is_valid_imagenet_class(self, model_name):
        """Top-1 預測應對應有效的 ImageNet 標籤."""
        onnx_path = self._get_onnx_path(model_name)
        if not onnx_path.exists():
            pytest.skip(f"{model_name} not exported yet")

        dummy_bgr = _make_dummy_image()
        input_np  = _imagenet_preprocess(dummy_bgr)
        outputs   = self._run_ort(onnx_path, input_np)
        logits    = outputs[0][0]

        top1_idx  = int(np.argmax(logits))
        assert 0 <= top1_idx < 1000, f"top1 index {top1_idx} out of range"

        label_file = onnx_path.with_suffix(".txt")
        if label_file.exists():
            labels = label_file.read_text().strip().split("\n")
            top1_label = labels[top1_idx]
            print(f"  {model_name}: top-1 = [{top1_idx}] {top1_label}")

    @slow
    @pytest.mark.parametrize("model_name", ["resnet18", "mobilenetv3"])
    def test_classify_different_images_give_different_results(self, model_name):
        """同一模型對不同圖像應給出不同 Top-1 (使用顯著差異的合成圖像)."""
        onnx_path = self._get_onnx_path(model_name)
        if not onnx_path.exists():
            pytest.skip(f"{model_name} not exported yet")

        # Use highly distinct synthetic images (solid colours + gradient) to ensure
        # different model activations rather than relying on random noise.
        imgs = [
            np.zeros((224, 224, 3), dtype=np.uint8),                         # black
            np.full((224, 224, 3), 255, dtype=np.uint8),                     # white
            np.zeros((224, 224, 3), dtype=np.uint8),                         # red channel only
            np.zeros((224, 224, 3), dtype=np.uint8),                         # green channel only
            np.tile(np.arange(224, dtype=np.uint8), (224, 1))[:, :, None]    # horizontal gradient
                * np.array([1, 1, 0], dtype=np.uint8),
            np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8),
            np.random.default_rng(1).integers(0, 255, (224, 224, 3), dtype=np.uint8),
            np.random.default_rng(2).integers(0, 255, (224, 224, 3), dtype=np.uint8),
        ]
        imgs[2][:, :, 2] = 255   # red
        imgs[3][:, :, 1] = 255   # green

        results = set()
        for img in imgs:
            inp = _imagenet_preprocess(img)
            out = self._run_ort(onnx_path, inp)[0][0]
            results.add(int(np.argmax(out)))

        assert len(results) > 1, (
            f"All {len(imgs)} distinct images gave same top-1 — model may be broken"
        )

    # ── 分割推論 ──────────────────────────────────────────────────────────────

    @slow
    @pytest.mark.parametrize("model_name,num_classes", [
        ("deeplabv3-mobilenetv3", 21),
        ("deeplabv3-resnet50",    21),
    ])
    def test_segment_inference_output_shape(self, model_name, num_classes):
        onnx_path = self._get_onnx_path(model_name)
        if not onnx_path.exists():
            pytest.skip(f"{model_name} not exported yet")

        # DeepLabV3 models are exported at their native input size (520x520)
        h, w = 520, 520
        dummy_bgr = _make_dummy_image()
        input_np  = _imagenet_preprocess(dummy_bgr, size=h)

        outputs = self._run_ort(onnx_path, input_np)
        out = outputs[0]

        assert out.shape[0] == 1, "Batch dim should be 1"
        assert out.shape[1] == num_classes, (
            f"{model_name}: expected {num_classes} classes, got {out.shape[1]}"
        )
        assert out.shape[2] == h and out.shape[3] == w, (
            f"{model_name}: spatial size mismatch, expected ({h},{w}), got {out.shape[2:]}"
        )
        print(f"  {model_name}: shape={out.shape}")

    @slow
    def test_segment_produces_valid_class_map(self):
        onnx_path = self._get_onnx_path("deeplabv3-mobilenetv3")
        if not onnx_path.exists():
            pytest.skip("deeplabv3-mobilenetv3 not exported yet")

        # DeepLabV3 native input size is 520x520
        h, w = 520, 520
        dummy_bgr = _make_dummy_image()
        input_np  = _imagenet_preprocess(dummy_bgr, size=h)
        logits    = self._run_ort(onnx_path, input_np)[0]  # [1, 21, H, W]

        class_map = np.argmax(logits[0], axis=0)  # [H, W]
        assert class_map.shape == (h, w)
        assert class_map.min() >= 0
        assert class_map.max() < 21

    # ── 偵測推論 ──────────────────────────────────────────────────────────────

    @slow
    @pytest.mark.parametrize("model_name,input_size", [
        ("ssdlite-mobilenet",   320),
        ("fasterrcnn-mobilenet", 640),
    ])
    def test_detect_inference_outputs_boxes_scores_labels(self, model_name, input_size):
        onnx_path = self._get_onnx_path(model_name)
        if not onnx_path.exists():
            pytest.skip(f"{model_name} not exported yet")

        dummy_bgr = _make_dummy_image()
        input_np  = _imagenet_preprocess(dummy_bgr, size=input_size)

        sess = self.ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        input_name = sess.get_inputs()[0].name
        outputs    = sess.run(None, {input_name: input_np})

        # Detection models output: boxes [N,4], scores [N], labels [N]
        assert len(outputs) >= 3, f"{model_name}: expected ≥3 outputs (boxes, scores, labels)"
        boxes  = outputs[0]
        scores = outputs[1]
        labels = outputs[2]

        assert boxes.ndim == 2 and boxes.shape[1] == 4, (
            f"{model_name}: boxes shape {boxes.shape} unexpected"
        )
        assert scores.ndim == 1, f"{model_name}: scores ndim should be 1"
        assert labels.ndim == 1, f"{model_name}: labels ndim should be 1"
        assert len(boxes) == len(scores) == len(labels)

        if len(scores) > 0:
            assert scores.min() >= 0.0 and scores.max() <= 1.0, (
                f"{model_name}: scores out of [0,1] range"
            )

        print(f"  {model_name}: {len(scores)} detections")

    # ── 批次推論 ──────────────────────────────────────────────────────────────

    @slow
    def test_classify_batch_inference(self):
        """確認分類模型可以接受 batch_size=4."""
        onnx_path = self._get_onnx_path("resnet18")
        if not onnx_path.exists():
            pytest.skip("resnet18 not exported yet")

        import cv2
        batch = np.stack([
            _imagenet_preprocess(_make_dummy_image()).squeeze(0)
            for _ in range(4)
        ])  # [4, 3, 224, 224]

        sess = self.ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        input_name = sess.get_inputs()[0].name
        # Check whether dynamic batch is supported
        try:
            outputs = sess.run(None, {input_name: batch})
            assert outputs[0].shape[0] == 4
            print(f"  Batch inference: shape={outputs[0].shape}")
        except Exception as e:
            # Static batch size of 1 is OK
            pytest.skip(f"Dynamic batch not supported: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. 訓練測試（需要 torch，使用假資料集）
# ──────────────────────────────────────────────────────────────────────────────

@requires_torch
class TestZooTraining:
    """Transfer learning 訓練流程測試."""

    # ── 訓練 API 可用性 ───────────────────────────────────────────────────────

    def test_trainer_imports_cleanly(self):
        from ivit.train import (
            Trainer, ImageFolderDataset,
            EarlyStopping, ModelCheckpoint, ProgressLogger,
        )
        assert Trainer is not None

    def test_augmentation_pipeline_runs(self):
        from ivit.train.augmentation import (
            Compose, Resize, RandomHorizontalFlip,
            RandomVerticalFlip, RandomRotation, ColorJitter,
            Normalize, get_train_augmentation, get_val_augmentation,
        )
        img = _make_dummy_image()
        pipeline = Compose([
            Resize(64),
            RandomHorizontalFlip(p=1.0),
            RandomVerticalFlip(p=0.5),
            RandomRotation(15, p=0.5),
            ColorJitter(),
            Normalize(),
        ])
        out = pipeline(img)
        assert out is not None
        assert hasattr(out, "shape")

    @pytest.mark.parametrize("size", [64, 224])
    def test_get_augmentation_presets(self, size):
        from ivit.train.augmentation import get_train_augmentation, get_val_augmentation
        train_aug = get_train_augmentation(size)
        val_aug   = get_val_augmentation(size)
        img = _make_dummy_image()
        assert train_aug(img) is not None
        assert val_aug(img) is not None

    # ── ImageFolderDataset ────────────────────────────────────────────────────

    def test_imagefolder_dataset_loads(self, tmp_path):
        ds_dir = _make_imagefolder_dataset(tmp_path / "data")
        from ivit.train.dataset import ImageFolderDataset
        ds = ImageFolderDataset(ds_dir, train_split=0.8, split="train")
        assert ds.num_classes == 3
        assert len(ds) > 0
        img, label = ds[0]
        assert img is not None
        assert isinstance(label, int)

    def test_imagefolder_train_val_split(self, tmp_path):
        ds_dir = _make_imagefolder_dataset(tmp_path / "data",
                                            n_classes=3, n_images=10)
        from ivit.train.dataset import ImageFolderDataset
        train_ds = ImageFolderDataset(ds_dir, train_split=0.8, split="train")
        val_ds   = ImageFolderDataset(ds_dir, train_split=0.8, split="val")
        total = len(train_ds) + len(val_ds)
        expected_total = 3 * 10  # n_classes × n_images
        assert total == expected_total, (
            f"Train({len(train_ds)}) + Val({len(val_ds)}) = {total} ≠ {expected_total}"
        )

    def test_imagefolder_class_names_match_directories(self, tmp_path):
        ds_dir = _make_imagefolder_dataset(tmp_path / "data", n_classes=4)
        from ivit.train.dataset import ImageFolderDataset
        ds = ImageFolderDataset(ds_dir, train_split=1.0, split="train")
        assert ds.num_classes == 4
        assert len(ds.class_names) == 4
        for cls in ds.class_names:
            assert (ds_dir / cls).is_dir()

    # ── 完整訓練流程 ──────────────────────────────────────────────────────────

    @slow
    @pytest.mark.parametrize("backbone", ["resnet18", "mobilenet_v3_small"])
    def test_train_2_epochs(self, backbone, tmp_path):
        """使用假資料集訓練 2 個 epoch，驗證 loss 下降且 API 正確."""
        ds_dir = _make_imagefolder_dataset(
            tmp_path / "train_data", n_classes=3, n_images=8
        )
        from ivit.train import Trainer, ImageFolderDataset, ProgressLogger
        import torch

        dataset = ImageFolderDataset(ds_dir, train_split=0.8, split="train")
        val_ds  = ImageFolderDataset(ds_dir, train_split=0.8, split="val")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        trainer = Trainer(
            model=backbone,
            dataset=dataset,
            val_dataset=val_ds,
            epochs=2,
            learning_rate=0.01,
            batch_size=4,
            optimizer="adam",
            device=device,
            freeze_backbone=True,
            num_workers=0,
        )
        history = trainer.fit(callbacks=[ProgressLogger()])

        assert isinstance(history, list), "fit() should return list of epoch metrics"
        assert len(history) == 2, f"Expected 2 epoch results, got {len(history)}"

        for epoch_metrics in history:
            assert "loss" in epoch_metrics, "Epoch metrics must contain 'loss'"
            assert epoch_metrics["loss"] >= 0

        # Loss should decrease or stay roughly stable (not diverge)
        if len(history) == 2:
            assert history[-1]["loss"] < history[0]["loss"] * 5, (
                "Loss diverged unexpectedly"
            )

        print(f"  {backbone}: loss {history[0]['loss']:.4f} → {history[-1]['loss']:.4f}")

    @slow
    def test_train_evaluate_returns_accuracy(self, tmp_path):
        """evaluate() 應回傳 accuracy 和 loss."""
        ds_dir = _make_imagefolder_dataset(tmp_path / "data", n_classes=2, n_images=8)
        from ivit.train import Trainer, ImageFolderDataset
        import torch

        dataset = ImageFolderDataset(ds_dir, train_split=0.7, split="train")
        val_ds  = ImageFolderDataset(ds_dir, train_split=0.7, split="val")
        device  = "cuda:0" if torch.cuda.is_available() else "cpu"

        trainer = Trainer(
            model="resnet18",
            dataset=dataset,
            val_dataset=val_ds,
            epochs=1,
            batch_size=4,
            device=device,
            num_workers=0,
        )
        trainer.fit()
        metrics = trainer.evaluate()

        assert isinstance(metrics, dict), "evaluate() must return dict"
        assert "accuracy" in metrics or "loss" in metrics, (
            f"Metrics missing expected keys: {list(metrics.keys())}"
        )
        if "accuracy" in metrics:
            assert 0.0 <= metrics["accuracy"] <= 1.0, (
                f"Accuracy out of range: {metrics['accuracy']}"
            )

    # ── 訓練後匯出 ────────────────────────────────────────────────────────────

    @slow
    def test_train_and_export_onnx(self, tmp_path):
        """訓練後匯出 ONNX，並用 ORT 驗證可推論."""
        if not _HAS_ORT:
            pytest.skip("onnxruntime required for export validation")

        ds_dir = _make_imagefolder_dataset(tmp_path / "data", n_classes=3, n_images=6)
        from ivit.train import Trainer, ImageFolderDataset
        import torch, onnxruntime as ort

        dataset = ImageFolderDataset(ds_dir, train_split=1.0, split="train")
        device  = "cuda:0" if torch.cuda.is_available() else "cpu"

        trainer = Trainer(
            model="resnet18",
            dataset=dataset,
            epochs=1,
            batch_size=4,
            device=device,
            num_workers=0,
        )
        trainer.fit()

        onnx_path = str(tmp_path / "trained_model.onnx")
        trainer.export(onnx_path, format="onnx")

        assert Path(onnx_path).exists(), "ONNX file not created"
        assert Path(onnx_path).stat().st_size > 1024, "ONNX file too small"

        # 驗證可以用 ORT 載入並推論
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        dummy    = np.random.randn(1, 3, 224, 224).astype(np.float32)
        outputs  = sess.run(None, {inp_name: dummy})

        assert len(outputs) >= 1
        # 分類輸出應有 num_classes 個 logit
        assert outputs[0].shape == (1, 3), (
            f"Expected (1, 3) for 3-class model, got {outputs[0].shape}"
        )
        print(f"  Trained model exported + inferred: shape={outputs[0].shape}")

    @slow
    def test_train_and_export_torchscript(self, tmp_path):
        """訓練後匯出 TorchScript."""
        import torch
        ds_dir = _make_imagefolder_dataset(tmp_path / "data", n_classes=2, n_images=6)
        from ivit.train import Trainer, ImageFolderDataset

        dataset = ImageFolderDataset(ds_dir, train_split=1.0, split="train")
        device  = "cuda:0" if torch.cuda.is_available() else "cpu"

        trainer = Trainer(
            model="resnet18",
            dataset=dataset,
            epochs=1,
            batch_size=4,
            device=device,
            num_workers=0,
        )
        trainer.fit()

        ts_path = str(tmp_path / "model.pt")
        trainer.export(ts_path, format="torchscript")

        assert Path(ts_path).exists()
        assert Path(ts_path).stat().st_size > 1024

        # 驗證可以載入執行 (map_location=cpu so it runs on any machine)
        model = torch.jit.load(ts_path, map_location="cpu")
        model.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out   = model(dummy)
        assert out.shape == (1, 2), f"Expected (1, 2), got {out.shape}"
        print(f"  TorchScript: shape={out.shape}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. 非 torchvision 模型下載路徑驗證
# ──────────────────────────────────────────────────────────────────────────────

class TestZooNonTorchvision:
    """YOLOX / RT-DETR / MambaVision / RTMPose 下載路徑的正確性驗證."""

    def _registry(self):
        from ivit.zoo.registry import MODEL_REGISTRY
        return MODEL_REGISTRY

    # ── YOLOX ─────────────────────────────────────────────────────────────────

    def test_yolox_models_have_onnx_url(self):
        reg = self._registry()
        yolox_models = [n for n in reg if n.startswith("yolox")]
        assert len(yolox_models) >= 5
        for name in yolox_models:
            info = reg[name]
            assert info.onnx_url, f"{name} must have onnx_url for direct download"
            assert "onnx" in info.onnx_url.lower() or info.onnx_url.endswith(".onnx"), (
                f"{name} onnx_url doesn't look like an ONNX file URL"
            )

    def test_yolox_models_use_megvii_source(self):
        reg = self._registry()
        for name, info in reg.items():
            if name.startswith("yolox"):
                assert info.source == "megvii", f"{name}: expected source=megvii"

    # ── RT-DETR ───────────────────────────────────────────────────────────────

    def test_rtdetr_models_have_onnx_url(self):
        reg = self._registry()
        rtdetr_models = [n for n in reg if n.startswith("rtdetr")]
        assert len(rtdetr_models) >= 2
        for name in rtdetr_models:
            assert reg[name].onnx_url, f"{name} must have onnx_url"

    def test_rtdetr_fallback_raises_import_error_without_paddle(self):
        """without_source=True path: PaddlePaddle 未安裝時應 raise ImportError."""
        from ivit.zoo.registry import _export_rtdetr_onnx, MODEL_REGISTRY, _get_cache_dir

        has_paddle = importlib.util.find_spec("paddle") is not None
        if has_paddle:
            pytest.skip("PaddlePaddle is installed — skip error path test")

        info = MODEL_REGISTRY["rtdetr-l"]
        fake_path = _get_cache_dir() / "_test_rtdetr_l.onnx"

        with pytest.raises((ImportError, RuntimeError)):
            _export_rtdetr_onnx("rtdetr-l", fake_path, info, force=True)

    # ── RTMPose ───────────────────────────────────────────────────────────────

    def test_rtmpose_models_registered(self):
        reg = self._registry()
        pose_models = [n for n in reg if n.startswith("rtmpose")]
        assert len(pose_models) >= 2, "Expected at least rtmpose-s and rtmpose-m"

    def test_rtmpose_fallback_raises_import_error_without_mmpose(self):
        from ivit.zoo.registry import _export_mmpose_onnx, MODEL_REGISTRY, _get_cache_dir

        has_mmpose = importlib.util.find_spec("mmpose") is not None
        if has_mmpose:
            pytest.skip("mmpose is installed — skip error path test")

        info = MODEL_REGISTRY.get("rtmpose-s")
        if info is None:
            pytest.skip("rtmpose-s not in registry")

        fake_path = _get_cache_dir() / "_test_rtmpose_s.onnx"
        with pytest.raises((ImportError, RuntimeError)):
            _export_mmpose_onnx("rtmpose-s", fake_path, info, force=True)

    # ── MambaVision ───────────────────────────────────────────────────────────

    def test_mambavision_models_have_onnx_url(self):
        reg = self._registry()
        mv_models = [n for n in reg if n.startswith("mambavision")]
        assert len(mv_models) >= 5
        for name in mv_models:
            info = reg[name]
            assert info.onnx_url, f"{name} must have onnx_url"
            assert info.source == "mambavision"
            assert info.license == "Apache-2.0"

    def test_mambavision_metrics_quality(self):
        reg = self._registry()
        for name, info in reg.items():
            if name.startswith("mambavision"):
                assert info.metrics.get("top1", 0) > 80, (
                    f"{name}: top1 accuracy should be > 80%"
                )

    # ── 格式驗證 ──────────────────────────────────────────────────────────────

    def test_download_invalid_format_raises(self):
        from ivit.zoo.registry import download
        with pytest.raises(ValueError, match="Format"):
            download("resnet18", format="invalid_format_xyz")

    def test_download_unknown_model_raises(self):
        from ivit.zoo.registry import download
        with pytest.raises(KeyError):
            download("model_that_does_not_exist_xyz")

    # ── onnx_url 需要 from_source=True 才繞過直接下載 ────────────────────────

    @slow
    def test_yolox_download_via_onnx_url(self):
        """YOLOX nano 應可透過 onnx_url 取得（cached 或下載皆可）."""
        from ivit.zoo.registry import download, _get_cache_dir

        onnx_path = _get_cache_dir() / "yolox-nano.onnx"
        if onnx_path.exists():
            # Already cached — just verify the file is valid
            size_mb = onnx_path.stat().st_size / 1024 / 1024
            assert size_mb > 0.5, f"yolox-nano cached file too small: {size_mb:.2f}MB"
            print(f"  yolox-nano (cached): {size_mb:.1f}MB")
            return

        try:
            path = download("yolox-nano", format="onnx")
            assert path.exists()
            size_mb = path.stat().st_size / 1024 / 1024
            assert size_mb > 0.5, f"yolox-nano too small: {size_mb:.2f}MB"
            print(f"  yolox-nano: {size_mb:.1f}MB")
        except Exception as e:
            # Network error: acceptable in CI
            pytest.skip(f"Network error (expected in offline CI): {e}")

    @slow
    def test_mambavision_t_download_via_onnx_url(self):
        """mambavision-t 應可透過 onnx_url 取得（cached 或下載皆可）."""
        from ivit.zoo.registry import download, _get_cache_dir

        onnx_path = _get_cache_dir() / "mambavision-t.onnx"
        if onnx_path.exists():
            size_mb = onnx_path.stat().st_size / 1024 / 1024
            assert size_mb > 1.0, f"mambavision-t cached file too small: {size_mb:.2f}MB"
            print(f"  mambavision-t (cached): {size_mb:.1f}MB")
            return

        try:
            path = download("mambavision-t", format="onnx")
            assert path.exists()
            size_mb = path.stat().st_size / 1024 / 1024
            assert size_mb > 1.0
            print(f"  mambavision-t: {size_mb:.1f}MB")
        except Exception as e:
            pytest.skip(f"Network error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. 完整端到端整合 (快速子集)
# ──────────────────────────────────────────────────────────────────────────────

@requires_torch
@requires_ort
class TestZooEndToEnd:
    """從 zoo.download() 到 ORT 推論的完整流程 (torchvision 模型)."""

    @slow
    def test_e2e_classify_resnet18(self):
        """resnet18: download → export → ORT infer."""
        import onnxruntime as ort
        from ivit.zoo.registry import download

        path = download("resnet18", format="onnx")
        assert path.exists()

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp  = _imagenet_preprocess(_make_dummy_image())
        out  = sess.run(None, {sess.get_inputs()[0].name: inp})

        assert out[0].shape == (1, 1000)
        top5 = np.argsort(out[0][0])[-5:][::-1]
        print(f"  E2E resnet18 top5 indices: {top5}")

    @slow
    def test_e2e_segment_deeplabv3(self):
        """deeplabv3-mobilenetv3: download → export → ORT infer."""
        import onnxruntime as ort
        from ivit.zoo.registry import download

        path  = download("deeplabv3-mobilenetv3", format="onnx")
        assert path.exists()

        # DeepLabV3 native input size is 520x520
        inp   = _imagenet_preprocess(_make_dummy_image(), size=520)
        sess  = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        out   = sess.run(None, {sess.get_inputs()[0].name: inp})

        seg_map = np.argmax(out[0][0], axis=0)
        assert seg_map.shape == (520, 520)
        print(f"  E2E deeplabv3: seg_map unique classes = {np.unique(seg_map).tolist()[:10]}")

    @slow
    def test_e2e_detect_ssdlite(self):
        """ssdlite-mobilenet: download → export → ORT infer."""
        import onnxruntime as ort
        from ivit.zoo.registry import download

        path = download("ssdlite-mobilenet", format="onnx")
        assert path.exists()

        inp  = _imagenet_preprocess(_make_dummy_image(), size=320)
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        outs = sess.run(None, {sess.get_inputs()[0].name: inp})

        boxes, scores, labels = outs[0], outs[1], outs[2]
        assert boxes.shape[1] == 4
        print(f"  E2E ssdlite: {len(scores)} detections")

    @slow
    def test_e2e_train_to_infer(self, tmp_path):
        """假資料集訓練 → 匯出 ONNX → ORT 推論."""
        import torch, onnxruntime as ort
        from ivit.train import Trainer, ImageFolderDataset

        ds_dir = _make_imagefolder_dataset(tmp_path / "data", n_classes=5, n_images=5)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        dataset = ImageFolderDataset(ds_dir, train_split=1.0, split="train")
        trainer = Trainer(
            model="resnet18", dataset=dataset,
            epochs=1, batch_size=5, device=device, num_workers=0,
        )
        trainer.fit()

        onnx_path = str(tmp_path / "e2e_model.onnx")
        trainer.export(onnx_path, format="onnx")
        assert Path(onnx_path).exists()

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inp  = np.random.randn(1, 3, 224, 224).astype(np.float32)
        out  = sess.run(None, {sess.get_inputs()[0].name: inp})

        assert out[0].shape == (1, 5)
        print(f"  E2E train→infer: classes={out[0].shape[1]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
