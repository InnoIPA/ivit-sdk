"""
YOLOX (Megvii, Apache-2.0) 訓練 backend。

YOLOX 採用 `Exp` 系統管理超參數與資料集設定。本 backend：
1. 動態產生一個 Exp 子類，注入 num_classes、data_dir、annotation 路徑、input_size
2. 用 yolox.core.Trainer 跑訓練
3. 將其 logger 結果折算成 epoch metric dict 給 iVIT callback 使用（best-effort）
4. 輸出 ckpt 後可走 yolox 的 export utility 轉 ONNX

需要使用者自行準備 COCO 格式的資料集（images/、annotations/instances_train2017.json 等）。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# 對應官方 yolox.exp.default 中的 Exp 子類（這些子類覆寫 get_model() 加上
# depthwise / 特殊 augmentation 等變體專屬設定，與 Megvii 官方 .pth 的權重命名一致）
_YOLOX_OFFICIAL_EXP_MODULES = {
    "yolox-nano": "yolox.exp.default.yolox_nano",
    "yolox-tiny": "yolox.exp.default.yolox_tiny",
    "yolox-s": "yolox.exp.default.yolox_s",
    "yolox-m": "yolox.exp.default.yolox_m",
    "yolox-l": "yolox.exp.default.yolox_l",
}


class YoloxBackend:
    """
    動態建立 yolox.exp.Exp 子類並驅動 yolox.core.Trainer。

    Args:
        model: "yolox-nano" / "yolox-tiny" / "yolox-s" / "yolox-m" / "yolox-l"
        data_dir: COCO root（含 images/ 與 annotations/）
        train_ann: 訓練 annotation 檔名（在 data_dir/annotations/ 內）
        val_ann: 驗證 annotation 檔名
        num_classes: 自訂類別數
        epochs / imgsz / batch_size / device / output_dir / pretrained_weights
    """

    def __init__(
        self,
        model: str,
        data_dir: str,
        train_ann: str = "instances_train2017.json",
        val_ann: str = "instances_val2017.json",
        num_classes: int = 80,
        epochs: int = 100,
        imgsz: int = 640,
        batch_size: int = 8,
        device: str = "cuda:0",
        output_dir: str = "YOLOX_outputs",
        pretrained_weights: Optional[str] = None,
        **exp_overrides,
    ):
        try:
            import yolox  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "YOLOX training needs the yolox package. Its PyPI release (0.3.0) does not "
                "build with current setuptools; put the YOLOX source tree on PYTHONPATH "
                "instead (e.g. unpack `pip download --no-deps --no-binary :all: yolox`)."
            ) from e

        if model not in _YOLOX_OFFICIAL_EXP_MODULES:
            raise ValueError(
                f"Unknown YOLOX model: {model}. "
                f"Available: {list(_YOLOX_OFFICIAL_EXP_MODULES.keys())}"
            )

        self.model_name = model
        self.data_dir = str(Path(data_dir).resolve())
        self.train_ann = train_ann
        self.val_ann = val_ann
        self.num_classes = num_classes
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.device = device
        self.output_dir = str(Path(output_dir).resolve())
        self.pretrained_weights = pretrained_weights
        self.exp_overrides = exp_overrides

        self._exp = self._build_exp()
        self._best_ckpt: Optional[str] = None

    def _build_exp(self) -> Any:
        """動態建立 yolox Exp 子類，繼承官方對應變體（保留 depthwise / 特殊 aug 設定）。"""
        import importlib

        module_name = _YOLOX_OFFICIAL_EXP_MODULES[self.model_name]
        official = importlib.import_module(module_name).Exp

        backend = self

        class _IvitExp(official):  # type: ignore[misc, valid-type]
            def __init__(self):
                super().__init__()
                # 只覆寫 dataset / 訓練超參，不動 depth / width / depthwise / aug 等架構相關欄位
                self.num_classes = backend.num_classes
                self.data_dir = backend.data_dir
                self.train_ann = backend.train_ann
                self.val_ann = backend.val_ann
                self.input_size = (backend.imgsz, backend.imgsz)
                self.test_size = (backend.imgsz, backend.imgsz)
                self.max_epoch = backend.epochs
                self.exp_name = backend.model_name
                self.output_dir = backend.output_dir
                # 套用使用者覆寫
                for k, v in backend.exp_overrides.items():
                    setattr(self, k, v)

        return _IvitExp()

    def fit(self) -> Dict[str, Any]:
        """跑 yolox.core.Trainer.train()。"""
        from yolox.core import Trainer

        # 沒指定 pretrained 時，從 ivit Model Zoo registry 自動抓 COCO 預訓練 .pth
        # （從零訓 yolox-nano 在小資料集上 100 epoch 也只能到 AP 0.01，沒實用價值）
        ckpt_path = self.pretrained_weights or self._auto_pretrained()

        # yolox Trainer 期待一個 argparse Namespace 風格的 args 物件
        args = _make_yolox_args(
            batch_size=self.batch_size,
            devices=self._device_count(),
            ckpt=ckpt_path,
            experiment_name=self.model_name,
        )
        trainer = Trainer(self._exp, args)
        trainer.train()

        # 優先 best_ckpt.pth，沒有的話（如 epoch 太短 best AP 沒被更新）退回 latest / last
        run_dir = Path(self.output_dir) / self.model_name
        for fname in ("best_ckpt.pth", "latest_ckpt.pth", "last_epoch_ckpt.pth"):
            candidate = run_dir / fname
            if candidate.exists():
                self._best_ckpt = str(candidate)
                break
        else:
            self._best_ckpt = None
        return {"best_ckpt": self._best_ckpt, "run_dir": str(run_dir)}

    def evaluate(
        self,
        ckpt: Optional[str] = None,
        conf_thres: float = 0.001,
        nms_thres: float = 0.65,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        對訓練後的 yolox 模型跑 COCO mAP 評估（用 yolox 原生 `COCOEvaluator`）。

        與 `YoloxPredictor.evaluate()` 的差別：
            - **這條路徑**：載入 .pth → 純 PyTorch forward → yolox 自家 COCOEvaluator
              （數字會與 `train_log.txt` 最後幾行的 yolox 自評完全一致）
            - YoloxPredictor.evaluate(): ONNX → TRT/OpenVINO C++ runtime → 我們的後處理
              （驗證部署管線；可能因 FP32→FP16 / NMS 配置差異有 ±1% 誤差）

        Args:
            ckpt: 要評估的 .pth 路徑；None 時用 `self._best_ckpt`（fit() 結束後自動填）
            conf_thres: 偵測 confidence 門檻（COCO eval 預設極低）
            nms_thres: NMS IoU 門檻
            verbose: 是否印出 pycocotools summary

        Returns:
            {
                "mAP": ...,          # AP @ IoU=0.50:0.95
                "mAP_50": ...,       # AP @ IoU=0.50
                "mAP_75": ...,       # AP @ IoU=0.75
                "AR_100": ...,       # AR @ maxDets=100
                "ckpt": "...",
            }
        """
        import io
        import contextlib

        import torch

        ckpt_path = ckpt or self._best_ckpt
        if not ckpt_path:
            raise RuntimeError(
                "No checkpoint available — call fit() first or pass ckpt=..."
            )

        if verbose:
            print(f"[yolox.evaluate] loading ckpt: {ckpt_path}")

        # 用我們的 Exp 子類拿 model + 載權重
        model = self._exp.get_model()
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state)

        if self.device.startswith("cuda") and torch.cuda.is_available():
            model = model.cuda().eval()
        else:
            model = model.cpu().eval()

        # 暫時覆寫 Exp 的 test_conf / nmsthre，讓 evaluator 拿到指定門檻
        orig_test_conf = getattr(self._exp, "test_conf", 0.01)
        orig_nmsthre = getattr(self._exp, "nmsthre", 0.65)
        self._exp.test_conf = conf_thres
        self._exp.nmsthre = nms_thres

        # 拿 yolox 自家 evaluator（內部會建 val DataLoader，跑 forward，餵 pycocotools）
        # get_evaluator(batch_size, is_distributed, testdev=False, legacy=False)
        evaluator = self._exp.get_evaluator(
            batch_size=self.batch_size,
            is_distributed=False,
            testdev=False,
            legacy=False,
        )

        if verbose:
            print(
                f"[yolox.evaluate] running COCO eval "
                f"(conf={conf_thres}, iou={nms_thres}) ..."
            )

        # evaluator.evaluate() 回傳 (ap50_95, ap50, summary_str)
        # 印 summary 到 stdout 是 pycocotools 內部做的，verbose=False 時吞掉
        if verbose:
            ap50_95, ap50, summary = evaluator.evaluate(
                model,
                distributed=False,
                half=False,
            )
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                ap50_95, ap50, summary = evaluator.evaluate(
                    model,
                    distributed=False,
                    half=False,
                )

        # 還原 Exp 設定
        self._exp.test_conf = orig_test_conf
        self._exp.nmsthre = orig_nmsthre

        # summary 字串內含 12 行 pycocotools stats，從中抽 mAP_75 與 AR_100
        # 格式類似：
        #   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.570
        mAP_75 = self._parse_stat_line(
            summary, "IoU=0.75", "area=   all", "maxDets=100"
        )
        AR_100 = self._parse_stat_line(
            summary, "IoU=0.50:0.95", "area=   all", "maxDets=100", metric="(AR)"
        )

        return {
            "mAP": float(ap50_95),
            "mAP_50": float(ap50),
            "mAP_75": float(mAP_75) if mAP_75 is not None else float("nan"),
            "AR_100": float(AR_100) if AR_100 is not None else float("nan"),
            "ckpt": ckpt_path,
        }

    @staticmethod
    def _parse_stat_line(
        summary: str,
        iou_tag: str,
        area_tag: str,
        max_dets_tag: str,
        metric: str = "(AP)",
    ) -> Optional[float]:
        """從 pycocotools summary 字串裡撿出特定一行的數值。"""
        for line in summary.splitlines():
            if (
                metric in line
                and iou_tag in line
                and area_tag in line
                and max_dets_tag in line
            ):
                # 行末 "= 0.570" 取數字
                try:
                    return float(line.rsplit("=", 1)[1].strip())
                except (IndexError, ValueError):
                    return None
        return None

    def _auto_pretrained(self) -> Optional[str]:
        """從 ivit Model Zoo registry 取對應 yolox 模型的 COCO 預訓練 .pth。

        快取於 ~/.cache/ivit/pretrained/<model>.pth；若 registry 沒登錄或下載失敗回 None。
        """
        try:
            from ivit.zoo.registry import MODEL_REGISTRY
        except Exception as e:
            logger.warning(f"Cannot access ivit zoo registry: {e}")
            return None

        info = MODEL_REGISTRY.get(self.model_name)
        if not info or not getattr(info, "url", None):
            logger.warning(
                f"No pretrained URL registered for {self.model_name}; "
                "training will start from random init (likely won't converge on small data)."
            )
            return None

        cache_dir = Path.home() / ".cache" / "ivit" / "pretrained"
        cache_dir.mkdir(parents=True, exist_ok=True)
        local = cache_dir / f"{self.model_name}_coco.pth"

        if local.exists():
            logger.info(f"Using cached pretrained weights: {local}")
            return str(local)

        from urllib.request import urlretrieve

        logger.info(f"Downloading pretrained weights from {info.url}")
        try:
            urlretrieve(info.url, local)
            logger.info(f"Cached to {local}")
            return str(local)
        except Exception as e:
            logger.warning(f"Failed to download pretrained: {e}")
            return None

    def _device_count(self) -> int:
        if self.device == "cpu":
            return 0
        try:
            import torch

            return torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            return 0

    def export(self, path: str, format: str = "onnx", **kwargs) -> str:
        """把訓練好的 ckpt export 成 ONNX（呼叫 yolox 內建的 export 邏輯）。"""
        if format.lower() != "onnx":
            raise NotImplementedError(
                "YOLOX backend only supports ONNX export directly; "
                "convert ONNX to other formats via ModelExporter."
            )
        if not self._best_ckpt:
            raise RuntimeError("No checkpoint found — call fit() first.")

        import torch

        model = self._exp.get_model()
        ckpt = torch.load(self._best_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        # 訓練後 self._exp.model 可能還掛在 GPU 上，搬回 CPU 讓 ONNX export 純 CPU trace
        model = model.cpu().eval()
        # 與 iVIT C++ Detector::postprocess_yolo（src/vision/detector.cpp:391-446）對齊：
        # C++ 端會自行做 grid decode（cx+gx）*stride、exp(w)*stride 等，所以 ONNX 必須輸出
        # **raw grid-relative** 預測（decode_in_inference=False，與 Megvii 官方 export_onnx.py 一致）。
        # 若你的下游 post-processor 期待 pixel-decoded 輸出，可傳 kwargs decode_in_inference=True。
        if hasattr(model, "head"):
            model.head.decode_in_inference = bool(
                kwargs.pop("decode_in_inference", True)
            )

        dummy = torch.zeros(1, 3, self.imgsz, self.imgsz)
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # 對齊 iVIT Model Zoo 預建 ONNX（C++ post-processor 預期）：
        # - input name "images"（不是 "input"）
        # - 固定 batch=1、固定 num_dets，shape 都是 static
        # - opset 11
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            input_names=["images"],
            output_names=["output"],
            opset_version=kwargs.get("opset_version", 11),
            do_constant_folding=True,
            dynamo=False,
        )
        logger.info(f"YOLOX ONNX export complete: {out_path}")
        return str(out_path)

    @property
    def model(self) -> Any:
        """回傳底層 PyTorch model（exp.get_model()）。"""
        return self._exp.get_model()


def _make_yolox_args(**fields: Any) -> Any:
    """建立 yolox.core.Trainer 期待的 args namespace。"""
    import argparse

    defaults = {
        "experiment_name": None,
        "name": None,
        "dist_backend": "nccl",
        "dist_url": None,
        "batch_size": 8,
        "devices": 1,
        "exp_file": None,
        "resume": False,
        "ckpt": None,
        "start_epoch": None,
        "num_machines": 1,
        "machine_rank": 0,
        "fp16": False,
        # yolox 內部 assert `cache_img is None`，必須給 None 而不是 False
        "cache": None,
        "occupy": False,
        "logger": "tensorboard",
        "opts": [],
    }
    defaults.update(fields)
    return argparse.Namespace(**defaults)
