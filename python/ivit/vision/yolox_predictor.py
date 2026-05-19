"""
YOLOX-aware predictor 包裝 `ivit.load(...).infer()` 的 raw tensor 介面。

設計動機：iVIT C++ 層的 `Model::predict` 走的是 model-agnostic 的簡單 YOLO 後處理
（src/core/model.cpp:360 `postprocess_detection`），它假設 cx/cy 是 normalized [0,1]，
完全不認 YOLOX 的 grid-relative output。`Detector::predict`（src/vision/detector.cpp）
雖然有 yolox-aware 邏輯但沒被 `Model::predict` dispatch 到。

在 C++ 層修好之前，這個 Python wrapper 提供正確的 yolox preprocess + decode + NMS：
- preprocess：letterbox 416×416、BGR、不 normalize（與 Megvii ValTransform 對齊）
- postprocess：grid-relative decode（cx+gx）×stride、exp(w)×stride，再 obj×cls + NMS
- 座標轉回原圖（移除 letterbox padding）
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

# COCO80 預設類別名稱（與 Megvii YOLOX 預訓練對齊）
_COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


@dataclass
class YoloxDetection:
    """單一偵測結果。"""
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2 in original image pixels
    class_id: int
    label: str
    confidence: float


@dataclass
class EvalReport:
    """COCO-style evaluation 主要指標。"""
    mAP: float            # AP @ IoU=0.50:0.95（COCO 主指標）
    mAP_50: float         # AP @ IoU=0.50（PASCAL-style）
    mAP_75: float         # AP @ IoU=0.75（嚴格定位）
    AR_100: float         # AR @ maxDets=100
    per_class_AP: Dict[str, float]   # {label: AP@0.5:0.95}
    num_images: int
    num_detections: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mAP": self.mAP,
            "mAP_50": self.mAP_50,
            "mAP_75": self.mAP_75,
            "AR_100": self.AR_100,
            "per_class_AP": dict(self.per_class_AP),
            "num_images": self.num_images,
            "num_detections": self.num_detections,
        }


class YoloxPredictor:
    """
    YOLOX 偵測器 wrapper。

    Args:
        model_path: ONNX path（會用 `ivit.load` 載入；自動走 TensorRT / OpenVINO 等後端）
        device: 推論裝置
        labels: 類別名稱（None → 嘗試讀 model_path.with_suffix('.txt')，再不行用 COCO80）
        input_size: 訓練時的 letterbox 解析度（必須與 ONNX 對齊）

    Examples:
        >>> from ivit.vision.yolox_predictor import YoloxPredictor
        >>> p = YoloxPredictor("runs/yolox-nano/yolox-nano.onnx", device="cuda:0")
        >>> dets = p.predict("bus.jpg", conf=0.3)
        >>> for d in dets:
        ...     print(d.label, d.confidence, d.bbox)
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cuda:0",
        labels: Optional[Sequence[str]] = None,
        input_size: int = 416,
    ):
        import ivit  # lazy: 避免在沒裝 C++ binding 時 import 失敗

        self.model_path = str(model_path)
        self.input_size = (input_size, input_size)
        self._model = ivit.load(self.model_path, device=device)
        self.labels = list(labels) if labels else self._auto_labels(self.model_path)

        # 從 ONNX I/O 取得實際 input name 與 input H/W（覆寫上面的 default）
        if self._model.input_info:
            info = self._model.input_info[0]
            self._input_name = info.name
            sh = list(info.shape)
            if len(sh) == 4 and sh[2] > 0 and sh[3] > 0:
                self.input_size = (int(sh[2]), int(sh[3]))
        else:
            self._input_name = "images"

    @staticmethod
    def _auto_labels(model_path: str) -> List[str]:
        labels_path = Path(model_path).with_suffix(".txt")
        if labels_path.exists():
            return [l.strip() for l in labels_path.read_text().splitlines() if l.strip()]
        return list(_COCO80)

    # ------------------------------------------------------------------
    # 主要 API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        annotation_file: Union[str, Path],
        images_dir: Union[str, Path],
        conf: float = 0.001,
        iou: float = 0.65,
        max_detections: int = 300,
        category_id_map: Optional[Dict[int, int]] = None,
        verbose: bool = True,
    ) -> EvalReport:
        """
        在 COCO-format 資料集上跑 COCO mAP 評估。

        Args:
            annotation_file: COCO annotation JSON 路徑（如 instances_val2017.json）
            images_dir: 對應的影像資料夾
            conf: 偵測 confidence 門檻；評估時建議很低（0.001）以最大化 recall，
                  讓 pycocotools 自己 sweep PR curve
            iou: NMS IoU 門檻
            max_detections: 每張圖最多保留多少偵測（COCO eval 規定預設 100，YOLOX 預設 300）
            category_id_map: model class_id（continuous 0~num_classes-1）→ COCO category_id
                             （離散，可能 1~90）的對應。預設依 annotation 順序 1:1。
            verbose: 是否印出 pycocotools 的 summary table

        Returns:
            EvalReport（含 mAP、mAP@50、mAP@75、AR、per-class AP）
        """
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except ImportError as e:
            raise ImportError(
                "pycocotools is required for evaluation. "
                "Install with: pip install pycocotools"
            ) from e

        ann_path = Path(annotation_file)
        img_root = Path(images_dir)
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")
        if not img_root.exists():
            raise FileNotFoundError(f"Images directory not found: {img_root}")

        if verbose:
            print(f"[evaluate] loading annotations: {ann_path}")
        coco_gt = COCO(str(ann_path))

        # 預設 class_id mapping：model 的 0,1,2,... → annotation 內 categories 出現順序
        cat_ids = coco_gt.getCatIds()
        if category_id_map is None:
            category_id_map = {i: cid for i, cid in enumerate(sorted(cat_ids))}

        img_ids = coco_gt.getImgIds()
        if verbose:
            print(f"[evaluate] running inference on {len(img_ids)} images "
                  f"(conf={conf}, iou={iou}) ...")

        results: List[Dict[str, Any]] = []
        for n, img_id in enumerate(img_ids, 1):
            img_meta = coco_gt.loadImgs(img_id)[0]
            img_path = img_root / img_meta["file_name"]
            if not img_path.exists():
                if verbose:
                    print(f"  [warn] missing image: {img_path}")
                continue

            dets = self.predict(str(img_path), conf=conf, iou=iou)
            # 限制每張圖 top-k 偵測（COCO 標準 maxDets=100，YOLOX eval 慣用 300）
            dets = sorted(dets, key=lambda d: -d.confidence)[:max_detections]

            for d in dets:
                x1, y1, x2, y2 = d.bbox
                cat = category_id_map.get(d.class_id)
                if cat is None:
                    continue
                results.append({
                    "image_id": int(img_id),
                    "category_id": int(cat),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(d.confidence),
                })

            if verbose and n % 50 == 0:
                print(f"  [{n}/{len(img_ids)}] {len(results)} dets so far")

        if verbose:
            print(f"[evaluate] inference done. total detections: {len(results)}")

        if not results:
            return EvalReport(
                mAP=0.0, mAP_50=0.0, mAP_75=0.0, AR_100=0.0,
                per_class_AP={}, num_images=len(img_ids), num_detections=0,
            )

        # 跑 COCO eval（保留 pycocotools 預設 maxDets=[1, 10, 100]，否則 summarize
        # 找不到 maxDets=100 對應的 mind，整排 stats 會回 -1）
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        # summarize() 會印 12 行 stats 並把結果寫進 coco_eval.stats。
        # verbose=False 時把 stdout 吞掉但仍要呼叫，否則 stats 是空 list。
        if verbose:
            coco_eval.summarize()
        else:
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                coco_eval.summarize()

        stats = coco_eval.stats  # 長度 12 的標準 COCO eval stats array

        # 算 per-class AP（IoU=0.5:0.95，跨 area, maxDets=last）
        # precisions: (T, R, K, A, M) — IoU, recall, class, area, maxDet
        per_class_AP: Dict[str, float] = {}
        prec = coco_eval.eval["precision"]   # type: ignore[index]
        a_idx = 0   # area=all
        m_idx = len(coco_eval.params.maxDets) - 1
        for k_idx, cat_id in enumerate(sorted(cat_ids)):
            cls_prec = prec[:, :, k_idx, a_idx, m_idx]
            cls_prec = cls_prec[cls_prec > -1]
            ap = float(cls_prec.mean()) if cls_prec.size else float("nan")
            cat_name = coco_gt.loadCats(cat_id)[0]["name"]
            per_class_AP[cat_name] = ap

        return EvalReport(
            mAP=float(stats[0]),
            mAP_50=float(stats[1]),
            mAP_75=float(stats[2]),
            AR_100=float(stats[8]),
            per_class_AP=per_class_AP,
            num_images=len(img_ids),
            num_detections=len(results),
        )

    def predict(
        self,
        image: Union[str, np.ndarray],
        conf: float = 0.3,
        iou: float = 0.45,
    ) -> List[YoloxDetection]:
        """跑一張圖。回傳 List[YoloxDetection]。"""
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Failed to read image: {image}")
        else:
            img = image
        orig_h, orig_w = img.shape[:2]

        # YOLOX preprocess：letterbox + BGR + no-normalize
        x, scale, pad_w, pad_h = self._preprocess(img)

        # ivit C++ runtime（TensorRT / OpenVINO / ONNX Runtime）跑 raw tensor
        out = self._model.infer({self._input_name: x})
        raw = next(iter(out.values()))   # (1, num_anchors, 5+num_classes)

        return self._postprocess(raw, orig_w, orig_h, scale, pad_w, pad_h, conf, iou)

    # ------------------------------------------------------------------
    # Pre / post
    # ------------------------------------------------------------------

    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        h, w = img.shape[:2]
        th, tw = self.input_size
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        pad_w, pad_h = (tw - nw) // 2, (th - nh) // 2

        resized = cv2.resize(img, (nw, nh))
        letterbox = np.full((th, tw, 3), 114, dtype=np.uint8)
        letterbox[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized

        # BGR、保留 0~255
        x = letterbox.astype(np.float32).transpose(2, 0, 1)[None]  # (1,3,H,W)
        return x, scale, pad_w, pad_h

    def _postprocess(
        self,
        raw: np.ndarray,
        orig_w: int, orig_h: int,
        scale: float, pad_w: int, pad_h: int,
        conf_thres: float, iou_thres: float,
    ) -> List[YoloxDetection]:
        out = raw[0]  # (num_anchors, 85)
        num_anchors, num_values = out.shape

        # YOLOX grid: input H/W 各除 stride 8/16/32
        ih, iw = self.input_size
        levels: List[Tuple[int, int, int]] = []  # (stride, grid_w, offset)
        cumulative = 0
        for s in (8, 16, 32):
            gw = iw // s
            gh = ih // s
            levels.append((s, gw, cumulative))
            cumulative += gh * gw

        if cumulative != num_anchors:
            # ONNX shape 不是預期 grid 大小（可能是 decode_in_inference=True 已 decode 過的版本）
            # 那就直接拿 cx,cy,w,h 做 pixel coords，不再 grid decode
            return self._postprocess_decoded(
                out, orig_w, orig_h, scale, pad_w, pad_h, conf_thres, iou_thres,
            )

        # 為每個 anchor 算 stride / grid (gx, gy)
        anchor_strides = np.empty(num_anchors, dtype=np.float32)
        anchor_gx = np.empty(num_anchors, dtype=np.float32)
        anchor_gy = np.empty(num_anchors, dtype=np.float32)
        for s, gw, off in levels:
            gh = ih // s
            n = gh * gw
            local = np.arange(n)
            anchor_gx[off:off + n] = local % gw
            anchor_gy[off:off + n] = local // gw
            anchor_strides[off:off + n] = s

        cx = (out[:, 0] + anchor_gx) * anchor_strides
        cy = (out[:, 1] + anchor_gy) * anchor_strides
        w = np.exp(out[:, 2]) * anchor_strides
        h = np.exp(out[:, 3]) * anchor_strides

        return self._finalize(
            cx, cy, w, h, out[:, 4], out[:, 5:],
            orig_w, orig_h, scale, pad_w, pad_h, conf_thres, iou_thres,
        )

    def _postprocess_decoded(
        self,
        out: np.ndarray,
        orig_w: int, orig_h: int,
        scale: float, pad_w: int, pad_h: int,
        conf_thres: float, iou_thres: float,
    ) -> List[YoloxDetection]:
        cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        return self._finalize(
            cx, cy, w, h, out[:, 4], out[:, 5:],
            orig_w, orig_h, scale, pad_w, pad_h, conf_thres, iou_thres,
        )

    def _finalize(
        self,
        cx, cy, w, h, obj_score, cls_scores,
        orig_w, orig_h, scale, pad_w, pad_h, conf_thres, iou_thres,
    ) -> List[YoloxDetection]:
        best_cls = cls_scores.argmax(axis=1)
        best_score = cls_scores.max(axis=1)
        confidence = obj_score * best_score
        keep = confidence >= conf_thres
        if not keep.any():
            return []
        cx, cy, w, h = cx[keep], cy[keep], w[keep], h[keep]
        confidence = confidence[keep]
        best_cls = best_cls[keep]

        # 座標：letterbox → 原圖
        x1 = (cx - w / 2 - pad_w) / scale
        y1 = (cy - h / 2 - pad_h) / scale
        x2 = (cx + w / 2 - pad_w) / scale
        y2 = (cy + h / 2 - pad_h) / scale
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        keep_idx = self._nms_per_class(boxes, confidence, best_cls, iou_thres)

        results: List[YoloxDetection] = []
        for i in keep_idx:
            cls = int(best_cls[i])
            label = self.labels[cls] if cls < len(self.labels) else f"class_{cls}"
            results.append(YoloxDetection(
                bbox=(float(boxes[i, 0]), float(boxes[i, 1]),
                      float(boxes[i, 2]), float(boxes[i, 3])),
                class_id=cls,
                label=label,
                confidence=float(confidence[i]),
            ))
        results.sort(key=lambda d: -d.confidence)
        return results

    @staticmethod
    def _nms_per_class(
        boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thres: float,
    ) -> List[int]:
        """Per-class NMS（YOLOX 標準做法：不同類別 box 不互相壓制）。"""
        keep_all: List[int] = []
        for c in np.unique(classes):
            idx = np.where(classes == c)[0]
            if idx.size == 0:
                continue
            cls_boxes = boxes[idx]
            cls_scores = scores[idx]
            order = cls_scores.argsort()[::-1]

            cls_keep: List[int] = []
            while order.size > 0:
                i = order[0]
                cls_keep.append(idx[i])
                if order.size == 1:
                    break
                rest = order[1:]
                xx1 = np.maximum(cls_boxes[i, 0], cls_boxes[rest, 0])
                yy1 = np.maximum(cls_boxes[i, 1], cls_boxes[rest, 1])
                xx2 = np.minimum(cls_boxes[i, 2], cls_boxes[rest, 2])
                yy2 = np.minimum(cls_boxes[i, 3], cls_boxes[rest, 3])
                inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
                area_i = (cls_boxes[i, 2] - cls_boxes[i, 0]) * (cls_boxes[i, 3] - cls_boxes[i, 1])
                area_rest = ((cls_boxes[rest, 2] - cls_boxes[rest, 0]) *
                             (cls_boxes[rest, 3] - cls_boxes[rest, 1]))
                iou = inter / (area_i + area_rest - inter + 1e-9)
                order = rest[iou < iou_thres]
            keep_all.extend(cls_keep)
        return keep_all
