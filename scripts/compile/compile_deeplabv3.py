"""
Compile deeplabv3-mobilenetv3.onnx to HEF for Hailo-8.

Segmentation model — no NMS needed. torchvision DeepLabV3 uses the same
ImageNet mean/std normalization as ResNet, which we bake into the HAR
via Hailo's normalization command so the deployment pipeline sends plain
RGB [0,1] and the chip handles mean/std internally.
"""

import argparse
import glob
import sys

import cv2
import numpy as np
from hailo_sdk_client import ClientRunner


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--onnx", default="models/onnx/deeplabv3-resnet50.onnx")
    p.add_argument("--hef", default="models/hailo/deeplabv3_r50.hef")
    p.add_argument("--har", default="models/hailo/deeplabv3_r50.har")
    p.add_argument("--model-name", default="deeplabv3_r50")
    p.add_argument("--mbv3-gap-fix", action="store_true",
                   help="Apply MobileNetV3-specific global_avgpool_reduction: "
                        "avgpool1-3 (65x65) split by [5,5]; avgpool4-9 (33x33) "
                        "split by [3,3]. Factors must divide the spatial dim.")
    p.add_argument("--perf-max", action="store_true",
                   help="Add performance_param(compiler_optimization_level=max). "
                        "Lets Hailo pick best partition; may avoid AddShortcut "
                        "C++ assertion on MobileNetV3 inverted-residuals. "
                        "Significantly increases compile time (20-40 min).")
    p.add_argument("--end-backbone-only", action="store_true",
                   help="Truncate Hailo graph at backbone output (hardswish "
                        "Mul before classifier). Mirrors YOLO's end_node trick "
                        "-- ASPP + classifier head run on CPU.")
    p.add_argument("--input-hw", type=int, default=520,
                   help="Model input spatial size (square).")
    p.add_argument("--calib-dir", default="calib_images",
                   help="COCO fits PASCAL VOC 21-class domain reasonably well.")
    p.add_argument("--calib-size", type=int, default=1024)
    p.add_argument("--hw-arch", default="hailo8",
                   choices=["hailo8", "hailo8r", "hailo8l"])
    p.add_argument("--no-bake-normalize", action="store_true")
    return p.parse_args()


def letterbox_like_seg(img, size):
    """Segmentor uses plain resize (no aspect-preserve), so match that."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def load_calibration(calib_dir, calib_size, input_hw):
    paths = sorted(glob.glob(f"{calib_dir}/*.jpg"))[:calib_size]
    if not paths:
        sys.exit(f"no calibration images in {calib_dir}")
    imgs = []
    for p in paths:
        bgr = cv2.imread(p)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        imgs.append(letterbox_like_seg(rgb, input_hw))
    calib = np.stack(imgs).astype(np.float32) / 255.0
    print(f"calibration set: {calib.shape} "
          f"range=[{calib.min():.3f}, {calib.max():.3f}] "
          f"from {len(paths)} images")
    return calib


def main():
    args = parse_args()

    print(f"=== [1/3] Translate {args.onnx} -> {args.har} ===")
    runner = ClientRunner(hw_arch=args.hw_arch)
    translate_kwargs = {}
    if args.end_backbone_only:
        # backbone.16.2 is the final hardswish Mul; its output tensor feeds
        # all 5 ASPP branches. Truncating here moves classifier to CPU.
        translate_kwargs["end_node_names"] = [
            "/m/backbone/backbone.16/backbone.16.2/Mul",
        ]
    runner.translate_onnx_model(args.onnx, args.model_name, **translate_kwargs)
    runner.save_har(args.har)

    print(f"\n=== [2/3] Optimize ===")
    opt_runner = ClientRunner(hw_arch=args.hw_arch)
    opt_runner.load_har(args.har)

    script_lines = []
    if not args.no_bake_normalize:
        # [0, 1] scale — Hailo's normalization layer sees dequantized floats,
        # not raw uint8 (see Bug #9 in hailo-compile-bug-log.md).
        script_lines.append(
            "normalization1 = normalization("
            "[0.485, 0.456, 0.406], "
            "[0.229, 0.224, 0.225])"
        )
    if args.mbv3_gap_fix:
        # 65 = 5*13, 33 = 3*11. Factor must divide spatial dim.
        # avgpool1-3 in 65x65 (stride 8), avgpool4-8 in 33x33 (stride 16).
        # avgpool9 sits inside the ASPP image-pool branch — drop it when the
        # graph is truncated at backbone output.
        script_lines.append(
            "pre_quantization_optimization(global_avgpool_reduction, "
            "layers=[avgpool1, avgpool2, avgpool3], division_factors=[5, 5])"
        )
        stride16_layers = ["avgpool4", "avgpool5", "avgpool6",
                           "avgpool7", "avgpool8"]
        if not args.end_backbone_only:
            stride16_layers.append("avgpool9")
        script_lines.append(
            "pre_quantization_optimization(global_avgpool_reduction, "
            f"layers=[{', '.join(stride16_layers)}], "
            "division_factors=[3, 3])"
        )
    if args.perf_max:
        script_lines.append("performance_param(compiler_optimization_level=max)")
    opt_runner.load_model_script("\n".join(script_lines) + "\n")

    calib = load_calibration(args.calib_dir, args.calib_size, args.input_hw)
    opt_runner.optimize(calib)

    print(f"\n=== [3/3] Compile -> {args.hef} ===")
    hef = opt_runner.compile()
    with open(args.hef, "wb") as f:
        f.write(hef)
    print(f"\nWrote {args.hef} ({len(hef) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
