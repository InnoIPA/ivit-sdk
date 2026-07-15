"""
Compile efficientnet-b0.onnx to HEF for Hailo-8.

EfficientNet-B0 is a classification model (224x224, ImageNet 1000 classes)
with SE blocks scattered through every inverted-residual stage. The 16 SE
GlobalAveragePool layers (plus 1 final classifier GAP) span spatial
[112, 56, 28, 14, 7], and the wider ones blow past Hailo's int8 shift
delta > 2 limit during quantization (same failure mode as DeepLabV3-
MobileNetV3, see Bug #10 in docs/development/hailo-compile-bug-log.md).

The fix is the same as Bug #10: pre_quantization_optimization(
global_avgpool_reduction, ...) with division_factors that divide the
spatial dim exactly. Mapping from Hailo internal layer name to spatial:

    avgpool1     112x112  (features.1.0  / 32 ch)
    avgpool2,3   56x56    (features.2.x  / 96, 144 ch)
    avgpool4,5   28x28    (features.3.x  / 144, 240 ch)
    avgpool6..11 14x14    (features.4-5  / 240, 480, 672 ch)
    avgpool12..17 7x7     (features.6-7 + final / 672, 1152, 1280 ch)

7 is prime — can't reduce — but log2(49) is small enough that those
GAPs usually fit within the shift delta budget without reduction.
"""

import argparse
import glob
import sys

import cv2
import numpy as np
from hailo_sdk_client import ClientRunner


# (avgpool layer indices, spatial size, division_factors).
# 7x7 GAPs deliberately omitted: 7 is prime, can't be divided, and shift
# delta on 7x7 (log2(49) ~ 5.6) often fits the budget. Re-enable selectively
# if Hailo still flags them.
GAP_FIX_GROUPS = [
    ([1],            112, [8, 8]),   # 112 = 8 * 14
    ([2, 3],         56,  [8, 8]),   # 56  = 8 * 7
    ([4, 5],         28,  [7, 7]),   # 28  = 7 * 4
    ([6, 7, 8, 9, 10, 11], 14, [7, 7]),  # 14  = 7 * 2
]


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--onnx", default="models/onnx/efficientnet-b0.onnx")
    p.add_argument("--hef", default="models/hailo/efficientnet_b0.hef")
    p.add_argument("--har", default="models/hailo/efficientnet_b0.har")
    p.add_argument("--model-name", default="efficientnet_b0")
    p.add_argument("--input-hw", type=int, default=224)
    p.add_argument("--calib-dir", default="calib_images")
    p.add_argument("--calib-size", type=int, default=512)
    p.add_argument("--hw-arch", default="hailo8",
                   choices=["hailo8", "hailo8r", "hailo8l"])
    p.add_argument("--no-bake-normalize", action="store_true")
    p.add_argument("--gap-fix", action="store_true",
                   help="Apply global_avgpool_reduction to SE block GAPs "
                        "(avgpool1-11). Required for shift delta > 2 errors.")
    p.add_argument("--gap-fix-7x7", action="store_true",
                   help="Also include the 7x7 GAPs (avgpool12-17). 7 is "
                        "prime so this is a no-op reduction; only use as "
                        "an explicit override if Hailo flags them.")
    p.add_argument("--se-a16", action="store_true",
                   help="Quantize SE block layers (fc1..fc32 squeeze/excite "
                        "+ ew_mult1..ew_mult16) at a16_w8 to combat "
                        "quantization activation collapse on swish+SE "
                        "(see Bug #14).")
    return p.parse_args()


def center_crop_resize(img, size):
    """Resize short side to size*256/224 then center-crop (ImageNet eval)."""
    h, w = img.shape[:2]
    scale = (size * 256 // 224) / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (nh - size) // 2
    left = (nw - size) // 2
    return resized[top:top + size, left:left + size]


def load_calibration(calib_dir, calib_size, input_hw):
    paths = sorted(glob.glob(f"{calib_dir}/*.jpg"))[:calib_size]
    if not paths:
        sys.exit(f"no calibration images in {calib_dir}")
    imgs = []
    for p in paths:
        bgr = cv2.imread(p)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        imgs.append(center_crop_resize(rgb, input_hw))
    calib = np.stack(imgs).astype(np.float32) / 255.0
    print(f"calibration set: {calib.shape} "
          f"range=[{calib.min():.3f}, {calib.max():.3f}] "
          f"from {len(paths)} images")
    return calib


def build_model_script(args):
    lines = []
    if not args.no_bake_normalize:
        # ImageNet mean/std in [0, 1] scale (Hailo dequantizes UINT8 -> [0, 1]
        # before this layer; see Bug #9).
        lines.append(
            "normalization1 = normalization("
            "[0.485, 0.456, 0.406], "
            "[0.229, 0.224, 0.225])"
        )
    if args.gap_fix:
        for indices, _spatial, factors in GAP_FIX_GROUPS:
            layers = "[" + ", ".join(f"avgpool{i}" for i in indices) + "]"
            lines.append(
                f"pre_quantization_optimization(global_avgpool_reduction, "
                f"layers={layers}, division_factors=[{factors[0]}, {factors[1]}])"
            )
        if args.gap_fix_7x7:
            seven_layers = "[" + ", ".join(
                f"avgpool{i}" for i in range(12, 18)) + "]"
            lines.append(
                f"pre_quantization_optimization(global_avgpool_reduction, "
                f"layers={seven_layers}, division_factors=[7, 7])"
            )
    if args.se_a16:
        # SE chain: avgpool -> fc (squeeze) -> fc (excite) -> resize -> ew_mult.
        # On hailo8, supported precision modes for each layer:
        #   - dense (fc): a8_w8, a8_w8_a8, a8_w8_a16, a16_w16, a16_w16_a16
        #   - ew_mult:    a8_w8, a8_w8_a8  (NO 16-bit input/output)
        # Pure a16_w16_a16 on fc fails at mapping ("precision mode is not
        # accurate") because the ew_mult downstream can't accept a16 input.
        # Use a8_w8_a16 instead: input stays a8 (aligns with upstream/GAP),
        # output widens to a16 so the sigmoid + excite-conv get finer
        # attention values; the resize/ew_mult re-quantize back to a8 but
        # the attention is already differentiated by then.
        se_fcs = "[" + ", ".join(f"fc{i}" for i in range(1, 33)) + "]"
        lines.append(
            f"quantization_param({se_fcs}, precision_mode=a8_w8_a16)"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def main():
    args = parse_args()

    print(f"=== [1/3] Translate {args.onnx} -> {args.har} ===")
    runner = ClientRunner(hw_arch=args.hw_arch)
    runner.translate_onnx_model(args.onnx, args.model_name)
    runner.save_har(args.har)

    print(f"\n=== [2/3] Optimize ===")
    opt_runner = ClientRunner(hw_arch=args.hw_arch)
    opt_runner.load_har(args.har)

    script = build_model_script(args)
    if script.strip():
        print(f"model script:\n{script}")
        opt_runner.load_model_script(script)

    calib = load_calibration(args.calib_dir, args.calib_size, args.input_hw)
    opt_runner.optimize(calib)

    print(f"\n=== [3/3] Compile -> {args.hef} ===")
    hef = opt_runner.compile()
    with open(args.hef, "wb") as f:
        f.write(hef)
    print(f"\nWrote {args.hef} ({len(hef) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
