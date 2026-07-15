"""
Compile MambaVision-T backbone (levels.0 + levels.1) to a Hailo HEF.

MambaVision is a hybrid Mamba+Transformer ImageNet classifier. The Mamba
selective-scan (S6) operator and ONNX Einsum nodes aren't supported by
the Hailo Dataflow Compiler — translate fails at the SSM mixer (officially
confirmed by Hailo support).

The workaround (suggested by Hailo) is to truncate the graph at the last
pure-conv op (levels.1 downsample reduction conv), encode that conv-only
backbone as a HEF, and run the SSM/transformer head on host CPU. Same
hybrid pattern as DeepLabV3-MobileNetV3 (see Bug #11 in
hailo-compile-bug-log.md and docs/development/hef-segmentation-hybrid.md).

Output of the HEF is the levels.2 input feature map (NCHW after host
transpose); the CPU-side classifier head (levels.2 + levels.3 + final
norm/GAP/FC) is exported separately as mambavision_t_head.onnx.
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
from hailo_sdk_client import ClientRunner


# Last conv before levels.2 starts the SSM/transformer pipeline.
BACKBONE_END_NODE = "/m/model/levels.1/downsample/reduction/reduction.0/Conv"


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--onnx", default=os.path.expanduser("~/.cache/ivit/models/mambavision-t.onnx"))
    p.add_argument("--hef", default="models/hailo/mambavision_t.hef")
    p.add_argument("--har", default="models/hailo/mambavision_t.har")
    p.add_argument("--model-name", default="mambavision_t")
    p.add_argument("--input-hw", type=int, default=224)
    p.add_argument(
        "--calib-dir",
        default="calib_images")
    p.add_argument("--calib-size", type=int, default=2048)
    p.add_argument("--hw-arch", default="hailo8",
                   choices=["hailo8", "hailo8r", "hailo8l"])
    p.add_argument("--no-bake-normalize", action="store_true")
    p.add_argument("--end-node", default=BACKBONE_END_NODE,
                   help="ONNX node name to truncate at. Default: last conv "
                        "before levels.2 SSM/transformer.")
    return p.parse_args()


def center_crop_resize(img, size):
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


def main():
    args = parse_args()

    print(f"=== [1/3] Translate {args.onnx} -> {args.har} ===")
    print(f"  end_node: {args.end_node}")
    runner = ClientRunner(hw_arch=args.hw_arch)
    runner.translate_onnx_model(
        args.onnx, args.model_name, end_node_names=[args.end_node])
    runner.save_har(args.har)

    print(f"\n=== [2/3] Optimize ===")
    opt_runner = ClientRunner(hw_arch=args.hw_arch)
    opt_runner.load_har(args.har)

    if not args.no_bake_normalize:
        # ImageNet mean/std in [0, 1] scale (Hailo dequantizes UINT8 -> [0, 1]
        # before this layer; see Bug #9).
        opt_runner.load_model_script(
            "normalization1 = normalization("
            "[0.485, 0.456, 0.406], "
            "[0.229, 0.224, 0.225])\n"
        )

    calib = load_calibration(args.calib_dir, args.calib_size, args.input_hw)
    opt_runner.optimize(calib)

    print(f"\n=== [3/3] Compile -> {args.hef} ===")
    hef = opt_runner.compile()
    with open(args.hef, "wb") as f:
        f.write(hef)
    print(f"\nWrote {args.hef} ({len(hef) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
