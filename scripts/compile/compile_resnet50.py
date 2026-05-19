"""
Compile resnet50.onnx to HEF for Hailo-8.

Classification model — no NMS postprocess needed. ResNet50 expects
ImageNet-normalized input (mean/std), which we bake into the HAR via
Hailo's normalization model-script command so the deployed pipeline
can feed plain RGB [0,1] without additional preprocessing.
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
    p.add_argument("--onnx", default="models/onnx/resnet50.onnx", help="Input ONNX path.")
    p.add_argument("--hef", default="models/hailo/resnet50.hef", help="Output HEF path.")
    p.add_argument("--har", default="models/hailo/resnet50.har", help="Intermediate HAR path.")
    p.add_argument("--model-name", default="resnet50",
                   help="Model name inside the HAR.")
    p.add_argument("--input-hw", type=int, default=224,
                   help="Model input spatial size (square).")
    p.add_argument("--calib-dir", default="/home/dos/coco_val2017/val2017",
                   help="Directory of calibration JPGs.")
    p.add_argument("--calib-size", type=int, default=512,
                   help="Number of calibration images to use.")
    p.add_argument("--hw-arch", default="hailo8",
                   choices=["hailo8", "hailo8r", "hailo8l"])
    p.add_argument("--no-bake-normalize", action="store_true",
                   help="Skip baking ImageNet mean/std normalize into the "
                        "HAR. Use this if your deployment pipeline already "
                        "handles normalization before sending to the chip.")
    return p.parse_args()


def center_crop_resize(img, size):
    """Resize short side to `size * 256/224` then center-crop to size (ImageNet eval preset)."""
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
    # [0, 1] range — matches ResNet training input AFTER mean/std step,
    # but the normalization layer we bake in will handle subtracting mean
    # and dividing by std internally; calibration just needs consistent
    # pre-normalize values.
    calib = np.stack(imgs).astype(np.float32) / 255.0
    print(f"calibration set: {calib.shape} "
          f"range=[{calib.min():.3f}, {calib.max():.3f}] "
          f"from {len(paths)} images")
    return calib


def main():
    args = parse_args()

    # [1/3] translate ONNX -> HAR
    print(f"=== [1/3] Translate {args.onnx} -> {args.har} ===")
    runner = ClientRunner(hw_arch=args.hw_arch)
    runner.translate_onnx_model(args.onnx, args.model_name)
    runner.save_har(args.har)

    # [2/3] optimize (calibrate + quantize)
    print(f"\n=== [2/3] Optimize ===")
    opt_runner = ClientRunner(hw_arch=args.hw_arch)
    opt_runner.load_har(args.har)

    # Bake ImageNet normalization into the HAR so the deployment pipeline
    # can just send plain RGB [0,1] (which iVIT-SDK's Detector/Classifier
    # already produces). Values below are the standard torchvision ResNet
    # ImageNet stats scaled to [0, 255] because Hailo's normalization layer
    # operates on the uint8 input range.
    if not args.no_bake_normalize:
        # Hailo's normalize layer operates in the model's internal float
        # scale (which Hailo treats as [0, 1] after auto-dequantizing the
        # UINT8 input), so mean/std values should be in [0, 1] range.
        # Using the *255 variants here would give catastrophic wrong output.
        model_script = (
            "normalization1 = normalization("
            "[0.485, 0.456, 0.406], "
            "[0.229, 0.224, 0.225])\n"
        )
        opt_runner.load_model_script(model_script)

    calib = load_calibration(args.calib_dir, args.calib_size, args.input_hw)
    opt_runner.optimize(calib)

    # [3/3] compile -> HEF
    print(f"\n=== [3/3] Compile -> {args.hef} ===")
    hef = opt_runner.compile()
    with open(args.hef, "wb") as f:
        f.write(hef)
    print(f"\nWrote {args.hef} ({len(hef) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
