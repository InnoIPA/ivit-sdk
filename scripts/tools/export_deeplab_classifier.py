"""
Extract the classifier subgraph from deeplabv3-mobilenetv3.onnx.

Pairs with compile_deeplabv3.py --end-backbone-only: the Hailo HEF ends at
the backbone's final hardswish Mul and emits [1, 33, 33, 960]; this head
ONNX consumes that same tensor (as [1, 960, 33, 33] NCHW) and runs the full
ASPP + classifier + bilinear resize on CPU.

See docs/development/hailo-compile-bug-log.md Bug #11.
"""

import argparse
import onnx


BACKBONE_OUT = "/m/backbone/backbone.16/backbone.16.2/Mul_output_0"


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--onnx", default="deeplabv3-mobilenetv3.onnx")
    p.add_argument("--out", default="deeplabv3_mbv3_head.onnx")
    p.add_argument("--input-tensor", default=BACKBONE_OUT,
                   help="Tensor name where the Hailo HEF output feeds in.")
    p.add_argument("--output-tensor", default=None,
                   help="Override output tensor name. Default: the model's "
                        "first graph output.")
    return p.parse_args()


def main():
    args = parse_args()

    m = onnx.load(args.onnx)
    if args.output_tensor is None:
        args.output_tensor = m.graph.output[0].name
        print(f"Using model output tensor: {args.output_tensor}")

    print(f"Extracting subgraph from '{args.input_tensor}' to "
          f"'{args.output_tensor}' -> {args.out}")
    onnx.utils.extract_model(
        args.onnx,
        args.out,
        input_names=[args.input_tensor],
        output_names=[args.output_tensor],
    )

    extracted = onnx.load(args.out)
    print(f"\nExtracted subgraph:")
    print(f"  nodes: {len(extracted.graph.node)}")
    for inp in extracted.graph.input:
        dims = [d.dim_value or d.dim_param or "?" for d in inp.type.tensor_type.shape.dim]
        print(f"  input : {inp.name} {dims}")
    for out in extracted.graph.output:
        dims = [d.dim_value or d.dim_param or "?" for d in out.type.tensor_type.shape.dim]
        print(f"  output: {out.name} {dims}")


if __name__ == "__main__":
    main()
