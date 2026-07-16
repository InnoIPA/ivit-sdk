"""
Extract the SSM/Transformer head subgraph from mambavision-t.onnx.

Pairs with compile_mambavision_t.py: that script truncates the Hailo HEF at
the last pure-conv op (levels.1 downsample reduction conv) and emits a
[14, 14, 320] feature map. This script extracts everything from that
boundary tensor through to the final `logits` output, producing an ONNX
subgraph that the C++ Classifier auto-loads as `<stem>_head.onnx` and runs
on CPU via the OpenVINORuntime.

See docs/development/hailo-compile-bug-log.md Bug #15 and
docs/development/hef-classification-hybrid.md.
"""

import argparse
import os
from pathlib import Path

import onnx


# Model cache follows the SDK convention: $IVIT_CACHE_DIR, else
# ~/.cache/ivit/models (see python/ivit/zoo/registry.py:_get_cache_dir).
CACHE_DIR = Path(os.environ.get(
    "IVIT_CACHE_DIR", str(Path.home() / ".cache" / "ivit" / "models")))
DEFAULT_ONNX = str(CACHE_DIR / "mambavision-t.onnx")


# Output tensor name of the last pure-conv op before the SSM pipeline.
# This is the same tensor that compile_mambavision_t.py uses as its
# end_node_names, except here we need the tensor name (with `_output_0`
# suffix) rather than the node name.
BACKBONE_OUT = (
    "/m/model/levels.1/downsample/reduction/reduction.0/Conv_output_0"
)


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--onnx",
        default=DEFAULT_ONNX,
        help="Source mambavision ONNX (full graph).",
    )
    p.add_argument(
        "--out",
        default="mambavision_t_head.onnx",
        help="Destination head subgraph. Should sit next to the HEF and "
             "match its stem (e.g. mambavision_t.hef + mambavision_t_head.onnx) "
             "so Classifier auto-loads it.",
    )
    p.add_argument(
        "--input-tensor",
        default=BACKBONE_OUT,
        help="Tensor name where the Hailo HEF output feeds in.",
    )
    p.add_argument(
        "--output-tensor",
        default=None,
        help="Override output tensor name. Default: the model's first "
             "graph output (typically 'logits').",
    )
    return p.parse_args()


def main():
    args = parse_args()

    m = onnx.load(args.onnx)
    if args.output_tensor is None:
        args.output_tensor = m.graph.output[0].name
        print(f"Using model output tensor: {args.output_tensor}")

    print(f"Extracting subgraph "
          f"from '{args.input_tensor}' "
          f"to '{args.output_tensor}' "
          f"-> {args.out}")
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
        dims = [d.dim_value or d.dim_param or "?"
                for d in inp.type.tensor_type.shape.dim]
        print(f"  input : {inp.name} {dims}")
    for out in extracted.graph.output:
        dims = [d.dim_value or d.dim_param or "?"
                for d in out.type.tensor_type.shape.dim]
        print(f"  output: {out.name} {dims}")


if __name__ == "__main__":
    main()
