"""
Strip SE (Squeeze-and-Excite) blocks from an EfficientNet-B0 ONNX.

Pattern matched (per SE block in torchvision EfficientNet-B0):

    feature_tensor ┬───────────────────────────────────────────────┐
                   │                                                ▼
                   └──> GAP --> Conv1x1 --> Sigmoid*x (swish)        Mul ──> next
                                  --> Conv1x1 --> Sigmoid ─────────►(att)
                                                                   ▲
                                                          (SE attention scale)

We rewrite to:

    feature_tensor ─────────────────────────────────────────────────► next

equivalent to setting the per-channel attention scale to 1 everywhere.

Why: EfficientNet's SE blocks cause int8 quantization activation collapse
on Hailo-8 (see Bug #14 in docs/development/hailo-compile-bug-log.md). The
SE-end ew_mult only supports a8_w8 and the dense layers can't bridge to
a16, so the attention values get crushed during quantization. Removing SE
costs ~5% top-1 accuracy on ImageNet but gives a quantization-robust
"plain MBConv" backbone that maps cleanly to Hailo dataflow hardware.
"""

import argparse
import collections

import onnx


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in", dest="in_path", default="efficientnet-b0.onnx")
    p.add_argument("--out", default="efficientnet-b0_no_se.onnx")
    return p.parse_args()


def find_se_blocks(graph):
    """
    Return list of (gap_node, se_mul_node, feature_tensor, chain_nodes).

    - gap_node is the GlobalAveragePool that opens the SE branch
    - se_mul_node is the Mul that applies attention back onto the feature
    - feature_tensor is the tensor name shared by the GAP input and the
      SE Mul's "other" input (the passthrough)
    - chain_nodes is the set of intermediate SE-only nodes to delete
    """
    # producer (output tensor name -> producing node)
    producer = {}
    for n in graph.node:
        for o in n.output:
            producer[o] = n
    # consumers (input tensor name -> [consuming nodes])
    consumers = collections.defaultdict(list)
    for n in graph.node:
        for i in n.input:
            consumers[i].append(n)

    blocks = []
    for gap in graph.node:
        if gap.op_type != "GlobalAveragePool":
            continue
        feat = gap.input[0]

        # SE Mul = a Mul whose inputs include `feat` AND whose other input
        # traces back to `gap` via the SE chain. Final classifier GAP has
        # no such Mul (its output goes straight into Flatten/Gemm).
        candidates = [n for n in consumers.get(feat, []) if n.op_type == "Mul"]
        se_mul = None
        for cand in candidates:
            other_inputs = [i for i in cand.input if i != feat]
            # BFS upstream from each `other_input`; if we hit `gap` we found it
            for oi in other_inputs:
                seen = set()
                queue = collections.deque([oi])
                hit = False
                while queue:
                    t = queue.popleft()
                    if t in seen:
                        continue
                    seen.add(t)
                    p_node = producer.get(t)
                    if p_node is None:
                        continue
                    if p_node is gap:
                        hit = True
                        break
                    queue.extend(p_node.input)
                if hit:
                    se_mul = cand
                    break
            if se_mul:
                break

        if se_mul is None:
            # final classifier GAP (no SE Mul) — leave untouched
            continue

        # Collect chain = every node on a path from `gap` (inclusive) down
        # to `se_mul` (inclusive). Forward BFS from gap, stop expanding past
        # se_mul. NodeProto isn't hashable so we key on id().
        chain_ids = set()
        chain_nodes = []
        queue = collections.deque([gap])
        while queue:
            n = queue.popleft()
            if id(n) in chain_ids:
                continue
            chain_ids.add(id(n))
            chain_nodes.append(n)
            if n is se_mul:
                continue
            for o in n.output:
                for cons in consumers.get(o, []):
                    if id(cons) not in chain_ids:
                        queue.append(cons)

        blocks.append((gap, se_mul, feat, chain_nodes))

    return blocks


def main():
    args = parse_args()
    m = onnx.load(args.in_path)
    g = m.graph

    blocks = find_se_blocks(g)
    print(f"Found {len(blocks)} SE block(s) to strip.")
    for gap, se_mul, feat, chain in blocks:
        print(f"  GAP={gap.name}  SE_Mul={se_mul.name}  "
              f"feature={feat}  chain_size={len(chain)}")

    # 1) Rewrite consumers: every node consuming se_mul.output[0] now
    #    consumes the feature tensor directly. Same for graph outputs.
    rename = {se_mul.output[0]: feat for _, se_mul, feat, _ in blocks}
    for n in g.node:
        for i, inp in enumerate(n.input):
            if inp in rename:
                n.input[i] = rename[inp]
    for o in g.output:
        if o.name in rename:
            o.name = rename[o.name]

    # 2) Delete every chain node (GAP, internal SE conv/sigmoid/mul, SE Mul).
    to_remove = set()
    for _, _, _, chain in blocks:
        for n in chain:
            to_remove.add(id(n))

    keep_nodes = [n for n in g.node if id(n) not in to_remove]
    del g.node[:]
    g.node.extend(keep_nodes)

    # 3) Drop dangling initializers + value_infos that nothing consumes now.
    used_tensors = set()
    for n in g.node:
        used_tensors.update(n.input)
        used_tensors.update(n.output)
    for o in g.output:
        used_tensors.add(o.name)
    for i in g.input:
        used_tensors.add(i.name)

    keep_init = [t for t in g.initializer if t.name in used_tensors]
    del g.initializer[:]
    g.initializer.extend(keep_init)

    keep_vi = [vi for vi in g.value_info if vi.name in used_tensors]
    del g.value_info[:]
    g.value_info.extend(keep_vi)

    onnx.checker.check_model(m)
    onnx.save(m, args.out)
    print(f"\nWrote {args.out}")
    print(f"  nodes: before={len(g.node) + len(to_remove)} "
          f"after={len(g.node)} (removed {len(to_remove)})")


if __name__ == "__main__":
    main()
