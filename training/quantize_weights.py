"""Weight-only INT8 quantisation (W8A32) of an ONNX model.

Every Conv/Gemm weight is stored as INT8 with one scale per output channel
(symmetric, zero point 0) and turned back into FP32 by a DequantizeLinear node;
activations stay FP32.

Why not full static INT8 (W8A8)? For these fine-tuned MobileNetV3 models it
collapsed (age MAE 5.1 → 16.4 years, expression accuracy 68 % → 29 %). Layer-wise
analysis showed that the weights quantise almost losslessly (per-channel INT8:
MAE +0.03 y), but a few activations have extreme outliers (first layer: up to +297
while 99.99 % of values lie in −0.4 … 56). With one 8-bit scale per tensor, the
small values that carry most of the information are rounded away, and clipping the
outliers destroys the model as well. See docs/results/quantization.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def quantize_weights_int8(src: str | Path, dst: str | Path, op_types=("Conv", "Gemm")) -> dict:
    """Weight-only INT8 quantisation (W8A32): Conv/Gemm weights are stored as per-channel
    symmetric INT8 with a DequantizeLinear node; activations stay FP32. Returns statistics."""
    model = onnx.load(str(src))
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    done, dq_nodes, stats = {}, [], {"tensors": 0, "params": 0}

    for node in g.node:
        if node.op_type not in op_types or len(node.input) < 2 or node.input[1] not in inits:
            continue
        name = node.input[1]
        if name not in done:
            w = numpy_helper.to_array(inits[name]).astype(np.float32)
            axis = 0
            if node.op_type == "Gemm":
                trans_b = next((a.i for a in node.attribute if a.name == "transB"), 0)
                axis = 0 if trans_b else 1
            reduce = tuple(i for i in range(w.ndim) if i != axis)
            max_abs = np.abs(w).max(axis=reduce)
            scale = np.where(max_abs > 0, max_abs / 127.0, 1.0).astype(np.float32)
            shape = [1] * w.ndim
            shape[axis] = -1
            q = np.clip(np.round(w / scale.reshape(shape)), -127, 127).astype(np.int8)

            g.initializer.remove(inits[name])
            g.initializer.extend([
                numpy_helper.from_array(q, f"{name}_int8"),
                numpy_helper.from_array(scale, f"{name}_scale"),
                numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8), f"{name}_zero"),
            ])
            dq_nodes.append(helper.make_node(
                "DequantizeLinear", [f"{name}_int8", f"{name}_scale", f"{name}_zero"], [f"{name}_dq"],
                axis=axis, name=f"{name}_DequantizeLinear"))
            done[name] = f"{name}_dq"
            stats["tensors"] += 1
            stats["params"] += int(w.size)
        node.input[1] = done[name]

    # DequantizeLinear nodes only read initializers, so they can go first (topological order kept)
    nodes = dq_nodes + list(g.node)
    del g.node[:]
    g.node.extend(nodes)
    onnx.checker.check_model(model)
    onnx.save_model(model, str(dst))
    return stats
