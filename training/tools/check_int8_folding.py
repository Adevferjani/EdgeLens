"""Check that ONNX Runtime turns the INT8 weights back into FP32 once at load time.

    python training/tools/check_int8_folding.py

Prints the ONNX Runtime version and the latency of the FP32 and INT8 models with and
without the 'session.disable_quant_qdq' option; results -> runs/int8_folding_check.json
"""
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REPO_ROOT, cpu_name, save_json  # noqa: E402

ort.set_default_logger_severity(3)
MODELS = REPO_ROOT / "pwa" / "models"


def bench(path, fold, threads=None, n=300):
    so = ort.SessionOptions()
    if fold:
        so.add_session_config_entry("session.disable_quant_qdq", "1")
    if threads:
        so.intra_op_num_threads = threads
    opt = Path(tempfile.mkdtemp()) / "optimised.onnx"
    so.optimized_model_filepath = str(opt)
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    size = s.get_inputs()[0].shape[2]
    x = np.random.default_rng(0).random((1, 3, size, size), dtype=np.float32)
    for _ in range(30):
        s.run(None, {"input": x})
    times = []
    for _ in range(n):
        t = time.perf_counter()
        s.run(None, {"input": x})
        times.append((time.perf_counter() - t) * 1000)
    dq_left = sum(n.op_type == "DequantizeLinear" for n in onnx.load(str(opt)).graph.node)
    return {"mean_ms": round(float(np.mean(times)), 3), "median_ms": round(float(np.median(times)), 3),
            "dequantize_nodes_after_optimisation": dq_left}


def main():
    res = {"onnxruntime": ort.__version__, "python": platform.python_version(), "cpu": cpu_name(), "results": {}}
    for task in ("age_gender", "expression"):
        for threads in (None, 1):
            key = f"{task} ({'all threads' if threads is None else '1 thread'})"
            res["results"][key] = {
                "fp32": bench(MODELS / f"{task}_fp32.onnx", False, threads),
                "int8_default": bench(MODELS / f"{task}_int8.onnx", False, threads),
                "int8_folded": bench(MODELS / f"{task}_int8.onnx", True, threads),
            }
    out = REPO_ROOT / "runs" / "int8_folding_check.json"
    save_json(res, out)
    print(f"ONNX Runtime {res['onnxruntime']} on {res['cpu']}")
    for key, r in res["results"].items():
        print(f"  {key:28s} FP32 {r['fp32']['median_ms']:.2f} ms | INT8 {r['int8_default']['median_ms']:.2f} ms "
              f"(DQ left: {r['int8_default']['dequantize_nodes_after_optimisation']}) | INT8 folded "
              f"{r['int8_folded']['median_ms']:.2f} ms "
              f"(DQ left: {r['int8_folded']['dequantize_nodes_after_optimisation']})")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
