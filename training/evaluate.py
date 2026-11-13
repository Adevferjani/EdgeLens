"""Evaluate a trained run on the test split and benchmark inference latency.

    python training/evaluate.py --run runs/age_gender_cuda --data data/UTKFace --onnx
    python training/evaluate.py --run runs/expression_cuda --data data/fer2013 --onnx

* Test metrics of the PyTorch checkpoint (and of the exported FP32/INT8 ONNX models
  with --onnx, to measure the accuracy cost of quantisation).
* Latency at batch size 1 (as in the app): PyTorch CPU, PyTorch GPU (if available),
  ONNX Runtime CPU. Results -> <run>/evaluation.json and a Markdown table on screen.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

from common import REPO_ROOT, cpu_name, portable_path, save_json
from data import build_datasets
from engine import compute_metrics, predict_torch
from metrics import scalar_metrics
from models import build_model, count_parameters, export_wrapper


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--onnx", action="store_true", help="also evaluate pwa/models/<task>_{fp32,int8}.onnx")
    p.add_argument("--models-dir", default=str(REPO_ROOT / "pwa" / "models"))
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def benchmark(fn, warmup, iters) -> dict:
    """Latency of `fn` after `warmup` calls: mean, median, 95th percentile (ms) and FPS."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    t = np.array(times)
    return {"mean_ms": round(t.mean(), 3), "p50_ms": round(np.percentile(t, 50), 3),
            "p95_ms": round(np.percentile(t, 95), 3), "fps": round(1000 / t.mean(), 1)}


def onnx_predict(task, path, dataset, threads) -> dict:
    """Predictions of an ONNX model on `dataset`, one image at a time (batch size 1,
    as in the app), in the same format as `engine.predict_torch`."""
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    if "int8" in Path(path).name:  # weight-only INT8: dequantise once at load time
        so.add_session_config_entry("session.disable_quant_qdq", "1")
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    acc = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        out = sess.run(None, {"input": sample[0].unsqueeze(0).numpy()})
        if task == "age_gender":
            acc.setdefault("age_pred", []).append(float(out[0][0]))
            acc.setdefault("p_female", []).append(float(out[1][0]))
            acc.setdefault("age_true", []).append(float(sample[1]))
            acc.setdefault("gender_true", []).append(float(sample[2]))
        else:
            acc.setdefault("probs", []).append(out[0][0])
            acc.setdefault("y_true", []).append(int(sample[1]))
    return {k: np.asarray(v) for k, v in acc.items()}, sess


def main():
    args = parse_args()
    ort.set_default_logger_severity(3)
    if args.threads:
        torch.set_num_threads(args.threads)
    run = Path(args.run)
    ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    task, size = ckpt["task"], ckpt["img_size"]
    split_dir = Path(ckpt["args"].get("split_dir") or REPO_ROOT / "splits")
    test_ds = build_datasets(task, args.data, size, ckpt["args"]["seed"], split_dir, train_aug=False)["test"]

    net = build_model(task, pretrained=False)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    results = {"task": task, "run": portable_path(run), "cpu": cpu_name(), "img_size": size,
               "parameters": count_parameters(net), "test_samples": len(test_ds),
               "checkpoint_mb": round((run / "best.pt").stat().st_size / 1e6, 3),
               "metrics": {}, "latency": {}}

    # ---- accuracy: PyTorch checkpoint (FP32) ----
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=args.workers)
    pred = predict_torch(task, net, loader, torch.device("cpu"))
    results["metrics"]["pytorch_fp32"] = compute_metrics(task, pred)

    # ---- accuracy: ONNX models (the files deployed to the phone) ----
    sessions = {}
    if args.onnx:
        for variant in ("fp32", "int8"):
            path = Path(args.models_dir) / f"{task}_{variant}.onnx"
            if not path.exists():
                print(f"skip {path} (not found)")
                continue
            p, sess = onnx_predict(task, path, test_ds, args.threads)
            results["metrics"][f"onnx_{variant}"] = compute_metrics(task, p)
            results[f"onnx_{variant}_mb"] = round(path.stat().st_size / 1e6, 3)
            sessions[f"onnxruntime_cpu_{variant}"] = sess

    # ---- latency at batch size 1 ----
    model = export_wrapper(task, net).eval()
    x = torch.rand(1, 3, size, size)
    with torch.inference_mode():
        results["latency"]["pytorch_cpu_fp32"] = benchmark(lambda: model(x), args.warmup, args.iters)
        if torch.cuda.is_available():
            gpu_model, xg = model.to("cuda"), x.to("cuda")

            def run_gpu():
                gpu_model(xg)
                torch.cuda.synchronize()

            results["latency"]["pytorch_gpu_fp32"] = benchmark(run_gpu, args.warmup, args.iters)
            results["gpu"] = torch.cuda.get_device_name(0)
    for name, sess in sessions.items():
        feed = {"input": x.numpy()}
        results["latency"][name] = benchmark(lambda s=sess, f=feed: s.run(None, f), args.warmup, args.iters)

    save_json(results, run / "evaluation.json")

    # ---- Markdown summary ----
    print(f"\n### {task} – test set (n={len(test_ds)})\n")
    keys = (["age_mae", "age_mae_18plus", "gender_accuracy", "age_group_accuracy_18plus"]
            if task == "age_gender" else ["accuracy", "macro_f1", "happy_sad_accuracy_2way"])
    print("| Model | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for name, m in results["metrics"].items():
        s = scalar_metrics(m)
        print(f"| {name} | " + " | ".join(str(s.get(k)) for k in keys) + " |")
    print("\n| Runtime (batch 1) | mean ms | p95 ms | FPS |\n|---|---|---|---|")
    for name, lat in results["latency"].items():
        print(f"| {name} | {lat['mean_ms']} | {lat['p95_ms']} | {lat['fps']} |")
    print(f"\nSaved {run / 'evaluation.json'}")


if __name__ == "__main__":
    main()
