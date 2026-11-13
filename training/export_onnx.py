"""Export a trained run to ONNX for the web app (FP32 + INT8).

    python training/export_onnx.py --run runs/age_gender_cuda --data data/UTKFace
    python training/export_onnx.py --run runs/expression_cuda --data data/fer2013

Writes to pwa/models/:
    <task>_fp32.onnx   single-file FP32 model (input 'input': 1x3xHxW, RGB in [0, 1])
    <task>_int8.onnx   INT8 model; default: weight-only INT8 (W8A32, see quantize_weights.py).
                       --int8-mode static: full static INT8 (W8A8, QDQ, calibrated on
                       validation images) - kept for the quantisation experiment; it
                       destroys the accuracy of these models.
    <task>.json        model card read by the web app (shapes, labels, thresholds)
"""
from __future__ import annotations

import argparse
import tempfile
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from common import ELDERLY_MIN_AGE, EXPRESSIONS, GENDERS, REPO_ROOT, load_json, save_json
from data import build_datasets
from models import build_model, export_wrapper, make_edge_friendly
from quantize_weights import quantize_weights_int8

OUTPUT_NAMES = {"age_gender": ["age", "p_female"], "expression": ["probs"]}
# Used until training/calibrate_crop.py has measured the real framing.
DEFAULT_CROP = {
    "age_gender": {"scale": 1.5, "shift_x": 0.0, "shift_y": 0.0},
    "expression": {"scale": 1.1, "shift_x": 0.0, "shift_y": 0.0},
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="training run folder containing best.pt")
    p.add_argument("--data", default=None, help="dataset path (only needed for --int8-mode static calibration)")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "pwa" / "models"))
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--calib-samples", type=int, default=300)
    p.add_argument("--no-int8", action="store_true")
    p.add_argument("--int8-mode", choices=["weight", "static"], default="weight",
                   help="weight: INT8 weights, FP32 activations (default); static: INT8 weights + activations")
    p.add_argument("--placeholder", action="store_true",
                   help="mark the models as untrained demo models (the app shows a warning)")
    return p.parse_args()


def export_onnx(model, x, path: Path, output_names, opset):
    """TorchScript exporter first (clean opset-17 graph); dynamo exporter as fallback."""
    common = dict(input_names=["input"], output_names=output_names)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(model, (x,), str(path), opset_version=opset, dynamo=False, **common)
    except Exception as e:  # legacy exporter removed in a future PyTorch release
        print(f"TorchScript exporter unavailable ({type(e).__name__}); using dynamo exporter.")
        torch.onnx.export(model, (x,), str(path), opset_version=max(opset, 18), dynamo=True,
                          external_data=False, **common)
    m = onnx.load(str(path), load_external_data=True)
    onnx.save_model(m, str(path), save_as_external_data=False)  # one self-contained file
    onnx.checker.check_model(str(path))


def ort_session(path) -> ort.InferenceSession:
    """CPU session; for INT8 models the DequantizeLinear nodes are folded into FP32
    weights once at load time (same setting as the web app)."""
    so = ort.SessionOptions()
    # weight-only INT8: fold DequantizeLinear into FP32 weights once at load time
    if "int8" in Path(path).name:
        so.add_session_config_entry("session.disable_quant_qdq", "1")
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


class CalibrationReader:
    """Feeds validation images to the ONNX Runtime static quantizer."""

    def __init__(self, dataset, n, seed=0):
        idx = np.random.default_rng(seed).permutation(len(dataset))[:n]
        self.samples = [dataset[int(i)][0].unsqueeze(0).numpy() for i in idx]
        self.it = iter(self.samples)

    def get_next(self):
        x = next(self.it, None)
        return None if x is None else {"input": x}

    def rewind(self):
        self.it = iter(self.samples)


def quantize_int8(fp32_path: Path, int8_path: Path, reader: CalibrationReader) -> None:
    """Static INT8 quantisation (QDQ, per channel) with MinMax calibration. Only used with
    --int8-mode static; the deployed models use weight-only INT8 (see quantize_weights.py)."""
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    with tempfile.TemporaryDirectory() as tmp:
        pre = Path(tmp) / "pre.onnx"
        quant_pre_process(str(fp32_path), str(pre), skip_symbolic_shape=True)
        quantize_static(
            str(pre), str(int8_path), reader,
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
        )


def main():
    args = parse_args()
    ort.set_default_logger_severity(3)  # hide harmless optimizer warnings
    run = Path(args.run)
    ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    task, size = ckpt["task"], ckpt["img_size"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    net = build_model(task, pretrained=False)
    net.load_state_dict(ckpt["state_dict"])
    reference = export_wrapper(task, net).eval()
    edge_net = build_model(task, pretrained=False)
    edge_net.load_state_dict(ckpt["state_dict"])
    model = export_wrapper(task, make_edge_friendly(edge_net)).eval()

    x = torch.rand(1, 3, size, size)
    with torch.no_grad():
        ref_out = [t.numpy() for t in (reference(x) if task == "age_gender" else (reference(x),))]

    fp32_path = out_dir / f"{task}_fp32.onnx"
    export_onnx(model, x, fp32_path, OUTPUT_NAMES[task], args.opset)
    onnx_out = ort_session(fp32_path).run(None, {"input": x.numpy()})
    parity = {n: float(np.abs(a - b).max()) for n, a, b in zip(OUTPUT_NAMES[task], ref_out, onnx_out, strict=True)}
    print(f"FP32 ONNX -> {fp32_path} ({fp32_path.stat().st_size / 1e6:.2f} MB), max |PyTorch - ONNX| = {parity}")

    files = {"fp32": fp32_path.name}
    sizes = {"fp32_mb": round(fp32_path.stat().st_size / 1e6, 3)}
    if not args.no_int8:
        int8_path = out_dir / f"{task}_int8.onnx"
        if args.int8_mode == "static":
            if not args.data:
                raise SystemExit("--int8-mode static needs --data for calibration images")
            split_dir = Path(ckpt["args"].get("split_dir") or REPO_ROOT / "splits")
            val = build_datasets(task, args.data, size, ckpt["args"]["seed"], split_dir, train_aug=False)["val"]
            quantize_int8(fp32_path, int8_path, CalibrationReader(val, args.calib_samples))
        else:
            quantize_weights_int8(fp32_path, int8_path)
        q_out = ort_session(int8_path).run(None, {"input": x.numpy()})
        q_diff = {n: float(np.abs(a - b).max()) for n, a, b in zip(OUTPUT_NAMES[task], onnx_out, q_out, strict=True)}
        print(f"INT8 ONNX -> {int8_path} ({int8_path.stat().st_size / 1e6:.2f} MB), "
              f"max |FP32 - INT8| on a random input = {q_diff}")
        files["int8"] = int8_path.name
        sizes["int8_mb"] = round(int8_path.stat().st_size / 1e6, 3)
        int8_mode = {"weight": "weight-only INT8 (W8A32)", "static": "static INT8 (W8A8)"}[args.int8_mode]

    card = {
        "task": task,
        "files": files,
        "sizes": sizes,
        "input": {"name": "input", "shape": [1, 3, size, size], "layout": "NCHW", "color": "RGB", "range": [0, 1]},
        "outputs": OUTPUT_NAMES[task],
        "trained_epoch": ckpt["epoch"],
        "parity_max_abs_diff": parity,
    }
    if not args.no_int8:
        card["int8_mode"] = int8_mode
    if task == "age_gender":
        card.update({"genders": GENDERS, "gender_output": "p_female", "elderly_min_age": ELDERLY_MIN_AGE})
    else:
        card.update({"labels": EXPRESSIONS})
    if args.placeholder:
        card["placeholder"] = True

    # Face framing: keep a calibrated crop (training/calibrate_crop.py) if one exists.
    card_path = out_dir / f"{task}.json"
    calib_path = REPO_ROOT / "runs" / "crop_calibration.json"
    if calib_path.exists() and task in load_json(calib_path):
        card["crop"] = {**load_json(calib_path)[task]["crop"], "source": "training/calibrate_crop.py"}
    elif card_path.exists() and "crop" in load_json(card_path):
        card["crop"] = load_json(card_path)["crop"]
    else:
        card["crop"] = {**DEFAULT_CROP[task], "source": "default (run training/calibrate_crop.py)"}
    save_json(card, card_path)
    print(f"Model card -> {out_dir / f'{task}.json'}")


if __name__ == "__main__":
    main()
