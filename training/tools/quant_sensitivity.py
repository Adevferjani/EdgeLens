"""Why does static INT8 fail? Simulated quantisation of a trained model in PyTorch.

    python training/tools/quant_sensitivity.py --run runs/age_gender_cuda --data data/UTKFace
    python training/tools/quant_sensitivity.py --run runs/expression_cuda --data data/fer2013

On validation images (first half: ranges, second half: evaluation) it reports:
  1. FP32 accuracy
  2. weights quantised to INT8 (per channel / per tensor), activations FP32
  3. all activations quantised to UINT8 (min/max per tensor), weights FP32
  4. sensitivity: accuracy when only ONE layer's output is quantised
  5. mixed precision: all activations quantised except the k most sensitive ones
Results -> <run>/quant_sensitivity.json
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REPO_ROOT, save_json  # noqa: E402
from data import build_datasets  # noqa: E402
from models import AGE_SCALE, EdgeHardswish, build_model  # noqa: E402

torch.set_grad_enabled(False)
QUANT_MODULES = (nn.Conv2d, nn.Linear, nn.ReLU, nn.Hardswish, nn.Hardsigmoid, EdgeHardswish)


def fold_bn(model):
    m = copy.deepcopy(model)

    def rec(mod):
        names = list(mod._modules)
        for i, n in enumerate(names):
            c = mod._modules[n]
            next_is_bn = i + 1 < len(names) and isinstance(mod._modules[names[i + 1]], nn.BatchNorm2d)
            if isinstance(c, nn.Conv2d) and next_is_bn:
                mod._modules[n] = fuse_conv_bn_eval(c, mod._modules[names[i + 1]])
                mod._modules[names[i + 1]] = nn.Identity()
            else:
                rec(c)

    rec(m)
    return m


def quant_weights(model, per_channel=True):
    m = copy.deepcopy(model)
    for mod in m.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            w = mod.weight.data
            mx = w.abs().amax(dim=tuple(range(1, w.dim())), keepdim=True) if per_channel else w.abs().max()
            s = mx.clamp(min=1e-12) / 127
            mod.weight.data = (w / s).round().clamp(-127, 127) * s
    return m


def quant_acts(model, ranges, skip=()):
    m = copy.deepcopy(model)
    for name, mod in m.named_modules():
        if name in ranges and name not in skip:
            lo, hi = min(ranges[name][0], 0.0), max(ranges[name][1], 0.0)
            s = (hi - lo) / 255 or 1e-8
            z = round(-lo / s)
            mod.register_forward_hook(lambda _m, _i, o, s=s, z=z: ((o / s).round() + z).clamp(0, 255).sub(z).mul(s))
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--n", type=int, default=400, help="validation images (half for ranges, half for evaluation)")
    args = p.parse_args()

    ckpt = torch.load(Path(args.run) / "best.pt", map_location="cpu", weights_only=True)
    task, size = ckpt["task"], ckpt["img_size"]
    split_dir = Path(ckpt["args"].get("split_dir") or REPO_ROOT / "splits")
    val = build_datasets(task, args.data, size, ckpt["args"]["seed"], split_dir, train_aug=False)["val"]
    idx = np.random.default_rng(0).permutation(len(val))[: args.n]
    batch = [val[int(i)] for i in idx]
    X = torch.stack([b[0] for b in batch])
    calib, Xe = X[: len(X) // 2], X[len(X) // 2:]
    if task == "age_gender":
        ages = np.array([float(b[1]) for b in batch])[len(X) // 2:]
        genders = np.array([float(b[2]) for b in batch])[len(X) // 2:]
    else:
        labels = np.array([int(b[1]) for b in batch])[len(X) // 2:]

    def score(m):
        if task == "age_gender":
            a, g = m(Xe)
            return {"age_mae": round(float(np.abs(a.numpy() * AGE_SCALE - ages).mean()), 3),
                    "gender_acc": round(float(((torch.sigmoid(g).numpy() >= 0.5) == genders).mean()), 4)}
        return {"accuracy": round(float((m(Xe).argmax(1).numpy() == labels).mean()), 4)}

    key = "age_mae" if task == "age_gender" else "accuracy"


    net = build_model(task, pretrained=False)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    model = fold_bn(net)
    acts = {}
    hooks = [mod.register_forward_hook(lambda _m, _i, o, n=name: acts.__setitem__(n, o.flatten().numpy()))
             for name, mod in model.named_modules() if isinstance(mod, QUANT_MODULES)]
    model(calib)
    for h in hooks:
        h.remove()
    ranges = {n: (float(a.min()), float(a.max())) for n, a in acts.items()}

    res = {"task": task, "n_eval": len(Xe), "fp32": score(model),
           "weights_int8_per_channel": score(quant_weights(model)),
           "weights_int8_per_tensor": score(quant_weights(model, False)),
           "activations_uint8_all": score(quant_acts(model, ranges))}
    sens = sorted(((score(quant_acts(model, {n: ranges[n]}))[key], n) for n in ranges),
                  key=lambda t: t[0], reverse=task == "age_gender")
    res["single_layer_sensitivity"] = [{"layer": n, key: v} for v, n in sens[:10]]
    res["widest_activation_ranges"] = [
        {"layer": n, "min": round(lo, 2), "max": round(hi, 2),
         "p0.01": round(float(np.percentile(acts[n], 0.01)), 2),
         "p99.99": round(float(np.percentile(acts[n], 99.99)), 2)}
        for n, (lo, hi) in sorted(ranges.items(), key=lambda kv: kv[1][1] - kv[1][0], reverse=True)[:6]]
    order = [n for _, n in sens]
    res["mixed_precision"] = [
        {"fp32_layers": k, **score(quant_acts(model, ranges, skip=set(order[:k])))} for k in (1, 2, 4, 6, 10)
    ]

    save_json(res, Path(args.run) / "quant_sensitivity.json")
    print(f"\n{task}: evaluation on {len(Xe)} validation images ({key})")
    for k in ("fp32", "weights_int8_per_channel", "weights_int8_per_tensor", "activations_uint8_all"):
        print(f"  {k:28s} {res[k]}")
    print("  most sensitive single layers:", [(d["layer"], d[key]) for d in res["single_layer_sensitivity"][:5]])
    widest = [(d["layer"], d["min"], d["max"], d["p99.99"]) for d in res["widest_activation_ranges"][:3]]
    print("  widest ranges:", widest)
    for d in res["mixed_precision"]:
        print(f"  all activations UINT8 except {d['fp32_layers']:2d} most sensitive: {d}")
    print(f"saved {Path(args.run) / 'quant_sensitivity.json'}")



if __name__ == "__main__":
    main()
