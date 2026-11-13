"""Pack the test sets into one file for the on-device accuracy check.

    python training/tools/export_testset.py                       # full test sets (~20 MB)
    python training/tools/export_testset.py --max-utk 500 --max-fer 500   # quick subset

Writes testset.zip (uncompressed ZIP) with
  utk/<file>.jpg            UTKFace test split (original files, splits/utkface_split.csv)
  fer/<index>_<label>.png   FER2013 PrivateTest (lossless 48x48 grayscale)
  manifest.json             true labels, the PC predictions of the deployed ONNX models
                            (ONNX Runtime CPU, FP32 and INT8) for every image, PC metrics,
                            and SHA-256 hashes of the model files.

Open the file in the app (Benchmark tab -> "Accuracy on the test sets"). The phone runs the
same models on the same images and compares its metrics and its per-image predictions with
the PC. Faces are already cropped, so no face detection is involved.

The ZIP contains dataset images (UTKFace: non-commercial research use only): keep it
private, do not commit or publish it (testset*.zip is in .gitignore).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ELDERLY_MIN_AGE, EXPRESSIONS, REPO_ROOT, cpu_name  # noqa: E402
from data import build_transform, fer2013_splits, utkface_splits  # noqa: E402
from metrics import age_gender_metrics, expression_metrics, scalar_metrics  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--utkface", default=str(REPO_ROOT / "data" / "UTKFace"))
    p.add_argument("--fer", default=str(REPO_ROOT / "data" / "fer2013"))
    p.add_argument("--split-dir", default=str(REPO_ROOT / "splits"))
    p.add_argument("--models-dir", default=str(REPO_ROOT / "pwa" / "models"))
    p.add_argument("--out", default=str(REPO_ROOT / "testset.zip"))
    p.add_argument("--max-utk", type=int, default=0, help="random subset size (0 = whole test split)")
    p.add_argument("--max-fer", type=int, default=0, help="random subset size (0 = whole test split)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def subset(items, k, seed):
    if not k or k >= len(items):
        return items
    idx = np.sort(np.random.default_rng(seed).choice(len(items), k, replace=False))
    return [items[i] for i in idx]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def session(path: Path):
    so = ort.SessionOptions()
    if "int8" in path.name:  # weight-only INT8: dequantise once at load time (as in the app)
        so.add_session_config_entry("session.disable_quant_qdq", "1")
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def main():
    args = parse_args()
    ort.set_default_logger_severity(3)
    models = Path(args.models_dir)
    cards = {t: json.loads((models / f"{t}.json").read_text(encoding="utf-8")) for t in ("age_gender", "expression")}
    sizes = {t: cards[t]["input"]["shape"][2] for t in cards}
    variants = [v for v in ("fp32", "int8") if (models / cards["age_gender"]["files"][v]).exists()]

    utk_root = Path(args.utkface)
    utk = subset(utkface_splits(utk_root, Path(args.split_dir) / "utkface_split.csv")["test"], args.max_utk, args.seed)
    fer = subset(fer2013_splits(args.fer)["test"], args.max_fer, args.seed)
    print(f"UTKFace test images: {len(utk)}, FER2013 test images: {len(fer)}")

    # ---- encode the files exactly as they will be stored ----
    utk_files, fer_files = [], []
    for r in utk:
        utk_files.append((f"utk/{r['file']}", (utk_root / r["file"]).read_bytes()))
    for i, (src, label) in enumerate(fer):
        img = Image.fromarray(src) if isinstance(src, np.ndarray) else Image.open(src)
        buf = io.BytesIO()
        img.convert("L").save(buf, format="PNG", optimize=True)
        fer_files.append((f"fer/{i:05d}_{label}.png", buf.getvalue()))

    # ---- PC reference: the deployed ONNX models on exactly these files ----
    tf_ag = build_transform("age_gender", sizes["age_gender"], train=False)
    tf_ex = build_transform("expression", sizes["expression"], train=False)

    def tensors(files: list, transform, mode: str) -> np.ndarray | None:
        if not files:
            return None
        return np.stack([transform(Image.open(io.BytesIO(b)).convert(mode).convert("RGB")).numpy() for _, b in files])

    x_ag = tensors(utk_files, tf_ag, "RGB")
    x_ex = tensors(fer_files, tf_ex, "L")  # FER2013 is grayscale; the app converts to RGB the same way

    ag_pc, ag_metrics, ex_pc, ex_metrics, latency = {}, {}, {}, {}, {}
    age_true = [r["age"] for r in utk]
    gender_true = [r["gender"] for r in utk]
    y_true = [label for _, label in fer]
    for v in variants:
        t0 = time.perf_counter()
        if x_ag is not None:
            s = session(models / cards["age_gender"]["files"][v])
            outs = [s.run(None, {"input": x_ag[i : i + 1]}) for i in range(len(x_ag))]
            age = [float(o[0].reshape(-1)[0]) for o in outs]
            pf = [float(o[1].reshape(-1)[0]) for o in outs]
            ag_pc[v] = {"age": [round(a, 5) for a in age], "p_female": [round(p, 7) for p in pf]}
            ag_metrics[v] = scalar_metrics(age_gender_metrics(age_true, age, gender_true, pf))
        if x_ex is not None:
            s = session(models / cards["expression"]["files"][v])
            probs = np.stack([s.run(None, {"input": x_ex[i : i + 1]})[0].reshape(-1) for i in range(len(x_ex))])
            ex_pc[v] = {"probs": np.round(probs, 7).tolist()}
            m = expression_metrics(y_true, probs)
            ex_metrics[v] = {**scalar_metrics(m), "per_class_f1": m["per_class_f1"]}
        latency[v] = round(time.perf_counter() - t0, 1)
        print(f"PC reference {v.upper()}: {latency[v]} s")

    manifest = {
        "format": "agescope-testset-1",
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "pc": {"cpu": cpu_name(), "os": platform.platform(), "onnxruntime": ort.__version__,
               "note": "ONNX Runtime CPU on the PC, PIL decoding and resizing (training transform)"},
        "models": {
            t: {v: {"file": cards[t]["files"][v], "sha256": sha256(models / cards[t]["files"][v])} for v in variants}
            for t in cards
        },
        "age_gender": {
            "img_size": sizes["age_gender"], "n": len(utk), "subset": bool(args.max_utk), "seed": args.seed,
            "elderly_min_age": ELDERLY_MIN_AGE,
            "items": [
                {"file": f, "age": r["age"], "gender": r["gender"]} for (f, _), r in zip(utk_files, utk, strict=True)
            ],
            "pc": ag_pc, "pc_metrics": ag_metrics,
        },
        "expression": {
            "img_size": sizes["expression"], "n": len(fer), "subset": bool(args.max_fer), "seed": args.seed,
            "labels": EXPRESSIONS,
            "items": [{"file": f, "label": label} for (f, _), (_, label) in zip(fer_files, fer, strict=True)],
            "pc": ex_pc, "pc_metrics": ex_metrics,
        },
    }

    out = Path(args.out)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("manifest.json", json.dumps(manifest, separators=(",", ":")))
        for name, data in utk_files + fer_files:
            z.writestr(name, data)
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
    for v in variants:
        a, e = ag_metrics.get(v, {}), ex_metrics.get(v, {})
        print(f"  PC {v.upper()}: age MAE {a.get('age_mae')}, gender acc {a.get('gender_accuracy')}, "
              f"expression acc {e.get('accuracy')}, macro-F1 {e.get('macro_f1')}")
    print("Keep this file private (dataset licence). Copy it to the phone, e.g. via iCloud Drive or OneDrive.")


if __name__ == "__main__":
    main()
