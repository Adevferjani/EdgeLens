"""After training: calibrate the crop (if needed), export to ONNX, evaluate, update the app.

    python training/finalize.py --utkface data/UTKFace --fer data/fer2013

Steps (each skipped if its input is missing):
  1. calibrate_crop.py          -> runs/crop_calibration.json (only if it does not exist yet)
  2. export_onnx.py             -> pwa/models/ from runs/<task>_cuda (fallback: runs/<task>_cpu)
  3. evaluate.py --onnx         -> runs/<task>_cuda/evaluation.json (PyTorch vs ONNX FP32/INT8 + PC latency)
     evaluate.py                -> runs/<task>_cpu/evaluation.json (CPU-trained model)
  4. pwa/tools/build_sw.py      -> refresh the offline file list
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from common import REPO_ROOT

PY = sys.executable
TRAIN = REPO_ROOT / "training"


def run(*args) -> None:
    """Run a repository script with the current Python interpreter; stop on the first error."""
    cmd = [PY, *map(str, args)]
    print("\n>>> " + " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--utkface", default="data/UTKFace")
    p.add_argument("--fer", default="data/fer2013")
    p.add_argument("--recalibrate", action="store_true", help="measure the crop again")
    p.add_argument("--age-gender-run", default=None,
                   help="run folder to deploy for age+gender (default: runs/age_gender_cuda)")
    args = p.parse_args()
    data = {"age_gender": args.utkface, "expression": args.fer}
    runs = REPO_ROOT / "runs"

    if args.recalibrate or not (runs / "crop_calibration.json").exists():
        run(TRAIN / "calibrate_crop.py", "--utkface", args.utkface, "--fer", args.fer)

    for task in ("age_gender", "expression"):
        gpu, cpu = runs / f"{task}_cuda", runs / f"{task}_cpu"
        deploy = gpu if (gpu / "best.pt").exists() else cpu if (cpu / "best.pt").exists() else None
        if task == "age_gender" and args.age_gender_run:
            deploy = REPO_ROOT / args.age_gender_run
        if deploy is None:
            print(f"\n!!! no trained {task} model in runs/ - skipped")
            continue
        run(TRAIN / "export_onnx.py", "--run", deploy, "--data", data[task])
        run(TRAIN / "evaluate.py", "--run", deploy, "--data", data[task], "--onnx")
        if deploy != cpu and (cpu / "best.pt").exists():
            run(TRAIN / "evaluate.py", "--run", cpu, "--data", data[task])

    run(REPO_ROOT / "pwa" / "tools" / "build_sw.py")
    print("\nDone. Publish the new models with:\n"
          "  git add runs pwa splits\n"
          '  git commit -m "Trained models and results"\n'
          "  git push")


if __name__ == "__main__":
    main()
