"""Create a tiny fake UTKFace + FER2013 dataset to smoke-test the pipeline
(no real faces; random images with valid file names and folder structure).

    python training/tools/make_dummy_data.py            -> data_dummy/
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

EXPRESSIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data_dummy")
    p.add_argument("--n", type=int, default=200)
    args = p.parse_args()
    rng = np.random.default_rng(0)
    out = Path(args.out)

    utk = out / "UTKFace"
    utk.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        age, gender = int(rng.integers(1, 90)), int(rng.integers(0, 2))
        img = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
        Image.fromarray(img).save(utk / f"{age}_{gender}_0_2017{i:013d}.jpg.chip.jpg")
    (utk / "broken_name.jpg").write_bytes(b"")  # must be skipped by the parser

    fer = out / "fer2013"
    for split, n in (("train", args.n), ("test", args.n // 4)):
        for i in range(n):
            cls = EXPRESSIONS[i % len(EXPRESSIONS)]
            d = fer / split / cls
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 255, (48, 48), dtype=np.uint8)).save(d / f"{i}.png")

    with open(out / "fer2013.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["emotion", "pixels", "Usage"])
        for i in range(args.n):
            usage = ["Training"] * 8 + ["PublicTest", "PrivateTest"]
            w.writerow([i % 7, " ".join(map(str, rng.integers(0, 255, 48 * 48))), usage[i % 10]])
    print(f"Dummy data written to {out.resolve()}")


if __name__ == "__main__":
    main()
