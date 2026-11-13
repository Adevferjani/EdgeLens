"""Measure how training faces are framed, so the app crops camera faces the same way.

Training images are already cropped faces (UTKFace 200x200, FER2013 48x48). The
web app instead gets a face box from the YuNet detector. To avoid a train/deploy
mismatch, this script runs the same detector on training images and measures,
relative to the detector box (base = max(w, h)):

    scale   = image side / base
    shift_x = horizontal offset of the image centre from the box centre / base
    shift_y = vertical offset (in the eye-aligned frame)

The medians are written to runs/crop_calibration.json and into the model cards
pwa/models/<task>.json (field "crop") read by the app.

    python training/calibrate_crop.py --utkface data/UTKFace --fer data/fer2013/fer2013.csv
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from common import REPO_ROOT, load_json, save_json
from data import fer2013_splits, utkface_splits
from yunet import YuNet, eye_angle

PAD = 32  # replicated border so faces touching the image edge are still found


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--utkface", help="UTKFace folder")
    p.add_argument("--fer", help="FER2013 folder or fer2013.csv")
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-dir", default=str(REPO_ROOT / "splits"))
    p.add_argument("--models-dir", default=str(REPO_ROOT / "pwa" / "models"))
    return p.parse_args()


def measure(det: YuNet, images, upscale_to: int | None) -> tuple:
    """Run YuNet on dataset images and measure how the dataset crop relates to the detector box
    (scale and centre offset), so the app can reproduce the training crop from a detection."""
    rows, missed = [], 0
    for img in images:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if upscale_to and img.shape[0] != upscale_to:
            img = cv2.resize(img, (upscale_to, upscale_to), interpolation=cv2.INTER_CUBIC)
        side = img.shape[0]
        padded = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
        face = det.largest_face(padded)
        if face is None:
            missed += 1
            continue
        x, y, w, h = face[:4] - np.array([PAD, PAD, 0, 0])
        base = max(w, h)
        a = eye_angle(face)
        dx, dy = side / 2 - (x + w / 2), side / 2 - (y + h / 2)
        # express the offset in the eye-aligned frame (rotate by -angle)
        sx = math.cos(a) * dx + math.sin(a) * dy
        sy = -math.sin(a) * dx + math.cos(a) * dy
        rows.append((side / base, sx / base, sy / base, math.degrees(a), base / side))
    return np.array(rows), missed


def summarise(rows, missed) -> dict:
    """Quartiles of the measured crop parameters; the medians become the crop used by the app."""
    if len(rows) == 0:
        raise SystemExit(f"No face found in any of the {missed} sampled images - check the dataset path.")
    def q(col: int) -> list:
        return [round(float(np.percentile(rows[:, col], p)), 4) for p in (25, 50, 75)]

    return {
        "n_measured": int(len(rows)),
        "n_no_face": int(missed),
        "detection_rate": round(len(rows) / (len(rows) + missed), 4),
        "scale_q25_q50_q75": q(0),
        "shift_x_q25_q50_q75": q(1),
        "shift_y_q25_q50_q75": q(2),
        "eye_angle_deg_q25_q50_q75": q(3),
        "box_to_image_ratio_q25_q50_q75": q(4),
        "crop": {
            "scale": round(float(np.median(rows[:, 0])), 4),
            "shift_x": round(float(np.median(rows[:, 1])), 4),
            "shift_y": round(float(np.median(rows[:, 2])), 4),
        },
    }


def sample(items, n, seed) -> list:
    """Reproducible random subset of `n` items."""
    idx = np.random.default_rng(seed).permutation(len(items))[:n]
    return [items[i] for i in idx]


def main():
    args = parse_args()
    det = YuNet(score_threshold=0.5)
    out_path = REPO_ROOT / "runs" / "crop_calibration.json"
    results = load_json(out_path) if out_path.exists() else {}

    if args.utkface:
        recs = sample(utkface_splits(args.utkface, Path(args.split_dir) / "utkface_split.csv", args.seed)["train"],
                      args.samples, args.seed)
        imgs = (cv2.imread(str(Path(args.utkface) / r["file"])) for r in recs)
        rows, missed = measure(det, imgs, upscale_to=None)
        results["age_gender"] = summarise(rows, missed)
        save_json(results, out_path)  # keep this result even if the FER2013 step fails
        print("age_gender:", json.dumps(results["age_gender"], indent=2))
    if args.fer:
        items = sample(fer2013_splits(args.fer, args.seed)["train"], args.samples, args.seed)
        imgs = (src if isinstance(src, np.ndarray) else cv2.imread(src, cv2.IMREAD_GRAYSCALE) for src, _ in items)
        rows, missed = measure(det, imgs, upscale_to=192)
        results["expression"] = summarise(rows, missed)
        save_json(results, out_path)
        print("expression:", json.dumps(results["expression"], indent=2))
    for task, res in results.items():
        card_path = Path(args.models_dir) / f"{task}.json"
        if card_path.exists():
            card = load_json(card_path)
            card["crop"] = {**res["crop"], "source": "training/calibrate_crop.py"}
            save_json(card, card_path)
            print(f"updated {card_path}")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
