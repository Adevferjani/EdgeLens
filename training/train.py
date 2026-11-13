"""Train the age+gender model (UTKFace) or the expression model (FER2013) on CPU or GPU.

Examples (run from the repository root):
    python training/train.py --task age_gender --data data/UTKFace --device cuda
    python training/train.py --task age_gender --data data/UTKFace --device cpu
    python training/train.py --task expression --data data/fer2013 --device cuda

Outputs go to runs/<task>_<device>/ : best.pt, history.csv, train_summary.json, test_metrics.json
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from common import (
    DEFAULT_IMG_SIZE,
    EXPRESSIONS,
    REPO_ROOT,
    hardware_info,
    portable_path,
    resolve_device,
    save_json,
    set_seed,
)
from data import build_datasets, class_weights
from engine import compute_metrics, predict_torch, selection_score, trivial_age_mae
from metrics import scalar_metrics
from models import AGE_SCALE, build_model, count_parameters

# Arguments that hold paths; written relative to the repository in train_summary.json.
PATH_ARGS = ("data", "out", "split_dir")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=["age_gender", "expression"])
    p.add_argument("--data", required=True, help="UTKFace folder (age_gender) or FER2013 folder/CSV (expression)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--img-size", type=int, default=None, help="default: 160 (age_gender), 112 (expression)")
    p.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    p.add_argument("--threads", type=int, default=None, help="PyTorch CPU threads (default: all cores)")
    p.add_argument("--patience", type=int, default=8, help="early stopping patience (epochs)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-smoothing", type=float, default=0.1,
                   help="expression: cross-entropy smoothing; age_gender: gender targets 0/1 -> eps/2, 1-eps/2")
    p.add_argument("--age-loss-weight", type=float, default=1.0)
    p.add_argument("--balance-ages", action="store_true",
                   help="oversample rare age groups (e.g. 60+) with a weighted sampler over age decades")
    p.add_argument("--balance-power", type=float, default=0.5,
                   help="sample weight = (1 / count of the age decade) ** power; 1 = fully balanced")
    p.add_argument("--no-pretrained", action="store_true", help="train from scratch (no ImageNet weights)")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision on GPU")
    p.add_argument("--max-train-samples", type=int, default=None, help="subsample training set (quick tests)")
    p.add_argument("--out", default=None, help="output folder (default: runs/<task>_<device>)")
    p.add_argument("--split-dir", default=str(REPO_ROOT / "splits"), help="where the fixed UTKFace split is stored")
    return p.parse_args()


def make_scheduler(optimizer, steps_per_epoch: int, epochs: int, warmup_epochs: int = 1):
    """Linear warm-up for one epoch, then cosine decay (updated every step)."""
    total = steps_per_epoch * epochs
    warm = min(steps_per_epoch * warmup_epochs, total)

    def factor(step):
        if step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def compute_loss(task, out, batch, criterion, age_loss_weight, smoothing=0.0) -> torch.Tensor:
    """Age: L1 on age/AGE_SCALE; gender: BCE with label smoothing; expression: `criterion`."""
    if task == "age_gender":
        age_pred, gender_logit = out
        l_age = F.l1_loss(age_pred.float(), batch[1] / AGE_SCALE)
        target = batch[2] * (1 - smoothing) + smoothing / 2  # label smoothing against over-confidence
        l_gender = F.binary_cross_entropy_with_logits(gender_logit.float(), target)
        return age_loss_weight * l_age + l_gender
    return criterion(out.float(), batch[1])


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    size = args.img_size or DEFAULT_IMG_SIZE[args.task]
    out_dir = Path(args.out) if args.out else REPO_ROOT / "runs" / f"{args.task}_{device.type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- data ----------------
    ds = build_datasets(args.task, args.data, size, args.seed, Path(args.split_dir))
    train_ds = ds["train"]
    if args.max_train_samples and args.max_train_samples < len(train_ds):
        idx = np.random.default_rng(args.seed).permutation(len(train_ds))[: args.max_train_samples]
        train_ds = Subset(train_ds, idx.tolist())
    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    sampler = None
    if args.balance_ages:
        if args.task != "age_gender":
            raise SystemExit("--balance-ages is only for --task age_gender")
        recs = ds["train"].records
        if isinstance(train_ds, Subset):
            recs = [recs[i] for i in train_ds.indices]
        bins = np.minimum(np.array([r["age"] for r in recs]) // 10, 9)  # decades, 90+ together
        counts = np.bincount(bins, minlength=10).astype(float)
        sample_w = (1.0 / counts[bins]) ** args.balance_power
        sampler = WeightedRandomSampler(
            torch.tensor(sample_w, dtype=torch.double), num_samples=len(recs), replacement=True
        )
        share = {f"{10 * b}+": round(float(sample_w[bins == b].sum() / sample_w.sum()), 3) for b in range(10)}
        print(f"Age-balanced sampling (power {args.balance_power}): expected share per decade {share}")
    train_loader = DataLoader(train_ds, shuffle=sampler is None, sampler=sampler,
                              drop_last=len(train_ds) > args.batch_size, **loader_kw)
    val_loader = DataLoader(ds["val"], shuffle=False, **loader_kw)
    test_loader = DataLoader(ds["test"], shuffle=False, **loader_kw)

    # ---------------- model ----------------
    model = build_model(args.task, pretrained=not args.no_pretrained).to(device)
    criterion = None
    if args.task == "expression":
        labels = ds["train"].labels
        if isinstance(train_ds, Subset):
            labels = [labels[i] for i in train_ds.indices]
        weights = class_weights(labels, len(EXPRESSIONS)).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    age_mae_ref = None
    if args.task == "age_gender":
        recs = ds["train"].records
        if isinstance(train_ds, Subset):
            recs = [recs[i] for i in train_ds.indices]
        age_mae_ref = trivial_age_mae([r["age"] for r in recs])
        print(f"Reference for model selection: MAE of a median-age predictor = {age_mae_ref:.2f} years")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, len(train_loader), args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    hw = hardware_info(device)
    n_params = count_parameters(model)
    print(f"Task={args.task}  device={device}  amp={use_amp}  img={size}px  params={n_params:,}")
    print(f"Hardware: {hw}")
    print(f"Samples: train={len(train_ds)} val={len(ds['val'])} test={len(ds['test'])}")

    # ---------------- training loop ----------------
    history, best_score, best_epoch, bad_epochs = [], -float("inf"), 0, 0
    train_seconds = 0.0
    wall_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        loss_sum, n_seen = 0.0, 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            batch = [b.to(device, non_blocking=True) for b in batch]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(batch[0])
            loss = compute_loss(args.task, out, batch, criterion, args.age_loss_weight, args.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item() * batch[0].size(0)
            n_seen += batch[0].size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_train_s = time.perf_counter() - t0
        train_seconds += epoch_train_s

        val_pred = predict_torch(args.task, model, val_loader, device, use_amp)
        val_m = compute_metrics(args.task, val_pred)
        score = selection_score(args.task, val_m, age_mae_ref)
        row = {
            "epoch": epoch,
            "train_loss": round(loss_sum / max(1, n_seen), 5),
            "epoch_train_s": round(epoch_train_s, 2),
            "lr": optimizer.param_groups[0]["lr"],
            "val_score": round(score, 5),
            **{f"val_{k}": v for k, v in scalar_metrics(val_m).items()},
        }
        history.append(row)
        if args.task == "age_gender":
            msg = f"val MAE={val_m['age_mae']:.2f}y gender acc={val_m['gender_accuracy']:.3f}"
        else:
            msg = f"val acc={val_m['accuracy']:.3f} macro-F1={val_m['macro_f1']:.3f}"
        improved = score > best_score
        mark = "  *" if improved else ""
        print(f"epoch {epoch:3d}  loss={row['train_loss']:.4f}  {msg}  ({epoch_train_s:.0f}s){mark}")

        if improved:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            torch.save(
                {"task": args.task, "img_size": size, "epoch": epoch, "args": vars(args),
                 "state_dict": model.state_dict()},
                out_dir / "best.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs.")
                break
    wall_seconds = time.perf_counter() - wall_start

    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    # ---------------- test evaluation with the best checkpoint ----------------
    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    test_pred = predict_torch(args.task, model, test_loader, device, use_amp)
    test_m = compute_metrics(args.task, test_pred)
    save_json(test_m, out_dir / "test_metrics.json")

    summary = {
        "task": args.task,
        "hardware": hw,
        "img_size": size,
        "parameters": n_params,
        "samples": {"train": len(train_ds), "val": len(ds["val"]), "test": len(ds["test"])},
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "training_time_s": round(train_seconds, 1),
        "mean_epoch_time_s": round(train_seconds / len(history), 1),
        "wall_time_incl_validation_s": round(wall_seconds, 1),
        "best_val": {k.removeprefix("val_"): v for k, v in history[best_epoch - 1].items() if k.startswith("val_")},
        "selection": ("-(val MAE / median-predictor MAE + gender error / 0.5)" if args.task == "age_gender"
                      else "val macro-F1"),
        "age_mae_reference": None if age_mae_ref is None else round(age_mae_ref, 3),
        "test": scalar_metrics(test_m),
        "args": {k: portable_path(v) if k in PATH_ARGS and v is not None else v for k, v in vars(args).items()},
    }
    save_json(summary, out_dir / "train_summary.json")
    print(f"\nBest epoch {best_epoch}. Training time {train_seconds / 60:.1f} min.")
    print(f"Test metrics: {scalar_metrics(test_m)}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":  # required on Windows for DataLoader workers
    main()
