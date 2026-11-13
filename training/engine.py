"""Prediction loops shared by training and evaluation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from metrics import age_gender_metrics, expression_metrics
from models import AGE_SCALE


@torch.no_grad()
def predict_torch(task: str, model, loader, device, amp: bool = False) -> dict:
    """Run the model over a loader and collect predictions and labels as NumPy arrays
    (age in years, p_female, or softmax probabilities)."""
    model.eval()
    acc = {}
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(x)
        if task == "age_gender":
            age, g_logit = out
            acc.setdefault("age_pred", []).append((age.float() * AGE_SCALE).cpu())
            acc.setdefault("p_female", []).append(torch.sigmoid(g_logit.float()).cpu())
            acc.setdefault("age_true", []).append(batch[1])
            acc.setdefault("gender_true", []).append(batch[2])
        else:
            acc.setdefault("probs", []).append(F.softmax(out.float(), dim=1).cpu())
            acc.setdefault("y_true", []).append(batch[1])
    return {k: torch.cat(v).numpy() for k, v in acc.items()}


def compute_metrics(task: str, pred: dict) -> dict:
    """Task-specific metrics for the output of `predict_torch`."""
    if task == "age_gender":
        return age_gender_metrics(pred["age_true"], pred["age_pred"], pred["gender_true"], pred["p_female"])
    return expression_metrics(pred["y_true"], pred["probs"])


def trivial_age_mae(train_ages) -> float:
    """MAE of always predicting the median training age (reference for model selection)."""
    a = np.asarray(train_ages, float)
    return float(np.abs(a - np.median(a)).mean())


def selection_score(task: str, metrics: dict, age_mae_ref: float | None = None) -> float:
    """Higher is better.

    Age+gender: -(age MAE / MAE of a median predictor + gender error / 0.5), i.e.
    each head's error relative to a trivial predictor, so both count equally.
    (The validation loss was dominated by the gender cross-entropy, which rises
    once the gender head becomes over-confident, while age was still improving.)
    Expression: macro-F1 (robust to class imbalance).
    """
    if task == "age_gender":
        return float(-(metrics["age_mae"] / age_mae_ref + (1 - metrics["gender_accuracy"]) / 0.5))
    return float(metrics["macro_f1"])
