"""Evaluation metrics for both tasks (all values JSON-serialisable)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from common import ADULT_MIN_AGE, EXPRESSIONS, HAPPY, SAD, age_group


def _f(x):
    return None if x is None else round(float(x), 4)


def age_gender_metrics(age_true, age_pred, gender_true, p_female) -> dict:
    """Age MAE (overall, 18+, per group), gender accuracy/F1 and the adult vs. elderly
    classification derived from the predicted age (threshold ELDERLY_MIN_AGE)."""
    age_true, age_pred = np.asarray(age_true, float), np.asarray(age_pred, float)
    g_true = np.asarray(gender_true).astype(int)
    g_pred = (np.asarray(p_female) >= 0.5).astype(int)
    err = np.abs(age_pred - age_true)
    adult = age_true >= ADULT_MIN_AGE

    m = {
        "n": int(len(age_true)),
        "age_mae": _f(err.mean()),
        "gender_accuracy": _f((g_true == g_pred).mean()),
        "gender_macro_f1": _f(f1_score(g_true, g_pred, average="macro", labels=[0, 1], zero_division=0)),
        "gender_confusion": confusion_matrix(g_true, g_pred, labels=[0, 1]).tolist(),  # rows: true male/female
        "n_18plus": int(adult.sum()),
        "age_mae_18plus": None,
        "age_group_accuracy_18plus": None,
        "age_group_confusion_18plus": None,  # rows: true adult/elderly
        "adult_recall_18plus": None,
        "elderly_recall_18plus": None,
        "age_group_balanced_accuracy_18plus": None,
        "age_group_macro_f1_18plus": None,
        "age_mae_elderly": None,
    }
    if adult.any():
        gt, gp = age_group(age_true[adult]), age_group(age_pred[adult])
        m["age_mae_18plus"] = _f(err[adult].mean())
        m["age_group_accuracy_18plus"] = _f((gt == gp).mean())
        m["age_group_confusion_18plus"] = confusion_matrix(gt, gp, labels=[0, 1]).tolist()
        rec = [(gp[gt == c] == c).mean() if (gt == c).any() else np.nan for c in (0, 1)]
        m["adult_recall_18plus"], m["elderly_recall_18plus"] = _f(rec[0]), _f(rec[1])
        m["age_group_balanced_accuracy_18plus"] = _f(np.nanmean(rec))
        m["age_group_macro_f1_18plus"] = _f(f1_score(gt, gp, average="macro", labels=[0, 1], zero_division=0))
        elderly = age_group(age_true) == 1
        if elderly.any():
            m["age_mae_elderly"] = _f(err[elderly].mean())
    return m


def expression_metrics(y_true, probs) -> dict:
    """Seven-class accuracy, macro-F1, per-class F1 and confusion matrix, plus the
    happy-vs-sad accuracy on happy/sad samples (7-way argmax and forced 2-way choice)."""
    y_true, probs = np.asarray(y_true).astype(int), np.asarray(probs, float)
    y_pred = probs.argmax(1)
    labels = list(range(len(EXPRESSIONS)))
    per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

    hs = np.isin(y_true, [HAPPY, SAD])
    two_way = np.where(probs[:, HAPPY] >= probs[:, SAD], HAPPY, SAD)
    return {
        "n": int(len(y_true)),
        "accuracy": _f((y_true == y_pred).mean()),
        "macro_f1": _f(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "per_class_f1": {c: _f(v) for c, v in zip(EXPRESSIONS, per_class, strict=True)},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "happy_sad_n": int(hs.sum()),
        # 7-way argmax must be exactly 'happy' or 'sad':
        "happy_sad_accuracy_7way": _f((y_pred[hs] == y_true[hs]).mean()) if hs.any() else None,
        # restricted decision between the two classes (as used in the 20-image test):
        "happy_sad_accuracy_2way": _f((two_way[hs] == y_true[hs]).mean()) if hs.any() else None,
    }


def scalar_metrics(m: dict) -> dict:
    """Keep only scalar values (for CSV logs and one-line summaries)."""
    return {k: v for k, v in m.items() if isinstance(v, (int, float)) or v is None}
