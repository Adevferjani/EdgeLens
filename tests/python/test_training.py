"""Fast checks of the training code (no datasets, no GPU; about 10 s on a laptop CPU).

Run from the repository root:  python -m pytest tests/python -q
"""
import csv
from collections import Counter

import numpy as np
import pytest
import torch

from common import ELDERLY_MIN_AGE, EXPRESSIONS, HAPPY, REPO_ROOT, SAD, portable_path
from metrics import age_gender_metrics, expression_metrics
from models import build_model, export_wrapper


@pytest.mark.parametrize(("task", "size"), [("age_gender", 160), ("expression", 112)])
def test_networks_and_export_wrappers(task, size):
    torch.manual_seed(0)
    net = build_model(task, pretrained=False).eval()
    x = torch.rand(2, 3, size, size)
    with torch.no_grad():
        out = export_wrapper(task, net)(x)
    if task == "age_gender":
        age, p_female = out
        assert age.shape == (2,) and p_female.shape == (2,)
        assert torch.all((age >= 0) & (age <= 116))
        assert torch.all((p_female >= 0) & (p_female <= 1))
    else:
        assert out.shape == (2, len(EXPRESSIONS))
        assert torch.allclose(out.sum(1), torch.ones(2), atol=1e-5)


def test_age_gender_metrics():
    m = age_gender_metrics(
        age_true=[20, 30, 65, 70],
        age_pred=[22, 30, 55, 72],
        gender_true=[0, 1, 0, 1],
        p_female=[0.1, 0.9, 0.7, 0.8],
    )
    assert m["age_mae"] == pytest.approx(3.5)
    assert m["gender_accuracy"] == pytest.approx(0.75)
    # adult / 60+ is derived from the predicted age; 65 -> 55 is the only group error
    assert m["age_group_accuracy_18plus"] == pytest.approx(0.75)
    assert ELDERLY_MIN_AGE == 60


def test_expression_metrics_forced_choice():
    probs = np.full((3, len(EXPRESSIONS)), 0.01)
    probs[0, HAPPY] = 0.9  # happy, correct
    probs[1, EXPRESSIONS.index("neutral")] = 0.9  # sad, top class neutral ...
    probs[1, SAD] = 0.05  # ... but sad > happy in the forced choice
    probs[2, SAD] = 0.9  # sad, correct
    m = expression_metrics([HAPPY, SAD, SAD], probs)
    assert m["accuracy"] == pytest.approx(2 / 3, abs=1e-4)  # metrics are rounded to 4 decimals
    assert m["happy_sad_accuracy_2way"] == pytest.approx(1.0)


def test_committed_utkface_split():
    """The fixed split used for all reported results: 80/10/10, no image in two splits."""
    with open(REPO_ROOT / "splits" / "utkface_split.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    counts = Counter(r["split"] for r in rows)
    assert counts == {"train": 18966, "val": 2371, "test": 2371}
    assert len({r["file"] for r in rows}) == len(rows)


def test_result_files_get_repository_relative_paths():
    assert portable_path(REPO_ROOT / "runs" / "age_gender_cuda") == "runs/age_gender_cuda"
    assert portable_path(REPO_ROOT.parent / "elsewhere" / "UTKFace") == "UTKFace"  # no local folder names
