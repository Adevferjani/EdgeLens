"""Checks of the deployed ONNX models in pwa/models (the files the web app loads)."""
import json

import numpy as np
import onnxruntime as ort
import pytest

from common import REPO_ROOT

MODELS = REPO_ROOT / "pwa" / "models"


def session(path):
    so = ort.SessionOptions()
    if "int8" in path.name:  # weight-only INT8: dequantise once at load time, as in the app
        so.add_session_config_entry("session.disable_quant_qdq", "1")
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


@pytest.mark.parametrize("task", ["age_gender", "expression"])
def test_model_card_matches_files(task):
    card = json.loads((MODELS / f"{task}.json").read_text(encoding="utf-8"))
    assert card["task"] == task
    for variant in ("fp32", "int8"):
        path = MODELS / card["files"][variant]
        assert path.exists(), path
        sess = session(path)
        assert sess.get_inputs()[0].name == card["input"]["name"]
        assert [o.name for o in sess.get_outputs()] == card["outputs"]


@pytest.mark.parametrize("task", ["age_gender", "expression"])
def test_int8_close_to_fp32(task):
    card = json.loads((MODELS / f"{task}.json").read_text(encoding="utf-8"))
    x = np.random.default_rng(0).random(card["input"]["shape"], dtype=np.float32)
    fp32 = session(MODELS / card["files"]["fp32"]).run(None, {"input": x})
    int8 = session(MODELS / card["files"]["int8"]).run(None, {"input": x})
    if task == "age_gender":
        assert abs(float(fp32[0][0]) - float(int8[0][0])) < 3.0  # years
        assert abs(float(fp32[1][0]) - float(int8[1][0])) < 0.1  # p_female
    else:
        assert np.allclose(fp32[0].sum(), 1.0, atol=1e-4)
        assert np.abs(fp32[0] - int8[0]).max() < 0.1


def test_detector_loads():
    card = json.loads((MODELS / "detector.json").read_text(encoding="utf-8"))
    files = card.get("files") or {"fp32": card.get("file")}
    path = MODELS / next(iter(files.values()))
    assert path.exists() and session(path).get_inputs()
