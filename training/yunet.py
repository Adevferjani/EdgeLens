"""YuNet face detector (Wu et al., 2023) with ONNX Runtime + NumPy decoding.

Same pre-/post-processing as the web app (pwa/js/detector.js). The decoding was
checked against OpenCV's cv2.FaceDetectorYN on a synthetic scene: identical
boxes, landmarks and scores (max. abs. difference 2e-5 px).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnxruntime as ort

from common import REPO_ROOT

DEFAULT_MODEL = REPO_ROOT / "pwa" / "models" / "face_detection_yunet_2026may.onnx"
STRIDES = (8, 16, 32)


def _nms(boxes, scores, thr):
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 0] + boxes[i, 2], boxes[rest, 0] + boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 1] + boxes[i, 3], boxes[rest, 1] + boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (boxes[i, 2] * boxes[i, 3] + boxes[rest, 2] * boxes[rest, 3] - inter)
        order = rest[iou <= thr]
    return keep


class YuNet:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL, score_threshold=0.6, nms_threshold=0.3):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self.names = [o.name for o in self.sess.get_outputs()]
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

    def detect(self, bgr: np.ndarray) -> np.ndarray:
        """bgr: HxWx3 uint8. Returns N x 15: x, y, w, h, 5 landmarks (x, y), score."""
        h, w = bgr.shape[:2]
        ph, pw = math.ceil(h / 32) * 32, math.ceil(w / 32) * 32
        blob = np.zeros((1, 3, ph, pw), np.float32)
        blob[0, :, :h, :w] = bgr.transpose(2, 0, 1)
        out = dict(zip(self.names, self.sess.run(None, {"input": blob}), strict=True))
        boxes, scores, kps = [], [], []
        for s in STRIDES:
            cols = pw // s
            cls, obj = out[f"cls_{s}"][0, :, 0], out[f"obj_{s}"][0, :, 0]
            bb, kp = out[f"bbox_{s}"][0], out[f"kps_{s}"][0]
            sc = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
            for idx in np.nonzero(sc >= self.score_threshold)[0]:
                r, c = divmod(int(idx), cols)
                cx, cy = (c + bb[idx, 0]) * s, (r + bb[idx, 1]) * s
                bw, bh = math.exp(bb[idx, 2]) * s, math.exp(bb[idx, 3]) * s
                boxes.append([cx - bw / 2, cy - bh / 2, bw, bh])
                scores.append(sc[idx])
                kps.append([v for n in range(5) for v in ((kp[idx, 2 * n] + c) * s, (kp[idx, 2 * n + 1] + r) * s)])
        if not boxes:
            return np.zeros((0, 15), np.float32)
        boxes, scores, kps = np.array(boxes), np.array(scores), np.array(kps)
        keep = _nms(boxes, scores, self.nms_threshold)
        return np.hstack([boxes[keep], kps[keep], scores[keep, None]]).astype(np.float32)

    def largest_face(self, bgr: np.ndarray) -> np.ndarray | None:
        """Detection with the largest box, or None if no face was found."""
        dets = self.detect(bgr)
        if not len(dets):
            return None
        return dets[np.argmax(dets[:, 2] * dets[:, 3])]


def eye_angle(face) -> float:
    """Angle (radians) of the line from landmark 0 (image-left eye) to landmark 1."""
    return math.atan2(face[7] - face[5], face[6] - face[4])
