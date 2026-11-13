"""MobileNetV3-Small networks for age + gender and for facial expression.

Design choices:
* MobileNetV3-Small (Howard et al., 2019) is built for mobile CPUs/GPUs:
  ~1 M parameters, < 5 MB as FP32.
* ImageNet pre-training + fine-tuning (transfer learning) reduces training time.
* Input normalisation is part of the network, so the web app only has to crop the
  face, resize it and scale pixels to [0, 1].
* The expression network converts its input to grayscale internally, because
  FER2013 is grayscale. Camera images are therefore processed the same way as the
  training data (smaller domain gap).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from common import EXPRESSIONS, MAX_AGE

AGE_SCALE = 100.0  # age targets are divided by this during training
FEATURE_DIM = 576  # channels of the last MobileNetV3-Small feature map
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InputNorm(nn.Module):
    """RGB input in [0, 1], shape (N, 3, H, W) -> ImageNet-normalised input."""

    def __init__(self, grayscale: bool = False):
        super().__init__()
        self.grayscale = grayscale
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.grayscale:  # ITU-R BT.601 luma, replicated to 3 channels
            g = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            x = torch.cat([g, g, g], dim=1)
        return (x - self.mean) / self.std


def mobilenet_features(pretrained: bool) -> nn.Module:
    """Convolutional part of MobileNetV3-Small, optionally with ImageNet weights."""
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    return mobilenet_v3_small(weights=weights).features


class AgeGenderNet(nn.Module):
    """Multi-task network: shared backbone, age regression head and gender head.

    Returns (age / AGE_SCALE, gender logit); gender logit > 0 means female.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.norm = InputNorm()
        self.features = mobilenet_features(pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.neck = nn.Sequential(
            nn.Flatten(), nn.Linear(FEATURE_DIM, 256), nn.Hardswish(), nn.Dropout(dropout)
        )
        self.age_head = nn.Linear(256, 1)
        self.gender_head = nn.Linear(256, 1)

    def forward(self, x):
        h = self.neck(self.pool(self.features(self.norm(x))))
        return self.age_head(h).squeeze(1), self.gender_head(h).squeeze(1)


class ExpressionNet(nn.Module):
    """7-class facial expression classifier (FER2013 classes). Returns logits."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.norm = InputNorm(grayscale=True)
        self.features = mobilenet_features(pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FEATURE_DIM, 256),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(256, len(EXPRESSIONS)),
        )

    def forward(self, x):
        return self.head(self.pool(self.features(self.norm(x))))


class AgeGenderExport(nn.Module):
    """Inference wrapper: outputs (age in years, P(female))."""

    def __init__(self, net: AgeGenderNet):
        super().__init__()
        self.net = net

    def forward(self, x):
        age, gender_logit = self.net(x)
        return torch.clamp(age * AGE_SCALE, 0.0, float(MAX_AGE)), torch.sigmoid(gender_logit)


class ExpressionExport(nn.Module):
    """Inference wrapper: outputs class probabilities (softmax)."""

    def __init__(self, net: ExpressionNet):
        super().__init__()
        self.net = net

    def forward(self, x):
        return F.softmax(self.net(x), dim=1)


class EdgeHardsigmoid(nn.Module):
    """hardsigmoid(x) = relu6(x + 3) / 6 using only Add/Clip/Div.

    Mathematically identical to nn.Hardsigmoid, but exports to basic ONNX ops that
    every ONNX Runtime Web backend (WASM and WebGPU) supports.
    """

    def forward(self, x):
        return F.relu6(x + 3.0) / 6.0


class EdgeHardswish(nn.Module):
    """hardswish(x) = x * relu6(x + 3) / 6 using only basic ONNX ops."""

    def forward(self, x):
        return x * F.relu6(x + 3.0) / 6.0


def make_edge_friendly(module: nn.Module) -> nn.Module:
    """Replace Hardswish/Hardsigmoid modules in place by basic-op equivalents."""
    for name, child in module.named_children():
        if isinstance(child, nn.Hardswish):
            setattr(module, name, EdgeHardswish())
        elif isinstance(child, nn.Hardsigmoid):
            setattr(module, name, EdgeHardsigmoid())
        else:
            make_edge_friendly(child)
    return module


def build_model(task: str, pretrained: bool = True) -> nn.Module:
    """Training network for `task` ("age_gender" or "expression")."""
    if task == "age_gender":
        return AgeGenderNet(pretrained)
    if task == "expression":
        return ExpressionNet(pretrained)
    raise ValueError(task)


def export_wrapper(task: str, net: nn.Module) -> nn.Module:
    """Wrap a trained network with the post-processing used by the web app
    (age in years and p_female, or softmax probabilities) for ONNX export."""
    return AgeGenderExport(net) if task == "age_gender" else ExpressionExport(net)


def count_parameters(model: nn.Module) -> int:
    """Total number of parameters."""
    return sum(p.numel() for p in model.parameters())
