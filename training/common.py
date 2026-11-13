"""Shared constants and helper functions for training, evaluation and export."""
from __future__ import annotations

import json
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

# FER2013 label order (0..6) as defined in the original FER2013 CSV.
EXPRESSIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
HAPPY, SAD = EXPRESSIONS.index("happy"), EXPRESSIONS.index("sad")

# Age groups used in the project evaluation (WHO: "older people" = 60+).
ADULT_MIN_AGE = 18
ELDERLY_MIN_AGE = 60
MAX_AGE = 116  # oldest age label in UTKFace

# UTKFace gender convention: 0 = male, 1 = female.
GENDERS = ["male", "female"]

DEFAULT_IMG_SIZE = {"age_gender": 160, "expression": 112}


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all GPUs). CPU runs are then repeatable;
    cuDNN kernels on the GPU can still introduce small run-to-run differences."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    """Map "auto", "cpu" or "cuda" to a device; stop with a hint if CUDA is requested but unavailable."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Install the CUDA build of PyTorch (see README)."
        )
    return torch.device(name)


def cpu_name() -> str:
    """Human-readable CPU model name (used in the hardware tables)."""
    system = platform.system()
    try:
        if system == "Windows":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if system == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if system == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def hardware_info(device: torch.device) -> dict:
    """Hardware and software description stored with every run (used for the CPU vs. GPU comparison)."""
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": cpu_name(),
        "cpu_threads": torch.get_num_threads(),
        "device": str(device),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update(
            {
                "gpu": props.name,
                "gpu_memory_gb": round(props.total_memory / 1024**3, 1),
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            }
        )
    return info


def save_json(obj, path) -> None:
    """Write `obj` as indented UTF-8 JSON, creating the parent folder if necessary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def portable_path(path) -> str:
    """Path as written into result files: relative to the repository (with "/") when it lies inside it,
    otherwise only its last part. Committed JSON files then contain no local folder names."""
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.name if p.is_absolute() else p.as_posix()


def load_json(path) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def age_group(age) -> np.ndarray:
    """1 = elderly (60+), 0 = adult. Accepts scalars or arrays."""
    return (np.asarray(age) >= ELDERLY_MIN_AGE).astype(int)
