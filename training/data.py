"""Dataset loading for UTKFace (age + gender) and FER2013 (expression).

UTKFace: file names follow  [age]_[gender]_[race]_[date&time].jpg  with
gender 0 = male, 1 = female (Zhang et al., 2017). Images whose names cannot be
parsed are skipped. A fixed, stratified 80/10/10 split is written to
splits/utkface_split.csv on first use, so CPU and GPU runs use identical data.

FER2013 (Goodfellow et al., 2013) is accepted in two formats:
* folder format  <root>/train/<class>/*.png  and  <root>/test/<class>/*.png
  (test = official PrivateTest set; 10 % of train is held out for validation)
* original CSV   fer2013.csv  with columns emotion, pixels, Usage
  (Training -> train, PublicTest -> val, PrivateTest -> test)
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms as T

from common import EXPRESSIONS, MAX_AGE

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# UTKFace
# --------------------------------------------------------------------------- #
def _age_bucket(age: int) -> int:
    return 0 if age < 18 else 1 if age < 40 else 2 if age < 60 else 3


def scan_utkface(root: Path):
    """Return records [{file, age, gender}] and the number of skipped files."""
    records, seen, skipped = [], set(), 0
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() not in IMG_EXT or p.name in seen:
            continue
        parts = p.name.split("_")
        try:
            age, gender = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            skipped += 1
            continue
        if gender not in (0, 1) or not 0 <= age <= MAX_AGE:
            skipped += 1
            continue
        seen.add(p.name)
        records.append({"file": p.relative_to(root).as_posix(), "age": age, "gender": gender})
    return records, skipped


def _split_indices(n, strata, seed, val_frac, test_frac):
    idx = np.arange(n)
    try:
        trval, test = train_test_split(idx, test_size=test_frac, random_state=seed, stratify=strata)
        tr, va = train_test_split(
            trval,
            test_size=val_frac / (1 - test_frac),
            random_state=seed,
            stratify=[strata[i] for i in trval],
        )
    except ValueError:  # very small datasets (e.g. smoke test) cannot be stratified
        trval, test = train_test_split(idx, test_size=test_frac, random_state=seed)
        tr, va = train_test_split(trval, test_size=val_frac / (1 - test_frac), random_state=seed)
    return tr, va, test


def utkface_splits(root, split_csv, seed=42, val_frac=0.1, test_frac=0.1) -> dict:
    """Stratified train/val/test split of UTKFace (strata: gender x age bucket).

    The split is written to `split_csv` on first use and read back afterwards, so every run
    (GPU, CPU, evaluation, crop calibration) uses exactly the same images."""
    split_csv = Path(split_csv)
    if split_csv.exists():
        out = {s: [] for s in SPLITS}
        with open(split_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out[r["split"]].append(
                    {"file": r["file"], "age": int(r["age"]), "gender": int(r["gender"])}
                )
        missing = [r["file"] for r in out["test"][:20] if not (Path(root) / r["file"]).exists()]
        if missing:
            raise SystemExit(
                f"{split_csv} does not match the images in {root} (e.g. {missing[0]}). "
                "Delete the split file or point --data to the right folder."
            )
        return out

    records, skipped = scan_utkface(root)
    if not records:
        raise SystemExit(f"No UTKFace images found in {root}")
    strata = [r["gender"] * 4 + _age_bucket(r["age"]) for r in records]
    tr, va, te = _split_indices(len(records), strata, seed, val_frac, test_frac)
    out = {"train": [records[i] for i in tr], "val": [records[i] for i in va], "test": [records[i] for i in te]}

    split_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(split_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "age", "gender", "split"])
        for s in SPLITS:
            for r in out[s]:
                w.writerow([r["file"], r["age"], r["gender"], s])
    print(
        f"UTKFace: {len(records)} images ({skipped} skipped) -> "
        + ", ".join(f"{s}={len(out[s])}" for s in SPLITS)
        + f"; split saved to {split_csv}"
    )
    return out


class UTKFaceDataset(Dataset):
    def __init__(self, root, records, transform):
        self.root, self.records, self.transform = Path(root), records, transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = Image.open(self.root / r["file"]).convert("RGB")
        return (
            self.transform(img),
            torch.tensor(r["age"], dtype=torch.float32),
            torch.tensor(r["gender"], dtype=torch.float32),
        )


# --------------------------------------------------------------------------- #
# FER2013
# --------------------------------------------------------------------------- #
FOLDER_ALIASES = {
    "angry": "angry", "anger": "angry",
    "disgust": "disgust", "disgusted": "disgust",
    "fear": "fear", "fearful": "fear",
    "happy": "happy", "happiness": "happy",
    "sad": "sad", "sadness": "sad",
    "surprise": "surprise", "surprised": "surprise",
    "neutral": "neutral",
}


def _scan_class_folders(d: Path):
    items = []
    for cls_dir in sorted(p for p in d.iterdir() if p.is_dir()):
        key = cls_dir.name.lower()
        if key not in FOLDER_ALIASES:
            print(f"warning: ignoring unknown class folder {cls_dir}")
            continue
        label = EXPRESSIONS.index(FOLDER_ALIASES[key])
        items += [(str(p), label) for p in sorted(cls_dir.iterdir()) if p.suffix.lower() in IMG_EXT]
    return items


def _fer_csv(csv_path: Path):
    usage = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}
    out = {s: [] for s in SPLITS}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
            out[usage[row["Usage"].strip()]].append((pixels, int(row["emotion"])))
    return out


def find_fer2013(path) -> Path:
    """Accept the CSV itself or any folder above it (Kaggle archives add subfolders)."""
    path = Path(path)
    if path.is_file():
        return path
    search_root = path if path.is_dir() else path.parent
    if search_root.is_dir():
        if (search_root / "train").is_dir() and (search_root / "test").is_dir():
            return search_root
        hits = sorted(search_root.rglob("fer2013.csv"))
        if hits:
            if hits[0] != path:
                print(f"FER2013: using {hits[0]}")
            return hits[0]
    raise SystemExit(
        f"FER2013 not found at {path}. Pass the fer2013.csv file or the folder that contains it "
        "(or a folder with train/ and test/ subfolders)."
    )


def fer2013_splits(path, seed=42, val_frac=0.1) -> dict:
    """FER2013 train/val/test lists. The CSV version uses its own "Usage" column
    (Training / PublicTest / PrivateTest); the image-folder version splits `val_frac` off the
    training folder with a fixed seed."""
    path = find_fer2013(path)
    if path.is_file() and path.suffix.lower() == ".csv":
        return _fer_csv(path)
    train_dir, test_dir = path / "train", path / "test"
    if not (train_dir.is_dir() and test_dir.is_dir()):
        raise SystemExit(f"Expected {path}/train and {path}/test folders or a fer2013.csv file.")
    train_items = _scan_class_folders(train_dir)
    labels = [label for _, label in train_items]
    try:
        tr, va = train_test_split(train_items, test_size=val_frac, random_state=seed, stratify=labels)
    except ValueError:
        tr, va = train_test_split(train_items, test_size=val_frac, random_state=seed)
    return {"train": tr, "val": va, "test": _scan_class_folders(test_dir)}


class FERDataset(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    @property
    def labels(self) -> list:
        """Class index of every sample (used for stratification and class weights)."""
        return [label for _, label in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        src, label = self.items[i]
        img = Image.fromarray(src) if isinstance(src, np.ndarray) else Image.open(src)
        return self.transform(img.convert("L").convert("RGB")), label


# --------------------------------------------------------------------------- #
# Transforms and factory
# --------------------------------------------------------------------------- #
def build_transform(task: str, size: int, train: bool):
    """Pixels are only scaled to [0, 1]; normalisation happens inside the model."""
    if not train:
        return T.Compose([T.Resize((size, size)), T.ToTensor()])
    if task == "age_gender":
        return T.Compose(
            [
                T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.3, 0.3, 0.2, 0.02),
                T.RandomGrayscale(0.1),
                T.ToTensor(),
            ]
        )
    return T.Compose(
        [
            T.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(0.3, 0.3),
            T.ToTensor(),
        ]
    )


def build_datasets(task: str, data, size: int, seed: int, split_dir: Path, train_aug: bool = True) -> dict:
    """Datasets for the three splits; augmentation is applied to the training split only."""
    if task == "age_gender":
        recs = utkface_splits(data, Path(split_dir) / "utkface_split.csv", seed)
        return {
            s: UTKFaceDataset(data, recs[s], build_transform(task, size, train_aug and s == "train"))
            for s in SPLITS
        }
    items = fer2013_splits(data, seed)
    return {s: FERDataset(items[s], build_transform(task, size, train_aug and s == "train")) for s in SPLITS}


def class_weights(labels, num_classes: int) -> torch.Tensor:
    """Inverse square-root frequency weights (FER2013 is imbalanced, e.g. 'disgust')."""
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    w = 1.0 / np.sqrt(counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32)
