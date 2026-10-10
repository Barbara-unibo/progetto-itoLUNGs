"""Datasets, transforms, data indexing and patient-level splitting.

Human (LungHist700) and canine images are indexed by separate functions and are
never concatenated: LungHist700 is used only for pre-training, the canine
dataset only for fine-tuning and evaluation.
"""
from __future__ import annotations

import logging
import os
import random
import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .config import IMAGENET_MEAN, IMAGENET_STD, MACENKO_TARGET_IMG, Config
from .utils import seed_worker
from .utils import MacenkoTransform


logger = logging.getLogger(__name__)

macenko_norm = MacenkoTransform(target_image_path=MACENKO_TARGET_IMG)

REQUIRED_COLUMNS: tuple[str, ...] = ("case_id", "image_path", "label")
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"})

# Canine file names look like "<class>_<image id>_<objective>.jpg", e.g. "acin_491_20x.jpg".
_CANINE_FILE_RE = re.compile(r"^(?P<cls>[a-z?]+)_(?P<id>\d+)_(?P<mag>\d+x)", re.IGNORECASE)
CANINE_NORMAL_CLASSES: frozenset[str] = frozenset({"nn", "nor", "normal"})

# LungHist700 tokens (file-name parts or folder names) mapped to the binary task.
LUNGHIST_NORMAL_TOKENS: frozenset[str] = frozenset({"nor", "normal", "healthy"})
LUNGHIST_TUMOUR_TOKENS: frozenset[str] = frozenset(
    {"aca", "scc", "adenocarcinoma", "squamous", "carcinoma", "tumor", "tumour", "neoplastic"}
)


# ------------------------------------------------------------------ transforms
class RandomRot90:
    """Rotate by a random multiple of 90 degrees (lossless, no padding corners)."""

    _OPS = {1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180, 3: Image.Transpose.ROTATE_270}

    def __call__(self, img: Image.Image) -> Image.Image:
        k = random.randint(0, 3)
        return img.transpose(self._OPS[k]) if k else img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def build_transforms(img_size: int, train: bool, macenko_norm: MacenkoTransform = None) -> T.Compose:
    """Histology-friendly augmentation; evaluation resizes the whole field of view."""
    normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    transforms_list = []
    if macenko_norm is not None:
        transforms_list.append(macenko_norm)
    if train:
        transforms_list.extend([
                T.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3), antialias=True),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                RandomRot90(),
                # Mild colour jitter mimics H&E staining variability.
                T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03)], p=0.8),
                T.ToTensor(),
                normalize,
            ])
    else:
        transforms_list.extend([
            T.Resize((img_size, img_size), antialias=True),
            T.ToTensor(),
            normalize
        ]
    )
    return T.Compose(transforms_list)


# --------------------------------------------------------------------- dataset
class HistologyDataset(Dataset):
    """Returns ``(image_tensor, label, row_index)``; the index maps back to the DataFrame."""

    def __init__(self, df: pd.DataFrame, transform: T.Compose) -> None:
        missing = [c for c in ("image_path", "label") if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing columns {missing}")
        self.paths: list[str] = df["image_path"].astype(str).tolist()
        self.labels: list[int] = df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        with Image.open(self.paths[idx]) as img:
            image = img.convert("RGB")
        return self.transform(image), self.labels[idx], idx


def make_loader(df: pd.DataFrame, cfg: Config, train: bool, seed: int, persistent: bool = True) -> DataLoader:
    """Seeded DataLoader: identical seeds give identical batch order and augmentations.

    ``persistent=False`` for loaders iterated only once (avoids idle worker processes).
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs: dict = dict(
        batch_size=cfg.batch_size,
        shuffle=train,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )
    if cfg.num_workers > 0:
        kwargs.update(persistent_workers=persistent, prefetch_factor=2)
    return DataLoader(HistologyDataset(df, build_transforms(cfg.img_size, train, macenko_norm=macenko_norm)), **kwargs)


# ------------------------------------------------------------- CSV utilities
def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()
    except ValueError:  # different drive on Windows
        return path.resolve().as_posix()


def _log_dataset_summary(df: pd.DataFrame, name: str) -> None:
    counts = df["label"].value_counts().to_dict()
    logger.info(
        "%s: %d images, %d cases | Normal=%d, Neoplastic=%d",
        name,
        len(df),
        df["case_id"].nunique(),
        counts.get(0, 0),
        counts.get(1, 0),
    )


def load_image_csv(csv_path: Path, check_files: bool = True) -> pd.DataFrame:
    """Load a ``case_id, image_path, label`` CSV (extra columns are kept).

    Relative image paths are resolved against the CSV's own directory.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype={"case_id": str})
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{csv_path} is missing required columns {missing_cols}")
    df = df.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    df["case_id"] = df["case_id"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    invalid = sorted(set(df["label"].unique()) - {0, 1})
    if invalid:
        raise ValueError(f"Labels must be 0 (Normal) or 1 (Neoplastic); found {invalid}")

    base = csv_path.resolve().parent
    df["image_path"] = [
        str(p if p.is_absolute() else (base / p).resolve()) for p in map(Path, df["image_path"].astype(str))
    ]
    duplicated = df["image_path"].duplicated()
    if duplicated.any():
        logger.warning("Dropping %d duplicated image paths in %s", int(duplicated.sum()), csv_path.name)
        df = df[~duplicated]
    if check_files:
        missing = [p for p in df["image_path"] if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images listed in {csv_path} do not exist, e.g. {missing[:5]}")
    df = df.reset_index(drop=True)
    _log_dataset_summary(df, csv_path.name)
    return df


# ------------------------------------------------------- canine CSV builder
def _read_raw_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(path, dtype=str, encoding=encoding, sep=None, engine="python")
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path} as UTF-8 or CP1252")


def build_canine_csv(raw_csv: Path, image_dirs: Sequence[Path], out_csv: Path) -> pd.DataFrame:
    """Create ``case_id, image_path, label`` (+ ``subtype, magnification``) for the canine set.

    The raw spreadsheet (``PATIENT_ID, IMAGE_ID, OBJECTIVE, SUPERCLASS``) is used
    only to map each image ID to its patient. The image files on disk are the
    source of truth for the label (``nn_*`` = Normal, any other class = Neoplastic),
    because the file name and the spreadsheet sometimes disagree on the objective.
    Files whose ID is missing from the spreadsheet are excluded and reported.
    """
    raw_csv, out_csv = Path(raw_csv), Path(out_csv)
    raw = _read_raw_csv(raw_csv)
    raw.columns = [str(c).strip().upper() for c in raw.columns]
    for col in ("PATIENT_ID", "IMAGE_ID"):
        if col not in raw.columns:
            raise ValueError(f"{raw_csv} must contain a {col} column; found {list(raw.columns)}")
    raw = raw.dropna(subset=["PATIENT_ID", "IMAGE_ID"]).copy()
    raw["PATIENT_ID"] = raw["PATIENT_ID"].str.strip()
    raw["IMAGE_ID"] = pd.to_numeric(raw["IMAGE_ID"].str.strip(), errors="coerce")
    raw = raw.dropna(subset=["IMAGE_ID"])
    raw["IMAGE_ID"] = raw["IMAGE_ID"].astype(int)
    has_class = "SUPERCLASS" in raw.columns

    id_to_patient: dict[int, str] = {}
    id_to_class: dict[int, str] = {}
    for image_id, grp in raw.groupby("IMAGE_ID"):
        patients = grp["PATIENT_ID"].unique()
        if len(patients) > 1:
            logger.warning("IMAGE_ID %d is assigned to several patients %s: excluded", image_id, list(patients))
            continue
        if len(grp) > 1:
            logger.warning("IMAGE_ID %d appears %d times in the spreadsheet (patient %s)", image_id, len(grp), patients[0])
        id_to_patient[int(image_id)] = str(patients[0])
        if has_class and isinstance(grp["SUPERCLASS"].iloc[0], str):
            id_to_class[int(image_id)] = grp["SUPERCLASS"].iloc[0].strip().lower().rstrip("?")

    records: list[dict] = []
    seen_ids: dict[int, Path] = {}
    unparsable: list[str] = []
    unmatched: list[str] = []
    conflicting: list[str] = []
    for image_dir in map(Path, image_dirs):
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        files = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        for path in files:
            match = _CANINE_FILE_RE.match(path.name)
            if match is None:
                unparsable.append(path.name)
                continue
            image_id = int(match["id"])
            if image_id in seen_ids:
                logger.warning("Duplicate image ID %d: %s and %s (keeping the first)", image_id, seen_ids[image_id].name, path.name)
                continue
            seen_ids[image_id] = path
            if image_id not in id_to_patient:
                unmatched.append(path.name)
                continue
            cls = match["cls"].lower().rstrip("?")
            label = 0 if cls in CANINE_NORMAL_CLASSES else 1
            csv_cls = id_to_class.get(image_id)
            if csv_cls is not None and (csv_cls in CANINE_NORMAL_CLASSES) != (label == 0):
                conflicting.append(f"{path.name} (spreadsheet: {csv_cls})")
                continue
            records.append(
                {
                    "case_id": id_to_patient[image_id],
                    "image_path": _relative_or_absolute(path, out_csv.parent),
                    "label": label,
                    "subtype": "normal" if label == 0 else cls,
                    "magnification": match["mag"].lower(),
                }
            )

    if unparsable:
        logger.warning("%d files with unrecognised names skipped, e.g. %s", len(unparsable), unparsable[:10])
    if unmatched:
        logger.warning("%d files have no patient in the spreadsheet and were EXCLUDED: %s", len(unmatched), unmatched)
    if conflicting:
        logger.warning("%d files whose class contradicts the spreadsheet were EXCLUDED: %s", len(conflicting), conflicting)
    ids_without_file = sorted(set(id_to_patient) - set(seen_ids))
    if ids_without_file:
        logger.warning("%d spreadsheet IDs have no image file: %s", len(ids_without_file), ids_without_file)
    if not records:
        raise RuntimeError("No canine images could be indexed; check --raw-csv and --image-dirs.")

    df = pd.DataFrame(records).sort_values(["case_id", "image_path"]).reset_index(drop=True)
    per_case = df.groupby("case_id")["label"].nunique()
    single_class = per_case[per_case < 2].index.tolist()
    if single_class:
        logger.warning("Cases with images of a single class only: %s", single_class)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %s", out_csv)
    _log_dataset_summary(df, out_csv.name)
    logger.info("Subtypes: %s", df["subtype"].value_counts().to_dict())
    return df


# -------------------------------------------------------- LungHist700 index
def _lunghist_label(path: Path, root: Path) -> tuple[int, str] | None:
    """Binary label from file-name tokens first, then from folder names."""
    tokens = re.split(r"[_\-\s.]+", path.stem.lower())
    folders = [part.lower() for part in path.relative_to(root).parts[:-1]]
    for token in tokens + folders:
        if token in LUNGHIST_NORMAL_TOKENS:
            return 0, "normal"
        if token in LUNGHIST_TUMOUR_TOKENS:
            return 1, token
    return None


def _candidate_stems(meta: pd.DataFrame, cols: Mapping[str, str]) -> list[str | None]:
    """Reconstruct image file stems from a metadata table."""
    for key in ("filename", "file_name", "file", "image_name", "image_path", "path", "image", "name"):
        if key in cols:
            return [Path(v.strip()).stem if isinstance(v, str) else None for v in meta[cols[key]]]
    if {"superclass", "resolution", "image_id"} <= set(cols):
        stems: list[str | None] = []
        for _, row in meta.iterrows():
            parts = [
                row[cols["superclass"]],
                row[cols["subclass"]] if "subclass" in cols else None,
                row[cols["resolution"]],
                row[cols["image_id"]],
            ]
            kept = [str(p).strip() for p in parts if isinstance(p, str) and p.strip() and p.strip().lower() != "nan"]
            stems.append("_".join(kept))
        return stems
    return [None] * len(meta)


def _lunghist_patient_map(root: Path) -> dict[str, str]:
    """Map lower-case image stem -> patient ID using any metadata CSV under ``root``."""
    mapping: dict[str, str] = {}
    for csv_file in sorted(root.rglob("*.csv")):
        try:
            meta = pd.read_csv(csv_file, dtype=str)
        except Exception as exc:  # noqa: BLE001 - unreadable side files are simply ignored
            logger.debug("Skipping %s: %s", csv_file, exc)
            continue
        cols = {str(c).strip().lower(): c for c in meta.columns}
        patient_col = next((orig for low, orig in cols.items() if "patient" in low), None)
        if patient_col is None:
            continue
        before = len(mapping)
        for stem, patient in zip(_candidate_stems(meta, cols), meta[patient_col]):
            if stem and isinstance(patient, str) and patient.strip():
                mapping[stem.lower()] = patient.strip()
        logger.info("Patient IDs: %d entries read from %s", len(mapping) - before, csv_file.name)
    return mapping


def build_lunghist_index(root: Path, csv_path: Path | None = None) -> pd.DataFrame:
    """Index LungHist700 as a binary dataset (``nor`` -> 0; ``aca_*``, ``scc_*`` -> 1).

    If ``csv_path`` is given it is loaded directly (``case_id, image_path, label``).
    Otherwise images under ``root`` are scanned and patient IDs are recovered from
    the metadata CSV shipped with the dataset. Without patient IDs every image
    becomes its own group (image-level split) and a warning is logged.
    """
    if csv_path is not None:
        return load_image_csv(csv_path)
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"LungHist700 directory not found: {root}. Download the dataset (see README) "
            "or pass --lunghist-csv with columns case_id,image_path,label."
        )
    images = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No images found under {root}")

    stem_to_patient = _lunghist_patient_map(root)
    records: list[dict] = []
    skipped: list[str] = []
    for path in images:
        parsed = _lunghist_label(path, root)
        if parsed is None:
            skipped.append(path.name)
            continue
        label, superclass = parsed
        records.append(
            {
                "case_id": stem_to_patient.get(path.stem.lower()),
                "image_path": str(path.resolve()),
                "label": label,
                "subtype": superclass,
            }
        )
    if skipped:
        logger.warning("LungHist700: %d images without a recognisable class skipped, e.g. %s", len(skipped), skipped[:10])
    if not records:
        raise RuntimeError(f"No labelled LungHist700 images found under {root}")

    df = pd.DataFrame(records)
    n_missing = int(df["case_id"].isna().sum())
    if n_missing == len(df):
        logger.warning(
            "LungHist700: no patient IDs could be matched; falling back to an IMAGE-level "
            "train/validation split (provide --lunghist-csv for a patient-level split)."
        )
    elif n_missing:
        logger.warning("LungHist700: %d images without patient ID are treated as individual groups", n_missing)
    fallback = "img_" + df["image_path"].map(lambda s: Path(s).stem)
    df["case_id"] = df["case_id"].fillna(fallback).astype(str)
    _log_dataset_summary(df, "LungHist700 (binary)")
    return df


# ---------------------------------------------------------------- splitting
def make_patient_folds(df: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    """Assign every case to one of ``n_splits`` folds (1-based).

    StratifiedGroupKFold keeps each patient in a single fold (groups = ``case_id``)
    while balancing the Normal/Neoplastic label across folds.
    """
    n_cases = df["case_id"].nunique()
    if n_cases < n_splits:
        raise ValueError(f"Need at least {n_splits} cases for {n_splits}-fold CV, got {n_cases}")
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    assignment: dict[str, int] = {}
    splits = sgkf.split(np.zeros(len(df)), df["label"], df["case_id"])
    for fold, (_, test_idx) in enumerate(splits, start=1):
        for case_id in df["case_id"].iloc[test_idx].unique():
            assignment[str(case_id)] = fold
    return pd.DataFrame(sorted(assignment.items()), columns=["case_id", "fold"])


def get_or_create_folds(df: pd.DataFrame, folds_csv: Path, n_splits: int, seed: int) -> pd.DataFrame:
    """Load the persisted fold assignment, or create and save it.

    Persisting the folds guarantees that settings A and B (and any re-run) are
    evaluated on exactly the same patients.
    """
    if folds_csv.is_file():
        folds = pd.read_csv(folds_csv, dtype={"case_id": str})
        if set(folds["case_id"]) != set(df["case_id"]) or folds["fold"].nunique() != n_splits:
            raise ValueError(
                f"{folds_csv} does not match the current dataset / n_splits. "
                "Delete it (and old CV results) to generate new folds."
            )
        logger.info("Reusing fold assignment from %s", folds_csv)
    else:
        folds = make_patient_folds(df, n_splits, seed)
        folds_csv.parent.mkdir(parents=True, exist_ok=True)
        folds.to_csv(folds_csv, index=False)
        logger.info("Created %d patient-level folds (stratified by Normal/Neoplastic) -> %s", n_splits, folds_csv)

    merged = df.merge(folds, on="case_id", how="left")
    for fold, frame in merged.groupby("fold"):
        counts = frame["label"].value_counts().to_dict()
        logger.info(
            "  fold %d: %2d cases, %4d images (Normal=%d, Neoplastic=%d)",
            fold,
            frame["case_id"].nunique(),
            len(frame),
            counts.get(0, 0),
            counts.get(1, 0),
        )
    return folds


def split_train_val_by_group(df: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out ~``val_fraction`` of the cases (never single images), stratified by label."""
    n_splits = max(2, int(round(1.0 / val_fraction)))
    n_cases = df["case_id"].nunique()
    if n_cases < n_splits:
        raise ValueError(f"Need at least {n_splits} cases for a {val_fraction:.0%} validation split, got {n_cases}")
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(np.zeros(len(df)), df["label"], df["case_id"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def assert_patient_disjoint(splits: Mapping[str, pd.DataFrame]) -> None:
    """Raise if any case appears in more than one split (data leakage guard)."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = set(splits[a]["case_id"]) & set(splits[b]["case_id"])
            if overlap:
                raise RuntimeError(f"Patient leakage between '{a}' and '{b}': {sorted(overlap)[:10]}")
