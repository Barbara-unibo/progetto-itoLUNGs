"""Shared helpers: logging, seeding, device setup, metrics and plotting."""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend: figures are only written to disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# Metrics reported for every evaluation; the first one drives model selection.
METRIC_NAMES: tuple[str, ...] = (
    "balanced_accuracy",
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "mcc",
    "auc",
)


# --------------------------------------------------------------------- logging
def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    """Log to stdout and, optionally, to a UTF-8 file."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


# ------------------------------------------------------------ reproducibility
def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU + all GPUs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # Must be set before the first cuBLAS call to take full effect.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: derive Python/NumPy seeds from the torch worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def configure_torch() -> None:
    """Enable TF32 matmuls/convolutions on Ampere+ GPUs (safe for training, faster)."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    logger.warning("CUDA not available: running on CPU (AMP disabled, training will be slow).")
    return torch.device("cpu")


def log_environment(device: torch.device) -> None:
    logger.info("Python %s | PyTorch %s | platform %s", platform.python_version(), torch.__version__, platform.platform())
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        logger.info(
            "GPU: %s | %.1f GB | CUDA %s | bf16 supported: %s",
            props.name,
            props.total_memory / 1024**3,
            torch.version.cuda,
            torch.cuda.is_bf16_supported(),
        )


# --------------------------------------------------------------------- metrics
def compute_class_weights(labels: Iterable[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights ``n / (k * count_c)`` for CrossEntropyLoss."""
    counts = np.bincount(np.asarray(list(labels), dtype=int), minlength=num_classes).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"Every class must be present in the training set, got counts {counts.tolist()}")
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_prob: Sequence[float]) -> dict[str, float]:
    """Binary metrics with Neoplastic (1) as the positive class.

    Sensitivity = recall of Neoplastic, specificity = recall of Normal,
    balanced accuracy = their mean. AUC uses the predicted P(Neoplastic).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    both_classes = len(np.unique(y_true)) == 2
    return {
        "balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)) if both_classes else float("nan"),
    }


def summarize_metrics(
    per_fold: pd.DataFrame,
    metrics: Sequence[str] = METRIC_NAMES,
    group_col: str = "setting",
) -> pd.DataFrame:
    """Mean and sample standard deviation (ddof=1) of each metric per setting."""
    rows: list[dict[str, Any]] = []
    for group, frame in per_fold.groupby(group_col, sort=False):
        for metric in metrics:
            values = frame[metric].astype(float)
            rows.append(
                {
                    group_col: group,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                    "n_folds": int(values.notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def format_mean_std(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


# ------------------------------------------------------------------------- I/O
def save_json(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# -------------------------------------------------------------------- plotting
def _draw_confusion_matrix(ax: plt.Axes, cm: np.ndarray, class_names: Sequence[str], title: str) -> Any:
    cm = np.asarray(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
    image = ax.imshow(pct, cmap="Blues", vmin=0.0, vmax=1.0)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks, labels=class_names)
    ax.set_yticks(ticks, labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({pct[i, j]:.1%})",
                ha="center",
                va="center",
                color="white" if pct[i, j] > 0.5 else "black",
                fontsize=11,
            )
    return image


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], out_path: Path, title: str) -> None:
    """Counts with row-normalised percentages (i.e. per-class recall)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = _draw_confusion_matrix(ax, cm, class_names, title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrices_side_by_side(
    cms: Mapping[str, np.ndarray], class_names: Sequence[str], out_path: Path, suptitle: str
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(cms), figsize=(5 * len(cms), 4.8), squeeze=False)
    for ax, (title, cm) in zip(axes[0], cms.items()):
        _draw_confusion_matrix(ax, cm, class_names, title)
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_training_history(history: pd.DataFrame, out_path: Path, title: str) -> None:
    """Loss and balanced-accuracy curves; the selected (best) epoch is marked."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(history["epoch"], history["train_loss"], label="train")
    ax_loss.plot(history["epoch"], history["val_loss"], label="validation")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.legend()
    ax_acc.plot(history["epoch"], history["train_balanced_accuracy"], label="train")
    ax_acc.plot(history["epoch"], history["val_balanced_accuracy"], label="validation")
    if "is_best" in history and history["is_best"].any():
        best_epoch = int(history.loc[history["is_best"], "epoch"].iloc[-1])
        for ax in (ax_loss, ax_acc):
            ax.axvline(best_epoch, color="grey", linestyle="--", linewidth=1, label="selected")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("balanced accuracy")
    ax_acc.set_ylim(0.0, 1.02)
    ax_acc.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
