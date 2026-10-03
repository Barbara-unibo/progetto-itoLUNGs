"""Central configuration.

Every tunable value lives in :class:`Config`. The CLI in ``train.py`` overrides
fields by name, and the resolved configuration is saved as JSON next to the
logs of every run so that each experiment can be reproduced exactly.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Binary task: index 0 = Normal, index 1 = Neoplastic (positive class).
CLASS_NAMES: tuple[str, str] = ("Normal", "Neoplastic")

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# The two experimental settings compared on identical folds.
SETTING_IMAGENET = "imagenet"  # A) ImageNet -> fine-tune on canine
SETTING_LUNGHIST = "lunghist"  # B) ImageNet -> LungHist700 -> fine-tune on canine
SETTINGS: tuple[str, str] = (SETTING_IMAGENET, SETTING_LUNGHIST)
SETTING_LABELS: dict[str, str] = {
    SETTING_IMAGENET: "A) ImageNet -> Canine",
    SETTING_LUNGHIST: "B) ImageNet -> LungHist700 -> Canine",
}


def _default_num_workers() -> int:
    return min(8, os.cpu_count() or 1)


@dataclass
class Config:
    # ------------------------------------------------------------------ paths
    canine_csv: Path = PROJECT_ROOT / "data" / "canine.csv"
    canine_raw_csv: Path = PROJECT_ROOT / "data" / "CANINELUNGHISTO(Foglio1).csv"
    canine_image_dirs: tuple[Path, ...] = (
        PROJECT_ROOT / "data" / "nn",
        PROJECT_ROOT / "data" / "neo",
    )
    lunghist_root: Path = PROJECT_ROOT / "data" / "LungHist700"
    # Optional pre-built index (case_id, image_path, label) that bypasses auto-detection.
    lunghist_csv: Path | None = None
    output_dir: Path = PROJECT_ROOT / "results"
    # Optional explicit path of the pre-trained checkpoint used by setting B.
    pretrain_checkpoint_path: Path | None = None

    # ---------------------------------------------------------------- general
    seed: int = 42
    deterministic: bool = False  # True = bit-reproducible but slower (disables cudnn.benchmark)
    img_size: int = 512
    num_classes: int = 2
    batch_size: int = 12
    num_workers: int = field(default_factory=_default_num_workers)
    amp: bool = True
    amp_dtype: str = "float16"  # "float16" (with GradScaler) or "bfloat16"
    channels_last: bool = True

    # ----------------------------------------------------------- optimisation
    # Shared by pre-training and fine-tuning so both phases are comparable.
    label_smoothing: float = 0.1
    weight_decay: float = 0.05
    min_lr: float = 1e-6
    max_grad_norm: float | None = 1.0
    use_class_weights: bool = True  # inverse-frequency CE weights (LungHist700 is imbalanced)

    # ----------------------------------------------- pre-training (LungHist700)
    pretrain_epochs: int = 30
    pretrain_lr: float = 1e-4
    pretrain_val_fraction: float = 0.2

    # ----------------------------------------------------- fine-tuning (canine)
    finetune_epochs: int = 30
    finetune_lr: float = 1e-4
    # True: setting B re-initialises the classification head, so A and B differ
    # only in the backbone initialisation. False: keep the human-trained head.
    reinit_head: bool = True

    # ------------------------------------------------------- cross-validation
    n_splits: int = 5
    # Fraction of the training patients of each fold held out for checkpoint
    # selection. The outer test fold is never used for model selection.
    inner_val_fraction: float = 0.2

    # --------------------------------------------------------------- Grad-CAM
    gradcam_per_class: int = 4  # test images per class shown for each fold
    gradcam_target: str = "true"  # "true": explain the ground-truth class; "pred": each model's prediction

    # ------------------------------------------------------- derived locations
    @property
    def pretrain_dir(self) -> Path:
        return self.output_dir / "pretrain"

    @property
    def pretrain_checkpoint(self) -> Path:
        if self.pretrain_checkpoint_path is not None:
            return self.pretrain_checkpoint_path
        return self.pretrain_dir / "best_lunghist700.pt"

    @property
    def cv_dir(self) -> Path:
        return self.output_dir / "cv"

    @property
    def folds_csv(self) -> Path:
        return self.output_dir / "folds.csv"

    # ----------------------------------------------------------------- helpers
    def validate(self) -> None:
        """Fail fast on inconsistent settings."""
        if self.img_size % 32 != 0:
            raise ValueError(f"img_size must be a multiple of 32 for ConvNeXt, got {self.img_size}")
        if self.num_classes != 2:
            raise ValueError("This project implements binary classification only (num_classes=2).")
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        for name in ("inner_val_fraction", "pretrain_val_fraction"):
            value = getattr(self, name)
            if not 0.0 < value < 0.5:
                raise ValueError(f"{name} must be in (0, 0.5), got {value}")
        if self.amp_dtype not in ("float16", "bfloat16"):
            raise ValueError(f"amp_dtype must be 'float16' or 'bfloat16', got {self.amp_dtype!r}")
        if self.gradcam_target not in ("true", "pred"):
            raise ValueError(f"gradcam_target must be 'true' or 'pred', got {self.gradcam_target!r}")
        if self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("batch_size must be >= 1 and num_workers >= 0")

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (Paths become strings)."""
        out: dict[str, Any] = {}
        for key, value in asdict(self).items():
            if isinstance(value, Path):
                out[key] = str(value)
            elif isinstance(value, (list, tuple)):
                out[key] = [str(v) if isinstance(v, Path) else v for v in value]
            else:
                out[key] = value
        return out
