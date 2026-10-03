"""Side-by-side Grad-CAM comparison of settings A and B on the same test images."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, SETTING_IMAGENET, SETTING_LABELS, SETTING_LUNGHIST, Config
from .dataset import build_transforms
from .model import get_gradcam_target_layers, load_model_from_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class CamResult:
    pred: int
    prob_neoplastic: float
    target: int
    cam: np.ndarray  # HxW in [0, 1]


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """CHW normalised tensor -> HWC float32 RGB in [0, 1] (exactly what the model saw)."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = (tensor.detach().cpu().float() * std + mean).clamp(0.0, 1.0)
    return image.permute(1, 2, 0).numpy().astype(np.float32)


def select_samples(test_df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """Deterministic, class-balanced selection of test images."""
    parts = [
        group.sample(n=min(per_class, len(group)), random_state=seed)
        for _, group in test_df.groupby("label", sort=True)
    ]
    return pd.concat(parts).sort_values(["label", "case_id", "image_path"]).reset_index(drop=True)


def _explain(
    model: nn.Module, inputs: list[torch.Tensor], labels: list[int], target_mode: str, device: torch.device
) -> list[CamResult]:
    """Prediction + Grad-CAM for each input (fp32, no autocast, for faithful gradients)."""
    results: list[CamResult] = []
    with GradCAM(model=model, target_layers=get_gradcam_target_layers(model)) as cam:
        for x, label in zip(inputs, labels):
            x = x.to(device)
            with torch.no_grad():
                probs = torch.softmax(model(x).float(), dim=1)[0].cpu().numpy()
            pred = int(probs.argmax())
            target = label if target_mode == "true" else pred
            grayscale = cam(input_tensor=x, targets=[ClassifierOutputTarget(target)])[0]
            results.append(CamResult(pred=pred, prob_neoplastic=float(probs[1]), target=target, cam=grayscale))
    return results


def _plot_rows(rows: list[dict], out_path: Path, title: str, dpi: int) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(10.5, 3.7 * n), squeeze=False)
    for i, row in enumerate(rows):
        ax = axes[i][0]
        ax.imshow(row["image"])
        ax.set_title(
            f"{row['name']}\ncase {row['case_id']} | true: {CLASS_NAMES[row['label']]}",
            fontsize=8,
        )
        for j, setting in enumerate((SETTING_IMAGENET, SETTING_LUNGHIST), start=1):
            result: CamResult = row[setting]
            overlay = show_cam_on_image(row["image"], result.cam, use_rgb=True, image_weight=0.55)
            correct = result.pred == row["label"]
            axes[i][j].imshow(overlay)
            axes[i][j].set_title(
                f"{SETTING_LABELS[setting]}\n"
                f"pred: {CLASS_NAMES[result.pred]} (P_neo={result.prob_neoplastic:.2f}) | "
                f"CAM: {CLASS_NAMES[result.target]}",
                fontsize=8,
                color="darkgreen" if correct else "firebrick",
            )
        for ax in axes[i]:
            ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def run_gradcam_comparison(
    test_df: pd.DataFrame,
    checkpoint_a: Path,
    checkpoint_b: Path,
    out_dir: Path,
    cfg: Config,
    device: torch.device,
    fold: int,
) -> Path:
    """Explain the same test images with model A and model B and save the figures.

    Outputs: ``comparison_fold<k>.png`` (all samples) and one figure per image in
    ``images/``. Titles are green for correct and red for wrong predictions.
    """
    samples = select_samples(test_df, cfg.gradcam_per_class, seed=cfg.seed + fold)
    transform = build_transforms(cfg.img_size, train=False)
    inputs: list[torch.Tensor] = []
    for path in samples["image_path"]:
        with Image.open(path) as img:
            inputs.append(transform(img.convert("RGB")).unsqueeze(0))
    labels = samples["label"].astype(int).tolist()

    explanations: dict[str, list[CamResult]] = {}
    for setting, checkpoint in ((SETTING_IMAGENET, checkpoint_a), (SETTING_LUNGHIST, checkpoint_b)):
        model = load_model_from_checkpoint(checkpoint, cfg.num_classes, device)
        explanations[setting] = _explain(model, inputs, labels, cfg.gradcam_target, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows: list[dict] = []
    for i, sample in samples.iterrows():
        rows.append(
            {
                "image": denormalize(inputs[i][0]),
                "name": Path(sample["image_path"]).name,
                "case_id": sample["case_id"],
                "label": int(sample["label"]),
                SETTING_IMAGENET: explanations[SETTING_IMAGENET][i],
                SETTING_LUNGHIST: explanations[SETTING_LUNGHIST][i],
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"comparison_fold{fold}.png"
    _plot_rows(rows, summary_path, f"Grad-CAM comparison - fold {fold} (test patients)", dpi=120)
    for row in rows:
        safe_name = re.sub(r"[^\w.-]+", "_", Path(row["name"]).stem)
        _plot_rows([row], out_dir / "images" / f"{safe_name}.png", f"Fold {fold}", dpi=200)

    pd.DataFrame(
        [
            {
                "image_path": samples.loc[i, "image_path"],
                "case_id": samples.loc[i, "case_id"],
                "label": labels[i],
                **{f"pred_{s}": explanations[s][i].pred for s in explanations},
                **{f"prob_neoplastic_{s}": explanations[s][i].prob_neoplastic for s in explanations},
            }
            for i in range(len(samples))
        ]
    ).to_csv(out_dir / "gradcam_samples.csv", index=False)
    logger.info("Grad-CAM fold %d: %d images -> %s", fold, len(rows), summary_path)
    return summary_path
