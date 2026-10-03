"""Training engine: LungHist700 pre-training and canine patient-level cross-validation.

Pipeline
--------
1. ``run_pretraining``: ImageNet ConvNeXt-Tiny -> binary LungHist700 (human only).
2. ``run_cross_validation``: for every canine fold, fine-tune
   A) from ImageNet weights and B) from the LungHist700 checkpoint, using the
   same folds, inner validation patients, seeds, data order and hyper-parameters.
   The best epoch is chosen on the inner validation patients by balanced accuracy;
   the outer test fold is evaluated once with that checkpoint.
3. ``summarize_cv``: mean +/- std over folds, paired differences, aggregated
   confusion matrices.
"""
from __future__ import annotations

import contextlib
import gc
import logging
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import CLASS_NAMES, SETTING_IMAGENET, SETTING_LABELS, SETTING_LUNGHIST, SETTINGS, Config
from .dataset import (
    assert_patient_disjoint,
    build_lunghist_index,
    get_or_create_folds,
    load_image_csv,
    make_loader,
    split_train_val_by_group,
)
from .gradcam import run_gradcam_comparison
from .model import build_model, count_parameters, load_pretrained_checkpoint
from .utils import (
    METRIC_NAMES,
    compute_class_weights,
    compute_metrics,
    format_mean_std,
    load_json,
    plot_confusion_matrices_side_by_side,
    plot_confusion_matrix,
    plot_training_history,
    save_json,
    seed_everything,
    summarize_metrics,
)

logger = logging.getLogger(__name__)

BEST_CHECKPOINT = "best_model.pt"
TEST_METRICS = "test_metrics.json"


# ------------------------------------------------------------------ AMP utils
def _autocast(cfg: Config, device: torch.device) -> contextlib.AbstractContextManager:
    if not (cfg.amp and device.type == "cuda"):
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _make_scaler(cfg: Config, device: torch.device) -> torch.amp.GradScaler:
    # Loss scaling is only needed for float16; bfloat16 has the fp32 exponent range.
    enabled = cfg.amp and device.type == "cuda" and cfg.amp_dtype == "float16"
    return torch.amp.GradScaler("cuda", enabled=enabled)


def _to_device(images: torch.Tensor, labels: torch.Tensor, cfg: Config, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    memory_format = torch.channels_last if cfg.channels_last else torch.contiguous_format
    return (
        images.to(device, non_blocking=True, memory_format=memory_format),
        labels.to(device, non_blocking=True),
    )


def _build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """AdamW without weight decay on biases, norm weights and ConvNeXt layer-scale."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "layer_scale" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def _free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------- train / eval
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: Config,
    device: torch.device,
    desc: str,
) -> dict[str, float]:
    model.train()
    total_loss, n_seen = 0.0, 0
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    for images, labels, _ in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        images, labels = _to_device(images, labels, cfg, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(cfg, device):
            logits = model(images)
            loss = criterion(logits.float(), labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss ({loss.item()}) in {desc}")
        scaler.scale(loss).backward()
        if cfg.max_grad_norm:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        batch = labels.size(0)
        total_loss += loss.item() * batch
        n_seen += batch
        y_true.append(labels.cpu().numpy())
        y_pred.append(logits.argmax(dim=1).cpu().numpy())

    true, pred = np.concatenate(y_true), np.concatenate(y_pred)
    recalls = [np.mean(pred[true == c] == c) for c in np.unique(true)]
    return {"loss": total_loss / max(n_seen, 1), "balanced_accuracy": float(np.mean(recalls))}


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, cfg: Config, device: torch.device, desc: str
) -> dict[str, Any]:
    """Returns loss, labels, predictions, P(Neoplastic) and dataset indices."""
    model.eval()
    total_loss, n_seen = 0.0, 0
    y_true, y_pred, y_prob, indices = [], [], [], []
    for images, labels, idx in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        images, labels = _to_device(images, labels, cfg, device)
        with _autocast(cfg, device):
            logits = model(images)
        logits = logits.float()
        total_loss += criterion(logits, labels).item() * labels.size(0)
        n_seen += labels.size(0)
        probs = torch.softmax(logits, dim=1)
        y_true.append(labels.cpu().numpy())
        y_pred.append(probs.argmax(dim=1).cpu().numpy())
        y_prob.append(probs[:, 1].cpu().numpy())
        indices.append(idx.numpy())
    return {
        "loss": total_loss / max(n_seen, 1),
        "y_true": np.concatenate(y_true),
        "y_pred": np.concatenate(y_pred),
        "y_prob": np.concatenate(y_prob),
        "indices": np.concatenate(indices),
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    class_weights: torch.Tensor | None,
    checkpoint_path: Path,
    run_name: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train with AdamW + cosine annealing; keep the checkpoint with the best val balanced accuracy.

    Ties in balanced accuracy are broken by the lower validation loss.
    """
    weight = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.label_smoothing)
    val_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = _build_optimizer(model, lr, cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=cfg.min_lr)
    scaler = _make_scaler(cfg, device)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best: dict[str, Any] = {"epoch": 0, "balanced_accuracy": -1.0, "loss": float("inf"), "metrics": {}}
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, cfg, device, desc=f"{run_name} train {epoch}/{epochs}"
        )
        val = evaluate(model, val_loader, val_criterion, cfg, device, desc=f"{run_name} val")
        val_metrics = compute_metrics(val["y_true"], val["y_pred"], val["y_prob"])
        scheduler.step()

        bal_acc = val_metrics["balanced_accuracy"]
        improved = bal_acc > best["balanced_accuracy"] + 1e-6 or (
            abs(bal_acc - best["balanced_accuracy"]) <= 1e-6 and val["loss"] < best["loss"]
        )
        if improved:
            best = {"epoch": epoch, "balanced_accuracy": bal_acc, "loss": val["loss"], "metrics": val_metrics}
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": float(val["loss"]),
                    "val_metrics": val_metrics,
                    "run_name": run_name,
                    "class_names": list(CLASS_NAMES),
                    "img_size": cfg.img_size,
                    "config": cfg.to_dict(),
                    **(metadata or {}),
                },
                checkpoint_path,
            )

        elapsed = time.perf_counter() - start
        history.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_stats["loss"],
                "train_balanced_accuracy": train_stats["balanced_accuracy"],
                "val_loss": val["loss"],
                "val_balanced_accuracy": bal_acc,
                "val_auc": val_metrics["auc"],
                "is_best": improved,
                "seconds": elapsed,
            }
        )
        logger.info(
            "[%s] epoch %02d/%02d | lr %.2e | train loss %.4f bal_acc %.4f | val loss %.4f bal_acc %.4f auc %.4f | %.0fs%s",
            run_name,
            epoch,
            epochs,
            current_lr,
            train_stats["loss"],
            train_stats["balanced_accuracy"],
            val["loss"],
            bal_acc,
            val_metrics["auc"],
            elapsed,
            "  <- best" if improved else "",
        )

    logger.info("[%s] best epoch %d (val balanced accuracy %.4f) -> %s", run_name, best["epoch"], best["balanced_accuracy"], checkpoint_path)
    return best, pd.DataFrame(history)


def _load_weights(model: nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])


def evaluate_and_save(
    model: nn.Module,
    loader: DataLoader,
    df: pd.DataFrame,
    cfg: Config,
    device: torch.device,
    out_dir: Path,
    split_name: str,
    title: str,
) -> dict[str, float]:
    """Evaluate, then write ``<split>_predictions.csv`` and ``confusion_matrix_<split>.png``."""
    out = evaluate(model, loader, nn.CrossEntropyLoss(), cfg, device, desc=split_name)
    metrics = compute_metrics(out["y_true"], out["y_pred"], out["y_prob"])
    metrics["loss"] = float(out["loss"])

    predictions = df.iloc[out["indices"]].copy()
    if not np.array_equal(predictions["label"].to_numpy(), out["y_true"]):
        raise RuntimeError("Prediction/label alignment check failed")
    predictions["pred"] = out["y_pred"]
    predictions["prob_neoplastic"] = out["y_prob"]
    predictions["correct"] = predictions["pred"] == predictions["label"]
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / f"{split_name}_predictions.csv", index=False)

    cm = confusion_matrix(out["y_true"], out["y_pred"], labels=[0, 1])
    plot_confusion_matrix(cm, CLASS_NAMES, out_dir / f"confusion_matrix_{split_name}.png", title)
    return metrics


def _save_split(path: Path, **splits: pd.DataFrame) -> None:
    columns = [c for c in ("case_id", "image_path", "label", "subtype", "magnification")]
    frames = []
    for name, frame in splits.items():
        part = frame[[c for c in columns if c in frame.columns]].copy()
        part.insert(0, "split", name)
        frames.append(part)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


def _log_split(name: str, df: pd.DataFrame) -> None:
    counts = df["label"].value_counts().to_dict()
    logger.info(
        "  %-5s %2d cases, %4d images (Normal=%d, Neoplastic=%d)",
        name,
        df["case_id"].nunique(),
        len(df),
        counts.get(0, 0),
        counts.get(1, 0),
    )


# ----------------------------------------------------------- pre-training
def run_pretraining(cfg: Config, device: torch.device) -> Path:
    """Pre-train ConvNeXt-Tiny (ImageNet init) on binary LungHist700. Human images only."""
    logger.info("=== Pre-training on LungHist700 (human) ===")
    seed_everything(cfg.seed, cfg.deterministic)
    out_dir = cfg.pretrain_dir

    df = build_lunghist_index(cfg.lunghist_root, cfg.lunghist_csv)
    train_df, val_df = split_train_val_by_group(df, cfg.pretrain_val_fraction, cfg.seed)
    assert_patient_disjoint({"train": train_df, "val": val_df})
    _log_split("train", train_df)
    _log_split("val", val_df)
    _save_split(out_dir / "split.csv", train=train_df, val=val_df)

    model = build_model(cfg.num_classes, imagenet_pretrained=True)
    model = model.to(device, memory_format=torch.channels_last if cfg.channels_last else torch.contiguous_format)
    total, _ = count_parameters(model)
    logger.info("ConvNeXt-Tiny: %.1fM parameters", total / 1e6)

    train_loader = make_loader(train_df, cfg, train=True, seed=cfg.seed)
    val_loader = make_loader(val_df, cfg, train=False, seed=cfg.seed)
    class_weights = compute_class_weights(train_df["label"], cfg.num_classes) if cfg.use_class_weights else None
    if class_weights is not None:
        logger.info("Class weights: %s", [round(w, 3) for w in class_weights.tolist()])

    _, history = fit(
        model,
        train_loader,
        val_loader,
        cfg,
        device,
        epochs=cfg.pretrain_epochs,
        lr=cfg.pretrain_lr,
        class_weights=class_weights,
        checkpoint_path=cfg.pretrain_checkpoint,
        run_name="pretrain-lunghist700",
        metadata={"stage": "pretrain", "dataset": "LungHist700"},
    )
    history.to_csv(out_dir / "history.csv", index=False)
    plot_training_history(history, out_dir / "training_curves.png", "Pre-training on LungHist700")

    _load_weights(model, cfg.pretrain_checkpoint)
    metrics = evaluate_and_save(
        model, val_loader, val_df, cfg, device, out_dir, "val", "LungHist700 validation (best checkpoint)"
    )
    save_json(metrics, out_dir / "val_metrics.json")
    logger.info("LungHist700 validation: balanced accuracy %.4f, AUC %.4f", metrics["balanced_accuracy"], metrics["auc"])

    del model, train_loader, val_loader
    _free_memory()
    return cfg.pretrain_checkpoint


# ------------------------------------------------------- cross-validation
def load_canine_with_folds(cfg: Config) -> pd.DataFrame:
    """Canine DataFrame with a 1-based ``fold`` column (patient-level, stratified by label)."""
    df = load_image_csv(cfg.canine_csv)
    folds = get_or_create_folds(df, cfg.folds_csv, cfg.n_splits, cfg.seed)
    df = df.merge(folds, on="case_id", how="left", validate="many_to_one")
    if df["fold"].isna().any():
        raise RuntimeError("Some canine images have no fold assignment")
    df["fold"] = df["fold"].astype(int)
    return df


def run_single_setting(
    cfg: Config,
    device: torch.device,
    setting: str,
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    run_dir: Path,
) -> dict[str, Any]:
    """Fine-tune one setting on one fold and evaluate its best checkpoint on the test patients."""
    run_name = f"fold{fold}-{setting}"
    logger.info("--- %s | %s ---", run_name, SETTING_LABELS[setting])

    # Same seed for A and B: identical head initialisation, batch order and augmentations,
    # so the backbone initialisation is the only difference between the two settings.
    seed = cfg.seed + fold
    seed_everything(seed, cfg.deterministic)
    model = build_model(cfg.num_classes, imagenet_pretrained=True)
    if setting == SETTING_LUNGHIST:
        load_pretrained_checkpoint(model, cfg.pretrain_checkpoint, reinit_head=cfg.reinit_head)
    model = model.to(device, memory_format=torch.channels_last if cfg.channels_last else torch.contiguous_format)

    train_loader = make_loader(train_df, cfg, train=True, seed=seed)
    val_loader = make_loader(val_df, cfg, train=False, seed=seed)
    test_loader = make_loader(test_df, cfg, train=False, seed=seed, persistent=False)
    class_weights = compute_class_weights(train_df["label"], cfg.num_classes) if cfg.use_class_weights else None

    checkpoint_path = run_dir / BEST_CHECKPOINT
    best, history = fit(
        model,
        train_loader,
        val_loader,
        cfg,
        device,
        epochs=cfg.finetune_epochs,
        lr=cfg.finetune_lr,
        class_weights=class_weights,
        checkpoint_path=checkpoint_path,
        run_name=run_name,
        metadata={"stage": "finetune", "dataset": "canine", "setting": setting, "fold": fold},
    )
    history.to_csv(run_dir / "history.csv", index=False)
    plot_training_history(history, run_dir / "training_curves.png", f"Fold {fold} - {SETTING_LABELS[setting]}")

    _load_weights(model, checkpoint_path)
    metrics = evaluate_and_save(
        model, test_loader, test_df, cfg, device, run_dir, "test", f"Fold {fold} - {SETTING_LABELS[setting]}"
    )
    result = {
        "fold": fold,
        "setting": setting,
        "best_epoch": best["epoch"],
        "val_balanced_accuracy": best["balanced_accuracy"],
        "n_test_images": len(test_df),
        "n_test_cases": int(test_df["case_id"].nunique()),
        **metrics,
    }
    save_json(result, run_dir / TEST_METRICS)
    logger.info(
        "[%s] TEST balanced accuracy %.4f | sens %.4f | spec %.4f | AUC %.4f",
        run_name,
        metrics["balanced_accuracy"],
        metrics["sensitivity"],
        metrics["specificity"],
        metrics["auc"],
    )

    del model, train_loader, val_loader, test_loader
    _free_memory()
    return result


def run_cross_validation(
    cfg: Config,
    device: torch.device,
    settings: Sequence[str] = SETTINGS,
    folds: Iterable[int] | None = None,
    do_gradcam: bool = True,
    resume: bool = False,
) -> pd.DataFrame | None:
    """Patient-level stratified K-fold CV comparing settings A and B on identical folds."""
    logger.info("=== Canine cross-validation | settings: %s ===", ", ".join(settings))
    if SETTING_LUNGHIST in settings and not cfg.pretrain_checkpoint.is_file():
        raise FileNotFoundError(
            f"Setting B needs the LungHist700 checkpoint {cfg.pretrain_checkpoint}. "
            "Run `python train.py pretrain` first."
        )

    df = load_canine_with_folds(cfg)
    available = sorted(df["fold"].unique().tolist())
    selected = sorted(set(folds)) if folds else available
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"Unknown folds {sorted(unknown)}; available: {available}")

    for fold in selected:
        fold_dir = cfg.cv_dir / f"fold_{fold}"
        test_df = df[df["fold"] == fold].reset_index(drop=True)
        trainval_df = df[df["fold"] != fold].reset_index(drop=True)
        # Inner split is computed once per fold and shared by both settings.
        train_df, val_df = split_train_val_by_group(trainval_df, cfg.inner_val_fraction, cfg.seed + fold)
        assert_patient_disjoint({"train": train_df, "val": val_df, "test": test_df})
        _save_split(fold_dir / "split.csv", train=train_df, val=val_df, test=test_df)
        logger.info("=== Fold %d/%d ===", fold, cfg.n_splits)
        for name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
            _log_split(name, frame)

        for setting in settings:
            run_dir = fold_dir / setting
            if resume and (run_dir / TEST_METRICS).is_file():
                logger.info("Skipping fold %d / %s (already completed, --resume)", fold, setting)
                continue
            run_single_setting(cfg, device, setting, fold, train_df, val_df, test_df, run_dir)

        if do_gradcam:
            _gradcam_for_fold(cfg, device, fold, test_df)

    return summarize_cv(cfg)


def _gradcam_for_fold(cfg: Config, device: torch.device, fold: int, test_df: pd.DataFrame) -> None:
    fold_dir = cfg.cv_dir / f"fold_{fold}"
    checkpoints = {s: fold_dir / s / BEST_CHECKPOINT for s in SETTINGS}
    missing = [str(p) for p in checkpoints.values() if not p.is_file()]
    if missing:
        logger.warning("Grad-CAM fold %d skipped: missing checkpoints %s", fold, missing)
        return
    run_gradcam_comparison(
        test_df,
        checkpoints[SETTING_IMAGENET],
        checkpoints[SETTING_LUNGHIST],
        fold_dir / "gradcam",
        cfg,
        device,
        fold,
    )


def run_gradcam_for_folds(cfg: Config, device: torch.device, folds: Iterable[int] | None = None) -> None:
    """(Re)generate Grad-CAM comparisons from saved fold checkpoints."""
    df = load_canine_with_folds(cfg)
    selected = sorted(set(folds)) if folds else sorted(df["fold"].unique().tolist())
    for fold in selected:
        _gradcam_for_fold(cfg, device, fold, df[df["fold"] == fold].reset_index(drop=True))


# ------------------------------------------------------------------ summary
def summarize_cv(cfg: Config) -> pd.DataFrame | None:
    """Aggregate every completed (fold, setting) run found on disk."""
    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for metrics_file in sorted(cfg.cv_dir.glob(f"fold_*/*/{TEST_METRICS}")):
        result = load_json(metrics_file)
        rows.append(result)
        pred_file = metrics_file.parent / "test_predictions.csv"
        if pred_file.is_file():
            pred = pd.read_csv(pred_file, dtype={"case_id": str})
            pred["fold"] = result["fold"]
            pred["setting"] = result["setting"]
            predictions.append(pred)
    if not rows:
        logger.warning("No completed CV runs found in %s", cfg.cv_dir)
        return None

    per_fold = pd.DataFrame(rows).sort_values(["setting", "fold"]).reset_index(drop=True)
    per_fold.to_csv(cfg.cv_dir / "per_fold_metrics.csv", index=False)
    present = [s for s in SETTINGS if s in set(per_fold["setting"])]

    summary = summarize_metrics(per_fold, METRIC_NAMES)
    deltas: dict[str, tuple[float, float, int]] = {}
    if len(present) == 2:
        for metric in METRIC_NAMES:
            paired = per_fold.pivot(index="fold", columns="setting", values=metric).dropna()
            diff = paired[SETTING_LUNGHIST] - paired[SETTING_IMAGENET]
            std = float(diff.std(ddof=1)) if len(diff) > 1 else float("nan")
            deltas[metric] = (float(diff.mean()), std, len(diff))
        delta_rows = pd.DataFrame(
            [
                {"setting": "delta_B_minus_A", "metric": m, "mean": v[0], "std": v[1], "n_folds": v[2]}
                for m, v in deltas.items()
            ]
        )
        summary = pd.concat([summary, delta_rows], ignore_index=True)
    summary.to_csv(cfg.cv_dir / "summary_metrics.csv", index=False)

    # Markdown table ready to paste into the thesis.
    header = ["Metric"] + [SETTING_LABELS[s] for s in present] + (["Δ (B − A)"] if deltas else [])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for metric in METRIC_NAMES:
        cells = [metric]
        for setting in present:
            row = summary[(summary["setting"] == setting) & (summary["metric"] == metric)].iloc[0]
            cells.append(format_mean_std(row["mean"], row["std"]))
        if deltas:
            cells.append(format_mean_std(deltas[metric][0], deltas[metric][1]))
        lines.append("| " + " | ".join(cells) + " |")
    n_folds = per_fold.groupby("setting")["fold"].nunique().to_dict()
    note = (
        f"\nMean ± sample std over folds (completed folds: {n_folds}). "
        "Δ = per-fold paired difference, LungHist700 minus ImageNet.\n"
    )
    (cfg.cv_dir / "summary.md").write_text("\n".join(lines) + "\n" + note, encoding="utf-8")

    if predictions:
        pooled = pd.concat(predictions, ignore_index=True)
        pooled.to_csv(cfg.cv_dir / "all_test_predictions.csv", index=False)
        cms = {}
        for setting in present:
            subset = pooled[pooled["setting"] == setting]
            cms[SETTING_LABELS[setting]] = confusion_matrix(subset["label"], subset["pred"], labels=[0, 1])
            plot_confusion_matrix(
                cms[SETTING_LABELS[setting]],
                CLASS_NAMES,
                cfg.cv_dir / f"confusion_matrix_{setting}_all_folds.png",
                f"{SETTING_LABELS[setting]} (all test folds)",
            )
        plot_confusion_matrices_side_by_side(
            cms, CLASS_NAMES, cfg.cv_dir / "confusion_matrices_comparison.png", "Pooled test predictions over all folds"
        )

    logger.info("=== Cross-validation summary (mean +/- std over folds) ===")
    for setting in present:
        for metric in METRIC_NAMES:
            row = summary[(summary["setting"] == setting) & (summary["metric"] == metric)].iloc[0]
            logger.info("  %-9s %-18s %.4f +/- %.4f", setting, metric, row["mean"], row["std"])
    for metric, (mean, std, n) in deltas.items():
        logger.info("  delta B-A %-18s %+.4f +/- %.4f (n=%d)", metric, mean, std, n)
    logger.info("Summary written to %s", cfg.cv_dir / "summary.md")
    return summary
