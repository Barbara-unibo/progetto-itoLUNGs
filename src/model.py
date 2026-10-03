"""ConvNeXt-Tiny model factory and checkpoint helpers (same architecture for both phases)."""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

logger = logging.getLogger(__name__)

# torchvision ConvNeXt classifier = Sequential(LayerNorm2d, Flatten, Linear); index 2 is the head.
HEAD_PREFIX = "classifier.2."


def build_model(num_classes: int = 2, imagenet_pretrained: bool = True) -> nn.Module:
    """ConvNeXt-Tiny with a freshly initialised ``num_classes`` linear head."""
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
    model = convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


def _read_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    # Checkpoints are produced by this project (trusted), and also store config/metrics.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint.get("model_state", checkpoint)


def load_pretrained_checkpoint(model: nn.Module, checkpoint_path: Path, reinit_head: bool = True) -> nn.Module:
    """Initialise ``model`` from the LungHist700 pre-trained checkpoint.

    With ``reinit_head=True`` only the backbone (and final LayerNorm) is loaded and
    the classification head keeps its fresh initialisation.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Pre-trained checkpoint not found: {checkpoint_path}")
    state = _read_state_dict(checkpoint_path)
    if reinit_head:
        state = {k: v for k, v in state.items() if not k.startswith(HEAD_PREFIX)}
        result = model.load_state_dict(state, strict=False)
        missing = [k for k in result.missing_keys if not k.startswith(HEAD_PREFIX)]
        if missing or result.unexpected_keys:
            raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={result.unexpected_keys}")
    else:
        model.load_state_dict(state, strict=True)
    logger.info("Loaded LungHist700 weights from %s (head %s)", checkpoint_path, "re-initialised" if reinit_head else "kept")
    return model


def load_model_from_checkpoint(checkpoint_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    """Rebuild a trained model for inference / Grad-CAM (eval mode)."""
    model = build_model(num_classes, imagenet_pretrained=False)
    model.load_state_dict(_read_state_dict(Path(checkpoint_path)), strict=True)
    return model.to(device).eval()


def get_gradcam_target_layers(model: nn.Module) -> list[nn.Module]:
    """Last ConvNeXt stage (16x16 feature map for a 512x512 input)."""
    return [model.features[-1]]


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
