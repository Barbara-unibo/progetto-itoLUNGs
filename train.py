"""Entry point.

Examples
--------
    python train.py prepare                 # build data/canine.csv from the raw spreadsheet
    python train.py pretrain                # ImageNet -> LungHist700 (human only)
    python train.py cv                      # 5-fold CV: settings A and B + Grad-CAM + summary
    python train.py all                     # pretrain + cv
    python train.py gradcam --folds 1 2     # regenerate Grad-CAM figures from checkpoints
    python train.py summarize               # rebuild summary tables/figures from saved runs

Run ``python train.py <command> -h`` for all options.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.config import SETTINGS, Config
from src.dataset import build_canine_csv
from src.train import run_cross_validation, run_gradcam_for_folds, run_pretraining, summarize_cv
from src.utils import configure_torch, get_device, log_environment, save_json, setup_logging

logger = logging.getLogger("train")


def _common_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("general")
    g.add_argument("--output-dir", type=Path, help="results directory (default: results/)")
    g.add_argument("--seed", type=int)
    g.add_argument("--img-size", type=int)
    g.add_argument("--batch-size", type=int)
    g.add_argument("--num-workers", type=int)
    g.add_argument("--no-amp", dest="amp", action="store_false", default=None, help="disable mixed precision")
    g.add_argument("--amp-dtype", choices=["float16", "bfloat16"])
    g.add_argument("--deterministic", action="store_true", default=None, help="bit-reproducible (slower)")
    g.add_argument("--no-class-weights", dest="use_class_weights", action="store_false", default=None)
    g.add_argument("--label-smoothing", type=float)
    g.add_argument("--weight-decay", type=float)
    return p


def _canine_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--canine-csv", type=Path, help="case_id,image_path,label CSV (default: data/canine.csv)")
    return p


def _pretrain_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("pre-training (LungHist700)")
    g.add_argument("--lunghist-root", type=Path, help="LungHist700 folder (default: data/LungHist700)")
    g.add_argument("--lunghist-csv", type=Path, help="optional case_id,image_path,label index for LungHist700")
    g.add_argument("--pretrain-epochs", type=int)
    g.add_argument("--pretrain-lr", type=float)
    g.add_argument("--pretrain-val-fraction", type=float)
    return p


def _folds_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--folds", type=int, nargs="+", help="1-based folds to run (default: all)")
    return p


def _gradcam_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("Grad-CAM")
    g.add_argument("--gradcam-per-class", type=int, help="test images per class per fold")
    g.add_argument("--gradcam-target", choices=["true", "pred"], help="class explained by the CAM")
    return p


def _cv_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("cross-validation / fine-tuning (canine)")
    g.add_argument("--settings", nargs="+", choices=list(SETTINGS), default=list(SETTINGS))
    g.add_argument("--finetune-epochs", type=int)
    g.add_argument("--finetune-lr", type=float)
    g.add_argument("--n-splits", type=int)
    g.add_argument("--inner-val-fraction", type=float)
    g.add_argument("--pretrain-checkpoint", dest="pretrain_checkpoint_path", type=Path)
    g.add_argument(
        "--keep-pretrained-head",
        dest="reinit_head",
        action="store_false",
        default=None,
        help="setting B keeps the LungHist700 classification head",
    )
    g.add_argument("--resume", action="store_true", help="skip (fold, setting) runs that already finished")
    g.add_argument("--no-gradcam", dest="gradcam", action="store_false", help="skip Grad-CAM figures")
    return p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ConvNeXt-Tiny: ImageNet vs. LungHist700 pre-training for canine lung histology.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    common, canine = _common_parser(), _canine_parser()

    prepare = sub.add_parser("prepare", parents=[common, canine], help="build the canine CSV")
    prepare.add_argument("--raw-csv", dest="canine_raw_csv", type=Path, help="raw spreadsheet export")
    prepare.add_argument("--image-dirs", dest="canine_image_dirs", type=Path, nargs="+", help="canine image folders")

    sub.add_parser("pretrain", parents=[common, _pretrain_parser()], help="pre-train on LungHist700")
    sub.add_parser(
        "cv", parents=[common, canine, _cv_parser(), _folds_parser(), _gradcam_parser()], help="canine cross-validation"
    )
    sub.add_parser(
        "all",
        parents=[common, canine, _pretrain_parser(), _cv_parser(), _folds_parser(), _gradcam_parser()],
        help="pretrain + cv",
    )
    sub.add_parser(
        "gradcam", parents=[common, canine, _folds_parser(), _gradcam_parser()], help="Grad-CAM from checkpoints"
    )
    sub.add_parser("summarize", parents=[common], help="rebuild the CV summary")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    """Override :class:`Config` fields with every CLI value that was provided."""
    cfg = Config()
    for key, value in vars(args).items():
        if value is not None and key in Config.__dataclass_fields__:
            setattr(cfg, key, tuple(value) if isinstance(value, list) else value)
    cfg.validate()
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = cfg.output_dir / "logs"
    setup_logging(log_dir / f"{args.command}_{stamp}.log")
    save_json(cfg.to_dict(), log_dir / f"config_{args.command}_{stamp}.json")
    logger.info("Command: %s", " ".join(sys.argv))

    try:
        if args.command == "prepare":
            build_canine_csv(cfg.canine_raw_csv, cfg.canine_image_dirs, cfg.canine_csv)
            return 0
        if args.command == "summarize":
            summarize_cv(cfg)
            return 0

        configure_torch()
        device = get_device()
        log_environment(device)

        if args.command in ("pretrain", "all"):
            run_pretraining(cfg, device)
        if args.command in ("cv", "all"):
            run_cross_validation(
                cfg,
                device,
                settings=args.settings,
                folds=args.folds,
                do_gradcam=args.gradcam,
                resume=args.resume,
            )
        if args.command == "gradcam":
            run_gradcam_for_folds(cfg, device, args.folds)
        return 0
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception:
        logger.exception("Run failed")
        return 1


if __name__ == "__main__":  # required on Windows: DataLoader workers re-import this module
    sys.exit(main())
