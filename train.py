from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from multi_affective import seed_everything
from multi_affective.config import get_text_cfg, make_image_transform
from multi_affective.data import ImageDataset, LabeledIndex, MultiDataset, TextDataset
from multi_affective.io_utils import save_csv, save_json
from multi_affective.labels import ID2LABEL, LABEL2ID
from multi_affective.models import ImageOnlyModel, MultiModalGatedFusionModel, TextOnlyModel
from multi_affective.training import (
    compute_class_weights,
    evaluate,
    freeze_module,
    plot_confusion_matrix,
    plot_curves,
    save_history,
    split_params,
    train_one_epoch,
)


Mode = Literal["text", "image", "multimodal", "all"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="multimodal", choices=["text", "image", "multimodal", "all"])
    parser.add_argument("--data-dir", type=Path, default=Path("./datasets"))
    parser.add_argument("--train-index", type=Path, default=Path("./outputs/splits/train_split.txt"))
    parser.add_argument("--val-index", type=Path, default=Path("./outputs/splits/val_split.txt"))
    parser.add_argument("--preprocess-dir", type=Path, default=Path("./outputs/preprocess"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fusion-dim", type=int, default=256)
    parser.add_argument("--text-model", type=str, default=None)
    parser.add_argument("--image-encoder", type=str, default="resnet18")
    parser.add_argument("--no-pretrained-image", action="store_true")
    parser.add_argument("--lr-encoder", type=float, default=2e-5)
    parser.add_argument("--lr-head", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--freeze-encoders", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=2)
    return parser


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _run_one_mode(mode: Mode, args: argparse.Namespace, device: torch.device, run_root: Path) -> dict[str, Any]:
    train_index = LabeledIndex(args.train_index)
    val_index = LabeledIndex(args.val_index)

    class_weights = compute_class_weights(train_index.rows, num_classes=len(LABEL2ID)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    text_model_name, max_len = get_text_cfg(args.preprocess_dir, args.text_model)
    image_transform = make_image_transform(args.preprocess_dir)
    pretrained_image = not bool(args.no_pretrained_image)

    if mode == "text":
        train_ds = TextDataset(args.data_dir, train_index, tokenizer_name=text_model_name, max_len=max_len)
        val_ds = TextDataset(args.data_dir, val_index, tokenizer_name=text_model_name, max_len=max_len)
        model: nn.Module = TextOnlyModel(text_model_name=text_model_name, dropout=float(args.dropout))
        head_keywords = ("head",)
        if args.freeze_encoders:
            freeze_module(getattr(model, "encoder"))
    elif mode == "image":
        train_ds = ImageDataset(args.data_dir, train_index, transform=image_transform)
        val_ds = ImageDataset(args.data_dir, val_index, transform=image_transform)
        model = ImageOnlyModel(
            image_encoder_name=str(args.image_encoder),
            pretrained=pretrained_image,
            dropout=float(args.dropout),
        )
        head_keywords = ("head",)
        if args.freeze_encoders:
            freeze_module(getattr(model, "encoder"))
    elif mode == "multimodal":
        train_text = TextDataset(args.data_dir, train_index, tokenizer_name=text_model_name, max_len=max_len)
        val_text = TextDataset(args.data_dir, val_index, tokenizer_name=text_model_name, max_len=max_len)
        train_img = ImageDataset(args.data_dir, train_index, transform=image_transform)
        val_img = ImageDataset(args.data_dir, val_index, transform=image_transform)
        train_ds = MultiDataset(train_text, train_img)
        val_ds = MultiDataset(val_text, val_img)
        model = MultiModalGatedFusionModel(
            text_model_name=text_model_name,
            image_encoder_name=str(args.image_encoder),
            pretrained_image=pretrained_image,
            d=int(args.fusion_dim),
            dropout=float(args.dropout),
        )
        head_keywords = ("text_proj", "image_proj", "gate", "head")
        if args.freeze_encoders:
            freeze_module(getattr(model, "text_encoder"))
            freeze_module(getattr(model, "image_encoder"))
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    setattr(model, "_loss_fn", loss_fn)
    model.to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    encoder_params, head_params = split_params(model, head_keywords=head_keywords)
    param_groups: list[dict[str, Any]] = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": float(args.lr_encoder)})
    if head_params:
        param_groups.append({"params": head_params, "lr": float(args.lr_head)})
    if not param_groups:
        raise RuntimeError("No trainable parameters. Disable --freeze-encoders or check model.")
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(args.weight_decay))

    out_dir = run_root / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "mode": mode,
        "data_dir": str(args.data_dir),
        "train_index": str(args.train_index),
        "val_index": str(args.val_index),
        "preprocess_dir": str(args.preprocess_dir),
        "text_model": text_model_name,
        "max_len": int(max_len),
        "image_encoder": str(args.image_encoder),
        "pretrained_image": pretrained_image,
        "dropout": float(args.dropout),
        "fusion_dim": int(args.fusion_dim),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr_encoder": float(args.lr_encoder),
        "lr_head": float(args.lr_head),
        "weight_decay": float(args.weight_decay),
        "freeze_encoders": bool(args.freeze_encoders),
        "class_weights": class_weights.detach().cpu().tolist(),
        "seed": int(args.seed),
        "deterministic": bool(args.deterministic),
        "device": str(device),
        "torch_version": torch.__version__,
    }
    save_json(out_dir / "run_config.json", run_config)

    history: list[dict[str, Any]] = []
    best = {"epoch": 0, "val_macro_f1": -1.0}
    patience_left = int(args.early_stop_patience)

    for epoch in range(1, int(args.epochs) + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, device)
        val_stats = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["acc"],
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["acc"],
            "val_macro_f1": val_stats["macro_f1"],
        }
        history.append(row)
        save_history(out_dir, history)

        if val_stats["macro_f1"] > float(best["val_macro_f1"]):
            best = {"epoch": epoch, "val_macro_f1": float(val_stats["macro_f1"])}
            torch.save(
                {
                    "mode": mode,
                    "model_state_dict": model.state_dict(),
                    "run_config": run_config,
                    "label2id": LABEL2ID,
                    "id2label": ID2LABEL,
                    "best": best,
                },
                out_dir / "best.pt",
            )
            plot_confusion_matrix(val_stats["confusion_matrix"], out_dir / "confusion_matrix.png")
            patience_left = int(args.early_stop_patience)
        else:
            patience_left -= 1
            if patience_left < 0:
                break

    plot_curves(history, out_dir / "curves.png")
    final_val = evaluate(model, val_loader, device)
    save_json(out_dir / "final_metrics.json", {"best": best, "final": final_val, "history_tail": history[-5:]})
    return {"mode": mode, "best_epoch": best["epoch"], "best_val_macro_f1": best["val_macro_f1"], "final": final_val}


def main() -> int:
    args = _build_parser().parse_args()
    seed_everything(args.seed, deterministic=args.deterministic)

    device = torch.device(_resolve_device(args.device))
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.output_dir / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    modes: list[Mode]
    if args.mode == "all":
        modes = ["text", "image", "multimodal"]
    else:
        modes = [args.mode]

    summary: list[dict[str, Any]] = []
    for m in modes:
        summary.append(_run_one_mode(m, args, device, run_root))
    save_json(run_root / "summary.json", summary)
    save_csv(run_root / "summary.csv", summary)
    print(json.dumps({"run_root": str(run_root), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

