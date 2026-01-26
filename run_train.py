from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from typing import Any

import train as train_module


@dataclass(frozen=True)
class TrainConfig:
    mode: str = "multimodal"
    data_dir: str = "./datasets"
    train_index: str = "./outputs/splits/train_split.txt"
    val_index: str = "./outputs/splits/val_split.txt"
    preprocess_dir: str = "./outputs/preprocess"
    output_dir: str = "./outputs"
    run_name: str | None = "baseline"
    seed: int = 42
    deterministic: bool = True
    device: str = "auto"
    epochs: int = 5
    batch_size: int = 16
    num_workers: int = 0
    dropout: float = 0.1
    fusion_dim: int = 256
    text_model: str | None = "./models/bert-base-uncased"
    image_encoder: str = "resnet18"
    no_pretrained_image: bool = False
    lr_encoder: float = 2e-5
    lr_head: float = 2e-4
    weight_decay: float = 0.01
    freeze_encoders: bool = False
    early_stop_patience: int = 2


def _build_argv(cfg: TrainConfig) -> list[str]:
    args: list[str] = [
        "--mode",
        str(cfg.mode),
        "--data-dir",
        str(cfg.data_dir),
        "--train-index",
        str(cfg.train_index),
        "--val-index",
        str(cfg.val_index),
        "--preprocess-dir",
        str(cfg.preprocess_dir),
        "--output-dir",
        str(cfg.output_dir),
        "--seed",
        str(int(cfg.seed)),
        "--device",
        str(cfg.device),
        "--epochs",
        str(int(cfg.epochs)),
        "--batch-size",
        str(int(cfg.batch_size)),
        "--num-workers",
        str(int(cfg.num_workers)),
        "--dropout",
        str(float(cfg.dropout)),
        "--fusion-dim",
        str(int(cfg.fusion_dim)),
        "--image-encoder",
        str(cfg.image_encoder),
        "--lr-encoder",
        str(float(cfg.lr_encoder)),
        "--lr-head",
        str(float(cfg.lr_head)),
        "--weight-decay",
        str(float(cfg.weight_decay)),
        "--early-stop-patience",
        str(int(cfg.early_stop_patience)),
    ]

    if cfg.run_name:
        args.extend(["--run-name", str(cfg.run_name)])
    if cfg.text_model:
        args.extend(["--text-model", str(cfg.text_model)])
    if cfg.deterministic:
        args.append("--deterministic")
    if cfg.no_pretrained_image:
        args.append("--no-pretrained-image")
    if cfg.freeze_encoders:
        args.append("--freeze-encoders")
    return args


def run_training(cfg: TrainConfig, *, extra_overrides: dict[str, Any] | None = None) -> int:
    if extra_overrides:
        cfg = replace(cfg, **extra_overrides)
    old_argv = sys.argv[:]
    try:
        sys.argv = ["train.py", *_build_argv(cfg)]
        return int(train_module.main())
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    run_training(TrainConfig(
        mode="all",
        run_name="baseline",
        seed=42,
        deterministic=True,
        device="auto",
        epochs=50,
        batch_size=256,
        num_workers=0,
        dropout=0.1,
        fusion_dim=256,
        text_model="./models/bert-base-uncased",
        image_encoder="resnet18",
        no_pretrained_image=False,
        lr_encoder=2e-5,
        lr_head=2e-4,
        weight_decay=0.01,
        freeze_encoders=False,
        early_stop_patience=4,
    ))
