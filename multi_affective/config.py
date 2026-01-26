from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import read_json


def get_text_cfg(preprocess_dir: Path, override_model: str | None) -> tuple[str, int]:
    cfg = read_json(preprocess_dir / "text_config.json")
    if cfg is None:
        tokenizer_name = override_model or "./models/bert-base-uncased"
        return tokenizer_name, 128
    tokenizer_name = override_model or str(cfg.get("tokenizer_name", "./models/bert-base-uncased"))
    max_len = int(cfg.get("max_len", 128))
    return tokenizer_name, max_len


def make_image_transform(preprocess_dir: Path):
    from torchvision import transforms  # type: ignore

    image_cfg: dict[str, Any] | None = read_json(preprocess_dir / "image_config.json")
    if image_cfg is None:
        resize_shorter = 256
        crop_size = 224
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        resize_shorter = int(image_cfg.get("resize_shorter", 256))
        crop_size = int(image_cfg.get("crop_size", 224))
        mean = tuple(image_cfg.get("mean", [0.485, 0.456, 0.406]))
        std = tuple(image_cfg.get("std", [0.229, 0.224, 0.225]))
    return transforms.Compose(
        [
            transforms.Resize(resize_shorter),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

