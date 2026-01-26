from __future__ import annotations

from dataclasses import dataclass

from torchvision import transforms


@dataclass(frozen=True)
class ImagePreprocessConfig:
    resize_shorter: int
    crop_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


def build_baseline_image_transform(cfg: ImagePreprocessConfig) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(cfg.resize_shorter),
            transforms.CenterCrop(cfg.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )


def default_image_preprocess_config(*, crop_size: int = 224) -> ImagePreprocessConfig:
    return ImagePreprocessConfig(
        resize_shorter=256,
        crop_size=crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

