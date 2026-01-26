from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        from transformers import AutoModel  # type: ignore

        self.model_name = model_name
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = getattr(self.backbone.config, "hidden_size", None)
        if hidden is None:
            raise ValueError("Cannot infer hidden_size from transformer config")
        self.out_dim = int(hidden)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = getattr(out, "pooler_output", None)
        if pooled is not None:
            return pooled
        return out.last_hidden_state[:, 0]


class ImageEncoder(nn.Module):
    def __init__(self, name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        from torchvision import models  # type: ignore

        self.name = name
        if name == "resnet18":
            if pretrained:
                try:
                    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                except Exception:
                    try:
                        backbone = models.resnet18(pretrained=True)
                    except Exception:
                        backbone = models.resnet18(weights=None)
            else:
                backbone = models.resnet18(weights=None)
            out_dim = int(backbone.fc.in_features)
            backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported image encoder: {name}")
        self.backbone = backbone
        self.out_dim = out_dim

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)


class TextOnlyModel(nn.Module):
    def __init__(self, text_model_name: str, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encoder = TextEncoder(text_model_name)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.encoder.out_dim, num_classes))

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        x = self.encoder(batch["input_ids"], batch["attention_mask"])
        return self.head(x)


class ImageOnlyModel(nn.Module):
    def __init__(
        self,
        image_encoder_name: str = "resnet18",
        pretrained: bool = True,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = ImageEncoder(image_encoder_name, pretrained=pretrained)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.encoder.out_dim, num_classes))

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        x = self.encoder(batch["image"])
        return self.head(x)


class MultiModalGatedFusionModel(nn.Module):
    def __init__(
        self,
        text_model_name: str,
        image_encoder_name: str = "resnet18",
        pretrained_image: bool = True,
        d: int = 256,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_encoder = TextEncoder(text_model_name)
        self.image_encoder = ImageEncoder(image_encoder_name, pretrained=pretrained_image)
        self.text_proj = nn.Linear(self.text_encoder.out_dim, d)
        self.image_proj = nn.Linear(self.image_encoder.out_dim, d)
        self.gate = nn.Linear(d * 2, d)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, num_classes))

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        t = self.text_encoder(batch["input_ids"], batch["attention_mask"])
        v = self.image_encoder(batch["image"])
        t = self.text_proj(t)
        v = self.image_proj(v)
        g = torch.sigmoid(self.gate(torch.cat([t, v], dim=-1)))
        h = g * t + (1.0 - g) * v
        return self.head(h)

