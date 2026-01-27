from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from .labels import LABEL2ID
from .text_preprocess import clean_text


class LabeledIndex(Dataset):
    def __init__(self, index_path: Path):
        self.rows: list[tuple[str, int]] = []
        with index_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for i, parts in enumerate(reader):
                if not parts:
                    continue
                if i == 0 and parts[0].strip().lower() == "guid":
                    continue
                if len(parts) < 2:
                    continue
                guid = parts[0].strip()
                tag = parts[1].strip().lower()
                if not guid:
                    continue
                if tag not in LABEL2ID:
                    raise ValueError(f"Unknown tag '{tag}' in {index_path}")
                self.rows.append((guid, LABEL2ID[tag]))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.rows[idx]


def _augment_text(text: str, aug: str, rng: random.Random) -> str:
    if aug == "baseline":
        return text
    tokens = text.split()
    if len(tokens) < 2:
        return text
    if aug == "weak":
        kept = [t for t in tokens if rng.random() > 0.1]
        if not kept:
            kept = [tokens[rng.randrange(len(tokens))]]
        if len(kept) >= 2 and rng.random() < 0.2:
            i = rng.randrange(len(kept) - 1)
            kept[i], kept[i + 1] = kept[i + 1], kept[i]
        return " ".join(kept)
    if aug == "strong":
        kept = [t for t in tokens if rng.random() > 0.2]
        if not kept:
            kept = [tokens[rng.randrange(len(tokens))]]
        for i in range(len(kept) - 1):
            if rng.random() < 0.3:
                kept[i], kept[i + 1] = kept[i + 1], kept[i]
        if rng.random() < 0.1:
            kept.insert(rng.randrange(len(kept) + 1), kept[rng.randrange(len(kept))])
        return " ".join(kept)
    raise ValueError(f"Unsupported text augmentation: {aug}")


class TextDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        index: LabeledIndex,
        tokenizer_name: str,
        max_len: int,
        *,
        text_aug: str = "baseline",
        train: bool = False,
    ):
        from transformers import AutoTokenizer  # type: ignore

        self.data_dir = data_dir
        self.index = index
        self.tokenizer_name = tokenizer_name
        self.max_len = int(max_len)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.text_aug = str(text_aug)
        self.train = bool(train)
        self.rng = random.Random()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        guid, y = self.index[idx]
        text_path = self.data_dir / "data" / f"{guid}.txt"
        raw = text_path.read_text(encoding="utf-8", errors="ignore")
        text = clean_text(raw)
        if self.train and self.text_aug != "baseline":
            text = _augment_text(text, self.text_aug, self.rng)
        encoded = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "guid": guid,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(y, dtype=torch.long),
        }


class ImageDataset(Dataset):
    def __init__(self, data_dir: Path, index: LabeledIndex, transform: Any):
        self.data_dir = data_dir
        self.index = index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        guid, y = self.index[idx]
        image_path = self.data_dir / "data" / f"{guid}.jpg"
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            x = self.transform(im)
        return {"guid": guid, "image": x, "label": torch.tensor(y, dtype=torch.long)}


class MultiDataset(Dataset):
    def __init__(self, text_ds: TextDataset, image_ds: ImageDataset):
        if len(text_ds) != len(image_ds):
            raise ValueError("Text and image datasets must have same length")
        self.text_ds = text_ds
        self.image_ds = image_ds

    def __len__(self) -> int:
        return len(self.text_ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        t = self.text_ds[idx]
        v = self.image_ds[idx]
        if t["guid"] != v["guid"]:
            raise ValueError("Text/image guid mismatch")
        return {
            "guid": t["guid"],
            "input_ids": t["input_ids"],
            "attention_mask": t["attention_mask"],
            "image": v["image"],
            "label": t["label"],
        }

