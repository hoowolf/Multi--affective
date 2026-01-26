from __future__ import annotations

import csv
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


class TextDataset(Dataset):
    def __init__(self, data_dir: Path, index: LabeledIndex, tokenizer_name: str, max_len: int):
        from transformers import AutoTokenizer  # type: ignore

        self.data_dir = data_dir
        self.index = index
        self.tokenizer_name = tokenizer_name
        self.max_len = int(max_len)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        guid, y = self.index[idx]
        text_path = self.data_dir / "data" / f"{guid}.txt"
        raw = text_path.read_text(encoding="utf-8", errors="ignore")
        text = clean_text(raw)
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

