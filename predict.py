from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from multi_affective.config import make_image_transform
from multi_affective.models import (
    ImageOnlyModel,
    MultiModalConcatFusionModel,
    MultiModalGatedFusionModel,
    MultiModalLateFusionModel,
    TextOnlyModel,
)
from multi_affective.text_preprocess import clean_text


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("./datasets"))
    parser.add_argument("--test-file", type=Path, default=Path("./datasets/test_without_label.txt"))
    parser.add_argument("--output-file", type=Path, default=Path("./outputs/test_predictions.txt"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class TestIndex:
    def __init__(self, test_file: Path):
        self.header: list[str] | None = None
        self.guids: list[str] = []
        with test_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for i, parts in enumerate(reader):
                if not parts:
                    continue
                first = parts[0].strip()
                if i == 0 and first.lower() == "guid":
                    self.header = [p.strip() for p in parts]
                    continue
                if not first:
                    continue
                self.guids.append(first)

    def __len__(self) -> int:
        return len(self.guids)

    def __getitem__(self, idx: int) -> str:
        return self.guids[idx]


class TestTextDataset(Dataset):
    def __init__(self, data_dir: Path, index: TestIndex, tokenizer_name: str, max_len: int):
        from transformers import AutoTokenizer  # type: ignore

        self.data_dir = data_dir
        self.index = index
        self.tokenizer_name = tokenizer_name
        self.max_len = int(max_len)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        guid = self.index[idx]
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
        }


class TestImageDataset(Dataset):
    def __init__(self, data_dir: Path, index: TestIndex, transform: Any):
        self.data_dir = data_dir
        self.index = index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        guid = self.index[idx]
        image_path = self.data_dir / "data" / f"{guid}.jpg"
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            x = self.transform(im)
        return {"guid": guid, "image": x}


class TestMultiDataset(Dataset):
    def __init__(self, text_ds: TestTextDataset, image_ds: TestImageDataset):
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
        }


def _normalize_id2label(obj: Any) -> dict[int, str]:
    if not isinstance(obj, dict):
        raise ValueError("checkpoint id2label must be a dict")
    out: dict[int, str] = {}
    for k, v in obj.items():
        try:
            kk = int(k)
        except Exception as e:
            raise ValueError(f"Invalid id2label key: {k}") from e
        out[kk] = str(v)
    return out


def _build_model_from_checkpoint(payload: dict[str, Any]) -> tuple[torch.nn.Module, dict[int, str], dict[str, Any]]:
    run_cfg = payload.get("run_config", None)
    if not isinstance(run_cfg, dict):
        raise ValueError("Missing run_config in checkpoint")
    mode = str(payload.get("mode", run_cfg.get("mode", "")))
    if not mode:
        raise ValueError("Missing mode in checkpoint")
    id2label = _normalize_id2label(payload.get("id2label", {}))

    dropout = float(run_cfg.get("dropout", 0.1))
    if mode == "text":
        model = TextOnlyModel(text_model_name=str(run_cfg["text_model"]), dropout=dropout)
    elif mode == "image":
        model = ImageOnlyModel(
            image_encoder_name=str(run_cfg.get("image_encoder", "resnet18")),
            pretrained=bool(run_cfg.get("pretrained_image", True)),
            dropout=dropout,
        )
    elif mode == "multimodal":
        arch = str(run_cfg.get("multimodal_arch", "gated"))
        if arch == "gated":
            model = MultiModalGatedFusionModel(
                text_model_name=str(run_cfg["text_model"]),
                image_encoder_name=str(run_cfg.get("image_encoder", "resnet18")),
                pretrained_image=bool(run_cfg.get("pretrained_image", True)),
                d=int(run_cfg.get("fusion_dim", 256)),
                dropout=dropout,
            )
        elif arch == "concat":
            model = MultiModalConcatFusionModel(
                text_model_name=str(run_cfg["text_model"]),
                image_encoder_name=str(run_cfg.get("image_encoder", "resnet18")),
                pretrained_image=bool(run_cfg.get("pretrained_image", True)),
                d=int(run_cfg.get("fusion_dim", 256)),
                dropout=dropout,
            )
        elif arch == "late":
            model = MultiModalLateFusionModel(
                text_model_name=str(run_cfg["text_model"]),
                image_encoder_name=str(run_cfg.get("image_encoder", "resnet18")),
                pretrained_image=bool(run_cfg.get("pretrained_image", True)),
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported multimodal architecture in checkpoint: {arch}")
    else:
        raise ValueError(f"Unsupported mode in checkpoint: {mode}")

    state = payload.get("model_state_dict", None)
    if not isinstance(state, dict):
        raise ValueError("Missing model_state_dict in checkpoint")
    cleaned_state = {k: v for k, v in state.items() if not str(k).startswith("_loss_fn.")}
    model.load_state_dict(cleaned_state, strict=True)
    return model, id2label, run_cfg


def main() -> int:
    args = _build_parser().parse_args()
    device = torch.device(_resolve_device(str(args.device)))

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict produced by train.py")
    model, id2label, run_cfg = _build_model_from_checkpoint(ckpt)
    model.to(device)
    model.eval()

    preprocess_dir = Path(str(run_cfg.get("preprocess_dir", "./outputs/preprocess")))
    test_index = TestIndex(args.test_file)

    mode = str(ckpt.get("mode", run_cfg.get("mode", "")))
    if mode == "text":
        test_ds: Dataset = TestTextDataset(
            args.data_dir,
            test_index,
            tokenizer_name=str(run_cfg["text_model"]),
            max_len=int(run_cfg.get("max_len", 128)),
        )
    elif mode == "image":
        transform = make_image_transform(preprocess_dir, aug="baseline", train=False)
        test_ds = TestImageDataset(args.data_dir, test_index, transform=transform)
    elif mode == "multimodal":
        transform = make_image_transform(preprocess_dir, aug="baseline", train=False)
        text_ds = TestTextDataset(
            args.data_dir,
            test_index,
            tokenizer_name=str(run_cfg["text_model"]),
            max_len=int(run_cfg.get("max_len", 128)),
        )
        img_ds = TestImageDataset(args.data_dir, test_index, transform=transform)
        test_ds = TestMultiDataset(text_ds, img_ds)
    else:
        raise ValueError(f"Unsupported mode for prediction: {mode}")

    loader = DataLoader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    preds: list[str] = []
    with torch.no_grad():
        for batch in loader:
            guids = batch["guid"]
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            logits = model(batch)
            p = torch.argmax(logits, dim=-1).detach().cpu().tolist()
            for i, pid in enumerate(p):
                pred_tag = id2label.get(int(pid), str(pid))
                _ = guids[i]
                preds.append(pred_tag)

    if len(preds) != len(test_index):
        raise RuntimeError("Prediction count mismatch with test file rows")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if test_index.header is not None:
            writer.writerow(["guid", "tag"])
        for guid, tag in zip(test_index.guids, preds):
            writer.writerow([guid, tag])

    payload = {
        "checkpoint": str(args.checkpoint),
        "mode": mode,
        "rows": len(test_index),
        "output_file": str(args.output_file),
        "device": str(device),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
