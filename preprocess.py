from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

from multi_affective.image_preprocess import default_image_preprocess_config
from multi_affective.stats import quantiles
from multi_affective.text_preprocess import clean_text


ALLOWED = {"positive", "neutral", "negative"}
TYPO_MAP = {"netative": "negative"}


@dataclass(frozen=True)
class LabeledRow:
    guid: str
    tag: str


@dataclass(frozen=True)
class TextPreprocessConfig:
    tokenizer_name: str
    max_len: int


def _read_labeled_index(path: Path) -> list[LabeledRow]:
    rows: list[LabeledRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for i, parts in enumerate(reader):
            if not parts:
                continue
            if i == 0 and parts[0].strip().lower() == "guid":
                continue
            if len(parts) < 2:
                continue
            guid = parts[0].strip()
            tag = parts[1].strip()
            if guid and tag:
                rows.append(LabeledRow(guid=guid, tag=tag))
    return rows


def _normalize_tag(tag: str) -> tuple[str, bool]:
    t = tag.strip().lower()
    mapped = TYPO_MAP.get(t, t)
    return mapped, mapped != t


def _write_labeled_index(path: Path, rows: list[LabeledRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["guid", "tag"])
        for r in rows:
            writer.writerow([r.guid, r.tag])


def _read_guid_only_index(path: Path) -> list[str]:
    guids: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for i, parts in enumerate(reader):
            if not parts:
                continue
            if i == 0 and parts[0].strip().lower() == "guid":
                continue
            guid = parts[0].strip()
            if guid:
                guids.append(guid)
    return guids


def _token_lengths_with_transformers(texts: list[str], tokenizer_name: str = "./models/bert-base-uncased") -> list[int] | None:
    try:
        from transformers import AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        return [len(tokenizer(t, add_special_tokens=True, truncation=False)["input_ids"]) for t in texts]
    except Exception:
        return None


def _clean_labels(train_file: Path, output_dir: Path, *, drop_invalid: bool) -> dict:
    src_rows = _read_labeled_index(train_file)
    raw_counter = Counter([r.tag for r in src_rows])
    normalized_counter = Counter()
    fixed_typo = 0
    invalid = 0

    cleaned: list[LabeledRow] = []
    for r in src_rows:
        normalized, changed_by_typo = _normalize_tag(r.tag)
        if changed_by_typo:
            fixed_typo += 1
        normalized_counter[normalized] += 1
        if normalized not in ALLOWED:
            invalid += 1
            if drop_invalid:
                continue
        cleaned.append(LabeledRow(guid=r.guid, tag=normalized))

    indices_dir = output_dir / "indices"
    clean_index = indices_dir / "train_clean.txt"
    report_file = indices_dir / "train_clean_report.json"
    _write_labeled_index(clean_index, cleaned)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "train_file": str(train_file),
        "output_file": str(clean_index),
        "drop_invalid": bool(drop_invalid),
        "rows_before": len(src_rows),
        "rows_after": len(cleaned),
        "fixed_typo_count": fixed_typo,
        "invalid_tag_count": invalid,
        "raw_tag_counts": dict(raw_counter),
        "normalized_tag_counts": dict(normalized_counter),
        "allowed": sorted(ALLOWED),
        "typo_map": TYPO_MAP,
    }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"clean_index": clean_index, "clean_report": report_file, "report": report}


def _build_preprocess_configs(
    data_dir: Path,
    train_clean_index: Path,
    output_dir: Path,
    *,
    tokenizer_name: str,
    default_max_len: int,
    image_crop_size: int,
    image_sample: int,
) -> dict:
    data_files_dir = data_dir / "data"
    guids = _read_guid_only_index(train_clean_index)

    texts: list[str] = []
    missing_text = 0
    for guid in guids:
        p = data_files_dir / f"{guid}.txt"
        if not p.exists():
            missing_text += 1
            continue
        raw = p.read_text(encoding="utf-8", errors="ignore")
        texts.append(clean_text(raw))

    char_lens = [len(t) for t in texts]
    ws_token_lens = [len(t.split()) for t in texts]
    transformer_token_lens = _token_lengths_with_transformers(texts, tokenizer_name)

    resolved_max_len = int(default_max_len)
    resolved_token_len_source = "arg"
    if transformer_token_lens:
        q = quantiles(transformer_token_lens)
        p95 = int(round(q.p95))
        resolved_max_len = min(max(32, p95), 256)
        resolved_token_len_source = "transformers_p95_capped_256"
    elif ws_token_lens:
        q = quantiles(ws_token_lens)
        p95 = int(round(q.p95))
        resolved_max_len = min(max(32, p95 * 2), 256)
        resolved_token_len_source = "whitespace_p95_x2_capped_256"

    text_cfg = TextPreprocessConfig(tokenizer_name=tokenizer_name, max_len=resolved_max_len)
    img_cfg = default_image_preprocess_config(crop_size=image_crop_size)

    image_ok = 0
    image_fail = 0
    for guid in guids[: max(0, int(image_sample))]:
        image_path = data_files_dir / f"{guid}.jpg"
        if not image_path.exists():
            image_fail += 1
            continue
        try:
            with Image.open(image_path) as im:
                im.convert("RGB")
            image_ok += 1
        except Exception:
            image_fail += 1

    preprocess_dir = output_dir / "preprocess"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    text_cfg_path = preprocess_dir / "text_config.json"
    img_cfg_path = preprocess_dir / "image_config.json"
    report_path = preprocess_dir / "preprocess_report.json"

    text_cfg_path.write_text(json.dumps(asdict(text_cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    img_cfg_path.write_text(json.dumps(asdict(img_cfg), ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "paths": {
            "data_dir": str(data_dir),
            "train_index": str(train_clean_index),
            "data_files_dir": str(data_files_dir),
            "output_dir": str(preprocess_dir),
        },
        "text": {
            "rows_in_index": len(guids),
            "loaded_texts": len(texts),
            "missing_text_files": missing_text,
            "char_len": {"min": min(char_lens) if char_lens else 0, "max": max(char_lens) if char_lens else 0, "quantiles": asdict(quantiles(char_lens))},
            "whitespace_token_len": {
                "min": min(ws_token_lens) if ws_token_lens else 0,
                "max": max(ws_token_lens) if ws_token_lens else 0,
                "quantiles": asdict(quantiles(ws_token_lens)),
            },
            "transformers_token_len": None
            if transformer_token_lens is None
            else {"min": min(transformer_token_lens), "max": max(transformer_token_lens), "quantiles": asdict(quantiles(transformer_token_lens))},
            "text_config": asdict(text_cfg),
            "max_len_source": resolved_token_len_source,
        },
        "image": {"image_config": asdict(img_cfg), "sampled": int(image_sample), "opened_ok": image_ok, "opened_fail_or_missing": image_fail},
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"text_config": text_cfg_path, "image_config": img_cfg_path, "report": report_path}


def _split_train_val(train_clean_index: Path, output_dir: Path, *, val_ratio: float, seed: int) -> dict:
    rows = _read_labeled_index(train_clean_index)
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0, 1)")

    try:
        from sklearn.model_selection import train_test_split  # type: ignore
    except Exception as e:
        raise RuntimeError("scikit-learn is required for splitting") from e

    X = [r.guid for r in rows]
    y = [r.tag.strip().lower() for r in rows]
    train_guids, val_guids, train_tags, val_tags = train_test_split(
        X, y, test_size=float(val_ratio), random_state=int(seed), stratify=y, shuffle=True
    )
    train_rows = [LabeledRow(guid=g, tag=t) for g, t in zip(train_guids, train_tags)]
    val_rows = [LabeledRow(guid=g, tag=t) for g, t in zip(val_guids, val_tags)]

    splits_dir = output_dir / "splits"
    train_out = splits_dir / "train_split.txt"
    val_out = splits_dir / "val_split.txt"
    _write_labeled_index(train_out, train_rows)
    _write_labeled_index(val_out, val_rows)

    def _cnt(rs: list[LabeledRow]) -> dict[str, int]:
        return dict(Counter([r.tag for r in rs]))

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "train_index": str(train_clean_index),
        "output_dir": str(splits_dir),
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "counts": {"all": {"rows": len(rows), "by_tag": _cnt(rows)}, "train": {"rows": len(train_rows), "by_tag": _cnt(train_rows)}, "val": {"rows": len(val_rows), "by_tag": _cnt(val_rows)}},
        "outputs": {"train_split": str(train_out), "val_split": str(val_out)},
    }
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"train_split": train_out, "val_split": val_out, "report": splits_dir / "split_report.json"}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("./datasets"))
    p.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--tokenizer", type=str, default="bert-base-uncased")
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--image-crop-size", type=int, default=224)
    p.add_argument("--image-sample", type=int, default=200)
    p.add_argument("--drop-invalid", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    train_file = data_dir / "train.txt"

    if not train_file.exists():
        raise FileNotFoundError(str(train_file))
    if not (data_dir / "data").exists():
        raise FileNotFoundError(str(data_dir / "data"))

    label_result = _clean_labels(train_file, output_dir, drop_invalid=bool(args.drop_invalid))
    config_result = _build_preprocess_configs(
        data_dir,
        label_result["clean_index"],
        output_dir,
        tokenizer_name=str(args.tokenizer),
        default_max_len=int(args.max_len),
        image_crop_size=int(args.image_crop_size),
        image_sample=int(args.image_sample),
    )
    split_result = _split_train_val(label_result["clean_index"], output_dir, val_ratio=float(args.val_ratio), seed=int(args.seed))

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "clean_labels": {k: str(v) for k, v in label_result.items() if k != "report"},
        "preprocess_configs": {k: str(v) for k, v in config_result.items()},
        "split": {k: str(v) for k, v in split_result.items()},
        "label_report": label_result["report"],
    }
    out = output_dir / "preprocess_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

