from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


@dataclass(frozen=True)
class IndexRow:
    guid: str
    tag: str | None


def _read_index_file(path: Path) -> list[IndexRow]:
    rows: list[IndexRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for i, parts in enumerate(reader):
            if not parts:
                continue
            if i == 0 and parts[0].strip().lower() == "guid":
                continue
            guid = parts[0].strip()
            tag = parts[1].strip() if len(parts) > 1 else ""
            rows.append(IndexRow(guid=guid, tag=tag if tag else None))
    return rows


def _collect_data_files(data_dir: Path) -> dict[str, dict[str, list[str]]]:
    per_guid: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"text": [], "image": [], "other": []})
    for p in data_dir.iterdir():
        if not p.is_file():
            continue
        guid = p.stem
        suffix = p.suffix.lower()
        if suffix == ".txt":
            per_guid[guid]["text"].append(p.name)
        elif suffix in IMAGE_EXTS:
            per_guid[guid]["image"].append(p.name)
        else:
            per_guid[guid]["other"].append(p.name)
    return per_guid


def _summarize_split(name: str, index_rows: list[IndexRow], per_guid_files: dict[str, dict[str, list[str]]]) -> dict:
    missing_text: list[str] = []
    missing_image: list[str] = []
    missing_both: list[str] = []
    duplicate_text: dict[str, list[str]] = {}
    duplicate_image: dict[str, list[str]] = {}

    for row in index_rows:
        files = per_guid_files.get(row.guid)
        texts = files["text"] if files else []
        images = files["image"] if files else []
        if len(texts) == 0 and len(images) == 0:
            missing_both.append(row.guid)
            continue
        if len(texts) == 0:
            missing_text.append(row.guid)
        if len(images) == 0:
            missing_image.append(row.guid)
        if len(texts) > 1:
            duplicate_text[row.guid] = texts
        if len(images) > 1:
            duplicate_image[row.guid] = images

    return {
        "name": name,
        "rows": len(index_rows),
        "missing_text": {"count": len(missing_text), "guids": missing_text},
        "missing_image": {"count": len(missing_image), "guids": missing_image},
        "missing_both": {"count": len(missing_both), "guids": missing_both},
        "duplicate_text": {"count": len(duplicate_text), "items": duplicate_text},
        "duplicate_image": {"count": len(duplicate_image), "items": duplicate_image},
    }


def _summarize_tags(index_rows: list[IndexRow]) -> dict:
    raw = Counter()
    normalized = Counter()
    for r in index_rows:
        if r.tag is None:
            raw["<empty>"] += 1
            normalized["<empty>"] += 1
            continue
        raw[r.tag] += 1
        normalized[r.tag.strip().lower()] += 1
    return {"raw": dict(raw), "normalized_lower": dict(normalized)}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("./datasets"))
    p.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    p.add_argument("--save-json", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    data_dir = args.data_dir
    index_train = data_dir / "train.txt"
    index_test = data_dir / "test_without_label.txt"
    data_files_dir = data_dir / "data"

    if not index_train.exists():
        raise FileNotFoundError(str(index_train))
    if not index_test.exists():
        raise FileNotFoundError(str(index_test))
    if not data_files_dir.exists():
        raise FileNotFoundError(str(data_files_dir))

    train_rows = _read_index_file(index_train)
    test_rows = _read_index_file(index_test)
    per_guid_files = _collect_data_files(data_files_dir)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "paths": {
            "data_dir": str(data_dir),
            "index_train": str(index_train),
            "index_test": str(index_test),
            "data_files_dir": str(data_files_dir),
        },
        "counts": {
            "data_dir_total_files": sum(1 for _ in data_files_dir.iterdir() if _.is_file()),
            "unique_guids_in_data_dir": len(per_guid_files),
            "train_index_rows": len(train_rows),
            "test_index_rows": len(test_rows),
        },
        "train": _summarize_split("train", train_rows, per_guid_files),
        "test": _summarize_split("test", test_rows, per_guid_files),
        "train_tags": _summarize_tags(train_rows),
        "test_tags": _summarize_tags(test_rows),
    }

    if args.save_json:
        out_dir = args.output_dir / "checks"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"data_check_{ts}.json"
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["saved_to"] = str(out_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

