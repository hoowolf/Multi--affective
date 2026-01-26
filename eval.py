from __future__ import annotations

import argparse
import json
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("./datasets"))
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    payload = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "split": args.split,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

