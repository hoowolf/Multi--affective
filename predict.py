from __future__ import annotations

import argparse
import json
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("./datasets"))
    parser.add_argument("--test-file", type=Path, default=Path("./datasets/test_without_label.txt"))
    parser.add_argument("--output-file", type=Path, default=Path("./outputs/test_predictions.txt"))
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    payload = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "test_file": str(args.test_file),
        "output_file": str(args.output_file),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

