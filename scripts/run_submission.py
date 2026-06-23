#!/usr/bin/env python3
"""Batch inference on private_test images → Kaggle-ready CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.data_utils import find_private_root, load_private_catalog, private_images_dir  # noqa: E402
from solution import predict_from_image  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run team solution on private test images")
    parser.add_argument("--limit", type=int, default=0, help="Max images (0 = all on disk)")
    parser.add_argument(
        "-o",
        "--output",
        default="outputs/submission_private.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    root = find_private_root()
    if root is None:
        raise SystemExit("private_test not found under data/private_test/")

    catalog = load_private_catalog(root)
    img_dir = private_images_dir(root)
    available = sorted(p.stem for p in img_dir.glob("*.jpg"))
    if args.limit:
        available = available[: args.limit]

    rows = []
    for iid in available:
        path = img_dir / f"{iid}.jpg"
        pred = predict_from_image(Image.open(path).convert("RGB"))
        rows.append(
            {
                "image_id": iid,
                "ocr_text": pred["ocr_text"] or " ",
                "brand_name": pred["brand_name"] or " ",
                "product_name": pred["product_name"] or " ",
            }
        )
        print(f"  {iid}: ocr={len(pred['ocr_text'])} brand={pred['brand_name']!r}")

    done = {r["image_id"] for r in rows}
    for iid in catalog["image_id"]:
        if iid not in done:
            rows.append(
                {
                    "image_id": iid,
                    "ocr_text": " ",
                    "brand_name": " ",
                    "product_name": " ",
                }
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows).sort_values("image_id")
    out.to_csv(out_path, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    print(f"Wrote {out_path} ({len(out)} rows, {len(available)} OCR'd)")


if __name__ == "__main__":
    main()
