"""
Team solution pipeline — replace this module (and siblings) with your own approach.

The Streamlit demo and submission script import:
    predict_from_image(img) -> {"ocr_text", "brand_name", "product_name"}
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

from shared.data_utils import load_train_labels
from solution.brand_rules import extract_brand_product, extract_product
from solution.product_model import ProductPredictor
from team_config import DEFAULT_MIN_CONF


# Image preprocessing + OCR (baseline: EasyOCR vi+en, CPU)



def preprocess(img: Image.Image, max_dim: int = 1280) -> Image.Image:
    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.35)
    return img.filter(ImageFilter.SHARPEN)


def postprocess_ocr(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    if not tokens:
        return ""
    deduped = [tokens[0]]
    for tok in tokens[1:]:
        if tok.lower() != deduped[-1].lower():
            deduped.append(tok)
    return " ".join(deduped)


@lru_cache(maxsize=1)
def get_ocr_reader():
    import easyocr

    return easyocr.Reader(["vi", "en"], gpu=False, verbose=False)


def run_ocr_on_image(img: Image.Image, reader, min_conf: float = DEFAULT_MIN_CONF) -> str:
    img = preprocess(img.convert("RGB"))
    try:
        results = reader.readtext(np.array(img), detail=1, paragraph=False)
        results = sorted(results, key=lambda r: (r[0][0][1], r[0][0][0]))
        lines = [r[1] for r in results if r[2] > min_conf]
        return postprocess_ocr(" ".join(lines))
    except Exception:
        return ""



# Brand + product extraction (baseline: regex rules + optional sklearn head)



@lru_cache(maxsize=1)
def _product_predict_fn() -> Callable[[str], str] | None:
    labels = load_train_labels()
    if labels is None:
        return None
    model = ProductPredictor(min_class_count=3, prob_threshold=0.60, max_features=3000)
    model.fit(labels, extract_product)
    return model.predict


def predict_private(
    ocr_text: str,
    product_fn: Callable[[str], str] | None = None,
) -> tuple[str, str]:
    brand, product = extract_brand_product(ocr_text or "")
    fn = product_fn or _product_predict_fn()
    if fn and not brand and not product:
        product = fn(ocr_text or "")
    return brand, product


def predict_from_text(ocr_text: str) -> tuple[str, str]:
    """Extract brand + product from raw OCR text (no image)."""
    return predict_private(ocr_text, _product_predict_fn())


def predict_from_image(
    img: Image.Image,
    min_conf: float = DEFAULT_MIN_CONF,
) -> dict[str, str]:
    """
    Main entry point for Streamlit + batch submission.

    Returns dict with keys: ocr_text, brand_name, product_name
    """
    ocr_text = run_ocr_on_image(img, get_ocr_reader(), min_conf)
    brand, product = predict_private(ocr_text, _product_predict_fn())
    return {
        "ocr_text": ocr_text,
        "brand_name": brand,
        "product_name": product,
    }
