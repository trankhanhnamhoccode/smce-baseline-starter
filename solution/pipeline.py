from __future__ import annotations

import os
import re
import sys
import time
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

# Force CPU mode before Paddle/PaddleOCR import.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("FLAGS_use_cuda", "0")
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("SMCE_MODEL_DIR", "/tmp/smce_models/ocr_router"))
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ura_ocr.product.rulebase_clean_v3 import CleanRulebaseV3
from ura_ocr.product.open_phrase_layer_v2_conservative import predict_product_phrase_conservative
from ura_ocr.product.brand_expander_v1 import build_brand_alias_map, predict_brand_name
from ura_ocr.product.curated_rules_v1 import apply_curated_rules

try:
    from team_config import DEFAULT_MIN_CONF
except Exception:
    DEFAULT_MIN_CONF = 0.30

PRODUCT_THRESHOLD = 6.3
BRAND_THRESHOLD = 5.0
BRAND_MARGIN = 0.5
USE_BLANK_GATE = False  # Private route usually selected OCR candidate without applying blank gate.
ENABLE_MERGE_CANDIDATES = False  # Live demo: keep False to avoid duplicated text from merged variants.

VARIANTS = [
    "raw",
    "center_70_resize_960",
    "bottom_60_resize_960",
    "bottom_50_resize_960",
    "bottom_45_resize_960",
    "middle_bottom_70_resize_960",
    "center_60_resize_960",
    "upper_60_resize_960",
]

REQUIRED_COLS = ["image_id", "ocr_text", "brand_name", "product_name"]


VIET_CHARS = "a-zA-ZÀ-ỹĐđ"
JUNK_TOKEN_RE = re.compile(rf"^[{VIET_CHARS}0-9]+$")

_OCR_READER = None


def get_ocr_reader():
    """
    Stable Streamlit/Kaggle CPU PaddleOCR singleton.

    Important:
    - Only initialize PaddleOCR once per Python process.
    - Avoid PaddleX reinitialization error on Streamlit rerun/hot reload.
    - Use predict(np.array(img)), not predict(path).
    """
    global _OCR_READER

    if _OCR_READER is not None:
        return _OCR_READER

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("FLAGS_use_cuda", "0")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_use_onednn", "0")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    try:
        import paddle
        paddle.set_device("cpu")
    except Exception:
        pass

    from paddleocr import PaddleOCR

    _OCR_READER = PaddleOCR(
        lang="vi",
        device="cpu",
        ocr_version="PP-OCRv6",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
        engine="paddle",
    )

    return _OCR_READER

def is_junk_ocr_line(text: str, score: float | None = None) -> bool:
    s = str(text or "").strip()
    if not s:
        return True

    # Very low confidence lines are usually noise.
    if score is not None and score < 0.45:
        return True

    # Remove lines that are mostly punctuation/symbols.
    alnum = re.findall(rf"[{VIET_CHARS}0-9]", s)
    if len(alnum) < 2:
        return True

    tokens = re.findall(rf"[{VIET_CHARS}0-9]+", s)
    if not tokens:
        return True

    # Too many ultra-short uppercase fragments: "F FE BA D C HY OOO..."
    short_tokens = [t for t in tokens if len(t) <= 2]
    upper_tokens = [t for t in tokens if t.isupper() and len(t) <= 3]

    if len(tokens) >= 6:
        short_ratio = len(short_tokens) / max(1, len(tokens))
        upper_ratio = len(upper_tokens) / max(1, len(tokens))

        if short_ratio >= 0.45 and upper_ratio >= 0.40:
            return True

    # Long line with almost no Vietnamese-looking words is suspicious.
    meaningful = [
        t for t in tokens
        if len(t) >= 4 or re.search(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]", t)
    ]

    if len(tokens) >= 8 and len(meaningful) <= 2:
        return True

    return False

def clean_ocr_lines(lines):
    cleaned = []
    for item in lines:
        text = item.get("text", "")
        score = item.get("score", None)

        if is_junk_ocr_line(text, score):
            continue

        cleaned.append(item)

    return cleaned

def _clean(s: Any) -> str:
    s = "" if s is None else str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def resize_long_side(img: Image.Image, long_side: int = 960) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    m = max(w, h)
    if m <= 0:
        return img
    if m == long_side:
        return img
    ratio = long_side / m
    return img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.LANCZOS)


def crop_vertical(img: Image.Image, top_ratio: float, bottom_ratio: float) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    top = int(max(0.0, min(1.0, top_ratio)) * h)
    bottom = int(max(0.0, min(1.0, bottom_ratio)) * h)
    if bottom <= top:
        return img
    return img.crop((0, top, w, bottom))


def light_enhance(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(1.20)
    img = img.filter(ImageFilter.SHARPEN)
    return img


def make_variant(img: Image.Image, variant: str) -> Image.Image:
    img = img.convert("RGB")

    if variant == "raw":
        return light_enhance(resize_long_side(img, 1280))

    if variant == "center_70_resize_960":
        crop = crop_vertical(img, 0.15, 0.85)
    elif variant == "bottom_60_resize_960":
        crop = crop_vertical(img, 0.40, 1.00)
    elif variant == "bottom_50_resize_960":
        crop = crop_vertical(img, 0.50, 1.00)
    elif variant == "bottom_45_resize_960":
        crop = crop_vertical(img, 0.55, 1.00)
    elif variant == "middle_bottom_70_resize_960":
        crop = crop_vertical(img, 0.30, 1.00)
    elif variant == "center_60_resize_960":
        crop = crop_vertical(img, 0.20, 0.80)
    elif variant == "upper_60_resize_960":
        crop = crop_vertical(img, 0.00, 0.60)
    else:
        crop = img

    return light_enhance(resize_long_side(crop, 960))





def _parse_paddle_result(result: Any, min_conf: float) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    def add_item(text: Any, score: Any = None, box: Any = None):
        text = _clean(text)
        if not text:
            return

        score_f = _safe_float(score, 1.0)
        if score_f < min_conf:
            return
        if is_junk_ocr_line(text, score_f):
            return

        x0, y0 = 0.0, 0.0
        box_w, box_h, box_area = 0.0, 0.0, 0.0
        try:
            pts = np.asarray(box, dtype=float)
            if pts.ndim >= 2 and pts.shape[-1] >= 2:
                xs = pts[..., 0]
                ys = pts[..., 1]
                x0 = float(np.min(xs))
                y0 = float(np.min(ys))
                box_w = max(0.0, float(np.max(xs) - np.min(xs)))
                box_h = max(0.0, float(np.max(ys) - np.min(ys)))
                box_area = box_w * box_h
        except Exception:
            pass

        items.append(
            {
                "text": text,
                "score": score_f,
                "x": x0,
                "y": y0,
                "box_w": box_w,
                "box_h": box_h,
                "box_area": box_area,
                "area_ratio": 0.0,
            }
        )

    def visit(x: Any):
        if x is None:
            return

        if isinstance(x, dict):
            rec_texts = x.get("rec_texts") or x.get("texts")
            rec_scores = x.get("rec_scores") or x.get("scores")
            rec_boxes = x.get("rec_polys") or x.get("rec_boxes") or x.get("dt_polys") or x.get("boxes")

            if isinstance(rec_texts, (list, tuple)):
                for i, t in enumerate(rec_texts):
                    s = rec_scores[i] if isinstance(rec_scores, (list, tuple)) and i < len(rec_scores) else 1.0
                    b = rec_boxes[i] if isinstance(rec_boxes, (list, tuple)) and i < len(rec_boxes) else None
                    add_item(t, s, b)

            for v in x.values():
                if isinstance(v, (list, tuple, dict)):
                    visit(v)
            return

        if isinstance(x, (list, tuple)):
            # PaddleOCR 2.x leaf: [box, (text, score)]
            if len(x) >= 2 and isinstance(x[1], (list, tuple)) and len(x[1]) >= 1 and isinstance(x[1][0], str):
                box = x[0]
                text = x[1][0]
                score = x[1][1] if len(x[1]) > 1 else 1.0
                add_item(text, score, box)
                return

            # Possible leaf: (text, score)
            if len(x) >= 1 and isinstance(x[0], str):
                text = x[0]
                score = x[1] if len(x) > 1 else 1.0
                add_item(text, score, None)
                return

            for y in x:
                visit(y)
            return

    visit(result)
    items = sorted(items, key=lambda d: (d["y"], d["x"]))

    dedup = []
    seen = set()
    for it in items:
        key = it["text"].lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(it)
    return dedup


def join_lines(lines: list[dict[str, Any]] | list[str]) -> str:
    texts: list[str] = []
    seen = set()
    for x in lines:
        t = x.get("text", "") if isinstance(x, dict) else x
        t = _clean(t)
        if not t:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        texts.append(t)
    return _clean(" ".join(texts))


def text_stats(text: str) -> tuple[int, int]:
    text = _clean(text)
    return len(text), len(text.split())


def junk_ratio(text: str) -> float:
    text = _clean(text)
    if not text:
        return 1.0
    good = sum(1 for ch in text if ch.isalnum() or ch.isspace() or ch in "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯàáâãèéêìíòóôõùúăđĩũơưẠ-ỹ+-/&%.,:()")
    return max(0.0, min(1.0, 1.0 - good / max(1, len(text))))


ACCENT_RE_ROUTER = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵ"
    r"èéẹẻẽêềếệểễ"
    r"ìíịỉĩ"
    r"òóọỏõôồốộổỗơờớợởỡ"
    r"ùúụủũưừứựửữ"
    r"ỳýỵỷỹđ"
    r"ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ"
    r"ÈÉẸẺẼÊỀẾỆỂỄ"
    r"ÌÍỊỈĨ"
    r"ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ"
    r"ÙÚỤỦŨƯỪỨỰỬỮ"
    r"ỲÝỴỶỸĐ]"
)


def _ratio(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def _line_area_ratio(x: dict[str, Any]) -> float:
    return _safe_float(x.get("area_ratio"), 0.0) if isinstance(x, dict) else 0.0


def text_feature_stats(text: str) -> dict[str, float]:
    s = _clean(text)
    char_count = len(s)
    alpha_count = sum(ch.isalpha() for ch in s)
    digit_count = sum(ch.isdigit() for ch in s)
    upper_count = sum(ch.isupper() for ch in s)
    accent_count = len(ACCENT_RE_ROUTER.findall(s))

    return {
        "is_blank": float(not bool(s)),
        "char_count": float(char_count),
        "digit_count": float(digit_count),
        "alpha_count": float(alpha_count),
        "upper_count": float(upper_count),
        "accent_count": float(accent_count),
        "digit_ratio": _ratio(digit_count, char_count),
        "alpha_ratio": _ratio(alpha_count, char_count),
        "upper_ratio": _ratio(upper_count, alpha_count),
        "accent_ratio": _ratio(accent_count, alpha_count),
        "has_digit": float(digit_count > 0),
        "has_accent": float(accent_count > 0),
        "has_dot": float("." in s),
        "has_plus": float("+" in s),
    }


def candidate_row(
    name: str,
    ctype: str,
    source_variant: str,
    text: str,
    lines: list[dict[str, Any]],
    raw_stats: dict[str, float],
) -> dict[str, Any]:
    scores = [_safe_float(x.get("score"), 0.0) for x in lines if isinstance(x, dict)]
    areas = [_line_area_ratio(x) for x in lines if isinstance(x, dict) and _line_area_ratio(x) > 0]

    avg_score = float(np.mean(scores)) if scores else 0.0
    median_score = float(np.median(scores)) if scores else 0.0
    avg_area_ratio = float(np.mean(areas)) if areas else 0.0

    n_lines = len([x for x in lines if _clean(x.get("text", "") if isinstance(x, dict) else x)])
    tl, wc = text_stats(text)
    jr = junk_ratio(text)
    feat = text_feature_stats(text)

    raw_tl = float(raw_stats.get("text_len", 0.0))
    raw_wc = float(raw_stats.get("word_count", 0.0))
    raw_nl = float(raw_stats.get("num_lines", 0.0))
    raw_avg = float(raw_stats.get("avg_line_score", 0.0))
    raw_jr = float(raw_stats.get("junk_ratio", 0.0))
    raw_digit_ratio = float(raw_stats.get("digit_ratio", 0.0))
    raw_alpha_ratio = float(raw_stats.get("alpha_ratio", 0.0))

    row = {
        "candidate_name": name,
        "candidate_type": ctype,
        "source_variant": source_variant,
        "ocr_text": text,
        "text": text,
        "final_ocr_text": text,

        "avg_line_score": avg_score,
        "median_line_score": median_score,
        "num_lines": float(n_lines),
        "avg_area_ratio": avg_area_ratio,

        "text_len": float(tl),
        "word_count": float(wc),
        "junk_ratio": jr,

        "line_score_x_num_lines": avg_score * n_lines,
        "area_x_num_lines": avg_area_ratio * n_lines,

        "raw_text_len": raw_tl,
        "raw_word_count": raw_wc,
        "raw_num_lines": raw_nl,
        "raw_avg_line_score": raw_avg,
        "raw_junk_ratio": raw_jr,
        "raw_digit_ratio": raw_digit_ratio,
        "raw_alpha_ratio": raw_alpha_ratio,

        "rel_text_len_vs_raw": tl / max(1.0, raw_tl),
        "rel_word_count_vs_raw": wc / max(1.0, raw_wc),
        "rel_num_lines_vs_raw": n_lines / max(1.0, raw_nl),

        "delta_score_vs_raw": avg_score - raw_avg,
        "delta_junk_vs_raw": jr - raw_jr,
    }
    row.update(feat)
    return row


def _has_accent_text(text: str) -> bool:
    return bool(ACCENT_RE_ROUTER.search(_clean(text)))


def _candidate_from_policy(
    candidates: list[dict[str, Any]],
    variant: str,
    policy: str,
    lines: list[dict[str, Any]],
    raw_stats: dict[str, float],
    *,
    ctype: str = "single_variant_line_policy",
    source_variant: str | None = None,
) -> None:
    if not lines:
        return
    text = join_lines(lines)
    if not text:
        return
    source = source_variant or variant
    candidates.append(
        candidate_row(
            f"line::{variant}::{policy}",
            ctype,
            source,
            text,
            lines,
            raw_stats,
        )
    )


def _generate_line_policy_candidates(
    candidates: list[dict[str, Any]],
    variant: str,
    lines: list[dict[str, Any]],
    raw_stats: dict[str, float],
) -> None:
    # Candidate names/types are aligned with the final notebook line-candidate router.
    _candidate_from_policy(candidates, variant, "all_clean", lines, raw_stats)

    score_ge_07 = [x for x in lines if _safe_float(x.get("score"), 0.0) >= 0.70]
    _candidate_from_policy(candidates, variant, "score_ge_07", score_ge_07, raw_stats)

    score_ge_085 = [x for x in lines if _safe_float(x.get("score"), 0.0) >= 0.85]
    _candidate_from_policy(candidates, variant, "score_ge_085", score_ge_085, raw_stats)

    with_accent = [x for x in lines if _has_accent_text(str(x.get("text", "")))]
    _candidate_from_policy(candidates, variant, "with_accent", with_accent, raw_stats)

    long_lines = [x for x in lines if len(_clean(x.get("text", ""))) >= 8]
    _candidate_from_policy(candidates, variant, "long_lines", long_lines, raw_stats)

    if len(lines) >= 2:
        by_area = sorted(lines, key=lambda x: _line_area_ratio(x), reverse=True)
        _candidate_from_policy(candidates, variant, "top5_area", by_area[:5], raw_stats)

        area_vals = [_line_area_ratio(x) for x in lines]
        if any(v > 0 for v in area_vals):
            q50 = float(np.quantile(area_vals, 0.50))
            q65 = float(np.quantile(area_vals, 0.65))
            _candidate_from_policy(
                candidates,
                variant,
                "area_ge_q50",
                [x for x in lines if _line_area_ratio(x) >= q50],
                raw_stats,
            )
            _candidate_from_policy(
                candidates,
                variant,
                "area_ge_q65",
                [x for x in lines if _line_area_ratio(x) >= q65],
                raw_stats,
            )


def run_multivariant_ocr(img: Image.Image, min_conf: float = DEFAULT_MIN_CONF) -> tuple[str, list[str], pd.DataFrame, pd.DataFrame]:
    variant_lines: dict[str, list[dict[str, Any]]] = {}
    variant_rows: list[dict[str, Any]] = []

    for variant in VARIANTS:
        try:
            lines = ocr_variant(img, variant, min_conf=min_conf)
        except Exception as e:
            print(f"[ocr] variant failed: {variant}: {e}", flush=True)
            lines = []

        variant_lines[variant] = lines
        text = join_lines(lines)
        scores = [_safe_float(x.get("score"), 0.0) for x in lines]
        areas = [_line_area_ratio(x) for x in lines if _line_area_ratio(x) > 0]

        avg_score = float(np.mean(scores)) if scores else 0.0
        median_score = float(np.median(scores)) if scores else 0.0
        avg_area_ratio = float(np.mean(areas)) if areas else 0.0
        tl, wc = text_stats(text)

        variant_rows.append(
            {
                "source_variant": variant,
                "ocr_text": text,
                "text": text,
                "avg_line_score": avg_score,
                "median_line_score": median_score,
                "avg_area_ratio": avg_area_ratio,
                "num_lines": len(lines),
                "text_len": tl,
                "word_count": wc,
                "junk_ratio": junk_ratio(text),
                **text_feature_stats(text),
            }
        )

    raw_row = next((r for r in variant_rows if r["source_variant"] == "raw"), None) or {}
    raw_stats = {
        "text_len": raw_row.get("text_len", 0),
        "word_count": raw_row.get("word_count", 0),
        "num_lines": raw_row.get("num_lines", 0),
        "avg_line_score": raw_row.get("avg_line_score", 0),
        "junk_ratio": raw_row.get("junk_ratio", 0),
        "digit_ratio": raw_row.get("digit_ratio", 0),
        "alpha_ratio": raw_row.get("alpha_ratio", 0),
    }

    candidates: list[dict[str, Any]] = []

    for r in variant_rows:
        v = r["source_variant"]
        text = r["ocr_text"]
        lines = variant_lines[v]

        # Existing full-variant candidate from notebook final schema.
        candidates.append(candidate_row(f"variant::{v}", "existing_variant", v, text, lines, raw_stats))

        # Rich line-policy candidates from notebook final schema.
        _generate_line_policy_candidates(candidates, v, lines, raw_stats)

    # Merge candidates are useful offline but caused duplicated live-demo text in several Streamlit cases.
    # Keep disabled by default; set ENABLE_MERGE_CANDIDATES=True if you want to reproduce merge policies.
    if ENABLE_MERGE_CANDIDATES:
        merge_sets = [
            ["raw", "bottom_50_resize_960"],
            ["raw", "bottom_60_resize_960"],
            ["raw", "center_70_resize_960"],
            ["raw", "middle_bottom_70_resize_960"],
            ["center_70_resize_960", "bottom_50_resize_960"],
            ["upper_60_resize_960", "bottom_50_resize_960"],
        ]
        for vs in merge_sets:
            lines: list[dict[str, Any]] = []
            for v in vs:
                lines.extend(variant_lines.get(v, []))

            score_ge_085 = [x for x in lines if _safe_float(x.get("score"), 0.0) >= 0.85]
            if not score_ge_085:
                continue

            source = "+".join(vs)
            name_variant = source
            text = join_lines(score_ge_085)
            candidates.append(
                candidate_row(
                    f"merge::{source}::score_ge_085",
                    "merged_line_policy",
                    name_variant,
                    text,
                    score_ge_085,
                    raw_stats,
                )
            )

    cand_df = pd.DataFrame(candidates).fillna("")
    var_df = pd.DataFrame(variant_rows).fillna("")

    print(
        "[ocr] candidates:",
        len(cand_df),
        "types:",
        cand_df["candidate_type"].value_counts().to_dict() if "candidate_type" in cand_df.columns else {},
        flush=True,
    )

    selected_text = select_ocr_candidate(cand_df, var_df)
    evidence_lines = [selected_text] + [str(r["ocr_text"]) for r in variant_rows if _clean(r.get("ocr_text", ""))]
    return selected_text, evidence_lines, cand_df, var_df


ROUTER_KNOWN_NUMERIC_COLS = [
    "num_lines",
    "avg_line_score",
    "median_line_score",
    "avg_area_ratio",
    "text_len",
    "word_count",
    "char_count",
    "digit_count",
    "alpha_count",
    "upper_count",
    "accent_count",
    "digit_ratio",
    "alpha_ratio",
    "upper_ratio",
    "accent_ratio",
    "junk_ratio",
    "is_blank",
    "has_digit",
    "has_accent",
    "has_dot",
    "has_plus",
    "line_score_x_num_lines",
    "area_x_num_lines",
    "raw_text_len",
    "raw_word_count",
    "raw_num_lines",
    "raw_avg_line_score",
    "raw_digit_ratio",
    "raw_alpha_ratio",
    "raw_junk_ratio",
    "rel_text_len_vs_raw",
    "rel_word_count_vs_raw",
    "rel_num_lines_vs_raw",
    "delta_score_vs_raw",
    "delta_junk_vs_raw",
]


def _bundle_get_predictor(obj: Any) -> Any:
    if hasattr(obj, "predict"):
        return obj

    if isinstance(obj, dict):
        for key in ["model", "router", "estimator", "selector", "regressor", "clf", "pipeline"]:
            val = obj.get(key)
            if hasattr(val, "predict"):
                print(f"[router] using bundle['{key}']", flush=True)
                return val

        for key, val in obj.items():
            if hasattr(val, "predict"):
                print(f"[router] using bundle['{key}']", flush=True)
                return val

    return None


def _bundle_feature_columns(bundle: Any, model: Any) -> tuple[list[str], list[str], list[str]]:
    cat_cols: list[str] = []
    num_cols: list[str] = []
    feature_cols: list[str] = []

    if isinstance(bundle, dict):
        raw_cat = bundle.get("cat_cols") or bundle.get("categorical_cols") or bundle.get("category_cols")
        raw_num = bundle.get("num_cols") or bundle.get("numeric_cols") or bundle.get("numeric_features")
        raw_feat = bundle.get("feature_cols") or bundle.get("features") or bundle.get("input_cols") or bundle.get("columns")

        if raw_cat is not None:
            cat_cols = [str(c) for c in list(raw_cat)]
        if raw_num is not None:
            num_cols = [str(c) for c in list(raw_num)]
        if raw_feat is not None:
            feature_cols = [str(c) for c in list(raw_feat)]

    if not cat_cols:
        cat_cols = ["candidate_name", "candidate_type", "source_variant"]

    if not num_cols:
        # The released final notebook router expects these numeric features.
        num_cols = ROUTER_KNOWN_NUMERIC_COLS.copy()

    if not feature_cols:
        if hasattr(model, "feature_names_in_"):
            try:
                feature_cols = [str(c) for c in list(model.feature_names_in_)]
            except Exception:
                feature_cols = []
        if not feature_cols:
            feature_cols = cat_cols + num_cols

    # Preserve order and deduplicate.
    def dedup(xs: list[str]) -> list[str]:
        out = []
        seen = set()
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return dedup(cat_cols), dedup(num_cols), dedup(feature_cols)


def _prepare_router_frame(cand_df: pd.DataFrame, bundle: Any, model: Any) -> pd.DataFrame:
    out = cand_df.copy()

    cat_cols, num_cols, feature_cols = _bundle_feature_columns(bundle, model)

    # Categorical columns used by the final notebook ColumnTransformer.
    for col in cat_cols:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)

    # Ensure all known/required numeric columns exist and are numeric.
    required_numeric = list(dict.fromkeys(num_cols + ROUTER_KNOWN_NUMERIC_COLS))
    for col in required_numeric:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], 0).fillna(0.0)

    # If the bundle/model exposes feature columns, make sure every one exists.
    for col in feature_cols:
        if col not in out.columns:
            out[col] = "" if col in cat_cols else 0.0
        if col in cat_cols:
            out[col] = out[col].fillna("").astype(str)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], 0).fillna(0.0)

    return out


def select_ocr_candidate(cand_df: pd.DataFrame, var_df: pd.DataFrame) -> str:
    print("[router] select_ocr_candidate called", flush=True)

    if cand_df.empty:
        print("[router] cand_df empty", flush=True)
        return ""

    print(
        "[router] cand_df:",
        len(cand_df),
        "types:",
        cand_df["candidate_type"].value_counts().to_dict()
        if "candidate_type" in cand_df.columns
        else "no_candidate_type",
        flush=True,
    )

    bundle = load_router_bundle()
    model = _bundle_get_predictor(bundle)
    print("[router] model loaded:", model is not None, "type:", type(model), flush=True)

    selected = ""

    if model is not None:
        try:
            scored = _prepare_router_frame(cand_df, bundle, model)
            cat_cols, num_cols, feature_cols = _bundle_feature_columns(bundle, model)

            # Use only the columns the training bundle/model expects when available.
            X = scored[feature_cols] if feature_cols else scored
            scored["pred_btc_cer"] = model.predict(X)

            scored = scored.sort_values(
                ["pred_btc_cer", "junk_ratio", "text_len"],
                ascending=[True, True, False],
            )

            best = scored.iloc[0]
            print(
                "[router] selected:",
                best.get("candidate_name", ""),
                "| type:",
                best.get("candidate_type", ""),
                "| variant:",
                best.get("source_variant", ""),
                "| pred:",
                best.get("pred_btc_cer", ""),
                flush=True,
            )

            selected = _clean(best.get("ocr_text", ""))

        except Exception as e:
            print("[router] model failed:", repr(e), flush=True)
            selected = ""
    else:
        print("[router] no model, using fallback rank_score", flush=True)

    if not selected:
        fallback = cand_df.copy()
        fallback["rank_score"] = (
            fallback["avg_line_score"].astype(float).fillna(0) * 4.0
            + np.minimum(fallback["word_count"].astype(float).fillna(0), 40) / 10.0
            - fallback["junk_ratio"].astype(float).fillna(0) * 3.0
            - (fallback["candidate_type"].astype(str).eq("merged_line_policy").astype(float) * 2.0)
        )
        fallback = fallback.sort_values(["rank_score", "text_len"], ascending=[False, False])

        best = fallback.iloc[0]
        print(
            "[router] fallback selected:",
            best.get("candidate_name", ""),
            "| type:",
            best.get("candidate_type", ""),
            "| variant:",
            best.get("source_variant", ""),
            "| rank:",
            best.get("rank_score", ""),
            flush=True,
        )

        selected = _clean(best.get("ocr_text", ""))

    if USE_BLANK_GATE:
        try:
            if should_blank_with_model(cand_df, var_df):
                print("[router] blank gate applied", flush=True)
                return ""
        except Exception as e:
            print("[router] blank gate failed:", repr(e), flush=True)

    return selected


def should_blank_with_model(cand_df: pd.DataFrame, var_df: pd.DataFrame) -> bool:
    obj = load_blank_model()
    if not isinstance(obj, dict) or "model" not in obj:
        return False
    model = obj["model"]
    cols = obj.get("blank_feat_cols") or getattr(model, "feature_names_in_", [])
    threshold = float(obj.get("threshold", 0.5))

    row: dict[str, float] = {}
    row["cand_count"] = float(len(cand_df))
    row["nonblank_cand_count"] = float(cand_df["ocr_text"].astype(str).str.strip().ne("").sum()) if "ocr_text" in cand_df.columns else 0.0
    row["max_text_len"] = float(cand_df["text_len"].max()) if "text_len" in cand_df.columns and len(cand_df) else 0.0
    row["mean_text_len"] = float(cand_df["text_len"].mean()) if "text_len" in cand_df.columns and len(cand_df) else 0.0
    row["max_word_count"] = float(cand_df["word_count"].max()) if "word_count" in cand_df.columns and len(cand_df) else 0.0
    row["mean_word_count"] = float(cand_df["word_count"].mean()) if "word_count" in cand_df.columns and len(cand_df) else 0.0
    row["max_score"] = float(cand_df["avg_line_score"].max()) if "avg_line_score" in cand_df.columns and len(cand_df) else 0.0
    row["mean_score"] = float(cand_df["avg_line_score"].mean()) if "avg_line_score" in cand_df.columns and len(cand_df) else 0.0
    row["max_num_lines"] = float(cand_df["num_lines"].max()) if "num_lines" in cand_df.columns and len(cand_df) else 0.0
    row["mean_num_lines"] = float(cand_df["num_lines"].mean()) if "num_lines" in cand_df.columns and len(cand_df) else 0.0
    row["min_junk_ratio"] = float(cand_df["junk_ratio"].min()) if "junk_ratio" in cand_df.columns and len(cand_df) else 1.0
    row["mean_junk_ratio"] = float(cand_df["junk_ratio"].mean()) if "junk_ratio" in cand_df.columns and len(cand_df) else 1.0

    for _, r in var_df.iterrows():
        v = str(r.get("source_variant", ""))
        if not v:
            continue
        for col in ["avg_line_score", "junk_ratio", "num_lines", "text_len", "word_count"]:
            row[f"{col}__{v}"] = _safe_float(r.get(col), 0.0)

    X = pd.DataFrame([{c: row.get(c, 0.0) for c in cols}])
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0, 1])
        return prob >= threshold
    return bool(model.predict(X)[0])


@lru_cache(maxsize=1)
def load_train_labels_cached() -> pd.DataFrame:
    candidates = [
        ROOT / "data" / "train_labels.csv",
        ROOT / "clean_restart_bundle" / "required" / "train_labels.csv",
        Path("/kaggle/input/datasets/huoijo/clean-start/required/train_labels.csv"),
        Path("/kaggle/input/competitions/the-2nd-ura-hackathon/train_labels.csv"),
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p, dtype=str, keep_default_na=False).fillna("")
    raise FileNotFoundError("Missing train_labels.csv. Put it at data/train_labels.csv in the demo repo.")


@lru_cache(maxsize=1)
def get_rulebase() -> CleanRulebaseV3:
    return CleanRulebaseV3(load_train_labels_cached())


@lru_cache(maxsize=1)
def get_brand_alias_map():
    return build_brand_alias_map(load_train_labels_cached())


def extract_brand_product_from_ocr(ocr_text: str, evidence_lines: list[str] | None = None) -> tuple[str, str, dict[str, Any]]:
    ocr_text = _clean(ocr_text)
    evidence_lines = evidence_lines or []
    evidence_text = _clean(" || ".join([ocr_text] + [_clean(x) for x in evidence_lines if _clean(x)]))

    audit: dict[str, Any] = {"ocr_text": ocr_text, "evidence_text": evidence_text}
    brand_name = ""
    product_name = ""

    pred_v3 = get_rulebase().predict(evidence_text)
    v3_brand = _clean(getattr(pred_v3, "brand_name", ""))
    v3_product = _clean(getattr(pred_v3, "product_name", ""))
    audit["v3_brand"] = v3_brand
    audit["v3_product"] = v3_product
    audit["v3_brand_reason"] = getattr(pred_v3, "brand_reason", "")
    audit["v3_product_reason"] = getattr(pred_v3, "product_reason", "")
    if v3_brand:
        brand_name = v3_brand
    if v3_product:
        product_name = v3_product

    if not product_name and ocr_text:
        pred_p = predict_product_phrase_conservative(ocr_text=ocr_text, brand_name=brand_name, threshold=PRODUCT_THRESHOLD)
        p = _clean(pred_p.product_name)
        audit["v4a_product"] = p
        audit["v4a_product_score"] = float(pred_p.product_score)
        audit["v4a_product_reason"] = pred_p.product_reason
        if p:
            product_name = p
    else:
        audit["v4a_product_reason"] = "skip_existing_product"

    if not brand_name and ocr_text:
        pred_b = predict_brand_name(
            ocr_text=ocr_text,
            variant_texts=[evidence_text],
            alias_map=get_brand_alias_map(),
            threshold=BRAND_THRESHOLD,
            margin=BRAND_MARGIN,
        )
        b = _clean(pred_b.brand_name)
        audit["v5_brand"] = b
        audit["v5_brand_score"] = float(pred_b.brand_score)
        audit["v5_brand_reason"] = pred_b.brand_reason
        audit["v5_brand_alias"] = pred_b.matched_alias
        audit["v5_brand_hits"] = int(pred_b.variant_hits)
        if b:
            brand_name = b
    else:
        audit["v5_brand_reason"] = "skip_existing_brand"

    sub = pd.DataFrame([{"image_id": "live_image", "ocr_text": ocr_text, "brand_name": brand_name, "product_name": product_name}])
    evi = pd.DataFrame([{"image_id": "live_image", "ocr_text": ocr_text, "evidence_text": evidence_text}])
    try:
        sub2, audit_curated = apply_curated_rules(sub, evi)
        brand_name = _clean(sub2.iloc[0].get("brand_name", brand_name))
        product_name = _clean(sub2.iloc[0].get("product_name", product_name))
        if isinstance(audit_curated, pd.DataFrame) and len(audit_curated):
            for c in audit_curated.columns:
                audit[f"curated_{c}"] = _clean(audit_curated.iloc[0].get(c, ""))
    except Exception as e:
        audit["curated_error"] = repr(e)

    audit["final_brand_name"] = brand_name
    audit["final_product_name"] = product_name
    return brand_name, product_name, audit


def predict_from_text(ocr_text: str) -> tuple[str, str]:
    brand, product, _ = extract_brand_product_from_ocr(ocr_text, [ocr_text])
    return brand, product


def predict_from_image(img: Image.Image, min_conf: float = DEFAULT_MIN_CONF, *, include_timing: bool = True) -> dict[str, Any]:
    t0 = time.perf_counter()

    t_ocr = time.perf_counter()
    try:
        ocr_text, evidence_lines, cand_df, var_df = run_multivariant_ocr(
            img.convert("RGB"),
            min_conf=min_conf,
        )
    except Exception as e:
        print("[predict] OCR failed:", repr(e), flush=True)
        ocr_text, evidence_lines = "", []
        cand_df, var_df = pd.DataFrame(), pd.DataFrame()

    ocr_ms = (time.perf_counter() - t_ocr) * 1000

    t_extract = time.perf_counter()
    try:
        brand_name, product_name, audit = extract_brand_product_from_ocr(
            ocr_text,
            evidence_lines,
        )
    except Exception as e:
        print("[predict] extractor failed:", repr(e), flush=True)
        brand_name = ""
        product_name = ""
        audit = {
            "ocr_text": ocr_text,
            "extractor_error": repr(e),
        }

    extract_ms = (time.perf_counter() - t_extract) * 1000

    result: dict[str, Any] = {
        "ocr_text": ocr_text,
        "brand_name": brand_name,
        "product_name": product_name,
    }

    if include_timing:
        try:
            router_loaded = load_router_model() is not None
        except Exception as e:
            print("[predict] router audit failed:", repr(e), flush=True)
            router_loaded = False

        try:
            blank_loaded = load_blank_model() is not None
        except Exception as e:
            print("[predict] blank audit failed:", repr(e), flush=True)
            blank_loaded = False

        result["timing_ms"] = {
            "ocr": round(ocr_ms, 1),
            "extract": round(extract_ms, 1),
            "total": round((time.perf_counter() - t0) * 1000, 1),
        }
        result["audit"] = {
            **audit,
            "ocr_variants": int(len(var_df)),
            "ocr_candidates": int(len(cand_df)),
            "router_model_loaded": router_loaded,
            "blank_model_loaded": blank_loaded,
            "blank_gate_enabled": USE_BLANK_GATE,
        }

    return result


def get_model_profile() -> dict[str, Any]:
    router_path = ROOT / "models" / "ocr_router" / "line_candidate_selector.pkl"
    blank_path = ROOT / "models" / "ocr_router" / "blank_classifier.pkl"
    return {
        "name": "PaddleOCR CPU multi-variant + trained OCR router + deterministic product extractor",
        "device": "CPU only",
        "uses_gpu": False,
        "ocr": "PaddleOCR CPU over 8 preprocess variants",
        "router": f"line_candidate_selector.pkl present={router_path.exists()}, blank_classifier.pkl present={blank_path.exists()}",
        "extractor": "CleanRulebaseV3 + conservative product phrase + brand expander + curated evidence rules",
        "notes": "No SLM/LLM at inference. End-to-end from PIL image to OCR text, brand_name, product_name.",
    }

# ============================================================
# FINAL DEMO OVERRIDE: stable Kaggle CPU OCR, raw variant only
# ============================================================

from functools import lru_cache as _demo_lru_cache
import os as _demo_os
import numpy as _demo_np

# Live demo default: one OCR pass per image.
# Full 8-variant mode is too slow on CPU.
VARIANTS = [
    "raw",
    "center_70_resize_960",
    "bottom_60_resize_960",
    "bottom_50_resize_960",
    "bottom_45_resize_960",
    "middle_bottom_70_resize_960",
    "center_60_resize_960",
    "upper_60_resize_960",
]





def ocr_variant(img, variant_name, min_conf=DEFAULT_MIN_CONF):
    """
    Run one OCR variant using numpy input.

    This matches the working CPU notebook behavior. In Kaggle CPU,
    predict(path) may crash or become unstable, while predict(np.array(img))
    works with engine="paddle" and enable_mkldnn=False.
    """
    vimg = make_variant(img, variant_name)
    reader = get_ocr_reader()

    arr = _demo_np.array(vimg.convert("RGB"))
    result = reader.predict(arr)

    lines = _parse_paddle_result(result, min_conf=min_conf)

    # Normalize bbox area by variant image area so avg_area_ratio matches the final router schema.
    img_area = float(max(1, arr.shape[0] * arr.shape[1]))
    for line in lines:
        try:
            line["area_ratio"] = _safe_float(line.get("box_area"), 0.0) / img_area
        except Exception:
            line["area_ratio"] = 0.0

    return lines

# ============================================================
# END FINAL DEMO OVERRIDE
# ============================================================


# ============================================================
# FINAL OVERRIDE: auto-download OCR router models for Streamlit Cloud
# ============================================================

def _ensure_router_models_for_cloud():
    try:
        from scripts.download_models import ensure_router_models
        ensure_router_models()
    except Exception as e:
        print(f"[models] Warning: could not auto-download router models: {e}")


@lru_cache(maxsize=1)
def load_router_bundle():
    _ensure_router_models_for_cloud()

    try:
        import joblib

        p = MODEL_DIR / "line_candidate_selector.pkl"
        print(f"[router] loading bundle: {p} exists={p.exists()}", flush=True)

        if not p.exists():
            return None

        obj = joblib.load(p)
        if isinstance(obj, dict):
            print("[router] bundle keys:", list(obj.keys()), flush=True)
        else:
            print("[router] loaded object:", type(obj), flush=True)
        return obj

    except Exception as e:
        print(f"[router] Failed to load line_candidate_selector.pkl: {e}", flush=True)

    return None


@lru_cache(maxsize=1)
def load_router_model():
    bundle = load_router_bundle()
    return _bundle_get_predictor(bundle)


@lru_cache(maxsize=1)
def load_blank_model():
    _ensure_router_models_for_cloud()

    try:
        import joblib
        p = MODEL_DIR / "blank_classifier.pkl"
        print(f"[blank] loading: {p} exists={p.exists()}", flush=True)
        if p.exists():
            obj = joblib.load(p)
            if isinstance(obj, dict):
                print("[blank] bundle keys:", list(obj.keys()), flush=True)
            else:
                print("[blank] loaded object:", type(obj), flush=True)
            return obj
    except Exception as e:
        print(f"[router] Failed to load blank_classifier.pkl: {e}", flush=True)

    return None

# ============================================================
# END FINAL OVERRIDE
# ============================================================

