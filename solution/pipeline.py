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


@lru_cache(maxsize=1)
def get_ocr_reader():
    try:
        import paddle
        paddle.set_device("cpu")
    except Exception:
        pass

    from paddleocr import PaddleOCR

    configs = [
        dict(use_angle_cls=True, lang="vi", use_gpu=False, show_log=False),
        dict(use_angle_cls=True, lang="vi", use_gpu=False),
        dict(lang="vi", use_gpu=False),
        dict(lang="vi"),
        dict(use_angle_cls=True, lang="en", use_gpu=False, show_log=False),
        dict(use_angle_cls=True, lang="en", use_gpu=False),
        dict(lang="en", use_gpu=False),
        dict(lang="en"),
        dict(),
    ]
    last_error = None
    for cfg in configs:
        try:
            return PaddleOCR(**cfg)
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Cannot initialize PaddleOCR CPU reader: {last_error}")


def _parse_paddle_result(result: Any, min_conf: float) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    def add_item(text: Any, score: Any = None, box: Any = None):
        text = _clean(text)
        if not text:
            return
        score_f = _safe_float(score, 1.0)
        if score_f < min_conf:
            return

        x0, y0 = 0.0, 0.0
        try:
            pts = np.asarray(box, dtype=float)
            if pts.ndim >= 2 and pts.shape[-1] >= 2:
                x0 = float(np.min(pts[..., 0]))
                y0 = float(np.min(pts[..., 1]))
        except Exception:
            pass

        items.append({"text": text, "score": score_f, "x": x0, "y": y0})

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


def ocr_variant(img: Image.Image, variant: str, min_conf: float) -> list[dict[str, Any]]:
    vimg = make_variant(img, variant)
    reader = get_ocr_reader()
    arr = np.array(vimg)

    try:
        result = reader.ocr(arr, cls=True)
    except TypeError:
        result = reader.ocr(arr)
    except Exception:
        # Some PaddleOCR builds are more stable with a file path.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
        try:
            vimg.save(tmp)
            try:
                result = reader.ocr(tmp, cls=True)
            except TypeError:
                result = reader.ocr(tmp)
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass

    return _parse_paddle_result(result, min_conf=min_conf)


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


def candidate_row(name: str, ctype: str, source_variant: str, text: str, lines: list[dict[str, Any]], raw_stats: dict[str, float]) -> dict[str, Any]:
    scores = [_safe_float(x.get("score"), 0.0) for x in lines if isinstance(x, dict)]
    avg_score = float(np.mean(scores)) if scores else 0.0
    median_score = float(np.median(scores)) if scores else 0.0
    n_lines = len([x for x in lines if _clean(x.get("text", "") if isinstance(x, dict) else x)])
    tl, wc = text_stats(text)
    raw_tl = float(raw_stats.get("text_len", 0.0))
    raw_wc = float(raw_stats.get("word_count", 0.0))
    raw_avg = float(raw_stats.get("avg_line_score", 0.0))
    return {
        "candidate_name": name,
        "candidate_type": ctype,
        "source_variant": source_variant,
        "ocr_text": text,
        "text": text,
        "final_ocr_text": text,
        "avg_line_score": avg_score,
        "median_line_score": median_score,
        "num_lines": n_lines,
        "text_len": tl,
        "word_count": wc,
        "line_score_x_num_lines": avg_score * n_lines,
        "raw_text_len": raw_tl,
        "raw_word_count": raw_wc,
        "raw_avg_line_score": raw_avg,
        "rel_text_len_vs_raw": tl / max(1.0, raw_tl),
        "rel_word_count_vs_raw": wc / max(1.0, raw_wc),
        "junk_ratio": junk_ratio(text),
    }


def run_multivariant_ocr(img: Image.Image, min_conf: float = DEFAULT_MIN_CONF) -> tuple[str, list[str], pd.DataFrame, pd.DataFrame]:
    variant_lines: dict[str, list[dict[str, Any]]] = {}
    variant_rows: list[dict[str, Any]] = []

    for variant in VARIANTS:
        try:
            lines = ocr_variant(img, variant, min_conf=min_conf)
        except Exception:
            lines = []
        variant_lines[variant] = lines
        text = join_lines(lines)
        scores = [_safe_float(x.get("score"), 0.0) for x in lines]
        avg_score = float(np.mean(scores)) if scores else 0.0
        median_score = float(np.median(scores)) if scores else 0.0
        tl, wc = text_stats(text)
        variant_rows.append({
            "source_variant": variant,
            "ocr_text": text,
            "text": text,
            "avg_line_score": avg_score,
            "median_line_score": median_score,
            "num_lines": len(lines),
            "text_len": tl,
            "word_count": wc,
            "junk_ratio": junk_ratio(text),
        })

    raw_row = next((r for r in variant_rows if r["source_variant"] == "raw"), None) or {}
    raw_stats = {
        "text_len": raw_row.get("text_len", 0),
        "word_count": raw_row.get("word_count", 0),
        "avg_line_score": raw_row.get("avg_line_score", 0),
    }

    candidates: list[dict[str, Any]] = []
    for r in variant_rows:
        v = r["source_variant"]
        text = r["ocr_text"]
        candidates.append(candidate_row(f"single::{v}", "single", v, text, variant_lines[v], raw_stats))

        high = [x for x in variant_lines[v] if _safe_float(x.get("score"), 0.0) >= 0.80]
        if high:
            candidates.append(candidate_row(f"score_ge_08::{v}", "line_policy", v, join_lines(high), high, raw_stats))

        # Top-area proxy after sorting by y: first half of detected lines.
        if len(variant_lines[v]) >= 2:
            top = variant_lines[v][: max(1, len(variant_lines[v]) // 2)]
            candidates.append(candidate_row(f"top_half::{v}", "line_policy", v, join_lines(top), top, raw_stats))

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
        text = join_lines(lines)
        name = "merge::" + "+".join(vs)
        candidates.append(candidate_row(name, "merged", "+".join(vs), text, lines, raw_stats))

    cand_df = pd.DataFrame(candidates).fillna("")
    var_df = pd.DataFrame(variant_rows).fillna("")

    selected_text = select_ocr_candidate(cand_df, var_df)
    evidence_lines = [selected_text] + [str(r["ocr_text"]) for r in variant_rows if _clean(r.get("ocr_text", ""))]
    return selected_text, evidence_lines, cand_df, var_df


@lru_cache(maxsize=1)
def load_router_model():
    p = ROOT / "models" / "ocr_router" / "line_candidate_selector.pkl"
    if not p.exists():
        return None
    obj = joblib.load(p)
    if isinstance(obj, dict):
        return obj.get("model") or obj.get("router") or obj
    return obj


@lru_cache(maxsize=1)
def load_blank_model():
    p = ROOT / "models" / "ocr_router" / "blank_classifier.pkl"
    if not p.exists():
        return None
    return joblib.load(p)


def select_ocr_candidate(cand_df: pd.DataFrame, var_df: pd.DataFrame) -> str:
    if cand_df.empty:
        return ""

    model = load_router_model()
    scored = cand_df.copy()

    if model is not None:
        try:
            scored["pred_btc_cer"] = model.predict(scored)
            scored = scored.sort_values(["pred_btc_cer", "junk_ratio", "text_len"], ascending=[True, True, False])
            selected = _clean(scored.iloc[0].get("ocr_text", ""))
        except Exception:
            # Fallback when sklearn/model version mismatch occurs.
            selected = ""
    else:
        selected = ""

    if not selected:
        fallback = cand_df.copy()
        fallback["rank_score"] = (
            fallback["avg_line_score"].astype(float).fillna(0) * 4.0
            + np.minimum(fallback["word_count"].astype(float).fillna(0), 40) / 10.0
            - fallback["junk_ratio"].astype(float).fillna(0) * 3.0
        )
        fallback = fallback.sort_values(["rank_score", "text_len"], ascending=[False, False])
        selected = _clean(fallback.iloc[0].get("ocr_text", ""))

    if USE_BLANK_GATE:
        try:
            if should_blank_with_model(cand_df, var_df):
                return ""
        except Exception:
            pass

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
    ocr_text, evidence_lines, cand_df, var_df = run_multivariant_ocr(img.convert("RGB"), min_conf=min_conf)
    ocr_ms = (time.perf_counter() - t_ocr) * 1000

    t_extract = time.perf_counter()
    brand_name, product_name, audit = extract_brand_product_from_ocr(ocr_text, evidence_lines)
    extract_ms = (time.perf_counter() - t_extract) * 1000

    result: dict[str, Any] = {
        "ocr_text": ocr_text,
        "brand_name": brand_name,
        "product_name": product_name,
    }
    if include_timing:
        result["timing_ms"] = {
            "ocr": round(ocr_ms, 1),
            "extract": round(extract_ms, 1),
            "total": round((time.perf_counter() - t0) * 1000, 1),
        }
        result["audit"] = {
            **audit,
            "ocr_variants": int(len(var_df)),
            "ocr_candidates": int(len(cand_df)),
            "router_model_loaded": load_router_model() is not None,
            "blank_model_loaded": load_blank_model() is not None,
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
