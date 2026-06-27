"""Team-facing configuration — edit this file after forking the template."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


# Team identity

TEAM_NAME = "Team 1 - Ngũ Long Công Tửu"
TEAM_MEMBERS = (
    "Trần Khánh Nam, "
    "Đặng Võ Minh Nhựt, "
    "Trương Quang Khải, "
    "Nguyễn Thiện Nhân, "
    "Nguyễn Hoàng"
)

GITHUB_REPO = "https://github.com/trankhanhnamhoccode/smce-baseline-starter"
OTHER_RESOURCE = "https://www.kaggle.com/competitions/the-2nd-ura-hackathon/overview"
STREAMLIT_APP_URL = "https://team1ngulongcongtuu.streamlit.app/"


# Streamlit page copy

SUBTITLE = (
    "CPU-friendly OCR, Brand Extraction & Product Name Extraction "
    "from Vietnamese Social Media Images"
)

PAGE_TITLE = f"The 2nd URA Hackathon - {TEAM_NAME}"
BROWSER_TITLE = PAGE_TITLE


# Branding assets

ASSETS_DIR = REPO_ROOT / "assets"
FAVICON = ASSETS_DIR / "kaggle_144224_logos_thumb76_76.png"
LOGO = ASSETS_DIR / "bk_name_en.png"
LOGO_WIDTH = 280


# UI theme

THEME_PRIMARY = "#1565C0"
THEME_PRIMARY_DARK = "#0D47A1"
THEME_BG = "#FFFFFF"
THEME_TEXT = "#1A2B4A"
THEME_MUTED = "#5C6B8A"


# Default inference settings

DEFAULT_MIN_CONF = 0.35


# Model footprint / demo profile
# Keep this in sync with solution/pipeline.py.

MODEL_PROFILE: dict[str, str | float | None] = {
    "pipeline": (
        "PaddleOCR PP-OCRv6 CPU OCR "
        "+ Vietnamese text cleanup "
        "+ brand/product rules "
        "+ sklearn/router post-processing"
    ),
    "runtime_device": "CPU",
    "product_head_mb": None,
    "ocr_backend_note": (
        "PaddleOCR 3.x with PP-OCRv6, lang='vi', device='cpu', "
        "engine='paddle'. The live demo uses raw variant only for stable "
        "CPU latency."
    ),
    "lightweight_notes": (
        "Optimized for Streamlit/Kaggle CPU live demo. "
        "Full 8-variant OCR is disabled in the demo path because it is much "
        "slower on CPU; the current runtime uses VARIANTS = ['raw']."
    ),
}