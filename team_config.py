"""Team-facing configuration — edit this file after forking the template."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


# Team identity (required after fork)

TEAM_NAME = "Team ABC"
TEAM_MEMBERS = "[Member 1], [Member 2], [Member 3]"
GITHUB_REPO = "https://github.com/your-team/ura-hackathon-team-abc"
OTHER_RESOURCE = "https://example.com/other-resource"


# Streamlit page copy

SUBTITLE = (
    "OCR & Product Name Extraction from Social Media Images "
    "by HCMUT URA Research Group"
)
PAGE_TITLE = f"The 2nd URA Hackathon - {TEAM_NAME}"
BROWSER_TITLE = PAGE_TITLE


# Branding assets (replace files under assets/ if needed)

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


# Default inference settings (override inside solution/pipeline.py if needed)

DEFAULT_MIN_CONF = 0.35
