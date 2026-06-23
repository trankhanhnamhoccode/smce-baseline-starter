"""Dataset paths and loaders used by scripts and optional local scoring."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

PRIVATE_ROOT_CANDIDATES = [
    REPO_ROOT / "data" / "private_test",
    REPO_ROOT / "private_test",
    Path(os.environ.get("SMCE_PRIVATE_TEST_DIR", "")),
]


def _has_private_images(root: Path) -> bool:
    for sub in ("images", "images_sample"):
        d = root / sub
        if d.is_dir() and any(d.glob("priv_*.jpg")):
            return True
    return False


def find_private_root() -> Path | None:
    for root in PRIVATE_ROOT_CANDIDATES:
        if not root or not Path(root).exists():
            continue
        root = Path(root).resolve()
        test_csv = root / "private_test.csv"
        if test_csv.is_file() and _has_private_images(root):
            return root
        if _has_private_images(root):
            return root
    return None


def private_images_dir(root: Path) -> Path:
    """Prefer full ``images/``; fall back to bundled ``images_sample/``."""
    full = root / "images"
    if full.is_dir() and any(full.glob("priv_*.jpg")):
        return full
    sample = root / "images_sample"
    if sample.is_dir() and any(sample.glob("priv_*.jpg")):
        return sample
    return full


def load_private_catalog(root: Path) -> pd.DataFrame:
    test_csv = root / "private_test.csv"
    if test_csv.is_file():
        return pd.read_csv(test_csv, keep_default_na=False)
    img_dir = private_images_dir(root)
    ids = sorted(p.stem for p in img_dir.glob("*.jpg"))
    return pd.DataFrame({"image_id": ids})


def load_private_solution(root: Path) -> pd.DataFrame | None:
    for name in ("solution_private.csv", "solution_private_eval.csv"):
        path = root / name
        if path.is_file():
            return pd.read_csv(path, keep_default_na=False)
    return None


def load_train_labels() -> pd.DataFrame | None:
    for path in [
        REPO_ROOT / "data" / "train_labels.csv",
        REPO_ROOT / "train_labels.csv",
    ]:
        if path.is_file():
            return pd.read_csv(path, keep_default_na=False)
    return None


def setup_full_private_images(source_dir: Path, dest_dir: Path | None = None) -> int:
    """Copy ``source_dir/images/*.jpg`` into ``data/private_test/images/``."""
    dest = dest_dir or (REPO_ROOT / "data" / "private_test" / "images")
    src = source_dir / "images"
    if not src.is_dir():
        raise FileNotFoundError(f"Missing {src}")
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for jpg in src.glob("*.jpg"):
        shutil.copy2(jpg, dest / jpg.name)
        n += 1
    return n
