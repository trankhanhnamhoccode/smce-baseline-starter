from pathlib import Path
import urllib.request


import os
from pathlib import Path

MODEL_DIR = Path(os.environ.get("SMCE_MODEL_DIR", "/tmp/smce_models/ocr_router"))

MODEL_URLS = {
    "line_candidate_selector.pkl": "https://github.com/trankhanhnamhoccode/smce-baseline-starter/releases/download/ocr-router-v1/line_candidate_selector.pkl",
    "blank_classifier.pkl": "https://github.com/trankhanhnamhoccode/smce-baseline-starter/releases/download/ocr-router-v1/blank_classifier.pkl",
}


def ensure_router_models() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in MODEL_URLS.items():
        out = MODEL_DIR / filename

        if out.exists() and out.stat().st_size > 1024 * 100:
            print(f"[models] OK: {out} ({out.stat().st_size / 1024 / 1024:.2f} MB)")
            continue

        print(f"[models] Downloading {filename}...")
        urllib.request.urlretrieve(url, out)
        print(f"[models] Saved: {out} ({out.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    ensure_router_models()
