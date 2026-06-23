<div align="center">

# The 2nd URA Hackathon — Team Submission Template

**Fork-friendly starter: Streamlit demo + batch submission + baseline OCR pipeline**

<p>
  <img src="https://img.shields.io/badge/python-3.11+-blue?style=for-the-badge" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/Type-Team%20Template-7c3aed?style=for-the-badge" alt="Team Template"/>
  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="MIT"/>
</p>

<p>
  <img src="https://skillicons.dev/icons?i=python,github" alt="Python, GitHub"/>
</p>

</div>

## Overview

Template repository for **The 2nd URA Hackathon 2026** (SMCE private round). Teams **fork** this repo, customize branding, and **replace the code under `solution/`** with their own OCR + brand/product extraction pipeline.

Included out of the box:

- **Streamlit demo** — Live test (upload image) + About tab for team presentation
- **Baseline pipeline** — EasyOCR + regex brands + optional sklearn product head
- **Batch script** — `outputs/submission_private.csv` (Kaggle-ready)
- **Sample data** — 6 private-test JPEGs + 1,202 IDs; full images installed separately
- **Official metric** — [`private_test/metric.py`](private_test/metric.py) (read-only)

**Private score:** `0.4 × F1_brand + 0.35 × (1 − CER) + 0.25 × F1_product`

## Fork in 3 steps

1. **Fork** this repo → edit [`team_config.py`](team_config.py) (team name, members, links, logo).
2. **Replace** [`solution/`](solution/) with your pipeline — keep `predict_from_image()` working.
3. **Run** `streamlit run streamlit_app.py` and `python scripts/run_submission.py`.

Full checklist: **[docs/TEAM_SETUP.md](docs/TEAM_SETUP.md)**

## Repository layout

```text
ura-hackathon-template/
├── team_config.py           # ← Edit: branding & team info
├── streamlit_app.py         # Demo UI (Live test + About)
├── solution/                # ← Replace: your ML pipeline
│   ├── pipeline.py          #   predict_from_image() entry point
│   ├── brand_rules.py
│   ├── product_model.py
│   ├── baseline_notebook.ipynb
│   └── README.md
├── shared/                  # Data path helpers (keep)
│   └── data_utils.py
├── scripts/
│   ├── setup_private_images.py
│   └── run_submission.py
├── data/
│   ├── train_labels.csv
│   └── private_test/
├── assets/                  # Logos & favicon
├── outputs/                 # Generated submissions (gitignored)
├── private_test/metric.py   # Official scoring — do not edit
└── docs/
    ├── TEAM_SETUP.md
    └── SUBMISSION.md
```

## What teams customize vs keep

| Customize | Keep as-is |
|-----------|------------|
| `team_config.py` | `shared/`, `scripts/`, `data/` layout |
| `solution/*.py` + notebook | `private_test/metric.py` |
| `assets/` logos | `streamlit_app.py` structure (About text optional) |
| About tab content | Submission column format |

## Quickstart

### Prerequisites

- Python **3.11+**
- ~**1 GB RAM** (EasyOCR downloads ~200 MB weights on first run)

### Install

```bash
git clone https://github.com/YOUR_ORG/ura-hackathon-team-abc.git
cd ura-hackathon-team-abc
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure team info

Edit [`team_config.py`](team_config.py) — set `TEAM_NAME`, `TEAM_MEMBERS`, `GITHUB_REPO`, etc.

### Streamlit demo

```bash
streamlit run streamlit_app.py
```

### Batch submission

```bash
python scripts/setup_private_images.py ../private_test   # optional: full 1,202 images
python scripts/run_submission.py --limit 6                 # smoke test
python scripts/run_submission.py                           # full CSV
```

Output: [`outputs/submission_private.csv`](outputs/submission_private.csv)

### Notebook (experiments)

```bash
jupyter notebook solution/baseline_notebook.ipynb
```

Port winning experiments into `solution/pipeline.py`.

## Solution API (contract)

```python
from PIL import Image
from solution import predict_from_image

result = predict_from_image(Image.open("path/to.jpg"))
# {"ocr_text": "...", "brand_name": "...", "product_name": "..."}
```

See [`solution/README.md`](solution/README.md).

## Useful docs

- [Team setup guide](docs/TEAM_SETUP.md)
- [Submission format & local scoring](docs/SUBMISSION.md)
- [Private test data](data/private_test/README.md)

## Deploy Streamlit Cloud

1. Push your fork to GitHub (public).
2. [share.streamlit.io](https://share.streamlit.io) → New app → `streamlit_app.py`.

## Current status

| Component | Status |
|-----------|--------|
| Team template structure | Ready |
| Streamlit demo (Live test + About) | Ready |
| Baseline solution in `solution/` | Reference implementation |
| Sample private images (6) | Included |
| Full 1,202 images | Install via script |
| Ground truth solution | BTC only — not public |

## License

MIT — see [LICENSE](LICENSE).
