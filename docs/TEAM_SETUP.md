# Team setup guide — fork & customize

Follow this checklist after forking the template repository.

## 1. Fork & clone

```bash
git clone https://github.com/YOUR_ORG/ura-hackathon-team-abc.git
cd ura-hackathon-team-abc
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Customize team branding (≈5 min)

Edit **[`team_config.py`](../team_config.py)** at repo root:

| Field | Example |
|-------|---------|
| `TEAM_NAME` | `"Team Phoenix"` |
| `TEAM_MEMBERS` | `"Alice, Bob, Carol"` |
| `GITHUB_REPO` | Your public fork URL |
| `OTHER_RESOURCE` | Slide deck, report, Kaggle notebook, … |
| `SUBTITLE` | One-line project description |
| `LOGO` / `FAVICON` | Paths under [`assets/`](../assets/) |

Optional: replace PNG files in [`assets/`](../assets/).

Run the demo:

```bash
streamlit run streamlit_app.py
```

## 3. Replace the solution (main work)

All inference code lives in **[`solution/`](../solution/)**.

### Minimum change

Keep [`solution/pipeline.py`](../solution/pipeline.py) but improve:

- [`brand_rules.py`](../solution/brand_rules.py) — add regex / brand dictionary entries
- [`product_model.py`](../solution/product_model.py) — train a better product classifier

### Full replacement

Swap OCR, NER, LLM, or end-to-end model — **keep this function signature**:

```python
# solution/pipeline.py
def predict_from_image(img: PIL.Image.Image, min_conf: float = 0.35) -> dict[str, str]:
    return {
        "ocr_text": "...",
        "brand_name": "...",
        "product_name": "...",
    }
```

[`streamlit_app.py`](../streamlit_app.py) and [`scripts/run_submission.py`](../scripts/run_submission.py) import only this API.

### Notebook

Replace [`solution/baseline_notebook.ipynb`](../solution/baseline_notebook.ipynb) with your experiments. The notebook is **not** wired into Streamlit automatically — use it for R&D, then port logic into `solution/pipeline.py`.

## 4. Install full private images (recommended)

Repo ships **6 sample JPEGs** in `data/private_test/images_sample/`.

For all **1,202** images (from BTC bundle or hackathon monorepo):

```bash
python scripts/setup_private_images.py /path/to/private_test
# e.g.
python scripts/setup_private_images.py ../private_test
```

## 5. Batch inference & submit

```bash
python scripts/run_submission.py --limit 6     # quick smoke test
python scripts/run_submission.py               # full run → outputs/submission_private.csv
```

Upload `outputs/submission_private.csv` to Kaggle.

Details: [SUBMISSION.md](./SUBMISSION.md)

## 6. About tab (presentation)

Edit `_render_about_tab()` in [`streamlit_app.py`](../streamlit_app.py) with your pipeline description, results, and links for judges / reviewers.

## Repository map — what to touch

| Path | Fork action |
|------|-------------|
| [`team_config.py`](../team_config.py) | **Edit** — team name, links, theme |
| [`solution/`](../solution/) | **Replace** — your ML pipeline |
| [`assets/`](../assets/) | **Optional** — logos |
| [`streamlit_app.py`](../streamlit_app.py) | **Light edit** — About tab content |
| [`shared/`](../shared/) | **Keep** — data path helpers |
| [`data/`](../data/) | **Keep** — IDs, sample images, train labels |
| [`private_test/metric.py`](../private_test/metric.py) | **Do not edit** — official metric |
| [`scripts/`](../scripts/) | **Keep** — setup + batch runner |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `private_test not found` | Ensure `data/private_test/private_test.csv` + `images_sample/` exist |
| EasyOCR slow first run | Downloads ~200 MB weights once |
| Streamlit upload shows wrong OCR | Confirm `predict_from_image` returns updated dict keys |
| Empty submission rows | Run `setup_private_images.py` or use `--limit` on sample IDs only |

## Deploy Streamlit (optional)

1. Push your fork to GitHub (public).
2. [share.streamlit.io](https://share.streamlit.io) → New app → `streamlit_app.py`.
3. Live test tab works with **Upload** only unless you bundle images separately.
