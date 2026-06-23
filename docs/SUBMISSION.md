# Submission format — The 2nd URA Hackathon (private round)

## CSV columns

| Column | Description |
|--------|-------------|
| `image_id` | `priv_h_*` or `priv_d_*` |
| `ocr_text` | Raw OCR output |
| `brand_name` | Brand entity |
| `product_name` | Product entity |

## Empty values

Use a **single space** `" "` in CSV for empty fields (Kaggle convention).

## Generate submission

```bash
python scripts/run_submission.py --limit 20          # smoke test
python scripts/run_submission.py                     # all images on disk
python scripts/run_submission.py -o outputs/my.csv   # custom path
```

Default output: [`outputs/submission_private.csv`](../outputs/submission_private.csv)

## Scoring formula

```
0.4 × F1_brand + 0.35 × (1 − CER) + 0.25 × F1_product
```

Implementation: [`private_test/metric.py`](../private_test/metric.py) (do not edit).

## Local evaluation (optional, BTC only)

1. Copy `solution_private.csv` to `data/private_test/` (gitignored).
2. Score:

```python
import pandas as pd
import sys
sys.path.insert(0, "private_test")
from metric import score

sol = pd.read_csv("data/private_test/solution_private.csv", keep_default_na=False)
sub = pd.read_csv("outputs/submission_private.csv", keep_default_na=False)
print(score(sol, sub, "image_id"))
```

Never commit or publish ground-truth files.
