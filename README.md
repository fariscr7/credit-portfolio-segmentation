# Credit Portfolio Segmentation and Risk Pattern Analysis

Unsupervised segmentation of Lending Club borrowers using K-Means, Hierarchical
Clustering and DBSCAN, with historical default-pattern analysis, anomaly
detection and an interactive Streamlit dashboard.

> **Status:** Stage 0 (project setup) in progress.
> Methodology, results and dashboard instructions are added as each stage is completed.

## Dataset

- File: `loans_granting_model_dataset.csv` (Lending Club loans, provided by the course).
- The CSV is **not stored in this repository** because it is larger than
  GitHub's 100 MiB file limit. Place it in `data/raw/` before running any code.
- Full variable descriptions are added in Stage 1.

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install the dependencies
pip install -r requirements.txt

# 3. Check that everything is ready
python check_environment.py
```

## Project structure

```
credit-portfolio-segmentation/
├── data/
│   ├── raw/            # original CSV (not committed)
│   └── processed/      # cleaned data created by the notebooks (not committed)
├── notebooks/          # one notebook per project stage
├── src/                # reusable Python modules (config, preprocessing, ...)
├── models/             # saved scaler / PCA / K-Means used by the dashboard
├── reports/figures/    # exported charts for the reports and presentation
├── check_environment.py
├── requirements.txt
└── README.md
```
