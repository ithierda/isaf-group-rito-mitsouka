<p align="center"><img src="assets/logo/horizontal_blue.png" width="520" alt="Les Rito Mitsouka"></p>

# ISAF project — Les Rito Mitsouka

Group project for **Interpretability, Stability & Algorithmic Fairness** (X-HEC M2, Prof. C. Pérignon, 2026-27).

**Client:** *Les Rito Mitsouka*, a (fictional) dating app deciding which pairs of profiles to recommend to each other.
**Task:** predict `match` (both participants say yes) on the Speed Dating Experiment data, and compare
three models — white-box, machine learning, tabular foundation model — on **performance** (statistical &
economic), **interpretability**, **stability** and **fairness** (protected attribute: ethnicity).

**Team:** Célia Morais · Ithier d'Aramon · Alec Seugnet · Simone Capizzi · Julian Fix · Justin Martin

## Repository layout

```
├── app/            Streamlit app for the client
├── assets/logo/    Client logo (PNG + SVG)
├── data/
│   ├── raw/        Original dataset (never modified)
│   └── processed/  Generated files (git-ignored)
├── docs/           Codebook of the dataset
├── models/         Saved models (git-ignored)
├── notebooks/      Analysis notebooks, numbered (01_eda, 02_features, …)
├── reports/figures Figures for the slides
└── src/            Reusable Python code (data loading, features, metrics)
```

## Setup

Requires Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name isaf --display-name "Python (isaf)"
```

In VS Code: *Python: Select Interpreter* → `.venv`, and pick the **Python (isaf)** kernel in notebooks.

> On macOS, XGBoost needs OpenMP: `brew install libomp`.

## Data

See [`data/README.md`](data/README.md). Source: Fisman, Iyengar, Kamenica & Simonson (2006),
*Gender Differences in Mate Selection: Evidence from a Speed Dating Experiment*, QJE —
[Kaggle](https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment).

## Git workflow

- `main` stays runnable; work on a branch per task (`eda`, `features`, `model-xgb`, …) and open a pull request.
- Pull before you start: `git pull origin main`.
- Clear notebook outputs before committing large plots, and never commit `.venv/` or `data/processed/`.
