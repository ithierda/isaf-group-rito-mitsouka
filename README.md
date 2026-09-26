<p align="center"><img src="assets/logo/horizontal_blue.png" width="520" alt="Les Rito Mitsouka"></p>

# ISAF project: Les Rito Mitsouka

Group project for **Interpretability, Stability & Algorithmic Fairness** (X-HEC M2, Prof. C. Pérignon, 2026-27).

**Client:** *Les Rito Mitsouka*, a (fictional) dating app deciding which pairs of profiles to recommend to each other.
**Task:** predict `match` (both participants say yes) on the Speed Dating Experiment data, and compare
three models (white-box, machine learning, tabular foundation model) on **performance** (statistical &
economic), **interpretability**, **stability** and **fairness** (protected attribute: ethnicity).

**Team:** Célia Morais · Ithier d'Aramon · Alec Seugnet · Simone Capizzi · Julian Fix · Justin Martin

## Repository layout

```
├── app/            Streamlit app for the client (streamlit run app/streamlit_app.py)
├── assets/logo/    Client logo (PNG + SVG)
├── data/
│   ├── raw/        Original dataset (never modified)
│   └── processed/  Tables the notebooks produce, committed so nobody has to rebuild them
├── docs/           Codebook of the dataset
├── notebooks/      The analysis, numbered and self-contained
└── reports/
    ├── figures/    Figures for the slides
    └── tables/     The tables the notebooks export, as CSV
```

The tables in `data/processed/` are **in the repository**. Pull and they are there: you do not need to
run 01 and 02 before starting on 04, 05 or 06. Rerun a notebook only when you change it, and commit
the table it rewrites along with it.

## Setup

Python 3.11. If you do not have it, `uv` installs one without touching your system Python
(`curl -LsSf https://astral.sh/uv/install.sh | sh` if you do not have `uv` either):

```bash
uv python install 3.11
uv venv --python 3.11 .venv

# or, if python3.11 is already on your machine:
# python3.11 -m venv .venv

source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name isaf --display-name "Python (isaf)"
```

In VS Code: *Python: Select Interpreter* → `.venv`, and pick the **Python (isaf)** kernel in notebooks.

> **macOS:** XGBoost needs OpenMP and will not import without it. With Homebrew, `brew install libomp`.
> Without Homebrew, copy the copy Anaconda ships into the Python you just created the venv from:
> ```bash
> cp /opt/anaconda3/lib/libomp.dylib "$(python -c 'import sys; print(sys.base_prefix)')/lib/"
> ```
>
> **Reproducibility:** every `LogisticRegression` is seeded with `random_state=0`. liblinear shuffles
> internally and without the seed the white box moves between runs. XGBoost and TabPFN are unaffected.
>
> **If the repository sits in an iCloud-synced folder** (anything under `~/Documents` with "Optimise
> Mac Storage" on), iCloud will evict `.venv` and every `import pandas` then waits on a download,
> minutes per cell. Either keep the venv outside iCloud (`uv venv --python 3.11 ~/.venvs/isaf`) or
> right-click the folder and choose *Keep Downloaded*.

## Notebooks

Each notebook is self-contained: imports, cleaning and analysis are all inside it, there is no shared
module to import. They expect to run from the repository root, and the first cell moves there by
itself if you launched Jupyter from `notebooks/`.

| Notebook | What it does | Output |
|---|---|---|
| `01_eda.ipynb` | The raw data: how it is organised, which columns leak, what the variables mean and how they are coded. Builds one row per pair. | `data/processed/pairs.csv` (4,184 × 138) |
| `02_features.ipynb` | Cleaning, comparison features, readable names, encoding, and the fold assignment. | `data/processed/model_table.csv` (4,184 × 66) |
| `03_models.ipynb` | Logistic regression and XGBoost, predicting the two decisions and multiplying them. Saves the out-of-fold predictions. | `data/processed/oof_predictions.csv` |
| `03b_tabpfn_colab.ipynb` | TabPFN, on a Colab GPU. Writes the predictions that 03 merges in. | `data/processed/tabpfn_oof.csv` |
| `04_interpretability.ipynb` | Coefficients and marginal effects, impurity importance, PDP and ICE, SHAP and LIME, permutation importance for all three engines, XPER, the surrogate tree, and the economics. | figures and tables in `reports/` |
| `05_stability.ipynb` | Distance between 25 versions of each model, coefficient drift, leave-one-wave-out, and the cost of imposing stability. | figures and tables in `reports/` |
| `06_fairness.ipynb` | Statistical parity, conditional parity (Cochran, Mantel and Haenszel), amplification, mitigation, proxy recovery, equal opportunity, calibration, equivalence. | figures and tables in `reports/` |
| `07_recommendation.ipynb` | The scorecard, three engines by four dimensions, and what we recommend to the client. | `reports/tables/07_scorecard.csv` |

Notebooks 04 to 06 are independent of each other and can be written in parallel; 07 reads what they
export. They all start from `oof_predictions.csv`, where `logit`, `xgboost` and `tabpfn` are the
two-stage models and the `*_direct` columns are the baseline that predicts `match` in one step. They
also rebuild the pipelines from 03, because an explanation, a distance between models and a
mitigation variant all need a fitted model rather than a column of scores.

**They follow the course.** The methods, the tables and their order come from the lectures. Three
things are ours and are labelled as such in the text: Ŷ = 1 means "in the top decile" rather than a
0.5 threshold, because no pair ever scores above 0.5 at a 16.5% base rate; the shortlist overlap
between refits in 05; and the amplification ratio in 06.

## Method

- **One row per pair.** Each date appears twice in the raw file, once from each side.
- **Pre-date information only.** 108 of the 195 columns are filled in during or after the date.
- **Split on `wave`, `StratifiedGroupKFold(5)`.** Everyone meets everyone of the opposite sex in
  their evening, so the "dated each other" graph has 21 connected components for 21 waves and there
  is no finer cut that keeps a person on one side. A random split reads 0.72 ROC-AUC against 0.60,
  and the gap is the model recognising people. The wave is a **grouping key, never a feature**: 02
  removes it along with everything constant within a wave.
- **A match is two decisions.** Every engine predicts `her_dec` and `his_dec` separately and
  multiplies them: 8,368 decision rows instead of 4,184 pairs, a target at 37% and 47% instead of
  16.5%, and a top decile at 29.8% against 23.6% predicting `match` directly. 03 keeps both so the
  choice can be defended.
- **Ethnicity stays in the table** on both sides, because 06 needs to compare a model fitted with it
  against one fitted without.
- **Readable column names**, set in 02: `her_` and `his_` for a person, a plain word for the pair,
  one underscore at most.

## The app

```bash
streamlit run app/streamlit_app.py
```

Written for the client, not for us: matches, money and fairness, with the technical metrics kept
but explained in a line each. A headline row of four figures, then four tabs (*Compare the
engines · How it decides · What it is worth · Is it fair?*) and two controls at the top that drive
everything below: which engines to compare, and how much of the catalogue the app would show.

It refits nothing. Performance, economics and fairness are recomputed live from
`oof_predictions.csv` at whatever shortlist length you pick; interpretability and stability are
read from `reports/`. TabPFN appears on its own once `tabpfn_oof.csv` is in `data/processed/`.

## Data

See [`data/README.md`](data/README.md). Source: Fisman, Iyengar, Kamenica & Simonson (2006),
*Gender Differences in Mate Selection: Evidence from a Speed Dating Experiment*, QJE.
[Kaggle](https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment).

## Git workflow

- Pull before you start: `git pull origin main`.
- **One notebook, one person at a time.** Notebooks are JSON and every run rewrites the outputs, so
  two people editing the same file give a conflict that is miserable to resolve. Say which number you
  are taking. Notebooks 03 to 06 are independent by design.
- Small, safe changes go straight to `main`; anything that might break something gets a branch.
- Do commit `data/processed/` when you change the notebook that writes it: that is how everyone else
  gets the tables. Never commit `.venv/`.
