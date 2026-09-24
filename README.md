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
| `03b_tabpfn_colab.ipynb` | TabPFN. Runs on Colab with a GPU, see the header of the notebook. | `tabpfn_oof.csv`, to drop into `data/processed/` |
| `04_interpretability.ipynb` | Coefficients and marginal effects, impurity importance, PDP and ICE, SHAP, permutation importance, XPER, and the economic comparison against random matching. | figures and tables in `reports/` |
| `05_stability.ipynb` | Distance between 25 versions of each model, stability of the recommended list, leave-one-wave-out, and the cost of imposing stability. | figures and tables in `reports/` |
| `06_fairness.ipynb` | Statistical parity, conditional parity (Cochran, Mantel and Haenszel), amplification, mitigation, proxy recovery, equivalence testing. | figures and tables in `reports/` |

Notebooks 04 to 06 are independent of each other and can be written in parallel. They all read
`oof_predictions.csv`, where `logit` and `xgboost` are the two-stage models we keep and
`logit_direct` / `xgboost_direct` the baseline. They also rebuild the pipelines from 03, because an
explanation, a distance between models and a mitigation variant all need a fitted model rather than
a column of scores. The recipe is copied from 03 unchanged, and it takes a few seconds.

**Three conventions hold across 04, 05 and 06.** A decision is "in the top decile", not a 0.5
threshold, because no pair ever scores above 0.5 at a 16.5% base rate. Every number carries a
bootstrap interval resampled within waves, and a gap smaller than its interval is reported as
inconclusive. The two stages are combined in logs, never on the probability scale.

## Method, decided so far

- **One row per pair.** Each date appears twice in the raw file, once from each side. Keeping both
  duplicates every observation and puts the same date on both sides of a split.
- **Pre-date information only.** 108 of the 195 columns are filled in during or after the date and
  would leak the answer.
- **Split on `wave`, with `StratifiedGroupKFold(5)`.** Every participant attends exactly one of the 21
  evenings, so the wave is the only grouping that keeps a person on a single side. Measured on the
  same features: a random split scores 0.72 ROC-AUC against 0.60 for the grouped one, and the
  difference is the model recognising people it has already seen, not skill.
- **No feature that is constant within a wave.** Such a column is the wave under another name, and a
  tree will use it to memorise an evening's match rate.
- **Ethnicity stays in the table** on both sides. The fairness notebook needs it to compare a model
  fitted with it against one fitted without.
- **A match is two decisions, not one event.** Every model predicts `her_dec` and `his_dec`
  separately and multiplies them, rather than predicting `match` directly: 8,368 decision rows
  instead of 4,184 pairs, and a target at 37% / 47% instead of 16.5%. It costs a little ROC-AUC and
  gains where the app operates: 30% of the top decile match, against 23% predicting `match`
  directly. `03_models` keeps both so the choice can be defended.
- **Readable column names**, set in 02: `her_` / `his_` for a person, a plain word for a property of
  the pair, one underscore at most (`her_racepref`, `agegap`, `samefield`, `her_fit`).

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
