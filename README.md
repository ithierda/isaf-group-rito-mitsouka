<p align="center"><img src="assets/logo/horizontal_blue.png" width="520" alt="Les Rito Mitsouka"></p>

# ISAF project: Les Rito Mitsouka

**Interpretability, Stability & Algorithmic Fairness**, X-HEC M2, Prof. C. Pérignon, 2026-27.
Célia Morais · Ithier d'Aramon · Alec Seugnet · Simone Capizzi · Julian Fix · Justin Martin

A dating app has to choose which pairs of profiles to put in front of each other. We build the thing
that chooses, on the Speed Dating Experiment data, and then judge three engines on four dimensions:
**performance**, **interpretability**, **stability** and **fairness**. The protected attribute is
ethnicity.

Everything is in the eight notebooks. They run in order and each one opens with an **In plain words**
paragraph before any code.

## What we found

| Engine | PR-AUC | Top decile | Lift over chance | Instability | Amplification |
|---|---|---|---|---|---|
| Lasso logistic regression | 0.228 | 0.298 | 1.81x | 0.56 | 1.16 |
| XGBoost | 0.218 | 0.286 | 1.74x | 0.30 | 1.01 |
| TabPFN | 0.234 | 0.289 | 1.75x | n/a | 1.27 |

Out-of-fold on 4,184 dates, base match rate 16.5%. Every figure comes from
`reports/tables/07_scorecard.csv`, which notebook 07 writes.

- **The engine works.** Showing pairs at random gives 165 matches per 1,000. The top decile gives
  **298**, a lift of **1.81**.
- **Performance cannot choose between the three.** The interval on the top decile is eight points
  wide and the engines are one point apart.
- **Fairness can.** All three recommend Asian participants less often, but XGBoost barely amplifies
  the same-background preference (1.01) where TabPFN reaches 1.27.
- **Deleting the ethnicity column makes fairness worse**, not better: amplification rises from 1.16
  to 1.35, because the model rebuilds ethnicity from income missingness and lifestyle answers.

**Instability** is the typical distance between two refits of the same model, divided by the size of
the average model, so lower is steadier. **Amplification** above 1.00 means the engine shows more
same-background pairs than actually match.

## The notebooks

Each one is self-contained: imports, cleaning and analysis inside it, no shared module. They run from
the repository root, and the first cell moves there if you launched Jupyter from `notebooks/`.

| Notebook | What it does | Output |
|---|---|---|
| `01_eda` | How the raw data is organised, which columns leak, what the variables mean. Builds one row per pair. | `pairs.csv`, 4,184 × 140 |
| `02_features` | Cleaning, comparison features, readable names, fold assignment. | `model_table.csv`, 4,184 × 66 |
| `03_models` | The three engines on the same folds, two-stage and direct. | `oof_predictions.csv` |
| `03b_tabpfn_colab` | TabPFN, on a Colab GPU. Writes the predictions 03 merges in. | `tabpfn_oof.csv` |
| `04_interpretability` | Coefficients, impurity, PDP, ICE, SHAP, LIME, permutation importance for all three engines, XPER, and the economics. | figures and tables |
| `05_stability` | Distance between 25 refits, coefficient drift, leave one wave out, anchoring. | figures and tables |
| `06_fairness` | Parity, conditional parity, amplification, mitigation, equal opportunity, calibration, equivalence. | tables |
| `07_recommendation` | The scorecard and what we recommend. | `07_scorecard.csv` |

04 to 06 are independent and can be read in any order; 07 reads what they export. All of them start
from `oof_predictions.csv`, where `logit`, `xgboost` and `tabpfn` are the two-stage engines and the
`*_direct` columns are the one-step baseline.

**They follow the course.** The methods, the tables and their order come from the lectures. Three
things are ours and are labelled as such where they appear: Ŷ = 1 means "in the top decile" rather
than a 0.5 threshold, because no pair ever scores above 0.5 at a 16.5% base rate; the shortlist
overlap between refits in 05; and the amplification ratio in 06.

## The two decisions that shape everything

**We split on the wave, not at random.** Everyone meets everyone of the opposite sex at their own
evening, so the "these two dated" graph has 21 connected components for 21 waves, and there is no
finer cut that keeps a person on one side. A random split reads 0.72 ROC-AUC against 0.60, and the
gap is the model recognising people rather than learning. The wave is a **grouping key, never a
feature**: 02 removes it, along with everything constant within a wave.

**A match is two decisions, not one event.** Every engine predicts `her_dec` and `his_dec`
separately and multiplies them: 8,368 decision rows instead of 4,184 pairs, a target at 37% and 47%
instead of 16.5%, and a top decile at 29.8% against 23.6% predicting `match` directly. 03 keeps both
so the choice can be defended.

Three smaller ones. Only pre-date information, since 108 of the 195 raw columns are filled in during
or after the date. Ethnicity stays in the table on both sides, because 06 compares a model fitted
with it against one fitted without. And column names are readable: `her_` and `his_` for a person, a
plain word for the pair, one underscore at most.

## The data

`data/raw/speed_dating.csv`, never modified. Fisman, Iyengar, Kamenica & Simonson (2006), *Gender
Differences in Mate Selection: Evidence from a Speed Dating Experiment*, QJE. Collected at Columbia
Business School, 2002 to 2004; public copy of the Kaggle
[Speed Dating Experiment](https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment) file.

| | |
|---|---|
| Rows | 8,378, one row = one participant's view of one four-minute date |
| Columns | 195, of which 108 are filled in during or after the date |
| Participants | 551 (`iid`) across 21 sessions (`wave`) |
| Pairs | 4,184, all woman-man, 690 matched (16.5%) |
| Encoding | Mac Roman, so `pd.read_csv(..., encoding="mac_roman")` |

`docs/speed_dating_data_key.doc` is the original codebook and defines every column. Four things it is
worth knowing before reading any coefficient: each date appears **twice**, once from each side;
`date` and `go_out` are **reverse coded**, 1 is several times a week and 7 almost never; `field_cd`,
`career_c`, `goal` and `race` are labels stored as numbers, so they are one-hot encoded and never
used as integers; and `income` is missing for 62% of Latino and 60% of Asian participants against
31% of Black ones, which is information about origin rather than a hole to impute.

`data/processed/` is committed, so 04, 05 and 06 open without running 01 to 03 first.

## Running it

Python 3.11, and **the environment goes outside the repository**. Not a style preference: if the repo
sits under `~/Documents` with iCloud on, iCloud evicts the packages and every `import shap` waits on
a download, which looks exactly like a notebook that will not run.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # if you do not have uv
uv python install 3.11
uv venv --python 3.11 ~/.venvs/isaf
uv pip install --python ~/.venvs/isaf/bin/python -r requirements.txt
~/.venvs/isaf/bin/python -m ipykernel install --user --name isaf --display-name "Python (isaf)"
```

In VS Code: *Python: Select Interpreter* → `~/.venvs/isaf/bin/python`, then the **Python (isaf)**
kernel in notebooks.

> **macOS and XGBoost:** it needs OpenMP and will not import without it. `brew install libomp`, or
> copy the one Anaconda ships: `cp /opt/anaconda3/lib/libomp.dylib ~/.venvs/isaf/lib/`
>
> **Reproducibility:** every `LogisticRegression` carries `random_state=0`. liblinear shuffles
> internally, and without the seed the white box moves between two identical runs. XGBoost and
> TabPFN are unaffected.

### The app

```bash
streamlit run app/streamlit_app.py
```

Opens on `http://localhost:8501`. It refits nothing. Performance, economics and fairness are
recomputed live from `oof_predictions.csv` at whatever shortlist length you choose; interpretability
and stability are read from `reports/tables/`. It is written for the client, with four tabs:
*Compare the engines · How it decides · What it is worth · Is it fair?*

## Layout

```
├── app/            the Streamlit app
├── assets/logo/    client logo
├── data/
│   ├── raw/        the original dataset, never modified
│   └── processed/  what the notebooks build, committed
├── docs/           the dataset codebook
├── notebooks/      the analysis, numbered, run in order
└── reports/
    ├── figures/    every chart the notebooks draw
    └── tables/     every number quoted above
```
