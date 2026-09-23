# CLAUDE.md — ISAF project "Les Rito Mitsouka"

## Project
- X-HEC M2 course *Interpretability, Stability & Algorithmic Fairness* (Prof. C. Pérignon). Group of 6.
- Client: **Les Rito Mitsouka**, a fictional dating app deciding which pairs of profiles to recommend.
- Task: predict **`match`** (both said yes). Binary, 16.5% positives.
- Three models: **lasso logistic regression** (white box), **XGBoost**, **TabPFN** (foundation model, run on Colab).
- Four dimensions: performance (statistical + economic), interpretability, stability, fairness.
- **Protected attribute: ethnicity.** Goal: check the recommendation engine is not racially biased.
- Deliverables (Mon 28/09, 09:40): slides, notebooks, Streamlit app.

## Golden rules
- Everything in **English** (code, comments, notebooks, slides).
- **Nothing fancy.** Concise, clear, precise. No over-engineering, no decorative output.
- Propose, explain briefly, wait for validation before big changes.
- Never modify `data/raw/`. `data/processed/` **is committed** — the team needs the tables without rerunning anything.

## Data (see `data/README.md`, codebook in `docs/`)
- `data/raw/speed_dating.csv`: 8,378 rows x 195 cols, 551 participants, 21 waves, Mac Roman encoding.
- Each date appears **twice**; 4,184 pairs, all woman–man → **one row per pair**, `her_*` and `his_*`.
- 108 of the 195 columns are filled in during or after the date → leakage, removed in 01.
- `date` and `go_out` are **reverse coded** (1 = several times a week, 7 = almost never).
- `field_cd`, `career_c`, `goal`, `race` are labels stored as numbers → one-hot, never integers.
- `income` is missing for 62% of Latino and 60% of Asian participants against 31% of Black ones: the
  missingness itself carries information about origin, so it has its own flag.
- Only the `*4_1` preference block changes scale (absent in waves 1–5, 1–10 in waves 6–9). `*1_1` and
  `*2_1` are on the 100-point scale everywhere, contrary to what the codebook suggests.

## Method
- **Split on `wave` with `StratifiedGroupKFold(5)`**, fold stored in `model_table.csv`. Every participant
  attends exactly one wave, so it is the only grouping that keeps a person on one side. A random split
  gives 0.72 ROC-AUC against 0.60 — the gap is the model recognising people, not skill.
- **No feature constant within a wave** (that column is the wave under another name).
- Encoding with `OneHotEncoder` **inside the pipeline**, fitted on the training fold only.
- Column names: `her_` / `his_` for a person, a plain word for the pair, one underscore at most.

## Layout
`data/raw` · `data/processed` (committed) · `docs/` · `notebooks/` (01_eda, 02_features, 03_models,
03b_tabpfn_colab) · `app/` (Streamlit) · `reports/figures/` · `assets/logo/`

Notebooks are self-contained — no shared module — and move to the repo root on their first cell.

## Environment
Python 3.11 in `.venv`, created with `uv` (no Homebrew on the machine). XGBoost needs `libomp`; see the README.

## Open
- **Cost matrix** — undecided, and economic performance is graded.
- Notebooks 04 (interpretability), 05 (stability), 06 (fairness) read `oof_predictions.csv` and refit nothing.
