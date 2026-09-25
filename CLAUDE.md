# CLAUDE.md: ISAF project "Les Rito Mitsouka"

## Project
- X-HEC M2 course *Interpretability, Stability & Algorithmic Fairness* (Prof. C. Pérignon). Group of 6.
- Client: **Les Rito Mitsouka**, a fictional dating app deciding which pairs of profiles to recommend.
- Task: predict **`match`** (both said yes). Binary, 16.5% positives.
- Three models: **lasso logistic regression** (white box), **XGBoost**, **TabPFN** (foundation model, run on Colab).
- All three predict **the two decisions separately and multiply them**, not `match` directly.
- Four dimensions: performance (statistical + economic), interpretability, stability, fairness.
- **Protected attribute: ethnicity.** Goal: check the recommendation engine is not racially biased.
- Deliverables (Mon 28/09, 09:40): slides, notebooks, Streamlit app.

## Golden rules
- Everything in **English** (code, comments, notebooks, slides).
- **Nothing fancy.** Concise, clear, precise. No over-engineering, no decorative output.
- Propose, explain briefly, wait for validation before big changes.
- Never modify `data/raw/`. `data/processed/` **is committed**, so the team has the tables without rerunning anything.

## Data (see `data/README.md`, codebook in `docs/`)
- `data/raw/speed_dating.csv`: 8,378 rows x 195 cols, 551 participants, 21 waves, Mac Roman encoding.
- Each date appears **twice**; 4,184 pairs, all woman/man, **one row per pair**, `her_*` and `his_*`.
- 108 of the 195 columns are filled in during or after the date → leakage, removed in 01.
- `date` and `go_out` are **reverse coded** (1 = several times a week, 7 = almost never).
- `field_cd`, `career_c`, `goal`, `race` are labels stored as numbers → one-hot, never integers.
- `income` is missing for 62% of Latino and 60% of Asian participants against 31% of Black ones: the
  missingness itself carries information about origin, so it has its own flag.
- Only the `*4_1` preference block changes scale (absent in waves 1 to 5, scored 1 to 10 in waves 6 to 9). `*1_1` and
  `*2_1` are on the 100-point scale everywhere, contrary to what the codebook suggests.

## Method
- **Split on `wave` with `StratifiedGroupKFold(5)`**, fold stored in `model_table.csv`. Every participant
  attends exactly one wave, so it is the only grouping that keeps a person on one side. A random split
  gives 0.72 ROC-AUC against 0.60, and the gap is the model recognising people, not skill.
- **No feature constant within a wave** (that column is the wave under another name).
- Encoding with `OneHotEncoder` **inside the pipeline**, fitted on the training fold only.
- Column names: `her_` / `his_` for a person, a plain word for the pair, one underscore at most.
- `raceclash` = someone cares about background **and** the pair is mixed. Strongest single feature
  (12.6% match against 18.0%) and built from ethnicity, so the fairness notebook must examine it.
- **Two-stage target.** A match is she says yes and he says yes. Each pair becomes two decision rows
  (`self_` = the decider, `other_` = the partner, plus a `female` flag), the model predicts `dec`, and
  the two probabilities are multiplied. 8,368 rows, target at 37%/47% instead of a 16.5% joint event.
  Top decile: 30% matches against 23% predicting `match` directly. `dec` is a target, never a feature.

## Conventions in 04, 05, 06, 07
- **Course only.** The methods, tables and their order come from the lectures. Exactly three things
  are ours and each is labelled in the text: Ŷ = top decile instead of a 0.5 threshold (no pair ever
  reaches 0.5), the shortlist overlap in 05, and the amplification ratio in 06. Nothing else.
- **Every section ends with "What it says" and "Conclusion".** Keep that shape.
- **The two stages add up in logs.** SHAP is computed per stage on the log-odds; `Σφ + φ₀ = margin`
  holds, the probability product does not decompose additively.
- 04 rebuilds the pipelines from 03: the **final model** (all 8,368 decisions) is what we explain,
  the **five fold models** are what we measure performance on. 05 and 06 refit by design.
- **The logit is not reproducible across environments.** liblinear with L1 moves individual
  predictions by up to 0.1 between package versions; XGBoost is bit-identical. Aggregate performance
  does not move, but 05's coefficient distances do.

## Layout
`data/raw` · `data/processed` (committed) · `docs/` · `notebooks/` (01_eda, 02_features, 03_models,
03b_tabpfn_colab, 04_interpretability, 05_stability, 06_fairness, 07_recommendation) ·
`app/` (Streamlit) ·
`reports/figures/` · `reports/tables/` · `assets/logo/`

Notebooks are self-contained, with no shared module, and move to the repo root on their first cell.

## Environment
Python 3.11 in `.venv`, created with `uv` (no Homebrew on the machine). XGBoost needs `libomp`; see the README.
- **iCloud evicts `.venv`** when the repo lives under `~/Documents`: imports then hang for minutes.
- `XPER` (the professor's package) is in `requirements.txt`. Two traps: `kernel=False` is broken in
  0.0.92, and `N_coalition_sampled` must stay at or below `2**p - 2` or it raises `IndexError`.

## Open
- **Cost matrix** is undecided, and economic performance is graded.
- Slides not started. TabPFN is done (Julian), merged into `oof_predictions.csv`.
