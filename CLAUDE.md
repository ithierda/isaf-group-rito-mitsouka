# CLAUDE.md — ISAF project "Les Rito Mitsouka"

## Project
- X-HEC M2 course *Interpretability, Stability & Algorithmic Fairness* (Prof. C. Pérignon). Group of 6.
- Client: **Les Rito Mitsouka**, a fictional dating app deciding which pairs of profiles to recommend.
- Task: predict **`match`** (both said yes). Binary, 16.5% positives.
- Compare 3 models: white-box (logit / AdaLogit), ML (XGBoost), tabular foundation model (TabPFN or other, TBD).
- Assess each on 4 dimensions: performance (statistical + economic via a cost matrix), interpretability, stability, fairness.
- **Protected attribute: ethnicity** (`race`, `race_o`, `samerace`). Goal: check the model is not racially biased.
- Deliverables (Mon 28/09, 09:40): slides, notebook/code, Streamlit app. Internal target: done by Fri 25/09.

## Golden rules
- Everything in **English** (code, comments, notebooks, slides).
- **Nothing fancy.** Concise, clear, precise. No over-engineering, no unnecessary abstractions, no decorative output.
- Build **step by step with the user**; propose, explain briefly, wait for validation before big changes.
- Never modify `data/raw/`. Generated data goes to `data/processed/` (git-ignored).

## Data (see `data/README.md`, codebook in `docs/`)
- `data/raw/speed_dating.csv`: 8,378 rows x 195 cols, 551 participants, 21 waves. Load with `src.data.load_raw()` (Mac Roman encoding).
- Each date appears **twice** (`iid` / `pid`); 4,184 pairs, `match` identical on both rows. 10 rows have no `pid`.
- All pairs are woman–man → **decision: one row per pair**, woman's features vs man's features.
- Variable timing: 75 known **before** the date (usable), 21 **during** (`dec`, `attr`…`met`, `_o` versions → leakage), 87 **after** (`*_s`, `*_2`, `*_3` → leakage), 11 IDs/design.
- The partner's profile is not on a row (only `age_o`, `race_o`, `pf_o_*`): join via `pid` to compare profiles.
- Preference scales (`*1_1`, `*2_1`, `*4_1`): codebook says 1–10 for waves 6–9 and 100-point allocation otherwise → check and normalise.
- Ethnicity groups are unbalanced (Black ~5%, no Native American): fairness tests will have wide intervals.

## Method choices
- Split **by wave** (GroupKFold) — never split rows of the same pair or wave across train/test.
- Feature engineering = simple, mostly profile comparisons (age gap, shared interests, same field, preference alignment…).

## Layout
`data/` · `docs/` · `notebooks/` (numbered: 01_eda, 02_prep, …) · `src/` (shared code) · `app/` (Streamlit) · `models/` · `reports/figures/` · `assets/logo/`

## Environment
Python 3.11, `.venv`, `pip install -r requirements.txt`. macOS: `brew install libomp` for XGBoost.
