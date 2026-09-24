"""Les Rito Mitsouka — what the recommendation engine does, for the client.

Everything shown here is measured on pairs the model never saw during training, so it is what a
brand new user would get. Run it from the repository root: streamlit run app/streamlit_app.py
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score, roc_auc_score

if not Path("data").is_dir():
    os.chdir("..")

BLUE, YELLOW, INK, GREY = "#1E34C8", "#FFD60A", "#101014", "#B8BCC8"
RACE = {1: "Black", 2: "White", 3: "Latino", 4: "Asian", 5: "Native American", 6: "Other"}
NAMES = {"logit": "Logistic regression", "xgboost": "XGBoost", "tabpfn": "TabPFN",
         "logit_direct": "Logistic regression, direct", "xgboost_direct": "XGBoost, direct"}

# Measured in 04: 2.8 microseconds to score one pair, 30,000 scores per user per month.
COST_PER_1000_USERS_PER_YEAR = 0.042

st.set_page_config(page_title="Les Rito Mitsouka", page_icon="assets/logo/icon_blue.png",
                   layout="wide")


@st.cache_data
def load():
    oof = pd.read_csv("data/processed/oof_predictions.csv")
    table = pd.read_csv("data/processed/model_table.csv")
    sides = pd.concat([table.her_race, table.his_race], ignore_index=True).map(RACE)
    return oof, table, sides


def shown(scores, k):
    return (scores >= np.quantile(scores, 1 - k)).to_numpy()


oof, table, sides = load()
available = [c for c in NAMES if c in oof.columns]
base = oof.match.mean()

logo, headline = st.columns([1, 2])
logo.image("assets/logo/horizontal_blue.png")
headline.markdown(f"## Which pairs should we put in front of each other?\n"
                  f"Measured on **{len(oof):,} real dates** the engine had never seen. "
                  f"Left to chance, **{base:.1%}** of pairs match.")

with st.container(border=True):
    left, right = st.columns([2, 3])
    models = left.multiselect("Engines to compare", available,
                              default=[m for m in ["logit", "xgboost"] if m in available],
                              format_func=lambda m: NAMES[m])
    k = right.slider("How much of the catalogue the app shows", 1, 30, 10, 1, format="%d%%") / 100
    right.caption("Showing fewer pairs means better ones, but fewer matches in total. "
                  "This is the product decision, and every number below follows it.")
    if "tabpfn" not in oof.columns:
        left.caption("TabPFN joins the list once `tabpfn_oof.csv` is in `data/processed/`.")

if not models:
    st.warning("Pick at least one engine above.")
    st.stop()

scored = {m: pd.DataFrame({"match": oof.match, "shown": shown(oof[m], k)}) for m in models}
rate = {m: d.match[d.shown].mean() for m, d in scored.items()}
best = max(models, key=lambda m: rate[m])

results, engine, worth, fair = st.tabs(
    ["Does it work?", "How it decides", "What it is worth", "Is it fair?"])

with results:
    st.subheader("Matches per 1,000 pairs shown")
    columns = st.columns(len(models) + 1)
    columns[0].metric("Left to chance", f"{1000 * base:.0f}")
    for column, m in zip(columns[1:], models):
        column.metric(NAMES[m], f"{1000 * rate[m]:.0f}", f"{1000 * (rate[m] - base):+.0f} vs chance")
    st.markdown(f"At a {k:.0%} shortlist, {NAMES[best]} turns {1000 * base:.0f} matches per thousand "
                f"recommendations into {1000 * rate[best]:.0f} — **{rate[best] / base:.2f}× better "
                f"than picking at random.** That multiplier is the promise the app can make to a "
                f"new user.")

    steps = np.arange(0.02, 0.32, 0.02)
    figure, axis = plt.subplots(figsize=(8, 3.2))
    for m, colour in zip(models, [BLUE, YELLOW, INK, GREY]):
        axis.plot([f"{s:.0%}" for s in steps],
                  [oof.match[shown(oof[m], s)].mean() / base for s in steps],
                  "o-", color=colour, label=NAMES[m], linewidth=2)
    axis.axhline(1, color=GREY, linestyle="--", linewidth=1)
    axis.set_ylabel("times better than chance")
    axis.set_xlabel("share of the catalogue shown")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    st.pyplot(figure)
    st.caption("The shorter the shortlist, the better each recommendation — and the fewer matches "
               "in total. Where to sit on this curve depends on what a bad recommendation costs "
               "you in user patience.")

    st.subheader("The standard scores, for the record")
    scores = pd.DataFrame({
        NAMES[m]: {"PR-AUC": average_precision_score(oof.match, oof[m]),
                   "ROC-AUC": roc_auc_score(oof.match, oof[m]),
                   f"Match rate in the top {k:.0%}": rate[m],
                   "Lift over chance": rate[m] / base} for m in models}).round(3)
    st.dataframe(scores, width="stretch")
    st.caption("**PR-AUC** is the one to read: only 16.5% of pairs match, and it measures how well "
               "the engine finds those few. **ROC-AUC** looks low on purpose — it scores the whole "
               "ranking, including the bottom, which the app never shows. Predicting attraction "
               "between two strangers from a questionnaire is genuinely hard; what matters "
               "commercially is the top of the list.")

with engine:
    st.subheader("What makes two people say yes")
    st.markdown("The engine predicts **each person's decision separately and multiplies the two**, "
                "because a match is she says yes *and* he says yes. Below are the answers that move "
                "that decision most, in percentage points of probability, for a one standard "
                "deviation change.")
    coefficients = pd.read_csv("reports/tables/04_white_box_coefficients.csv", index_col=0)
    effects = coefficients[coefficients.Step == "+1 sd"].head(10)["Marginal Effect (%)"].iloc[::-1]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.barh(effects.index, effects, color=np.where(effects > 0, BLUE, YELLOW))
    axis.axvline(0, color=INK, linewidth=0.8)
    axis.set_xlabel("change in the chance of a yes (percentage points)")
    axis.spines[["top", "right"]].set_visible(False)
    st.pyplot(figure)
    st.caption("No single answer moves the decision by more than four points. The engine works by "
               "adding up many small signals, which is also why it can be read line by line — "
               "every recommendation comes with the reasons behind it.")

    st.subheader("Does it say the same thing every time?")
    stability = pd.read_csv("reports/tables/05_ranking_stability.csv")
    folds = stability[stability.versions == "fold"].set_index("model")
    columns = st.columns(max(len(models), 2))
    for column, m in zip(columns, [m for m in models if m in folds.index]):
        column.metric(f"{NAMES[m]}: shortlist kept after retraining",
                      f"{folds.loc[m, 'Jaccard on the top decile']:.0%}")
    st.caption("Retrain on a different sample of dates and about half the shortlist changes. That "
               "is normal at this sample size, and it is the reason to refresh recommendations "
               "regularly rather than present them as a verdict.")
    st.image("reports/figures/04_shap_beeswarm.png",
             caption="Each dot is one date. The engine's reasoning can be opened up one "
                     "recommendation at a time.")

with worth:
    st.subheader("What the engine is worth to you")
    st.markdown("Two numbers are yours, not ours: how many of your users pay, and how much longer "
                "a satisfied user stays. Set them here.")
    left, middle, right = st.columns(3)
    price = left.slider("Subscription price, per month", 5, 40, 15, 1, format="$%d")
    conversion = middle.slider("Share of users on a paid plan", 5, 25, 12, 1, format="%d%%") / 100
    months = right.slider("Extra paid months per subscriber per year", 0.0, 3.0, 1.0, 0.25)
    value = 1000 * conversion * price * months

    left, middle, right = st.columns(3)
    left.metric("Value per 1,000 users per year", f"${value:,.0f}")
    middle.metric("Cost of running it", f"${COST_PER_1000_USERS_PER_YEAR:.2f}")
    right.metric("Return", f"{value / COST_PER_1000_USERS_PER_YEAR:,.0f}×" if value else "—")
    st.caption(f"Scoring one pair takes 2.8 microseconds, so the engine costs "
               f"${COST_PER_1000_USERS_PER_YEAR:.2f} per 1,000 users per year to run. It cannot be "
               f"priced on what it costs. The real spending is collecting profile data and "
               f"monitoring fairness — people, not servers. The market band for the paid share is "
               f"8% to 15%: Tinder reports 8.6 million payers against roughly 60 million monthly "
               f"users, Grindr publishes 8.4%.")

with fair:
    st.subheader("Who gets recommended")
    picked = st.radio("Engine", models, format_func=lambda m: NAMES[m], horizontal=True)
    person = pd.concat([pd.Series(scored[picked].shown)] * 2, ignore_index=True)
    seen = sides.notna()
    selection = person[seen].groupby(sides[seen]).mean().sort_values()

    figure, axis = plt.subplots(figsize=(7, 3))
    axis.barh(selection.index, 100 * selection, color=BLUE)
    axis.axvline(100 * k, color=YELLOW, linewidth=2.5, label="equal treatment")
    axis.set_xlabel("share of a group's dates that get recommended (%)")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    st.pyplot(figure)
    st.markdown(f"If the engine treated every group the same, all bars would sit on the yellow "
                f"line. The widest gap is **{100 * (selection.max() - selection.min()):.1f} "
                f"points**, between {selection.idxmax()} and {selection.idxmin()} participants.")

    same = table.samerace.to_numpy()
    amplification = same[scored[picked].shown].mean() / same[oof.match == 1].mean()
    left, right = st.columns(2)
    left.metric("Same-background pairs in the shortlist",
                f"{100 * same[scored[picked].shown].mean():.0f}%")
    right.metric("Among the pairs that really matched",
                 f"{100 * same[oof.match == 1].mean():.0f}%",
                 f"engine is {amplification:.2f}× more segregated", delta_color="inverse")
    st.caption("People do prefer their own background — that is in the data, not in the model. "
               "What matters is whether the engine **exaggerates** it. Above 1.00 it does. Removing "
               "the effect entirely is possible, and costs about 50 matches per 1,000 "
               "recommendations. Dropping ethnicity from the model does not work: the engine "
               "rebuilds it from income, going-out habits and self-rated attractiveness, and the "
               "segregation gets worse.")

st.divider()
st.caption("Out-of-fold results on the Speed Dating Experiment (Fisman, Iyengar, Kamenica & "
           "Simonson, 2006). Every pair was scored by a model trained without it, and without "
           "either of the two people — so these are cold-start numbers, what a new user gets.")
