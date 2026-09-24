"""Les Rito Mitsouka — the matching engine, for the client.

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

YELLOW, WHITE, SKY, PALE = "#FFD60A", "#FFFFFF", "#7FE3FF", "#9AA6E8"
SERIES = [YELLOW, WHITE, SKY, PALE]
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


def canvas(width, height):
    """A matplotlib figure that sits on the blue background instead of a white box."""
    figure, axis = plt.subplots(figsize=(width, height))
    figure.patch.set_alpha(0)
    axis.set_facecolor("none")
    axis.tick_params(colors=WHITE)
    for side in ["bottom", "left"]:
        axis.spines[side].set_color(PALE)
    axis.spines[["top", "right"]].set_visible(False)
    axis.xaxis.label.set_color(WHITE)
    axis.yaxis.label.set_color(WHITE)
    return figure, axis


oof, table, sides = load()
available = [c for c in NAMES if c in oof.columns]
base = oof.match.mean()

logo, headline = st.columns([1, 2])
logo.image("assets/logo/horizontal_blue.png")
headline.markdown(f"## The matching engine, in numbers\n"
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

PRICE, CONVERSION, MONTHS = 15, 0.12, 1.0
headline_value = 1000 * CONVERSION * PRICE * MONTHS

st.markdown("#### Headline")
a, b, c, d = st.columns(4)
a.metric("Matches per 1,000 shown", f"{1000 * rate[best]:.0f}",
         f"{1000 * (rate[best] - base):+.0f} vs chance")
b.metric("Better than chance", f"{rate[best] / base:.2f}×")
c.metric("Worth per 1,000 users a year", f"${headline_value:,.0f}",
         help="At $15 a month, 12% of users paying, one extra paid month each. "
              "Change the assumptions in 'What it is worth'.")
d.metric("Costs to run, same basis", f"${COST_PER_1000_USERS_PER_YEAR:.2f}",
         f"{headline_value / COST_PER_1000_USERS_PER_YEAR:,.0f}× return", delta_color="off")
st.caption(f"Best of the engines you selected: **{NAMES[best]}**, at a {k:.0%} shortlist.")

compare, engine, worth, fair = st.tabs(
    ["Compare the engines", "How it decides", "What it is worth", "Is it fair?"])

with compare:
    st.subheader("Matches per 1,000 pairs shown")
    figure, axis = plt.subplots(figsize=(8, 0.6 + 0.5 * len(models)))
    figure.patch.set_alpha(0)
    axis.set_facecolor("none")
    labels = ["Left to chance"] + [NAMES[m] for m in models]
    values = [1000 * base] + [1000 * rate[m] for m in models]
    bars = axis.barh(labels, values, color=[PALE] + SERIES[:len(models)])
    axis.bar_label(bars, fmt="%.0f", padding=4, color=WHITE, fontsize=11)
    axis.set_xlim(0, max(values) * 1.18)
    axis.tick_params(colors=WHITE)
    axis.get_xaxis().set_visible(False)
    axis.spines[:].set_visible(False)
    st.pyplot(figure)

    st.subheader("How the advantage changes with the length of the shortlist")
    steps = np.arange(0.02, 0.32, 0.02)
    figure, axis = canvas(8, 3.4)
    for m, colour in zip(models, SERIES):
        axis.plot([f"{s:.0%}" for s in steps],
                  [oof.match[shown(oof[m], s)].mean() / base for s in steps],
                  "o-", color=colour, label=NAMES[m], linewidth=2)
    axis.axhline(1, color=PALE, linestyle="--", linewidth=1)
    here = f"{min(steps, key=lambda s: abs(s - k)):.0%}"        # snap to a point on the curve
    axis.axvline(here, color=WHITE, alpha=0.16, linewidth=12, zorder=0)
    axis.annotate("your setting", xy=(here, axis.get_ylim()[1]), color=WHITE, fontsize=8,
                  ha="center", va="bottom", alpha=0.7)
    axis.set_ylabel("times better than chance")
    axis.set_xlabel("share of the catalogue shown")
    axis.legend(frameon=False, labelcolor=WHITE)
    st.pyplot(figure)
    st.markdown(
        "**Read it left to right.** A short shortlist is the most accurate — the engine is putting "
        "forward only the pairs it is surest about — but it delivers few matches in total. Stretch "
        "the list and every extra pair is a little less likely to work, so the curve falls towards "
        "1.00, which is chance. The highlighted band is where your slider sits.\n\n"
        "**Where the engines differ.** They are close, and that is the honest finding: on this data "
        "no engine is clearly ahead of the other. What separates them is not accuracy but what "
        "comes after — whether the recommendation can be explained, whether it survives "
        "retraining, and whether it treats groups evenly. The next three tabs.")

    st.subheader("The standard scores, for the record")
    scores = pd.DataFrame({
        NAMES[m]: {"PR-AUC": average_precision_score(oof.match, oof[m]),
                   "ROC-AUC": roc_auc_score(oof.match, oof[m]),
                   f"Match rate in the top {k:.0%}": rate[m],
                   "Lift over chance": rate[m] / base} for m in models}).round(3)
    st.dataframe(scores, width="stretch")
    left, right = st.columns(2)
    left.markdown("**PR-AUC** — the one to read. Only 16.5% of pairs match, and this measures how "
                  "well the engine finds those few rather than how well it recognises the many that "
                  "do not. Higher is better; chance is 0.165.")
    right.markdown("**ROC-AUC** — looks low on purpose. It scores the entire ranking, bottom "
                   "included, and the app never shows the bottom. Predicting attraction between two "
                   "strangers from a questionnaire is genuinely hard; what pays is the top of the "
                   "list, which is the row above.")

with engine:
    st.subheader("What makes two people say yes")
    st.markdown("The engine predicts **each person's decision separately and multiplies the two**, "
                "because a match is she says yes *and* he says yes. Below are the answers that move "
                "that decision most, in percentage points, for a one standard deviation change.")
    coefficients = pd.read_csv("reports/tables/04_white_box_coefficients.csv", index_col=0)
    effects = coefficients[coefficients.Step == "+1 sd"].head(10)["Marginal Effect (%)"].iloc[::-1]
    figure, axis = canvas(7, 4)
    axis.barh(effects.index, effects, color=np.where(effects > 0, YELLOW, SKY))
    axis.axvline(0, color=WHITE, linewidth=0.8)
    axis.set_xlabel("change in the chance of a yes (percentage points)")
    st.pyplot(figure)
    st.caption("No single answer moves the decision by more than four points. The engine works by "
               "adding up many small signals, which is also why it can be read line by line — every "
               "recommendation comes with the reasons behind it.")

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

with worth:
    st.subheader("What the engine is worth to you")
    st.markdown("Two numbers are yours, not ours: how many of your users pay, and how much longer a "
                "satisfied user stays. Set them here and the figures follow.")
    left, middle, right = st.columns(3)
    price = left.slider("Subscription price, per month", 5, 40, PRICE, 1, format="$%d")
    conversion = middle.slider("Share of users on a paid plan", 5, 25,
                               int(CONVERSION * 100), 1, format="%d%%") / 100
    months = right.slider("Extra paid months per subscriber per year", 0.0, 3.0, MONTHS, 0.25)
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

    figure, axis = canvas(7, 3)
    axis.barh(selection.index, 100 * selection, color=SKY)
    axis.axvline(100 * k, color=YELLOW, linewidth=2.5, label="equal treatment")
    axis.set_xlabel("share of a group's dates that get recommended (%)")
    axis.legend(frameon=False, labelcolor=WHITE)
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
    st.caption("People do prefer their own background — that is in the data, not in the model. What "
               "matters is whether the engine **exaggerates** it. Above 1.00 it does. Removing the "
               "effect entirely is possible, and costs about 50 matches per 1,000 recommendations. "
               "Dropping ethnicity from the model does not work: the engine rebuilds it from "
               "income, going-out habits and self-rated attractiveness, and segregation gets worse.")

st.divider()
st.caption("Out-of-fold results on the Speed Dating Experiment (Fisman, Iyengar, Kamenica & "
           "Simonson, 2006). Every pair was scored by a model trained without it, and without "
           "either of the two people — so these are cold-start numbers, what a new user gets.")
