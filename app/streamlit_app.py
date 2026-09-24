"""Les Rito Mitsouka: the matching engine, for the client.

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
TRAIT = {"selfintel": "thinks they are intelligent", "selfattr": "thinks they are good looking",
         "selfamb": "thinks they are ambitious", "selffun": "thinks they are fun",
         "selfsinc": "thinks they are sincere", "optimism": "expects a good evening",
         "wantshar": "wants shared interests", "wantsinc": "wants someone sincere",
         "wantattr": "wants someone attractive", "wantamb": "wants someone ambitious",
         "wantintel": "wants someone intelligent", "wantfun": "wants someone fun",
         "datefreq": "dates rarely", "outfreq": "goes out rarely",
         "racepref": "cares about background", "religpref": "cares about religion",
         "income": "earns more", "age": "is older", "noincome": "left income blank",
         "fit": "got what they asked for"}
PAIR_FEATURE = {"raceclash": "One cares about background, and the pair is mixed",
                "samerace": "Same background", "samefield": "Same field of study",
                "samecareer": "Same career", "agegap": "Age difference",
                "agediff": "Age difference", "interests": "How alike their interests are",
                "passions": "Shared passions", "fitmin": "The worse of the two fits",
                "outgap": "Different going out habits", "dategap": "Different dating habits",
                "olderwoman": "She is older", "racepref_max": "At least one cares about background",
                "religpref_max": "At least one cares about religion",
                "female": "The decider is the woman"}


def plain(column):
    """Turn a model column name into something a client can read."""
    if column in PAIR_FEATURE:
        return PAIR_FEATURE[column]
    who, _, trait = column.partition("_")
    if who in {"self", "other"} and trait in TRAIT:
        return f"{'Themselves' if who == 'self' else 'Partner'}: {TRAIT[trait]}"
    return column.replace("_", " ")
NAMES = {"logit": "Logistic regression", "xgboost": "XGBoost", "tabpfn": "TabPFN",
         "logit_direct": "Logistic regression, direct", "xgboost_direct": "XGBoost, direct"}

# Measured in 04: 2.8 microseconds to score one pair, 30,000 scores per user per month.
COST_PER_1000_USERS_PER_YEAR = 0.042

st.set_page_config(page_title="Les Rito Mitsouka", page_icon="assets/logo/icon_blue.png",
                   layout="wide")
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 2.9rem; font-weight: 800; line-height: 1.1; }
[data-testid="stMetricLabel"] p { font-size: 0.95rem; font-weight: 600; opacity: 0.85; }
[data-testid="stMetricDelta"] { font-size: 1rem; font-weight: 600; }
.stTabs [data-baseweb="tab-list"] { gap: 10px; border-bottom: none; }
.stTabs [data-baseweb="tab"] {
  background: rgba(255,255,255,0.10); border-radius: 10px 10px 0 0;
  padding: 14px 26px; font-size: 1.1rem; font-weight: 700; }
.stTabs [aria-selected="true"] { background: #FFD60A; color: #101014; }
.stTabs [data-baseweb="tab-highlight"] { display: none; }
</style>""", unsafe_allow_html=True)


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
         f"{1000 * (rate[best] - base):+.0f} vs chance",
         help="Of 1,000 pairs the app puts in front of people, how many say yes to each other.")
b.metric("Better than chance", f"{rate[best] / base:.2f}×",
         help="That match rate divided by the 16.5% you get picking pairs at random.")
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
    st.markdown("A short list is picky and accurate. A long list finds more matches but each one "
                "is less likely. At 1.00 the engine is no better than chance. The yellow band is "
                "your slider.")
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

    st.subheader("The standard scores")
    st.markdown(
        f"| Metric | What it measures | Chance | Better is |\n|---|---|---|---|\n"
        f"| PR-AUC | Finding the few pairs that match. **The one to read.** | 0.165 | higher |\n"
        f"| ROC-AUC | The whole ranking, including the bottom the app never shows. | 0.500 | higher |\n"
        f"| Match rate in the top {k:.0%} | Of the pairs shown, the share that match. | 16.5% | higher |\n"
        f"| Lift over chance | That rate divided by 16.5%. | 1.00× | higher |")
    scores = pd.DataFrame({
        NAMES[m]: {"PR-AUC": average_precision_score(oof.match, oof[m]),
                   "ROC-AUC": roc_auc_score(oof.match, oof[m]),
                   f"Match rate in the top {k:.0%}": rate[m],
                   "Lift over chance": rate[m] / base} for m in models}).round(3)
    st.dataframe(scores, width="stretch")

with engine:
    st.subheader("What makes two people say yes")
    st.markdown("A match is she says yes **and** he says yes, so the engine predicts each decision "
                "and multiplies them. Below: how much each answer moves the chance of a yes, in "
                "percentage points. Nothing moves it more than four points, so the engine adds up "
                "many small signals rather than following one rule.")
    coefficients = pd.read_csv("reports/tables/04_white_box_coefficients.csv", index_col=0)
    effects = coefficients[coefficients.Step == "+1 sd"].head(10)["Marginal Effect (%)"].iloc[::-1]
    effects.index = [plain(c) for c in effects.index]
    figure, axis = canvas(8.5, 4)
    axis.barh(effects.index, effects, color=np.where(effects > 0, YELLOW, SKY))
    axis.axvline(0, color=WHITE, linewidth=0.8)
    axis.set_xlabel("change in the chance of a yes (percentage points)")
    st.pyplot(figure)

    st.subheader("Does it say the same thing every time?")
    st.markdown("Retrain on a different sample of dates and about half the shortlist changes. "
                "Refresh recommendations regularly rather than present them as a verdict.")
    stability = pd.read_csv("reports/tables/05_ranking_stability.csv")
    folds = stability[stability.versions == "fold"].set_index("model")
    columns = st.columns(max(len(models), 2))
    for column, m in zip(columns, [m for m in models if m in folds.index]):
        column.metric(f"{NAMES[m]}: shortlist kept after retraining",
                      f"{folds.loc[m, 'Jaccard on the top decile']:.0%}")

with worth:
    st.subheader("What the engine is worth to you")
    st.markdown("Two numbers are yours: how many users pay, and how much longer a satisfied one "
                "stays. Set them and the figures follow.")
    left, middle, right = st.columns(3)
    price = left.slider("Subscription price, per month", 5, 40, PRICE, 1, format="$%d")
    conversion = middle.slider("Share of users on a paid plan", 5, 25,
                               int(CONVERSION * 100), 1, format="%d%%") / 100
    months = right.slider("Extra paid months per subscriber per year", 0.0, 3.0, MONTHS, 0.25)
    value = 1000 * conversion * price * months

    st.caption(f"Scoring one pair takes 2.8 microseconds, so running the engine costs "
               f"${COST_PER_1000_USERS_PER_YEAR:.2f} per 1,000 users per year. The real spending is "
               f"collecting profile data and monitoring fairness: people, not servers. For "
               f"reference, the paid share across the market runs 8% to 15% (Tinder 8.6M payers "
               f"against roughly 60M monthly users, Grindr 8.4%).")
    left, middle, right = st.columns(3)
    left.metric("Value per 1,000 users per year", f"${value:,.0f}")
    middle.metric("Cost of running it", f"${COST_PER_1000_USERS_PER_YEAR:.2f}")
    right.metric("Return", f"{value / COST_PER_1000_USERS_PER_YEAR:,.0f}×" if value else "n/a")

with fair:
    st.subheader("Who gets recommended")
    picked = st.radio("Engine", models, format_func=lambda m: NAMES[m], horizontal=True)
    person = pd.concat([pd.Series(scored[picked].shown)] * 2, ignore_index=True)
    seen = sides.notna()
    selection = person[seen].groupby(sides[seen]).mean().sort_values()

    st.markdown(f"Equal treatment means every bar on the yellow line. The widest gap is "
                f"**{100 * (selection.max() - selection.min()):.1f} points**, between "
                f"{selection.idxmax()} and {selection.idxmin()} participants.")
    figure, axis = canvas(7, 3)
    axis.barh(selection.index, 100 * selection, color=SKY)
    axis.axvline(100 * k, color=YELLOW, linewidth=2.5, label="equal treatment")
    axis.set_xlabel("share of a group's dates that get recommended (%)")
    axis.legend(frameon=False, labelcolor=WHITE)
    st.pyplot(figure)

    same = table.samerace.to_numpy()
    amplification = same[scored[picked].shown].mean() / same[oof.match == 1].mean()
    st.markdown("People prefer their own background: that is in the data, not the model. The "
                "question is whether the engine **exaggerates** it. Above 1.00 it does. Removing "
                "the effect costs about 50 matches per 1,000 recommendations, and simply deleting "
                "the ethnicity column makes it worse, because the engine rebuilds it from income "
                "and lifestyle answers.")
    left, right = st.columns(2)
    left.metric("Same-background pairs in the shortlist",
                f"{100 * same[scored[picked].shown].mean():.0f}%",
                help="Of the pairs the app would show, the share where both have the same "
                     "background.")
    right.metric("Among the pairs that really matched",
                 f"{100 * same[oof.match == 1].mean():.0f}%",
                 f"engine is {amplification:.2f}× more segregated", delta_color="inverse",
                 help="The same share among the dates that actually ended in a match. The engine "
                      "should not exceed it.")

st.divider()
st.caption("Speed Dating Experiment (Fisman, Iyengar, Kamenica & Simonson, 2006). Every pair was "
           "scored by a model trained without it and without either person, so these are "
           "cold-start numbers: what a brand new user gets.")
