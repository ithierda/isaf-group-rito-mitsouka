# Fixes for `ISAF Project - Rito Mitsouka.pptx`

Read against the deck of 27/09, 14 slides. Ordered by urgency.

---

## Urgent, these are visible on screen

**Three working notes are still in the deck.** They will be projected.

| Slide | Text to delete |
|---|---|
| 5 | *"À remplacer par les vrais chiffres après:"* |
| 6 | *"Peut-être plutôt mettre le bar chart avec le threshold (ce que Justin a fait sur le streamlit)"* |
| 13 | *"Pas convaincu de cette slide de claude"* |

---

## Numbers that no longer match the tables

The logistic regressions were reseeded on 26/09, which moved a few figures. Current values:

| Slide | Says | Should say |
|---|---|---|
| 7 | Top 5%: 314, +149 | **319, +154** |
| 8 | Top 10% Match Rate 29.4% | **29.8%** |
| 12 | Same-background among recommendations 48% | **47.5%** |
| 12 | 95% CI [1.03, 1.30] | **[1.03, 1.29]** |
| 12 | Ethnicity neutralised 1.57 / 24.8% | **1.59 / 24.6%** |

---

## The one real error: slide 9 shows the wrong half of the model

Every row in the table is a dummy, a 0 to 1 jump. That is why the list is topped by
`self_goal=rare`, `other_field=social work` and `other_race=rare`, which have **p-values of 0.567 and
0.673**. A jury reading a "top features" table with insignificant rare-category buckets in it will
ask why, and the honest answer is that the table sorted on a scale that favours dummies.

A 0 to 1 jump is mechanically a bigger move than one standard deviation, so dummies always win that
ranking. The profile answers, which are what the client cares about, are pushed off the slide.

**Replace the table with this one** (continuous columns, `04_white_box_coefficients.csv`):

| Feature | Coefficient | P-value | Odds ratio | Marginal effect | Scale |
|---|---|---|---|---|---|
| `female` | −0.314 | < 0.0001 | 0.730 | **−13.7 pts** | 0 to 1 |
| `other_selfintel` | −0.186 | < 0.0001 | 0.830 | −4.0 pts | +1 sd |
| `self_optimism` | +0.161 | < 0.0001 | 1.175 | +3.5 pts | +1 sd |
| `raceclash` | −0.156 | < 0.0001 | 0.856 | −3.4 pts | +1 sd |
| `self_fit` | +0.137 | 0.072 | 1.147 | +3.0 pts | +1 sd |
| `self_wantshar` | +0.127 | 0.010 | 1.136 | +2.8 pts | +1 sd |

Keep `female` at the top, it is genuinely the largest effect in the model. Then the continuous ones.

**The line to say**, and it is a strength rather than an admission: *"a model whose strongest profile
variable moves the answer by four points works by combining many small effects, not by finding a
rule."*

**Add one line of method**, because the p-values need it: the lasso selects the columns, an
unpenalised logit is refitted on them for the standard errors, so read the p-values as an ordering.

---

## What is missing, and it is graded

### 1. There is no slide on how we validated

This is the strongest technical content we have and the deck goes from the data straight to the
business model. The first question a jury asks is how you know the number is real.

**New slide, after slide 4.** Title: *Why we do not split at random*.

- Everyone meets everyone of the opposite sex in their evening, so the "dated each other" graph has
  **21 connected components for 21 waves**. There is no finer cut that keeps a person on one side.
- **Random split 0.72 ROC-AUC. Grouped split 0.60.** The gap is the model recognising people.
- **The wave is a grouping key, never a feature.** We remove it, and everything constant within it.
- Removing the identifiers is not enough: four ordinary profile columns identify 90% of participants.

*Say:* the client is launching, so every user is new. The grouped number is the one they will get.

### 2. Interpretability is one dimension out of four and has one slide

Slide 9 is a coefficient table. There is no SHAP, no LIME, and no XPER, which is the professor's own
package.

**New slide, after 9.** Title: *One recommendation, explained*.

- Insert `reports/figures/04_shap_waterfall.png`.
- Three LIME rules as text, from the same pair.
- **The four importance methods share 30% of their top five. Impurity and XPER share nothing.**

*Say:* SHAP allocates the prediction exactly, LIME phrases it as conditions, and they agree. The two
stages add up in logs, not in probabilities. Then the punchline: *"this is the argument for not
putting a single importance chart on a slide. If one has to be chosen it is XPER, because the client
asks where the skill comes from, not where the tree looked."*

Optional inset: `reports/figures/04_tabpfn_importance.png`. TabPFN has no coefficients and no trees,
and permutation importance puts **`race` first** for it too.

### 3. Room for both: merge slide 8 into slide 6

Slide 8 repeats slide 6 with a stale number. Fold the three big figures (29.8%, 1.8×, 16.5%) into
slide 6 as the headline above the table, and the deck stays at 14 slides.

---

## Two improvements

**Slide 5, cut the "calibrated advantage" line.** *"Enough to create value without making the user
experience feel solved too quickly"* reads as deliberately holding the engine back, in a course on
trustworthy AI. It is also unnecessary: **70% of our best recommendations still do not match**, so
there is no scenario where the engine empties the app. Replace with the retention argument, which is
the same business point without the exposure.

Slide 5 also has no figures. Ours: scoring one pair takes microseconds, the engine runs for **a few
cents per 1,000 users per year**, and at $15 a month with 12% conversion it is worth **$1,800 per
1,000 users per year**. The market band for the paid share is 8% to 15%: Tinder reports 8.6 million
payers against roughly 60 million monthly users, Grindr publishes 8.4%.

**Slide 6, insert `reports/figures/07_top_decile.png`.** The three intervals overlap almost entirely
and each is eight points wide against a gap of one point between engines. It makes *"performance
cannot choose the engine"* something the room sees rather than something we assert. That is also what
the note on slide 6 was asking for.

**Slide 12, insert `reports/figures/06_amplification.png`.** The right-hand panel shows XGBoost's
interval crossing 1.00, so its amplification is not distinguishable from none, where TabPFN reaches
1.27. The tables on that slide are correct but they hide it.

---

## Slide 13 is fine

The note says otherwise, but the seven recommendations are the right seven and in the right order.
One thing to add out loud rather than on the slide: **XGBoost is the fairest of the three**,
amplification 1.01 against 1.16, and choosing it instead is defensible for about one point of
top-decile rate. Saying it before anyone asks is stronger than being asked.
