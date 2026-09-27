# Presentation plan. 15 minutes, 6 speakers

Jury: Prof. Pérignon, Monday 28/09. 15 min talk, 10 min Q&A.
Graded: technical /5, presentation /5, Q&A /5, slides, code and app /10.

Each slide below has **On the slide**, which is what to put in PowerPoint, and **Say**, which is not
written anywhere on it. Every number comes from `reports/tables/`. Copy, do not retype.

| Block | Slides | Speaker | Time |
|---|---|---|---|
| A. Client and data | 1 to 3 | 1 | 2:00 |
| B. The two choices that decide everything | 4 to 5 | 2 | 2:00 |
| C. Performance | 6 to 7 | 3 | 2:00 |
| D. Interpretability | 8 to 10 | 4 | 2:30 |
| E. Stability | 11 to 12 | 5 | 2:00 |
| F. Fairness | 13 to 15 | 6 | 2:30 |
| G. Recommendation and app | 16 to 18 | 1 again | 2:00 |

Rehearse at 13:30. A jury talk always runs long.

---

## Block A. Client and data, speaker 1, 2 min

### Slide 1. Title, 10 s
**On the slide:** logo, *Les Rito Mitsouka*, six names, course, date.
**Say:** our client is a dating app, it has to decide which pairs of users to put in front of each
other.

### Slide 2. The client's question, 50 s
**On the slide:**
- **Y = 1 if both people say yes. 16.5% of pairs.**
- The app shows a **shortlist**, not a yes or no on every pair.
- Three engines: lasso logistic regression, XGBoost, TabPFN.
- Four dimensions: performance, interpretability, stability, fairness.
- Protected attribute: **ethnicity**.

**Say:** the shortlist sentence justifies every methodological choice that follows, so land it.

### Slide 3. The data, 60 s
**On the slide:**
- Fisman, Iyengar, Kamenica and Simonson (2006), QJE.
- **8,378 rows → 4,184 pairs · 551 participants · 21 waves · 16.5% match.**
- Each date appears **twice** → one row per pair.
- **108 of 195 columns** are filled in during or after the date → leakage, removed.
- **Income is missing for 62% of Latino and 60% of Asian participants against 31% of Black ones.**

**Say:** flag the last one, it comes back in blocks F and G. No EDA gallery, three decisions, move on.

---

## Block B. The two choices, speaker 2, 2 min

The strongest technical content in the deck. Do not rush it.

### Slide 4. We split on the wave, 60 s
**On the slide:**
- The "dated each other" graph has **21 connected components for 21 waves**.
- Random split **0.72 ROC-AUC**, grouped split **0.60**. The gap is the model recognising people.
- **The wave is a grouping key, never a feature.** 02 removes it and everything constant within it.

**Say:** kill the objection before it is raised. Removing the identifiers is not enough: four ordinary
profile columns identify 90% of participants.

### Slide 5. A match is two decisions, 60 s
**On the slide:**
- A match is she says yes **and** he says yes.
- **8,368 decisions** instead of 4,184 pairs, target **37% and 47%** instead of 16.5%.
- Top decile **29.8% against 23.6%** predicting the match in one step.

**Say:** it costs nothing on ROC-AUC and gains a third where the app operates.

---

## Block C. Performance, speaker 3, 2 min

### Slide 6. Statistical performance, 60 s
**On the slide:** `reports/figures/07_top_decile.png`, full width.

| model | PR-AUC | top decile | lift |
|---|---|---|---|
| logit, two-stage | 0.228 | **0.298** | 1.81 |
| XGBoost, two-stage | 0.218 | 0.286 | 1.74 |
| TabPFN | **0.234** | 0.289 | 1.75 |

**Say:** the chart makes the argument on its own. *"The interval is eight points wide, the gap between
engines is one. Performance cannot choose for us."* Add that eighteen further engineered features all
landed inside the same interval, and we stopped rather than tune on noise.

### Slide 7. Economic performance, 60 s
**On the slide:**
- **+133 matches per 1,000 recommendations.** 298 against 165 at random.

| list shown | matches /1,000 | extra /1,000 | extra matches over the catalogue |
|---|---|---|---|
| top 5% | 319 | +154 | +32 |
| top 10% | 298 | +133 | +56 |
| top 20% | 235 | +70 | +59 |

- **Value per 1,000 = 133 × v**, v the net value of one match.
- Cost: **a few cents per 1,000 users per year**. Value at $15/month and 12% conversion: **$1,800**.

**Say:** the last two columns pull opposite ways, so the length of the list is the client's decision.
The transferable number is the **lift ×1.81**, not the absolute rate, because these people had just
met face to face. Then the bridge: a lift can be bought with exclusion, slides 13 to 15 check that.

---

## Block D. Interpretability, speaker 4, 2:30

### Slide 8. The white box, 60 s
**On the slide:** the lasso keeps **77 of 101 columns**.

| variable | effect | scale |
|---|---|---|
| `female` | **−13.7 pts** | 0 to 1 |
| `other_selfintel` | −4.0 pts | +1 sd |
| `self_optimism` | +3.5 pts | +1 sd |
| `raceclash` | −3.4 pts | +1 sd |

**Say:** the lasso selects, an unpenalised logit is refitted for the standard errors, so read the
p-values as an ordering. Then the honest line, which is a strength: *"a model whose strongest profile
variable moves the answer by four points works by combining many small effects, not by finding a
rule."* Name `raceclash` here, it is the hinge into slide 13.

### Slide 9. One prediction explained, 45 s
**On the slide:** `reports/figures/04_shap_waterfall.png`, plus three LIME rules as text.
**Say:** the two stages add up in logs, not in probabilities. SHAP allocates the prediction exactly,
LIME phrases it as conditions, and they agree. Product framing: *"why would she say yes to him"*.

### Slide 10. The four methods disagree, 45 s
**On the slide:**
- **The four share 30% of their top five. Impurity and XPER share nothing.**
- The comparison table, impurity · SHAP · permutation · XPER.
- Optional inset: `reports/figures/04_tabpfn_importance.png`.

**Say:** impurity ranks the field columns first because eighteen categories offer more places to
split. Permutation and XPER, both performance measures, put `female` and the ethnicity columns on
top. Punchline: *"if one has to be chosen it is XPER, because the client asks where the skill comes
from, not where the tree looked."* TabPFN has no coefficients and no trees, and permutation
importance puts **`race` first** for it too.

---

## Block E. Stability, speaker 5, 2 min

### Slide 11. Is it the same model twice, 60 s
**On the slide:** `reports/figures/05_distance_heatmaps.png`.

| model | instability | shortlist kept after retraining |
|---|---|---|
| logit | **0.56** | 48% |
| XGBoost | **0.30** | 44% |

**Say:** Turney's definition first, two datasets from the same population should give approximately
the same model, so stability is a distance between models and not the spread of a score. 25 versions
each, 5 folds plus 20 bootstraps within waves. Then the twist, our addition: the user sees a **list**,
and half of it changes on retraining. Only two of 55 live coefficients ever change sign.

### Slide 12. Over time, and the price of stability, 60 s
**On the slide:** `reports/figures/05_leave_one_wave_out.png`.
- **PR-AUC from 0.10 to 0.46** depending on the evening held out.
- Anchoring the fold-1 model **84% towards** fold 0 **raises** PR-AUC from **0.287 to 0.312**.

**Say:** give the client the range, not the average. The drift column correlates 0.96 with the number
of pairs removed, so it is a sample-size effect. And there is **no trade-off to arbitrate** here:
anchoring is free regularisation. That feeds slide 17.

---

## Block F. Fairness, speaker 6, 2:30

Frame it in the first ten seconds or the jury argues the wrong question.
**Say:** *"not whether the participants preferred their own background, they did and that is
descriptive, but whether the engine disadvantages a group and whether it amplifies that preference."*

### Slide 13. The gaps, and whether they survive, 60 s
**On the slide:**
- Mapping, small: Y = matched · **Ŷ = in the top decile** · D = protected.
- **Asian participants appear in 5.8% of shown pairs against 11.3% for everyone else. Black 19.3%.**
- Conditional parity, Cochran Mantel Haenszel on four legitimate variables: **5 of the 6 gaps
  survive.** Only White is green.
  Asian odds ratio **0.54**, Black **2.12**, mixed pairs **0.69**.
- **Among the pairs that really matched, the app surfaces 10.5% of the Asian ones against 20.1%.**

**Say:** the last line out loud, because it is a loss and not a ratio.

### Slide 14. Amplification, 45 s. Our contribution, sell it.
**On the slide:** `reports/figures/06_amplification.png`, full width.
- **41.0% of real matches are same-background, 47.5% of the shortlist. Ratio 1.16, CI [1.03, 1.29].**

**Say:** *"the engine did not invent a preference, it sharpened one."* Then the right-hand panel:
**XGBoost's interval crosses 1.00**, so its amplification is not distinguishable from none, where
TabPFN reaches 1.27. Fairness separates the engines where performance could not.

### Slide 15. The obvious fix does not work, 45 s
**On the slide:**

| variant | amplification | top decile |
|---|---|---|
| full model | 1.16 | 0.298 |
| without ethnicity | **1.35** | 0.251 |
| ethnicity neutralised at prediction time | **1.59** | 0.246 |
| full unawareness | **1.07** | 0.248 |

- With no ethnicity column at all, ethnicity is still recoverable at **AUC 0.66**.

**Say:** removing the column makes it worse, the model leans harder on income missingness and
lifestyle answers. The trade-off, handed over: **about 50 matches per 1,000 recommendations** for a
recommendation that no longer sharpens the preference. If there is room, one line on TOST: below 10
points of tolerance the three small groups cannot conclude, which is absence of power and not a clean
bill of health.

---

## Block G. Recommendation and app, speaker 1 again, 2 min

### Slide 16. The scorecard, 45 s
**On the slide:** one matrix. This is the slide the jury photographs.

| | logit | XGBoost | TabPFN |
|---|---|---|---|
| PR-AUC | 0.228 | 0.218 | **0.234** |
| Matches per 1,000 | **298** | 286 | 289 |
| Interpretability | 77 readable coefficients | SHAP, four methods disagree | permutation importance only |
| Instability | 0.56 | **0.30** | needs a GPU |
| Amplification | 1.16 | **1.01** | 1.27 |

**Say:** *"the three engines are separated by less than the width of our error bars on performance, so
we chose on the other three dimensions."*

### Slide 17. What we recommend, 60 s
**On the slide:** six lines, in this order.
1. **Deploy the two-stage lasso logistic regression**, and say why not TabPFN.
2. **Ship it anchored** to the previous version, λ ≈ 0.3. Costs nothing, gains.
3. **Fairness is a product decision and we price it:** ≈50 matches per 1,000.
4. **Collect interaction history:** ×1.81 on day one, **×2.89** once the app has seen a few decisions.
5. **Make some profile fields mandatory.** Income missingness is not random.
6. **Re-run the fairness checks with more users.**

**Say:** on point 1, the caveat before anyone asks. **XGBoost is the fairest of the three**,
amplification 1.01 against 1.16, and choosing it instead is defensible for about one point of
top-decile rate. We recommend the logit because an engine that decides who meets whom has to be
auditable.

### Slide 18. The app, 45 s
**On the slide:** two screenshots, or a live demo.
**Say:** *"the client compares the three engines and moves the shortlist length themselves."* Decide
live or screenshots by Saturday. If live: one rehearsed scenario, app already running, browser
already open, no typing.

---

## Backup slides, after the thank-you

`04_pdp.png` and `04_ice.png`, and why a PDP averages over combinations that never happen ·
`04_permutation_importance.png` with its error bars · `04_xper.png` and its additivity gap ·
`04_impurity_importance.png` · `06_calibration.png` · `05_stability_trade_off.png` · the full TOST
table · the eighteen engineered features that did not work.

---

## Q&A. The eight questions that will come

1. **"0.59 ROC-AUC is barely better than a coin flip."** It is the number the product operates at. A
   random split reads 0.72 and the difference is memorisation. What matters is the top decile, 29.8%
   against 16.5%.
2. **"Why top decile and not a 0.5 threshold?"** No pair scores above 0.5 at a 16.5% base rate with a
   product of two probabilities. Checked at 5% and 20%.
3. **"Isn't multiplying two decisions a hack?"** It is the generative structure of the outcome. Twice
   the rows, target at 37% and 47%, six points of top-decile rate.
4. **"Just drop ethnicity."** Measured: amplification 1.16 to 1.35, and five points of match rate
   lost. Ethnicity is recoverable at AUC 0.66 from what remains.
5. **"Where is your cost matrix?"** At a fixed top-K, TP + FP = K, so minimising cost is maximising
   the top-K match rate, which is what we optimised. Costs matter for choosing K, not the model.
6. **"Your groups are tiny, 420 rows for Black participants."** Which is why every table carries the
   group size and why we ran TOST. Absence of power, not absence of a gap.
7. **"Why TabPFN if you do not ship it?"** Best PR-AUC, no native explanation, and it amplifies the
   most at 1.27. That combination is itself the argument.
8. **"Who did what?"** Every member must be able to defend any section.

---

## Before Monday

- [ ] Decide live demo or screenshots for slide 18.
- [ ] Two full rehearsals with a timer. Target 13:30.
- [ ] Send slides, notebooks and app before Monday 09:40.
