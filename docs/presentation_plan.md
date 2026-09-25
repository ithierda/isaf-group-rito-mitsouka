# Presentation plan. 15 minutes, 6 speakers

Jury: Prof. Pérignon, Monday 28/09. 15 min talk, 10 min Q&A.
Graded: technical /5, presentation /5, Q&A /5, slides, code and app /10.

**18 slides, six blocks, one speaker per block**, so nobody hands over mid-argument.
Everything on the slides is in English. **Every number is in `reports/tables/`. Copy, do not retype.**

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

## Block A. Client and data (speaker 1, 2 min)

**Slide 1, title, 10 s.** Logo, six names, course, date. Say: *our client is a dating app, it has to
decide which pairs of users to put in front of each other.*

**Slide 2, the client's question, 50 s.**
- Large: **Y = 1 if both people say yes. 16.5% of pairs.**
- The product shows a **shortlist**, not a yes or no on every pair. This sentence justifies every
  methodological choice later, so say it clearly.
- Three engines: lasso logistic regression, XGBoost, TabPFN. Four dimensions: performance,
  interpretability, stability, fairness. Protected attribute: **ethnicity**.

**Slide 3, the data, 60 s.** Fisman, Iyengar, Kamenica and Simonson (2006), QJE.
**8,378 rows to 4,184 pairs, 551 participants, 21 waves, 16.5% match.** Three cleaning decisions:
1. each date appears **twice**, so one row per pair;
2. **108 of 195 columns** are filled in during or after the date, so they leak;
3. **income is missing for 62% of Latino and 60% of Asian participants against 31% of Black ones**,
   so the missingness itself carries information. Flag it, it returns in blocks F and G.

No EDA gallery. One slide, three decisions, move on.

---

## Block B. The two choices that decide everything (speaker 2, 2 min)

The strongest technical content in the deck. Do not rush it.

**Slide 4, we split on the wave, 60 s.**
- Everyone meets everyone of the opposite sex in their evening, so the "dated each other" graph has
  **21 connected components for 21 waves**. There is no finer cut that keeps a person on one side.
- A random split scores **0.72 ROC-AUC against 0.60**. The gap is the model recognising people.
- Kill the obvious objection before it is raised: **the wave is a grouping key, never a feature.**
  02 removes it and everything constant within a wave, `pool_size` included.

**Slide 5, a match is two decisions, 60 s.**
- A match is she says yes **and** he says yes. One model on **8,368 decisions** instead of 4,184
  pairs, target at 37% and 47% instead of 16.5%, the two probabilities multiplied.
- It costs nothing on ROC-AUC and gains where the app operates:
  **top decile 29.8% against 23.6%** predicting the match in one step.

---

## Block C. Performance (speaker 3, 2 min)

**Slide 6, statistical performance, 60 s.** From `07_scorecard.csv`.

| model | PR-AUC | top decile | lift |
|---|---|---|---|
| logit, two-stage | 0.228 | **0.298** | 1.81 |
| XGBoost, two-stage | 0.218 | 0.286 | 1.74 |
| TabPFN | **0.234** | 0.289 | 1.75 |

- Put the **error bars on the chart**, not only in the table. Base rate 16.5% as a dashed line.
- The sentence, repeated at the end: **"the interval is eight points wide. The two-stage gain is
  real. The gap between the three engines is not. Performance cannot choose for us."**
- A point in your favour: eighteen further engineered features all landed inside the same interval.
  We stopped rather than tune on noise.

**Slide 7, economic performance, 60 s.** From `04_economic_performance.csv`.

| list shown | matches /1,000 | extra /1,000 | extra matches over the catalogue |
|---|---|---|---|
| top 5% | 314 | +149 | +31 |
| top 10% | 298 | +133 | +56 |
| top 20% | 235 | +70 | +59 |

- Headline: **+133 matches per 1,000 recommendations**, 298 against 165 at random.
- **Value per 1,000 = 133 × v**, v the net value of one match. One number for the client to supply.
- The last two columns pull opposite ways, so the **length of the list** is the client's decision.
- Cost, from `04_unit_economics.csv`: scoring a pair takes microseconds, the engine runs for **a few
  cents per 1,000 users per year**. At $15 a month and 12% conversion it is worth **$1,800**. It
  cannot be priced on cost.
- The transferable number is the **lift, ×1.81**, not the absolute rate: these people had just met
  face to face. It is also what the app tells the user. Add in one breath that it is an offline
  cold-start estimate the client must confirm with an A/B test.
- **Bridge into fairness:** a lift can be bought with exclusion. Slides 13 to 15 check that.

---

## Block D. Interpretability (speaker 4, 2:30)

**Slide 8, the white box, 60 s.** From `04_white_box_coefficients.csv`.
- The lasso keeps **77 of 101 columns**. Method in one line: the lasso selects, an unpenalised logit
  is refitted to get standard errors, so read the p-values as an ordering.

| variable | effect | scale |
|---|---|---|
| `female` | **−13.7 pts** | 0 to 1 |
| `other_selfintel` | −4.0 pts | +1 sd |
| `self_optimism` | +3.5 pts | +1 sd |
| `raceclash` | −3.4 pts | +1 sd |

- The honest sentence, and it is a strength: **"a model whose strongest profile variable moves the
  answer by four points works by combining many small effects, not by finding a rule."**
- Name `raceclash` here: someone cares about background **and** the pair is mixed. It is the hinge
  into slide 13.

**Slide 9, one prediction explained, 45 s.** `04_shap_waterfall.png`, then LIME on the same pair.
- One technical line: **the two stages add up in logs, not in probabilities.** Σφ + φ₀ = margin holds
  per decision; the pair score is a product.
- SHAP allocates the prediction exactly, LIME returns readable rules. Show both, say they agree.
- Product framing: *"why would she say yes to him"*, which is what the app could show.

**Slide 10, the four methods disagree, 45 s.**
- **The four share 30% of their top five. Impurity and XPER share nothing.**
- Why: impurity ranks the field columns first because eighteen categories offer more places to
  split. Permutation and XPER, both performance measures, put `female`, `self_optimism` and the
  **ethnicity columns** on top.
- Punchline: **"this is the argument for not putting a single importance chart on a slide. If one
  has to be chosen it is XPER, because the client asks where the skill comes from."**
- Small type: XPER runs on the 8 strongest columns in approximate mode.
- TabPFN has no coefficients and no trees, so permutation importance by group is the only way in,
  and it puts **`race` first**. `04_tabpfn_importance.png`.

---

## Block E. Stability (speaker 5, 2 min)

**Slide 11, is it the same model twice, 60 s.**
- Definition first: **Turney, two datasets from the same population should produce approximately the
  same model.** Stability is a distance between models, not the spread of a score.
- **25 versions of each engine**: 5 folds plus 20 bootstraps drawn within waves.
  `05_distance_heatmaps.png` and `05_instability.csv`:

| model | instability |
|---|---|
| logit | **0.56** |
| XGBoost | **0.30** |

- Then the twist, our one addition here, `05_shortlist_overlap.csv`: **the user sees a list, not
  coefficients.** Retrain and about **half the top decile changes**, 48% kept for the logit, 44% for
  XGBoost. A finding, not a defect to hide.
- Only two engines: comparing TabPFN refits needs a GPU.

**Slide 12, over time and the price of stability, 60 s.**
- Leave one wave out, `05_leave_one_wave_out.png`: **PR-AUC from 0.10 to 0.46** depending on the
  evening held out. *"Give the client the range, not the average."*
- A caveat that shows you looked: the drift column correlates 0.96 with the number of pairs removed,
  so it is a sample-size effect.
- The counter-intuitive result, worth 30 seconds: pulling the fold-1 model **84% towards** fold 0
  **raises** PR-AUC from **0.287 to 0.312**. **No trade-off to arbitrate.** Feeds slide 17.

---

## Block F. Fairness (speaker 6, 2:30)

Frame it in the first ten seconds or the jury argues the wrong question: **"not whether the
participants preferred their own background, they did and that is descriptive, but whether the engine
disadvantages a group and whether it amplifies that preference."**

**Slide 13, the gaps and whether they survive, 60 s.**
- Mapping, small: Y = matched, **Ŷ = in the top decile** (no pair scores above 0.5), D = protected.
- Raw parity: **Asian participants in 5.8% of shown pairs against 11.3% for everyone else. Black
  19.5%. Mixed pairs −3.4 points.**
- Conditional parity, Cochran Mantel Haenszel on four legitimate variables (`other_selfattr`,
  `other_selfsinc`, `interests`, `agegap`). Verdict table, colour coded: **4 of 6 gaps survive.**
  Asian odds ratio **0.54**, Black **2.12**, mixed pairs **0.69**. Only White is green.
- Say this one out loud, because it is a loss and not a ratio: **among the pairs that really
  matched, the app surfaces 10.5% of the Asian ones against 20.1% for everyone else.**

**Slide 14, amplification, 45 s. Our contribution, so sell it.**
- Share of same-background pairs among the pairs **shown**, divided by their share among the pairs
  that **matched**. Above 1, the recommendation is more segregated than reality.
- **41% of real matches are same-background, 48% of shown pairs. Ratio 1.16, CI [1.03, 1.30].**
- One sentence: **"the engine did not invent a preference, it sharpened one."**
- And the result nobody expects, `06_by_model.csv`: the three engines differ.
  **XGBoost 1.01, logit 1.16, TabPFN 1.27.** Fairness separates them where performance could not.
- You still need this chart. A bar pair, 41% against 48%, with the interval.

**Slide 15, the obvious fix does not work, 45 s.** From `06_mitigation.csv`.

| variant | amplification | top decile |
|---|---|---|
| full model | 1.16 | 0.298 |
| without ethnicity | **1.35** | 0.251 |
| ethnicity neutralised at prediction time | **1.57** | 0.248 |
| full unawareness | **1.07** | 0.248 |

- **"Removing the ethnicity column makes the segregation worse."** The model leans harder on whatever
  correlates with it and pays five points of match rate.
- The proof, `06_proxy_recovery.csv`: with no ethnicity column, no `samerace`, no `raceclash` and no
  background preference, ethnicity is still recoverable at **AUC 0.66**. Income missingness,
  going-out habits and self-rated looks carry it.
- The trade-off, handed to the client: **about 50 matches per 1,000 recommendations** for a
  recommendation that no longer sharpens the preference.
- If there is room, one line on TOST: only White is equivalent at δ = 3, White and mixed pairs at
  δ = 5, and below 10 points the three small groups cannot conclude. *Absence of power, not a clean
  bill of health.*

---

## Block G. Recommendation and app (speaker 1 again, 2 min)

**Slide 16, the scorecard, 45 s.** From `07_scorecard.csv`. Three engines, four dimensions. This is
the slide the jury photographs.

| | logit | XGBoost | TabPFN |
|---|---|---|---|
| PR-AUC | 0.228 | 0.218 | **0.234** |
| Matches per 1,000 | **298** | 286 | 289 |
| Interpretability | 77 readable coefficients | SHAP, 4 methods disagree | permutation importance only |
| Instability | 0.56 | **0.30** | needs a GPU |
| Amplification | 1.16 | **1.01** | 1.27 |

The sentence that ties the deck together: **"the three engines are separated by less than the width
of our error bars on performance, so we chose on the other three dimensions."**

**Slide 17, what we recommend, 60 s.** From `07_recommendation.ipynb`.

1. **Deploy the two-stage lasso logistic regression.** Not because it scores best, TabPFN does.
   Because at equal performance it is the only one we can hand over coefficient by coefficient.
   **Say the caveat before anyone asks: XGBoost is the fairest of the three**, amplification 1.01
   against 1.16, and choosing it instead is defensible for about one point of top-decile rate.
2. **Ship it anchored** to the previous version, λ ≈ 0.3. It costs nothing and gains: PR-AUC 0.287
   to 0.312, and 84% less drift between retrainings.
3. **Fairness is a product decision, and we price it.** Removing the amplification costs about
   **50 matches per 1,000 recommendations**. Removing only the ethnicity column makes it worse.
4. **Collect interaction history.** ×1.81 on the day a user signs up, **×2.89** once the app has seen
   a handful of their decisions, and ×2.06 after only two. The single strongest lever the product has.
5. **Make some profile fields mandatory**, with the message *"the more you give, the better your
   recommendations"*. Income missingness is not random and 06 shows the model uses it to reconstruct
   ethnicity. Closes the loop opened on slide 3.
6. **Re-run the fairness checks when there are more users.** Below 10 points of tolerance the three
   small groups cannot conclude. This belongs in the retraining routine.

**Slide 18, the app, 45 s.** Two screenshots or a live demo, decided by Saturday. If live: one
rehearsed scenario, app already running, browser already open, no typing. One line: *"the client can
compare the three engines and move the shortlist length themselves."*

---

## Backup slides, after the thank-you

- PDP and ICE, and why a PDP averages over combinations that never happen.
- Permutation importance with error bars, and XPER with its additivity gap.
- The depth-3 surrogate tree and its fidelity of 0.24.
- Calibration by group, and the full TOST table.
- The eighteen engineered features that did not work.

---

## Q&A. The eight questions that will come

1. **"0.59 ROC-AUC is barely better than a coin flip."** It is the number the product operates at. On
   a random split the same model reads 0.72, and the difference is memorisation. What matters
   commercially is the top decile: 29.8% against a 16.5% base rate.
2. **"Why top decile and not a 0.5 threshold?"** No pair scores above 0.5 at a 16.5% base rate with a
   product of two probabilities, so a threshold classifier predicts zero everywhere. The app shows a
   shortlist. Checked at 5% and 20%.
3. **"Isn't multiplying two decisions a hack?"** It is the generative structure of the outcome. It
   doubles the rows, moves the target to 37% and 47%, and gains six points of top-decile rate.
4. **"Just drop ethnicity."** We measured it: amplification rises from 1.16 to 1.35 and it costs five
   points of match rate. Ethnicity is recoverable at AUC 0.66 from the remaining columns.
5. **"Where is your cost matrix?"** At a fixed top-K the list length is constant, so TP + FP = K and
   minimising cost is maximising the top-K match rate, which is what we optimised. Costs matter for
   choosing K, not the model, so we give the client the curve.
6. **"Your groups are tiny, 420 rows for Black participants."** Which is why every table carries the
   group size and why we ran TOST. We report absence of power, not absence of a gap.
7. **"Why TabPFN if you do not ship it?"** It has the best PR-AUC and no native explanation, which is
   itself the argument. It also amplifies the most, 1.27.
8. **"Who did what?"** Have the answer ready. Every member must be able to defend any section.

---

## Before Monday

- [ ] Build the amplification chart, 41% against 48% with the interval. No PNG exists yet.
- [ ] Build the top-decile chart with error bars for slide 6.
- [ ] Decide live demo or screenshots for slide 18.
- [ ] Two full rehearsals with a timer. Target 13:30.
- [ ] Send slides, notebooks and app before Monday 09:40.
