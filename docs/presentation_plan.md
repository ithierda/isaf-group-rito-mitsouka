# Presentation plan. 15 minutes, 6 speakers

Jury: Prof. Pérignon, Monday 28/09. 15 min talk + 10 min Q&A.
Graded: technical skills /5, presentation /5, Q&A /5, slides & code/app /10.

**18 slides.** Six blocks, one speaker per block, so nobody has to hand over mid-argument.
Everything on the slides is in English. Every number below is in `reports/tables/`.
do not retype from memory, copy from the CSV.

| Block | Slides | Speaker | Time |
|---|---|---|---|
| A. Client, data | 1 to 3 | 1 | 2:00 |
| B. Two choices that decide everything | 4 to 5 | 2 | 2:00 |
| C. Performance | 6 to 7 | 3 | 2:00 |
| D. Interpretability | 8 to 10 | 4 | 2:30 |
| E. Stability | 11 to 12 | 5 | 2:00 |
| F. Fairness | 13 to 15 | 6 | 2:30 |
| G. Recommendation + app | 16 to 18 | 1 (bookend) | 2:00 |

Total 15:00. Rehearse at 13:30, because a jury talk always runs long.

---

## Block A. the client and the data (speaker 1, 2 min)

### Slide 1. Title (10 s)
- Logo `assets/logo/horizontal_blue.png`.
- *Les Rito Mitsouka: which pairs should the app recommend?*
- Six names, course, date.
- Say while it is up: "Our client is a dating app. It has to decide, out of every possible
  pair of users, which ones to put in front of each other."

### Slide 2. The client's question (50 s)
- One line, large: **Y = 1 if both people say yes. 16.5% of pairs.**
- The product: the app shows a **short list**, not a yes/no on every pair.
  → this is the sentence that justifies every methodological choice later. Say it clearly.
- The three models the brief asks for, as three logos/boxes:
  lasso logistic regression (white box) · XGBoost (ML) · TabPFN (tabular foundation model).
- The four dimensions as four icons: performance · interpretability · stability · fairness.
- Bottom line: **the protected attribute is ethnicity.**

### Slide 3. The data (60 s)
- Source: Fisman, Iyengar, Kamenica & Simonson (2006), QJE, the Speed Dating Experiment.
- Numbers: **8,378 rows → 4,184 pairs · 551 participants · 21 waves · 16.5% match.**
- Three cleaning decisions, one line each. These are what a jury rewards:
  1. **Each date appears twice**, once from each side → one row per pair, `her_*` / `his_*`.
  2. **108 of the 195 columns are filled in during or after the date** → leakage, removed.
  3. **`income` is missing for 62% of Latino and 60% of Asian participants against 31% of
     Black ones** → the missingness itself carries information, so it gets its own flag.
     Flag this one, it comes back in the fairness block and in the recommendation.
- Do **not** put an EDA gallery here. One slide, three decisions, move on.

---

## Block B. the two choices that decide everything (speaker 2, 2 min)

This block is the strongest technical content in the deck. Do not rush it.

### Slide 4. We split on the wave, not at random (60 s)
- The table, from `02_features.ipynb`:

  | split | ROC-AUC | PR-AUC |
  |---|---|---|
  | random | 0.715 | 0.364 |
  | by wave (StratifiedGroupKFold) | 0.594 | 0.216 |

- One sentence: **"the 12 points of AUC are the model recognising people it has already seen."**
  Every participant attends exactly one evening, and each attends ~18 dates. A random split puts
  the same person on both sides.
- Second line: **no feature that is constant within a wave.** Such a column is the wave under
  another name, and a tree uses it to memorise an evening's match rate.
- Consequence to say out loud: **all our numbers look worse than what you will see in most
  projects, and they are the honest ones.**

### Slide 5. A match is two decisions, not one event (60 s)
- Diagram: pair → (she decides | he decides) → p̂(her yes) × p̂(his yes).
- Why: 16.5% joint event vs **37% / 47%** on each side. 8,368 decision rows instead of 4,184 pairs.
- The payoff, `reports/tables/04_economic_performance.csv`:
  **top decile match rate 29.8% two-stage vs 23.6% predicting the match directly.**
- The line the jury will like: **ROC-AUC ranks the models in almost the reverse order of the
  metric the client cares about.** Direct XGBoost has the best ROC-AUC (0.589) and the worst
  top decile (0.230).
- `dec` is a target, never a feature. Say it before they ask.

---

## Block C. performance (speaker 3, 2 min)

### Slide 6. Statistical performance (60 s)
- Table, from `03_models`:

  | model | ROC-AUC | PR-AUC | top 10% | 95% CI |
  |---|---|---|---|---|
  | logit (two-stage) | 0.586 | 0.227 | 0.294 | [0.248, 0.339] |
  | XGBoost (two-stage) | 0.585 | 0.218 | 0.284 | [0.241, 0.327] |
  | TabPFN | *fill from 03b* | | | |
  | logit direct | 0.573 | 0.203 | 0.232 | [0.193, 0.274] |
  | XGBoost direct | 0.589 | 0.212 | 0.229 | [0.191, 0.269] |

- **Put the error bars on the chart, not just in the table.** One horizontal bar chart of the
  top-decile rate with the 95% interval; the base rate 16.5% as a dashed vertical line.
- The conclusion, said in one sentence and repeated at the end:
  **"The interval is 8 points wide. The two-stage gain is real. The gap between our three
  models is not. Performance cannot choose the model for us. The other three dimensions will."**
- Add, because it is a point in your favour: we tried 18 more engineered features, attribute-level
  preference fits, directional `raceclash`, income gaps, and every variant landed inside the same
  interval. We stopped rather than tune on noise.

### Slide 7. Economic performance (60 s)
- The baseline the client has on day one is showing pairs at random. Everything on this slide is
  the difference against it, per 1,000 recommendations.
- The headline, one line, large: **+133 matches per 1,000 recommendations**, 298 against 165.
- The table, from `04_economic_performance.csv`, three rows:

  | List shown | Matches /1,000 | Extra /1,000 | Extra matches over the catalogue |
  |---|---|---|---|
  | top 5% | 314 | +149 | +31 |
  | top 10% | 298 | +133 | +56 |
  | top 20% | 235 | +70 | +59 |

- **Value per 1,000 recommendations = 133 × v**, v = net value of one additional match. One number
  for the client to supply, out of their own business plan. Say it in exactly one sentence.
- The decision those numbers drive is **how long the list should be**, and the last two columns pull
  in opposite directions: a short list is more efficient per slot, a long list delivers more matches
  in total. Where to sit depends on what a wasted recommendation costs in user fatigue, a product
  decision, and a good thing to hand back to the jury.
- One caveat, said in a breath: more matches → more use → more data → better matches. Nobody in this
  dataset attends twice, so that loop is a reason to invest, not a figure we quote.
- **Unit economics, from `04_unit_economics.csv`**, two lines that land hard:
  - **Cost:** scoring a pair takes 2.8 µs, so the engine runs for **$0.04 per 1,000 users per year**.
    It cannot be priced on cost; it is priced on value. The real expense is data collection and the
    fairness monitoring, in person-days.
  - **Value:** at $15/month, 12% conversion (market band 8 to 15%: Tinder 8.6M payers / ~60M users,
    Grindr 8.4%) and one extra paid month per subscriber, **$1,800 per 1,000 users per year**,
    about 40,000× the cost. `c` (conversion) and `r` (extra paid months) are the client's numbers,
    and saying so is stronger than inventing them.
- Say the transferable quantity out loud: **×1.81 at equal recommendation volume.** The absolute
  16.5% base rate does not transfer, because these people had just met face to face. This is also the
  number the app shows the user in-product: *"recommended profiles are 1.8× more likely to match"*.
  Add the honest qualifier in one breath: it is an offline cold-start estimate, and the client has
  to confirm it with an A/B test before advertising it.
- **Bridge into fairness:** a lift can be bought with exclusion. A model that quietly stops
  recommending one group also raises the top-decile rate. Slides 13 to 15 check exactly that.

---

## Block D. interpretability (speaker 4, 2:30)

### Slide 8. The white box, coefficient by coefficient (60 s)
- The lasso keeps **46 features out of 99** for three thousandths of AUC. Half as many
  coefficients in front of the client is worth more than that.
- Method in one line: lasso **selects**, an unpenalised logit is refitted on the selected columns
  to get standard errors. Those p-values ignore the selection step, so read them as an ordering.
- The four effects to show (marginal effect on the probability of a yes,
  `04_white_box_coefficients.csv`):

  | variable | effect | scale |
  |---|---|---|
  | `female` | **−13.7 pts** | 0 → 1 |
  | `other_selfintel` | −4.0 pts | +1 sd |
  | `self_optimism` | +3.5 pts | +1 sd |
  | `raceclash` | −3.4 pts | +1 sd |

- The honest sentence, and it is a strength not a weakness:
  **"A model whose strongest profile variable moves the answer by four points is a weak model.
  The engine works by combining many small effects, not by finding a rule."**
- `raceclash` = someone cares about background **and** the pair is mixed. Name it here, it is the
  hinge into slide 13.

### Slide 9. One prediction, explained (45 s)
- `reports/figures/04_shap_waterfall.png`, one pair.
- The technical point to make in one line: **the two stages add up in logs, not in probabilities.**
  XGBoost's raw output is a margin and Σφ + φ₀ = margin holds exactly; the pair score is a product,
  so we explain each stage on its own scale and read them side by side.
- Product framing: what the app could actually show a user: *"why would she say yes to him"*.

### Slide 10. The four methods disagree (45 s)
- Four small panels or one comparison table: impurity · SHAP · permutation · XPER.
- The number: **the four methods share 30% of their top five. Impurity and XPER share nothing.**
- Why: impurity ranks `self_field` / `other_field` first because a column with eighteen categories
  offers more places to split. Permutation and XPER, which both measure *performance*, put
  `female`, `self_optimism` and the **ethnicity columns** on top.
- The punchline: **"this is the argument for not putting a single importance bar chart on a slide.
  If one has to be chosen it is XPER, because the client asks where the skill comes from,
  not where the tree looked."**
- XPER caveat, on the slide in small type: run on the 8 top columns in approximate mode
  (2^101 coalitions is out of reach), so the decomposition describes that reduced model.

---

## Block E. stability (speaker 5, 2 min)

### Slide 11. Is it the same model twice? (60 s)
- Definition first, one line: **Turney: two datasets from the same population should produce
  approximately the same model.** Stability is a distance between models, not the spread of a score.
- Setup: **25 versions of each model**, 5 CV folds plus 20 bootstrap resamples drawn within waves.
- `reports/figures/05_distance_heatmaps.png` + the two numbers from `05_instability.csv`:

  | model | instability (mean distance / size) |
  |---|---|
  | logit | **0.56** |
  | XGBoost | **0.30** |

- Then the twist, from `05_ranking_stability.csv`. **The user does not see coefficients, they see
  a list**:

  | model | Jaccard, top decile | Kendall τ |
  |---|---|---|
  | logit | **0.48** | **0.66** |
  | XGBoost | 0.44 | 0.56 |

- The sentence: **"XGBoost is the more stable object; the logit produces the more stable list.
  The list is what the client ships."** Retrain the engine and roughly half the shortlist changes,
  for both. That is a finding, not a defect to hide.

### Slide 12. Over time, and the price of stability (60 s)
- Leave-one-wave-out, `05_leave_one_wave_out.png`: **PR-AUC from 0.10 to 0.46 depending on the
  evening held out.** Say: *"a model retrained next month lands somewhere in that range, and the
  client should be given the range, not the average."*
- Caveat that shows you looked: the distance column is almost entirely a sample-size effect
  (correlates 0.96 with the number of pairs removed). Wave 21 moves the model more than wave 6
  because it is 484 pairs against 25.
- Buying stability, `05_stability_trade_off.csv`, the counter-intuitive result, worth 30 seconds:
  pulling the fold-1 model **84% of the way** towards the fold-0 model **raises** PR-AUC from
  **0.287 to 0.312**. **On this data there is no trade-off to arbitrate.** With 55 live coefficients
  and 3,300 pairs per fold, anchoring is simply extra regularisation.
  → This feeds directly into the recommendation on slide 16.

---

## Block F. fairness (speaker 6, 2:30)

Frame it in the first ten seconds or the jury will argue the wrong question:
**"Not whether the participants preferred their own background. They did, and that is descriptive.
Whether the engine disadvantages a group, and whether it amplifies that preference."**

### Slide 13. The gaps, and whether they survive (60 s)
- Mapping on the slide, small: Y = matched · **Ŷ = in the top decile** (no pair ever scores above
  0.5 at a 16.5% base rate, so a threshold classifier is degenerate) · D = the protected attribute.
- Raw statistical parity: **Asian participants appear in 5.8% of shown pairs against 11.3% for
  everyone else. Black participants 19.5%. Mixed pairs −3.4 points.**
- Conditional parity (Cochran, Mantel and Haenszel), conditioning on four variables we commit to as
  legitimate (`other_selfattr`, `other_selfsinc`, `interests`, `agegap`), and Benjamini-Hochberg
  over the whole family of 12 tests.
- The verdict table from `06_parity_verdicts.csv`, colour-coded:
  **4 of the 6 gaps survive.** Asian odds ratio **0.54**, Black **2.12**, mixed pairs **0.69**.
  Only White is green, which is what a majority group usually does.
- The cost version, `06_equal_opportunity.csv`, and it is the one to say out loud because it is a
  loss and not a ratio: **among the pairs that really did match, the app surfaces 10.5% of the
  Asian ones against 20.1% for everyone else.**

### Slide 14. Amplification (45 s), your original contribution, so sell it
- Definition: share of same-background pairs among the pairs the app **shows**, divided by their
  share among the pairs that actually **matched**. Above 1 = the recommendation is more segregated
  than reality.
- **41% of real matches are same-background. 48% of shown pairs are. Ratio 1.16, CI [1.03, 1.30].**
- One sentence: **"the engine did not invent a preference. It sharpened one."**
- Honest limit: the effect grows with the length of the list and disappears into the noise at the
  top 5%, where there are only 209 pairs.
- You will need to build this chart, there is no PNG for it yet. A single bar pair (41% vs 48%)
  with the bootstrap interval is enough.

### Slide 15. The obvious fix does not work (45 s)
- `06_mitigation.csv`, four variants, one line each:

  | variant | amplification | top decile |
  |---|---|---|
  | full model | 1.16 | 0.298 |
  | without ethnicity | **1.35** | 0.251 |
  | ethnicity neutralised at prediction time | **1.57** | 0.248 |
  | full unawareness (ethnicity + `raceclash` + `samerace` + both preference columns) | **1.07** | 0.248 |

- **"Removing the ethnicity column makes the segregation worse."** Deprived of it, the model leans
  harder on whatever correlates with it, and pays 5 points of match rate for the privilege.
- The proof that unawareness is no defence, `06_proxy_recovery.csv`: from a table with **no**
  ethnicity column, no `samerace`, no `raceclash`, no background preference, ethnicity is still
  recoverable at **AUC 0.66** (White and Asian). Income missingness, going-out frequency and
  self-rated attractiveness carry it.
- The trade-off, stated plainly and handed to the client: **≈5 points of top-decile match rate in
  exchange for a recommendation that no longer sharpens the same-background preference.**
- If you have room, one line on TOST (`06_equivalence.csv`): only White is equivalent at δ = 3 pts,
  White and mixed pairs at δ = 5. For the three small groups the test cannot conclude at any
  tolerance below 10 points. *"No significant difference" there is an absence of power, not a
  clean bill of health.*

---

## Block G. recommendation (speaker 1 again, 2 min)

### Slide 16. The scorecard (45 s)
One matrix, 3 models × 4 dimensions, ticks and crosses. This is the slide the jury photographs.

| | logit (two-stage) | XGBoost | TabPFN |
|---|---|---|---|
| **Statistical perf.** | 0.227 PR-AUC | 0.218 | *fill* |
| **Economic perf.** | 298 / 1,000 (×1.81) | 286 / 1,000 | *fill* |
| **Interpretability** | 46 readable coefficients | SHAP only, 4 methods disagree | opaque, no native explanation |
| **Stability (model)** | 0.56 | **0.30** | *fill* |
| **Stability (the list)** | **0.48 Jaccard** | 0.44 | *fill* |
| **Fairness** | 4/6 gaps survive, amplification 1.16 | same direction | *fill* |

The sentence that ties the deck together: **"The three models are separated by less than the width
of our error bars on performance. So we chose on the other three dimensions."**

### Slide 17. What we recommend to Les Rito Mitsouka (60 s)
Three recommendations, one line of justification each. Keep them in this order.

1. **Deploy the two-stage lasso logistic regression.** Not because it scores best, it does not,
   measurably. Because at equal performance it is the only one of the three we can hand the client
   coefficient by coefficient, and it produces the more stable shortlist.
2. **Ship it with the anchoring penalty (λ ≈ 0.3).** Stability normally costs performance; here it
   pays for itself: PR-AUC 0.287 to 0.312 and 84% less drift between retrainings. There is no
   arbitration to make, so make it.
3. **Fairness is a product decision, not a modelling one.** We can remove the same-background
   amplification (1.16 → 1.07), and it costs ≈5 points of top-decile match rate. Removing only the
   ethnicity column makes things *worse*. The client chooses; we price the choice.

Then two things the models cannot do for themselves, and these are worth saying because they are
advice rather than output:

4. **Make some profile fields mandatory**, with the message *"the more information you give, the
   more accurate your recommendations"*. Income is missing for half the participants and **not at
   random** (62% of Latino and 60% of Asian against 31% of Black ones), so incomplete profiles
   both weaken the model *and* skew it across groups. Slide 3 set this up; close the loop here.
5. **Collect and use interaction history.** Our own split experiment measures what identity is
   worth: 0.715 ROC-AUC when the model has already seen a person against 0.594 when it has not.
   We had to throw that 12-point gap away because nobody in this dataset attends twice, but a real
   app *does* see its users again. It is the single strongest lever the product has.

### Slide 18. The app *(LEAVE EMPTY FOR NOW)*
Placeholder while the Streamlit app is built. Reserve **45 s** for it in the rehearsal.
What it should end up containing:
- 2 screenshots: pick a pair → score + the top SHAP contributions, and a model switcher.
- One line: *"the client can test the three models on any pair and see why a pair is recommended."*
- Decide by Saturday whether this is a live demo or screenshots. If live: one rehearsed scenario,
  app already running, browser already open, no typing.

### Backup slides (after the thank-you, not counted in the 15 min)
Build these, they are where the 10 minutes of Q&A get won:
- PDP + ICE (`04_pdp.png`, `04_ice.png`) and why PDP averages over combinations that never happen.
- Permutation importance with error bars (`04_permutation_importance.png`).
- XPER decomposition (`04_xper.png`) and the additivity gap.
- Depth-3 surrogate tree (`04_surrogate_tree.png`) and its fidelity.
- Calibration by group (`06_calibration.png`) and the proxy map (`06_proxies.png`).
- Full TOST table (`06_equivalence.csv`).
- The `bootstrap_ci` function: what is resampled and why within waves.
- The list of 18 engineered features that did not work.

---

## Q&A. the seven questions that will come, and the answer

1. **"Your AUC is 0.59, that is barely better than a coin flip."**
   → It is honest, and it is the number the product operates at. On a random split the same model
   reads 0.715, and the difference is memorisation. What matters commercially is the top decile:
   29.8% against a 16.5% base rate, a lift of 1.81.
2. **"Why top decile and not a 0.5 threshold?"**
   → At a 16.5% base rate with a product of two probabilities, no pair scores above 0.5. A threshold
   classifier predicts zero everywhere and every confusion-matrix metric is degenerate. The app
   shows a shortlist, so Ŷ = 1 means "in the list". Sensitivity checked at 5% and 20%.
3. **"Isn't predicting two decisions and multiplying them just a hack?"**
   → It is the generative structure of the outcome: a match *is* she says yes and he says yes.
   It doubles the rows, moves the target from 16.5% to 37/47%, and gains 7 points of top-decile
   match rate. It costs nothing on ROC-AUC.
4. **"Should you not just drop ethnicity? It is illegal to use it."**
   → We measured it. Dropping the column raises amplification from 1.16 to 1.35 and costs 5 points
   of match rate. Ethnicity is recoverable at AUC 0.66 from the remaining columns. Unawareness is
   not a defence: you need the full feature set removed, and that is the 1.07 variant.
5. **"Why is TabPFN not everywhere in the deck?"**
   → Answer with whatever is true on Sunday night. If it only has performance numbers: it runs on
   the same five folds with the same two-stage target, so it is comparable on performance; it has
   no native explanation, which is itself part of the recommendation.
6. **"Your groups are tiny, 420 rows for Black participants."**
   → Which is why every table carries the group size, and why we ran TOST: for the three small
   groups we cannot certify equality at any tolerance below 10 points. We report that as absence
   of power, not as absence of a gap.
7. **"Where is your cost matrix?"**
   → At a fixed top-K the list length is constant, so TP + FP = K and the cost collapses to
   `const − (c_FP + c_FN)·TP`: every cost matrix ranks the models identically, and minimising cost
   is maximising the top-K match rate, which is what we optimised. The costs matter for choosing K,
   not for choosing the model, which is why we give the client the curve at 5/10/20% instead of a
   number we invented.
8. **"Who did what?"**
   → Have the answer ready. Every member must be able to defend any section: the brief says so
   explicitly and grades can vary across team members.

## Before Monday
- [ ] Run 03b on Colab, merge `tabpfn_oof.csv`, fill every *fill* cell above.
- [ ] Build the amplification chart (41% vs 48% + CI), no PNG exists yet.
- [ ] Slide 16: the price of fairness, full model 298 matches /1,000 against 248 for the mitigated
      one, so removing the amplification costs ≈50 matches per 1,000 recommendations. Number printed
      at the end of section 4 of `06_fairness.ipynb`; convert to dollars with the slide 7 table.
- [ ] Build the top-decile chart with error bars for slide 6.
- [ ] Streamlit app + slide 18.
- [ ] Two full rehearsals with a timer. Target 13:30.
- [ ] Send slides + notebooks + app by email before Monday 09:40.
