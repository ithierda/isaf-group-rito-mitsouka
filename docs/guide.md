# The guide

Everything, in plain words, with no jargon that is not defined here. Every number is read from
`reports/tables/`.

---

## 1. The project

A dating app cannot show everyone to everyone, so it has to pick. We build the thing that picks, and
check it four ways: does it work, can we explain it, does it stay the same when rebuilt, is it fair.

**The data.** 551 students, 21 speed dating evenings at Columbia, 2002 to 2004. Everyone met everyone
of the opposite sex for four minutes and ticked yes or no. Both yes is a **match**, and 16.5% of
dates ended that way. We use only what people wrote about themselves **before** the evening, because
that is all an app has. 108 of the 195 raw columns were filled in during or after the date, so they
go.

---

## 2. The two decisions that shape everything

**We predict each person's yes, then multiply.** A match needs two yeses, and each one alone is
easier to guess: she said yes to 37% of her dates, he to 47%, against 16.5% for matches. Top decile
29.8% against 23.6% predicting the match in one step.

**We test on people the model has never seen.** Each person had about 16 dates, so they appear in 16
rows. Mix those between practice and test and the model recognises the person instead of learning:
**0.72 ROC-AUC against 0.60**. So everyone from one evening stays on the same side.

Everyone meets everyone at their own evening, so the "these two dated" graph has **21 connected
components for 21 waves**. There is no finer cut. And the wave is a **grouping key, never a
feature**: notebook 02 removes it, and everything constant within it.

It matters because the app is launching. Every user is new.

---

## 3. What we found

| Engine | PR-AUC | Top decile | Lift | Instability | Amplification |
|---|---|---|---|---|---|
| Lasso logistic regression | 0.228 | **0.298** | 1.81 | 0.56 | 1.16 |
| XGBoost | 0.218 | 0.286 | 1.74 | **0.30** | **1.01** |
| TabPFN | **0.234** | 0.289 | 1.75 | needs a GPU | 1.27 |

- Random gives 165 matches per 1,000. The engine gives **298**.
- The three are **equally good**: the interval on the top decile is eight points wide, they are one
  point apart.
- No single answer matters much. The strongest moves the chance of a yes by **four points**.
- Rebuild it and **half the recommended list changes**.
- It recommends **Asian participants about half as often**, and the gap survives every check.
- It **exaggerates** the same-background preference by **16%**.
- **Deleting the ethnicity column makes that worse**, 1.16 to 1.35.

---

## 4. Every method, and what it gave

### Performance

| | What it is | Chance | Ours |
|---|---|---|---|
| **Base rate** | what you get doing nothing clever | | 16.5% |
| **Top decile** | of the 419 pairs the app shows, the share that match | 16.5% | 29.8% |
| **Lift** | top decile divided by chance | 1.00 | **1.81** |
| **PR-AUC** | precision across all list lengths, the one to read when matches are rare | 0.165 | 0.228 |
| **ROC-AUC** | chance the engine ranks a real match above a non-match | 0.50 | 0.586 |

ROC-AUC looks poor because it grades the whole ranking including the bottom, which the app never
shows. Our top decile is a point on the PR curve: precision 0.298 at recall 0.181.

**Worth keeping:** ROC-AUC ranks the engines in nearly the reverse order of the client's metric.
Direct XGBoost has the best ROC-AUC, 0.589, and the worst top decile, 0.232.

### Interpretability

| Method | What it answers | What we got |
|---|---|---|
| **Coefficients** | the weight of one answer in the simple model | the lasso keeps 77 of 101 columns |
| **Marginal effect** | the same, in percentage points | `female` −13.7, nothing else above 4 |
| **Impurity importance** | how often the tree split there | the categorical blocks, which is its known bias |
| **Permutation importance** | score lost when you shuffle the column | `other_race`, `raceclash`, `self_income`, `female` |
| **SHAP** | for **one** recommendation, what each answer contributed | six or seven small pieces, never one big |
| **LIME** | the same, as readable rules | agrees with SHAP |
| **PDP** | average prediction as one column moves | `raceclash` 39% to 32%, `agediff` flat |
| **ICE** | the same, drawn once per pair | `other_selfattr` fans out, `agediff` does not |
| **XPER** | splits the **performance** between columns | `female` produces most of the skill |

**The traps, all of which we name:**

- Impurity favours columns with many cut points. Eighteen fields beat one yes/no.
- PDP averages over combinations that never happen. `other_selfattr` jumps between 3 and 4, and only
  4.8% of people rate themselves 4 or below, so that jump is extrapolation.
- SHAP adds up on the **log-odds of one decision**, not on the pair score, which is a product.
- XPER needs 2^p coalitions, so it runs on the eight strongest columns, approximately.
- **p-values after a lasso are not valid.** The lasso selects, an unpenalised logit is refitted for
  the standard errors. Read them as an ordering.

**The headline:** the four methods share **30% of their top five**, and impurity and XPER share
nothing. One importance chart on a slide would be a choice presented as a fact.

### Stability

Turney's definition: two samples from the same population should give approximately the **same
model**, so stability is a distance between models, not the spread of a score. We build 25 versions
of each engine, five folds plus twenty bootstraps within waves.

| | Instability | Shortlist kept after retraining |
|---|---|---|
| logit | **0.56** | 48% |
| XGBoost | **0.30** | 44% |

- The **white box is twice as unstable** as the black box, which is not what you expect from the
  model chosen for being readable.
- **Half the shortlist changes** between refits. That is what the user sees.
- Only 2 of the 55 live coefficients ever change sign, so the story stays stable even when the
  selection does not.
- **Leave one wave out:** PR-AUC from **0.10 to 0.46** depending on the evening, so give the client
  the range. The drift correlates 0.96 with the number of pairs removed, so it is a sample-size
  effect.
- **Anchoring** a refit to the previous model 84% of the way **raises** PR-AUC from 0.287 to 0.312.
  No trade-off to arbitrate: the free fit was overfitting its fold.

### Fairness

Y = 1 the pair matched. **Ŷ = 1 the pair is in the top decile**, because at a 16.5% base rate with a
product of two probabilities no pair reaches 0.5 and a threshold classifier predicts nothing. D is
the protected group.

**Statistical parity**, is every group recommended as often:

| Group | n | Their rate | Everyone else | Gap |
|---|---|---|---|---|
| Asian | 1,978 | 5.8% | 11.3% | **−5.5** |
| White | 4,722 | 9.7% | 10.3% | −0.6 |
| Latino | 664 | 13.3% | 9.7% | +3.6 |
| Other | 521 | 16.5% | 9.5% | +7.0 |
| Black | 420 | 19.3% | 9.5% | **+9.8** |
| mixed pair | 2,526 | | | −3.3 |

**Conditional parity**, the same among people alike on four things the app may legitimately use: how
the partner rates their own looks and sincerity, how alike their interests are, the age gap. Tested
with **Cochran, Mantel and Haenszel**.

**Five of the six gaps survive.** Asian common odds ratio 0.54, Black 2.12, mixed pairs 0.69. Only
White is green. So it is not "people were biased and the model reflects it": the model produces gaps
the legitimate variables do not explain.

**Amplification, our own measure.** Same-background pairs are 41% of real matches and 47.5% of what
we show, so **1.16, interval [1.03, 1.29]**. Above 1 means the engine exaggerates a preference it
found. The engines differ: **XGBoost 1.01, logit 1.16, TabPFN 1.27**.

**Equal opportunity.** Among the pairs that **really matched**, the app surfaces **10.5% of the Asian
ones against 20.1%** for everyone else. A loss, not a ratio.

**Calibration.** When the engine says 30%, is it 30% for everyone? Roughly yes. The problem is not
the numbers, it is where the cut falls.

**Mitigation:**

| Variant | Amplification | Top decile |
|---|---|---|
| full model | 1.16 | 0.298 |
| without ethnicity | **1.35** | 0.251 |
| ethnicity neutralised, no refit | **1.59** | 0.246 |
| full unawareness | **1.07** | 0.248 |

Removing the column makes it worse: the model rebuilds ethnicity from income missingness, going-out
habits and self-rated looks, still recoverable at **AUC 0.66**. Only removing ethnicity,
`raceclash`, `samerace` and both preference columns works, and it costs about **50 matches per
1,000**.

**Equivalence, TOST.** "No significant difference" can mean too little data. This reverses the null
and asks whether the gap is provably small. δ = 5 points, from the four-fifths rule and our own
noise. Only White is equivalent at δ = 3, White and mixed pairs at δ = 5, and for the smallest
groups we cannot conclude: **absence of power, not a clean result**.

---

## 5. Economics

| Shortlist | Matches /1,000 | Extra | Extra over the catalogue |
|---|---|---|---|
| top 5% | 319 | +154 | +32 |
| top 10% | 298 | +133 | +56 |
| top 20% | 235 | +70 | +59 |

The last two columns pull opposite ways, so the length of the list is the client's decision.

**Where ×1.81 comes from.** 690 of 4,184 pairs matched, so random gives 16.5%. The top decile is 419
pairs of which 125 matched, 29.8%. 0.298 / 0.165 = 1.81, interval [1.55, 2.04]. Say the comparator
out loud: 1.8 times more likely **than a profile picked at random**.

**Cost.** Scoring one pair is a dot product over 101 columns, so a few cents per 1,000 users per
year. The engine cannot be priced on what it costs; the real spending is collecting the data and
monitoring fairness.

**Value.** Two numbers are the client's: the share who pay, market band 8% to 15%, and the extra paid
months a satisfied subscriber stays. At $15 a month, 12% and one extra month, **$1,800 per 1,000
users per year**.

**No cost matrix, and that is a result.** At a fixed top-K the list length is constant, so TP + FP = K
and minimising cost is maximising the top-K match rate, which is what we optimised. Every cost matrix
ranks the engines identically. Costs matter for choosing K, not the model.

**History is the biggest lever.** ×1.81 on day one. Give the engine the user's own past decisions,
leave-one-out and still on the wave split, and it reaches **×2.89**. Two interactions already give
×2.06.

---

## 6. What we recommend

1. **Ship the lasso logistic regression.** Not the most accurate, TabPFN is, but at equal accuracy it
   is the only one we can hand over line by line. Say the caveat first: **XGBoost is the fairest**,
   and choosing it costs about one point of top-decile rate.
2. **Anchor each retrain** to the previous version. Costs nothing, drifts less.
3. **Fairness is the client's decision, and we price it:** about 50 matches per 1,000.
4. **Collect interaction history.** ×1.81 becoming ×2.89.
5. **Make some profile fields compulsory.** People who skip the income question are not a random
   group.
6. **Re-run the fairness checks** as the user base grows.

---

## 7. Words used in this repository

**Wave** one speed dating evening, there were 21 · **Pair** one date, seen once · **Leakage** using
information you would not have when you decide · **Cold start** scoring someone never seen before ·
**Proxy** a column standing in for another, like income missingness for background · **Fold** one of
five slices, train on four and test on the fifth · **Out-of-fold** the score a pair got from a model
that never saw it, which is every number here · **Grouped split** cutting so all of one person's rows
stay on the same side · **Lasso** a way of keeping a model simple that pushes useless columns to
exactly zero · **One-hot** turning a label into one yes/no column per value · **Bootstrap** redrawing
the data many times to see how much a number wobbles, which gives the **confidence interval** ·
**Precision** of what you show, the share that works · **Recall** of what would have worked, the
share you found · **Odds ratio** how many times more likely, 1.00 being no difference · **p-value**
how surprised we would be if the column did nothing

---

## 8. Questions we expect

**"0.59 ROC-AUC is barely better than a coin flip."** It is the number the product operates at. A
random split reads 0.72 and the difference is memorisation. What matters is the top decile, 29.8%
against 16.5%.

**"Why the top decile and not a threshold?"** No pair scores above 0.5 at this base rate with a
product of two probabilities. Checked at 5% and 20%.

**"Isn't multiplying two decisions a hack?"** It is the structure of the outcome. Twice the rows,
target at 37% and 47%, six points of top-decile rate.

**"Just drop ethnicity."** Measured: amplification 1.16 to 1.35, five points of match rate lost, and
ethnicity still recoverable at AUC 0.66.

**"Where is your cost matrix?"** At a fixed top-K, minimising cost is maximising the top-K match
rate. Costs choose K, not the model.

**"Your groups are tiny."** Which is why every table carries the group size and why we ran TOST. We
report absence of power, not absence of a gap.

**"Why TabPFN if you do not ship it?"** Best PR-AUC, no native explanation, and it amplifies the
most. That combination is the argument.

**"Is 1.8× a promise for a user?"** An offline cold-start estimate, interval [1.55, 2.04], against
random recommendation. Confirm with an A/B test before advertising it.
