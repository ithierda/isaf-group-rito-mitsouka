# Everything in this repository, and how to defend it

Written so that any of the six of us can answer any question, technical, methodological or
theoretical, without opening a notebook. Every number here is read from `reports/tables/`.

---

## 1. The project in one page

| | |
|---|---|
| Client | *Les Rito Mitsouka*, a dating app choosing which pairs of profiles to recommend |
| Data | Speed Dating Experiment, Fisman, Iyengar, Kamenica and Simonson (2006), QJE |
| Target | `match` = 1 when both people said yes after a four-minute date |
| Unit | one **pair**, 4,184 of them, 16.5% positive (690 matches) |
| Engines | lasso logistic regression, XGBoost, TabPFN |
| Dimensions | performance (statistical and economic), interpretability, stability, fairness |
| Protected attribute | **ethnicity** |

The product shows a **shortlist**, not a yes or no on every pair. That single sentence justifies most
of what follows.

---

## 2. The data

8,378 rows, 195 columns, 551 participants, 21 evenings called **waves**. Everyone met every
participant of the opposite sex in their wave, and each date produced **two** rows, one filled in by
her and one by him.

**Three things we did to it.**

1. **One row per pair.** Both rows of a date carry the same outcome, so keeping both would duplicate
   every observation and put the same date on both sides of a split. 8,378 rows become 4,184 pairs,
   `her_*` and `his_*`.
2. **Pre-date information only.** 108 of the 195 columns are filled in during the date (the
   scorecard, `dec`, `attr`, `sinc`, `intel`, `fun`, `amb`, `shar`, and the same seen by the partner)
   or after it (the follow-up surveys, suffixes `_s`, `_2`, `_3`). They predict the target extremely
   well and are unusable: the client ranks pairs **before** anyone meets.
3. **Four families of variable, and they are not interchangeable.** Labels stored as numbers
   (`field_cd`, `career_c`, `goal`, `race`) need one-hot encoding. Ratings from 1 to 10 are genuine
   ordinal scales. `date` and `go_out` are **reverse coded**, 1 meaning several times a week. The
   `1_1`, `2_1`, `4_1` blocks are 100 points spread over six qualities, so they are shares, not
   levels.

**The one data quirk to know.** `income` is missing for 62% of Latino and 60% of Asian participants
against 31% of Black ones. The missingness is not random: it carries information about origin. It
gets its own flag, and it comes back in the fairness notebook as one of the columns that lets the
model reconstruct ethnicity.

---

## 3. The two decisions that shape everything

### 3.1 We split on the wave

**What we do.** `StratifiedGroupKFold(5)` grouped on `wave`, fold stored in `model_table.csv`.

**Why, and this is the argument to give.** Everyone meets everyone of the opposite sex in their
evening, so the graph "these two dated" has **21 connected components for 21 waves**. There is no
finer cut that keeps a person entirely on one side. Grouping by person and grouping by wave are the
same operation on this data.

**Why it matters.** Each participant appears in about 16 pairs. A random split scatters them across
both sides, the model learns that this profile says yes a lot, and the score becomes recognition
rather than skill. Measured on the same features: **random 0.72 ROC-AUC against 0.60 grouped.**

**The objection to pre-empt.** "You removed the identifiers, so a random split is fine." No. The 22
profile answers are constant across a person's 16 rows and act as a fingerprint: four ordinary
columns identify 90% of participants, six identify 99%.

**The other objection.** "You split on the wave but then remove the wave, that is contradictory."
No: the wave is a **grouping key, never a feature**. Notebook 02 removes it and everything constant
within a wave, `pool_size` included, so the model never sees which evening a pair came from. The
wave only decides who lands in which fold.

**Scope.** The client is launching the app, so every user is new. Cold start is not a pis-aller, it
is the situation.

### 3.2 A match is two decisions

**What we do.** Instead of one model on 4,184 pairs predicting `match`, one model on **8,368
decisions** predicting `dec`, then the two probabilities multiplied.

Each pair becomes two rows: whoever decides is `self_`, the partner `other_`, plus a `female` flag
for the asymmetry.

**Why.** `match` is a rare joint event at 16.5%. The two decisions on their own are 37% for her and
47% for him. Twice the rows and a nearly balanced target.

**What it buys.** Nothing on ROC-AUC, and a third on the metric that matters:
**top decile 29.8% against 23.6%** predicting the match in one step.

`dec` is unusable as a feature, since it is the answer, but it is a perfectly legitimate target.

---

## 4. The notebooks

| | What it does | Output |
|---|---|---|
| `01_eda` | Unit of analysis, leakage boundary, what the variables mean, one row per pair | `pairs.csv` |
| `02_features` | Cleaning, comparison features, readable names, fold assignment | `model_table.csv` |
| `03_models` | The three engines on the same folds, two-stage and direct | `oof_predictions.csv` |
| `03b_tabpfn_colab` | TabPFN on a Colab GPU | `tabpfn_oof.csv` |
| `04_interpretability` | Coefficients, impurity, PDP, ICE, SHAP, LIME, permutation, XPER, economics | figures and tables |
| `05_stability` | Distance between 25 versions, coefficient drift, leave one wave out, anchoring | figures and tables |
| `06_fairness` | Parity, conditional parity, amplification, mitigation, equal opportunity, TOST | tables |
| `07_recommendation` | The scorecard and what we recommend | `07_scorecard.csv` |

**Out-of-fold predictions** are the hinge. Every pair carries a prediction from a model that never
saw it, so 04 to 07 read scores rather than refitting from scratch. They do rebuild the pipelines,
because an explanation, a distance between models and a mitigation variant all need a fitted model.

**60 features** in `model_table.csv`, **101 columns** after one-hot encoding.

---

## 5. Every method, and the sentence to say about it

### Performance

| Metric | What it is | Chance | Ours (logit) |
|---|---|---|---|
| **ROC-AUC** | probability that a random matching pair is ranked above a random non-matching one | 0.50 | 0.586 |
| **PR-AUC** | average precision across all recall levels, so how well the few positives are found | 0.165 | 0.228 |
| **Top-decile rate** | of the 419 pairs the app would show, the share that match | 0.165 | 0.298 |
| **Lift** | that rate divided by chance | 1.00 | 1.81 |

**Why PR-AUC and not ROC-AUC.** The false-positive rate divides by 3,494 non-matches, so putting 100
bad pairs at the top moves it by 2.9 points, almost nothing, while it wrecks the shortlist. ROC-AUC
scores the whole ranking including the bottom the app never shows. **Our top decile is a point on
the PR curve**: precision 0.298 at recall 0.181.

**The result to keep.** ROC-AUC ranks the models in almost the reverse order of the client's metric:
direct XGBoost has the best ROC-AUC, 0.589, and the worst top decile, 0.232.

### Interpretability

| Method | What it answers | What we found |
|---|---|---|
| **Coefficients + AME** | how much does one answer move the probability | `female` 13.7 points, no profile answer above 4 |
| **Impurity importance** | how often did the tree split here | the categorical blocks first, which is the known bias |
| **PDP** | average prediction as one column moves | `raceclash` 39% to 32%, `agediff` flat |
| **ICE** | the same, one line per pair | `other_selfattr` fans out, `agediff` does not |
| **SHAP** | exact contribution of each column to **one** prediction | six or seven small contributions, never one big |
| **LIME** | a local linear surrogate, returns **rules** | same story, readable as conditions |
| **Permutation importance** | how much out-of-fold performance a column is worth | top four: `other_race`, `raceclash`, `self_income`, `female` |
| **XPER** | splits the performance measure itself, PM = φ₀ + Σφⱼ | `female` produces most of the skill; some contributions negative |

**The traps to name before anyone asks.**

- **Impurity importance** favours columns with many cut points and says nothing about direction.
  Eighteen fields offer eighteen times more places to split.
- **PDP averages over the marginal distribution** of the other columns, so the curve covers
  combinations that never occur. Our example: `other_selfattr` jumps between 3 and 4, and only 4.8%
  of participants rate their own looks 4 or below. That jump is extrapolation.
- **SHAP is additive on the log-odds of one decision**, not on the pair score. `Σφ + φ₀ = margin`
  holds exactly, verified to three millionths. The pair score is a product, so we explain each stage
  separately.
- **XPER** needs 2^p coalitions, so it runs on the eight strongest columns refitted into their own
  model, in approximate mode, which leaves a small gap between φ₀ + Σφⱼ and the measured AUC.
- **p-values after a lasso** are not valid. The lasso selects, an unpenalised logit is refitted to
  get standard errors, and those p-values ignore the selection step. Read them as an ordering.

**The headline of the block.** The four importance methods share **30% of their top five**, and
impurity and XPER share nothing at all. One importance chart on a slide would be a choice presented
as a fact.

### Stability

**Turney's definition, the one the course uses.** Two datasets drawn from the same population should
produce approximately the **same model**. Stability is a distance between models, not the spread of
a score.

**Setup.** 25 versions of each engine: the 5 folds plus 20 bootstrap resamples drawn within waves.
Distance on standardised coefficients for the logit, on normalised importances for XGBoost.

| | Instability (mean distance / size) | Shortlist kept after retraining |
|---|---|---|
| logit | **0.56** | 48% |
| XGBoost | **0.30** | 44% |

**Three findings.**

1. The white box is about twice as unstable as the black box in coefficient space, which is the
   opposite of what one expects from a model chosen for being readable.
2. **About half the shortlist changes** between two refits. That is what the user sees, so it is the
   number that matters commercially. It is our one addition to the course here.
3. Only 2 of the 55 live coefficients ever change sign, so the story we tell the client is stable
   even when the selection is not.

**Over time.** Leave one wave out: PR-AUC runs from **0.10 to 0.46** depending on the evening held
out. Give the client the range, not the average. The drift column correlates 0.96 with the number of
pairs removed, so it is a sample-size effect rather than a property of any evening.

**Buying stability.** Refit the fold-1 model with λ‖θ − θ̂₀‖² pulling it towards the fold-0 model.
Pulling it **84% of the way raises** PR-AUC from 0.287 to 0.312. On this data there is no trade-off
to arbitrate: with 55 live coefficients and 3,300 pairs per fold, the free fit is overfitting its
fold and anchoring is free regularisation.

### Fairness

**The mapping, and state it first.** Y = 1 the pair matched. **Ŷ = 1 the pair is in the top decile**,
so the app would show it. D = 1 the protected group. Two definitions of D: a person's ethnicity, one
group against all others on the 8,368 person-pair rows; and a mixed pair against a same-background
pair on the 4,184 pairs.

**Why top-K and not a 0.5 threshold.** At a 16.5% base rate with a score that is a product of two
probabilities, no pair reaches 0.5. A threshold classifier predicts zero everywhere and every
confusion-matrix metric is degenerate.

**Statistical parity**, Pr(Ŷ=1|D=1) = Pr(Ŷ=1|D=0), two-proportion z-test:

| Group | n | Selection rate | Everyone else | Gap |
|---|---|---|---|---|
| Asian | 1,978 | 5.8% | 11.3% | **−5.5 pts** |
| White | 4,722 | 9.7% | 10.3% | −0.6 pts |
| Latino | 664 | 13.3% | 9.7% | +3.6 pts |
| Other | 521 | 16.5% | 9.5% | +7.0 pts |
| Black | 420 | 19.3% | 9.5% | **+9.8 pts** |
| mixed pair | 2,526 | | | −3.3 pts |

**Conditional parity**, Ŷ ⊥ D | X_c. X_c is four variables we commit to as legitimate and defend:
`other_selfattr` and `other_selfsinc`, how the partner rates their own looks and sincerity;
`interests`, how alike the two activity profiles are; `agegap`. Combined into one score, cut into
four strata, then the **Cochran, Mantel and Haenszel** test asks whether D and Ŷ are still associated
inside the strata.

**Verdict: 4 of the 6 gaps survive.** Asian common odds ratio 0.54, Black 2.12, mixed pairs 0.69.
Only White is green, which is what a majority group usually does. So it is not "the participants were
biased and the model reflects it": the model produces gaps the legitimate variables do not explain.

**Amplification, our contribution.** The share of same-background pairs among the pairs **shown**,
divided by their share among the pairs that **matched**. Above 1, the recommendation is more
segregated than reality. **41% of real matches are same-background, 47.5% of the shortlist. Ratio
1.16, interval [1.03, 1.29].** The engine did not invent a preference, it sharpened one.

**And the three engines differ**, which is the result nobody expects:

| Engine | Amplification | Asian gap |
|---|---|---|
| XGBoost | **1.01** | −2.7 pts |
| logit | 1.16 | −5.5 pts |
| TabPFN | **1.27** | −6.1 pts |

Fairness separates the engines where performance could not, and not in favour of the one we
recommend.

**Mitigation, and this is the useful result.**

| Variant | Amplification | Top decile |
|---|---|---|
| full model | 1.16 | 0.298 |
| without `raceclash` | 1.22 | 0.279 |
| without ethnicity | **1.35** | 0.251 |
| ethnicity neutralised without a refit | **1.59** | 0.246 |
| full unawareness | **1.07** | 0.248 |

**Removing the ethnicity column makes the segregation worse.** Deprived of it, the model leans harder
on whatever correlates with it and pays five points of match rate. Only removing ethnicity,
`raceclash`, `samerace` and both preference columns at once works.

**The proof that unawareness is no defence.** From a table with no ethnicity column, no `samerace`,
no `raceclash` and no background preference, ethnicity is still recoverable at **AUC 0.66** for White
and Asian participants. Income missingness, going-out habits and self-rated looks carry it. For the
three small groups the AUC is near 0.5, which is a statement about 420 rows rather than the absence
of a proxy.

**Equal opportunity**, Ŷ ⊥ D | Y. Among the pairs that **really matched**, the app would have
surfaced **10.5% of the Asian ones against 20.1% for everyone else**. Say this one out loud: it is a
loss, not a ratio.

**Calibration** by group, five bins: close to the diagonal for the two large groups, wandering for
the small ones. The scores mean roughly the same thing across groups. The problem is not calibration,
it is where the cut falls.

**Equivalence, TOST.** A non-significant difference is not evidence of equality. The two one-sided
tests reverse the null to "the gap is at least δ", and rejecting it on both sides is evidence the gap
is smaller than δ. Variance **unpooled**, because under the equivalence null the two proportions are
not assumed equal. **δ = 5 points**, justified by the four-fifths rule and by the noise on our own
estimates; δ = 3 and δ = 10 reported alongside. Only White is equivalent at δ = 3, White and mixed
pairs at δ = 5, and below 10 points the three small groups cannot conclude. **Absence of power, not
a clean bill of health.**

---

## 6. Economics

The baseline the client has on day one is showing pairs at random, 165 matches per 1,000.

| Shortlist | Matches /1,000 | Extra /1,000 | Extra matches over the catalogue |
|---|---|---|---|
| top 5% | 319 | +154 | +32 |
| top 10% | 298 | +133 | +56 |
| top 20% | 235 | +70 | +59 |

**The two columns pull in opposite directions**, so the length of the list is the client's decision:
a short list is more efficient per slot, a long one delivers more matches in total.

**Where the ×1.81 comes from.** 690 of 4,184 pairs matched, so random gives 16.5%. The top decile is
419 pairs of which 125 matched, 29.8%. 0.298 / 0.165 = 1.81. Out-of-fold and grouped by wave, so it
is what a **new** user gets, with an interval of [1.55, 2.04]. The comparator must be said out loud:
1.8 times more likely **than a profile picked at random**.

**Cost.** Scoring one pair is a dot product over 101 columns, microseconds, so the engine runs for a
few cents per 1,000 users per year. **It cannot be priced on what it costs.** The real expense is
collecting profile data and the fairness monitoring, in person-days.

**Value.** Two numbers come from the client: `c`, the share who pay, market band 8% to 15% (Tinder
8.6 million payers against roughly 60 million monthly users, Grindr 8.4%), and `r`, the extra paid
months a satisfied subscriber stays. At $15 a month, 12% conversion and one extra month, **$1,800 per
1,000 users per year**, tens of thousands of times the running cost.

**Why there is no cost matrix, if asked.** At a fixed top-K the list length is constant, so
TP + FP = K and the cost collapses to `const − (c_FP + c_FN)·TP`. Every cost matrix ranks the models
identically, and minimising cost is maximising the top-K match rate, which is what we optimised. The
costs matter for choosing K, not the model, so we hand the client the curve instead of a number we
invented.

**The history lever.** ×1.81 on the day a user signs up. Give the engine that user's own past
decisions, computed leave-one-out and still on the wave split, and it reaches **×2.89**. Two
interactions already take it to ×2.06, eight to ×2.74. The onboarding should harvest a few decisions
fast, because that is where the curve is steepest.

---

## 7. The code, and the traps inside it

**The pipeline.** Encoding lives **inside** the model, so it is fitted on the training fold only:

```python
ColumnTransformer([("categories", OneHotEncoder(min_frequency=0.03,
                    handle_unknown="infrequent_if_exist"), SIDE_CATEGORICAL)],
                  remainder="passthrough")
→ SimpleImputer(strategy="median") → [StandardScaler] → estimator
```

`min_frequency=0.03` merges categories under 3% of rows. `handle_unknown` covers a category that only
turns up in a test wave.

**Seven traps we hit, in case one is asked about.**

1. **`LogisticRegression` must carry `random_state=0`.** liblinear shuffles internally. Without the
   seed two identical runs of 03 gave predictions differing by up to 0.009 per pair while XGBoost was
   bit-identical. Aggregate metrics never moved, which is why it hid for so long.
2. **`career_c` left in as an integer** was a label treated as a quantity, exactly the error we warn
   about in 01. It is now dropped.
3. **`pool_size` was the wave in disguise.** One value per wave, so a tree can isolate an evening and
   memorise its match rate. Removing it raised the worst fold from 0.556 to 0.569.
4. **The encoder learns different categories on different resamples**, so model widths vary from 99
   to 105 across the 25 versions in 05. Signatures are reindexed onto a common column set, with an
   unseen column counting as zero.
5. **Marginal effects on a standardised design** cannot move a dummy from 0 to 1, because the two
   levels are no longer 0 and 1. The code moves between the column's actual two values.
6. **`pair` is read back from CSV as a string**, so joining on a tuple key silently produced all-NaN
   and an AUC of exactly 0.5 until the key was rebuilt as a string on both sides.
7. **XPER 0.0.92 has a bug**: `kernel=False` passes a list where an integer is expected, and
   `N_coalition_sampled` above `2**p - 2` raises `IndexError`.

**Environment.** Python 3.11 in `~/.venvs/isaf`, **outside the repository**. When it lived in `.venv`
under `~/Documents`, iCloud evicted the packages and every import waited on a download, which looks
exactly like a notebook that will not run.

---

## 8. The scorecard, and what we recommend

| | logit | XGBoost | TabPFN |
|---|---|---|---|
| PR-AUC | 0.228 | 0.218 | **0.234** |
| Top decile | **0.298** | 0.286 | 0.289 |
| Matches per 1,000 | **298** | 286 | 289 |
| Interpretability | 77 readable coefficients | SHAP, four methods disagree | permutation importance only |
| Instability | 0.56 | **0.30** | needs a GPU to measure |
| Amplification | 1.16 | **1.01** | 1.27 |

**The three engines are separated by less than the width of our error bars on performance**, the
interval on the top decile being eight points wide. So the choice is made on the other three
dimensions.

**What we recommend.**

1. **Deploy the two-stage lasso logistic regression.** Not because it scores best, TabPFN does.
   Because at equal performance it is the only one we can hand over coefficient by coefficient. Say
   the caveat before anyone asks: **XGBoost is the fairest of the three**, and choosing it instead is
   defensible for about one point of top-decile rate.
2. **Ship it anchored** to the previous version, λ ≈ 0.3. It costs nothing and gains.
3. **Fairness is a product decision, and we price it**: removing the amplification costs about 50
   matches per 1,000 recommendations. Removing only the ethnicity column makes it worse.
4. **Collect interaction history**, ×1.81 becoming ×2.89. The strongest lever the product has.
5. **Make some profile fields mandatory.** Income missingness is not random and the model uses it to
   reconstruct ethnicity.
6. **Re-run the fairness checks with more users.** Below 10 points of tolerance the three small
   groups cannot conclude.

---

## 9. Questions, and the answers

**"0.59 ROC-AUC is barely better than a coin flip."** It is the number the product operates at. On a
random split the same model reads 0.72 and the difference is memorisation. What matters commercially
is the top decile, 29.8% against a 16.5% base rate.

**"Why the top decile and not a threshold?"** No pair scores above 0.5 at a 16.5% base rate with a
product of two probabilities, so a threshold classifier predicts zero everywhere. Checked at 5% and
20%.

**"Isn't multiplying two decisions a hack?"** It is the generative structure of the outcome. It
doubles the rows, moves the target to 37% and 47%, and gains six points of top-decile rate.

**"Your split throws away information."** It throws away the ability to recognise people, which we
would not have in production. Grouping by person is grouping by wave here: 21 connected components
for 21 waves.

**"You split on the wave then delete it."** The wave is a grouping key, not a feature. It decides
folds, it never enters the model.

**"Just drop ethnicity."** Measured: amplification rises from 1.16 to 1.35 and it costs five points
of match rate. Ethnicity is recoverable at AUC 0.66 from the remaining columns.

**"Where is your cost matrix?"** At a fixed top-K, minimising cost is maximising the top-K match
rate, which is what we optimised. Costs matter for choosing K, not the model.

**"Your groups are tiny, 420 rows for Black participants."** Which is why every table carries the
group size and why we ran TOST. We report absence of power, not absence of a gap.

**"Why TabPFN if you do not ship it?"** It has the best PR-AUC and no native explanation, which is
itself the argument. It also amplifies the most, 1.27.

**"Is 1.8× a promise you can make to a user?"** It is an offline cold-start estimate with an interval
of [1.55, 2.04], measured against random recommendation. The client should confirm it with an A/B
test before advertising it.
