# The whole project in plain words

No jargon. Every technical word used anywhere in this repository is defined at the bottom, in one
line each, with our own numbers as the example. If a sentence here is unclear, that is a bug: tell
us and we fix it.

---

## What we were asked to do

A dating app has a lot of users. It cannot show everyone to everyone, so it has to pick. We build
the thing that picks, and then we check it four ways: does it work, can we explain it, does it stay
the same when we rebuild it, and is it fair.

## What the data is

A real experiment. 551 students went to speed dating evenings at Columbia between 2002 and 2004.
Each evening, every woman met every man for four minutes. After each date both people ticked yes or
no on a card. **If both ticked yes, that is a match.** 16.5% of dates ended that way.

We know what each person wrote about themselves before the evening: age, studies, what they like
doing, what they say they want in a partner. That is all our model is allowed to use, because a
dating app knows those things before anyone meets.

## The two ideas that matter

**One. We predict each person's yes separately, then multiply.**

A match needs two yeses. Rather than guess "will this pair match", we guess "will she say yes" and
"will he say yes", then multiply the two chances. It works better because each question on its own
is easier: she said yes to 37% of her dates, he to 47%, while matches are only 16.5%.

**Two. We test the model on people it has never seen.**

Each person had about 16 dates, so they appear in 16 rows of our table. If we mix those rows between
the practice set and the test set, the model recognises the person instead of learning anything, and
its score looks much better than it is. So we keep all the people from one evening together: they
either all go in the practice set, or all in the test set.

This matters because the app is launching. Every user will be new. The honest question is "how good
is it on somebody it has never seen", and that is the question we answer.

## What we found, in one line each

- Showing pairs at random, 165 out of 1,000 match. Our engine gets **298**. That is **1.8 times
  better**.
- All three engines we tried are **equally good**. The difference between them is smaller than our
  own measurement error.
- No single answer on the form matters much. The strongest one moves the chance of a yes by **four
  points**. The engine works by adding up many small things.
- Rebuild the engine on a different sample and **half the recommended list changes**. That is normal
  at this size, and worth telling the client.
- The engine **recommends Asian participants about half as often** as everyone else, and that gap
  survives every check we ran.
- It also **exaggerates** people's own preference for dating within their background, by 16%.
- **Deleting the ethnicity column makes that worse, not better**, because the engine rebuilds it from
  other answers like whether someone left the income question blank.

## What we tell the client

1. Use the simple model, the one we can read line by line. It is not the most accurate, but at equal
   accuracy we can explain and audit it.
2. When you retrain it, tie it to the previous version. It costs nothing and it drifts less.
3. Fixing the unfairness is possible and costs about **50 matches per 1,000 recommendations**. That
   is your call, not ours, and now you know the price.
4. Ask users for a few decisions early. Once the app has seen a handful, the engine goes from
   **1.8 times better to 2.9 times better**. This is the biggest lever you have.
5. Make some profile fields compulsory. People who skip the income question are not a random group.
6. Run the fairness checks again when you have more users. On our smallest groups we could not
   conclude either way.

---

## Every technical word, in one line

### About the data

**Wave.** One speed dating evening. There were 21. Everyone met everyone of the opposite sex at their
own evening, and nobody attended two.

**Pair.** One date, seen once. The raw file has each date twice, her card and his card, so we merged
them into a single row.

**Leakage.** Using information you would not have at the moment you decide. Here, anything written
during or after the date. It makes a model look brilliant and useless.

**Cold start.** Scoring somebody the app has never seen before. The situation on launch day.

**Proxy.** A column that quietly stands in for another one. Whether someone answered the income
question is a proxy for their background, so removing ethnicity does not remove ethnicity.

### About training and testing

**Fold.** One of five slices of the data. We train on four and test on the fifth, five times, so
every pair gets tested once.

**Out-of-fold prediction.** The score a pair got from a model that had never seen it. Every number in
this project is out-of-fold.

**Grouped split.** Cutting the data so that all of one person's rows stay on the same side. Here it
means cutting by evening.

**Regularisation.** Telling a model to keep its answer simple, to stop it memorising. The **lasso** is
a kind of regularisation that pushes useless columns to exactly zero, which is also why it is easy to
read.

**One-hot encoding.** Turning a label like "field of study" into one yes/no column per field, because
the numbers 1 to 18 in the file are names, not quantities.

**Bootstrap.** Redrawing the data at random, many times, to see how much a number wobbles. It gives
the **confidence interval**, the range the true value plausibly sits in.

### About measuring performance

**Base rate.** What you get by doing nothing clever. Here 16.5%, the share of all pairs that match.

**Top decile.** The best 10% of pairs according to the engine, which is what the app would show. Our
number: 29.8% of them match.

**Lift.** The top decile divided by the base rate. 29.8 / 16.5 = **1.81**. "Our recommendations are
1.8 times more likely to work than a random one."

**Precision.** Of what you show, the share that works. **Recall.** Of what would have worked, the
share you found.

**PR-AUC.** One number summarising precision across all list lengths. Good when what you care about
is rare. Chance is 0.165 here, we get 0.228.

**ROC-AUC.** The chance that the engine ranks a real match above a non-match. Chance is 0.50, we get
0.586. It looks poor because it grades the whole ranking including the bottom, which the app never
shows.

### About explaining the model

**Coefficient.** In the simple model, the weight given to one answer. Bigger absolute value, bigger
effect.

**Marginal effect.** The same thing in plain units: how many percentage points the chance of a yes
moves. Easier to say out loud than a coefficient.

**p-value.** How surprised we would be to see this effect if the column actually did nothing. Small
means "probably not a fluke".

**Impurity importance.** How often the tree used a column. Misleading: a column with many possible
values gets used more regardless of whether it helps.

**Permutation importance.** Shuffle one column and see how much the score falls. Answers "is this
column worth having".

**SHAP.** For one single recommendation, how much each answer pushed it up or down. The pieces add up
exactly to the prediction.

**LIME.** Same question, different method: it fits a tiny model around that one case and returns
readable rules like "optimism below 4 pushes this down".

**PDP, partial dependence.** The average prediction as one column moves, everything else left alone.
Its weakness: the average includes combinations that never occur in real life.

**ICE.** The same curve drawn once per pair instead of averaged. If the lines fan out, the column
matters for some people and not others.

**XPER.** The professor's method. It splits the performance itself between columns: this much of our
skill comes from this column. Different from SHAP, which splits a prediction, not a score.

### About stability

**Stability.** Rebuild the model on a comparable sample. Do you get the same model? If not, the
explanation you gave the client changes every time you retrain.

**Instability, our 0.56 and 0.30.** The typical distance between two rebuilds, divided by the size of
the average model. Lower is steadier. The simple model is the less steady of the two, which is not
what you would expect.

**Jaccard.** How much two lists overlap. Two rebuilds share 48% of their top 10%, so about half the
recommended list changes.

**Leave one wave out.** Train on 20 evenings, test on the 21st, 21 times. Our score ranges from 0.10
to 0.46 depending which evening, which is the range the client should be told.

**Anchoring.** Asking a retrained model to stay close to the previous one. Here it costs nothing and
actually improves the score.

### About fairness

**Protected attribute.** The characteristic we must not discriminate on. Here, ethnicity.

**Statistical parity.** Every group should be recommended as often as every other. Asian participants
appear in 5.8% of shown pairs against 11.3% for others, so this fails.

**Conditional parity.** The same, but comparing only people who are alike on things the app is
entitled to use. If the gap disappears, it was explained. **Ours does not disappear**, which is the
finding.

**Cochran, Mantel and Haenszel.** The test for that: it splits people into comparable groups and asks
whether the gap remains inside each one.

**Equal opportunity.** Among the pairs that really did match, did the app show them equally? We show
10.5% of Asian real matches against 20.1% for others. This is the number that costs the client
something.

**Calibration.** When the engine says 30%, is it really 30%, for everyone? Ours is roughly fine. The
problem is not the numbers, it is where the cut falls.

**Amplification, our own measure.** People do prefer their own background. Does the engine exaggerate
it? Same-background pairs are 41% of real matches and 47.5% of what we show, so **1.16**. Above 1
means yes.

**Unawareness.** Hoping that deleting the sensitive column makes the model fair. It does not: from a
table with no ethnicity at all we can still guess someone's ethnicity correctly most of the time.

**TOST, equivalence testing.** "No significant difference" can just mean too little data. This test
turns it round and asks whether the gap is provably small. For our three smallest groups the answer
is that we cannot tell, and saying so is the honest result.
