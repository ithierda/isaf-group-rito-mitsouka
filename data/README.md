# Data

## `raw/speed_dating.csv`

**Speed Dating Experiment** — Fisman, Iyengar, Kamenica & Simonson (2006),
*Gender Differences in Mate Selection: Evidence from a Speed Dating Experiment*, QJE.
Collected at Columbia Business School, 2002–2004. Public copy of the Kaggle
"Speed Dating Experiment" file (identical values).

| | |
|---|---|
| Rows | 8,378 (one row = one participant's view of one 4-minute date) |
| Columns | 195 |
| Participants | 551 (`iid`), across 21 sessions (`wave`) |
| Target | `match` = 1 if both said yes — 1,380 positives (16.5%) |
| Encoding | ASCII, comma-separated |

Load with:

```python
import pandas as pd
df = pd.read_csv("data/raw/speed_dating.csv")
```

## Codebook

`docs/speed_dating_data_key.doc` describes every variable (coding of `race`,
`field_cd`, `goal`, `go_out`, preference scales, etc.).

## Things to keep in mind

- **Each date appears twice** (once from each partner's side: `iid`/`pid`).
  There are 4,184 unique pairs. `match` is identical on both rows, so the
  train/test split must keep both rows of a pair (and ideally a whole wave)
  on the same side, otherwise the test set leaks.
- **Post-date variables** (`dec`, `dec_o`, `attr`…`like`, `prob`, `met` and their
  `_o` versions, plus all `*_2` / `*_3` follow-up survey columns) are known
  only after the date → to be handled in the leakage step.
- Some columns are mostly empty (e.g. `num_in_3` 92% missing, `expnum` 79%).
- `race` is missing for 63 rows.
