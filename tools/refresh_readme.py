"""Rewrite the results table in README.md from reports/tables/07_scorecard.csv.

Run it after re-executing the notebooks so the README can never quote a stale number:
    python tools/refresh_readme.py
"""

from pathlib import Path

import pandas as pd

START, END = "<!-- scorecard:start -->", "<!-- scorecard:end -->"

score = pd.read_csv("reports/tables/07_scorecard.csv", index_col=0)
columns = {"PR-AUC": "{:.3f}", "Top decile": "{:.3f}", "Lift over chance": "{:.2f}x",
           "Instability of the model": "{:.2f}", "Amplification": "{:.2f}"}

header = "| Engine | " + " | ".join(c.replace(" of the model", "") for c in columns) + " |"
rule = "|" + "---|" * (len(columns) + 1)
rows = [
    "| " + engine + " | "
    + " | ".join("n/a" if pd.isna(line[c]) else fmt.format(line[c]) for c, fmt in columns.items())
    + " |"
    for engine, line in score.iterrows()
]
table = "\n".join([header, rule, *rows])

readme = Path("README.md").read_text()
before, _, rest = readme.partition(START)
_, _, after = rest.partition(END)
Path("README.md").write_text(f"{before}{START}\n{table}\n{END}{after}")
print(table)
