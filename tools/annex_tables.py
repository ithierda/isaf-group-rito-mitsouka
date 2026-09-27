"""Render the annex tables as slide-sized PNGs, straight from reports/tables/.

    python tools/annex_tables.py

Nothing here computes anything: it only formats what the notebooks exported, so the annex can never
disagree with the analysis. Re-run it after the notebooks.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BLUE, YELLOW, INK, TINT = "#1E33C8", "#FFD709", "#101014", "#F2F3F9"
Path("reports/figures").mkdir(parents=True, exist_ok=True)


def render(frame, title, note, out, width=None, highlight=None, row_width=0.55):
    """One dataframe, one PNG. `highlight` marks cells to paint yellow."""
    height = 1.5 + row_width * len(frame)
    figure, axis = plt.subplots(figsize=(width or 2 + 2.1 * len(frame.columns), height))
    axis.axis("off")
    table = axis.table(cellText=frame.astype(str).to_numpy(), colLabels=frame.columns,
                       rowLabels=frame.index, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(2)
        if row == 0:
            cell.set_facecolor(BLUE)
            cell.set_text_props(color="white", weight="bold")
        elif col == -1:
            cell.set_facecolor(TINT)
            cell.set_text_props(weight="bold", color=INK)
        elif highlight and highlight(frame.iloc[row - 1, col]):
            cell.set_facecolor(YELLOW)
            cell.set_text_props(color=INK, weight="bold")
        else:
            cell.set_facecolor(TINT)
            cell.set_text_props(color=INK)
    axis.set_title(title, loc="left", fontsize=16, color=INK, pad=18)
    figure.text(0.01, 0.015, note, fontsize=10.5, color="#555869")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print("wrote", out)


# A5, equivalence. One column per tolerance, so the verdict is read across.
eq = pd.read_csv("reports/tables/06_equivalence.csv")
wide = eq.pivot(index="D", columns="delta", values="equivalent")[["3%", "5%", "10%"]]
wide.columns = [f"delta = {c}" for c in wide.columns]
wide.insert(0, "rows", eq.drop_duplicates("D").set_index("D").n.reindex(wide.index))
wide.index.name = "Protected group"
render(wide, "Equivalence testing (TOST): can we certify that a gap is small?",
       "reports/tables/06_equivalence.csv. 'Yes' means the gap is shown to be under the tolerance. "
       "'Cannot conclude' is an absence of power, not a clean result.",
       "reports/figures/annex_equivalence.png",
       highlight=lambda v: v == "yes")

# A5, the verdicts next to it.
ver = pd.read_csv("reports/tables/06_parity_verdicts.csv", index_col=0)
view = pd.DataFrame({
    "rows": ver.n,
    "gap (pts)": ver["difference (pts)"].round(1),
    "parity p": ver["parity p"].apply(lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"),
    "conditional p": ver["conditional p"].apply(lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"),
    "verdict": ver.verdict.str.split(":").str[0]})
view.index.name = "Protected group"
render(view, "Statistical parity, then conditional parity",
       "reports/tables/06_parity_verdicts.csv. Conditional on attractiveness, sincerity, shared "
       "interests and age gap, tested with Cochran, Mantel and Haenszel.",
       "reports/figures/annex_parity_verdicts.png",
       highlight=lambda v: v == "RED")

# A6, what the 0.56 is made of.
full = pd.read_csv("reports/tables/05_coefficient_drift.csv", index_col=0)
live = full[full["mean"].abs() > 0.02]
moves = ((live["sign changes (%)"] > 0).sum(), (live["dropped by the lasso (%)"] > 20).sum(), len(live))
drift = full.head(10)
drift = pd.DataFrame({
    "mean coefficient": drift["mean"].round(3),
    "standard deviation": drift["std"].round(3),
    "dropped by the lasso": drift["dropped by the lasso (%)"].round(0).astype(int).astype(str) + "%",
    "sign changes": drift["sign changes (%)"].round(0).astype(int).astype(str) + "%"})
drift.index.name = "Column"
render(drift, "How much each coefficient moves across 25 refits",
       f"reports/tables/05_coefficient_drift.csv, ten strongest columns. The instability of 0.56 is "
       f"the typical distance between two of those 25 versions, divided by the size of the average. "
       f"The strongest columns are steady; of the {moves[2]} columns with a real coefficient, "
       f"{moves[0]} change sign and {moves[1]} are dropped by the lasso in more than a fifth of the "
       f"refits, and that is where the 0.56 comes from.",
       "reports/figures/annex_coefficient_drift.png", width=12)

# A8, where the 1,800 comes from.
econ = pd.read_csv("reports/tables/04_unit_economics.csv", index_col=0)
econ.index.name = "Subscription price"
cost = pd.read_csv("reports/tables/04_running_cost.csv", index_col=0, header=None).squeeze()
render(econ.map(lambda v: f"${v:,}"),
       "Value per 1,000 users per year, at 12% of users paying",
       f"reports/tables/04_unit_economics.csv. Columns are the extra paid months the engine buys per "
       f"subscriber. Running cost, measured in 04: {cost['microseconds per pair']:.1f} microseconds "
       f"per pair, ${cost['dollars per 1000 users per year']:.3f} per 1,000 users per year.",
       "reports/figures/annex_unit_economics.png", width=11,
       highlight=lambda v: v == "$1,800")
