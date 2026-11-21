"""Optional plot: run after analyze_mdp.py; requires matplotlib."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
with (HERE / "results/employment_cohorts.csv").open() as f:
    cohorts = list(csv.DictReader(f))
with (HERE / "results/sensitivity_summary.csv").open() as f:
    sensitivity = {r["intervention"]: r for r in csv.DictReader(f)}

cohorts.sort(key=lambda r: float(r["shortfall_pct"]))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.15]})
fig.patch.set_facecolor("#fafafa")
for ax in axes:
    ax.set_facecolor("#fafafa")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)

labels = [r["employment_status"].replace("_", " ").title() for r in cohorts]
values = [float(r["shortfall_pct"]) for r in cohorts]
axes[0].barh(labels, values, color="#397b9b", height=.65)
for i, r in enumerate(cohorts):
    axes[0].text(values[i]+1.3, i, f'{r["cash_shortfall_profiles"]}/{r["profiles"]} ({values[i]:.1f}%)', va="center", fontsize=10)
axes[0].set_xlim(0, 110)
axes[0].set_xlabel("Profiles with negative month-end cash (%)")
axes[0].set_title("Recorded shortfalls concentrate in two cohorts", loc="left", pad=15, fontsize=12, weight="bold")

items = [("Baseline", "baseline"), ("+CHF 250/month", "monthly_net_plus_250"),
         ("+CHF 500/month", "monthly_net_plus_500"), ("+CHF 1,000/month", "monthly_net_plus_1000"),
         ("Rent reduced 20%", "rent_reduction_20pct"),
         ("Liquidate opening investments", "liquidate_all_opening_investments")]
axes[1].barh([x[0] for x in items], [int(sensitivity[x[1]]["replay_shortfall_profiles"]) for x in items],
             color=["#9b5252"]+["#397b9b"]*5, height=.65)
for i, (_, key) in enumerate(items):
    n = int(sensitivity[key]["replay_shortfall_profiles"])
    axes[1].text(n+1, i, str(n), va="center")
axes[1].invert_yaxis()
axes[1].set_xlim(0, 70)
axes[1].set_xlabel("Profiles still experiencing a cash shortfall")
axes[1].set_title("Single-lever replays show different liquidity effects", loc="left", pad=15, fontsize=12, weight="bold")
fig.suptitle("MDP action priorities: evidence from 1,200 synthetic profiles", x=.05, ha="left", fontsize=17, weight="bold")
fig.text(.05, .075, "January 2025–December 2026. Replays hold recorded net flows fixed and assume immediate changes.\n"
         "No moving costs or investment fees/returns. Liquidation changes liquidity, not wealth. These are not causal estimates.",
         fontsize=10, color="#555555", va="top")
fig.tight_layout(rect=(0, .14, 1, .91), w_pad=2.5)
fig.savefig(HERE / "results/action_priorities.png", dpi=180, facecolor=fig.get_facecolor())
plt.close(fig)
print(HERE / "results/action_priorities.png")
