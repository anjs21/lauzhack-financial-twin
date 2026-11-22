"""Recompute the numbers marked † in pipeline/07-validation.md and 05-persona-policy.md (~2 min, CPU).

    python pipeline/diagnostics.py [out.json]

1. Learned markets vs fixed rates for three people, 2,000 lives each, same random numbers.
2. Time for one exploration (2,000 lives x 120 months).
3. Where the spread of 2034 net worth comes from (one source of randomness left on at a time).
4. Yearly action rates of the rule persona over 240 random profiles x 50 lives.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from twin import config as C  # noqa: E402
from twin.data import profiles  # noqa: E402
from twin.explore import goal_met, person_table, simulate, spread_sources, yearly_action_rates  # noqa: E402
from twin.policy.traits import from_profile  # noqa: E402
from twin.world.env import Config, World  # noqa: E402

MIA = "Mia (example persona)"


def people_for(key):
    if key == MIA:
        return person_table(custom=json.loads((ROOT / "web" / "src" / "example.json").read_text())["person"])
    return person_table(key)


def stats(p):
    high, _ = goal_met(p)
    out_months = ((p.mode[:, :-1] == C.NOT_WORKING) & (p.age[:, :-1] < 65)).sum(1)
    return {"high_goal_odds": round(float(high.mean()), 3),
            "credit_breach_odds": round(float(p.credit_breach.any(1).mean()), 3),
            "months_not_working_before_65_mean": round(float(out_months.mean()), 2),
            "median_final_net_worth": round(float(np.median(p.nw[:, -1])))}


def main(out_path=None):
    out = {}
    world, fixed = World.load(), World.load(config=Config(markets=False))

    out["markets_learned_vs_fixed"] = {}
    for key in ["P00001", "P00003", MIA]:
        people = people_for(key)
        persona = from_profile(people.iloc[0])
        out["markets_learned_vs_fixed"][key] = {
            "learned": stats(simulate(world, people, persona, 2000, C.SEED)),
            "fixed_rates": stats(simulate(fixed, people, persona, 2000, C.SEED))}

    people = person_table("P00003")
    t0 = time.time()
    simulate(world, people, from_profile(people.iloc[0]), 2000, C.SEED)
    out["seconds_2000_lives"] = round(time.time() - t0, 2)

    out["spread_p10_p90_chf"] = {}
    for key in ["P00001", MIA]:
        people = people_for(key)
        widths = spread_sources(world, people, from_profile(people.iloc[0]))
        out["spread_p10_p90_chf"][key] = {k: round(v) for k, v in widths.items()}

    ids = np.random.default_rng(C.SEED).permutation(profiles().index.values)[:240]
    prof = profiles().loc[ids]
    rows = []
    for pid, r in prof.iterrows():
        p = simulate(world, prof.loc[[pid]], from_profile(r), 50, C.SEED)
        rows.append({"status": r["status"], **yearly_action_rates(p)})
    df = pd.DataFrame(rows)
    out["persona_rates_per_year"] = {"all": df.drop(columns="status").mean().round(4).to_dict(),
                                     **df.groupby("status").mean().round(3).to_dict(orient="index")}

    text = json.dumps(out, indent=1)
    print(text)
    if out_path:
        Path(out_path).write_text(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
