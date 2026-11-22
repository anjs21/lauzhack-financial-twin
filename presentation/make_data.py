"""Simulate Jaycent's three study paths for the presentation and write presentation/data.json (~2 min).

    python presentation/make_data.py

The game-theory transition model (twin_game_theory/world/transition.py) is the one with a PhD market.
It has no countries or currencies: the three paths run in the Swiss-calibrated world, in CHF.

Illustrative starting point: 22, finishing a bachelor's in Lausanne, CHF 8,000 saved, shared flat.
Paths (decision rules for the study choice only; housing, spending and savings stay as they are):
    work      graduate, then search and accept the first job offer
    master    graduate, apply for a master's, then search and accept the first job offer
    phd       as master, then apply for a PhD for up to 12 months (it needs a master's), then work
Each path: 2,000 lives x 120 months; life i uses refit i mod 5 (model uncertainty).
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twin_game_theory.world import spec as sp  # noqa: E402
from twin_game_theory.world.transition import Action, State, TransitionModel  # noqa: E402

LIVES, MONTHS, CHUNK = 2000, 120, 125
START = State(t=0, age=22.0, W=sp.S, prog=sp.BACHELOR, edu=sp.NONE, k=1, H=0, F=sp.SINGLE, C=8000.0,
              city="Lausanne")
PATHS = {"work": "Work now", "master": "Master's", "phd": "PhD (after a master's)"}
WORKING = (sp.E60, sp.E80, sp.E100)


def choose(model, s, path):
    want = "stay"
    if s.pend == sp.P_JOB_OFFER:
        want = "accept"
    elif s.pend == sp.P_ADMIT_MASTER:
        want = "accept" if path in ("master", "phd") else "stay"
    elif s.pend == sp.P_ADMIT_PHD:
        want = "accept" if path == "phd" else "stay"
    elif s.W == sp.N and s.pend == sp.P_NONE:
        if path in ("master", "phd") and s.edu == sp.BACHELOR:
            want = "apply_master"
        elif path == "phd" and s.edu == sp.MASTER and s.u < 12:
            want = "apply_phd"
        else:
            want = "plus"
    return Action(want if want in model.allowed(s)["status"] else "stay")


def run_chunk(args):
    path, first = args
    model = TransitionModel()
    n = min(CHUNK, LIVES - first)
    nw = np.zeros((n, MONTHS + 1))
    first_job = np.full(n, np.nan)
    edu, working_end, breach = np.zeros(n, int), np.zeros(n, bool), np.zeros(n, bool)
    for j in range(n):
        i = first + j
        rng, s = np.random.default_rng([2026, i]), START
        nw[j, 0] = s.C + s.I
        for t in range(MONTHS):
            s, _ = model.sample(s, choose(model, s, path), rng, draw=i % 5)
            nw[j, t + 1] = s.C + s.I
            if np.isnan(first_job[j]) and s.W in WORKING:
                first_job[j] = s.age
        edu[j], working_end[j], breach[j] = s.edu, s.W in WORKING, s.breach
    return path, first, nw, first_job, edu, working_end, breach


def main():
    jobs = [(p, f) for p in PATHS for f in range(0, LIVES, CHUNK)]
    parts = {p: [] for p in PATHS}
    with ProcessPoolExecutor(max_workers=16) as pool:
        for path, first, *rest in pool.map(run_chunk, jobs):
            parts[path].append((first, rest))
    age = START.age + np.arange(MONTHS + 1) / 12
    out = {"start": {"age": START.age, "cash_chf": START.C, "city": START.city, "lives_per_path": LIVES},
           "age": age[::3].round(3).tolist(), "paths": {}}
    for path, label in PATHS.items():
        cols = [np.concatenate([r[k] for _, r in sorted(parts[path], key=lambda x: x[0])]) for k in range(5)]
        nw, first_job, edu, working_end, breach = cols
        q = np.percentile(nw, [10, 25, 50, 75, 90], axis=0)[:, ::3] / 1000
        rng = np.random.default_rng(7)
        sample = nw[rng.choice(LIVES, 40, replace=False)][:, ::3] / 1000
        out["paths"][path] = {
            "label": label,
            "fan_k": dict(zip(["p10", "p25", "p50", "p75", "p90"], q.round(2).tolist())),
            "sample_lives_k": sample.round(2).tolist(),
            "median_net_wealth_at_32_k": round(float(np.median(nw[:, -1])) / 1000, 1),
            "p10_net_wealth_at_32_k": round(float(np.percentile(nw[:, -1], 10)) / 1000, 1),
            "p90_net_wealth_at_32_k": round(float(np.percentile(nw[:, -1], 90)) / 1000, 1),
            "share_with_master": round(float((edu >= sp.MASTER).mean()), 3),
            "share_with_phd": round(float((edu == sp.PHD).mean()), 3),
            "median_age_first_job": round(float(np.nanmedian(first_job)), 1),
            "share_working_at_32": round(float(working_end.mean()), 3),
            "share_credit_limit_hit": round(float(breach.mean()), 3),
        }
        print(label, {k: v for k, v in out["paths"][path].items() if not isinstance(v, (dict, list))})
    (Path(__file__).parent / "data.json").write_text(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
