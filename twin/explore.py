"""Explore one person's possible lives: sample many paths, then summarise the spread.

    paths = simulate(world, person_table("P00042"), persona, n_paths=2000, seed=1)
    result = explore("P00042")          # simulate + summaries, JSON-ready

Sources of spread on every path: traits (who they are), the policy's draws (what they
choose), events, flows and markets (luck), and the ensemble member (what the model does
not know). `spread_sources` switches them off one at a time.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .data import profiles
from .policy.foresight import life_utility
from .policy.rules import RulePolicy
from .policy.traits import Persona, from_profile, sample_traits
from .world.env import Streams, World
from .timeline import Timeline, observed_month

EVENTS = ["job_loss", "offer", "graduated", "partner", "child", "separation", "forced_move", "windfall", "side_job"]


@dataclass
class Paths:
    people: pd.DataFrame
    n_paths: int
    cash: np.ndarray        # (N, T+1), state at the start of each month; index T = end of horizon
    inv: np.ndarray
    debt: np.ndarray
    income: np.ndarray
    mode: np.ndarray
    tier: np.ndarray
    household: np.ndarray
    lifestyle: np.ndarray
    econ: np.ndarray
    age: np.ndarray
    action: np.ndarray      # (N, T, 4) actions taken
    events: dict            # name -> (N, T)
    below_zero: np.ndarray  # (N, T)
    traits: np.ndarray
    credit_breach: np.ndarray

    @property
    def nw(self):
        return self.cash + self.inv - self.debt


def person_table(profile_id: str | None = None, custom: dict | None = None) -> pd.DataFrame:
    """One person in the format of data.profiles(): a profile from the data or a custom one."""
    if custom is None:
        return profiles().loc[[profile_id]]
    d = dict(custom)
    status, hh = d["status"], d["household_type"]
    row = dict(name=d.get("name", "custom"), age=float(d["age"]), city=d["city"], city_idx=C.CITY[d["city"]],
               status=status, mode=C.STATUS_TO_MODE[status], self_employed=status == "self_employed",
               pensioner=status == "retired", household_type=hh, household=C.HOUSEHOLD_OF[hh], tier=C.TIER_OF[hh],
               income0=float(d["income"]), rent0=float(d["rent"]), cash0=float(d["cash"]),
               inv0=float(d.get("investments", 0)), debt0=float(d.get("debt", 0)), rate0=float(d.get("rate", 0)),
               high_goal=d.get("high_goal", "Emergency savings"), high_target=float(d.get("high_target", 20000)),
               med_goal=d.get("med_goal"), med_target=float(d.get("med_target", np.nan)),
               tuition=float(d.get("tuition", np.nan)),
               study_left0=max(1, int(d.get("study_months_left", 1))) if status == "student" else 0,
               masters0=bool(d.get("in_masters", False)) and status == "student")
    return pd.DataFrame([row], index=[d.get("id", "custom")])


def goals_per_path(people: pd.DataFrame, n_paths: int) -> dict:
    rep = lambda c: np.repeat(people[c].values, n_paths)  # noqa: E731
    return {"high_target": rep("high_target").astype(float),
            "high_cash_only": rep("high_goal") == "Emergency savings",
            "med_target": rep("med_target").astype(float)}


def simulate(world: World, people: pd.DataFrame, personas, n_paths=2000, seed=C.SEED, horizon=C.HORIZON,
             shared=(), policy_cls=RulePolicy, timeline=None) -> Paths:
    personas = personas if isinstance(personas, (list, tuple)) else [personas] * len(people)
    N = len(people) * n_paths
    streams = Streams(seed, N, shared)
    z = streams.normal("traits", 0, 8)
    traits = np.vstack([sample_traits(p, n_paths, z[:, i * n_paths:(i + 1) * n_paths]) for i, p in enumerate(personas)])
    temperature = np.repeat([p.temperature for p in personas], n_paths)
    s = world.initial_state(people, n_paths, streams, buffer_months=policy_cls.buffer_months(traits))
    intentions = [it for p in personas[:1] for it in p.intentions]      # intentions belong to the explored person
    policy = policy_cls(world, traits, temperature, goals_per_path(people, n_paths), intentions)
    policy.reset(s, streams)
    T = horizon
    rec = {k: np.empty((N, T + 1)) for k in ["cash", "inv", "debt", "income", "age"]}
    rec.update({k: np.empty((N, T + 1), np.int8) for k in ["mode", "tier", "household", "lifestyle", "econ"]})
    action = np.empty((N, T, 4), np.int8)
    events = {k: np.zeros((N, T), bool) for k in EVENTS}
    below = np.zeros((N, T), bool)

    def keep(k, st):
        for name in rec:
            rec[name][:, k] = getattr(st, name)
    keep(0, s)
    timeline = timeline or Timeline()
    breaches = np.zeros_like(below)
    count, bias = 0, 0.0
    for t in range(T):
        offer = timeline.offers.get(t)
        if t in timeline.observations:
            count, bias, breaches[:, t] = observed_month(
                world, s, timeline.observations[t], offer, count, bias)
            keep(t + 1, s)
            below[:, t] = s.cash < 0
            if offer:
                action[:, t, 0] = np.where(s.mode == offer[0], 2 if offer[0] == C.PART_TIME else 3, 0)
            continue
        a, _ = policy.act(s, streams)
        if offer:
            a[:, 0] = 0
        s, info = world.step(s, a, streams, offer=offer)
        breaches[:, t] = info["credit_breach"]
        policy.observe(info, s)
        keep(t + 1, s)
        action[:, t] = info["action"]
        for k in EVENTS:
            events[k][:, t] = info[k]
        below[:, t] = info["below_zero"]
    return Paths(people=people, n_paths=n_paths, action=action, events=events, below_zero=below, traits=traits, credit_breach=breaches, **rec)


# ------------------------------------------------------------------------------- summaries
def fan(x, qs=(5, 25, 50, 75, 95)):
    return np.percentile(x, qs, axis=0)


def goal_met(paths: Paths):
    """Per path: high goal met at its deadline (Dec 2029), medium goal at Dec 2032 (nan if none)."""
    p = paths.people.iloc[0]
    k = C.T_HIGH_GOAL + 1
    have = paths.cash[:, k] if p["high_goal"] == "Emergency savings" else paths.nw[:, k]
    high = (have >= p["high_target"]) & ~paths.credit_breach[:, :k].any(1)
    med = np.full(len(high), np.nan)
    if pd.notna(p["med_target"]):
        med = ((paths.nw[:, C.T_MED_GOAL + 1] >= p["high_target"] + p["med_target"])
               & ~paths.credit_breach[:, :C.T_MED_GOAL + 1].any(1)).astype(float)
    return high, med


MODE_WORD = {C.STUDENT: "studying", C.PART_TIME: "part-time", C.FULL_TIME: "full-time", C.NOT_WORKING: "not working"}


def story_of(paths: Paths, i: int):
    """(label, year of the first work change or None) for one path; spells under 6 months are skipped."""
    m = paths.mode[i]
    seq, first, last, run_start = [], None, None, 0
    for t in range(len(m) + 1):
        if t == len(m) or m[t] != m[run_start]:
            if t - run_start >= 6 or (run_start == 0) or t == len(m):
                if m[run_start] != last:
                    if last is not None and first is None:
                        first = C.START_YEAR + run_start // 12
                    seq.append(MODE_WORD[int(m[run_start])])
                    last = m[run_start]
            run_start = t
    t0, t1 = paths.tier[i, 0], paths.tier[i, -1]
    home = "" if t0 == t1 else (" · ends in a bigger flat" if t1 > t0 else " · ends in a cheaper flat")
    fam = "".join(f" · {w}" for k, w in (("partner", "partner moves in"), ("child", "a child"),
                                         ("separation", "separates")) if paths.events[k][i].any())
    return " → ".join(seq) + home + fam, first


def stories(paths: Paths, top=5):
    """The most common life stories: work-mode sequence, housing direction, family events."""
    told = [story_of(paths, i) for i in range(len(paths.cash))]
    labels = [s for s, _ in told]
    high, _ = goal_met(paths)
    nw = paths.nw[:, -1]
    out = []
    for label, n in Counter(labels).most_common(top):
        idx = np.array([i for i, l in enumerate(labels) if l == label])
        years = [told[i][1] for i in idx if told[i][1] is not None]
        out.append(dict(story=label, share=n / len(labels), median_final_net_worth=float(np.median(nw[idx])),
                        high_goal_odds=float(high[idx].mean()), shortfall_odds=float(paths.below_zero[idx].any(1).mean()),
                        typical_change_year=int(np.median(years)) if years else None, example=int(idx[0])))
    return out


def lasting_shortfall(paths: Paths, months=3):
    """Per path: cash below zero for at least `months` months in a row."""
    b = paths.below_zero.astype(int)
    run = np.zeros(len(b), int)
    best = np.zeros(len(b), int)
    for t in range(b.shape[1]):
        run = (run + 1) * b[:, t]
        best = np.maximum(best, run)
    return best >= months


def path_features(paths: Paths) -> dict:
    a, ev = paths.action, paths.events
    started = paths.mode[:, 0]
    return {
        "starts a master's": (a[:, :, 0] == 1).any(1),
        "goes part-time": ((paths.mode == C.PART_TIME).any(1)) & (started != C.PART_TIME),
        "leaves paid work before 65": ((a[:, :, 0] == 4) & (paths.age[:, :-1] < 65)).any(1),
        "loses a job": ev["job_loss"].any(1),
        "moves somewhere cheaper": (a[:, :, 1] == -1).any(1),
        "moves somewhere bigger": (a[:, :, 1] == 1).any(1),
        "raises lifestyle": (paths.lifestyle == 2).any(1),
        "cuts lifestyle": (paths.lifestyle == 0).any(1),
        "lives through a recession": (paths.econ == 2).any(1),
        "has a child": ev["child"].any(1),
    }


def turning_points(paths: Paths, top=6):
    high, _ = goal_met(paths)
    nw = paths.nw[:, -1]
    rows = []
    for name, f in path_features(paths).items():
        share = f.mean()
        if 0.03 <= share <= 0.97:
            rows.append(dict(feature=name, share=float(share),
                             high_goal_with=float(high[f].mean()), high_goal_without=float(high[~f].mean()),
                             median_nw_with=float(np.median(nw[f])), median_nw_without=float(np.median(nw[~f]))))
    rows.sort(key=lambda r: -abs(r["median_nw_with"] - r["median_nw_without"]))
    return rows[:top]


def moment_masks(paths: Paths) -> dict:
    """(N, T) masks, one per big decision or life event, true in the month it happens."""
    m0, m1 = paths.mode[:, :-1], paths.mode[:, 1:]
    idle = np.isin(m0, [C.STUDENT, C.NOT_WORKING])
    work0, work1 = np.isin(m0, [C.PART_TIME, C.FULL_TIME]), np.isin(m1, [C.PART_TIME, C.FULL_TIME])
    ev, age = paths.events, paths.age[:, :-1]
    stop = work0 & (m1 == C.NOT_WORKING) & ~ev["job_loss"]
    tier0, tier1 = paths.tier[:, :-1], paths.tier[:, 1:]
    return {
        ("decision", "Starts a master's"): paths.action[:, :, 0] == 1,
        ("decision", "Takes a job alongside studies"): ev["side_job"],
        ("event", "Finishes studies"): ev["graduated"],
        ("decision", "Starts a full-time job"): idle & (m1 == C.FULL_TIME),
        ("decision", "Starts a part-time job"): idle & (m1 == C.PART_TIME),
        ("decision", "Moves up to full-time"): (m0 == C.PART_TIME) & (m1 == C.FULL_TIME),
        ("decision", "Cuts back to part-time"): (m0 == C.FULL_TIME) & (m1 == C.PART_TIME),
        ("event", "Loses a job"): ev["job_loss"],
        ("decision", "Stops working"): stop & (age < 60),
        ("decision", "Retires"): stop & (age >= 60),
        ("decision", "Moves to a bigger home"): tier1 > tier0,
        ("decision", "Moves somewhere cheaper"): tier1 < tier0,
        ("event", "Partner moves in"): ev["partner"],
        ("event", "Has a child"): ev["child"],
        ("event", "Separates"): ev["separation"],
    }


def key_moments(paths: Paths, top=6, min_share=0.10) -> list:
    """The big decisions and life events that happen in at least `min_share` of lives: how often and when.

    share: lives where it happens at least once; typical_year: median year it first happens;
    share_by_goal_deadline: lives where it has happened by December 2029. Ordered by typical year.
    """
    deadline = C.T_HIGH_GOAL + 1
    rows = []
    for (kind, label), mask in moment_masks(paths).items():
        ever = mask.any(1)
        share = float(ever.mean())
        if share < min_share:
            continue
        first = mask.argmax(1)[ever]
        rows.append(dict(moment=label, kind=kind, share=round(share, 4),
                         typical_year=int(C.START_YEAR + np.median(first) // 12),
                         share_by_goal_deadline=round(float((mask[:, :deadline].any(1)).mean()), 4)))
    rows = sorted(rows, key=lambda r: -r["share"])[:top]
    return sorted(rows, key=lambda r: (r["typical_year"], -r["share"]))


def yearly_action_rates(paths: Paths) -> dict:
    """Share of paths taking each kind of action per year (for plausibility checks)."""
    a = paths.action
    yrs = a.shape[1] / 12
    return {"status change": float((a[:, :, 0] > 0).sum(1).mean() / yrs),
            "start a master's": float((a[:, :, 0] == 1).sum(1).mean() / yrs),
            "housing move": float((a[:, :, 1] != 0).sum(1).mean() / yrs),
            "lifestyle change": float((a[:, :, 2] != 0).sum(1).mean() / yrs),
            "asset move": float((a[:, :, 3] != 0).sum(1).mean() / yrs)}


def summarize(paths: Paths) -> dict:
    high, med = goal_met(paths)
    years = np.arange(C.START_YEAR, C.START_YEAR + C.HORIZON // 12)
    below_year = paths.below_zero.reshape(len(paths.cash), -1, 12).any(2)
    return {
        "months": [C.month_label(t) for t in range(C.HORIZON + 1)],
        "fan_cash": fan(paths.cash).round(0).tolist(),
        "fan_net_worth": fan(paths.nw).round(0).tolist(),
        "fan_income": fan(paths.income).round(0).tolist(),
        "high_goal_odds": float(high.mean()),
        "medium_goal_odds": None if np.isnan(med).all() else float(np.nanmean(med)),
        "shortfall_odds": float(paths.below_zero.any(1).mean()),
        "credit_breach_odds": float(paths.credit_breach.any(1).mean()),
        "lasting_shortfall_odds": float(lasting_shortfall(paths).mean()),
        "median_life_utility": float(np.median(life_utility(paths))),
        "shortfall_by_year": dict(zip(years.tolist(), below_year.mean(0).round(4).tolist())),
        "final_net_worth": dict(zip(["p5", "p25", "p50", "p75", "p95"], fan(paths.nw[:, -1:])[:, 0].round(0).tolist())),
        "mode_share_by_year": {MODE_WORD[m]: [round(float((paths.mode[:, 12 * y] == m).mean()), 4) for y in range(11)]
                               for m in range(4)},
        "stories": stories(paths),
        "key_moments": key_moments(paths),
        "turning_points": turning_points(paths),
        "action_rates_per_year": yearly_action_rates(paths),
    }


def spread_sources(world, people, persona, n_paths=600, seed=C.SEED, policy_cls=RulePolicy):
    """Spread of final net worth when only one source of randomness is left on."""
    groups = {"who they are (traits)": ["traits"], "what they choose": ["policy"],
              "luck (events, flows, markets)": ["events", "flow", "market"], "what the model does not know": ["model"]}
    every = sum(groups.values(), [])

    def width(shared):
        nw = simulate(world, people, persona, n_paths, seed, shared=shared, policy_cls=policy_cls).nw[:, -1]
        return float(np.percentile(nw, 90) - np.percentile(nw, 10))
    full = width(())
    return {"all sources": full,
            **{name: width([s for s in every if s not in keep]) for name, keep in groups.items()}}


def compare(base, alternative):
    """Common-random-number comparison; interval is Monte Carlo error conditional on this model."""
    before, _ = goal_met(base)
    after, _ = goal_met(alternative)
    diff = after.astype(float) - before.astype(float)
    se = np.std(diff, ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")
    return {
        "high_goal_odds": [float(before.mean()), float(after.mean())],
        "shortfall_odds": [float(p.below_zero.any(1).mean()) for p in (base, alternative)],
        "credit_breach_odds": [float(p.credit_breach.any(1).mean()) for p in (base, alternative)],
        "median_final_net_worth": [float(np.median(p.nw[:, -1])) for p in (base, alternative)],
        "goal_change_pp": float(100 * diff.mean()),
        "goal_change_mc95_pp": [float(100 * np.clip(diff.mean() + sign * 1.96 * se, -1, 1))
                                for sign in (-1, 1)] if len(diff) > 1 else None,
    }


def default_what_ifs(person: pd.Series, persona: Persona) -> dict:
    """Up to three stated plans worth comparing for this person (each fires for sure)."""
    from dataclasses import replace

    from .policy.traits import Intention
    plans = []
    mode, age = int(person["mode"]), float(person["age"])
    if mode == C.STUDENT:
        plans.append(("Starts a master's at the opening review", Intention("status", 1, 1.0, when="2025-01")))
        plans.append(("Looks for a full-time job now", Intention("status", 3, 1.0, when="2025-01")))
    if mode != C.STUDENT and age <= 40:
        plans.append(("Starts a master's in January 2026", Intention("status", 1, 1.0, when="2026-01")))
    if mode == C.FULL_TIME and age < 60:
        plans.append(("Goes part-time from January 2026", Intention("status", 2, 1.0, when="2026-01")))
    if mode in (C.PART_TIME, C.FULL_TIME) and age >= 52:
        plans.append(("Leaves paid work at 60", Intention("status", 4, 1.0, min_age=60)))
    if int(person["tier"]) > (1 if int(person["household"]) == C.FAMILY else 0):
        plans.append(("Moves somewhere cheaper in January 2025", Intention("housing", -1, 1.0, when="2025-01")))
    plans.append(("Cuts lifestyle to lean now", Intention("spending", -1, 1.0, when="2025-01")))
    return {name: replace(persona, intentions=list(persona.intentions) + [it]) for name, it in plans[:3]}


def explore(profile_id=None, custom=None, persona: Persona | None = None, n_paths=2000, seed=C.SEED,
            world: World | None = None, what_ifs: dict | None = None, with_spread=True, policy_cls=RulePolicy,
            foresight_levels=None):
    """Everything the explorer shows for one person, JSON-ready (plus the raw paths)."""
    world = world or World.load()
    people = person_table(profile_id, custom)
    persona = persona or from_profile(people.iloc[0])
    paths = simulate(world, people, persona, n_paths, seed, policy_cls=policy_cls)
    p = people.iloc[0]
    out = {"person": {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in p.items()},
           "persona": {"traits": persona.traits, "temperature": persona.temperature, "spread": persona.spread,
                       "intentions": [vars(i) if not isinstance(i, dict) else i for i in persona.intentions],
                       "policy": policy_cls.name},
           "n_paths": n_paths, "seed": seed, **summarize(paths)}
    if what_ifs:
        out["what_ifs"] = {}
        for name, alt in what_ifs.items():
            alt_paths = simulate(world, people, alt, n_paths, seed, policy_cls=policy_cls)   # same seed: common random numbers
            out["what_ifs"][name] = compare(paths, alt_paths)
            out["what_ifs"][name]["fan_net_worth"] = fan(alt_paths.nw).round(0).tolist()
    if with_spread:
        out["spread_sources"] = spread_sources(world, people, persona, seed=seed, policy_cls=policy_cls)
    if foresight_levels:
        from .policy.foresight import at_level
        out["foresight"] = {}
        for lv in foresight_levels:                                   # same seed: common random numbers
            fp = simulate(world, people, persona, n_paths, seed, policy_cls=at_level(lv))
            h, _ = goal_met(fp)
            out["foresight"][f"{lv:g}"] = {
                "high_goal_odds": float(h.mean()), "shortfall_odds": float(fp.below_zero.any(1).mean()),
                "lasting_shortfall_odds": float(lasting_shortfall(fp).mean()),
                "median_final_net_worth": float(np.median(fp.nw[:, -1])),
                "median_life_utility": float(np.median(life_utility(fp))),
                "top_story": stories(fp, 1)[0], "fan_net_worth_median": fan(fp.nw, (50,))[0].round(0).tolist(),
                "action_rates_per_year": yearly_action_rates(fp)}
    return out, paths
