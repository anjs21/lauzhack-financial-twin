"""Carry out a plan in the simulator: how often each step really happens, when, and what follows.

The predictor (persona habits + world model) runs the plan over many lives. Each step reports
the share of lives where it was carried out and the typical year, the share where it was due
but never possible (blocked), the share where the person was asked and declined, the share where they
tried but the world turned it down (no admission, no flat found), and how often its trigger came up.
"""
from __future__ import annotations

import copy

import numpy as np

from .. import config as C
from ..explore import MODE_WORD, Paths, simulate, summarize
from ..policy.rules import RulePolicy
from .schema import EVENT_TEXT, NOOP, OPTION_TEXT, LifePlan

TIER_WORD = ["a shared flat", "a small flat", "a family-size flat"]
WORLD_SAYS_NO = {"housing": "no flat found before the next review, or the landlord's rule said no: rent at most "
                            "a third of income, or 12 months' rent saved",
                 "status": "no admission, or still looking for a job"}
BLOCKED_WHY = {
    ("status", "start_masters"): "a master's needs no master's degree yet, age 45 or under, and any current "
                                 "course in its last month",
    ("status", "part_time"): "not possible while already part-time or already looking for part-time work, or from 70",
    ("status", "full_time"): "not possible while already full-time or already looking for full-time work, or from 70",
    ("status", "stop_working"): "only possible while working",
    ("housing", "cheaper"): "already in the cheapest home allowed (a family needs at least a small flat)",
    ("housing", "bigger"): "already in a family-size flat",
    ("spending", "cut"): "lifestyle is already lean",
    ("spending", "raise"): "lifestyle is already comfortable",
    ("assets", "sell_to_buffer"): "only when cash is below the buffer and there are investments to sell",
    ("assets", "invest_surplus"): "only when cash is above the buffer",
}


def with_plan(persona, plan: LifePlan):
    p = copy.deepcopy(persona)
    p.intentions = plan.intentions()
    return p


def run(world, people, persona, plan: LifePlan, n_paths=300, seed=C.SEED, policy_cls=RulePolicy) -> Paths:
    return simulate(world, people, with_plan(persona, plan), n_paths, seed, policy_cls=policy_cls)


def step_report(plan: LifePlan, paths: Paths) -> list[dict]:
    log = paths.plan_log
    out = []
    for s in plan.steps:
        k = log["ids"].index(s.id)
        done, due, open_, tried = log["done_at"][k], log["was_due"][k], log["was_open"][k], log["tried"][k]
        taken = done >= 0
        row = {"id": s.id, "label": s.label, "head": s.head, "option": s.option,
               "taken_share": round(float(taken.mean()), 3),
               "typical_year": int(C.START_YEAR + np.median(done[taken]) // 12) if taken.any() else None,
               "blocked_share": round(float((due & ~open_).mean()), 3),
               "declined_share": round(float((open_ & ~tried).mean()), 3),
               "turned_down_share": round(float((tried & ~taken).mean()), 3),
               "trigger_share": round(float(due.mean()), 3)}
        if row["blocked_share"] >= 0.05:                         # the UI explains it; the agent is warned from 20%
            row["blocked_because"] = BLOCKED_WHY.get((s.head, s.option), "its option was not open")
        out.append(row)
    return out


def off_plan_changes(plan: LifePlan, paths: Paths) -> float:
    """Changes of work or home per life that no step asked for (habits, within spontaneity)."""
    a = paths.action
    changes = ((a[:, :, 0] > 0) | (a[:, :, 1] != 0)).sum(1)
    log = paths.plan_log
    planned = sum((log["done_at"][log["ids"].index(s.id)] >= 0).astype(int)
                  for s in plan.steps if s.head in ("status", "housing") and s.option not in ("continue", "stay"))
    return round(float(np.maximum(changes - planned, 0).mean()), 3)


def outcomes(summary: dict) -> dict:
    keep = ("high_goal_odds", "medium_goal_odds", "shortfall_odds", "lasting_shortfall_odds", "credit_breach_odds",
            "final_net_worth")
    out = {k: summary[k] for k in keep}
    out["key_moments"] = summary["key_moments"]
    out["stories"] = [{k: s[k] for k in ("story", "share")} for s in summary["stories"][:3]]
    return out


def money_trouble(paths: Paths) -> dict | None:
    """Where the first credit-limit breach happens: work situation, home and typical year (None if rare)."""
    hit = paths.credit_breach.any(1)
    if hit.mean() < 0.05:
        return None
    first = paths.credit_breach.argmax(1)[hit]
    rows = np.flatnonzero(hit)
    mode = paths.mode[rows, first]
    return {"share_of_lives": round(float(hit.mean()), 3),
            "typical_year": int(C.START_YEAR + np.median(first) // 12),
            "while": {MODE_WORD[m]: round(float((mode == m).mean()), 3) for m in range(4) if (mode == m).any()},
            "income_then": round(float(np.median(paths.income[rows, first])), 0),
            "home_then": TIER_WORD[int(np.median(paths.tier[rows, first]))]}


def evaluate(world, people, persona, plan: LifePlan, n_paths=300, seed=C.SEED, policy_cls=RulePolicy):
    """(evaluation for the agent and the UI, full summary with fan charts, the paths)."""
    paths = run(world, people, persona, plan, n_paths, seed, policy_cls)
    summary = summarize(paths)
    ev = {"n_lives": int(len(paths.cash)), "steps": step_report(plan, paths),
          "off_plan_changes_per_life": off_plan_changes(plan, paths), "outcomes": outcomes(summary),
          "money_trouble": money_trouble(paths),
          "high_goal_counts": "cash" if people.iloc[0]["high_goal"] == "Emergency savings" else "net worth"}
    return ev, summary, paths


def feedback(plan: LifePlan, ev: dict) -> list[str]:
    """Plain warnings the planner should look at; feasibility and consistency, not optimisation."""
    notes = []
    by_id = {s.id: s for s in plan.steps}
    for r in ev["steps"]:
        s = by_id[r["id"]]
        if r.get("blocked_because") and r["blocked_share"] >= 0.2:
            notes.append(f"'{s.id}' could not happen in {r['blocked_share']:.0%} of lives when due: "
                         f"{r['blocked_because']}. Move its window, add a step that makes it possible, or drop it.")
        elif r["turned_down_share"] >= 0.25:
            notes.append(f"'{s.id}' was tried in {r['turned_down_share'] + r['taken_share']:.0%} of lives but the world "
                         f"turned it down in {r['turned_down_share']:.0%} ({WORLD_SAYS_NO.get(s.head, 'not granted')}); "
                         "it is retried each January in its window. An earlier trigger or another lever may be needed.")
        elif s.event and r["trigger_share"] < 0.05:
            notes.append(f"'{s.id}' waits for '{EVENT_TEXT[s.event]}', which happened in only "
                         f"{r['trigger_share']:.0%} of lives; fine as a fallback, but it rarely matters.")
        elif not s.contingent and s.after is None and r["taken_share"] < 0.5 * s.commitment:
            notes.append(f"'{s.id}' happened in only {r['taken_share']:.0%} of lives against a commitment of "
                         f"{s.commitment:.0%}: check its window and conditions.")
        elif s.after and r["trigger_share"] < 0.3:
            notes.append(f"'{s.id}' waits for '{s.after}', which rarely happened, so '{s.id}' was rarely due "
                         f"({r['trigger_share']:.0%} of lives).")
    if ev.get("high_goal_counts") == "cash" and any(s.option == "invest_surplus" for s in plan.steps):
        notes.append("Your high-priority goal (emergency savings) counts cash only, and 'invest_surplus' moves cash "
                     "into investments: that step works against the goal. Cash grows by spending less.")
    o, trouble = ev["outcomes"], ev.get("money_trouble")
    if o["credit_breach_odds"] > 0.25 and trouble:
        where = ", ".join(f"{share:.0%} {mode}" for mode, share in sorted(trouble["while"].items(), key=lambda x: -x[1]))
        notes.append(f"In {o['credit_breach_odds']:.0%} of lives cash fell past the credit limit, typically in "
                     f"{trouble['typical_year']}, while {where}, on an income of about CHF {format(trouble['income_then'], ',.0f').replace(',', "'")} "
                     f"a month in {trouble['home_then']}. Investing does not help here: only more income, lower rent "
                     "(a cheaper home), a leaner lifestyle, more cash before that point, or changing that step does; "
                     "a fallback triggered by cash_months_below (e.g. 1) can say what this person does when cash runs "
                     "low. Would this person accept the risk, or change something?")
    elif o["lasting_shortfall_odds"] > 0.3:
        notes.append(f"In {o['lasting_shortfall_odds']:.0%} of lives cash stayed below zero for 3+ months: "
                     "does the plan say what this person does then?")
    return notes


def option_rules() -> str:
    lines = []
    for head, opts in OPTION_TEXT.items():
        lines.append(f"- {head}: " + "; ".join(f'"{o}" = {t}' for o, t in opts.items() if o not in NOOP))
    return "\n".join(lines)
