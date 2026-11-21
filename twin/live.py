"""Persistent local decision history and paired previews for the browser explorer."""
import json
import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass

import numpy as np

from . import config as C
from .explore import goal_met, simulate
from .timeline import Timeline


@dataclass
class Journal:
    profile: str
    revision: int
    timeline: Timeline

    @property
    def current_month(self):
        return len(self.timeline.observations)


def read_journal(db, profile):
    with closing(sqlite3.connect(db)) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS entries (
                    profile TEXT, month INTEGER, kind TEXT, payload TEXT NOT NULL,
                    PRIMARY KEY(profile,month,kind))""")
        rows = con.execute("SELECT month,kind,payload FROM entries WHERE profile=? ORDER BY month", (profile,)).fetchall()
    timeline = Timeline()
    for month, kind, data in rows:
        if kind == "offer":
            timeline.offers[month] = tuple(json.loads(data))
        else:
            timeline.observations[month] = json.loads(data)
    return Journal(profile, len(rows), timeline)


def record(db, journal, month, kind, payload):
    if kind == "offer":
        mode, salary = payload
        if month != C.next_review(journal.current_month) or month >= C.HORIZON:
            raise ValueError("Offers take effect at the next available January review.")
        if mode not in (C.PART_TIME, C.FULL_TIME) or not math.isfinite(salary) or salary <= 0:
            raise ValueError("Choose a work mode and a positive monthly net salary.")
    elif kind == "observation":
        if month != journal.current_month or not 0 <= month < C.HORIZON:
            raise ValueError("Record consecutive closing months within the ten-year horizon.")
        allowed = {"cash", "investments", "rent", "income", "net_flow", "status"}
        if "cash" not in payload or set(payload) - allowed:
            raise ValueError("Supply closing cash and supported observation fields.")
        if "status" in payload and payload["status"] not in C.STATUS_TO_MODE:
            raise ValueError("Unknown employment status.")
        for k,v in payload.items():
            if k != "status" and (not math.isfinite(v) or (k in ("investments", "income") and v < 0)
                                  or (k == "rent" and v <= 0)):
                raise ValueError(f"Invalid {k} value.")
    else:
        raise ValueError("Unknown journal entry.")
    with closing(sqlite3.connect(db)) as con, con:
        con.execute("BEGIN IMMEDIATE")
        existing = con.execute("SELECT payload FROM entries WHERE profile=? AND month=? AND kind=?",
                               (journal.profile, month, kind)).fetchone()
        data = json.dumps(payload, sort_keys=True, allow_nan=False)
        if existing and existing[0] == data:
            return
        version = con.execute("SELECT COUNT(*) FROM entries WHERE profile=?", (journal.profile,)).fetchone()[0]
        if version != journal.revision or existing:
            raise ValueError("The saved history changed. Refresh before adding another entry.")
        con.execute("INSERT INTO entries VALUES (?,?,?,?)", (journal.profile, month, kind, data))


def branch(journal, mode, salary):
    month = C.next_review(journal.current_month)
    if month >= C.HORIZON or month in journal.timeline.offers:
        raise ValueError("This review already has an offer or lies beyond the horizon.")
    return Timeline({**journal.timeline.offers, month: (mode, salary)}, journal.timeline.observations)


def outcomes(paths):
    goal, _ = goal_met(paths)
    return dict(goal=float(goal.mean()), breach=float(paths.credit_breach.any(1).mean()),
                shortfall=float(paths.below_zero.any(1).mean()), wealth=float(np.median(paths.nw[:, -1])))


def salary_search(world, people, persona, journal, salaries, target, n_paths, seed):
    if not 0 <= target <= 1 or not salaries or any(not math.isfinite(v) or v <= 0 for v in salaries):
        raise ValueError("Supply positive salaries and a probability between 0 and 1.")
    if C.next_review(journal.current_month) > C.T_HIGH_GOAL:
        raise ValueError("The next review falls after this goal's deadline.")
    rows = []
    for salary in sorted(set(salaries)):
        timeline = branch(journal, C.PART_TIME, salary)
        paths = simulate(world, people, persona, n_paths, seed, timeline=timeline)
        rows.append(dict(salary=salary, **outcomes(paths)))
    minimum = next((r["salary"] for r in rows if r["goal"] >= target), None)
    return minimum, rows
