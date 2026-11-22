"""Learned market probabilities: job offers, master's admission and flats.

The ledger never shows anyone getting a job, a study place or a flat, so these three probabilities
cannot be learned from it. They come from `twin_game_theory/world`: a simulated Swiss society
(societies of 200 to 2,000 people competing for limited openings, calibrated to SECO and BFS
statistics) on which three small models were fitted by maximum likelihood:

    P(job offer)   complementary log-log in degree, grades, months without work, pay level,
                   log(applications), log(market tightness), economy, student
    P(admitted)    logit in programme, grades, log(tightness)
    P(flat)        logit in target tier, downsizing, log(income / new rent), log(tightness),
                   once the landlord's rule is met: rent <= income / 3, or 12 months' rent saved,
                   or a student

The person never sees the market's tightness theta (openings per application), so each
probability is averaged over the tightness the simulated society produced in that economy (mean
field). Each path uses one of the five bootstrap refits, picked by its ensemble member, so the
market's model uncertainty moves together with the flow network's ("what the model does not know").

This module only reads `world_model.json`; rebuild that file with `python -m twin_game_theory.world.build`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import expit

NONE, BACHELOR, MASTER, PHD = range(4)
EDUCATION = {"none": NONE, "bachelor": BACHELOR, "master": MASTER, "phd": PHD}

# Market assumptions used when priors.yaml has no `markets` block (it normally has one).
DEFAULTS = {
    "model": "twin_game_theory/world/world_model.json",
    "applications_per_month": 2,
    "grades": 0.0,
    "education_unknown": {"tertiary_share_by_pay_quintile": [0.25, 0.40, 0.55, 0.70, 0.80],
                          "degree_mix": [0.45, 0.45, 0.10]},
}
# Floors on the tightness before taking logs, as in twin_game_theory/world/transition.py.
THETA_FLOOR = {"job_offer": 1e-3, "admission": 1e-2, "flat": 1e-2}


def pay_level(rank) -> np.ndarray:
    """Pay level 1..5 of the market model: the quintile of the person's pay rank."""
    return 1 + np.clip(np.floor(np.asarray(rank, float) * 5), 0, 4).astype(int)


def draw_education(level, u_tertiary, u_degree, prior: dict) -> np.ndarray:
    """Highest degree for people whose education the data does not give (an assumption, per path)."""
    share = np.asarray(prior["tertiary_share_by_pay_quintile"], float)[np.asarray(level) - 1]
    edges = np.cumsum(prior["degree_mix"])[:-1]
    degree = BACHELOR + (np.asarray(u_degree)[:, None] > edges[None, :]).sum(1)
    return np.where(np.asarray(u_tertiary) < share, degree, NONE)


class MarketModel:
    def __init__(self, wm: dict):
        mk = wm["markets"]
        self.built = wm.get("built")
        self.models = {}
        for name in ("job_offer", "admission", "flat"):
            m = mk[name]
            self.models[name] = dict(link=m["link"], features=list(m["features"]),
                                     theta=m["features"].index("log_theta"),
                                     coef=np.array(m["coef_draws"], float))      # (refits, features)
        th = mk["theta"]
        log = lambda q, name: np.log(np.maximum(np.asarray(q, float), THETA_FLOOR[name]))  # noqa: E731
        self.log_theta_job = log([th["job"][str(e)] for e in range(3)], "job_offer")                 # (economy, Q)
        self.log_theta_master = log(th["admission"][str(MASTER)], "admission")                       # (Q,)
        self.log_theta_flat = log([[th["flat"][f"{h}_{e}"] for e in range(3)] for h in range(3)], "flat")  # (tier, economy, Q)

    @classmethod
    def load(cls, path) -> "MarketModel":
        return cls(json.loads(Path(path).read_text()))

    @property
    def n_refits(self) -> int:
        return len(self.models["job_offer"]["coef"])

    # --------------------------------------------------------------------------- core
    def _design(self, name, n, **cols):
        """Columns in the model's own feature order; the tightness column is filled in later."""
        out = []
        for f in self.models[name]["features"]:
            if f == "const":
                out.append(np.ones(n))
            elif f == "log_theta":
                out.append(np.zeros(n))
            else:
                out.append(np.broadcast_to(np.asarray(cols[f], float), (n,)))
        return np.column_stack(out)

    def _prob(self, name, X, log_theta, refit):
        """Mean over the tightness quantiles of link(x b + b_theta log theta). X (N, d), log_theta (N, Q)."""
        m = self.models[name]
        B = m["coef"][np.asarray(refit, int) % len(m["coef"])]
        j = m["theta"]
        eta = (X * B).sum(1)[:, None] + B[:, j][:, None] * log_theta
        p = -np.expm1(-np.exp(np.clip(eta, -30, 4))) if m["link"] == "cloglog" else expit(eta)
        return p.mean(1)

    # ------------------------------------------------------------------------ markets
    def p_offer(self, edu, grades, months_out, level, apps, econ, student, refit):
        """Chance a searcher gets at least one job offer this month."""
        edu, econ = np.asarray(edu), np.asarray(econ, int)
        n = len(edu)
        X = self._design("job_offer", n, edu_bachelor=edu == BACHELOR, edu_master=edu == MASTER, edu_phd=edu == PHD,
                         grades=grades, u_years=np.minimum(months_out, 24) / 12, k_minus_1=np.asarray(level) - 1,
                         log_apps=np.log(np.maximum(apps, 1)), inflation=econ == 1, recession=econ == 2,
                         student=student)
        return self._prob("job_offer", X, self.log_theta_job[econ], refit)

    def p_admission(self, grades, refit):
        """Chance an application to a master's is accepted."""
        n = len(np.atleast_1d(refit))
        X = self._design("admission", n, phd=0.0, grades=grades)
        return self._prob("admission", X, np.broadcast_to(self.log_theta_master, (n, len(self.log_theta_master))), refit)

    def p_flat(self, target, current, income, new_rent, cash, student, econ, refit):
        """Chance of getting a flat of tier `target` this month; zero if the landlord's rule fails."""
        target, econ = np.asarray(target, int), np.asarray(econ, int)
        new_rent = np.maximum(np.asarray(new_rent, float), 1.0)
        income = np.asarray(income, float)
        allowed = (new_rent <= income / 3) | (np.asarray(cash) >= 12 * new_rent) | np.asarray(student, bool)
        n = len(target)
        X = self._design("flat", n, to_small=target == 1, to_family=target == 2,
                         downsizing=target < np.asarray(current), log_income_over_rent=np.log(np.maximum(income, 1.0) / new_rent))
        return np.where(allowed, self._prob("flat", X, self.log_theta_flat[target, econ], refit), 0.0)
