"""Learned market chances, copied from twin_game_theory/world/transition.py and vectorised over paths.

twin_game_theory simulated societies of 200, 1,000 and 2,000 people (90,000 in all) competing for limited
job openings, study places and flats, calibrated to Swiss statistics (SECO unemployment, BFS
bachelor's -> master's and graduate unemployment, the Swiss moving rate), and fitted three models by
maximum likelihood with 5 bootstrap refits. The fitted coefficients live in artifacts/markets.json (a copy of
twin_game_theory/world/world_model.json). The individual never sees the market's tightness theta,
so each chance is averaged over the theta values that world produced in that situation, exactly as
TransitionModel._p does.

    job offer while searching   cloglog(degree, grades, years out of work, pay level, applications, theta, economy, student)
    master's admission          logit(programme, grades, theta)
    moving to another flat      logit(target size, downsizing, log(income / new rent), theta), after the landlord rule:
                                new rent <= income / 3, or 12 months of rent in cash, or a student

Mapping from this model's state (assumptions, not in either dataset):
    degree       a finished master's when `degree` is set and the person is no longer studying; otherwise a
                 bachelor's (the data say nothing about education). Change with markets.default_education.
    grades       0 (average), applications 2 a month (twin_game_theory's default habit)
    years out    months in the current spell without work (tau while not working)
    pay level    quintile of the pay rank (twin_game_theory's levels 1..5 are data quintiles)
    refit        each path uses bootstrap refit (ensemble member mod 5), so model uncertainty matches the flows
"""
from __future__ import annotations

import json

import numpy as np
from scipy.special import expit

from .. import config as C

PATH = C.ARTIFACTS / "markets.json"
EDU = {"none": 0, "bachelor": 1, "master": 2, "phd": 3}
MASTER = 2                      # programme code in twin_game_theory


class Markets:
    def __init__(self, path=PATH, default_education="bachelor", grades=0.0, applications=2):
        wm = json.loads(path.read_text())
        self.mk = wm["markets"]
        self.n_draws = len(self.mk["job_offer"]["coef_draws"])
        self.default_edu = EDU[default_education]
        self.grades, self.apps = float(grades), float(applications)

    # --------------------------------------------------------------------------- shared
    def _prob(self, name, X, keys, table, draw, floor):
        """Per-row chance, averaged over the theta values recorded for each row's situation `keys`."""
        m = self.mk[name]
        B = np.asarray(m["coef_draws"], float)[np.asarray(draw) % self.n_draws]          # (n, F)
        j = m["features"].index("log_theta")
        base = np.einsum("nf,nf->n", X, B)                                               # theta column of X is 0
        out = np.empty(len(X))
        for key in np.unique(keys):
            rows = keys == key
            th = np.log(np.maximum(np.asarray(table[str(key)], float), floor))            # (K,)
            eta = base[rows, None] + B[rows, j, None] * th[None, :]
            p = -np.expm1(-np.exp(np.clip(eta, -30, 4))) if m["link"] == "cloglog" else expit(eta)
            out[rows] = p.mean(1)
        return out

    # --------------------------------------------------------------------------- the three markets
    def p_job_offer(self, edu, u_months, rank, econ, student, draw):
        n = len(edu)
        k_minus_1 = np.clip((np.asarray(rank) * 5).astype(int), 0, 4)
        X = np.column_stack([np.ones(n), edu == 1, edu == 2, edu == 3, np.full(n, self.grades),
                             np.minimum(u_months, 24) / 12, k_minus_1, np.full(n, np.log(self.apps)), np.zeros(n),
                             econ == 1, econ == 2, student]).astype(float)
        return self._prob("job_offer", X, np.asarray(econ), self.mk["theta"]["job"], draw, 1e-3)

    def p_master_admission(self, draw):
        n = len(draw)
        X = np.column_stack([np.ones(n), np.zeros(n), np.full(n, self.grades), np.zeros(n)])
        return self._prob("admission", X, np.full(n, MASTER), self.mk["theta"]["admission"], draw, 1e-2)

    def p_flat(self, target, tier, income, new_rent, cash, student, econ, draw):
        n = len(target)
        rule = (new_rent <= income / 3) | (cash >= 12 * new_rent) | student                 # landlord's rule
        X = np.column_stack([np.ones(n), target == 1, target == 2, target < tier,
                             np.log(np.maximum(income, 1.0) / new_rent), np.zeros(n)]).astype(float)
        keys = np.char.add(np.char.add(np.asarray(target).astype(str), "_"), np.asarray(econ).astype(str))
        return np.where(rule, self._prob("flat", X, keys, self.mk["theta"]["flat"], draw, 1e-2), 0.0)

    def education(self, degree, studying):
        """Highest finished degree: a master's once it is done, else the default (bachelor's)."""
        return np.where(np.asarray(degree) & ~np.asarray(studying), EDU["master"], self.default_edu)
