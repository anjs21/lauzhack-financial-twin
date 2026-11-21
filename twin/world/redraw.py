"""What changes when a person changes work mode or flat, learned by comparing people.

Nobody switches in the data, so these models read the answer off people who are already
in the new state, keeping the person's rank:

* IncomeModel - take-home pay by income kind. In the data pay depends on work mode only
  (not on age, city or rent), so a person keeps their pay rank u and reads their new pay
  off the new kind's distribution.
* RentModel   - log rent = level + tier + city + residual (R^2 0.66). A person keeps their
  residual (their taste for pricier flats) when they move to another tier.
All amounts are January-2025 CHF; the world scales them by the pay and price indices.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.special import ndtr, ndtri

from .. import config as C

# Income kinds: work mode, with "not working" split into benefits (< 65) and pensions.
KINDS = ["student", "part_time", "full_time", "benefit", "pension"]
STUDENT_K, PART_K, FULL_K, BENEFIT_K, PENSION_K = range(5)


def income_kind(mode, pensioner):
    mode = np.asarray(mode)
    kind = mode.copy()          # student 0, part-time 1, full-time 2, not working 3
    kind[(mode == C.NOT_WORKING) & np.asarray(pensioner, bool)] = PENSION_K
    return kind


class IncomeModel:
    def __init__(self, values: list):
        self.values = [np.sort(np.asarray(v, float)) for v in values]

    @classmethod
    def fit(cls, prof):
        kind = income_kind(prof["mode"].values, prof["pensioner"].values)
        return cls([prof["income0"].values[kind == k] for k in range(len(KINDS))])

    def _grid(self, k):
        v = self.values[k]
        return (np.arange(len(v)) + 0.5) / len(v), v

    def quantile(self, kind, u):
        kind, u = np.asarray(kind), np.clip(np.asarray(u, float), 1e-4, 1 - 1e-4)
        out = np.empty(len(u))
        for k in range(len(KINDS)):
            m = kind == k
            if m.any():
                pos, v = self._grid(k)
                out[m] = np.interp(u[m], pos, v)
        return out

    def rank(self, kind, y):
        kind, y = np.asarray(kind), np.asarray(y, float)
        out = np.empty(len(y))
        for k in range(len(KINDS)):
            m = kind == k
            if m.any():
                pos, v = self._grid(k)
                out[m] = np.interp(y[m], v, pos)
        return out

    def to_dict(self):
        return {"kinds": KINDS, "values": [v.round(2).tolist() for v in self.values]}

    @classmethod
    def from_dict(cls, d):
        return cls(d["values"])


def shift_rank(u, step=0.0, noise=0.0, rng=None, size=None):
    """Move a pay rank in normal-score space: + step, + N(0, noise)."""
    z = ndtri(np.clip(np.asarray(u, float), 1e-4, 1 - 1e-4)) + step
    if noise:
        z = z + noise * (rng.standard_normal(size if size is not None else np.shape(u)))
    return ndtr(z)


class RentModel:
    def __init__(self, level, tier, city, sigma):
        self.level, self.tier, self.city, self.sigma = float(level), np.asarray(tier), np.asarray(city), float(sigma)

    @classmethod
    def fit(cls, prof):
        n = len(prof)
        X = np.zeros((n, 1 + 2 + len(C.CITIES) - 1))
        X[:, 0] = 1                                         # tier 0 (shared flat) is the reference
        X[:, 1] = prof["tier"].values == 1
        X[:, 2] = prof["tier"].values == 2
        ci = prof["city_idx"].values
        for j in range(1, len(C.CITIES)):                   # city 0 (Basel) is the reference
            X[:, 2 + j] = ci == j
        y = np.log(prof["rent0"].values)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        return cls(beta[0], [0.0, beta[1], beta[2]], [0.0, *beta[3:]], resid.std())

    def z_of(self, rent, tier, city_idx):
        mu = self.level + self.tier[np.asarray(tier)] + self.city[np.asarray(city_idx)]
        return (np.log(np.asarray(rent, float)) - mu) / self.sigma

    def rent(self, tier, city_idx, z):
        return np.exp(self.level + self.tier[np.asarray(tier)] + self.city[np.asarray(city_idx)]
                      + self.sigma * np.asarray(z, float))

    def r2(self, prof):
        y = np.log(prof["rent0"].values)
        fit = self.level + self.tier[prof["tier"].values] + self.city[prof["city_idx"].values]
        return 1 - np.var(y - fit) / np.var(y)

    def to_dict(self):
        return {"level": self.level, "tier": self.tier.tolist(), "city": self.city.tolist(),
                "sigma": self.sigma, "cities": C.CITIES}

    @classmethod
    def from_dict(cls, d):
        return cls(d["level"], d["tier"], d["city"], d["sigma"])


def save(path, income: IncomeModel, rent: RentModel):
    path.write_text(json.dumps({"income": income.to_dict(), "rent": rent.to_dict()}))


def load(path):
    d = json.loads(path.read_text())
    return IncomeModel.from_dict(d["income"]), RentModel.from_dict(d["rent"])
