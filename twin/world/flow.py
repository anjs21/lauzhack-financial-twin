"""The learned part of the world: next month's net cash flow as a distribution.

A mixture density network (MLP 2 x 64, 5 Gaussian components) maps a person-month to a
distribution over the month's net flow in real CHF thousands, so December bonuses,
February/August bills and student fees can sit in separate components. Five seeds form
an ensemble; each simulated path keeps one member, so model uncertainty shows up in the
spread. A per-person offset (shrunk mean residual) carries what the features cannot see,
such as one person's unusually high living costs.

Inputs never include the calendar year: drift enters through net_income and rent.
"""
from __future__ import annotations

import copy
import json
import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .. import config as C

FEATURES = (["student", "part_time", "full_time", "not_working", "self_employed", "pensioner",
             "log_income", "log_rent"] + [f"city_{c}" for c in C.CITIES] + [f"month_{m}" for m in range(1, 13)]
            + [f"{md}_x_month_{m}" for md in C.MODES for m in range(1, 13)])
SCALE = 1000.0   # model works in CHF thousands
# Mode x month inputs let 39 students' February/September fees and employees' December
# bonus be learned directly; without them the network under-fits these rare spikes.


def features(mode, self_employed, pensioner, income_real, rent_real, city_idx, moy) -> np.ndarray:
    mode = np.asarray(mode, int)
    n = len(mode)
    X = np.zeros((n, len(FEATURES)), np.float32)
    X[np.arange(n), mode] = 1
    X[:, 4] = np.asarray(self_employed, bool) & (mode == C.FULL_TIME)
    X[:, 5] = np.asarray(pensioner, bool) & (mode == C.NOT_WORKING)
    X[:, 6] = np.log1p(np.maximum(np.asarray(income_real, float), 0) / SCALE)
    X[:, 7] = np.log(np.maximum(np.asarray(rent_real, float), 50) / SCALE)
    X[np.arange(n), 8 + np.asarray(city_idx, int)] = 1
    moy = np.asarray(moy, int)
    X[np.arange(n), 8 + len(C.CITIES) + moy - 1] = 1
    X[np.arange(n), 8 + len(C.CITIES) + 12 + mode * 12 + moy - 1] = 1
    return X


def frame_features(df: pd.DataFrame) -> np.ndarray:
    return features(df["mode"], df["self_employed"], df["pensioner"], df["income_real"],
                    df["rent_real"], df["city_idx"], df["moy"])


class MDN(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64, k: int = 5):
        super().__init__()
        self.k = k
        self.body = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.head = nn.Linear(hidden, 3 * k)

    def forward(self, x):
        o = self.head(self.body(x))
        k = self.k
        return torch.log_softmax(o[:, :k], -1), o[:, k:2 * k], F.softplus(o[:, 2 * k:]) + 0.01


def mdn_nll(logw, mu, sigma, y):
    z = (y[:, None] - mu) / sigma
    lp = logw - torch.log(sigma) - 0.5 * z * z - 0.5 * math.log(2 * math.pi)
    return -torch.logsumexp(lp, -1)


def train_member(X, y, groups, seed, hidden=64, k=5, lr=1e-3, weight_decay=1e-4, batch=1024,
                 max_epochs=500, patience=80, val_share=0.1):
    """Train one MDN; early stopping on a held-out 10% of *people*, chosen by the seed."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    people = np.unique(groups)
    val_people = rng.choice(people, max(1, int(len(people) * val_share)), replace=False)
    vm = np.isin(groups, val_people)
    Xt, yt = torch.tensor(X[~vm]), torch.tensor(y[~vm], dtype=torch.float32)
    Xv, yv = torch.tensor(X[vm]), torch.tensor(y[vm], dtype=torch.float32)
    model = MDN(X.shape[1], hidden, k)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad, epoch = float("inf"), None, 0, 0
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            loss = mdn_nll(*model(Xt[idx]), yt[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = mdn_nll(*model(Xv), yv).mean().item()
        if v < best - 1e-4:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"seed": seed, "val_nll": best, "epochs": epoch + 1}


class FlowModel:
    """Ensemble of MDNs plus per-person offsets."""

    def __init__(self, members, info=None, hidden=64, k=5):
        self.members, self.info, self.hidden, self.k = members, info or [], hidden, k
        self.offsets = pd.DataFrame(columns=["b", "var"])   # real CHF thousands
        self.sigma_b2 = 0.0

    @classmethod
    def fit(cls, X, y, groups, seeds=(0, 1, 2, 3, 4), **kw):
        out = [train_member(X, y, groups, C.SEED + s, **kw) for s in seeds]
        return cls([m for m, _ in out], [i for _, i in out], kw.get("hidden", 64), kw.get("k", 5))

    def params(self, X, m):
        with torch.no_grad():
            logw, mu, sigma = self.members[m](torch.as_tensor(X))
        return logw.exp().numpy().astype(float), mu.numpy().astype(float), sigma.numpy().astype(float)

    def mean_var(self, X):
        """Predictive mean and variance of the ensemble (an equal mixture of members)."""
        means, seconds = [], []
        for m in range(len(self.members)):
            w, mu, s = self.params(X, m)
            means.append((w * mu).sum(1))
            seconds.append((w * (s ** 2 + mu ** 2)).sum(1))
        mean = np.mean(means, 0)
        return mean, np.mean(seconds, 0) - mean ** 2

    def sample(self, X, member, u, z):
        """One draw per row from its member's mixture: u picks the component, z the value."""
        y = np.empty(len(X))
        for m in range(len(self.members)):
            rows = np.flatnonzero(member == m)
            if len(rows):
                w, mu, s = self.params(X[rows], m)
                comp = (u[rows, None] > np.cumsum(w, 1)[:, :-1]).sum(1)
                r = np.arange(len(rows))
                y[rows] = mu[r, comp] + s[r, comp] * z[rows]
        return y

    def fit_offsets(self, X, y, groups):
        """Shrunk per-person mean residual (random intercept), method of moments."""
        mean, _ = self.mean_var(X)
        df = pd.DataFrame({"g": groups, "r": y - mean})
        agg = df.groupby("g").r.agg(["mean", "var", "count"])
        noise = (agg["var"] / agg["count"]).values
        self.sigma_b2 = float(max(1e-6, agg["mean"].var() - noise.mean()))
        kappa = self.sigma_b2 / (self.sigma_b2 + noise)
        self.offsets = pd.DataFrame({"b": kappa * agg["mean"].values, "var": kappa * noise}, index=agg.index)
        return self.offsets

    def offset_for(self, ids):
        """(b, var) per id; people without history get 0 with the population variance."""
        o = self.offsets.reindex(ids)
        return o["b"].fillna(0.0).values, o["var"].fillna(self.sigma_b2).values

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(self.members):
            torch.save(m.state_dict(), path / f"member_{i}.pt")
        self.offsets.to_csv(path / "offsets.csv")
        (path / "meta.json").write_text(json.dumps({"features": FEATURES, "hidden": self.hidden, "k": self.k,
                                                    "members": self.info, "sigma_b2": self.sigma_b2}, indent=1))

    @classmethod
    def load(cls, path):
        meta = json.loads((path / "meta.json").read_text())
        assert meta["features"] == FEATURES, "feature list changed; retrain with `python -m twin train`"
        members = []
        for i in range(len(meta["members"])):
            m = MDN(len(FEATURES), meta["hidden"], meta["k"])
            m.load_state_dict(torch.load(path / f"member_{i}.pt", weights_only=True))
            m.eval()
            members.append(m)
        fm = cls(members, meta["members"], meta["hidden"], meta["k"])
        fm.offsets = pd.read_csv(path / "offsets.csv", index_col=0)
        fm.sigma_b2 = meta["sigma_b2"]
        return fm


def crps(samples, y):
    """Sample-based CRPS per row: E|X - y| - 0.5 E|X - X'| (samples: n x K)."""
    s = np.sort(samples, 1)
    K = s.shape[1]
    term1 = np.abs(s - y[:, None]).mean(1)
    w = (2 * np.arange(1, K + 1) - K - 1) / (K * K)
    return term1 - (s * w).sum(1)
