"""Tests for TransitionModel. Run: python -m twin_game_theory.world.test_transition (or pytest)."""
from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np

from . import spec as sp
from .transition import DISCRETE, Action, State, TransitionModel, alexia

M = TransitionModel()
RNG = np.random.default_rng(0)


def states():
    base = alexia()
    yield base                                                              # studying
    yield replace(base, W=sp.N, prog=sp.NONE, edu=sp.BACHELOR, u=1)         # just graduated
    yield replace(base, W=sp.N, prog=sp.NONE, edu=sp.MASTER, u=14, M=sp.RECESSION, k=2)
    yield replace(base, W=sp.E100, prog=sp.NONE, edu=sp.MASTER, k=3, H=1, F=sp.COUPLE, age=31, C=40000, I=20000)
    yield replace(base, W=sp.N, prog=sp.NONE, edu=sp.BACHELOR, pend=sp.P_JOB_OFFER, offer_k=2)
    yield replace(base, W=sp.E80, prog=sp.NONE, edu=sp.PHD, k=4, H=2, F=sp.FAMILY, age=38, C=-2000, I=5000)


def actions(s):
    al = M.allowed(s)
    for st, h, sp_, a in product(al["status"], al["housing"], al["spending"], al["assets"]):
        yield Action(st, h, sp_, a)


def test_probabilities_sum_to_one():
    for s in states():
        for a in actions(s):
            total = sum(M.outcome_probs(s, a).values())
            assert abs(total - 1) < 1e-9, (s, a, total)


def test_sampler_matches_probabilities():
    """Total variation between 4,000 samples and the exact distribution stays small."""
    for s in list(states())[:4]:
        for a in [Action("plus"), Action("stay", housing=1 if s.H < 2 else 0)] + (
                [Action("apply_master")] if "apply_master" in M.allowed(s)["status"] else []):
            if a.status not in M.allowed(s)["status"]:
                continue
            exact = M.outcome_probs(s, a)
            counts = {}
            for _ in range(4000):
                s2, _ = M.sample(s, a, RNG)
                key = tuple(getattr(s2, k) for k in DISCRETE)
                counts[key] = counts.get(key, 0) + 1
            keys = set(exact) | set(counts)
            tv = 0.5 * sum(abs(exact.get(k, 0) - counts.get(k, 0) / 4000) for k in keys)
            assert tv < 0.03, (s, a, tv)


def test_money_adds_up():
    for s in states():
        for a in list(actions(s))[:20]:
            s2, led = M.sample(s, a, RNG)
            flows = sum(v for k, v in led.items() if k != "market_return")
            assert abs(s2.C - (s.C + flows)) < 1e-6                        # cash identity
            assert abs(s2.I - (s.I - led["invest"] - led["sell"] + led["market_return"])) < 1e-6   # investment identity


def test_wealth_identity():
    """Cash + investments change only by real flows and market returns; transfers cancel out."""
    for s in states():
        for a in actions(s):
            s2, led = M.sample(s, a, RNG)
            real = sum(v for k, v in led.items() if k not in ("market_return", "invest", "sell"))
            assert abs((s2.C + s2.I) - (s.C + s.I + real + led["market_return"])) < 1e-6, (a, led)


def test_actions_respect_rules():
    s = alexia()
    assert "accept" not in M.allowed(s)["status"]                          # nothing to accept
    assert "apply_phd" not in M.allowed(replace(s, W=sp.N, edu=sp.BACHELOR, prog=sp.NONE))["status"]
    fam = replace(s, F=sp.FAMILY, H=1)
    assert -1 not in M.allowed(fam)["housing"]                             # a family needs at least a small flat


def test_directions():
    from dataclasses import asdict
    d = asdict(replace(alexia(), W=sp.N, prog=sp.NONE, edu=sp.BACHELOR, u=1))
    assert M.p_job_offer({**d, "M": sp.NORMAL}) > M.p_job_offer({**d, "M": sp.RECESSION})
    assert M.p_job_offer({**d, "edu": sp.MASTER}) > M.p_job_offer(d)
    assert M.p_job_offer({**d, "u": 18}) < M.p_job_offer(d)
    assert M.p_admission({**d, "grades": 1.0}, sp.PHD) > M.p_admission({**d, "grades": -1.0}, sp.PHD)


def test_validation_recorded():
    """Calibration error under 3 points on new societies of the training sizes and on a new persona mix.
    Societies larger than any in training (5,000 people) are an extrapolation check with a looser 4-point bound."""
    trained = max(int(n) for n in M.wm["training_society_sizes"])
    for test, markets in M.wm["validation"].items():
        size = int(test.split("_")[-2]) if test.endswith("_people") else 0
        bound = 4.0 if size > trained else 3.0
        for name, r in markets.items():
            assert r["ece_pts"] < bound, (test, name, r)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
