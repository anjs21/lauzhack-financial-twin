"""The learned market model inside the world: job offers, master's admission, flats.

Run: python -m pytest tests/test_market.py -q
"""
from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pytest

from twin import config as C
from twin.data import profiles
from twin.explore import person_table
from twin.world.env import Config, Streams, World
from twin.world.market import BACHELOR, MASTER, NONE, MarketModel, pay_level

pytestmark = pytest.mark.filterwarnings("ignore")
HAVE_MODEL = (C.ARTIFACTS / "flow" / "meta.json").exists()
needs_model = pytest.mark.skipif(not HAVE_MODEL, reason="train first: python -m twin train")


@pytest.fixture(scope="module")
def market():
    return MarketModel.load(C.ROOT / "twin_game_theory" / "world" / "world_model.json")


def custom(**kw):
    base = dict(name="Test", age=30, city="Bern", status="employed", household_type="single", income=6000,
                rent=1600, cash=20000, investments=0, debt=0, rate=0, high_goal="Emergency savings",
                high_target=20000)
    base.update(kw)
    return person_table(custom=base)


def run(world, people, n, seed, months, action_at=None):
    """Fixed actions: `action_at(t)` gives the (status, housing, spending, assets) chosen in month t."""
    streams = Streams(seed, len(people) * n)
    s = world.initial_state(people, n, streams)
    states, infos = [s], []
    for t in range(months):
        a = np.zeros((s.n, 4), int)
        if action_at is not None and action_at(t) is not None:
            a[:] = action_at(t)
        s, info = world.step(s, a, streams)
        states.append(s)
        infos.append(info)
    return states, infos


# ------------------------------------------------------------------ the adapter is the learned model
def test_adapter_matches_the_transition_model(market):
    from twin_game_theory.world import spec as sp
    from twin_game_theory.world.transition import TransitionModel, alexia
    tm = TransitionModel()
    base = replace(alexia(), W=sp.N, prog=sp.NONE, edu=sp.BACHELOR, u=1)
    for edu in (sp.NONE, sp.BACHELOR, sp.MASTER, sp.PHD):
        for M in (sp.NORMAL, sp.INFLATION, sp.RECESSION):
            for u, k, grades, student in ((1, 1, 0.0, False), (14, 3, -1.0, False), (0, 2, 1.0, True)):
                d = asdict(replace(base, edu=edu, M=M, u=u, k=k, grades=grades, W=sp.S if student else sp.N))
                for refit in range(market.n_refits):
                    ours = market.p_offer([edu], grades, [u], [k], d["apps"], [M], [student], [refit])[0]
                    assert abs(ours - tm.p_job_offer(d, draw=refit)) < 1e-9
    for grades in (-1.0, 0.0, 1.5):
        d = asdict(replace(base, grades=grades))
        assert abs(market.p_admission(grades, [0])[0] - tm.p_admission(d, sp.MASTER, draw=0)) < 1e-9
    for H, target, income, M in ((0, 1, 9000, 0), (1, 2, 12000, 2), (2, 1, 5000, 1)):
        d = asdict(replace(base, H=H, M=M, W=sp.E100))
        rent = sp.rent(d["city"], target) * d["cost_index"]
        ours = market.p_flat([target], [H], [income], [rent], [d["C"]], [False], [M], [3])[0]
        assert abs(ours - tm.p_flat(d, target, income, draw=3)) < 1e-9


def test_market_directions(market):
    one = lambda **kw: market.p_offer(**{**dict(edu=[BACHELOR], grades=0.0, months_out=[1], level=[1], apps=2,  # noqa: E731
                                              econ=[0], student=[False], refit=[0]), **kw})[0]
    assert one(econ=[2]) < one(econ=[1]) <= one() * 1.1           # recessions cut offers
    assert one(edu=[NONE]) < one() < one(edu=[MASTER])            # degrees help
    assert one(months_out=[18]) < one()                           # long spells hurt
    assert one(apps=4) > one()                                    # more applications help
    assert market.p_admission(1.0, [0])[0] > market.p_admission(-1.0, [0])[0]
    rich = market.p_flat([2], [1], [15000], [3000], [0], [False], [0], [0])[0]
    poor = market.p_flat([2], [1], [6000], [3000], [0], [False], [0], [0])[0]
    assert rich > 0 and poor == 0                                 # the landlord's rule: rent <= income / 3
    assert market.p_flat([2], [1], [6000], [3000], [40000], [False], [0], [0])[0] > 0   # ... or 12 months' rent saved
    assert list(pay_level([0.01, 0.2, 0.5, 0.99])) == [1, 2, 3, 5]


# ----------------------------------------------------------------------------- inside the world
@needs_model
def test_offer_rate_follows_the_market_model():
    w = World.load()
    people = custom(status="unemployed", income=0.0, education="master")
    states, infos = run(w, people, 4000, 3, 1, lambda t: (3, 0, 0, 0))    # January: look for full-time work
    s0 = states[0]
    p = w.market.p_offer(s0.edu, s0.grades, s0.tau, pay_level(s0.rank), 2, s0.econ, np.zeros(s0.n, bool), s0.member)
    got = infos[0]["offer"].mean()
    assert abs(got - p.mean()) < 4 * np.sqrt(p.mean() * (1 - p.mean()) / s0.n) + 0.01
    off, offi = run(World.load(config=Config(markets=False)), people, 4000, 3, 1, lambda t: (3, 0, 0, 0))
    assert abs(offi[0]["offer"].mean() - 0.20) < 0.03              # markets off: the priors' 20% a month


@needs_model
def test_every_move_meets_the_landlords_rule():
    w = World.load()
    people = custom(income=1500, rent=700, household_type="shared_housing", cash=0, investments=0)
    up_every_january = lambda t: (0, 1, 0, 0) if C.annual_review(t) else None     # noqa: E731
    states, infos = run(w, people, 300, 5, 36, up_every_january)
    for before, info in zip(states, infos):
        rows = info["moved"]
        new_rent = w.rent_model.rent(before.tier[rows] + 1, before.city[rows], before.rent_z[rows]) * before.P[rows]
        assert ((new_rent <= before.income[rows] / 3) | (before.cash[rows] >= 12 * new_rent)).all()
    assert (states[-1].tier == 0).mean() > 0.9                     # only a windfall or a raise lets them move
    off, _ = run(World.load(config=Config(markets=False)), people, 300, 5, 1, up_every_january)
    assert (off[1].tier == 1).all()                                # markets off: moves are immediate


@needs_model
def test_flat_search_lasts_until_found_or_the_next_review():
    people = custom(income=12000, rent=1600, cash=150000)
    states, infos = run(World.load(), people, 1000, 7, 24, lambda t: (0, 1, 0, 0) if t == 0 else None)
    moved = np.array([i["moved"] for i in infos])                  # (months, paths)
    assert 0.05 < moved[0].mean() < 0.95                            # not certain in the first month
    assert moved[1:12].sum() > 0.2 * moved[0].sum()                # the search goes on after January
    assert not moved[12:].any()                                    # the next review (no choice) ends it
    assert (moved.sum(0) <= 1).all()                               # one move per search


@needs_model
def test_education_is_kept_or_earned():
    w = World.load()
    states, _ = run(w, custom(education="phd"), 50, 1, 1)
    assert (states[0].edu == 3).all()
    alexia = profiles().loc[["P00001"]]                            # a student finishing her course
    states, infos = run(w, alexia, 200, 2, 2)
    assert (states[0].edu == NONE).all()
    assert (states[-1].edu[infos[0]["graduated"]] == BACHELOR).all()
    with pytest.raises(ValueError):
        custom(education="high school")
