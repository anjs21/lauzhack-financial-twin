"""Checks from evaluation/technical-validation.md, plus the world model and persona contracts.

Run: python -m pytest tests -q   (needs a trained world model: python -m twin train)
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from twin import config as C
from twin.data import ledger, load_priors, profiles
from twin.explore import person_table, simulate
from twin.policy.traits import Intention, Persona, from_profile
from twin.world.env import Config, Streams, World
from twin.world.flow import FEATURES
from twin.world.redraw import IncomeModel, RentModel, income_kind

pytestmark = pytest.mark.filterwarnings("ignore")
HAVE_MODEL = (C.ARTIFACTS / "flow" / "meta.json").exists()
needs_model = pytest.mark.skipif(not HAVE_MODEL, reason="train first: python -m twin train")


@pytest.fixture(scope="module")
def world():
    return World.load()


def world_with(**changes):
    pr = copy.deepcopy(load_priors())
    for path, value in changes.items():
        d = pr
        keys = path.split(".")
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value
    return World.load(priors=pr)


def custom(**kw):
    base = dict(name="Test", age=30, city="Bern", status="employed", household_type="single", income=6000,
                rent=1600, cash=20000, investments=10000, debt=0, rate=0, high_goal="Emergency savings",
                high_target=20000)
    base.update(kw)
    return person_table(custom=base)


def steady(**traits):
    t = dict(risk_tolerance=0.5, frugality=0.5, career_drive=0.5, free_time=0.3, housing_aspiration=0.4,
             inertia=0.7, impatience=0.4, goal_focus=0.5)
    t.update(traits)
    return Persona(t, spread=0.0)


# ------------------------------------------------------------------------------------- data
def test_ledger_identity():
    led = ledger()
    err = (led.opening_cash_chf + led.net_transactions_chf - led.closing_cash_chf).abs().max()
    assert err < 1e-6


def test_redraws_round_trip():
    prof = profiles()
    inc, rent = IncomeModel.fit(prof), RentModel.fit(prof)
    kind = income_kind(prof["mode"].values, prof["pensioner"].values)
    assert np.allclose(inc.quantile(kind, inc.rank(kind, prof.income0.values)), prof.income0.values)
    z = rent.z_of(prof.rent0.values, prof.tier.values, prof.city_idx.values)
    assert np.allclose(rent.rent(prof.tier.values, prof.city_idx.values, z), prof.rent0.values)


def test_network_never_sees_the_calendar_year():
    assert not any(f in ("t", "year", "date") or f.startswith("year") for f in FEATURES)


@needs_model
def test_world_model_passed_acceptance_tests():
    res = json.loads((C.ARTIFACTS / "validation.json").read_text())
    assert res["passed"], res["verdict_network"]


# ------------------------------------------------------------------------- cash-flow identities
@needs_model
def test_cash_identity_every_month(world):
    people = profiles().iloc[:50]
    streams = Streams(7, len(people) * 4)
    s = world.initial_state(people, 4, streams)
    rng = np.random.default_rng(0)
    for _ in range(24):
        a = np.stack([rng.integers(0, 5, s.n), rng.integers(-1, 2, s.n), rng.integers(-1, 2, s.n),
                      rng.integers(-1, 2, s.n)], 1)
        s1, info = world.step(s, a, streams)
        assert np.allclose(s1.cash, s.cash - info["transfer"] + info["net"], atol=1e-6)
        s = s1


@needs_model
def test_transfers_do_not_change_net_worth():
    w = world_with(**{f"economy.returns_per_year.{e}": [0.0, 0.0] for e in C.ECONOMY})
    people = profiles().iloc[:100]
    streams = Streams(3, len(people))
    s = w.initial_state(people, 1, streams)
    for t in range(12):
        a = np.zeros((s.n, 4), int)
        a[:, 3] = 1 if t % 2 == 0 else -1                 # invest, then sell
        s1, info = w.step(s, a, streams)
        nw0, nw1 = s.cash + s.inv - s.debt, s1.cash + s1.inv - s1.debt
        expected = info["net"] + info["loan_payment"] - info["loan_interest"]
        assert np.allclose(nw1 - nw0, expected, atol=1e-6)
        assert (np.abs(info["transfer"]) > 0).any()
        s = s1


@needs_model
def test_loans_amortise_and_never_go_negative(world):
    people = profiles()
    people = people[people.debt0 > 0].iloc[:40]
    streams = Streams(5, len(people))
    s = world.initial_state(people, 1, streams)
    paid = interest = 0.0
    prev = s.debt.copy()
    for _ in range(72):
        s, info = world.step(s, np.zeros((s.n, 4), int), streams)
        assert (s.debt >= 0).all() and (s.debt <= prev + 1e-9).all()
        paid, interest, prev = paid + info["loan_payment"], interest + info["loan_interest"], s.debt.copy()
    assert np.allclose(s.debt, 0)                                     # 60-month term
    assert np.allclose(paid, people.debt0.values + interest)


# --------------------------------------------------------------------------- boundary cases
@needs_model
def test_zero_income_is_visible(world):
    paths = simulate(world, custom(status="unemployed", income=0.0, cash=5000), steady(), n_paths=200, seed=2,
                     horizon=12)
    assert paths.income[:, 0].max() == 0
    assert np.median(paths.cash[:, -1]) < 5000


def run_fixed(w, people, n, seed, months, first_action=None):
    """Drive the world with fixed actions (no persona), so one assumption can be isolated."""
    streams = Streams(seed, len(people) * n)
    s = w.initial_state(people, n, streams)
    out = [s]
    for t in range(months):
        a = np.zeros((s.n, 4), int)
        if t == 0 and first_action is not None:
            a[:] = first_action
        s, _ = w.step(s, a, streams)
        out.append(s)
    return out


@needs_model
def test_higher_tuition_never_lowers_costs():
    w = World.load(config=Config(events=False))
    start = (1, 0, 0, 0)                                          # start a master's now, nothing else
    cheap = run_fixed(w, custom(tuition=1200), 300, 4, 24, start)
    dear = run_fixed(w, custom(tuition=8000), 300, 4, 24, start)
    assert (cheap[1].mode == C.STUDENT).all()
    gap = cheap[-1].cash - dear[-1].cash                          # same luck: only tuition differs
    fees = load_priors()["living_costs"]["typical_student_fees_per_year"]   # already in the learned flows
    extra = (8000 - fees) * 2                                     # two years of tuition above typical fees
    assert (gap >= extra * 0.99).all()                            # never cheaper; delays and credit cost more
    assert np.median(gap) < extra * 1.1


@needs_model
def test_negative_returns_shrink_the_portfolio():
    w = world_with(**{f"economy.returns_per_year.{e}": [-0.24, 0.0] for e in C.ECONOMY})
    states = run_fixed(w, custom(investments=50000), 20, 1, 12)
    for a, b in zip(states, states[1:]):
        assert np.allclose(b.inv, a.inv * (1 - 0.02))            # -24% a year, monthly, no transfers


# ------------------------------------------------------------------ reproducibility, isolation
@needs_model
def test_same_seed_same_lives(world):
    a = simulate(world, profiles().iloc[[3]], steady(), n_paths=100, seed=9, horizon=36)
    b = simulate(world, profiles().iloc[[3]], steady(), n_paths=100, seed=9, horizon=36)
    c = simulate(world, profiles().iloc[[3]], steady(), n_paths=100, seed=10, horizon=36)
    assert np.array_equal(a.cash, b.cash) and np.array_equal(a.action, b.action)
    assert not np.array_equal(a.cash, c.cash)


@needs_model
def test_what_if_changes_nothing_before_the_plan(world):
    base = steady()
    alt = steady()
    alt.intentions = [Intention("status", 2, 1.0, when="2027-01")]
    people = custom()
    a = simulate(world, people, base, n_paths=200, seed=11, horizon=48)
    b = simulate(world, people, alt, n_paths=200, seed=11, horizon=48)
    k = C.month_index("2027-01")
    assert np.array_equal(a.cash[:, :k + 1], b.cash[:, :k + 1])
    assert (b.action[:, k, 0] == 2).mean() > 0.8               # everyone still employed takes the plan


# -------------------------------------------------------------------------------------- policy
@needs_model
def test_actions_respect_masks(world):
    paths = simulate(world, profiles().iloc[:30], [steady()] * 30, n_paths=40, seed=6, horizon=120)
    status, mode = paths.action[:, :, 0], paths.mode[:, :-1]
    assert not ((status == 4) & ~np.isin(mode, [C.PART_TIME, C.FULL_TIME])).any()
    assert not ((status == 1) & (mode == C.STUDENT)).any()
    assert ((status == 1).sum(1) <= 1).all()                    # one master's per life
    assert not ((paths.household[:, 1:] == C.FAMILY) & (paths.tier[:, 1:] == 0)).any()


@needs_model
def test_status_and_housing_only_at_decision_points(world):
    from twin.explore import goals_per_path
    from twin.policy.rules import RulePolicy
    people = profiles().iloc[:40]
    streams = Streams(13, len(people) * 10)
    traits = np.tile(steady().vector(), (len(people) * 10, 1))
    s = world.initial_state(people, 10, streams)
    pol = RulePolicy(world, traits, 1.0, goals_per_path(people, 10))
    pol.reset(s, streams)
    for _ in range(36):
        a, _ = pol.act(s, streams)
        quiet = ~pol.last_decide
        assert (a[quiet, 0] == 0).all() and (a[quiet, 1] == 0).all()
        s, info = world.step(s, a, streams)
        pol.observe(info, s)


@needs_model
def test_intentions_fire_with_their_probability(world):
    p = steady()
    p.intentions = [Intention("status", 2, 0.6, when="2026-01")]
    paths = simulate(world, custom(), p, n_paths=2000, seed=21, horizon=13)
    k = C.month_index("2026-01")
    employed = paths.mode[:, k] == C.FULL_TIME
    share = (paths.action[employed, k, 0] == 2).mean()
    assert abs(share - 0.6) < 0.04


@needs_model
def test_frugal_personas_upgrade_lifestyle_less(world):
    people = custom(cash=60000)
    frugal = simulate(world, people, steady(frugality=0.95, impatience=0.05), n_paths=500, seed=8)
    spender = simulate(world, people, steady(frugality=0.05, impatience=0.95), n_paths=500, seed=8)
    assert (frugal.lifestyle == 2).mean() < (spender.lifestyle == 2).mean()
    assert np.median(frugal.nw[:, -1]) > np.median(spender.nw[:, -1])
