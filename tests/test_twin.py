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
        assert np.allclose(s1.cash, s.cash - info["transfer"] + info["net"] + info["topup"], atol=1e-6)
        s = s1


@needs_model
def test_transfers_do_not_change_net_worth():
    w = world_with(**{f"economy.returns_per_year.{e}": [0.0, 0.0] for e in C.ECONOMY})
    people = profiles().iloc[:100]
    streams = Streams(3, len(people))
    s = w.initial_state(people, 1, streams)
    for t in range(24):
        a = np.zeros((s.n, 4), int)
        a[:, 3] = 1 if t < 12 else -1                 # invest in year one, sell in year two
        s1, info = w.step(s, a, streams)
        nw0, nw1 = s.cash + s.inv - s.debt, s1.cash + s1.inv - s1.debt
        expected = info["net"] + info["loan_payment"] - info["loan_interest"]
        assert np.allclose(nw1 - nw0, expected, atol=1e-6)
        moved = (np.abs(info["transfer"]) > 0).any()
        assert moved if t in (0, 12) else not moved   # transfers only at the January review
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
    # Opening students can enroll at the first review.
    assert not ((status[:, 1:] == 1) & (mode[:, 1:] == C.STUDENT)).any()
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


# --------------------------------------------------------------------------- LLM persona, foresight
def test_llm_answer_parsing():
    from twin.policy.llm import parse
    valid = {"status": np.array([1, 0, 0, 1, 1.0]), "housing": np.array([1, 1, 0.0]),
             "spending": np.array([1, 1, 1.0]), "assets": np.array([0, 1, 1.0])}
    ok = {"status": {"0": 0.7, "1": 0.2, "3": 0.1}, "housing": {"-1": 0.2, "0": 0.8, "1": 0.5},
          "spending": {"0": 1}, "assets": {"1": "0.6", "0": 0.4}}
    p = parse(ok, valid)
    heads = np.split(p, [5, 8, 11])
    assert all(abs(h.sum() - 1) < 1e-6 for h in heads)
    assert heads[0][1] == 0 and heads[1][2] == 0 and heads[3][0] == 0      # options not open stay at zero
    assert heads[0][0] > heads[0][3] > 0
    assert parse({"status": {"9": 1}}, valid) is None and parse({}, valid) is None


def test_llm_client_caches_and_never_needs_a_key_for_cached_prompts(tmp_path):
    import types
    from twin.policy.llm import Client
    calls = []

    class Fake:
        chat = property(lambda self: self)
        completions = property(lambda self: self)

        def create(self, **kw):
            calls.append(kw["model"])
            msg = types.SimpleNamespace(content='{"status": {"0": 1}}')
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])
    c = Client("m", cache=tmp_path / "c.jsonl", sdk_client=Fake())
    assert c.ask("hello") == {"status": {"0": 1}} and c.ask("hello") == {"status": {"0": 1}}
    again = Client("m", cache=tmp_path / "c.jsonl")                       # no SDK, no key: served from disk
    assert again.ask("hello") == {"status": {"0": 1}} and calls == ["m"]


@pytest.mark.skipif(not (C.ARTIFACTS / "foresight_net.pt").exists(), reason="python -m twin foresight")
def test_foresight_zero_stays_close_to_the_persona(world):
    from twin.explore import goal_met
    from twin.policy.foresight import at_level
    from twin.policy.net import NeuralPolicy
    people = profiles().iloc[[10]]
    p = from_profile(people.iloc[0])
    a = simulate(world, people, p, n_paths=600, seed=3, policy_cls=NeuralPolicy)
    b = simulate(world, people, p, n_paths=600, seed=3, policy_cls=at_level(0.0))
    assert abs(goal_met(a)[0].mean() - goal_met(b)[0].mean()) < 0.06
    assert abs(np.median(a.nw[:, -1]) / np.median(b.nw[:, -1]) - 1) < 0.1


@needs_model
def test_life_utility_follows_preferences(world):
    from twin.policy.foresight import life_utility
    lean = steady(frugality=0.98, impatience=0.02)
    lean.intentions = [Intention("spending", -1, 1.0, when="2025-01")]
    rich = custom(cash=500000, income=12000)
    u = life_utility(simulate(world, rich, lean, n_paths=200, seed=2, horizon=120))
    paths = simulate(world, rich, lean, n_paths=200, seed=2, horizon=120)
    assert np.allclose(u, life_utility(paths)) and np.isfinite(u).all()
    assert (u > paths.nw[:, -1] - 120 * 2500).all()                      # bounded monthly terms


# --------------------------------------------------------------------------- study progress
def test_data_students_keep_the_course_ending_at_the_opening_review():
    prof = profiles()
    students = prof["mode"] == C.STUDENT
    assert students.any()
    assert (prof.loc[students, "study_left0"] == 1).all() and (prof.loc[~students, "study_left0"] == 0).all()
    assert not prof["masters0"].any()


@needs_model
def test_study_months_left_sets_the_study_clock(world):
    default = custom(status="student", income=1200, rent=1500, cash=1000)
    longer = custom(status="student", income=1200, rent=1500, cash=1000, study_months_left=24, in_masters=True)
    worker = custom(study_months_left=24, in_masters=True)          # ignored unless studying
    s = world.initial_state(longer, 3, Streams(1, 3))
    assert (s.study_left == 24).all() and s.degree.all()
    s = world.initial_state(worker, 3, Streams(1, 3))
    assert (s.study_left == 0).all() and not s.degree.any()
    masks = world.masks(world.initial_state(longer, 3, Streams(1, 3)))[0]
    assert not masks[:, 1].any()                                    # no second master's while doing one
    mid = world.initial_state(custom(status="student", study_months_left=12), 3, Streams(1, 3))
    assert not world.masks(mid)[0][:, 1].any()                      # nor mid-course
    assert world.masks(world.initial_state(default, 3, Streams(1, 3)))[0][:, 1].all()   # final month: allowed

    n = 200                                     # comfortable finances, so no one drops out to pay the rent
    comfy = dict(status="student", income=2500, rent=1000, cash=60000)
    short = simulate(world, custom(**comfy), steady(career_drive=0.2), n, 3, horizon=30)
    long = simulate(world, custom(**comfy, study_months_left=24, in_masters=True), steady(career_drive=0.2), n, 3,
                    horizon=30)
    assert (short.mode[:, 12] == C.STUDENT).mean() < 0.2
    assert (long.mode[:, 12] == C.STUDENT).mean() > 0.6
    first_grad = np.where(long.events["graduated"].any(1), long.events["graduated"].argmax(1), 10**6)
    assert first_grad.min() >= 23                                   # no graduation before the course ends


@needs_model
def test_key_moments_say_when_big_decisions_happen(world):
    from twin.explore import key_moments
    from twin.timeline import Timeline
    people = custom()
    t = Timeline()
    t.offers[0] = (C.PART_TIME, 4000)
    paths = simulate(world, people, steady(), 200, 5, timeline=t)
    moments = key_moments(paths)
    first = moments[0]
    assert first["moment"] == "Cuts back to part-time" and first["typical_year"] == C.START_YEAR
    assert first["share"] == 1.0 and first["share_by_goal_deadline"] == 1.0 and first["kind"] == "decision"
    assert all(m["share"] >= 0.10 for m in moments) and len(moments) <= 6
    assert [m["typical_year"] for m in moments] == sorted(m["typical_year"] for m in moments)
    assert all(m["share_by_goal_deadline"] <= m["share"] for m in moments)
    assert all(m["kind"] in ("decision", "event") for m in moments)


@needs_model
def test_part_time_offer_alongside_studies_or_instead_of_them(world):
    from twin.timeline import Timeline
    student = custom(status="student", income=1200, rent=1000, cash=60000, study_months_left=24, in_masters=True)
    # Accepting directly: alongside keeps the course and the pay rank; otherwise the course (and master's) is left.
    s = world.initial_state(student, 4, Streams(1, 4))
    rank = s.rank.copy()
    job = world.accept_offer(s, (C.PART_TIME, 4000, True))
    assert not job.any() and (s.mode == C.STUDENT).all() and (s.study_left == 24).all() and s.degree.all()
    assert np.allclose(s.rank, rank) and np.allclose(s.income, 4000)
    s = world.initial_state(student, 4, Streams(1, 4))
    job = world.accept_offer(s, (C.PART_TIME, 4000))
    assert job.all() and (s.mode == C.PART_TIME).all() and (s.study_left == 0).all() and not s.degree.any()
    assert world.masks(s)[0][:, 1].all()                     # an unfinished master's can be started again
    worker = world.initial_state(custom(), 4, Streams(1, 4))
    assert world.accept_offer(worker, (C.PART_TIME, 4000, True)).all()   # the flag only matters for students

    n, persona = 200, steady(career_drive=0.2)
    t_along, t_leave = Timeline(), Timeline()
    t_along.offers[0], t_leave.offers[0] = (C.PART_TIME, 4000, True), (C.PART_TIME, 4000, False)
    base = simulate(world, student, persona, n, 3, horizon=48)
    along = simulate(world, student, persona, n, 3, horizon=48, timeline=t_along)
    leave = simulate(world, student, persona, n, 3, horizon=48, timeline=t_leave)
    assert along.events["side_job"][:, 0].all() and not leave.events["side_job"].any()
    assert (along.mode[:, 12] == C.STUDENT).mean() > 0.8 and (leave.mode[:, 12] == C.STUDENT).mean() < 0.05
    assert np.median(along.income[:, 12]) > 3 * np.median(base.income[:, 12])          # earns the offer while studying
    assert along.events["graduated"].any(1).mean() > 0.8 and leave.events["graduated"][:, :23].sum() == 0
    late = np.median(along.income[:, 40]) / np.median(base.income[:, 40])               # same graduate pay afterwards
    assert 0.85 < late < 1.15


@needs_model
def test_side_job_student_is_costed_as_a_part_timer_plus_tuition():
    w = World.load(config=Config(events=False))
    student = custom(status="student", income=1200, rent=1000, cash=60000, study_months_left=24, in_masters=True, tuition=4800)
    s = w.initial_state(student, 6, Streams(2, 6))
    w.accept_offer(s, (C.PART_TIME, 4000, True))
    assert s.side_job.all() and (s.mode == C.STUDENT).all()
    worker = w.initial_state(student, 6, Streams(2, 6))   # the same person, pay and rent, working part-time instead
    w.accept_offer(worker, (C.PART_TIME, 4000))
    assert np.allclose(worker.income, s.income) and not worker.side_job.any()
    still = np.zeros((s.n, 4), int)
    s1, a = w.step(s, still, Streams(3, 6))
    _, b = w.step(worker, still, Streams(3, 6))
    assert np.allclose(b["net"] - a["net"], 4800 / 12 * s.P)          # only the tuition differs
    s = s1
    for _ in range(30):                                               # the side job ends with the course
        s, info = w.step(s, still, Streams(3, 6))
    assert not s.side_job.any() and (s.mode != C.STUDENT).all()


@needs_model
def test_anchored_foresight_is_the_rule_persona_at_zero_and_tilts_above(world):
    from twin.policy.foresight import PATH, anchored_at_level
    if not PATH.exists():
        pytest.skip("train first: python -m twin foresight")
    people = custom()
    persona = from_profile(people.iloc[0])
    rules = simulate(world, people, persona, 150, 4, horizon=60)
    zero = simulate(world, people, persona, 150, 4, horizon=60, policy_cls=anchored_at_level(0.0))
    assert np.array_equal(rules.action, zero.action) and np.allclose(rules.nw, zero.nw)
    tilted = simulate(world, people, persona, 150, 4, horizon=60, policy_cls=anchored_at_level(1.0))
    assert not np.array_equal(rules.action, tilted.action)


# ------------------------------------------------------------------ learned markets (twin_game_theory)
def test_market_port_matches_twin_game_theory():
    from dataclasses import asdict, replace
    from twin.world.markets import Markets
    gt_mod = pytest.importorskip("twin_game_theory.world.transition")
    from twin_game_theory.world import spec as sp
    gt, mk = gt_mod.TransitionModel(), Markets()
    for draw in range(5):
        for edu, u, k, M, W in [(1, 0, 1, 0, sp.N), (2, 5, 3, 2, sp.N), (3, 30, 5, 1, sp.S)]:
            d = asdict(replace(gt_mod.alexia(), W=W, edu=edu, u=u, k=k, M=M))
            mine = mk.p_job_offer(np.array([edu]), np.array([u]), np.array([(k - .5) / 5]), np.array([M]),
                                  np.array([W == sp.S]), np.array([draw]))[0]
            assert abs(mine - gt.p_job_offer(d, draw)) < 1e-12
        assert abs(mk.p_master_admission(np.array([draw]))[0] - gt.p_admission(asdict(gt_mod.alexia()), sp.MASTER, draw)) < 1e-12


def test_markets_rank_degrees_spells_and_landlords():
    from twin.world.markets import Markets
    mk = Markets()
    one = lambda v: np.array([v])  # noqa: E731
    bachelor, master = (mk.p_job_offer(one(e), one(0), one(.3), one(0), one(False), one(0))[0] for e in (1, 2))
    fresh, long_out = (mk.p_job_offer(one(1), one(u), one(.3), one(0), one(False), one(0))[0] for u in (0, 18))
    normal, recession = (mk.p_job_offer(one(1), one(0), one(.3), one(M), one(False), one(0))[0] for M in (0, 2))
    assert master > bachelor and fresh > long_out and normal > 3 * recession
    args = dict(target=one(2), tier=one(1), new_rent=one(3000.0), student=one(False), econ=one(0), draw=one(0))
    assert mk.p_flat(income=one(6000.0), cash=one(5000.0), **args)[0] == 0      # rent > income / 3, little cash
    assert mk.p_flat(income=one(6000.0), cash=one(40000.0), **args)[0] > 0      # 12 months of rent saved
    assert mk.p_flat(income=one(9500.0), cash=one(0.0), **args)[0] > 0          # rent <= income / 3


@needs_model
def test_markets_switch_and_landlord_rule_in_the_world():
    assert World.load().markets is not None and World.load(config=Config(markets=False)).markets is None
    w = World.load(config=Config(events=False))
    poor = custom(income=2500, rent=1200, cash=500, investments=0, household_type="single")
    s = w.initial_state(poor, 50, Streams(9, 50))
    assert (s.tier < 2).all()
    up = np.zeros((s.n, 4), int)
    up[:, 1] = 1                                                    # everyone asks for a bigger flat in January
    s1, info = w.step(s, up, Streams(9, 50))
    assert not info["moved"].any()                                  # no landlord accepts rent > income / 3
    old = World.load(config=Config(events=False, markets=False))
    _, info_old = old.step(old.initial_state(poor, 50, Streams(9, 50)), up, Streams(9, 50))
    assert info_old["moved"].all()                                  # the old world always found a flat
