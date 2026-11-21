import numpy as np
import pytest

from twin import config as C
from twin.data import profiles
from twin.explore import goal_met, simulate
from twin.live import branch, read_journal, record, salary_search
from twin.policy.foresight import at_level
from twin.policy.net import NeuralPolicy
from twin.policy.rules import RulePolicy
from twin.policy.traits import from_profile, Intention
from twin.world.env import Config, World


@pytest.fixture(scope="module")
def world():
    return World.load(config=Config(events=False))


@pytest.mark.parametrize("policy", [RulePolicy, NeuralPolicy, at_level(0.5)])
def test_dated_major_plan_waits_for_january(policy, world):
    people = profiles().loc[["P00003"]]
    p = from_profile(people.iloc[0])
    # Keep mode in January. An off-cycle plan requests the opposite mode.
    p.intentions = [Intention("status", 0, when="2025-01"), Intention("status", 2, when="2025-03")]
    out = simulate(world, people, p, 40, 32, horizon=24, policy_cls=policy)
    quiet = np.arange(24) % 12 != 0
    assert (out.action[:, quiet, :2] == 0).all()
    assert (out.action[:, 12, 0] == 2).all()
    assert (out.action[:, quiet, 2:] != 0).any()


def test_live_offer_and_observation(tmp_path, world):
    db = tmp_path / "desk.sqlite"
    people = profiles().loc[["P00003"]]
    p = from_profile(people.iloc[0])
    j = read_journal(db, "P00003")
    alt = branch(j, C.PART_TIME, 12500)
    preview = simulate(world, people, p, 40, 2, horizon=2, timeline=alt)
    assert not read_journal(db, "P00003").timeline.offers
    record(db, j, 0, "offer", [C.PART_TIME, 12500])
    with pytest.raises(ValueError):
        record(db, j, 0, "observation", {"cash": 1})
    saved = read_journal(db, "P00003")
    committed = simulate(world, people, p, 40, 2, horizon=2, timeline=saved.timeline)
    assert np.array_equal(preview.cash, committed.cash)
    assert (committed.income[:, 1] >= 12500).all()
    with pytest.raises(ValueError):
        record(db, saved, 1, "observation", {"cash": 1})
    record(db, saved, 0, "observation", {"cash": 100, "investments": 50000, "rent": 1200, "income": 4200})
    observed = read_journal(db, "P00003")
    assert observed.current_month == 1
    updated = simulate(world, people, p, 40, 2, horizon=2, timeline=observed.timeline)
    assert np.all(updated.cash[:, 1] == 100) and np.all(updated.inv[:, 1] == 50000)
    assert np.allclose(updated.income[:, 1], 4200)
    assert np.all(updated.mode[:, 1] == C.PART_TIME)


def test_credit_breach_blocks_only_goals_at_or_after_breach(world):
    people = profiles().loc[["P00003"]]
    paths = simulate(world, people, from_profile(people.iloc[0]), 2, 4)
    paths.cash[:, C.T_HIGH_GOAL + 1] = 1e7
    paths.credit_breach[:] = False
    paths.credit_breach[0, C.T_HIGH_GOAL] = True
    paths.credit_breach[1, C.T_HIGH_GOAL + 1] = True
    assert goal_met(paths)[0].tolist() == [False, True]


def test_salary_grid_checks_all_offers(tmp_path, world):
    people = profiles().loc[["P00003"]]
    result, rows = salary_search(world, people, from_profile(people.iloc[0]), read_journal(tmp_path/"j", "P00003"),
                                 [6000,4000,5000], 0., 15, 42)
    assert result == 4000 and [r["salary"] for r in rows] == [4000,5000,6000]
    people = people.assign(high_target=1e20)
    result, _ = salary_search(world, people, from_profile(people.iloc[0]), read_journal(tmp_path/"j", "P00003"),
                              [6000], 1., 15, 42)
    assert result is None
