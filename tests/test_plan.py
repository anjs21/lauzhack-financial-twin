"""Long-term plans: schema, how the simulator carries them out, the planner loop and the API. No paid calls."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from twin import config as C
from twin.data import load_priors
from twin.explore import person_table, simulate
from twin.plan import store
from twin.plan.agent import PlannerAgent, PlanError, apply_edits, parse_reply, person_brief, repair
from twin.plan.evaluate import evaluate, with_plan
from twin.plan.schema import LAST, LifePlan, diff, freeze_past
from twin.policy.traits import from_profile
from twin.world.env import Config, World

pytestmark = pytest.mark.filterwarnings("ignore")
needs_model = pytest.mark.skipif(not (C.ARTIFACTS / "flow" / "meta.json").exists(), reason="train first: python -m twin train")
MIA = json.loads(Path("twin/examples/custom_person.json").read_text())


@pytest.fixture(scope="module")
def world():
    return World.load(config=Config(events=False))


def person(**changes):
    people = person_table(custom={**MIA, **changes})
    return people, from_profile(people.iloc[0])


def plan(steps, spontaneity=0.0, narrative="I follow my plan."):
    p, errors = LifePlan.from_dict({"narrative": narrative, "spontaneity": spontaneity, "steps": steps})
    assert not errors, errors
    return p


def test_plan_errors_are_readable_and_complete():
    bad, errors = LifePlan.from_dict({"narrative": "x", "spontaneity": 2, "steps": [
        {"id": "a", "head": "status", "option": "phd", "start": "2026-1"},
        {"id": "a", "head": "housing", "option": "bigger", "after": "nowhere"},
        {"id": "b", "head": "spending", "option": "cut", "event": "lottery", "until": "2040-01", "colour": "red"}]})
    assert bad is None
    text = " | ".join(errors)
    for expected in ("spontaneity", "phd", "YYYY-MM", "unknown step 'nowhere'", "use 'event'", "unique: a", "lottery", "horizon",
                     "unknown fields ['colour']"):
        assert expected in text, expected
    loop, errors = LifePlan.from_dict({"narrative": "x", "steps": [
        {"id": "a", "head": "spending", "option": "cut", "after": "b"},
        {"id": "b", "head": "spending", "option": "cut", "after": "a"}]})
    assert loop is None and any("cycle" in e for e in errors)
    noop, errors = LifePlan.from_dict({"narrative": "x", "steps": [{"id": "a", "head": "status", "option": "continue"}]})
    assert noop is None and "is not a step" in errors[0]


def test_plan_compiles_to_intentions_behind_the_habit_holds():
    p = plan([{"id": "m", "head": "status", "option": "start_masters", "start": "2026-03", "commitment": .7},
              {"id": "flat", "head": "housing", "option": "bigger", "after": "m", "after_months": 24},
              {"id": "inv", "head": "assets", "option": "invest_surplus", "repeat": True}], spontaneity=.25)
    its = p.intentions()
    assert [i.id for i in its] == ["_hold_status", "_hold_housing", "m", "flat", "inv"]
    assert its[0].prob == its[1].prob == .75 and not its[0].once
    m, flat, inv = its[2:]
    assert (m.option, m.prob, m.when, m.until) == (1, .7, "2026-03", "2027-12")   # its own year's review (Jan 2027)
    assert (flat.option, flat.after, flat.until) == (1, "m", LAST)
    assert inv.once is False and inv.until == LAST
    again, errors = LifePlan.from_dict(p.to_dict())
    assert not errors and again == p


@needs_model
def test_habits_only_change_work_and_home_within_spontaneity(world):
    people, persona = person(age=24)
    habits = simulate(world, people, persona, 300, 4)
    assert ((habits.action[:, :, 0] > 0) | (habits.action[:, :, 1] != 0)).any()
    still = simulate(world, people, with_plan(persona, plan([], spontaneity=0.0)), 300, 4)
    assert (still.action[:, :, 0] == 0).all() and (still.action[:, :, 1] == 0).all()
    free = simulate(world, people, with_plan(persona, plan([], spontaneity=1.0)), 300, 4)
    assert np.array_equal(free.action, habits.action) and np.array_equal(free.cash, habits.cash)


@needs_model
def test_a_window_retries_until_the_step_is_possible(world):
    people, persona = person(status="student", study_months_left=13, income=1200, cash=30000)
    windowed = plan([{"id": "m", "head": "status", "option": "start_masters", "start": "2025-01", "until": "2027-12",
                      "commitment": 1}])
    ev, _, paths = evaluate(world, people, persona, windowed, 200, 3)
    step = ev["steps"][0]
    assert step["blocked_share"] == 0 and step["taken_share"] > 0.95 and step["typical_year"] == 2026
    assert paths.plan_log["blocked"][2].all()           # due in 2025 while the course still ran
    half = plan([{**windowed.to_dict()["steps"][0], "commitment": .5}])
    step = evaluate(world, people, persona, half, 400, 3)[0]["steps"][0]
    assert step["blocked_share"] == 0 and abs(step["declined_share"] - .5) < .08    # declined, not blocked
    assert step["turned_down_share"] < .05
    once = plan([{"id": "m", "head": "status", "option": "start_masters", "start": "2025-01"}])
    ev, _, _ = evaluate(world, people, persona, once, 200, 3)
    assert ev["steps"][0]["taken_share"] == 0 and ev["steps"][0]["blocked_share"] == 1
    assert "last month" in ev["steps"][0]["blocked_because"]


@needs_model
def test_after_waits_for_the_step_it_follows(world):
    people, persona = person()
    p = plan([{"id": "cut", "head": "spending", "option": "cut", "start": "2025-01", "commitment": .5},
              {"id": "move", "head": "housing", "option": "cheaper", "after": "cut", "after_months": 12, "commitment": 1}])
    _, _, paths = evaluate(world, people, persona, p, 300, 5)
    cut, move = paths.plan_log["done_at"][2], paths.plan_log["done_at"][3]
    assert 0.35 < (cut >= 0).mean() < 0.65
    assert ((move >= 0) <= (cut >= 0)).all()
    assert (move[move >= 0] - cut[move >= 0] >= 12).all() and (move >= 0).sum() > 0.8 * (cut >= 0).sum()


@needs_model
def test_a_contingency_waits_for_its_event():
    pr = copy.deepcopy(load_priors())
    pr["jobs"]["loss_per_month"] = {e: 0.03 for e in C.ECONOMY}
    world = World.load(priors=pr)
    people, persona = person(age=30)
    p = plan([{"id": "cut", "head": "spending", "option": "cut", "event": "job_loss", "within_months": 12, "commitment": 1}])
    ev, _, paths = evaluate(world, people, persona, p, 300, 6)
    done, lost = paths.plan_log["done_at"][2], paths.events["job_loss"]
    assert (done >= 0).sum() > 30
    for i in np.flatnonzero(done >= 0):
        assert lost[i, max(0, done[i] - 12):done[i]].any()
    calm = World.load(config=Config(events=False))
    ev, _, _ = evaluate(calm, people, persona, p, 100, 6)
    assert ev["steps"][0]["trigger_share"] == 0 and ev["steps"][0]["taken_share"] == 0


def test_repair_fixes_repeated_slips():
    d = {"narrative": "x", "steps": [{"id": "keep", "head": "status", "option": "continue"},
                                     {"id": "flat", "head": "housing", "option": "bigger", "after": "promotion"},
                                     {"id": "cut", "head": "spending", "option": "cut", "after": "job_loss"},
                                     {"id": "more", "head": "spending", "option": "raise", "after": "keep"}]}
    fixed, notes = repair(d)
    p, errors = LifePlan.from_dict(fixed)
    assert not errors and len(notes) == 4
    assert [s.id for s in p.steps] == ["flat", "cut", "more"]
    assert p.step("flat").after is None and p.step("cut").event == "job_loss" and p.step("more").after is None
    assert repair({"narrative": "x", "steps": [GOOD["steps"][0]]})[1] == []


@needs_model
def test_money_trouble_is_diagnosed_where_it_happens(world):
    people, persona = person(cash=2000, investments=0)
    p = plan([{"id": "m", "head": "status", "option": "start_masters", "start": "2025-01", "commitment": 1},
              {"id": "inv", "head": "assets", "option": "invest_surplus", "repeat": True}])
    ev, _, _ = evaluate(world, people, persona, p, 200, 2)
    trouble = ev["money_trouble"]
    assert trouble["share_of_lives"] > 0.5 and max(trouble["while"], key=trouble["while"].get) == "studying"
    from twin.plan.evaluate import feedback
    notes = " ".join(feedback(p, ev))
    assert "while" in notes and "studying" in notes and "works against the goal" in notes


def test_nudge_edits_touch_only_what_they_name():
    base = plan(GOOD["steps"] + [{"id": "flat", "head": "housing", "option": "bigger", "after": "m",
                                  "after_months": 12}]).to_dict()
    out, errors = apply_edits(base, {"edits": {"remove": ["m", "unknown"], "narrative": "No master's.",
                                               "set": [{"id": "flat", "start": "2027-01", "after": None},
                                                       {"id": "lean", "head": "spending", "option": "cut",
                                                        "cash_months_below": 1}]}})
    assert not errors
    p, errors = LifePlan.from_dict(out)
    assert not errors and [s.id for s in p.steps] == ["cut_if_jobless", "flat", "lean"]
    assert p.step("cut_if_jobless") == plan(GOOD["steps"]).step("cut_if_jobless")
    assert p.step("flat").after is None and p.step("flat").start == "2027-01" and p.narrative == "No master's."
    assert apply_edits(base, {"plan": GOOD}) == (GOOD, [])
    assert apply_edits(base, {"thought": "hm"})[0] is None
    assert apply_edits(base, {"edits": {"set": [{"head": "status"}]}})[0] is None


def test_diff_and_freezing_the_past():
    old = plan([{"id": "invest", "head": "assets", "option": "invest_surplus", "start": "2025-01"},
                {"id": "m", "head": "status", "option": "start_masters", "start": "2026-01", "until": "2027-12"}])
    new = plan([{"id": "flat", "head": "housing", "option": "bigger", "start": "2027-01", "commitment": .6}])
    frozen, kept = freeze_past(old, new, "2026-01")
    assert kept == ["invest"] and [s.id for s in frozen.steps] == ["flat", "invest"]
    assert frozen.step("invest") == old.step("invest")
    d = diff(old, frozen)
    assert len(d["added"]) == 1 and "flat" not in d["removed"][0] and "start_masters" in d["removed"][0]
    assert d["changed"] == [] and d["spontaneity"] is None


def test_reply_parsing_and_private_brief():
    assert parse_reply('<think>{"a": 1} maybe</think>\n```json\n{"ready": true}\n```') == {"ready": True}
    assert parse_reply("no json here") == {} and parse_reply("[1, 2]") == {}
    people, persona = person()
    brief = person_brief(people.iloc[0], persona)
    assert "Mia" not in brief and "27" in brief and "Lausanne" in brief and "Education flexibility" in brief
    assert "you will be 37" in brief and "counts cash only" in brief


GOOD = {"narrative": "I keep my job and save, then do a master's.", "spontaneity": 0.2,
        "steps": [{"id": "m", "label": "Master's", "head": "status", "option": "start_masters", "start": "2027-01",
                   "until": "2028-12", "commitment": 0.8},
                  {"id": "cut_if_jobless", "head": "spending", "option": "cut", "event": "job_loss", "commitment": .7}]}


class Scripted:
    """A chat that replies from a list and records what it was sent."""

    def __init__(self, *replies):
        self.replies, self.sent = list(replies), []

    def __call__(self, messages):
        self.sent.append(copy.deepcopy(messages))
        return self.replies.pop(0)


def reply(plan=GOOD, ready=False, **extra):
    return json.dumps({"thought": "checked", "ready": ready, "plan": plan, **extra})


@needs_model
def test_agent_simulates_before_signing_and_recovers_from_bad_replies(world):
    people, persona = person()
    chat = Scripted("Sure! Here is my plan: none", reply(ready=True), reply(ready=True))
    res = PlannerAgent(world, chat, max_rounds=3, n_paths=100).draft(people, persona, "Master's in a few years")
    assert [bool(r.get("errors")) for r in res.rounds] == [True, False, False]
    assert len(chat.sent) == 3                                  # signed only after seeing a simulation
    last = chat.sent[-1][-1]["content"]
    assert "Simulation of your plan" in last and '"taken_share"' in last
    assert all("Mia" not in m["content"] for m in chat.sent[-1])
    assert "Master's in a few years" in chat.sent[0][1]["content"]
    assert res.plan.step("m").start == "2027-01" and res.evaluation["steps"][0]["typical_year"] in (2027, 2028)
    assert res.signed
    unsigned = PlannerAgent(world, Scripted(reply(), reply()), max_rounds=2, n_paths=100).draft(people, persona)
    assert not unsigned.signed and len(unsigned.rounds) == 2
    with pytest.raises(PlanError):
        PlannerAgent(world, Scripted("{}", "{}"), max_rounds=2, n_paths=100).draft(people, persona)


@needs_model
def test_nudge_revises_keeps_the_past_and_reports_the_diff(world, tmp_path):
    people, persona = person()
    agent = PlannerAgent(world, Scripted(reply(), reply(ready=True)), max_rounds=3, n_paths=100)
    first = agent.draft(people, persona, "")
    path = tmp_path / "mia.json"
    store.append(path, "mia", "draft", "", first)
    invest = {"id": "invest", "head": "assets", "option": "invest_surplus", "start": "2025-01"}
    first.plan.steps.insert(0, LifePlan.from_dict({"narrative": "x", "steps": [invest]})[0].steps[0])
    revised = {**GOOD, "steps": [{"id": "flat", "head": "housing", "option": "bigger", "start": "2027-01"}]}
    edits = {"remove": ["m", "cut_if_jobless", "invest"], "set": [revised["steps"][0]]}
    agent._chat = Scripted(json.dumps({"thought": "t", "edits": edits, "changes": "Dropped the master's, flat in 2027."}),
                           json.dumps({"thought": "t", "ready": True, "edits": {}}))
    res = agent.nudge(people, persona, first.plan, "No master's: I got a promotion.", as_of="2026-01",
                      evaluation=first.evaluation, history=store.history(path))
    prompt = agent._chat.sent[0][1]["content"]
    assert "No master's: I got a promotion." in prompt and "as of 2026-01" in prompt and "Your current plan" in prompt
    assert res.kept_past == ["invest"] and {s.id for s in res.plan.steps} == {"flat", "invest"}
    assert res.changes == "Dropped the master's, flat in 2027." and res.signed
    note = agent._chat.sent[1][-1]["content"]
    assert "Your plan now" in note and "Compared with the plan before this update: removed" in note
    assert any("start_masters" in r for r in res.diff["removed"]) and any("bigger" in a for a in res.diff["added"])
    store.append(path, "mia", "nudge", "No master's: I got a promotion.", res, "2026-01")
    saved, entry = store.latest(path)
    assert saved == res.plan and entry["version"] == 2 and "promotion" in store.history(path)[-1]
    with pytest.raises(ValueError):
        agent.nudge(people, persona, first.plan, "x", as_of="January")


# ------------------------------------------------------------------------------------- API
fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from twin import api  # noqa: E402

KEY = "test-workspace-key-not-a-real-secret"
AUTH = {"Authorization": f"Bearer {KEY}"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("TWIN_API_KEY", KEY)
    monkeypatch.setenv("TOGETHER_API_KEY", "test-provider-key")
    monkeypatch.setenv("TOGETHER_MODEL", "test-model")
    api.plans.clear()
    api.sessions.clear()
    with TestClient(api.app) as c:
        yield c
    api.plans.clear()
    api.sessions.clear()


def body():
    p = {k: v for k, v in MIA.items() if k not in ("id", "med_goal", "med_target")}
    return {"person": p, "traits": {"career_drive": .9}, "wishes": "Master's in 2027, then more pay.", "n_paths": 100}


def settle(client, pid, timeout=120):
    """Poll a plan job like the web client does, until it is no longer working."""
    import time
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        state = client.get(f"/plans/{pid}", headers=AUTH).json()
        if state["status"] != "working":
            return state
        time.sleep(0.2)
    raise AssertionError("plan job did not finish")


@needs_model
def test_plan_endpoints_draft_nudge_and_follow(client, monkeypatch):
    chat = Scripted(reply(), reply(ready=True),
                    reply({**GOOD, "steps": GOOD["steps"][1:]}, changes="No master's."),
                    reply({**GOOD, "steps": GOOD["steps"][1:]}, ready=True))
    monkeypatch.setattr(api, "planner_chat", lambda: chat)
    assert client.post("/plans", json=body()).status_code == 401
    r = client.post("/plans", json=body(), headers=AUTH)
    assert r.status_code == 202, r.text
    pid = r.json()["plan_id"]
    assert r.json()["status"] == "working" and r.json()["latest"] is None
    state = settle(client, pid)
    assert state["status"] == "ready" and state["error"] is None, state
    data = state["latest"]
    assert data["version"] == 1 and data["kind"] == "draft" and data["plan"]["steps"][0]["id"] == "m"
    assert data["signed"] and "summary" not in data and "high_goal_odds" in data["habits"]
    assert data["comparison"]["high_goal_odds"][1] == data["evaluation"]["outcomes"]["high_goal_odds"]
    assert "Master's in 2027" in chat.sent[0][1]["content"]
    bad = client.post(f"/plans/{pid}/nudges", json={"message": "x", "as_of": "2040-01"}, headers=AUTH)
    assert bad.status_code == 422
    r = client.post(f"/plans/{pid}/nudges", json={"message": "Forget the master's.", "as_of": "2026-01"}, headers=AUTH)
    assert r.status_code == 202, r.text
    assert client.post(f"/plans/{pid}/nudges", json={"message": "again"}, headers=AUTH).status_code in (409, 202)
    state = settle(client, pid)
    latest = state["latest"]
    assert latest["version"] == 2 and latest["changes"] == "No master's." and latest["diff"]["removed"]
    assert [v["kind"] for v in state["versions"]] == ["draft", "nudge"]
    assert state["versions"][1]["said"] == "Forget the master's." and state["versions"][1]["as_of"] == "2026-01"

    monkeypatch.setattr(api, "explain", lambda evidence, history, message: "ok")
    session = {k: v for k, v in body().items() if k != "wishes"}
    r = client.post("/sessions", json={**session, "plan_id": pid}, headers=AUTH)
    assert r.status_code == 200, r.text
    evidence = r.json()["evidence"]
    assert evidence["plan"]["steps"] and evidence["plan"]["carried_out"][0]["id"] == "cut_if_jobless"
    assert any("long-term plan" in a for a in evidence["assumptions"])
    assert client.delete(f"/plans/{pid}", headers=AUTH).json()["deleted"]
    assert client.post(f"/plans/{pid}/nudges", json={"message": "hi"}, headers=AUTH).status_code == 404
    assert client.post("/sessions", json={**session, "plan_id": pid}, headers=AUTH).status_code == 404


@needs_model
def test_plan_endpoint_errors(client, monkeypatch):
    monkeypatch.setattr(api, "planner_chat", lambda: Scripted("{}", "{}", "{}", "{}"))
    r = client.post("/plans", json=body(), headers=AUTH)
    state = settle(client, r.json()["plan_id"])
    assert state["status"] == "failed" and "did not produce a usable plan" in state["error"] and state["latest"] is None
    assert client.post(f"/plans/{r.json()['plan_id']}/nudges", json={"message": "hi"}, headers=AUTH).status_code == 409

    def down():
        raise RuntimeError("no provider")
    monkeypatch.setattr(api, "planner_chat", down)
    state = settle(client, client.post("/plans", json=body(), headers=AUTH).json()["plan_id"])
    assert state["status"] == "failed" and "unavailable" in state["error"]
    assert api.capacity.acquire(blocking=False) and api.capacity.acquire(blocking=False)   # both slots released
    try:
        assert client.post("/plans", json=body(), headers=AUTH).status_code == 429
    finally:
        api.capacity.release(), api.capacity.release()
    assert client.post("/plans", json={**body(), "traits": {"x": 1}}, headers=AUTH).status_code == 422
    monkeypatch.delenv("TOGETHER_API_KEY")
    assert client.post("/plans", json=body(), headers=AUTH).status_code == 503


def test_step_why_is_kept_and_rewording_is_no_change():
    from twin.plan.schema import LifePlan, diff
    raw = {"narrative": "I save first.", "steps": [{"id": "cut", "label": "Spend less", "head": "spending", "option": "cut",
                                                   "why": "  Rent is most of my costs.  "}]}
    plan, errors = LifePlan.from_dict(raw)
    assert not errors and plan.steps[0].why == "Rent is most of my costs."
    assert LifePlan.from_dict(plan.to_dict())[0].steps[0].why == plan.steps[0].why
    reworded, _ = LifePlan.from_dict({**raw, "steps": [{**raw["steps"][0], "why": "Other words."}]})
    assert diff(plan, reworded)["changed"] == []
    assert LifePlan.from_dict({**raw, "steps": [{**raw["steps"][0], "why": 3}]})[1]
