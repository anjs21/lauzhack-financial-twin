"""HTTP boundary and evidence grounding tests; no paid provider calls."""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from twin import api

KEY = "test-workspace-key-not-a-real-secret"
AUTH = {"Authorization": f"Bearer {KEY}"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("TWIN_API_KEY", KEY)
    monkeypatch.setenv("TOGETHER_API_KEY", "test-provider-key")
    monkeypatch.setenv("TOGETHER_MODEL", "test-model")
    api.sessions.clear()
    with TestClient(api.app) as client:
        yield client
    api.sessions.clear()


@pytest.fixture
def payload():
    person = json.loads(Path("twin/examples/custom_person.json").read_text())
    for key in ("id", "med_goal", "med_target"):
        person.pop(key)
    return {"person": person, "traits": {"free_time": .8}, "n_paths": 100,
            "scenario": {"kind": "part_time", "salary": 4000}}


def test_auth_and_cors(client, monkeypatch):
    assert client.get("/health").status_code == 401
    assert client.get("/health", headers=AUTH).json()["llm_ready"]
    preflight = client.options("/sessions", headers={"Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization,content-type"})
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert client.options("/sessions", headers={"Origin": "https://untrusted.example",
        "Access-Control-Request-Method": "POST"}).status_code == 400
    monkeypatch.delenv("TWIN_API_KEY")
    assert client.get("/health", headers=AUTH).status_code == 503


def test_open_workspace_needs_no_key(client, monkeypatch):
    monkeypatch.delenv("TWIN_API_KEY")
    monkeypatch.setenv("TWIN_OPEN", "1")
    assert client.get("/health").json()["status"] == "ok"
    monkeypatch.setenv("TWIN_OPEN", "0")                     # anything else keeps the key required
    assert client.get("/health").status_code == 503


@pytest.mark.parametrize("field,value", [("income", -1), ("city", "London"), ("age", 200), ("name", "  "), ("rent", 0),
                                         ("study_months_left", 0), ("study_months_left", 100), ("in_masters", "maybe")])
def test_person_validation(client, payload, field, value):
    payload["person"][field] = value
    assert client.post("/sessions", json=payload, headers=AUTH).status_code == 422


@pytest.mark.parametrize("traits", [{"free_time": 2}, {"unknown": .5}])
def test_trait_validation(client, payload, traits):
    payload["traits"] = traits
    assert client.post("/sessions", json=payload, headers=AUTH).status_code == 422


def test_llm_required_before_simulation(client, payload, monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY")
    monkeypatch.setattr(api, "run_projection", lambda _: pytest.fail("Must not simulate without an explanation provider"))
    assert client.post("/sessions", json=payload, headers=AUTH).status_code == 503


def test_real_projection_and_conversation_grounding(client, payload, monkeypatch):
    calls = []
    def explain(evidence, history, message):
        calls.append((evidence, list(history), message))
        return "The goal success estimate is conditional on this model. What matters most to you?"
    monkeypatch.setattr(api, "explain", explain)
    response = client.post("/sessions", json=payload, headers=AUTH)
    assert response.status_code == 200, response.text
    data = response.json()
    evidence = data["evidence"]
    assert evidence["traits"]["free_time"] == .8
    assert len(evidence["baseline"]["months"]) == 121
    assert evidence["baseline"]["fan_cash"][2][0] == payload["person"]["cash"]
    assert evidence["alternative"]["fan_income"][2][1] >= 4000
    assert evidence["comparison"]["high_goal_odds"][0] == evidence["baseline"]["high_goal_odds"]
    assert evidence["comparison"]["high_goal_odds"][1] == evidence["alternative"]["high_goal_odds"]
    sid = data["session_id"]
    seen = []
    def complete(messages, tools=None):
        seen.append((messages, tools))
        return SimpleNamespace(content="Grounded in the stored run.", tool_calls=None)
    monkeypatch.setattr(api, "complete", complete)
    answer = client.post(f"/sessions/{sid}/messages", json={"message": "What if I earned more?"}, headers=AUTH)
    assert answer.status_code == 200 and answer.json() == {"answer": "Grounded in the stored run.", "actions": [], "notes": []}
    messages, tools = seen[0]
    assert json.loads(messages[1]["content"].split(": ", 1)[1])["baseline"]["high_goal_odds"] == evidence["baseline"]["high_goal_odds"]
    assert [m["role"] for m in messages[2:]] == ["user", "assistant", "user"]       # history, then the new message
    assert {t["function"]["name"] for t in tools} == {"update_simulation", "explore_decision", "highlight_path"}
    assert client.post(f"/sessions/{sid}/messages", json={"message": " "}, headers=AUTH).status_code == 422
    assert client.post(f"/sessions/{sid}/messages", json={"message": "hello", "evidence": {}}, headers=AUTH).status_code == 422
    api.sessions[sid].expires = time.monotonic() - 1
    assert client.post(f"/sessions/{sid}/messages", json={"message": "Hello"}, headers=AUTH).status_code == 404
    assert sid not in api.sessions


def test_deleted_session_cannot_be_used(client):
    api.sessions["session"] = api.Session({})
    assert client.delete("/sessions/session", headers=AUTH).status_code == 200
    assert client.post("/sessions/session/messages", json={"message": "hello"}, headers=AUTH).status_code == 404


def test_provider_receives_sanitized_context_and_history(monkeypatch):
    seen = {}
    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def create(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="A grounded explanation."))])
    monkeypatch.setenv("TOGETHER_API_KEY", "test")
    monkeypatch.setenv("TOGETHER_MODEL", "test-model")
    monkeypatch.setattr(api, "OpenAI", Client)
    evidence = {"person": {"name": "Private nickname", "income": 5000},
                "baseline": {"months": ["2025-01"], "fan_cash": [[1]], "high_goal_odds": .5}, "alternative": None}
    result = api.explain(evidence, [{"role": "user", "content": "Earlier question"}], "Why?")
    assert result == "A grounded explanation."
    context = seen["messages"][1]["content"]
    assert "Private nickname" not in context and "fan_cash" not in context
    assert "high_goal_odds" in context
    assert seen["messages"][-2]["content"] == "Earlier question"
    assert evidence["person"]["name"] == "Private nickname"


def test_busy_laptop_returns_retryable_error(client, payload):
    api.capacity.acquire()
    api.capacity.acquire()
    try:
        assert client.post("/sessions", json=payload, headers=AUTH).status_code == 429
    finally:
        api.capacity.release()
        api.capacity.release()


def test_housing_intervention_respects_existing_model_limits(payload):
    payload["scenario"] = {"kind": "cheaper_home"}
    payload["person"]["household_type"] = "family"
    result = api.run_projection(api.SessionRequest(**payload))
    assert result["alternative"] is not None
    payload["person"]["household_type"] = "shared_housing"
    with pytest.raises(api.HTTPException) as error:
        api.run_projection(api.SessionRequest(**payload))
    assert error.value.status_code == 422


def test_deliberate_persona_uses_foresight_on_both_paths(payload):
    habitual = api.run_projection(api.SessionRequest(**payload))
    deliberate = api.run_projection(api.SessionRequest(**payload, deliberate=True))
    assert habitual["decision_style"] == "habitual" and not habitual["deliberate"]
    assert deliberate["decision_style"] == "deliberate" and deliberate["deliberate"]
    assert deliberate["baseline"]["final_net_worth"] != habitual["baseline"]["final_net_worth"]
    assert deliberate["alternative"]["final_net_worth"] != habitual["alternative"]["final_net_worth"]
    assert any("deliberate" in a for a in deliberate["assumptions"])
    with pytest.raises(ValueError):
        api.SessionRequest(**payload, deliberate="maybe")


def test_part_time_alongside_studies_is_validated_and_used(payload):
    payload["person"].update(status="student", study_months_left=24, in_masters=True)
    payload["scenario"]["alongside_studies"] = True
    result = api.run_projection(api.SessionRequest(**payload))
    assert result["scenario"]["alongside_studies"] is True
    assert result["alternative"]["key_moments"][0]["moment"] == "Takes a job alongside studies"
    payload["scenario"]["alongside_studies"] = "maybe"
    with pytest.raises(ValueError):
        api.SessionRequest(**payload)


def test_llm_calls_go_to_together(monkeypatch):
    from twin import llm_provider
    monkeypatch.setenv("TOGETHER_API_KEY", "k")
    monkeypatch.setenv("TOGETHER_MODEL", "some/model")
    monkeypatch.delenv("TOGETHER_BASE_URL", raising=False)
    seen = {}
    class Client:
        def __init__(self, **kwargs):
            seen["init"] = kwargs
            self.chat = SimpleNamespace(completions=self)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def create(self, **kwargs):
            seen["create"] = kwargs
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    monkeypatch.setattr(api, "OpenAI", Client)
    api.explain({"person": {"name": "x"}, "baseline": {}, "alternative": None}, [], "Why?")
    assert seen["init"]["base_url"] == llm_provider.BASE_URL == "https://api.together.xyz/v1"
    assert seen["init"]["api_key"] == "k" and seen["create"]["model"] == "some/model"
    assert "max_tokens" in seen["create"] and "max_completion_tokens" not in seen["create"]
    monkeypatch.delenv("TOGETHER_MODEL")
    assert not api.llm_ready()


def test_persona_demos_fall_back_without_json_mode(tmp_path, monkeypatch):
    from twin.policy.llm import Client, parse_json
    attempts = []
    class NoJsonMode:
        chat = property(lambda self: self)
        completions = property(lambda self: self)
        def create(self, **kw):
            attempts.append(sorted(kw))
            if "response_format" in kw:
                raise ValueError("This model does not support response_format json mode")
            msg = SimpleNamespace(content='Sure: ```json\n{"status": {"0": 1}}\n```')
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])
    c = Client("some/model", cache=tmp_path / "c.jsonl", sdk_client=NoJsonMode())
    assert c.ask("hello") == {"status": {"0": 1}} and len(attempts) == 2
    assert parse_json("no json here") == {}


def test_branches_fork_from_a_root_at_a_later_review(client, payload, monkeypatch):
    monkeypatch.setattr(api, "explain", lambda evidence, history, message: "A grounded explanation.")
    session = client.post("/sessions", json=payload, headers=AUTH).json()
    sid, evidence = session["session_id"], session["evidence"]
    branch = client.post(f"/sessions/{sid}/branches", json={"parent": "alternative", "month": 24, "choice": "full_time"},
                         headers=AUTH)
    assert branch.status_code == 200
    b = branch.json()
    assert b["id"] == "b1" and b["parent"] == "alternative" and b["label"] == "Work full-time · 2027"
    for k in ("fan_cash", "fan_net_worth"):                    # the same lives up to the fork
        assert np.allclose(np.array(b["summary"][k])[:, :25], np.array(evidence["alternative"][k])[:, :25])
    assert 0 <= b["applied"] <= 1 and b["summary"]["fan_net_worth"] != evidence["alternative"]["fan_net_worth"]
    deeper = client.post(f"/sessions/{sid}/branches", json={"parent": "b1", "month": 48, "choice": "spend_less"}, headers=AUTH)
    assert deeper.status_code == 200 and [d["month"] for d in deeper.json()["decisions"]] == [24, 48]
    assert set(api.sessions[sid].branches) == {"b1", "b2"}                                # known to the assistant
    for bad in ({"parent": "alternative", "month": 30, "choice": "full_time"},             # not a January
                {"parent": "b2", "month": 36, "choice": "invest"},                         # before the parent's decision
                {"parent": "alternative", "month": 24, "choice": "win_lottery"},
                {"parent": "nowhere", "month": 24, "choice": "invest"}):
        assert client.post(f"/sessions/{sid}/branches", json=bad, headers=AUTH).status_code == 422
    assert client.delete(f"/sessions/{sid}/branches/b1", headers=AUTH).json() == {"deleted": ["b1", "b2"]}
    assert api.sessions[sid].branches == {}
    assert client.post("/sessions/missing/branches", json={"parent": "baseline", "month": 24, "choice": "invest"},
                       headers=AUTH).status_code == 404
    payload["scenario"] = {"kind": "baseline"}
    plain = client.post("/sessions", json=payload, headers=AUTH).json()["session_id"]
    assert client.post(f"/sessions/{plain}/branches", json={"parent": "alternative", "month": 24, "choice": "invest"},
                       headers=AUTH).status_code == 422


def test_provider_refusals_explain_what_to_change(monkeypatch):
    import httpx
    from openai import PermissionDeniedError
    monkeypatch.setenv("TOGETHER_API_KEY", "k")
    monkeypatch.setenv("TOGETHER_MODEL", "Qwen/Qwen3.8-Flash")
    refusal = {"error": {"message": "This model sends your prompts to a third-party provider, and your organization has "
                                    "not enabled third-party data sharing."}}
    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def create(self, **kwargs):
            response = httpx.Response(403, request=httpx.Request("POST", "https://api.together.xyz/v1/chat/completions"))
            raise PermissionDeniedError("refused", response=response, body=refusal)
    monkeypatch.setattr(api, "OpenAI", Client)
    with pytest.raises(api.HTTPException) as error:
        api.explain({"person": {"name": "x"}, "baseline": {}, "alternative": None}, [], "Why?")
    detail = error.value.detail
    assert error.value.status_code == 502 and "(403)" in detail and "third-party" in detail
    assert "served by Together" in detail and "k" != detail



def test_assistant_tools_turn_words_into_simulation_changes(client, payload, monkeypatch):
    monkeypatch.setattr(api, "explain", lambda evidence, history, message: "Ready.")
    sid = client.post("/sessions", json=payload, headers=AUTH).json()["session_id"]
    call = lambda i, name, args: SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))  # noqa: E731
    script = [
        [call(1, "update_simulation", {"person": {"age": 31, "income": 7000}, "traits": {"free_time": 0.2}})],
        [call(2, "explore_decision", {"from_path": "alternative", "year": 2028, "choice": "full_time"}),
         call(3, "explore_decision", {"from_path": "b1", "year": 2030, "choice": "bigger_home"}),
         call(4, "highlight_path", {"path_id": "b2", "measure": "net_worth"})],
        [call(5, "explore_decision", {"from_path": "b2", "year": 2029, "choice": "invest"}),          # before b2's decision
         call(6, "update_simulation", {"person": {"city": "Paris"}}),                                  # unsupported
         call(7, "highlight_path", {"path_id": "b9"})],
        None,
    ]
    tool_results = []
    def complete(messages, tools=None):
        tool_results.extend(json.loads(m["content"]) for m in messages if m["role"] == "tool")
        step = script.pop(0)
        return SimpleNamespace(content="Done." if step is None else "", tool_calls=step)
    monkeypatch.setattr(api, "complete", complete)
    reply = client.post(f"/sessions/{sid}/messages", json={"message": "I'm 31, take home 7k ..."}, headers=AUTH).json()
    assert reply["answer"] == "Done."
    assert [a["type"] for a in reply["actions"]] == ["rerun", "branch", "branch", "highlight"]
    assert reply["actions"][0]["evidence"]["person"]["income"] == 7000
    assert reply["actions"][2]["branch"]["parent"] == "b1" and reply["actions"][3] == {"type": "highlight", "path_id": "b2", "measure": "net_worth"}
    assert reply["notes"][0].startswith("Updated age 31; income CHF 7,000; free time 0.2")
    s = api.sessions[sid]
    assert s.request.person.age == 31 and s.request.traits["free_time"] == 0.2 and s.request.person.name == payload["person"]["name"]
    assert set(s.branches) == {"b1", "b2"}
    errors = [r["error"] for r in tool_results if "error" in r]
    assert len(errors) == 3 and any("later review" in e for e in errors) and any("city" in e for e in errors)
    assert any("Unknown path" in e for e in errors)


def test_life_events_branch_from_now_and_decisions_from_2026(client, payload, monkeypatch):
    monkeypatch.setattr(api, "explain", lambda evidence, history, message: "Ready.")
    sid = client.post("/sessions", json=payload, headers=AUTH).json()["session_id"]
    now = client.post(f"/sessions/{sid}/branches", json={"parent": "baseline", "month": 0, "choice": "lose_job"}, headers=AUTH)
    assert now.status_code == 200 and now.json()["label"] == "Lose my job · 2025" and now.json()["applied"] > 0.5
    too_early = client.post(f"/sessions/{sid}/branches", json={"parent": "baseline", "month": 0, "choice": "full_time"}, headers=AUTH)
    assert too_early.status_code == 422 and "2026" in too_early.text
    then = client.post(f"/sessions/{sid}/branches", json={"parent": "b1", "month": 24, "choice": "recession"}, headers=AUTH)
    assert then.status_code == 200 and then.json()["applied"] == 1.0
    result, action, note = api.run_tool(api.sessions[sid], "explore_decision",
                                        json.dumps({"from_path": "baseline", "year": 2025, "choice": "lose_job"}))
    assert action["type"] == "branch" and note == "Grew a branch: Lose my job · 2025"


def test_compare_runs_a_second_scenario_as_its_own_path(payload):
    payload["person"].update(status="student", study_months_left=14, in_masters=True, income=2000, rent=950)
    payload["scenario"] = {"kind": "part_time", "salary": 3200, "alongside_studies": True}
    alone = api.run_projection(api.SessionRequest(**payload))
    request = api.SessionRequest(**payload, compare={"kind": "part_time", "salary": 3200, "alongside_studies": False})
    both = api.run_projection(request)
    assert both["alternative"]["final_net_worth"] == alone["alternative"]["final_net_worth"]   # untouched by the second
    assert both["compare_scenario"]["alongside_studies"] is False
    assert both["compare"]["final_net_worth"]["p50"] < both["alternative"]["final_net_worth"]["p50"]   # leaving the course
    branch = api.run_branch(request, api.BranchRequest(root="compare", decisions=[{"month": 24, "choice": "full_time"}]))
    assert branch["root"] == "compare"
    for bad in ({"kind": "part_time", "salary": 3200, "alongside_studies": True}, {"kind": "baseline"}):
        with pytest.raises(api.ValidationError):
            api.SessionRequest(**payload, compare=bad)
    session = api.Session(evidence=both, history=[], request=request)
    labels = {p["id"]: p["label"] for p in api.paths_of(session)}
    assert labels["alternative"].startswith("Job alongside studies") and labels["compare"].startswith("Job instead of studies")


def test_plan_explanation_runs_for_the_sessions_plan(client, payload):
    from twin.plan.schema import LifePlan
    example = json.loads(Path("web/src/example.json").read_text())
    versions = json.loads(Path("web/src/example-plan.json").read_text())["history"]
    request = api.SessionRequest(person=example["person"], traits=example["traits"], n_paths=100)
    plan = LifePlan.from_dict(versions[-1]["plan"])[0]
    with_plan = api.Session(evidence={}, request=request, plan=plan, plan_versions=versions)
    api.sessions.update(with_plan=with_plan, without=api.Session(evidence={}, request=request))
    report = api.run_explanation(with_plan, n_paths=100)
    assert report["paths"] == 100 and report["person"]["name"] == "P0067"
    assert [s["id"] for s in report["execution"]["step_report"]] == [s.id for s in plan.steps]
    assert {"plan", "habits", "plan_minus_habits", "offer_stress"} <= set(report["outcomes"])
    assert report["execution"]["opening_state_intervention"]["cash_comparisons"][0]["cash_chf"] == 3000
    with_plan.explanation = report                                         # computed once, then served
    assert client.get("/sessions/with_plan/explanation", headers=AUTH).json()["paths"] == 100
    assert client.get("/sessions/without/explanation", headers=AUTH).status_code == 409
