"""HTTP boundary and evidence grounding tests; no paid provider calls."""
import json
import time
from pathlib import Path
from types import SimpleNamespace

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
    answer = client.post(f"/sessions/{sid}/messages", json={"message": "What if I earned more?"}, headers=AUTH)
    assert answer.status_code == 200
    assert calls[1][0] is calls[0][0]  # Server evidence, not a client-submitted summary.
    assert calls[1][1][0]["role"] == "user"
    assert calls[1][1][1]["role"] == "assistant"
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
