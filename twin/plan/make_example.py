"""Regenerate the web UI's example: a fictional student, P0067 (examples/demo-character-elif.md).

    python -m twin.plan.make_example --projection-only                  # web/src/example.json
    TOGETHER_API_KEY=... TOGETHER_MODEL=... python -m twin.plan.make_example   # and example-plan.json

example.json is the projection POST /sessions returns for REQUEST (500 lives, the fixed seed).
For the plan, the agent drafts one from WISHES, then revises it after NUDGE, exactly as the API's
plan jobs do; both versions are written in the shape GET /plans/{id} returns, so the UI shows them
unchanged.
"""
from __future__ import annotations

import json
import sys

from .. import api
from .. import config as C
from .. import llm_provider as provider
from .schema import LifePlan

EXAMPLE = C.ROOT / "web" / "src" / "example.json"
OUT = C.ROOT / "web" / "src" / "example-plan.json"
REQUEST = dict(
    person=dict(name="P0067", age=24, city="Zurich", status="student", household_type="shared_housing",
                income=2000, rent=950, cash=9000, investments=0, debt=4000, rate=0.04,
                high_goal="Housing deposit", high_target=40000, study_months_left=14, in_masters=True),
    traits=dict(risk_tolerance=0.55, frugality=0.6, career_drive=0.85, free_time=0.35, housing_aspiration=0.6,
                inertia=0.3, impatience=0.45, goal_focus=0.7),
    scenario=dict(kind="part_time", salary=3200, alongside_studies=True),
    n_paths=500)
WISHES = ("I'll keep the start-up job at 60% while I finish my master's, then go full-time there. My partner and I "
          "want to move in together in 2027 and find something bigger once we both earn properly. I want the "
          "deposit by 2030. If money gets tight I'd rather cut spending than touch investments.")
NUDGE = ("The start-up is cutting everyone to 40% next year, so I'll look for a full-time job elsewhere after "
         "graduating instead.")
AS_OF = "2026-01"


def projection():
    result = api.run_projection(api.SessionRequest(**REQUEST))
    EXAMPLE.write_text(json.dumps(result, allow_nan=False, separators=(",", ":")))
    print(f"wrote {EXAMPLE} ({EXAMPLE.stat().st_size // 1024} kB)")


def main(n_paths=300):
    example = json.loads(EXAMPLE.read_text())
    person = api.Person(**example["person"])
    traits = example["traits"]
    record = api.PlanRecord(person, traits, n_paths)
    people, persona = api.persona_for(person, traits)
    agent = api.PlannerAgent(api.world(), chat=api.planner_chat(), n_paths=n_paths, log=print)
    draft = agent.draft(people, persona, WISHES)
    first = api.plan_entry(record, draft, people, persona)
    first.update(kind="draft", said=WISHES, as_of=None)
    record.versions.append(first)
    plan = LifePlan.from_dict(first["plan"])[0]
    revised = agent.nudge(people, persona, plan, NUDGE, AS_OF, first["evaluation"], [])
    second = api.plan_entry(record, revised, people, persona)
    second.update(kind="nudge", said=NUDGE, as_of=AS_OF)
    record.versions.append(second)
    state = {"status": "ready", "progress": "", "error": None, "latest": second,
             "versions": [{k: v.get(k) for k in ("version", "kind", "said", "as_of")} for v in record.versions],
             "history": record.versions, "model": provider.model()}
    OUT.write_text(json.dumps(state, allow_nan=False, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} kB): v1 signed={first['signed']}, v2 signed={second['signed']}")


if __name__ == "__main__":
    projection()
    if "--projection-only" not in sys.argv:
        main()
