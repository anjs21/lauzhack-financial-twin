"""Regenerate web/src/example-plan.json: a real planner run for the web UI's example persona (Mia).

    TOGETHER_API_KEY=... TOGETHER_MODEL=... python -m twin.plan.make_example

The agent drafts a plan from WISHES, then revises it after NUDGE, exactly as the API's plan jobs
do; both versions are written in the shape GET /plans/{id} returns, so the UI shows them unchanged.
"""
from __future__ import annotations

import json

from .. import api
from .. import config as C
from .. import llm_provider as provider
from .schema import LifePlan

EXAMPLE = C.ROOT / "web" / "src" / "example.json"
OUT = C.ROOT / "web" / "src" / "example-plan.json"
WISHES = ("I'd like to start a master's in data science next year, then go back to full-time work and move "
          "somewhere bigger. If money gets tight I'd rather cut back than touch my investments.")
NUDGE = "Change of plans: my employer offered me a promotion if I stay, so no master's. I'd still like a bigger flat, around 2027."
AS_OF = "2026-01"


def main(n_paths=300):
    example = json.loads(EXAMPLE.read_text())
    person = api.Person(**{**example["person"], "study_months_left": 1, "in_masters": False})
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
    main()
