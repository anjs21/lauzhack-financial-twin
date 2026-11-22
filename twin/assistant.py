"""Tools the assistant can call to change the simulation from natural language.

The model turns what the user says into the simulator's variables; the laptop API (twin/api.py)
validates them, runs the simulation and sends the client the updates for its chart:

    update_simulation   new facts about the person, their preferences, the scenario or the decision style
                        -> re-run every life (branches are cleared)
    explore_decision    a what-if at a later January review, from any path -> a new branch in the tree
    highlight_path      what the chart shows: which path, and cash or net worth

Schemas follow the OpenAI function-calling format, which Together's API accepts.
"""
from __future__ import annotations

from . import config as C
from .policy.traits import QUESTIONS, TRAITS

STATUS = {"employed": "working full-time", "part_time": "working part-time", "self_employed": "self-employed",
          "student": "studying", "unemployed": "not working", "retired": "retired"}
HOUSEHOLD = {"single": "living alone", "shared_housing": "sharing a flat", "couple": "living with a partner",
             "family": "family with children"}
GOALS = ["Emergency savings", "Housing deposit", "Education flexibility", "Career transition", "Retirement reserve"]
SCENARIO = {"baseline": "no change at the January 2025 review (only the current path)",
            "part_time": "a part-time job offer at the January 2025 review (needs salary)",
            "spend_less": "cut lifestyle spending at the January 2025 review",
            "cheaper_home": "move one housing tier cheaper at the January 2025 review"}


def _money(text):
    return {"type": "number", "minimum": 0, "description": f"{text}, CHF (numbers only: '7k' -> 7000)"}


def tools(choices: dict) -> list:
    """Function schemas; `choices` is api.CHOICES (key -> (head, option, label))."""
    person = {
        "age": {"type": "integer", "minimum": 18, "maximum": 90, "description": "age in January 2025"},
        "city": {"type": "string", "enum": C.CITIES, "description": "Swiss city (nearest supported one)"},
        "household_type": {"type": "string", "enum": list(HOUSEHOLD),
                           "description": "; ".join(f"{k}: {v}" for k, v in HOUSEHOLD.items())},
        "status": {"type": "string", "enum": list(STATUS), "description": "; ".join(f"{k}: {v}" for k, v in STATUS.items())},
        "income": _money("monthly take-home (net) income"),
        "rent": _money("monthly rent"),
        "cash": _money("cash and savings now"),
        "investments": _money("investments now"),
        "debt": _money("outstanding debt"),
        "rate": {"type": "number", "minimum": 0, "maximum": 0.5, "description": "annual debt interest as a fraction (5% -> 0.05)"},
        "high_goal": {"type": "string", "enum": GOALS, "description": "main money goal"},
        "high_target": _money("amount the main goal needs by December 2029"),
        "study_months_left": {"type": "integer", "minimum": 1, "maximum": 72,
                              "description": "students only: months until the current course ends"},
        "in_masters": {"type": "boolean", "description": "students only: the current course is a master's"},
    }
    traits = {t: {"type": "number", "minimum": 0, "maximum": 1,
                  "description": f"0 = disagree, 1 = agree: '{QUESTIONS[t]}'"} for t in TRAITS}
    scenario = {
        "kind": {"type": "string", "enum": list(SCENARIO), "description": "; ".join(f"{k}: {v}" for k, v in SCENARIO.items())},
        "salary": _money("part-time offer: monthly take-home pay"),
        "alongside_studies": {"type": "boolean", "description": "part-time offer for a student: keep studying alongside it"},
    }
    choice = {k: f"{label}" for k, (_, _, label) in choices.items()}
    return [
        {"type": "function", "function": {
            "name": "update_simulation",
            "description": ("Change facts about the person, their preferences, the scenario explored at the January 2025 "
                            "review, or the decision style, then re-run every simulated life (the chart updates; "
                            "branches are cleared). Use it when the user states new facts about themselves ('I earn 7k "
                            "now', 'I'm 31', 'I moved to Zurich', 'I'm doing a master's with 18 months left', 'I care "
                            "more about free time') or asks to change what is explored ('try part-time at 3,500'). "
                            "Include only what changes. Do not use it for questions."),
            "parameters": {"type": "object", "additionalProperties": False, "properties": {
                "person": {"type": "object", "additionalProperties": False, "properties": person},
                "traits": {"type": "object", "additionalProperties": False, "properties": traits},
                "scenario": {"type": "object", "additionalProperties": False, "properties": scenario},
                "deliberate": {"type": "boolean", "description": "true: decide deliberately (weigh each yearly choice "
                                                                 "by its five-year value); false: as the person usually does"},
            }}}},
        {"type": "function", "function": {
            "name": "explore_decision",
            "description": ("Grow a what-if branch: start from an existing path, impose one decision or life event at a "
                            "January review and simulate the rest of every life on the same random draws. The chart "
                            "shows the new branch. Decisions are the person's choices (from 2026); life events happen "
                            "to them: lose_job, recession, have_child, partner, separate (from 2025; 'now' = 2025). "
                            "Use it for 'what if I ... in 2028?' or 'what if I lose my job now?'; for 'and then ...' "
                            "branch from the branch you just made. The year must be later than the path's last "
                            "decision or event. Call it once per decision or event."),
            "parameters": {"type": "object", "additionalProperties": False, "required": ["from_path", "year", "choice"],
                           "properties": {
                "from_path": {"type": "string", "description": "id of the path to branch from (see the paths list)"},
                "year": {"type": "integer", "minimum": C.START_YEAR, "maximum": C.START_YEAR + 9,
                         "description": "the January when it happens: decisions 2026-2034, life events 2025-2034"},
                "choice": {"type": "string", "enum": list(choices),
                           "description": "; ".join(f"{k}: {v}" for k, v in choice.items())},
            }}}},
        {"type": "function", "function": {
            "name": "highlight_path",
            "description": "Show a path on the chart (shaded range, key moments), and optionally switch between cash and net worth.",
            "parameters": {"type": "object", "additionalProperties": False, "required": ["path_id"], "properties": {
                "path_id": {"type": "string", "description": "id from the paths list"},
                "measure": {"type": "string", "enum": ["cash", "net_worth"]},
            }}}},
    ]


TOOL_RULES = """You can change the simulation with tools. Translate what the user says into the simulator's
variables: monthly amounts in CHF ('7k' -> 7000), rates as fractions, years as January reviews.
- New facts about the user, their preferences or what to explore -> update_simulation (only what changes).
- A what-if decision in a later year, or a life event that happens to them ('what if I lose my job',
  'a recession', 'we have a baby') -> explore_decision, from the path the user means (default: the
  highlighted path, else the alternative, else the current path). 'Now' or 'today' means January 2025,
  the start of the simulation; events can be imposed then, decisions from 2026. Chain branches for 'and then'.
- To point at something on the chart -> highlight_path.
- Pure questions need no tool. If a request is ambiguous or outside the tools (e.g. a year outside
  2026-2034 or an unsupported city), ask a short question instead of guessing.
After tools run, explain what changed using only the tool results and the evidence, and keep the reply short."""
