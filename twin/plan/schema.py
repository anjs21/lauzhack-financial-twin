"""A person's long-term plan: what the planner agent writes and what the simulator carries out.

A plan is a short first-person narrative plus steps. Each step is one voluntary choice
(work, home, lifestyle or savings) with a window, a commitment (the chance the person
follows through when the step is possible) and optional triggers: another step done
first, or a life event (a contingency such as "if I lose my job, I cut back").

`LifePlan.intentions()` compiles it into the rule persona's Intention list. Between and
around the steps the person's habits still decide, damped by `spontaneity`: the chance
that an unplanned change of work or home the habits would make actually happens.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields

from .. import config as C
from ..policy.traits import EVENTS, Intention

OPTIONS = {
    "status": {"continue": 0, "start_masters": 1, "part_time": 2, "full_time": 3, "stop_working": 4},
    "housing": {"cheaper": -1, "stay": 0, "bigger": 1},
    "spending": {"cut": -1, "keep": 0, "raise": 1},
    "assets": {"sell_to_buffer": -1, "hold": 0, "invest_surplus": 1},
}
OPTION_TEXT = {
    "status": {"continue": "keep the current work or study situation",
               "start_masters": "start a full-time master's (24 months, tuition, small student income; "
                                "needs no master's yet, age 45 or under, and any current course in its last month)",
               "part_time": "work part-time: same job with fewer hours if working, otherwise start looking "
                            "for a part-time job (it begins when an offer comes; for a student, taking it ends "
                            "the course)",
               "full_time": "work full-time: same job with more hours if working, otherwise start looking "
                            "for a full-time job (it begins when an offer comes; for a student, taking it ends "
                            "the course)",
               "stop_working": "leave paid work (break, quit or retire; only while working)"},
    "housing": {"cheaper": "move one step cheaper (family flat -> small flat -> shared flat; CHF 2,000 to move)",
                "stay": "stay", "bigger": "move one step bigger (CHF 2,000 to move; a landlord must accept "
                                          "the income or savings)"},
    # Either move opens a flat search that can take months; it lasts until a flat is found or the next review.
    "spending": {"cut": "cut lifestyle (about 10% less living costs): the way to save more cash", "keep": "keep lifestyle",
                 "raise": "raise lifestyle (about 10% more living costs)"},
    "assets": {"sell_to_buffer": "sell investments to bring cash back up to the buffer",
               "hold": "leave savings as they are",
               "invest_surplus": "move the cash above the buffer into investments (lowers cash; it does not build "
                                 "an emergency fund, which counts cash only)"},
}
# Keeping things as they are is what happens without a step; spontaneity says how firmly.
NOOP = {"continue", "stay", "keep", "hold"}
EVENT_TEXT = {"job_loss": "you lose your job", "offer": "you get a job offer and start the job",
              "graduated": "you finish your studies", "partner": "a partner moves in", "child": "you have a child",
              "separation": "you separate", "windfall": "you receive a windfall such as an inheritance or gift"}
FIRST, LAST = C.month_label(0), C.month_label(C.HORIZON - 1)
MAX_STEPS = 16
MONTH = re.compile(r"^\d{4}-\d{2}$")


def _month(value, name, errors):
    if value is None:
        return None
    if not isinstance(value, str) or not MONTH.match(value) or not 1 <= int(value[5:]) <= 12:
        errors.append(f"{name} must be a month written YYYY-MM, got {value!r}")
        return None
    if not FIRST <= value <= LAST:
        errors.append(f"{name} {value} is outside the plan's horizon {FIRST} to {LAST}")
        return None
    return value


@dataclass
class Step:
    id: str
    label: str
    head: str
    option: str
    start: str | None = None           # first month the step may happen (None: from now)
    until: str | None = None           # last month it may still happen
    commitment: float = 0.8            # chance of following through when the step is possible
    after: str | None = None           # id of a step that must be done first ...
    after_months: int = 0              # ... at least this many months earlier
    event: str | None = None           # a life event that must have happened ...
    within_months: int = 12            # ... in the last this many months
    cash_months_below: float | None = None
    cash_months_above: float | None = None
    min_age: float | None = None
    max_age: float | None = None
    repeat: bool = False               # act at every review where it applies, not just once

    @property
    def contingent(self):
        return self.event is not None or self.cash_months_below is not None

    def window(self):
        """(start, until) as compiled: a dated step without an end gets its own year's review;
        a step waiting on another step or event stays open to the end of the horizon."""
        until = self.until
        if until is None:
            if self.start is not None and self.after is None and self.event is None and not self.repeat:
                review = C.next_review(C.month_index(self.start))
                until = C.month_label(min(review + 11, C.HORIZON - 1))
            else:
                until = LAST
        return self.start, until

    def intention(self) -> Intention:
        start, until = self.window()
        return Intention(self.head, OPTIONS[self.head][self.option], prob=self.commitment, when=start, until=until,
                         id=self.id, after=self.after, after_months=self.after_months, event=self.event,
                         within_months=self.within_months, cash_months_below=self.cash_months_below,
                         cash_months_above=self.cash_months_above, min_age=self.min_age, max_age=self.max_age,
                         once=not self.repeat)

    def describe(self) -> str:
        start, until = self.window()
        when = f"{start or 'now'} to {until}"
        bits = [f"{self.label} [{self.head}: {self.option}], {when}, commitment {self.commitment:.0%}"]
        if self.after:
            bits.append(f"after '{self.after}'" + (f" by {self.after_months} months" if self.after_months else ""))
        if self.event:
            bits.append(f"if {EVENT_TEXT[self.event]} (within {self.within_months} months)")
        if self.cash_months_below is not None:
            bits.append(f"if cash < {self.cash_months_below:g} months of outgoings")
        if self.cash_months_above is not None:
            bits.append(f"if cash > {self.cash_months_above:g} months of outgoings")
        if self.repeat:
            bits.append("every year")
        return ", ".join(bits)


STEP_FIELDS = {f.name for f in fields(Step)}


@dataclass
class LifePlan:
    narrative: str
    steps: list[Step] = field(default_factory=list)
    spontaneity: float = 0.35
    notes: list[str] = field(default_factory=list)

    def intentions(self) -> list[Intention]:
        """Habits are held back by (1 - spontaneity) for work and home; plan steps come after and win."""
        hold = 1.0 - self.spontaneity
        return ([Intention("status", 0, prob=hold, once=False, id="_hold_status"),
                 Intention("housing", 0, prob=hold, once=False, id="_hold_housing")]
                + [s.intention() for s in self.steps])

    def to_dict(self) -> dict:
        d = asdict(self)
        d["steps"] = [{k: v for k, v in s.items() if not (v is None or v is False)} for s in d["steps"]]
        for s in d["steps"]:
            for k, default in (("after_months", 0), ("within_months", 12)):
                if s.get(k) == default:
                    s.pop(k)
        return d

    def step(self, step_id) -> Step | None:
        return next((s for s in self.steps if s.id == step_id), None)

    @classmethod
    def from_dict(cls, d) -> tuple["LifePlan | None", list[str]]:
        """Parse and validate; returns (plan, []) or (None, readable errors for the agent)."""
        errors: list[str] = []
        if not isinstance(d, dict):
            return None, ["the plan must be a JSON object with narrative, spontaneity and steps"]
        narrative = d.get("narrative")
        if not isinstance(narrative, str) or not narrative.strip():
            errors.append("narrative must be a non-empty first-person summary")
            narrative = ""
        spont = d.get("spontaneity", 0.35)
        if not isinstance(spont, (int, float)) or isinstance(spont, bool) or not 0 <= spont <= 1:
            errors.append(f"spontaneity must be a number from 0 to 1, got {spont!r}")
            spont = 0.35
        notes = d.get("notes") or []
        if not isinstance(notes, list) or not all(isinstance(n, str) for n in notes):
            errors.append("notes must be a list of short strings")
            notes = []
        raw = d.get("steps", [])
        if not isinstance(raw, list):
            return None, errors + ["steps must be a list"]
        if len(raw) > MAX_STEPS:
            errors.append(f"at most {MAX_STEPS} steps; keep the big decisions and the main fallbacks")
        steps = []
        for i, r in enumerate(raw[:MAX_STEPS]):
            step, errs = _parse_step(r, i)
            errors += errs
            if step:
                steps.append(step)
        ids = [s.id for s in steps]
        raw_ids = [r.get("id") for r in raw if isinstance(r, dict) and isinstance(r.get("id"), str)]
        dup = sorted({i for i in raw_ids if raw_ids.count(i) > 1})
        if dup:
            errors.append(f"step ids must be unique: {', '.join(dup)}")
        for s in steps:
            if s.after is not None and s.after not in ids:
                errors.append(f"step '{s.id}' comes after unknown step '{s.after}': 'after' must be the id of another "
                              f"step in this plan. For something that happens to you use 'event' ({', '.join(EVENTS)}); "
                              "for a date use 'start'.")
            if s.after == s.id:
                errors.append(f"step '{s.id}' cannot come after itself")
        if not errors and _has_cycle(steps):
            errors.append("the 'after' links form a cycle")
        if errors:
            return None, errors
        return cls(narrative.strip()[:1200], steps, float(spont), [n[:300] for n in notes[:8]]), []


def _parse_step(r, i) -> tuple[Step | None, list[str]]:
    where = f"step {i + 1}"
    if not isinstance(r, dict):
        return None, [f"{where} must be an object"]
    errors = []
    unknown = sorted(set(r) - STEP_FIELDS)
    if unknown:
        errors.append(f"{where} has unknown fields {unknown}; allowed: {sorted(STEP_FIELDS)}")
    sid = r.get("id")
    if not isinstance(sid, str) or not re.match(r"^[a-z][a-z0-9_]{0,39}$", sid or ""):
        errors.append(f"{where}: id must be short snake_case, got {sid!r}")
    else:
        where = f"step '{sid}'"
    head, option = r.get("head"), r.get("option")
    if head not in OPTIONS:
        errors.append(f"{where}: head must be one of {list(OPTIONS)}, got {head!r}")
    elif option in NOOP:
        errors.append(f"{where}: '{option}' is not a step: keeping things as they are is what happens without one. "
                      "Drop it; set a low spontaneity if this person rarely changes course.")
    elif option not in OPTIONS[head]:
        errors.append(f"{where}: option for {head} must be one of "
                      f"{[o for o in OPTIONS[head] if o not in NOOP]}, got {option!r}")
    label = r.get("label") or (option.replace("_", " ") if isinstance(option, str) else "")
    start = _month(r.get("start"), f"{where} start", errors)
    until = _month(r.get("until"), f"{where} until", errors)
    if start and until and until < start:
        errors.append(f"{where}: until {until} is before start {start}")
    commitment = r.get("commitment", 0.8)
    if not isinstance(commitment, (int, float)) or isinstance(commitment, bool) or not 0 <= commitment <= 1:
        errors.append(f"{where}: commitment must be a probability from 0 to 1, got {commitment!r}")
    event = r.get("event")
    if event is not None and event not in EVENTS:
        errors.append(f"{where}: event must be one of {list(EVENTS)}, got {event!r}")
    nums = {}
    for k, lo, hi in (("after_months", 0, 120), ("within_months", 1, 120), ("cash_months_below", -12, 60),
                      ("cash_months_above", -12, 120), ("min_age", 16, 100), ("max_age", 16, 100)):
        v = r.get(k)
        if v is None:
            continue
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not lo <= v <= hi:
            errors.append(f"{where}: {k} must be a number from {lo} to {hi}, got {v!r}")
        else:
            nums[k] = int(v) if k.endswith("_months") else float(v)
    after = r.get("after")
    if after is not None and not isinstance(after, str):
        errors.append(f"{where}: after must be the id of another step")
    repeat = r.get("repeat", False)
    if not isinstance(repeat, bool):
        errors.append(f"{where}: repeat must be true or false")
    if errors:
        return None, errors
    return Step(id=sid, label=str(label)[:80], head=head, option=option, start=start, until=until,
                commitment=float(commitment), after=after, event=event, repeat=repeat, **nums), []


def _has_cycle(steps):
    after = {s.id: s.after for s in steps}
    for s in steps:
        seen, cur = set(), s.id
        while cur is not None:
            if cur in seen:
                return True
            seen.add(cur)
            cur = after.get(cur)
    return False


def diff(old: LifePlan, new: LifePlan) -> dict:
    """What a revision changed, by step id."""
    o, n = {s.id: s for s in old.steps}, {s.id: s for s in new.steps}
    changed = []
    for sid in o.keys() & n.keys():
        a, b = asdict(o[sid]), asdict(n[sid])
        fields_changed = {k: [a[k], b[k]] for k in a if a[k] != b[k]}
        if fields_changed:
            changed.append({"id": sid, "label": n[sid].label, "fields": fields_changed})
    return {"added": [n[s].describe() for s in n if s not in o],
            "removed": [o[s].describe() for s in o if s not in n],
            "changed": sorted(changed, key=lambda c: c["id"]),
            "spontaneity": None if old.spontaneity == new.spontaneity else [old.spontaneity, new.spontaneity],
            "narrative_changed": old.narrative != new.narrative}


def freeze_past(old: LifePlan, new: LifePlan, as_of: str) -> tuple[LifePlan, list[str]]:
    """Steps whose window closed before `as_of` already played out: keep them exactly as they were
    (with the steps they wait on), whatever the revision says about them."""
    olds = {s.id: s for s in old.steps}
    keep = [s.id for s in old.steps if s.window()[1] < as_of]
    for sid in keep:                                   # grows while we walk it: the 'after' closure
        target = olds[sid].after
        if target is not None and target not in keep:
            keep.append(target)
    steps = {s.id: s for s in new.steps}
    kept = [sid for sid in keep if steps.get(sid) != olds[sid]]
    order = [s.id for s in new.steps] + [sid for sid in keep if sid not in steps]
    steps.update({sid: olds[sid] for sid in kept})
    return LifePlan(new.narrative, [steps[sid] for sid in order], new.spontaneity, list(new.notes)), kept
