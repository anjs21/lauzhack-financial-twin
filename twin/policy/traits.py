"""Who is deciding: persona traits, uncertainty about them, and stated intentions.

Traits live in [0, 1]. Defaults come from the profile (age, work mode, household, savings
mix, goal types), a questionnaire can override them, and every path draws its own traits
around the person's values (logit-normal spread), so the twin also shows how much the
answer depends on not knowing the person exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit, logit

from .. import config as C

TRAITS = ["risk_tolerance", "frugality", "career_drive", "free_time", "housing_aspiration",
          "inertia", "impatience", "goal_focus"]
TRAIT = {t: i for i, t in enumerate(TRAITS)}

QUESTIONS = {
    "risk_tolerance": "I am comfortable holding investments that can lose value for a while.",
    "frugality": "I save first and spend what is left.",
    "career_drive": "Moving up in my career or studies matters a lot to me.",
    "free_time": "I would trade some income for more free time.",
    "housing_aspiration": "Living somewhere bigger or nicer is worth paying for.",
    "inertia": "Once things work, I rarely change them.",
    "impatience": "I would rather enjoy money now than later.",
    "goal_focus": "I keep a close eye on my savings goals and deadlines.",
}


EVENTS = ("job_loss", "offer", "graduated", "partner", "child", "separation", "windfall")


@dataclass
class Intention:
    """A plan the person states, acted on with probability `prob` when it applies.

    head: "status" (option 0..4), "housing", "spending" or "assets" (option -1/0/+1).
    when: "YYYY-MM" for a one-off plan; otherwise it applies whenever its conditions hold
    at a decision point, until it fires once (once=True) or every time (once=False).

    Long-term plans (twin/plan) also use:
    until: "YYYY-MM", the last month of a window starting at `when` (or now). Within the window
        the plan is retried every January review while its option is not possible, or the
        world turned it down (no admission, no flat); declining it (1 - prob) ends it.
    after: id of another intention that must have been carried out first, at least
        `after_months` months earlier.
    event: a life event (EVENTS) that must have happened in the last `within_months` months:
        a contingency such as "if I lose my job, I move somewhere cheaper".
    """
    head: str
    option: int
    prob: float = 1.0
    when: str | None = None
    min_age: float | None = None
    max_age: float | None = None
    cash_months_below: float | None = None
    cash_months_above: float | None = None
    once: bool = True
    id: str | None = None
    until: str | None = None
    after: str | None = None
    after_months: int = 0
    event: str | None = None
    within_months: int = 12

    def __post_init__(self):
        if self.event is not None and self.event not in EVENTS:
            raise ValueError(f"unknown event {self.event!r}; use one of {', '.join(EVENTS)}")

    @property
    def windowed(self):
        return self.until is not None

    @classmethod
    def parse(cls, d: dict) -> "Intention":
        return cls(**d)


@dataclass
class Persona:
    traits: dict
    temperature: float = 1.0       # higher = more random choices
    spread: float = 0.4            # sd of trait uncertainty, logit units (0 = traits known exactly)
    intentions: list = field(default_factory=list)

    def vector(self):
        return np.array([self.traits[t] for t in TRAITS], float)

    def with_answers(self, answers: dict, weight: float = 0.7) -> "Persona":
        """Blend questionnaire answers (1-5 per trait) into the traits."""
        tr = dict(self.traits)
        for k, a in answers.items():
            tr[k] = float(np.clip(weight * (a - 1) / 4 + (1 - weight) * tr[k], 0.02, 0.98))
        return Persona(tr, self.temperature, self.spread, list(self.intentions))


def from_profile(row) -> Persona:
    """Data-informed default traits for one profile row (see data.profiles())."""
    tr = dict(risk_tolerance=0.5, frugality=0.5, career_drive=0.5, free_time=0.3, housing_aspiration=0.4,
              inertia=0.7, impatience=0.4, goal_focus=0.5)
    liquid = row["cash0"] + row["inv0"]
    if liquid > 0:
        tr["risk_tolerance"] = 0.3 + 0.5 * row["inv0"] / liquid
    age = row["age"]
    tr["career_drive"] += 0.15 * (age < 30) + 0.15 * (row["mode"] == C.STUDENT) - 0.2 * (age > 55)
    tr["free_time"] += 0.2 * (age >= 55) + 0.25 * (row["mode"] == C.PART_TIME)
    tr["housing_aspiration"] += 0.1 * (row["household"] != C.SINGLE) - 0.1 * (row["tier"] == 0)
    high, med = row.get("high_goal"), row.get("med_goal")
    if high == "Emergency savings":
        tr["frugality"] += 0.1
        tr["goal_focus"] += 0.1
    nudges = {"Housing deposit": {"frugality": 0.15, "housing_aspiration": 0.1},
              "Education flexibility": {"career_drive": 0.2},
              "Career transition": {"career_drive": 0.15, "inertia": -0.15},
              "Retirement reserve": {"free_time": 0.15, "frugality": 0.1}}
    for k, v in nudges.get(med, {}).items():
        tr[k] += v
    return Persona({k: float(np.clip(v, 0.02, 0.98)) for k, v in tr.items()})


def sample_traits(persona: Persona, n: int, u_normal: np.ndarray) -> np.ndarray:
    """(n, 8) traits around the persona's values; u_normal is an (8, n) standard-normal draw."""
    base = logit(np.clip(persona.vector(), 0.02, 0.98))
    return expit(base[None, :] + persona.spread * np.asarray(u_normal).T)


def buffer_months(traits: np.ndarray) -> np.ndarray:
    """Cash buffer target: 6 months for the most cautious, 2 for the boldest."""
    return 6.0 - 4.0 * traits[:, TRAIT["risk_tolerance"]]
