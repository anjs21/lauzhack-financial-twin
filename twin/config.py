"""Paths, fixed vocabularies and seeds shared by the whole package."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
ARTIFACTS = ROOT / "twin" / "artifacts"
PRIORS = ROOT / "twin" / "world" / "priors.yaml"

SEED = 20261121
START_YEAR, START_MONTH = 2025, 1      # month index t = 0 is January 2025
HORIZON = 120                          # 2025-01 .. 2034-12
T_HIGH_GOAL = 59                       # 2029-12, deadline of every high-priority goal
T_MED_GOAL = 95                        # 2032-12, deadline of every medium-priority goal
REGULAR = [1, 3, 4, 5, 6, 7, 10, 11]   # months without bills (Feb, Aug), fees (Feb, Sep) or bonus (Dec)

MODES = ["student", "part_time", "full_time", "not_working"]
STUDENT, PART_TIME, FULL_TIME, NOT_WORKING = range(4)
STATUS_TO_MODE = {"student": STUDENT, "part_time": PART_TIME, "employed": FULL_TIME,
                  "self_employed": FULL_TIME, "unemployed": NOT_WORKING, "retired": NOT_WORKING}

HOUSEHOLDS = ["single", "couple", "family"]
SINGLE, COUPLE, FAMILY = range(3)
TIERS = ["shared flat", "small flat", "family flat"]
# The data's household_type mixes who lives together with what they rent.
HOUSEHOLD_OF = {"shared_housing": SINGLE, "single": SINGLE, "couple": COUPLE, "family": FAMILY}
TIER_OF = {"shared_housing": 0, "single": 1, "couple": 2, "family": 2}

CITIES = ["Basel", "Bern", "Biel", "Fribourg", "Geneva", "Lausanne", "Lucerne", "Lugano",
          "Neuchatel", "St Gallen", "Winterthur", "Zurich"]
CITY = {c: i for i, c in enumerate(CITIES)}

ECONOMY = ["normal", "inflation", "recession"]

# Action heads and their options, in the order the policy samples them.
STATUS_OPTIONS = ["continue", "start a master's", "accept part-time work", "accept full-time work",
                  "leave paid work"]
HOUSING_OPTIONS = [-1, 0, 1]     # index 0..2 <-> change -1, 0, +1
SPENDING_OPTIONS = [-1, 0, 1]
ASSET_OPTIONS = [-1, 0, 1]       # sell to the buffer, hold, invest above the buffer


def month_label(t: int) -> str:
    y, m = divmod(START_MONTH - 1 + t, 12)
    return f"{START_YEAR + y}-{m + 1:02d}"


def month_index(label: str) -> int:
    y, m = (int(x) for x in label.split("-"))
    return (y - START_YEAR) * 12 + (m - START_MONTH)


def annual_review(t):
    return (START_MONTH - 1 + t) % 12 == 0


def next_review(t):
    return t + (-(START_MONTH - 1 + t)) % 12


def next_checkin(t):
    """The December closing on or after month t: when the person checks in with year-end figures."""
    return t + (11 - (START_MONTH - 1 + t) % 12)
