"""Constants of the Financial Twin MDP (analysis/mdp_spec/mdp-spec.html) plus the market extension.

Every number is tagged in SOURCES: 'data' (fitted on the challenge ledger), 'swiss' (official statistic,
see swiss_targets.json), 'spec' (MDP spec placeholder) or 'assumed' (ours, to tune).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_cal = json.loads((ROOT / "analysis/action_importance/results/calibration.json").read_text())

# --- codes -------------------------------------------------------------------------------
S, E60, E80, E100, N = 0, 1, 2, 3, 4                        # work status W
W_NAMES = ["S", "E60", "E80", "E100", "N"]
FRAC = {E60: 0.6, E80: 0.8, E100: 1.0}
NONE, BACHELOR, MASTER, PHD = 0, 1, 2, 3                    # programme (in S) and highest degree (edu)
EDU_NAMES = ["none", "bachelor", "master", "phd"]
NORMAL, INFLATION, RECESSION = 0, 1, 2                      # economy M
SINGLE, COUPLE, FAMILY = 0, 1, 2                            # household F
P_NONE, P_APPLIED_MASTER, P_APPLIED_PHD, P_JOB_OFFER, P_ADMIT_MASTER, P_ADMIT_PHD = range(6)  # pending
PEND_NAMES = ["none", "applied_master", "applied_phd", "job_offer", "admitted_master", "admitted_phd"]

# Status action options (index into this list). 'accept' is allowed only with an offer pending.
STATUS = ["stop", "minus", "stay", "plus", "apply_master", "apply_phd", "accept"]
ST = {name: i for i, name in enumerate(STATUS)}

# --- money (CHF a month, January 2025 prices) --------------------------------------------
PAY = [4500.0, 6100.0, 7800.0, 9300.0, 10800.0]             # full-time pay level k=1..5 (data quintiles)
STUDENT_PAY = {BACHELOR: 1200.0, MASTER: 1200.0}            # data: student income
PHD_PAY = 47040.0 * 0.87 / 12                               # swiss: SNSF minimum 2025, ~13% deductions assumed
BENEFIT = 530.0                                             # data: not working under 65
PENSION_SHARE = 0.5                                         # spec
RENT_BASE = 2471.0                                          # data: rent = base x city x tier
TIER_MULT = [0.46, 0.68, 1.00]                              # shared, small, family flat
CITY_RENT = _cal["rent_model"]["city_mult"]
CITY_LIVING = _cal["other_living_city_median_chf"]
LIVING_INCOME_SHARE = 0.016                                 # data
STUDENT_LIVING_DISCOUNT = 126.0                             # data
LIFESTYLE_MULT = [0.9, 1.0, 1.1]                            # spec
TUITION_DEFAULT = 4800.0                                    # data: S001 median (CHF a year)
CHILD_COST = 1800.0                                         # spec
LEAVE_MONTHS, LEAVE_PAY = 4, 0.8                            # spec
MOVE_COST = 2000.0                                          # spec
BILL_P_FEB_AUG, BILL_P_OTHER, BILL_LO, BILL_HI = 0.37, 0.045, 450.0, 1780.0   # data
BONUS_P, BONUS_LO, BONUS_HI = 0.56, 0.23, 1.01              # data: December, employees
NOISE_BASE, NOISE_SHARE = 200.0, 0.05                       # data
PAY_GROWTH, COST_GROWTH, COST_GROWTH_INFL = 0.014, 0.010, 0.030   # data, data, spec
RETURNS = {NORMAL: (0.06, 0.10), INFLATION: (0.01, 0.12), RECESSION: (-0.15, 0.25)}  # spec (annual mean, sd)
CREDIT_RATE, CREDIT_LIMIT_MONTHS = 0.08, 3.0                # spec
BUFFER_MONTHS = 3.0                                         # spec: asset moves refill/invest above 3 months
RETIRE_AGE = 65

# Home value used by personas (CHF-equivalent a month, spec reward). Index [F][H]; None = not allowed.
HOME_VALUE = [[-700.0, 0.0, 300.0], [-1500.0, -500.0, 0.0], [None, -1200.0, 0.0]]
FREE_TIME = {S: 0.0, E60: 1800.0, E80: 1000.0, E100: 0.0, N: 2000.0}   # spec reward, CHF a month

# --- chance events (monthly probabilities) ------------------------------------------------
P_JOB_LOSS = {NORMAL: 0.003, INFLATION: 0.003, RECESSION: 0.010}      # spec; SECO 2.8% checked in the world
P_PROMOTION = {NORMAL: 0.007, INFLATION: 0.007, RECESSION: 0.003}     # spec
P_PARTNER, PARTNER_MAX_AGE = 0.004, 45                                # spec
P_CHILD, CHILD_MAX_AGE = 0.007, 42                                    # spec
P_SEPARATION = 0.002                                                  # spec
P_WINDFALL, WINDFALL_LO, WINDFALL_HI = 0.0005, 5000.0, 50000.0        # spec
ECONOMY = {NORMAL: {INFLATION: 0.005, RECESSION: 0.005}, INFLATION: {NORMAL: 0.04}, RECESSION: {NORMAL: 0.06}}  # spec
P_STUDY_END = {BACHELOR: 1 / 12, MASTER: 1 / 21, PHD: 1 / 48}        # assumed: remaining bachelor ~1 yr, master ~1.75 yr, PhD ~4 yr
P_FEWER_HOURS_OK, P_MORE_HOURS_OK = 0.7, 0.5                          # assumed: employer agrees this month
STUDY_MAX_AGE = 45                                                    # spec
APPLY_MONTHS = tuple(range(1, 13))                                    # assumed: rolling applications, once a year or when a spell without work starts
ADMISSION_EVERY_MONTH = True                                          # assumed: places decided monthly, start after accepting
ENTRY_LEVEL = {NONE: 1, BACHELOR: 1, MASTER: 2, PHD: 3}               # assumed from graduate pay (~CHF 77k gross, master)


def rent(city: str, tier: int) -> float:
    return RENT_BASE * CITY_RENT[city] * TIER_MULT[tier]


def target_level(edu: int, k_prev: int, u: int) -> int:
    """Pay level a searcher can expect: their degree's entry level, or their old level (one lower after a year out)."""
    return max(ENTRY_LEVEL[edu], k_prev - (1 if u >= 12 else 0))


SOURCES = {
    "pay levels, student pay, benefit, rent model, living costs, bills, bonus, noise, pay/cost growth": "data",
    "PhD pay (SNSF minimum CHF 47,040, 2025)": "swiss",
    "unemployment, spell length, bachelor->master, moving rate (calibration targets)": "swiss",
    "pension share, lifestyle, child, moving cost, returns, credit, event rates, economy switching": "spec",
    "study durations, hours-change approval, application window, entry pay levels, personas, market rules": "assumed",
}
