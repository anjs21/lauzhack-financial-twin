"""Assumption-based structural transitions; no causal effects are estimated here."""
from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_HALF_UP
import csv
from pathlib import Path


def money(value):
    return Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class Assumptions:
    study_months: int = 24
    annual_tuition: float = 8000
    annual_scholarship: float = 0
    work_income: float = 4200
    study_income: float = 1200
    graduate_income: float = 4800
    work_expenses: float = 3000
    study_expenses: float = 2100
    graduate_expenses: float = 3100
    job_search_expenses: float = 2100
    employment_delay_months: int = 0
    minimum_reserve: float = 0

    def __post_init__(self):
        for key in ('study_months', 'employment_delay_months'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < (1 if key == 'study_months' else 0):
                raise ValueError(f'Invalid {key}')
        for key in self.__dataclass_fields__:
            if key in ('study_months', 'employment_delay_months'):
                continue
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
                raise ValueError(f'Invalid {key}')
            if not Decimal(str(value)).is_finite() or value < 0:
                raise ValueError(f'Invalid {key}')
        if self.annual_scholarship > self.annual_tuition:
            raise ValueError('Tuition scholarship cannot exceed annual tuition')


@dataclass(frozen=True)
class State:
    month: int
    cash: Decimal
    study_progress: int = 0
    education: str = 'not_started'
    employment: str = 'not_working'
    months_since_graduation: int = 0


@dataclass(frozen=True)
class Transition:
    before: State
    action: str
    after: State
    income: Decimal
    living_costs: Decimal
    tuition: Decimal
    scholarship: Decimal
    cash_after_upfront_costs: Decimal
    minimum_cash: Decimal
    reserve_gap: Decimal


def transition(state: State, action: str, assumptions: Assumptions) -> Transition:
    """Tuition/grant at month start; income/costs at month end; graduate jobs next month earliest.

    Negative cash is an unfunded projection, never automatic borrowing.
    Tuition scholarships are paid before tuition. Part-year final tuition is not prorated.
    """
    a = assumptions
    studying = action in ('enroll', 'study')
    if action == 'enroll':
        if state.education != 'not_started':
            raise ValueError('Enrollment requires education not yet started')
    elif action == 'study':
        if state.education != 'studying':
            raise ValueError('Continue study requires active enrollment')
    elif action == 'work':
        if state.education == 'studying':
            raise ValueError('Study interruption is not modelled')
    else:
        raise ValueError(f'Unknown action: {action}')

    tuition = scholarship = money(0)
    progress, education = state.study_progress, state.education
    wait = state.months_since_graduation
    if studying:
        income, costs, employment = a.study_income, a.study_expenses, 'part_time'
        if progress % 12 == 0:
            tuition, scholarship = money(a.annual_tuition), money(a.annual_scholarship)
        progress += 1
        education = 'graduated' if progress == a.study_months else 'studying'
        wait = 0
    elif education == 'graduated':
        if wait >= a.employment_delay_months:
            income, costs, employment = a.graduate_income, a.graduate_expenses, 'graduate_job'
        else:
            income, costs, employment = 0, a.job_search_expenses, 'job_seeking'
        wait += 1
    else:
        income, costs, employment = a.work_income, a.work_expenses, 'job'
    income, costs = money(income), money(costs)
    upfront = state.cash + scholarship - tuition
    closing = upfront + income - costs
    lowest = min(state.cash, upfront, closing)
    after = State(state.month + 1, closing, progress, education, employment, wait)
    return Transition(state, action, after, income, costs, tuition, scholarship,
                      upfront, lowest, max(money(0), money(a.minimum_reserve) - lowest))


def simulate(opening_cash, assumptions, months=120, enroll_month=None):
    """Evaluate a work policy or work-then-study policy; this is not an MDP solver."""
    if isinstance(months, bool) or not isinstance(months, int) or months < 1:
        raise ValueError('months must be a positive integer')
    if enroll_month is not None and (isinstance(enroll_month, bool) or not isinstance(enroll_month, int) or not 0 <= enroll_month < months):
        raise ValueError('enroll_month must lie within the horizon')
    opening = Decimal(str(opening_cash))
    if not opening.is_finite() or opening < 0:
        raise ValueError('opening_cash must be finite and nonnegative')
    state, trace = State(0, money(opening)), []
    for month in range(months):
        action = 'enroll' if month == enroll_month else ('study' if state.education == 'studying' else 'work')
        step = transition(state, action, assumptions)
        trace.append(step)
        state = step.after
    return trace


def summary(trace):
    return {
        'closing_cash': trace[-1].after.cash,
        'minimum_cash': min(x.minimum_cash for x in trace),
        'extra_initial_funding': max(x.reserve_gap for x in trace),
        'first_shortfall_month': next((x.before.month + 1 for x in trace if x.minimum_cash < 0), None),
        'funded': all(x.minimum_cash >= 0 for x in trace),
    }


def load_profile(data_dir: Path, profile_id='P00001'):
    """Read starting cash and S001 inputs; retain row IDs for provenance."""
    def read(name):
        with (data_dir / name).open(newline='', encoding='utf-8') as stream:
            return list(csv.DictReader(stream))
    profile = next((x for x in read('synthetic-profiles.csv') if x['profile_id'] == profile_id), None)
    if profile is None:
        raise ValueError(f'Unknown profile: {profile_id}')
    rows = [x for x in read('scenario-assumptions.csv') if x['profile_id'] == profile_id and x['scenario_id'] == 'S001']
    parameters = {x['parameter']: x for x in rows}
    if not {'study_duration_years', 'annual_tuition_chf'} <= parameters.keys():
        raise ValueError(f'Missing S001 assumptions for {profile_id}')
    duration = parameters['study_duration_years']
    tuition = parameters['annual_tuition_chf']
    if (duration['unit'], duration['frequency'], tuition['unit'], tuition['frequency']) != ('years', 'once', 'CHF', 'annual'):
        raise ValueError('Unexpected scenario units or frequencies')
    months = Decimal(duration['value']) * 12
    if not months.is_finite() or months != months.to_integral_value():
        raise ValueError('Study duration must resolve to whole months')
    assumptions = Assumptions(study_months=int(months), annual_tuition=float(tuition['value']))
    return profile, assumptions, [x['assumption_id'] for x in rows]


def sensitivity(opening_cash, assumptions, months=120):
    """Fixed-policy comparison; one parameter at a time, no probabilities implied."""
    for parameter, values in {
        'annual_tuition': (2500, 8000, 12000),
        'graduate_income': (4200, 4800, 5400),
        'employment_delay_months': (0, 3, 6),
    }.items():
        for value in values:
            if parameter == 'annual_tuition' and value < assumptions.annual_scholarship:
                continue
            variant = replace(assumptions, **{parameter: value})
            work = summary(simulate(opening_cash, variant, months))
            study = summary(simulate(opening_cash, variant, months, 0))
            yield parameter, value, study, study['closing_cash'] - work['closing_cash']
