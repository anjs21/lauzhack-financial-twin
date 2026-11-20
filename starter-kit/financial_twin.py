"""Minimal deterministic cash-only Financial Twin simulation.

All amounts are CHF. Months are zero-indexed and phases use half-open intervals.
This is an educational scenario engine, not a forecasting or advice tool.
"""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SimulationResult:
    scenario_name: str
    opening_cash_chf: float
    closing_cash_chf: float
    minimum_cash_chf: float
    first_negative_month: int | None
    cash_by_month_chf: tuple[float, ...]


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        raise ValueError(f"{label} must be finite")
    return result


def simulate(profile: dict, scenario: dict) -> SimulationResult:
    """Simulate cash balances with explicit deterministic monthly assumptions."""
    opening = _number(profile["opening_cash_chf"], "opening_cash_chf")
    months = profile["horizon_months"]
    if isinstance(months, bool) or not isinstance(months, int) or months < 1:
        raise ValueError("horizon_months must be a positive integer")

    phases = scenario["phases"]
    if not isinstance(phases, list) or not phases:
        raise ValueError("phases must be a non-empty list")

    income = [None] * months
    expenses = [None] * months
    for phase in phases:
        start, end = phase["start_month"], phase["end_month"]
        if (isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, int) or not isinstance(end, int)
                or not 0 <= start < end <= months):
            raise ValueError("Invalid phase interval")
        inc = _number(phase["monthly_income_chf"], "monthly_income_chf")
        exp = _number(phase["monthly_expenses_chf"], "monthly_expenses_chf")
        if exp < 0:
            raise ValueError("monthly_expenses_chf cannot be negative")
        for m in range(start, end):
            if income[m] is not None:
                raise ValueError(f"Overlapping phases at month {m}")
            income[m], expenses[m] = inc, exp
    if any(v is None for v in income):
        raise ValueError("Phases must cover every month")

    costs = [0.0] * months
    for item in scenario.get("one_off_costs", []):
        month = item["month"]
        if isinstance(month, bool) or not isinstance(month, int) or not 0 <= month < months:
            raise ValueError("Invalid one-off cost month")
        amount = _number(item["amount_chf"], "one-off amount_chf")
        if amount < 0:
            raise ValueError("One-off costs cannot be negative")
        costs[month] += amount

    cash = opening
    balances = []
    for m in range(months):
        cash = cash + income[m] - expenses[m] - costs[m]
        balances.append(round(cash, 2))

    minimum = min([opening] + balances)
    first_negative = next((i for i, v in enumerate(balances) if v < 0), None)
    return SimulationResult(
        scenario_name=str(scenario["name"]),
        opening_cash_chf=round(opening, 2),
        closing_cash_chf=balances[-1],
        minimum_cash_chf=round(minimum, 2),
        first_negative_month=first_negative,
        cash_by_month_chf=tuple(balances),
    )
