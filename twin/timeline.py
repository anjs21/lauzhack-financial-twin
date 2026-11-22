"""Recorded months and accepted offers for an individual twin."""
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from .replay import update_bias
from .world.flow import SCALE, features


@dataclass
class Timeline:
    offers: dict = field(default_factory=dict)         # month -> (work mode, net CHF[, alongside studies])
    observations: dict = field(default_factory=dict)   # consecutive month ends


def observed_month(world, s, observation, offer, count, bias):
    """Advance recorded time. Closing cash is observed; unreported assets carry forward."""
    # Only the model's expected regular flow enters the bias update. The optional
    # observed flow explicitly excludes transfers, debt service and other overlays.
    if offer:
        world.accept_offer(s, offer)
    flow_key = "avg_net_flow" if "avg_net_flow" in observation else "net_flow"
    if flow_key in observation:
        # Same historical state for every path. Use the ensemble expectation: of this month for a
        # monthly flow, or the average over the twelve calendar months for a year-end check-in.
        months = np.arange(1, 13) if flow_key == "avg_net_flow" else np.array([s.t % 12 + 1])
        X = features(np.full(len(months), s.mode[0]), np.full(len(months), s.self_employed[0]),
                     np.full(len(months), s.pensioner[0]), np.full(len(months), s.income[0] / s.P[0]),
                     np.full(len(months), s.rent[0] / s.P[0]), np.full(len(months), s.city[0]), months)
        mean, _ = world.flow.mean_var(X)
        expected = (mean.mean() + s.offset.mean()) * SCALE * s.P[0] - bias
        new_bias = update_bias(bias, observation[flow_key] - expected, count + 1)
        s.offset += (new_bias - bias) / (SCALE * s.P)
        bias, count = new_bias, count + 1
    loan_due = s.debt * (1 + s.loan_rate / 12)
    s.debt = np.maximum(0, loan_due - s.loan_pay)
    s.loan_pay[s.debt == 0] = 0
    s.cash[:] = observation["cash"]
    if "investments" in observation:
        s.inv[:] = observation["investments"]
    growth = (1 + world.pr["drift"]["pay_growth_per_year"]) ** (1 / 12)
    prices = (1 + world.pr["drift"]["cost_growth_per_year"]) ** (1 / 12)
    s.age += 1 / 12
    s.G *= growth
    s.P *= prices
    s.rent *= prices
    s.t += 1
    s.tau += 1
    # Carry employment/status as last observed until explicitly updated.
    if "status" in observation:
        status = observation["status"]
        mode = C.STATUS_TO_MODE[status]
        changed = (s.mode != mode) | (s.pensioner != (status == "retired"))
        s.tau[changed], s.search[changed], s.wage_adjustment[changed] = 0, 0, 0
        s.mode[:], s.pensioner[:], s.self_employed[:] = mode, status == "retired", status == "self_employed"
        s.study_left[:] = 1 if mode == C.STUDENT else 0
        s.side_job &= s.mode == C.STUDENT
    world._set_income(s)
    if "income" in observation:
        world.set_salary(s, observation["income"])
    if "rent" in observation:
        s.rent[:] = observation["rent"]
        s.rent_z = world.rent_model.z_of(s.rent / s.P, s.tier, s.city)
    limit = np.maximum(world.pr["credit"]["limit_floor"], s.income * world.pr["credit"]["limit_months_of_pay"])
    return count, bias, s.cash < -limit
