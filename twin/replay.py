"""Chronological replay. Fit on 2025, predict each 2026 month before revealing it."""
import json
from pathlib import Path

import numpy as np

from . import config as C
from .data import ledger, load_priors, profile_months, profiles
from .world.flow import SCALE, FlowModel, crps, frame_features


def update_bias(previous, residual, count):
    """Mean model error in CHF with twelve zero-error prior observations."""
    return previous + (residual - previous) / (12 + count)


def evaluate(draws, observed, opening):
    cash = opening.copy()
    at_risk = cash >= 0
    accumulated = np.broadcast_to(cash[:, None], draws.shape[:2]).copy()
    bias = np.zeros_like(cash)
    methods = {k: {"flow_crps": [], "closing_cash_mae": [], "incident_brier": [],
                   "alerts": 0, "new_shortfalls_detected": 0} for k in ("frozen", "balances", "balances_and_surplus")}
    incidents = 0
    for i in range(observed.shape[1]):
        flows = draws[:, :, i]
        actual = observed[:, i]
        closing = cash + actual
        incident = at_risk & (closing < 0)
        incidents += int(incident.sum())
        for name, scores in methods.items():
            forecast = flows + bias[:, None] if name == "balances_and_surplus" else flows
            cash_draws = forecast + (accumulated if name == "frozen" else cash[:, None])
            probability = (cash_draws < 0).mean(axis=1)
            alerts = at_risk & (probability >= 0.5)
            scores["alerts"] += int(alerts.sum())
            scores["new_shortfalls_detected"] += int((alerts & incident).sum())
            scores["incident_brier"].extend((probability[at_risk] - incident[at_risk]) ** 2)
            scores["closing_cash_mae"].extend(abs(np.median(cash_draws, axis=1) - closing))
            scores["flow_crps"].extend(crps(forecast, actual))
        bias = update_bias(bias, actual - flows.mean(axis=1), i + 1)
        accumulated += flows
        cash = closing
        at_risk &= cash >= 0
    for scores in methods.values():
        for key in ("flow_crps", "closing_cash_mae", "incident_brier"):
            scores[key] = float(np.mean(scores[key]))
    return {"profiles": len(opening), "months": observed.shape[1], "initially_negative": int((opening < 0).sum()),
            "first_new_shortfalls": incidents, "alert_threshold": 0.5, "results": methods}


def run(path):
    # The shipped drift estimates also use 2026. Use flat ex-ante drift in all
    # three arms so validation has no access to the held-out year's trend.
    pr = load_priors()
    pr["drift"] = dict(pay_growth_per_year=0., cost_growth_per_year=0.)
    pm = profile_months(pr)
    train = pm[pm.t < 12]
    from .validate import nn_flows
    x, y, groups = frame_features(train), train.flow_real.values / SCALE, train.profile_id.values
    model = FlowModel.fit(x, y, groups, seeds=(0, 1, 2, 3, 4))
    model.fit_offsets(x, y, groups)
    people = profiles()
    draws = nn_flows(model, people, 12, 12, 200, C.SEED + 7, pr)
    rows = ledger()
    actual = rows.pivot(index="profile_id", columns="t", values="net_transactions_chf").loc[people.index]
    opening = rows[rows.t == 11].set_index("profile_id").loc[people.index, "closing_cash_chf"]
    result = evaluate(draws, actual.values[:, 12:], opening.values)
    result["design"] = "2025-only model and offsets, flat drift, 200 paired draws. Incident alerts count at-risk person-months."
    Path(path).write_text(json.dumps(result, indent=2) + "\n")
    return result
