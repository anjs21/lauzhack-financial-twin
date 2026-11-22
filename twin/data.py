"""Load the synthetic data and build the tables the models learn from.

Money in the ledger is nominal. The flow network learns in *real* terms (deflated by the
price index P_t) so that ten-year rollouts never ask it to extrapolate the calendar:
pay grows with G_t, prices with P_t, and both are tracked outside the network.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import yaml

from . import config as C


def load_priors(path=C.PRIORS) -> dict:
    return yaml.safe_load(path.read_text())


@lru_cache(maxsize=1)
def profiles() -> pd.DataFrame:
    """One row per person: static attributes, opening balances, goals, S001 tuition."""
    p = pd.read_csv(C.DATA / "synthetic-profiles.csv").set_index("profile_id")
    assets = pd.read_csv(C.DATA / "synthetic-assets.csv")
    loans = pd.read_csv(C.DATA / "synthetic-liabilities.csv")
    goals = pd.read_csv(C.DATA / "synthetic-goals.csv")
    s001 = pd.read_csv(C.DATA / "scenario-assumptions.csv")
    assert loans.profile_id.is_unique, "one loan per profile is assumed"
    assert set(p.city) <= set(C.CITIES), set(p.city) - set(C.CITIES)

    out = pd.DataFrame(index=p.index)
    out["name"] = p.display_name
    out["age"] = p.age_at_start.astype(float)
    out["city"] = p.city
    out["city_idx"] = p.city.map(C.CITY).astype(int)
    out["status"] = p.employment_status
    out["mode"] = p.employment_status.map(C.STATUS_TO_MODE).astype(int)
    out["self_employed"] = p.employment_status.eq("self_employed")
    out["pensioner"] = p.employment_status.eq("retired")
    out["household_type"] = p.household_type
    out["household"] = p.household_type.map(C.HOUSEHOLD_OF).astype(int)
    out["tier"] = p.household_type.map(C.TIER_OF).astype(int)
    out["income0"] = p.reference_monthly_net_income_chf.astype(float)
    out["rent0"] = p.reference_monthly_rent_chf.astype(float)
    out["cash0"] = p.opening_cash_chf.astype(float)
    inv = assets[assets.asset_type == "investment_portfolio"].groupby("profile_id").value_chf.sum()
    out["inv0"] = inv.reindex(p.index).fillna(0.0)
    loans = loans.set_index("profile_id")
    out["debt0"] = loans.opening_principal_chf.reindex(p.index).fillna(0.0)
    out["rate0"] = loans.annual_interest_rate.reindex(p.index).fillna(0.0)
    hi = goals[goals.priority == "high"].set_index("profile_id")
    md = goals[goals.priority == "medium"].set_index("profile_id")
    out["high_goal"] = hi.goal_name.reindex(p.index)
    out["high_target"] = hi.target_amount_chf.reindex(p.index)
    out["med_goal"] = md.goal_name.reindex(p.index)
    out["med_target"] = md.target_amount_chf.reindex(p.index)
    tuition = s001[s001.parameter == "annual_tuition_chf"].set_index("profile_id").value
    out["tuition"] = tuition.reindex(p.index)
    # The data identify students but not how far along they are: assume the course ends at the opening review.
    out["study_left0"] = np.where(out["mode"] == C.STUDENT, 1, 0)
    out["masters0"] = False
    return out


@lru_cache(maxsize=1)
def ledger() -> pd.DataFrame:
    led = pd.read_csv(C.DATA / "synthetic-cash-balances.csv")
    led["t"] = ((led.month.str[:4].astype(int) - C.START_YEAR) * 12
                + led.month.str[5:].astype(int) - C.START_MONTH)
    led["moy"] = led.month.str[5:].astype(int)
    return led.sort_values(["profile_id", "t"]).reset_index(drop=True)


def indices(t, priors: dict):
    """Pay index G_t and price index P_t, both 1 in January 2025 (normal economy)."""
    g = priors["drift"]["pay_growth_per_year"]
    pi = priors["drift"]["cost_growth_per_year"]
    t = np.asarray(t, float)
    return (1 + g) ** (t / 12), (1 + pi) ** (t / 12)


def profile_months(priors: dict | None = None) -> pd.DataFrame:
    """28,800 rows: one per person and month, with real-terms inputs and target."""
    priors = priors or load_priors()
    prof, led = profiles(), ledger()
    cols = ["mode", "self_employed", "pensioner", "city_idx", "income0", "rent0", "status"]
    pm = led[["profile_id", "t", "moy", "opening_cash_chf", "net_transactions_chf",
              "closing_cash_chf"]].join(prof[cols], on="profile_id")
    pm["G"], pm["P"] = indices(pm.t.values, priors)
    pm["income_real"] = pm.income0 * pm.G / pm.P
    pm["rent_real"] = pm.rent0                      # rents are indexed to prices
    pm["flow"] = pm.net_transactions_chf
    pm["flow_real"] = pm.flow / pm.P
    return pm


def person_folds(prof: pd.DataFrame, k: int = 5, seed: int = C.SEED) -> pd.Series:
    """Fold per person, balanced across work modes; nobody is in two folds."""
    rng = np.random.default_rng(seed)
    fold = pd.Series(0, index=prof.index)
    for _, ids in prof.groupby("mode").groups.items():
        ids = np.asarray(ids)[rng.permutation(len(ids))]
        fold.loc[ids] = np.arange(len(ids)) % k
    return fold
