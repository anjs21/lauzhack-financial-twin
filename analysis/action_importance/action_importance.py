#!/usr/bin/env python3
"""Forward-looking action-importance screen for the Financial Twin MDP.

Complements ../analyze_mdp.py (descriptive replays of the recorded ledger) and
../mdp_design.json (accounting contract). This script:

1. Decodes how the synthetic monthly cash flows were generated
   (income, rent, flat living costs, seasonal bills, December bonus, shocks).
2. Calibrates a stochastic monthly model for the exogenous noise w_t and
   back-tests it out of sample (calibrate on 2025, predict 2026).
3. Applies each candidate decision or shock once, at January 2025, to every
   eligible profile and simulates 120 months (to December 2034) under common
   random numbers, so differences come from the intervention alone.
4. Ranks interventions by how often they materially change a profile's
   outcome: cash shortfall, high-priority goal (2029-12), medium goal
   (2032-12), terminal net financial wealth.
5. Estimates which state variables drive the baseline outcomes.

Everything is synthetic and assumption-labelled (see dict A). Results show what
matters *inside this simulator*; they are not causal estimates, forecasts or
financial advice.

Run:   python analysis/action_importance/action_importance.py
Needs: numpy, pandas, scipy, scikit-learn. Takes about two minutes.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "Data"
OUT = HERE / "results"

SEED = 20261121
N_PATHS = 200
T_FWD = 120   # 2025-01 .. 2034-12, the horizon in ../mdp_design.json
T_HIGH = 59   # index of 2029-12, deadline of every high-priority goal
T_MED = 95    # index of 2032-12, deadline of every medium-priority goal
REGULAR = [1, 3, 4, 5, 6, 7, 10, 11]  # months without bills (Feb, Aug), fees (Feb, Sep) or bonus (Dec)

STATUS = ["employed", "part_time", "self_employed", "retired", "student", "unemployed",
          "startup", "masters"]  # the last two exist only inside interventions
SC = {s: i for i, s in enumerate(STATUS)}

# Analyst assumptions: none of these numbers is in the data. Change and re-run.
A = dict(
    invest_return=0.04,          # illustrative return used in docs/financial-concepts.md
    invest_vol=0.10,             # balanced-portfolio volatility
    loan_term=60,                # same 60-month annuity as ../analyze_mdp.py
    rate_floor=0.01,
    overdraft_rate=0.08,         # interest on negative cash; without it overdraft is free and exploitable
    job_search_months=3,
    entry_salary_quantile=0.25,  # quantile of employed reference income used as a new job's pay
    masters_months=24,
    masters_premium=0.10,        # pay uplift after graduating
    business_delay=12,           # months without revenue
    business_cost=15_000,
    moving_cost=2_000,
    relocation_cost=3_000,
    car_price=25_000,
    car_running=400,
    emergency_expense=10_000,
    new_loan=20_000,
    new_loan_rate=0.065,
    child_cost=1_800,            # the data has no household-size effect on spending
    leave_months=4,
    leave_income=0.8,
    invest_buffer_months=6,
    repay_buffer_months=3,
    surplus_share=0.5,
    surplus_buffer_months=3,
    topup_floor_months=1,        # closed-loop rule: sell investments when cash < 1 month of outgoings
    topup_target_months=3,       # ... and refill cash to 3 months
    job_loss_months=6,
    crash=0.30,
    rate_step=0.02,
    inflation_extra=0.02,
    rent_increase=0.15,
    raise_=0.10,
    workload_cut=0.20,
    spending_change=0.10,
    cheaper_flat=0.20,
    material=0.10,               # |change in a probability| that counts as material
)


# --------------------------------------------------------------------------- data
def load():
    prof = pd.read_csv(DATA / "synthetic-profiles.csv").set_index("profile_id")
    led = pd.read_csv(DATA / "synthetic-cash-balances.csv").sort_values(["profile_id", "month"])
    led["mon"] = led.month.str[5:].astype(int)
    led["yr"] = led.month.str[:4].astype(int)
    assets = pd.read_csv(DATA / "synthetic-assets.csv")
    loans = pd.read_csv(DATA / "synthetic-liabilities.csv")
    goals = pd.read_csv(DATA / "synthetic-goals.csv")
    s001 = pd.read_csv(DATA / "scenario-assumptions.csv")
    assert loans.profile_id.is_unique, "script assumes at most one loan per profile"
    assert (goals.priority == "high").groupby(goals.profile_id).sum().eq(1).all()
    assert (goals.priority == "medium").groupby(goals.profile_id).sum().le(1).all()

    inv = assets[assets.asset_type == "investment_portfolio"].groupby("profile_id").value_chf.sum()
    loans = loans.set_index("profile_id")
    hi = goals[goals.priority == "high"].set_index("profile_id")
    md = goals[goals.priority == "medium"].set_index("profile_id")
    tuition = s001[s001.parameter == "annual_tuition_chf"].set_index("profile_id").value
    prof["inv0"] = inv.reindex(prof.index).fillna(0.0)
    prof["debt0"] = loans.opening_principal_chf.reindex(prof.index).fillna(0.0)
    prof["rate0"] = loans.annual_interest_rate.reindex(prof.index).fillna(0.0)
    prof["high_goal"] = hi.goal_name.reindex(prof.index)
    prof["high_target"] = hi.target_amount_chf.reindex(prof.index)
    prof["med_goal"] = md.goal_name.reindex(prof.index)
    prof["med_target"] = md.target_amount_chf.reindex(prof.index)
    prof["tuition"] = tuition.reindex(prof.index)
    return prof, led


def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, 1 - resid.var() / y.var()


def robust_sd(x):
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


# -------------------------------------------------------------------- calibration
def calibrate(prof, led):
    """Decode the cash-flow generator from (a window of) the ledger."""
    Y = prof.reference_monthly_net_income_chf
    R = prof.reference_monthly_rent_chf
    L = led.join(prof[["employment_status", "reference_monthly_net_income_chf"]], on="profile_id")
    L["t"] = (L.yr - 2025) * 12 + L.mon - 1  # month index from the January 2025 snapshot
    reg = L[L.mon.isin(REGULAR)]
    base = reg.groupby("profile_id").net_transactions_chf.median().reindex(prof.index)
    L["dev"] = L.net_transactions_chf - L.profile_id.map(base)
    L["rel"] = L.dev / L.reference_monthly_net_income_chf
    reg = L[L.mon.isin(REGULAR)]
    se = prof.employment_status.eq("self_employed")

    # Drift: regular-month net flow trends upward. Fit slope_i = g*Y_i - pi*(R_i + other_i),
    # i.e. pay growth g and cost inflation pi per year (non-self-employed profiles).
    tc = reg.t - reg.groupby("profile_id").t.transform("mean")
    slope = ((tc * reg.net_transactions_chf).groupby(reg.profile_id).sum()
             / (tc ** 2).groupby(reg.profile_id).sum()).reindex(prof.index) * 12
    other_raw = Y - R - base
    (g, pi), *_ = np.linalg.lstsq(np.column_stack([Y[~se], -(R + other_raw)[~se]]),
                                  slope[~se].values, rcond=None)
    tbar = float(reg.groupby("profile_id").t.mean().median())
    fy, fc = (1 + g) ** (tbar / 12), (1 + pi) ** (tbar / 12)

    # Non-rent living costs implied by income - rent - typical net flow, restated at January 2025.
    other = (Y * fy - R * fc - base) / fc
    city_other = other[~se].groupby(prof.city[~se]).median()
    other_used = other.where(~se, prof.city.map(city_other))  # SE months too noisy per profile

    # Monthly noise (robust to the shock tail) by status.
    sig = {s: robust_sd(reg.dev[reg.employment_status == s])
           for s in ["employed", "part_time", "retired", "student", "unemployed"]}
    sig_rel_se = robust_sd(reg.rel[reg.employment_status == "self_employed"])

    # Parameter uncertainty: standard error of each profile's typical monthly net flow.
    n_reg = reg.groupby("profile_id").size().reindex(prof.index)
    base_se = 1.2533 * prof.employment_status.map(sig) / np.sqrt(n_reg)  # SE of a median
    base_se = base_se.where(~se, float(other[~se].std()))  # SE: city-median living costs

    # One-off expense shocks: N(0, sigma_status) + Bernoulli(p_month) * Uniform(lo, hi), by MLE.
    pool = L[L.employment_status.isin(["employed", "part_time", "retired", "unemployed"])]
    xr, xs = pool[pool.mon.isin(REGULAR)], pool[pool.mon.isin([2, 8])]
    sr, ss = xr.employment_status.map(sig).values, xs.employment_status.map(sig).values

    def dens(x, s, p, lo, hi):
        mix = (stats.norm.cdf((x + hi) / s) - stats.norm.cdf((x + lo) / s)) / (hi - lo)
        return (1 - p) * stats.norm.pdf(x, 0, s) + p * mix

    def nll(th):
        pr, ps = 1 / (1 + np.exp(-th[:2]))
        lo, hi = 1000 * abs(th[2]), 1000 * abs(th[2]) + 1000 * abs(th[3]) + 50
        return -(np.log(dens(xr.dev.values, sr, pr, lo, hi) + 1e-300).sum()
                 + np.log(dens(xs.dev.values, ss, ps, lo, hi) + 1e-300).sum())

    th = optimize.minimize(nll, [-3.0, -0.6, 0.5, 1.2], method="Nelder-Mead",
                           options=dict(maxiter=4000, xatol=1e-5, fatol=1e-4)).x
    p_reg, p_sea = (1 / (1 + np.exp(-th[:2]))).tolist()
    shock_lo = 1000 * abs(th[2])
    shock_hi = shock_lo + 1000 * abs(th[3]) + 50

    # Student fees: extra outflow in Feb and Sep beyond what non-students show.
    stu = L[L.employment_status == "student"]
    fee = {m: max(0.0, pool.dev[pool.mon == m].mean() - stu.dev[stu.mon == m].mean()) for m in (2, 9)}

    # December bonus (employed; reused for self-employed, whose noise hides the split).
    z = L.rel[(L.mon == 12) & (L.employment_status == "employed")]
    has = z > 0.17
    p_bonus = float(has.mean())
    bonus_lo, bonus_hi = z[has].quantile([0.025, 0.975]).tolist()

    # Structural relationships used by interventions.
    med_inc = prof.groupby("employment_status").reference_monthly_net_income_chf.median()
    emp_inc = Y[prof.employment_status == "employed"]
    hh = pd.get_dummies(prof.household_type).drop(columns="couple")
    city = pd.get_dummies(prof.city).drop(columns="Bern")
    X = np.column_stack([np.ones(len(prof)), hh.values, city.values]).astype(float)
    beta, r2_rent = ols(np.log(R.values), X)
    hh_mult = {"couple": 1.0, **{c: float(np.exp(b)) for c, b in zip(hh.columns, beta[1:1 + hh.shape[1]])}}
    city_mult = {"Bern": 1.0, **{c: float(np.exp(b)) for c, b in zip(city.columns, beta[1 + hh.shape[1]:])}}

    def r2(x):
        return float(np.corrcoef(base, x)[0, 1] ** 2)

    cal = dict(
        calibration_window=f"{led.month.min()}..{led.month.max()}",
        net_flow_model=("net_t = income_t - rent_t - other_living_t - fees_t(student; Feb, Sep) - shock_t "
                        "+ bonus_t(Dec; employed, self-employed) + noise_t + offset_i - debt_service_t; "
                        "income grows at pay_growth, rent/living costs/fees/shocks at cost_inflation"),
        drift=dict(pay_growth_per_year=float(g), cost_inflation_per_year=float(pi),
                   note="fitted on per-profile trends of regular-month net flow; approximate"),
        base_flow_r2_explained_by={"income": r2(Y), "rent": r2(R), "other_living": r2(other)},
        base_flow_standard_error_chf={"median": float(base_se.median())},
        other_living_chf={"median": float(other.median()), "p10": float(other.quantile(.1)),
                          "p90": float(other.quantile(.9)),
                          "corr_with_income": float(np.corrcoef(other[~se], Y[~se])[0, 1])},
        other_living_city_median_chf={k: float(v) for k, v in city_other.items()},
        noise_sd_chf=sig, noise_sd_self_employed_share_of_income=sig_rel_se,
        shock=dict(p_regular_month=p_reg, p_feb_aug=p_sea, size_uniform_chf=[shock_lo, shock_hi]),
        student_fee_chf={"feb": fee[2], "sep": fee[9]},
        december_bonus=dict(p=p_bonus, size_share_of_income_uniform=[bonus_lo, bonus_hi]),
        income_median_by_status_chf={k: float(v) for k, v in med_inc.items()},
        entry_salary_chf=float(emp_inc.quantile(A["entry_salary_quantile"])),
        full_time_over_part_time_income=float(med_inc.employed / med_inc.part_time),
        self_employed_over_employed_income=float(med_inc.self_employed / med_inc.employed),
        rent_model=dict(form="rent = 2,100-ish x household_mult x city_mult x noise", r2_log=float(r2_rent),
                        household_mult=hh_mult, city_mult=city_mult),
    )
    # Per-status-code parameter vectors for the simulator.
    n = len(STATUS)
    cal["_sig_abs"] = np.array([sig["employed"], sig["part_time"], 0.0, sig["retired"], sig["student"],
                                sig["unemployed"], sig["unemployed"], sig["student"]])
    cal["_sig_rel"] = np.zeros(n)
    cal["_sig_rel"][SC["self_employed"]] = sig_rel_se
    cal["_p_bonus"] = np.zeros(n)
    cal["_p_bonus"][[SC["employed"], SC["self_employed"]]] = p_bonus
    cal["_fee_feb"], cal["_fee_sep"] = np.zeros(n), np.zeros(n)
    cal["_fee_feb"][SC["student"]], cal["_fee_sep"][SC["student"]] = fee[2], fee[9]
    return cal, other_used, base_se


# ---------------------------------------------------------------------- simulator
@dataclass
class State:
    pid: np.ndarray
    code: np.ndarray
    Y: np.ndarray
    R: np.ndarray
    O: np.ndarray
    cash0: np.ndarray
    inv0: np.ndarray
    debt0: np.ndarray
    rate0: np.ndarray
    age: np.ndarray
    hh: np.ndarray
    city: np.ndarray
    high_goal: np.ndarray
    high_target: np.ndarray
    med_target: np.ndarray
    tuition: np.ndarray
    base_se: np.ndarray

    @property
    def n(self):
        return len(self.pid)


def make_state(prof, other, base_se, cash0=None):
    return State(
        pid=prof.index.values, code=prof.employment_status.map(SC).values.astype(int),
        Y=prof.reference_monthly_net_income_chf.values.astype(float),
        R=prof.reference_monthly_rent_chf.values.astype(float), O=other.values.astype(float),
        base_se=base_se.values.astype(float),
        cash0=(prof.opening_cash_chf.values if cash0 is None else np.asarray(cash0)).astype(float),
        inv0=prof.inv0.values, debt0=prof.debt0.values, rate0=prof.rate0.values,
        age=prof.age_at_start.values, hh=prof.household_type.values, city=prof.city.values,
        high_goal=prof.high_goal.values, high_target=prof.high_target.values,
        med_target=prof.med_target.values.astype(float), tuition=prof.tuition.values.astype(float))


@dataclass
class Plan:
    Y: np.ndarray            # (T, P) income
    R: np.ndarray            # (T, P) rent
    O: np.ndarray            # (T, P) other living costs
    status: np.ndarray       # (T, P) status code: noise, bonus and fee regime
    oneoff: np.ndarray       # (T, P) one-off outflows
    extra_fee: np.ndarray    # (T, P) tuition outside the student regime
    invest_t0: np.ndarray    # (P,) cash -> investments at t0 (negative = sale)
    surplus_share: np.ndarray
    inv_factor: np.ndarray   # (P,) market shock applied to opening investments
    cash_t0: np.ndarray      # (P,) loan proceeds
    repay_t0: np.ndarray     # (P,) lump-sum principal repayment
    rate_shift: np.ndarray   # (P,) change in existing loan rate
    new_loan: np.ndarray     # (P,) new principal, amortised at A["new_loan_rate"]
    topup: np.ndarray        # (P,) bool: closed-loop "sell investments when cash is low" rule


def base_plan(st, T):
    P = st.n
    z = np.zeros(P)
    return Plan(Y=np.tile(st.Y, (T, 1)), R=np.tile(st.R, (T, 1)), O=np.tile(st.O, (T, 1)),
                status=np.tile(st.code, (T, 1)), oneoff=np.zeros((T, P)), extra_fee=np.zeros((T, P)),
                invest_t0=z.copy(), surplus_share=z.copy(), inv_factor=np.ones(P), cash_t0=z.copy(),
                repay_t0=z.copy(), rate_shift=z.copy(), new_loan=z.copy(), topup=np.zeros(P, bool))


def annuity_schedule(principal, rate, T, term):
    """Fixed-payment loan: interest is an expense, principal reduces debt."""
    i = rate / 12
    with np.errstate(divide="ignore", invalid="ignore"):
        pay = np.where(i > 0, principal * i / (1 - (1 + i) ** -term), principal / term)
    pay = np.where(principal > 0, pay, 0.0)
    bal = principal.astype(float).copy()
    pays, bals = np.zeros((T, len(bal))), np.zeros((T, len(bal)))
    for t in range(T):
        interest = bal * i
        p_t = np.minimum(pay, bal + interest)
        bal = bal + interest - p_t
        bal[bal < 1e-6] = 0.0
        pays[t], bals[t] = p_t, bal
    return pays, bals


def simulate(st, cal, plan, T, K, seed, debt=True, t_offset=0, overdraft=True):
    """Monthly cash/investment/debt paths for every profile, K paths each.

    Plan amounts are in January-2025 CHF; pay growth and cost inflation are applied
    here, counted from January 2025 (t_offset = months after it). Random draws depend
    only on (seed, month), so every plan sees the same shocks (common random numbers).
    Negative cash is allowed and counted as a funding gap, as in the ledger's
    negative_balance_requires_financing flag; it accrues A["overdraft_rate"] unless
    overdraft=False (the back-test, because the ledger charges none).
    """
    P = st.n
    g, pi = cal["drift"]["pay_growth_per_year"], cal["drift"]["cost_inflation_per_year"]
    offset = st.base_se[:, None] * np.random.default_rng([seed, 999_999]).standard_normal((P, K))
    cash = np.repeat((st.cash0 + plan.cash_t0 - plan.invest_t0 - plan.repay_t0)[:, None], K, 1)
    inv = np.repeat((st.inv0 * plan.inv_factor + plan.invest_t0)[:, None], K, 1)
    if debt:
        rate = np.where(st.debt0 > 0, np.maximum(A["rate_floor"], st.rate0 + plan.rate_shift), 0.0)
        p1, b1 = annuity_schedule(st.debt0 - plan.repay_t0, rate, T, A["loan_term"])
        p2, b2 = annuity_schedule(plan.new_loan, np.full(P, A["new_loan_rate"]), T, A["loan_term"])
        pay, bal = p1 + p2, b1 + b2
    else:
        pay, bal = np.zeros((T, P)), np.zeros((T, P))
    mu, sd = A["invest_return"] / 12, A["invest_vol"] / np.sqrt(12)
    lo, hi = cal["shock"]["size_uniform_chf"]
    blo, bhi = cal["december_bonus"]["size_share_of_income_uniform"]
    breach = np.zeros((P, K), bool)
    breach60 = np.zeros((P, K), bool)
    min_cash = cash.copy()
    snap = {}
    for t in range(T):
        mon = (t_offset + t) % 12 + 1
        rng = np.random.default_rng([seed, t])
        z, u1, u2, u3, u4 = (rng.standard_normal((P, K)), rng.random((P, K)), rng.random((P, K)),
                             rng.random((P, K)), rng.random((P, K)))
        zm = rng.standard_normal(K)  # market draw shared by all profiles on a path
        code = plan.status[t]
        gy, gc = (1 + g) ** ((t_offset + t) / 12), (1 + pi) ** ((t_offset + t) / 12)
        Y = plan.Y[t][:, None] * gy
        p_sh = cal["shock"]["p_feb_aug"] if mon in (2, 8) else cal["shock"]["p_regular_month"]
        net = (Y - gc * (plan.R[t][:, None] + plan.O[t][:, None] + plan.extra_fee[t][:, None])
               - plan.oneoff[t][:, None] - pay[t][:, None]
               - gc * (u1 < p_sh) * (lo + (hi - lo) * u2)
               + (cal["_sig_abs"][code][:, None] + cal["_sig_rel"][code][:, None] * Y) * z
               + offset)
        if mon == 12:
            net += (u3 < cal["_p_bonus"][code][:, None]) * Y * (blo + (bhi - blo) * u4)
        if mon == 2:
            net -= gc * cal["_fee_feb"][code][:, None]
        if mon == 9:
            net -= gc * cal["_fee_sep"][code][:, None]
        if overdraft:
            net += np.minimum(cash, 0) * A["overdraft_rate"] / 12
        cash += net
        inv *= 1 + mu + sd * zm[None, :]
        if plan.surplus_share.any():
            buf = A["surplus_buffer_months"] * (gc * (plan.R[t] + plan.O[t]) + pay[t])
            x = plan.surplus_share[:, None] * np.maximum(net, 0) * (cash > buf[:, None])
            cash -= x
            inv += x
        if plan.topup.any():
            outgo = (gc * (plan.R[t] + plan.O[t]) + pay[t])[:, None]
            low = plan.topup[:, None] & (cash < A["topup_floor_months"] * outgo)
            sell = np.where(low, np.minimum(inv, np.maximum(0, A["topup_target_months"] * outgo - cash)), 0)
            cash += sell
            inv -= sell
        neg = cash < 0
        breach |= neg
        if t < 60:
            breach60 |= neg
        np.minimum(min_cash, cash, out=min_cash)
        if t in (T_HIGH, T_MED, T - 1):
            snap[t] = (cash.copy(), inv.copy(), bal[t].copy())
    return dict(breach=breach, breach60=breach60, min_cash=min_cash, snap=snap, cash=cash)


def summarize(st, res, T):
    out = dict(p_breach=res["breach"].mean(1), p_breach_5y=res["breach60"].mean(1),
               gap_mean=np.maximum(0, -res["min_cash"]).mean(1))
    c, i, d = res["snap"][T_HIGH]
    emerg = (st.high_goal == "Emergency savings")[:, None]
    met = np.where(emerg, c >= st.high_target[:, None], c + i - d[:, None] >= st.high_target[:, None])
    out["p_high_goal"] = met.mean(1)
    c, i, d = res["snap"][T_MED]
    need = (st.high_target + np.nan_to_num(st.med_target))[:, None]  # reserve stays + medium goal on top
    out["p_med_goal"] = np.where(np.isnan(st.med_target), np.nan, (c + i - d[:, None] >= need).mean(1))
    c, i, d = res["snap"][T - 1]
    nw = c + i - d[:, None]
    out["nw_median"] = np.median(nw, 1)
    out["nw_p10"] = np.percentile(nw, 10, axis=1)
    return pd.DataFrame(out, index=st.pid)


# ------------------------------------------------------------------ interventions
REGISTRY = []


def intervention(kind, family, catalogue, basis):
    def wrap(fn):
        REGISTRY.append(dict(name=fn.__name__, kind=kind, family=family, catalogue=catalogue,
                             basis=basis, doc=(fn.__doc__ or "").strip(), fn=fn))
        return fn
    return wrap


def working(st):
    return np.isin(st.code, [SC["employed"], SC["part_time"], SC["self_employed"]])


@intervention("decision", "career", "S011, S014 exit", "data: entry pay = employed p25; assumed 3-month search")
def take_full_time_job(st, pl, cal):
    """Student or unemployed person starts a full-time job."""
    m = np.isin(st.code, [SC["student"], SC["unemployed"]])
    d = A["job_search_months"]
    pl.Y[d:, m] = cal["entry_salary_chf"]
    pl.status[d:, m] = SC["employed"]
    return m


@intervention("decision", "career", "S020", "data: employed/part-time median income ratio")
def part_time_to_full_time(st, pl, cal):
    """Part-time worker moves to full-time."""
    m = st.code == SC["part_time"]
    pl.Y[:, m] *= cal["full_time_over_part_time_income"]
    pl.status[:, m] = SC["employed"]
    return m


@intervention("decision", "career", "S012, S013", "assumed +10%, certain")
def negotiate_raise(st, pl, cal):
    """Pay rise of 10% from month 0."""
    m = working(st)
    pl.Y[:, m] *= 1 + A["raise_"]
    return m


@intervention("decision", "career", "S017, S087, S096", "assumed income x0.8")
def reduce_workload_80pct(st, pl, cal):
    """Employed or self-employed person cuts workload to 80%."""
    m = np.isin(st.code, [SC["employed"], SC["self_employed"]])
    pl.Y[:, m] *= 1 - A["workload_cut"]
    return m


@intervention("decision", "career", "S111, S115", "data: SE/employed income ratio, SE volatility; assumed cost and delay")
def start_business(st, pl, cal):
    """Leave work, 12 months without revenue, CHF 15k set-up, then self-employed income."""
    m = np.isin(st.code, [SC["employed"], SC["part_time"]]) & (st.age <= 60)
    d = A["business_delay"]
    pl.Y[:d, m] = 0.0
    pl.status[:d, m] = SC["startup"]
    pl.Y[d:, m] = st.Y[m] * cal["self_employed_over_employed_income"]
    pl.status[d:, m] = SC["self_employed"]
    pl.oneoff[0, m] += A["business_cost"]
    return m


@intervention("decision", "education", "S001", "data: S001 tuition, student income; assumed +10% premium")
def full_time_masters(st, pl, cal):
    """Two-year master's on a student budget, then a job paying 10% more."""
    m = np.isin(st.code, [SC["employed"], SC["part_time"], SC["student"]]) & (st.age <= 35)
    n = A["masters_months"]
    tuition = np.where(np.isnan(st.tuition), np.nanmean(st.tuition), st.tuition)
    pl.Y[:n, m] = cal["income_median_by_status_chf"]["student"]
    pl.status[:n, m] = SC["masters"]
    for t in range(n):
        if t % 12 + 1 in (2, 9):
            pl.extra_fee[t, m] += tuition[m] / 2
    pl.Y[n:, m] = (np.maximum(st.Y, cal["entry_salary_chf"]) * (1 + A["masters_premium"]))[m]
    pl.status[n:, m] = SC["employed"]
    return m


@intervention("decision", "retirement", "S091", "data: retired median income")
def retire_early(st, pl, cal):
    """Worker aged 58+ retires now on the typical retired income."""
    m = working(st) & (st.age >= 58)
    pl.Y[:, m] = np.minimum(st.Y, cal["income_median_by_status_chf"]["retired"])[m]
    pl.status[:, m] = SC["retired"]
    return m


@intervention("shock", "career", "S014, S104", "data: unemployed median income")
def job_loss_6m(st, pl, cal):
    """Six months on the typical unemployed income, then back to the old job."""
    m = working(st)
    d = A["job_loss_months"]
    pl.Y[:d, m] = cal["income_median_by_status_chf"]["unemployed"]
    pl.status[:d, m] = SC["unemployed"]
    return m


@intervention("decision", "housing", "S032, S049", "data: household rent multipliers; assumed moving cost")
def move_to_shared_flat(st, pl, cal):
    """Single household moves into shared housing."""
    hm = cal["rent_model"]["household_mult"]
    m = st.hh == "single"
    pl.R[:, m] *= hm["shared_housing"] / hm["single"]
    pl.oneoff[0, m] += A["moving_cost"]
    return m


@intervention("decision", "housing", "S039", "assumed rent -20% and moving cost")
def move_to_cheaper_flat(st, pl, cal):
    """Any household moves to a flat 20% cheaper."""
    m = np.ones(st.n, bool)
    pl.R *= 1 - A["cheaper_flat"]
    pl.oneoff[0] += A["moving_cost"]
    return m


@intervention("decision", "relocation", "S021, S022, S088", "data: city rent and living-cost effects; income unaffected by city")
def relocate_to_cheapest_city(st, pl, cal):
    """Move to the city with the lowest modelled rent (St Gallen in this data)."""
    cm = cal["rent_model"]["city_mult"]
    co = cal["other_living_city_median_chf"]
    target = min(cm, key=cm.get)
    m = st.city != target
    pl.R[:, m] *= (cm[target] / np.array([cm[c] for c in st.city]))[m]
    pl.O[:, m] += (co[target] - np.array([co[c] for c in st.city]))[m]
    pl.oneoff[0, m] += A["relocation_cost"]
    return m


@intervention("decision", "housing", "S031", "data: household rent multipliers")
def rent_alone(st, pl, cal):
    """Shared-housing household moves to a flat of its own."""
    hm = cal["rent_model"]["household_mult"]
    m = st.hh == "shared_housing"
    pl.R[:, m] *= hm["single"] / hm["shared_housing"]
    pl.oneoff[0, m] += A["moving_cost"]
    return m


@intervention("shock", "housing", "S038, S109", "assumed +15%")
def rent_increase_15pct(st, pl, cal):
    """Landlord raises rent by 15%."""
    pl.R *= 1 + A["rent_increase"]
    return np.ones(st.n, bool)


@intervention("decision", "spending", "S045, S047, S050", "assumed -10% of non-rent living costs")
def cut_spending_10pct(st, pl, cal):
    """Cut non-rent living costs by 10% (about CHF 150 a month here)."""
    pl.O *= 1 - A["spending_change"]
    return np.ones(st.n, bool)


@intervention("decision", "spending", "S043, S044", "assumed +10% of non-rent living costs")
def lifestyle_inflation_10pct(st, pl, cal):
    """Raise non-rent living costs by 10%."""
    pl.O *= 1 + A["spending_change"]
    return np.ones(st.n, bool)


@intervention("decision", "spending", "S042, S047", "assumed price and running cost")
def buy_car_cash(st, pl, cal):
    """Buy a CHF 25k car from cash and run it for CHF 400 a month."""
    pl.oneoff[0] += A["car_price"]
    pl.O += A["car_running"]
    return np.ones(st.n, bool)


@intervention("shock", "spending", "S048, S055", "assumed CHF 10k")
def emergency_expense_10k(st, pl, cal):
    """Unplanned CHF 10k bill in month 0."""
    pl.oneoff[0] += A["emergency_expense"]
    return np.ones(st.n, bool)


@intervention("decision", "investing", "S056, S061", "assumed 4% return, 10% volatility; 6-month buffer")
def invest_excess_cash(st, pl, cal):
    """Move cash above six months of outgoings into the portfolio."""
    pay = annuity_schedule(st.debt0, st.rate0, 1, A["loan_term"])[0][0]
    excess = np.maximum(0, st.cash0 - A["invest_buffer_months"] * (st.R + st.O + pay))
    pl.invest_t0 = excess
    return excess > 0


@intervention("decision", "investing", "S052, S062", "assumed share and buffer")
def invest_half_of_surplus(st, pl, cal):
    """Each month, invest half of any positive net flow while cash exceeds a 3-month buffer."""
    pl.surplus_share[:] = A["surplus_share"]
    return np.ones(st.n, bool)


@intervention("decision", "investing", "S060, S068", "no fees, immediate access")
def liquidate_investments(st, pl, cal):
    """Sell the whole portfolio into cash at month 0."""
    m = st.inv0 > 0
    pl.invest_t0 = -st.inv0
    return m


@intervention("decision", "investing", "S060, S068", "rule: sell below 1 month of outgoings, refill to 3")
def sell_investments_when_cash_low(st, pl, cal):
    """Closed-loop rule: whenever cash drops below one month of outgoings, sell investments to restore three."""
    m = st.inv0 > 0
    pl.topup[m] = True
    return m


@intervention("decision", "investing", "S056, S061, S068", "invest_excess_cash plus the top-up rule")
def invest_excess_with_top_up(st, pl, cal):
    """Invest cash above six months of outgoings, then sell back whenever cash drops below one month."""
    pay = annuity_schedule(st.debt0, st.rate0, 1, A["loan_term"])[0][0]
    excess = np.maximum(0, st.cash0 - A["invest_buffer_months"] * (st.R + st.O + pay))
    m = excess > 0
    pl.invest_t0 = excess
    pl.topup[m] = True
    return m


@intervention("shock", "investing", "S065, S105", "assumed -30%")
def market_crash_30pct(st, pl, cal):
    """Portfolio loses 30% at month 0."""
    m = st.inv0 > 0
    pl.inv_factor[m] = 1 - A["crash"]
    return m


@intervention("decision", "debt", "S040, S072, S080", "assumed 3-month buffer kept")
def repay_debt_from_cash(st, pl, cal):
    """Repay as much loan principal as cash above a 3-month buffer allows."""
    room = np.maximum(0, st.cash0 - A["repay_buffer_months"] * (st.R + st.O))
    amount = np.minimum(st.debt0, room)
    pl.repay_t0 = amount
    return amount > 0


@intervention("decision", "debt", "S074, S077", "assumed -2 points, no fees")
def refinance_minus_2pts(st, pl, cal):
    """Refinance the loan two percentage points cheaper."""
    m = st.debt0 > 0
    pl.rate_shift[m] = -A["rate_step"]
    return m


@intervention("shock", "debt", "S073, S102", "assumed +2 points on a variable rate")
def rate_hike_2pts(st, pl, cal):
    """Loan rate rises two percentage points."""
    m = st.debt0 > 0
    pl.rate_shift[m] = A["rate_step"]
    return m


@intervention("decision", "debt", "S076, S079, S009", "assumed CHF 20k at 6.5% over 60 months")
def take_personal_loan_20k(st, pl, cal):
    """Borrow CHF 20k now and repay it over five years."""
    pl.cash_t0 += A["new_loan"]
    pl.new_loan[:] = A["new_loan"]
    return np.ones(st.n, bool)


@intervention("decision", "family", "S082, S083", "assumed child cost and leave; data has no household-size cost effect")
def have_child(st, pl, cal):
    """Couple or family aged 25-45 adds CHF 1,800 a month of costs and 4 months on 80% pay."""
    m = np.isin(st.hh, ["couple", "family"]) & (st.age >= 25) & (st.age <= 45)
    pl.O[:, m] += A["child_cost"]
    pl.Y[:A["leave_months"], m] *= A["leave_income"]
    return m


@intervention("shock", "macro", "S101, S106, S108", "assumed +2%/yr on rent and living costs, flat pay")
def inflation_2pct_no_raise(st, pl, cal):
    """Rent and living costs grow 2% a year faster than pay."""
    f = (1 + A["inflation_extra"]) ** (np.arange(pl.R.shape[0]) / 12)
    pl.R *= f[:, None]
    pl.O *= f[:, None]
    return np.ones(st.n, bool)


# ----------------------------------------------------------------------- analyses
def backtest(prof, led):
    """Calibrate on 2025 only, simulate 2026 from the recorded 2025-12 cash, compare."""
    l25 = led[led.yr == 2025]
    cal, other, base_se = calibrate(prof, l25)
    close25 = l25[l25.mon == 12].set_index("profile_id").closing_cash_chf.reindex(prof.index)
    st = make_state(prof, other, base_se, cash0=close25.values)
    res = simulate(st, cal, base_plan(st, 12), 12, N_PATHS, SEED + 1, debt=False, t_offset=12, overdraft=False)
    l26 = led[led.yr == 2026]
    obs_breach = (l26.groupby("profile_id").closing_cash_chf.min() < 0).reindex(prof.index).values
    obs_final = l26[l26.mon == 12].set_index("profile_id").closing_cash_chf.reindex(prof.index).values
    p = res["breach"].mean(1)
    sims = res["cash"]
    q05, q25, q75, q95 = np.percentile(sims, [5, 25, 75, 95], axis=1)
    pit = (sims < obs_final[:, None]).mean(1)
    count_paths = res["breach"].sum(0)
    from sklearn.metrics import roc_auc_score
    by_status = (pd.DataFrame({"status": prof.employment_status.values, "obs": obs_breach, "exp": p})
                 .groupby("status").sum().round(1))
    return dict(
        design="calibrate on 2025-01..2025-12, simulate 2026-01..2026-12 from recorded 2025-12 cash (no debt service, as in the ledger)",
        profiles_breaching_2026_observed=int(obs_breach.sum()),
        profiles_breaching_2026_expected=float(p.sum()),
        profiles_breaching_2026_90pct_interval=np.percentile(count_paths, [5, 95]).tolist(),
        breach_auc=float(roc_auc_score(obs_breach, p)),
        breach_brier=float(np.mean((p - obs_breach) ** 2)),
        final_cash_coverage_50pct=float(np.mean((obs_final >= q25) & (obs_final <= q75))),
        final_cash_coverage_90pct=float(np.mean((obs_final >= q05) & (obs_final <= q95))),
        final_cash_pit_deciles=np.histogram(pit, bins=np.linspace(0, 1, 11))[0].tolist(),
        breach_by_status_observed_vs_expected=by_status.to_dict(orient="index"),
    )


# Decisions that cost money in every simulated case: they only make sense as user-chosen
# branches or with an explicit preference utility in the reward.
LIFE_CHOICES = {"start_business", "full_time_masters", "retire_early", "reduce_workload_80pct",
                "have_child", "buy_car_cash", "lifestyle_inflation_10pct", "rent_alone"}


def role(spec):
    if spec["kind"] == "shock":
        return "exogenous_shock"
    return "life_choice" if spec["name"] in LIFE_CHOICES else "financial_lever"


def outcome_changes(st, base, res, elig):
    """Per-profile material changes: probabilities by >= A['material'], wealth by >= 10% (min CHF 10k)."""
    m = A["material"]
    d = res - base
    dpb, dph = d.p_breach.values, d.p_high_goal.values
    dpm = np.nan_to_num(d.p_med_goal.values)
    dnw = d.nw_median.values
    thr = np.maximum(0.10 * np.abs(base.nw_median.values), 10_000)
    risk = elig & ((np.abs(dpb) >= m) | (np.abs(dph) >= m) | (np.abs(dpm) >= m))
    wealth = elig & (np.abs(dnw) >= thr)
    better = elig & ((dpb <= -m) | (dph >= m) | (dpm >= m) | (dnw >= thr))
    worse = elig & ((dpb >= m) | (dph <= -m) | (dpm <= -m) | (dnw <= -thr))
    return d, dpb, dph, dpm, dnw, risk, wealth, better, worse


def rank_interventions(st, base, effects, prof):
    status = prof.employment_status.values
    rows, by_status, per_profile = [], [], []
    for spec in REGISTRY:
        res, elig = effects[spec["name"]]
        e = elig
        has_med = ~np.isnan(st.med_target)
        d, dpb, dph, dpm, dnw, risk, wealth, better, worse = outcome_changes(st, base, res, e)
        material = risk | wealth
        b_pb, r_pb = base.p_breach.values, res.p_breach.values
        b_ph, r_ph = base.p_high_goal.values, res.p_high_goal.values
        nb, nw_ = int(better.sum()), int(worse.sum())
        character = ("no financial upside" if nb == 0 else "dominant (no downside)" if nw_ == 0
                     else "trade-off")
        rows.append(dict(
            intervention=spec["name"], kind=spec["kind"], role=role(spec), family=spec["family"],
            catalogue=spec["catalogue"], basis=spec["basis"], description=spec["doc"],
            eligible=int(e.sum()),
            material_profiles=int(material.sum()),
            material_share_of_all=round(material.sum() / st.n, 4),
            material_share_of_eligible=round(material.sum() / max(1, e.sum()), 4),
            risk_material_share_of_all=round(risk.sum() / st.n, 4),
            better_profiles=nb, worse_profiles=nw_, character=character,
            both_better_and_worse=int((better & worse).sum()),
            breach_fixed=int((e & (b_pb >= .5) & (r_pb < .5)).sum()),
            breach_created=int((e & (b_pb < .5) & (r_pb >= .5)).sum()),
            high_goal_gained=int((e & (b_ph < .5) & (r_ph >= .5)).sum()),
            high_goal_lost=int((e & (b_ph >= .5) & (r_ph < .5)).sum()),
            mean_dp_breach_pts=round(100 * dpb[e].mean(), 2) if e.any() else 0.0,
            mean_dp_high_goal_pts=round(100 * dph[e].mean(), 2) if e.any() else 0.0,
            mean_dp_med_goal_pts=round(100 * d.p_med_goal.values[e & has_med].mean(), 2) if (e & has_med).any() else np.nan,
            median_dnw_chf=round(float(np.median(dnw[e])), 0) if e.any() else 0.0,
            median_dnw_p10_chf=round(float(np.median(d.nw_p10.values[e])), 0) if e.any() else 0.0,
        ))
        frame = pd.DataFrame(dict(status=status, e=e, dpb=dpb, dph=dph, dnw=dnw, mat=material,
                                  better=better, worse=worse))
        for s, g in frame[frame.e].groupby("status"):
            by_status.append(dict(intervention=spec["name"], role=role(spec), status=s, eligible=len(g),
                                  mean_dp_breach_pts=round(100 * g.dpb.mean(), 1),
                                  mean_dp_high_goal_pts=round(100 * g.dph.mean(), 1),
                                  median_dnw_chf=round(g.dnw.median(), 0),
                                  material_share=round(g.mat.mean(), 3),
                                  better_share=round(g.better.mean(), 3), worse_share=round(g.worse.mean(), 3)))
        pp = d[["p_breach", "p_high_goal", "p_med_goal", "nw_median"]].add_prefix("delta_")
        pp.insert(0, "intervention", spec["name"])
        per_profile.append(pp[e].reset_index(names="profile_id"))
    ranking = pd.DataFrame(rows)
    return ranking, pd.DataFrame(by_status), pd.concat(per_profile, ignore_index=True)


def best_lever(st, base, effects, prof, exclude=("negotiate_raise",)):
    """Best financial lever per profile vs. continuing, lexicographically:
    lower breach probability, then higher high-goal probability, then medium goal, then wealth.
    Differences under 2 points (or CHF 1,000) count as ties. A certain pay rise is excluded:
    its outcome is not the person's to choose."""
    names = ["continue"] + [s["name"] for s in REGISTRY
                            if role(s) == "financial_lever" and s["name"] not in exclude]
    P = st.n
    cols = {"continue": (base, np.ones(P, bool))} | {n: effects[n] for n in names[1:]}
    best = np.zeros(P, int)
    for i in range(P):
        cand = []
        for j, n in enumerate(names):
            res, elig = cols[n]
            if not elig[i]:
                continue
            r = res.iloc[i]
            cand.append((j, np.round(-r.p_breach / 0.02), np.round(r.p_high_goal / 0.02),
                         np.round(np.nan_to_num(r.p_med_goal) / 0.02), r.nw_median / 1_000))
        best[i] = max(cand, key=lambda c: (c[1], c[2], c[3], c[4]))[0]
    out = pd.DataFrame(dict(profile_id=st.pid, status=prof.employment_status.values,
                            best=np.array(names)[best]))
    return out


def tier(row, base_breach_n):
    """Tier 1: changes the outcome for >= 20% of all profiles or fixes >= 25% of baseline shortfalls.
    Tier 2: >= 5% of all profiles, or >= 40% of an eligible group of 20+. Tier 3: the rest."""
    if row.material_share_of_all >= 0.20 or row.breach_fixed >= 0.25 * base_breach_n:
        return 1
    if row.material_share_of_all >= 0.05 or (row.eligible >= 20 and row.material_share_of_eligible >= 0.40):
        return 2
    return 3


def run_scenario(prof, led, label):
    """Calibrate, simulate the baseline and every intervention, and rank them."""
    cal, other, base_se = calibrate(prof, led)
    st = make_state(prof, other, base_se)
    base = summarize(st, simulate(st, cal, base_plan(st, T_FWD), T_FWD, N_PATHS, SEED), T_FWD)
    effects = {}
    for spec in REGISTRY:
        plan = base_plan(st, T_FWD)
        elig = spec["fn"](st, plan, cal)
        effects[spec["name"]] = (summarize(st, simulate(st, cal, plan, T_FWD, N_PATHS, SEED), T_FWD), elig)
    ranking, by_status, per_profile = rank_interventions(st, base, effects, prof)
    base_breach_n = int((base.p_breach >= .5).sum())
    ranking["tier"] = ranking.apply(tier, axis=1, base_breach_n=base_breach_n)
    ranking["scenario"] = label
    print(f"  scenario {label}: done")
    return cal, st, base, effects, ranking, by_status, per_profile


def tier(row, base_breach_n):
    """Tier rule for decisions; shocks are tiered the same way for the transition model."""
    if row.material_share_of_all >= 0.20 or row.breach_fixed >= 0.25 * base_breach_n:
        return 1
    if row.material_share_of_all >= 0.05 or (row.eligible >= 20 and row.material_share_of_eligible >= 0.40):
        return 2
    return 3


def state_importance(prof, st, base):
    """Grouped permutation importance (5-fold CV random forest) of initial state on baseline outcomes."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    num = pd.DataFrame(dict(income=st.Y, rent=st.R, other_living=st.O, cash=st.cash0, investments=st.inv0,
                            debt=st.debt0, loan_rate=st.rate0, age=st.age,
                            has_medium_goal=~np.isnan(st.med_target),
                            goal_is_emergency=st.high_goal == "Emergency savings"), index=prof.index)
    cats = {c: pd.get_dummies(prof[c], prefix=c) for c in ["employment_status", "household_type", "city"]}
    X = pd.concat([num, *cats.values()], axis=1).astype(float)
    groups = {c: [c] for c in num.columns} | {c: list(v.columns) for c, v in cats.items()}
    targets = {"p_breach": base.p_breach, "p_high_goal": base.p_high_goal,
               "nw_median": base.nw_median, "p_med_goal": base.p_med_goal}
    rng = np.random.default_rng(SEED)
    rows = []
    for name, y in targets.items():
        ok = ~y.isna().values
        Xv, yv = X.values[ok], y.values[ok]
        drops = {g: [] for g in groups}
        r2s = []
        for tr, te in KFold(5, shuffle=True, random_state=SEED).split(Xv):
            rf = RandomForestRegressor(300, min_samples_leaf=3, random_state=SEED, n_jobs=-1).fit(Xv[tr], yv[tr])
            r2 = rf.score(Xv[te], yv[te])
            r2s.append(r2)
            for g, cols in groups.items():
                idx = [X.columns.get_loc(c) for c in cols]
                for _ in range(3):
                    Xp = Xv[te].copy()
                    Xp[:, idx] = Xp[rng.permutation(len(te))][:, idx]
                    drops[g].append(r2 - rf.score(Xp, yv[te]))
        total = sum(max(0.0, float(np.mean(v))) for v in drops.values())
        for g, v in drops.items():
            rows.append(dict(outcome=name, cv_r2=round(float(np.mean(r2s)), 3), feature=g,
                             r2_drop_when_permuted=round(float(np.mean(v)), 4),
                             share_of_importance=round(max(0.0, float(np.mean(v))) / total, 4)))
    out = pd.DataFrame(rows)
    return out.sort_values(["outcome", "r2_drop_when_permuted"], ascending=[True, False])


# Assumption sets for the robustness check (only keys that differ from A).
SENSITIVITY = {
    "pessimistic": dict(invest_return=0.02, invest_vol=0.15, entry_salary_quantile=0.10, masters_premium=0.0,
                        cheaper_flat=0.10, moving_cost=4_000, relocation_cost=6_000, child_cost=2_500,
                        business_cost=25_000, job_search_months=6),
    "optimistic": dict(invest_return=0.06, invest_vol=0.10, entry_salary_quantile=0.50, masters_premium=0.20,
                       cheaper_flat=0.30, moving_cost=1_000, relocation_cost=2_000, child_cost=1_200,
                       business_cost=8_000, job_search_months=1),
}


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return round(float(x), 4)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    prof, led = load()

    bt = backtest(prof, led)
    (OUT / "backtest.json").write_text(json.dumps(jsonable(bt), indent=2) + "\n")
    print("backtest:", {k: bt[k] for k in ["profiles_breaching_2026_observed", "profiles_breaching_2026_expected",
                                          "final_cash_coverage_50pct", "final_cash_coverage_90pct"]})

    cal, st, base, effects, ranking, by_status, per_profile = run_scenario(prof, led, "reference")
    (OUT / "calibration.json").write_text(json.dumps(jsonable(cal | {"assumptions": A}), indent=2) + "\n")
    base.assign(employment_status=prof.employment_status.values).round(4).to_csv(
        OUT / "baseline_by_profile.csv", index_label="profile_id")
    order = {"financial_lever": 0, "life_choice": 1, "exogenous_shock": 2}
    ranking = ranking.sort_values(["role", "tier", "material_share_of_all"],
                                  key=lambda c: c.map(order) if c.name == "role" else c,
                                  ascending=[True, True, False])
    ranking.to_csv(OUT / "action_ranking.csv", index=False)
    by_status.to_csv(OUT / "action_by_status.csv", index=False)
    per_profile.round(4).to_csv(OUT / "action_effects_by_profile.csv", index=False)

    best = best_lever(st, base, effects, prof)
    best_tab = pd.crosstab(best.best, best.status, margins=True, margins_name="all")
    best_tab.to_csv(OUT / "best_financial_lever_by_status.csv")

    imp = state_importance(prof, st, base)
    imp.to_csv(OUT / "state_importance.csv", index=False)

    # Robustness: does the tiering survive pessimistic/optimistic assumptions?
    saved = dict(A)
    tiers = [ranking[["intervention", "role", "tier", "material_share_of_all"]].assign(scenario="reference")]
    for label, overrides in SENSITIVITY.items():
        A.update(overrides)
        r = run_scenario(prof, led, label)[4]
        tiers.append(r[["intervention", "role", "tier", "material_share_of_all"]].assign(scenario=label))
        A.clear()
        A.update(saved)
    sens = pd.concat(tiers).pivot_table(index=["intervention", "role"], columns="scenario",
                                        values=["tier", "material_share_of_all"]).round(3)
    sens.columns = [f"{a}_{b}" for a, b in sens.columns]
    sens.to_csv(OUT / "sensitivity_tiers.csv")

    status = prof.employment_status.values
    base_breach_n = int((base.p_breach >= .5).sum())
    summary = dict(
        horizon=f"2025-01..2034-12 ({T_FWD} months), {N_PATHS} paths per profile, seed {SEED}",
        baseline_definition=("continue current situation; data-fitted pay growth and cost inflation; "
                             "loans amortised over 60 months and a 4%/10% portfolio (assumptions)"),
        baseline=dict(
            profiles_with_breach_prob_over_50pct=base_breach_n,
            by_status=(base.assign(status=status).groupby("status")
                       [["p_breach", "p_breach_5y", "p_high_goal", "p_med_goal", "nw_median"]].mean()
                       .round(3).to_dict(orient="index")),
            mean_p_high_goal=float(base.p_high_goal.mean()),
            mean_p_med_goal=float(base.p_med_goal.mean()),
        ),
        tier_changes_under_sensitivity=int((sens.filter(like="tier_").nunique(axis=1) > 1).sum()),
        runtime_seconds=round(time.time() - t0, 1),
    )
    (OUT / "summary.json").write_text(json.dumps(jsonable(summary), indent=2) + "\n")
    cols = ["intervention", "role", "tier", "character", "eligible", "material_share_of_all", "better_profiles",
            "worse_profiles", "breach_fixed", "high_goal_gained", "high_goal_lost", "mean_dp_breach_pts",
            "mean_dp_high_goal_pts", "median_dnw_chf"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(ranking[cols].to_string(index=False))
        print(best_tab)
        print(sens.to_string())
        print(imp[imp.share_of_importance > 0.01].to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
