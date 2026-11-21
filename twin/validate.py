"""Acceptance tests: the flow network must match or beat the hand-calibrated baseline.

T1  held-out people: CRPS of 2026 monthly flows, network (trained without those people)
    vs baseline. Both use only 2025 of the scored person's own data.
T2  trained on 2025 only, intervals for end-2026 cash cover about 50% and 90%.
T3  rolled forward from December 2025, about as many profiles go below zero in 2026 as did.
T4  simulated 2025-2026 ledgers show the data's patterns: December bonus, Feb/Aug bills,
    student fees, noise by pay, no carry-over between months, bills unrelated to pay.
The baseline runs through the same code, so the tests are checked against it too.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config as C
from .data import indices, ledger, load_priors, person_folds, profile_months, profiles
from .world.baseline import baseline_flows
from .world.flow import SCALE, FlowModel, crps, features, frame_features

K_PATHS = 200


def nn_flows(fm: FlowModel, rows: pd.DataFrame, t0: int, T: int, K: int, seed: int, priors: dict):
    """Monthly nominal flows (P x K x T) for people who keep their situation ('continue')."""
    P = len(rows)
    b, var = fm.offset_for(rows.index.values)
    rng0 = np.random.default_rng([seed, 999_999])
    member = rng0.integers(0, len(fm.members), P * K)
    off = np.repeat(b, K) + np.repeat(np.sqrt(var), K) * rng0.standard_normal(P * K)
    rep = lambda a: np.repeat(np.asarray(a), K)  # noqa: E731
    mode, se, pens, city = rep(rows["mode"]), rep(rows["self_employed"]), rep(rows["pensioner"]), rep(rows["city_idx"])
    inc0, rent0 = rep(rows["income0"]), rep(rows["rent0"])
    flows = np.empty((P, K, T))
    for t in range(T):
        tt = t0 + t
        G, Pi = indices(tt, priors)
        X = features(mode, se, pens, inc0 * G / Pi, rent0, city, np.full(P * K, tt % 12 + 1))
        rng = np.random.default_rng([seed, tt])
        y = fm.sample(X, member, rng.random(P * K), rng.standard_normal(P * K)) + off
        flows[:, :, t] = (y * SCALE * Pi).reshape(P, K)
    return flows


def robust_sd(x):
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def pattern_stats(flow: np.ndarray, prof: pd.DataFrame) -> dict:
    """Ledger patterns for flows of shape (P, 24), months 2025-01..2026-12."""
    T = flow.shape[1]
    moy = np.arange(T) % 12 + 1
    reg = np.isin(moy, C.REGULAR)
    resid = flow - np.median(flow[:, reg], 1)[:, None]
    income = prof["income0"].values
    status = prof["status"].values
    ft = status == "employed"
    pool = np.isin(status, ["employed", "part_time", "unemployed", "retired"])
    stu = status == "student"
    out = {}
    out["bonus_share"] = float((resid[ft][:, moy == 12] > 0.17 * income[ft][:, None]).mean())
    extra = resid[:, np.isin(moy, [2, 8])].mean(1) - resid[:, reg].mean(1)
    out["feb_aug_extra_cost"] = float(-extra[pool].mean())
    out["bill_cost_per_1000_pay"] = float(np.polyfit(income[ft] / 1000, -extra[ft], 1)[0])
    out["student_sep_fee"] = float(-(resid[stu][:, moy == 9].mean() - resid[stu][:, reg].mean()))
    q = pd.qcut(income[ft], 5, labels=False)
    out["noise_sd_by_pay_quintile"] = [robust_sd(resid[ft][q == i][:, reg]) for i in range(5)]
    pairs = [(t, t + 1) for t in range(T - 1) if reg[t] and reg[t + 1]]
    a = np.concatenate([resid[:, i] for i, _ in pairs])
    b = np.concatenate([resid[:, j] for _, j in pairs])
    out["next_month_corr"] = float(np.corrcoef(a, b)[0, 1])
    out["mean_flow_by_mode"] = [float(flow[prof["mode"].values == m].mean()) for m in range(4)]
    return out


def pattern_checks(sim: dict, real: dict) -> dict:
    """Tolerances for 'shows the same patterns'."""
    def close(a, b, rel, floor):
        return abs(a - b) <= max(floor, rel * abs(b))
    return {
        "bonus_share": abs(sim["bonus_share"] - real["bonus_share"]) <= 0.06,
        "feb_aug_extra_cost": close(sim["feb_aug_extra_cost"], real["feb_aug_extra_cost"], 0.25, 50),
        "bill_cost_per_1000_pay": abs(sim["bill_cost_per_1000_pay"]) <= 30,   # bills unrelated to pay
        "student_sep_fee": close(sim["student_sep_fee"], real["student_sep_fee"], 0.30, 100),
        "noise_sd_by_pay_quintile": all(0.8 <= s / r <= 1.25 for s, r in zip(sim["noise_sd_by_pay_quintile"],
                                                                                real["noise_sd_by_pay_quintile"])),
        "next_month_corr": abs(sim["next_month_corr"] - real["next_month_corr"]) <= 0.05,
        "mean_flow_by_mode": all(close(s, r, 0.05, 100) for s, r in zip(sim["mean_flow_by_mode"],
                                                                        real["mean_flow_by_mode"])),
    }


def backtest_metrics(flows, close25, obs26):
    """flows P x K x 12 for 2026; close25 P; obs26 P x 12 observed flows."""
    K = flows.shape[1]
    cash = close25[:, None, None] + np.cumsum(flows, 2)
    final = cash[:, :, -1]
    obs_cash = close25[:, None] + np.cumsum(obs26, 1)
    obs_final = obs_cash[:, -1]
    q05, q25, q75, q95 = np.percentile(final, [5, 25, 75, 95], axis=1)
    count = (cash.min(2) < 0).sum(0)
    pit = (final < obs_final[:, None]).mean(1)
    return {
        "final_cash_coverage_50": float(np.mean((obs_final >= q25) & (obs_final <= q75))),
        "final_cash_coverage_90": float(np.mean((obs_final >= q05) & (obs_final <= q95))),
        "final_cash_pit_deciles": np.histogram(pit, bins=np.linspace(0, 1, 11))[0].tolist(),
        "profiles_below_zero_observed": int((obs_cash.min(1) < 0).sum()),
        "profiles_below_zero_expected": float(count.mean()),
        "profiles_below_zero_90pct": np.percentile(count, [5, 95]).tolist(),
        "monthly_flow_crps": float(crps(flows.transpose(0, 2, 1).reshape(-1, K), obs26.ravel()).mean()),
        "final_cash_crps": float(crps(final, obs_final).mean()),
    }


def run(fm_final: FlowModel | None = None, seeds=(0, 1, 2, 3, 4), K=K_PATHS, log=print) -> dict:
    t_start = time.time()
    priors, prof, led = load_priors(), profiles(), ledger()
    pm = profile_months(priors)
    X, y, g = frame_features(pm), (pm.flow_real / SCALE).values, pm.profile_id.values
    first = pm.t.values < 12
    obs = led.pivot(index="profile_id", columns="t", values="net_transactions_chf").reindex(prof.index).values
    close25 = led[led.t == 11].set_index("profile_id").closing_cash_chf.reindex(prof.index).values
    res = {"design": {"paths_per_profile": K, "ensemble_seeds": list(seeds), "priors_drift": priors["drift"]}}

    # T2/T3 (and T1 for everyone): train on 2025, forecast 2026 -----------------------------
    log("backtest: training on 2025 ...")
    fm25 = FlowModel.fit(X[first], y[first], g[first], seeds)
    fm25.fit_offsets(X[first], y[first], g[first])
    nn26 = nn_flows(fm25, prof, 12, 12, K, C.SEED + 7, priors)
    ids, bl26 = baseline_flows(years=(2025,), t_offset=12, T=12, K=K, seed=C.SEED + 7)
    assert (ids == prof.index.values).all()
    res["backtest_network"] = backtest_metrics(nn26, close25, obs[:, 12:])
    res["backtest_baseline"] = backtest_metrics(bl26, close25, obs[:, 12:])

    # T1: people held out of training ------------------------------------------------------
    log("held-out people: 5 folds ...")
    folds = person_folds(prof)
    crps_nn = np.empty((len(prof), 12))
    for f in range(5):
        test = prof.index.values[folds.values == f]
        tm = np.isin(g, test)
        fm_f = FlowModel.fit(X[~tm], y[~tm], g[~tm], seeds=(10 + f,))
        fm_f.fit_offsets(X[tm & first], y[tm & first], g[tm & first])   # only their 2025 months
        fl = nn_flows(fm_f, prof.loc[test], 12, 12, K, C.SEED + 11, priors)
        rows = np.flatnonzero(folds.values == f)
        crps_nn[rows] = crps(fl.transpose(0, 2, 1).reshape(-1, K), obs[rows, 12:].ravel()).reshape(len(test), 12)
    crps_bl = crps(bl26.transpose(0, 2, 1).reshape(-1, K), obs[:, 12:].ravel()).reshape(len(prof), 12)
    res["held_out_people"] = {"monthly_flow_crps_network": float(crps_nn.mean()),
                              "monthly_flow_crps_baseline": float(crps_bl.mean()),
                              "by_status": pd.DataFrame({"status": prof.status.values, "network": crps_nn.mean(1),
                                                         "baseline": crps_bl.mean(1)})
                              .groupby("status").mean().round(1).to_dict(orient="index")}

    # T4: patterns of simulated 2025-2026 ledgers --------------------------------------------
    log("patterns ...")
    if fm_final is None:
        fm_final = FlowModel.fit(X, y, g, seeds)
        fm_final.fit_offsets(X, y, g)
    nn_all = nn_flows(fm_final, prof, 0, 24, 4, C.SEED + 13, priors)
    _, bl_all = baseline_flows(years=(2025, 2026), t_offset=0, T=24, K=4, seed=C.SEED + 13)
    real = pattern_stats(obs, prof)

    def avg(stats):
        return {k: (np.mean([s[k] for s in stats], 0).tolist() if isinstance(stats[0][k], list)
                    else float(np.mean([s[k] for s in stats]))) for k in stats[0]}
    sim_nn = avg([pattern_stats(nn_all[:, k], prof) for k in range(4)])
    sim_bl = avg([pattern_stats(bl_all[:, k], prof) for k in range(4)])
    res["patterns"] = {"data": real, "network": sim_nn, "baseline": sim_bl,
                       "checks_network": pattern_checks(sim_nn, real),
                       "checks_baseline": pattern_checks(sim_bl, real)}

    # Verdicts --------------------------------------------------------------------------------
    def verdict(bt, crps_model, crps_base, checks):
        lo, hi = bt["profiles_below_zero_90pct"]
        return {
            "T1_crps_no_worse_than_baseline": crps_model <= 1.02 * crps_base,
            "T2_coverage_50": abs(bt["final_cash_coverage_50"] - 0.5) <= 0.06,
            "T2_coverage_90": abs(bt["final_cash_coverage_90"] - 0.9) <= 0.04,
            "T3_below_zero_count": (lo - 1 <= bt["profiles_below_zero_observed"] <= hi + 1)
                                   or abs(bt["profiles_below_zero_expected"] - bt["profiles_below_zero_observed"]) <= 4,
            "T4_patterns": all(checks.values()),
        }
    hp = res["held_out_people"]
    res["verdict_network"] = verdict(res["backtest_network"], hp["monthly_flow_crps_network"],
                                     hp["monthly_flow_crps_baseline"], res["patterns"]["checks_network"])
    res["verdict_baseline"] = verdict(res["backtest_baseline"], hp["monthly_flow_crps_baseline"],
                                      hp["monthly_flow_crps_baseline"], res["patterns"]["checks_baseline"])
    res["passed"] = all(res["verdict_network"].values())
    res["seconds"] = round(time.time() - t_start, 1)
    return res


def summary_lines(res: dict) -> list:
    bn, bb = res["backtest_network"], res["backtest_baseline"]
    hp = res["held_out_people"]
    lines = [
        f"{'':34s}{'network':>12s}{'baseline':>12s}{'target':>14s}",
        f"{'T1 CRPS 2026 months, held-out':34s}{hp['monthly_flow_crps_network']:12.1f}{hp['monthly_flow_crps_baseline']:12.1f}{'<= baseline':>14s}",
        f"{'T2 end-2026 cash in 50% band':34s}{bn['final_cash_coverage_50']:12.3f}{bb['final_cash_coverage_50']:12.3f}{'0.50':>14s}",
        f"{'T2 end-2026 cash in 90% band':34s}{bn['final_cash_coverage_90']:12.3f}{bb['final_cash_coverage_90']:12.3f}{'0.90':>14s}",
        f"{'T3 profiles below zero in 2026':34s}{bn['profiles_below_zero_expected']:12.1f}{bb['profiles_below_zero_expected']:12.1f}"
        f"{bn['profiles_below_zero_observed']:>14d}",
        f"{'   end-2026 cash CRPS':34s}{bn['final_cash_crps']:12.0f}{bb['final_cash_crps']:12.0f}",
    ]
    pt = res["patterns"]
    for k in pt["data"]:
        d, n, b = pt["data"][k], pt["network"][k], pt["baseline"][k]
        fmt = (lambda v: " ".join(f"{x:.0f}" for x in v)) if isinstance(d, list) else (lambda v: f"{v:.3f}" if abs(v) < 5 else f"{v:.0f}")
        ok = "ok" if pt["checks_network"][k] else "FAIL"
        lines.append(f"T4 {k:31s} data {fmt(d)} | network {fmt(n)} | baseline {fmt(b)}  [{ok}]")
    lines.append("network verdict: " + ", ".join(f"{k}={'pass' if v else 'FAIL'}" for k, v in res["verdict_network"].items()))
    lines.append("baseline verdict: " + ", ".join(f"{k}={'pass' if v else 'FAIL'}" for k, v in res["verdict_baseline"].items()))
    return lines
