#!/usr/bin/env python3
"""Are income and expenses correlated in the synthetic Financial Twin data?

The ledger stores one net cash flow per profile and month. Income and rent come from the
profiles; everything else (non-rent living costs, one-off bills, student fees, December bonus,
monthly noise) has to be read off the net flow. The script checks:

1. Between people: do higher earners pay more rent, spend more on living costs, or carry more
   tuition or debt? Raw correlations, plus partial correlations that hold household type, city
   and employment status fixed.
2. Within a person over time: do the random parts of income and spending move together
   (monthly noise, December bonus, February/August bills), and does spending react to a bonus?

Living costs at January 2025 = income - rent - typical regular-month net flow, restated with the
calibrated drift exactly as in action_importance.py. A check that assumes no drift rate uses each
profile's own robust (Theil-Sen) trend line, extrapolated to January 2025.

Run:   python analysis/action_importance/income_expense_correlation.py
Needs: numpy, pandas, scipy. Takes about a minute.
Writes results/income_expense_correlation.json and results/income_expense_pairs.csv.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import optimize, stats

import action_importance as ai

N_BOOT = 1000
NON_SE = ["employed", "part_time", "retired", "student", "unemployed"]
BILL_POOL = ["employed", "part_time", "retired", "unemployed"]  # students pay fees in Feb; SE noise hides bills
CONTROLS = ["household", "city", "status"]
BONUS_REL = 0.17        # December surprise above 17% of income counts as a bonus (as in the calibration)
BILL_Z = -2.5           # surprise below -2.5 noise SDs counts as a detected bill
DOF = np.sqrt(16 / 14)  # residuals around a two-parameter line through 16 months understate the noise
HALF1, HALF2 = [1, 4, 6, 10], [3, 5, 7, 11]  # split of the regular months for the reliability check


# ---------------------------------------------------------------------------- helpers
def design(df, cats=(), nums=()):
    parts = [pd.Series(1.0, index=df.index, name="const")]
    parts += [pd.get_dummies(df[c], prefix=c, drop_first=True, dtype=float) for c in cats]
    parts += [df[c].astype(float) for c in nums]
    return pd.concat(parts, axis=1)


def ols(y, X, groups=None):
    """Least squares with HC1 standard errors, or cluster-robust ones when groups are given."""
    Xv, yv = X.to_numpy(float), np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    e = yv - Xv @ beta
    n, k = Xv.shape
    bread = np.linalg.pinv(Xv.T @ Xv)
    if groups is None:
        meat, scale = (Xv * e[:, None] ** 2).T @ Xv, n / (n - k)
    else:
        S = pd.DataFrame(Xv * e[:, None]).groupby(np.asarray(groups)).sum().to_numpy()
        meat, scale = S.T @ S, len(S) / (len(S) - 1) * (n - 1) / (n - k)
    se = np.sqrt(np.diag(bread @ meat @ bread) * scale)
    return pd.Series(beta, X.columns), pd.Series(se, X.columns), float(1 - e.var() / yv.var())


def partial_r(x, y, Z):
    """Correlation of x and y after removing what Z explains in each."""
    def res(v):
        return v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return float(np.corrcoef(res(x), res(y))[0, 1])


def fisher_ci(r, n, k=0):
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3 - k)
    return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]


def ci95(b, se):
    return [float(b - 1.96 * se), float(b + 1.96 * se)]


def spearman(a, b):
    r, p = stats.spearmanr(a, b)
    return dict(rho=float(r), p=float(p))


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d, se = a.mean() - b.mean(), np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return dict(diff_chf=float(d), ci=ci95(d, se), n=[len(a), len(b)])


def tercile(s):
    return pd.qcut(s.rank(method="first"), 3, labels=["low", "middle", "high"])


# ---------------------------------------------------------------------------- data
def living_costs(df, led, cal, months):
    """Non-rent living costs at January 2025: income - rent - median net flow, restated with the drift."""
    g, pi = cal["drift"]["pay_growth_per_year"], cal["drift"]["cost_inflation_per_year"]
    r = led[led.mon.isin(months)]
    base = r.groupby("profile_id").net_transactions_chf.median().reindex(df.index)
    tbar = float(r.groupby("profile_id").t.mean().median())
    fy, fc = (1 + g) ** (tbar / 12), (1 + pi) ** (tbar / 12)
    return (df.Y * fy - df.R * fc - base) / fc


def add_residuals(led):
    """Each profile's robust trend through its regular months; the residual is that month's surprise."""
    reg = led[led.mon.isin(ai.REGULAR)]
    fits = {pid: stats.theilslopes(grp.net_transactions_chf.to_numpy(), grp.t.to_numpy())[:2]
            for pid, grp in reg.groupby("profile_id")}
    trend = pd.DataFrame.from_dict(fits, orient="index", columns=["slope", "b0"])
    led = led.join(trend, on="profile_id")
    led["res"] = led.net_transactions_chf - (led.b0 + led.slope * led.t)
    return led, trend


def prepare():
    prof, led = ai.load()
    cal, _, _ = ai.calibrate(prof, led)
    df = pd.DataFrame(dict(
        Y=prof.reference_monthly_net_income_chf, R=prof.reference_monthly_rent_chf,
        status=prof.employment_status, household=prof.household_type, city=prof.city,
        tuition=prof.tuition, debt=prof.debt0, rate=prof.rate0))
    led = led.rename(columns={"status": "cash_status"}).join(df[["Y", "status"]], on="profile_id")
    led["t"] = (led.yr - 2025) * 12 + led.mon - 1
    led, trend = add_residuals(led)
    trend = trend.reindex(df.index)

    df["O"] = living_costs(df, led, cal, ai.REGULAR)
    df["O_half1"] = living_costs(df, led, cal, HALF1)
    df["O_half2"] = living_costs(df, led, cal, HALF2)
    df["O_trend"] = df.Y - df.R - trend.b0  # no drift assumption: own trend line at January 2025
    df["E"] = df.R + df.O
    return df, led, cal


# ---------------------------------------------------------------------------- 1. between people
def between_people(df):
    nse = df[df.status != "self_employed"]  # their months are too noisy to back out living costs
    pairs = []
    for label, d, col in [("rent", df, "R"), ("living costs (non-rent)", nse, "O"),
                          ("living costs, drift-free check", nse, "O_trend"),
                          ("rent + living costs", nse, "E")]:
        n, r = len(d), float(stats.pearsonr(d.Y, d[col])[0])
        Z = design(d, CONTROLS)
        rp = partial_r(d.Y.to_numpy(float), d[col].to_numpy(float), Z.to_numpy(float))
        b, se, _ = ols(d[col], design(d, CONTROLS, ["Y"]))
        pairs.append(dict(expense=label, n=n, pearson_r=r, pearson_ci=fisher_ci(r, n),
                          spearman_rho=float(stats.spearmanr(d.Y, d[col])[0]),
                          partial_r=rp, partial_ci=fisher_ci(rp, n, Z.shape[1] - 1),
                          chf_per_1000_income=float(1000 * b.Y), chf_per_1000_ci=ci95(1000 * b.Y, 1000 * se.Y)))

    # Living costs are estimated with noise, which shrinks r. Two disjoint halves of the regular months
    # give independent estimates; their covariance is the variance of the true living costs.
    def denoised(d):
        y, o, rr = d.Y.to_numpy(), d.O.to_numpy(), d.R.to_numpy()
        var_true = np.cov(d.O_half1, d.O_half2)[0, 1]
        sy = y.std(ddof=1)
        c_yo, c_yr, c_ro = np.cov(y, o)[0, 1], np.cov(y, rr)[0, 1], np.cov(rr, o)[0, 1]
        return (c_yo / (sy * np.sqrt(var_true)),
                (c_yo + c_yr) / (sy * np.sqrt(rr.var(ddof=1) + 2 * c_ro + var_true)),
                var_true / o.var(ddof=1))
    est = denoised(nse)
    rng = np.random.default_rng(ai.SEED)
    boot = np.array([denoised(nse.iloc[rng.integers(0, len(nse), len(nse))]) for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    denoise = dict(living_costs_r=est[0], living_costs_ci=[lo[0], hi[0]],
                   rent_plus_living_r=est[1], rent_plus_living_ci=[lo[1], hi[1]],
                   reliability_of_living_cost_estimate=est[2], reliability_ci=[lo[2], hi[2]])

    dl = df.assign(logY=np.log(df.Y), logR=np.log(df.R))
    b, se, _ = ols(dl.logR, design(dl, CONTROLS, ["logY"]))
    rent_elasticity = dict(estimate=float(b.logY), ci=ci95(b.logY, se.logY))

    b, se, _ = ols(nse.O, design(nse, CONTROLS, ["Y"]))
    status_fx = {k.replace("status_", ""): dict(chf=float(b[k]), ci=ci95(b[k], se[k]))
                 for k in b.index if k.startswith("status_")}

    sets = {"status": ["status"], "household": ["household"], "city": ["city"], "all three": CONTROLS}
    drivers = {}
    for name, d, y, inc in [("income (log)", dl, dl.logY, None), ("rent (log)", dl, dl.logR, "logY"),
                            ("living costs", nse, nse.O, "Y")]:
        row = {k: ols(y, design(d, v))[2] for k, v in sets.items()}
        if inc:
            row["all three + income"] = ols(y, design(d, CONTROLS, [inc]))[2]
        drivers[name] = row

    q = pd.qcut(nse.Y, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    tab = (nse.assign(q=q, share=(nse.R + nse.O) / nse.Y).groupby("q", observed=True)
           .agg(n=("Y", "size"), income=("Y", "median"), rent=("R", "median"),
                living=("O", "median"), spending_share=("share", "median")))
    quint = [dict(quintile=str(k), **{c: float(v) for c, v in r.items()}) for k, r in tab.iterrows()]

    within = {s: dict(n=len(d), rent=spearman(d.Y, d.R),
                      living=spearman(d.Y, d.O) if s != "self_employed" else None)
              for s, d in df.groupby("status")}
    return dict(pairs=pairs, denoised=denoise, rent_elasticity_to_income=rent_elasticity,
                living_cost_by_status_vs_employed=status_fx, r2_by_driver=drivers,
                by_income_quintile=quint, spearman_within_status=within)


def other_expenses(df, led):
    s001 = df[df.tuition.notna()]
    tuition = dict(n=len(s001), **spearman(s001.Y, s001.tuition),
                   levels_chf=sorted(float(v) for v in s001.tuition.unique()),
                   median_by_status=s001.groupby("status").tuition.median().to_dict())
    stu = df.index[df.status == "student"]
    sep = -led[(led.mon == 9) & led.profile_id.isin(stu)].groupby("profile_id").res.mean()
    fee = dict(n=len(sep), median_chf=float(sep.median()), **spearman(df.Y[sep.index], sep))
    has = df.debt > 0
    bor, emp = df[has], df[df.status == "employed"]
    loans = dict(n_borrowers=int(has.sum()),
                 principal_vs_income=spearman(bor.Y, bor.debt), rate_vs_income=spearman(bor.Y, bor.rate),
                 has_loan_vs_income=spearman(df.Y, has.astype(float)),
                 has_loan_vs_income_employed=spearman(emp.Y, (emp.debt > 0).astype(float)),
                 share_with_loan_by_status=df.assign(h=has).groupby("status").h.mean().to_dict(),
                 note="opening snapshot only; the ledger never services these loans")
    return dict(tuition_s001=tuition, student_fee_september=fee, loans=loans)


# ---------------------------------------------------------------------------- 2. over time
def noise_model(df, led):
    """Monthly noise SD by income quintile within status; fit sd^2 = fixed^2 + (share * income)^2."""
    reg = led[led.mon.isin(ai.REGULAR)].copy()
    q = df.groupby("status").Y.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)) + 1
    reg["q"] = reg.profile_id.map(q)
    fit, fixed2, share2 = {}, {}, {}
    for s, g in reg.groupby("status"):
        b = g.groupby("q")
        sd = b.res.apply(ai.robust_sd) * DOF
        yb = b.Y.mean()
        (a2, b2), _ = optimize.nnls(np.column_stack([np.ones(len(yb)), yb.to_numpy() ** 2]), sd.to_numpy() ** 2)
        per = g.groupby("profile_id").res.apply(ai.robust_sd) * DOF
        fit[s] = dict(fixed_chf=float(np.sqrt(a2)), share_of_income=float(np.sqrt(b2)),
                      per_profile_sd_vs_income=spearman(per, df.Y[per.index]),
                      per_profile_sd_vs_rent=spearman(per, df.R[per.index]),
                      per_profile_sd_vs_living=spearman(per, df.O[per.index]) if s != "self_employed" else None,
                      by_income_quintile=[dict(quintile=int(k), mean_income=float(yb[k]), noise_sd=float(sd[k]))
                                          for k in sd.index])
        fixed2[s], share2[s] = a2, b2
    sigma = np.sqrt(df.status.map(fixed2) + df.status.map(share2) * df.Y ** 2)
    return fit, sigma


def bill_mle(x_reg, s_reg, x_sea, s_sea, start=(-3.0, -0.6, 0.5, 1.2)):
    """Surprise = N(0, s) - bill, bill ~ Bernoulli(p) x Uniform(lo, hi); p differs in Feb/Aug."""
    def dens(x, s, p, lo, hi):
        mix = (stats.norm.cdf((x + hi) / s) - stats.norm.cdf((x + lo) / s)) / (hi - lo)
        return (1 - p) * stats.norm.pdf(x, 0, s) + p * mix

    def unpack(th):
        pr, ps = 1 / (1 + np.exp(-np.asarray(th[:2])))
        lo = 1000 * abs(th[2])
        return pr, ps, lo, lo + 1000 * abs(th[3]) + 50

    def nll(th):
        pr, ps, lo, hi = unpack(th)
        return -(np.log(dens(x_reg, s_reg, pr, lo, hi) + 1e-300).sum()
                 + np.log(dens(x_sea, s_sea, ps, lo, hi) + 1e-300).sum())

    res = optimize.minimize(nll, start, method="Nelder-Mead",
                            options=dict(maxiter=4000, xatol=1e-5, fatol=1e-4))
    pr, ps, lo, hi = unpack(res.x)
    return dict(p_regular_month=float(pr), p_feb_aug=float(ps), size_chf=[float(lo), float(hi)],
                mean_size_chf=float((lo + hi) / 2), _nll=float(res.fun), _x=res.x)


def bills(df, led, sigma):
    """Bill frequency and size by income tercile (mixture fit, descriptive), plus the model-free test:
    how much more a Feb/Aug month costs than a regular month, per person, regressed on income."""
    L = led[led.status.isin(BILL_POOL)].assign(s=lambda d: d.profile_id.map(sigma))

    def fit(g, start=(-3.0, -0.6, 0.5, 1.2)):
        r, s = g[g.mon.isin(ai.REGULAR)], g[g.mon.isin([2, 8])]
        return bill_mle(r.res.to_numpy(), r.s.to_numpy(), s.res.to_numpy(), s.s.to_numpy(), start)
    out = {}
    for name, ids in [("employed_part_time_retired_unemployed", df.index[df.status.isin(BILL_POOL)]),
                      ("employed_only", df.index[df.status == "employed"])]:
        G = L[L.profile_id.isin(ids)]
        extra = (G[G.mon.isin(ai.REGULAR)].groupby("profile_id").res.mean()
                 - G[G.mon.isin([2, 8])].groupby("profile_id").res.mean())
        b, se, _ = ols(extra, design(df.loc[extra.index], ["status"], ["Y"]))
        grp = tercile(df.Y[ids])
        G = G.assign(grp=G.profile_id.map(grp))
        pooled = fit(G)
        parts, moments = {}, {}
        for k, g in G.groupby("grp", observed=True):
            parts[k] = fit(g, pooled["_x"])
            e = extra[g.profile_id.unique()]
            moments[k] = dict(extra_cost_feb_aug_chf=float(e.mean()), ci=ci95(e.mean(), e.std(ddof=1) / np.sqrt(len(e))),
                              income_range_chf=[float(df.Y[e.index].min()), float(df.Y[e.index].max())])
        out[name] = dict(pooled=pooled, by_income_tercile=parts, extra_cost_of_feb_aug_month=moments,
                         extra_cost_chf_per_1000_income=dict(estimate=float(1000 * b.Y), ci=ci95(1000 * b.Y, 1000 * se.Y)))
    return out


def bonus(df, led):
    emp = df.index[df.status == "employed"]
    D = led[(led.mon == 12) & led.profile_id.isin(emp)].copy()
    D["rel"] = D.res / D.Y
    D["bonus"] = D.rel > BONUS_REL
    wide = D.pivot(index="profile_id", columns="yr", values="bonus").reindex(emp)
    b25, b26 = wide[2025].astype(bool), wide[2026].astype(bool)
    grp = tercile(df.Y[emp])
    rate = D.assign(grp=D.profile_id.map(grp)).groupby("grp", observed=True).bonus.mean()
    got = D[D.bonus]
    out = dict(
        share_with_bonus=float(D.bonus.mean()),
        share_by_income_tercile={str(k): float(v) for k, v in rate.items()},
        tercile_chi2_p=float(stats.chi2_contingency(pd.crosstab(D.profile_id.map(grp), D.bonus))[1]),
        size_chf_vs_income=spearman(got.Y, got.res), size_share_vs_income=spearman(got.Y, got.rel),
        median_size_share_of_income=float(got.rel.median()),
        same_people_both_years=dict(phi=float(np.corrcoef(b25, b26)[0, 1]),
                                    p_2026_if_bonus_2025=float(b26[b25].mean()),
                                    p_2026_if_none_2025=float(b26[~b25].mean())))

    # Spending response: did net flow fall from 2025 to 2026 more for those paid a bonus in Dec 2025?
    net = led[led.profile_id.isin(emp)]

    def change(months):
        m = net[net.mon.isin(months)].groupby(["profile_id", "yr"]).net_transactions_chf.mean().unstack()
        return (m[2026] - m[2025]).reindex(emp)
    X = pd.DataFrame(dict(const=1.0, bonus=b25.astype(float), Y=df.Y[emp], R=df.R[emp]), index=emp)
    resp = {}
    for label, months in [("january", [1]), ("regular_months", ai.REGULAR), ("feb_aug", [2, 8])]:
        b, se, _ = ols(change(months), X)
        resp[label] = dict(chf_per_month=float(b.bonus), ci=ci95(b.bonus, se.bonus))
    out["net_flow_change_2025_to_2026_if_bonus_dec_2025"] = resp
    out["mean_bonus_dec_2025_chf"] = float(D[(D.yr == 2025) & D.bonus].res.mean())

    # Bills earlier in the same year: are bonus years worse or better in Feb/Aug?
    fa = (led[led.mon.isin([2, 8]) & led.profile_id.isin(emp)]
          .groupby(["profile_id", "yr"]).res.mean().rename("fa").reset_index()
          .merge(D[["profile_id", "yr", "bonus"]], on=["profile_id", "yr"]))
    X = pd.DataFrame(dict(const=1.0, bonus=fa.bonus.astype(float), Y=fa.profile_id.map(df.Y)))
    b, se, _ = ols(fa.fa, X, groups=fa.profile_id)
    out["feb_aug_surprise_in_bonus_years"] = dict(chf_per_month=float(b.bonus), ci=ci95(b.bonus, se.bonus))
    return out


def persistence(df, led, sigma):
    z = led.assign(z=led.res / led.profile_id.map(sigma)).pivot(index="profile_id", columns="t", values="z")
    res = led.pivot(index="profile_id", columns="t", values="res")
    regt = [t for t in range(24) if t % 12 + 1 in ai.REGULAR]

    def pooled(pairs):
        x = np.concatenate([z[a].to_numpy() for a, _ in pairs])
        y = np.concatenate([z[b].to_numpy() for _, b in pairs])
        return float(np.corrcoef(x, y)[0, 1])
    lag1 = [(a, a + 1) for a in regt if a + 1 in regt]
    placebo = [(a, b) for i, a in enumerate(regt) for b in regt[i + 1:] if b - a >= 2]

    pool = df.index[df.status.isin(BILL_POOL)]
    zp, rp = z.loc[pool], res.loc[pool]
    bill = np.concatenate([(zp[a] < BILL_Z).to_numpy() for a in (1, 7, 13, 19)])  # Feb, Aug of both years
    nxt = np.concatenate([rp[a + 1].to_numpy() for a in (1, 7, 13, 19)])         # Mar, Sep
    feb = np.concatenate([(zp[a] < BILL_Z).to_numpy() for a in (1, 13)])
    aug = np.concatenate([(zp[a] < BILL_Z).to_numpy() for a in (7, 19)])
    return dict(lag1_r=pooled(lag1), placebo_r_non_adjacent=pooled(placebo),
                month_after_bill_minus_after_none=welch(nxt[bill], nxt[~bill]),
                share_feb_aug_with_detected_bill=float(bill.mean()),
                feb_bill_vs_aug_bill_same_year_phi=float(np.corrcoef(feb, aug)[0, 1]))


def persistence_null(df, led, sigma, bill_fit, reps=20):
    """The same statistics on simulated ledgers whose months are independent by construction.
    Fitting each profile's trend line alone makes neighbouring residuals slightly negatively correlated."""
    rng = np.random.default_rng(ai.SEED + 1)
    base = led[["profile_id", "t", "mon", "status"]].copy()
    s = base.profile_id.map(sigma).to_numpy()
    p = np.where(base.mon.isin([2, 8]), bill_fit["p_feb_aug"], bill_fit["p_regular_month"])
    lo, hi = bill_fit["size_chf"]
    sims = []
    for _ in range(reps):
        sim = base.assign(net_transactions_chf=rng.normal(0, s) - (rng.random(len(base)) < p) * rng.uniform(lo, hi, len(base)))
        sims.append(persistence(df, add_residuals(sim)[0], sigma))
    out = {}
    for k in ["lag1_r", "placebo_r_non_adjacent", "feb_bill_vs_aug_bill_same_year_phi"]:
        v = [x[k] for x in sims]
        out[k] = dict(mean=float(np.mean(v)), range95=np.percentile(v, [2.5, 97.5]).tolist())
    v = [x["month_after_bill_minus_after_none"]["diff_chf"] for x in sims]
    out["month_after_bill_minus_after_none"] = dict(mean=float(np.mean(v)), range95=np.percentile(v, [2.5, 97.5]).tolist())
    return out


def drift(df, led):
    nr = led[led.mon.isin(ai.REGULAR) & led.status.isin(NON_SE)].copy()
    nr["tc"] = (nr.t - nr.groupby("profile_id").t.transform("mean")) / 12
    nr["yc"] = nr.net_transactions_chf - nr.groupby("profile_id").net_transactions_chf.transform("mean")
    y, r, o = (nr.profile_id.map(df[c]) for c in ("Y", "R", "O"))
    out = {}
    for label, X in [("common_cost_inflation", pd.DataFrame(dict(pay=nr.tc * y, costs=-nr.tc * (r + o)))),
                     ("separate_rates", pd.DataFrame(dict(pay=nr.tc * y, rent=-nr.tc * r, living=-nr.tc * o)))]:
        b, se, _ = ols(nr.yc, X, groups=nr.profile_id)
        out[label] = {k: dict(per_year=float(b[k]), ci=ci95(b[k], se[k])) for k in X.columns}
    return out


# ---------------------------------------------------------------------------- report
def report(out):
    bp, ox, ot = out["between_people"], out["other_expenses"], out["over_time"]
    f = lambda v: f"{v:+.2f}"  # noqa: E731
    print("\n1. BETWEEN PEOPLE  (partial r and CHF slope hold household type, city and status fixed)")
    print(f"   {'expense':31s}{'n':>6s}   {'Pearson r [95% CI]':22s}{'Spearman':>9s}   {'partial r [95% CI]':22s} CHF per +1,000 income")
    for r in bp["pairs"]:
        print(f"   {r['expense']:31s}{r['n']:6d}   {f(r['pearson_r'])} [{f(r['pearson_ci'][0])}, {f(r['pearson_ci'][1])}]"
              f"{f(r['spearman_rho']):>9s}   {f(r['partial_r'])} [{f(r['partial_ci'][0])}, {f(r['partial_ci'][1])}]"
              f"   {r['chf_per_1000_income']:+6.1f} [{r['chf_per_1000_ci'][0]:+.1f}, {r['chf_per_1000_ci'][1]:+.1f}]")
    d = bp["denoised"]
    print(f"   living costs with estimation noise removed: r = {f(d['living_costs_r'])} "
          f"[{f(d['living_costs_ci'][0])}, {f(d['living_costs_ci'][1])}] (reliability {d['reliability_of_living_cost_estimate']:.2f}"
          f" [{d['reliability_ci'][0]:.2f}, {d['reliability_ci'][1]:.2f}]);"
          f" rent + living: r = {f(d['rent_plus_living_r'])}")
    e = bp["rent_elasticity_to_income"]
    print(f"   rent elasticity to income: {e['estimate']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]")
    print("   living costs vs employed at same income/city/household: "
          + ", ".join(f"{k} {v['chf']:+.0f}" for k, v in bp["living_cost_by_status_vs_employed"].items()))
    print("   share of variance explained (R2):")
    for k, v in bp["r2_by_driver"].items():
        print(f"     {k:14s} " + "  ".join(f"{a}={b:.2f}" for a, b in v.items()))
    print("   by income quintile (excl. self-employed):  median income / rent / living costs / spending share")
    for r in bp["by_income_quintile"]:
        print(f"     {r['quintile']}  n={r['n']:.0f}  {r['income']:7.0f}  {r['rent']:6.0f}  {r['living']:6.0f}  {r['spending_share']:5.2f}")
    print("   Spearman within status (income vs rent / living):")
    for s, v in bp["spearman_within_status"].items():
        liv = f"{v['living']['rho']:+.2f} (p={v['living']['p']:.2f})" if v["living"] else "n/a"
        print(f"     {s:14s} n={v['n']:4d}  rent {v['rent']['rho']:+.2f} (p={v['rent']['p']:.2f})  living {liv}")
    t, fee, lo = ox["tuition_s001"], ox["student_fee_september"], ox["loans"]
    print(f"   tuition (S001, n={t['n']}): rho={t['rho']:+.2f} p={t['p']:.2f}; levels {t['levels_chf']}")
    print(f"   student fee (Sep, n={fee['n']}): median {fee['median_chf']:.0f}, rho={fee['rho']:+.2f} p={fee['p']:.2f}")
    print(f"   loans (n={lo['n_borrowers']}): principal rho={lo['principal_vs_income']['rho']:+.2f} "
          f"p={lo['principal_vs_income']['p']:.2f}; rate rho={lo['rate_vs_income']['rho']:+.2f}; "
          f"has loan rho={lo['has_loan_vs_income']['rho']:+.2f} (employed {lo['has_loan_vs_income_employed']['rho']:+.2f})")

    print("\n2. OVER TIME (within a person)")
    print("   monthly noise SD by income quintile; fit sd^2 = fixed^2 + (share x income)^2")
    for s, v in ot["noise"].items():
        qs = "  ".join(f"{q['mean_income']:.0f}:{q['noise_sd']:.0f}" for q in v["by_income_quintile"])
        print(f"     {s:14s} fixed {v['fixed_chf']:5.0f} CHF + {100 * v['share_of_income']:4.1f}% of income | "
              f"rho(sd, income)={v['per_profile_sd_vs_income']['rho']:+.2f} rho(sd, rent)={v['per_profile_sd_vs_rent']['rho']:+.2f} | {qs}")
    for name, v in ot["bills"].items():
        s = v["extra_cost_chf_per_1000_income"]
        print(f"   bills, {name}: extra cost of a Feb/Aug month per +1,000 income {s['estimate']:+.1f} CHF"
              f" [{s['ci'][0]:+.1f}, {s['ci'][1]:+.1f}]")
        for k, p in v["by_income_tercile"].items():
            m = v["extra_cost_of_feb_aug_month"][k]
            print(f"     {k:7s} income {m['income_range_chf'][0]:6.0f}-{m['income_range_chf'][1]:6.0f}: p(Feb/Aug)={p['p_feb_aug']:.2f}"
                  f" p(other)={p['p_regular_month']:.3f} size {p['size_chf'][0]:.0f}-{p['size_chf'][1]:.0f};"
                  f" extra cost of a Feb/Aug month {m['extra_cost_feb_aug_chf']:.0f} [{m['ci'][0]:.0f}, {m['ci'][1]:.0f}]")
    b = ot["bonus"]
    print(f"   December bonus (employed): {100 * b['share_with_bonus']:.0f}% of years; by income tercile "
          + ", ".join(f"{k} {100 * v:.0f}%" for k, v in b["share_by_income_tercile"].items())
          + f" (chi2 p={b['tercile_chi2_p']:.2f}); size in CHF vs income rho={b['size_chf_vs_income']['rho']:+.2f},"
          f" as share of income rho={b['size_share_vs_income']['rho']:+.2f}; median {b['median_size_share_of_income']:.2f} x monthly pay")
    sp = b["same_people_both_years"]
    print(f"     same people both years? phi={sp['phi']:+.2f} (P 2026 | 2025 bonus {sp['p_2026_if_bonus_2025']:.2f} vs {sp['p_2026_if_none_2025']:.2f})")
    print(f"     net-flow change 2025->2026 for Dec-2025 bonus recipients (mean bonus {b['mean_bonus_dec_2025_chf']:.0f} CHF):")
    for k, v in b["net_flow_change_2025_to_2026_if_bonus_dec_2025"].items():
        print(f"       {k:15s} {v['chf_per_month']:+6.0f} CHF/month [{v['ci'][0]:+.0f}, {v['ci'][1]:+.0f}]")
    v = b["feb_aug_surprise_in_bonus_years"]
    print(f"     Feb/Aug surprise in bonus years vs other years: {v['chf_per_month']:+.0f} CHF [{v['ci'][0]:+.0f}, {v['ci'][1]:+.0f}]")
    p, nu = ot["persistence"], ot["persistence_if_months_independent"]
    print("   surprises, observed vs simulated with independent months (mean [95% range]):")
    for k, lab in [("lag1_r", "next-month correlation"), ("placebo_r_non_adjacent", "non-adjacent months"),
                   ("feb_bill_vs_aug_bill_same_year_phi", "Feb bill vs Aug bill (phi)")]:
        print(f"     {lab:28s} {p[k]:+.3f}   null {nu[k]['mean']:+.3f} [{nu[k]['range95'][0]:+.3f}, {nu[k]['range95'][1]:+.3f}]")
    m, n = p["month_after_bill_minus_after_none"], nu["month_after_bill_minus_after_none"]
    print(f"     {'month after a bill (CHF)':28s} {m['diff_chf']:+.0f} [{m['ci'][0]:+.0f}, {m['ci'][1]:+.0f}]"
          f"   null {n['mean']:+.0f} [{n['range95'][0]:+.0f}, {n['range95'][1]:+.0f}]")
    for k, v in ot["drift"].items():
        print(f"   drift ({k}): " + ", ".join(f"{a} {100 * c['per_year']:+.2f}%/yr [{100 * c['ci'][0]:+.2f}, {100 * c['ci'][1]:+.2f}]"
                                         for a, c in v.items()))


def main():
    df, led, _ = prepare()
    out = dict(between_people=between_people(df), other_expenses=other_expenses(df, led))
    noise, sigma = noise_model(df, led)
    bl = bills(df, led, sigma)
    out["over_time"] = dict(noise=noise, bills=bl, bonus=bonus(df, led),
                            persistence=persistence(df, led, sigma),
                            persistence_if_months_independent=persistence_null(
                                df, led, sigma, bl["employed_part_time_retired_unemployed"]["pooled"]),
                            drift=drift(df, led))
    report(out)
    ai.OUT.mkdir(exist_ok=True)
    (ai.OUT / "income_expense_correlation.json").write_text(json.dumps(ai.jsonable(out), indent=1))
    rows = [{**{k: v for k, v in r.items() if not isinstance(v, list)},
             "pearson_lo": r["pearson_ci"][0], "pearson_hi": r["pearson_ci"][1],
             "partial_lo": r["partial_ci"][0], "partial_hi": r["partial_ci"][1],
             "chf_per_1000_lo": r["chf_per_1000_ci"][0], "chf_per_1000_hi": r["chf_per_1000_ci"][1]}
            for r in out["between_people"]["pairs"]]
    pd.DataFrame(rows).round(4).to_csv(ai.OUT / "income_expense_pairs.csv", index=False)
    print(f"\nwrote {ai.OUT / 'income_expense_correlation.json'} and income_expense_pairs.csv")


if __name__ == "__main__":
    main()
