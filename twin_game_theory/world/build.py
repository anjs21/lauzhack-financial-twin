#!/usr/bin/env python3
"""Calibrate the world to Swiss statistics, learn the three market models, validate, write world_model.json.

Run: python -m twin_game_theory.world.build          (about 5-10 minutes on one CPU core)

Steps
  1. Calibrate 5 knobs (job openings, master and PhD preference, PhD positions) so a society of 1,000
     people matches swiss_targets.json. Coordinate search on common random numbers.
  2. Simulate the calibrated world at three society sizes (TRAIN_SIZES: 200, 1,000 and 2,000 people,
     30,000 people per size), 120 months each, so the market models do not learn one market size.
  3. Fit P(job offer), offer level, P(admitted), P(flat) from the pooled logs. 5 bootstrap refits
     (societies resampled within each size) give the parameter uncertainty.
  4. Validate on new societies of every training size, on 5,000-person societies (outside the
     training range) and on a different persona mix.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import expit

from . import spec as sp
from .society import Knobs, World, load_profiles

OUT = Path(__file__).resolve().parent
TRAIN_SIZES = {200: 150, 1000: 30, 2000: 15}            # people per society: number of societies (30,000 people each)
TEST_SIZES = {200: 50, 1000: 10, 2000: 5, 5000: 2}       # new seeds; 5,000 is outside the training range
CAL_SIZE, CAL_WORLDS = 1000, 8
TARGETS = json.loads((OUT / "swiss_targets.json").read_text())["targets"]

# ------------------------------------------------------------------ features of each market model
JOB_X = ["const", "edu_bachelor", "edu_master", "edu_phd", "grades", "u_years", "k_minus_1", "log_apps",
         "log_theta", "inflation", "recession", "student"]
ADM_X = ["const", "phd", "grades", "log_theta"]
FLAT_X = ["const", "to_small", "to_family", "downsizing", "log_income_over_rent", "log_theta"]


def job_design(d):
    e = d["edu"].values
    return np.column_stack([np.ones(len(d)), e == 1, e == 2, e == 3, d.grades, np.minimum(d.u, 24) / 12, d.k - 1,
                            np.log(d.apps), np.log(np.maximum(d.theta, 1e-3)), d.M == 1, d.M == 2, d.student]).astype(float)


def adm_design(d):
    return np.column_stack([np.ones(len(d)), d.prog == sp.PHD, d.grades, np.log(np.maximum(d.theta, 1e-2))]).astype(float)


def flat_design(d):
    return np.column_stack([np.ones(len(d)), d.tier_to == 1, d.tier_to == 2, d.tier_to < d.tier_from,
                            d.log_income_over_rent, np.log(np.maximum(d.theta, 1e-2))]).astype(float)


def prob(X, b, link):
    eta = X @ b
    return -np.expm1(-np.exp(np.clip(eta, -30, 4))) if link == "cloglog" else expit(eta)


def fit_glm(X, y, link, l2=1e-4):
    """Maximum likelihood with exact gradients (L-BFGS). Both log-likelihoods are concave."""
    def f(b):
        eta = X @ b
        if link == "logit":
            p = expit(eta)
            ll = y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))
            g = X.T @ (y - p)
        else:
            mu = np.exp(np.clip(eta, -30, 4))
            p = np.clip(-np.expm1(-mu), 1e-12, 1)
            ll = y * np.log(p) - (1 - y) * mu
            g = X.T @ (y * mu * np.exp(-mu) / p - (1 - y) * mu)
        n = len(y)
        return -ll.sum() / n + l2 * b @ b, -g / n + 2 * l2 * b
    b0 = np.zeros(X.shape[1])
    b0[0] = np.log(-np.log1p(-np.clip(y.mean(), 1e-4, 0.999))) if link == "cloglog" else np.log(y.mean() / (1 - y.mean()))
    return optimize.minimize(f, b0, jac=True, method="L-BFGS-B").x


def calibration_report(X, y, b, link):
    p = prob(X, b, link)
    bins = np.clip((np.searchsorted(np.quantile(p, np.linspace(0, 1, 11)), p, side="right") - 1), 0, 9)
    ece = sum(abs(p[bins == i].mean() - y[bins == i].mean()) * (bins == i).mean() for i in range(10) if (bins == i).any())
    return dict(rows=int(len(y)), mean_pred=round(float(p.mean()), 4), mean_obs=round(float(y.mean()), 4),
                ece_pts=round(100 * float(ece), 2),
                log_loss=round(float(-np.mean(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1)))), 4))


# ------------------------------------------------------------------ calibration
def loss(m):
    def z(value, target, scale):
        return 100.0 if value is None or np.isnan(value) else ((value - target) / scale) ** 2

    def t(key, scale):
        return z(m[key], TARGETS[key]["value"], scale)
    # PhD positions are assumed to be competitive: about 40% of applicants admitted
    return dict(jobs=t("unemployment_rate", 0.01) + t("median_unemployment_months", 1.5),
                master=t("bachelor_to_master_within_24m", 0.03),
                phd=t("master_to_phd_within_24m", 0.03) + z(m.get("phd_admit_rate"), 0.4, 0.1))


def simulate(prof, kn, sizes, seed, mix="reference"):
    """Run societies of several sizes and pool their logs. World ids are made unique; 'size' is recorded."""
    out, offset = {}, 0
    for i, (n_people, n_worlds) in enumerate(sizes.items()):
        tb = World(prof, n_worlds, n_people, kn, mix=mix, seed=seed + 1000 * i).run(120).tables()
        for k, d in tb.items():
            d = d.assign(world=d.world + offset, size=n_people)
            out[k] = d if k not in out else pd.concat([out[k], d], ignore_index=True)
        offset += n_worlds
    return out


def evaluate(prof, kn, n_worlds=CAL_WORLDS, seed=7, n_people=CAL_SIZE):
    w = World(prof, n_worlds, n_people, kn, seed=seed).run(120)
    m = w.moments()
    adm = w.tables()["adm"]
    m["phd_admit_rate"] = float(adm[adm.prog == sp.PHD].admitted.mean()) if (adm.prog == sp.PHD).any() else np.nan
    return m


def calibrate(prof, log):
    kn = Knobs()
    grids = {
        "jobs": [("openings", list(np.geomspace(0.005, 0.04, 12)))],
        "master": [("b_master", [4, 8, 12, 16, 20, 25])],
        "phd": [("phd_seats", [0.004, 0.008, 0.012, 0.018, 0.025]), ("b_phd", [-2.0, -1.0, 0.0, 1.0, 2.0])],
    }
    for sweep in range(2):
        for target, knobs in grids.items():
            names = [k for k, _ in knobs]
            combos = np.array(np.meshgrid(*[v for _, v in knobs])).reshape(len(knobs), -1).T
            best = None
            for c in combos:
                trial = replace(kn, **dict(zip(names, map(float, c))))
                m = evaluate(prof, trial)
                score = loss(m)[target]
                if best is None or score < best[0]:
                    best = (score, trial, m)
            kn = best[1]
            log(f"  sweep {sweep + 1}, {target}: {dict((n, round(getattr(kn, n), 4)) for n in names)} "
                f"loss {best[0]:.2f}")
    return kn, evaluate(prof, kn, n_worlds=2 * CAL_WORLDS, seed=99)


# ------------------------------------------------------------------ learning
def fit_markets(tb, rng):
    job, adm, flat = tb["job"], tb["adm"], tb["flat"]
    flat_e = flat[flat.eligible == 1]
    specs = {"job_offer": (job, job_design, "offer", "cloglog", JOB_X),
             "admission": (adm, adm_design, "admitted", "logit", ADM_X),
             "flat": (flat_e, flat_design, "got", "logit", FLAT_X)}
    models = {}
    for name, (d, design, ycol, link, names) in specs.items():
        X, y = design(d), d[ycol].values.astype(float)
        b = fit_glm(X, y, link)
        rows_of = {w: np.flatnonzero(d.world.values == w) for w in d.world.unique()}
        strata = [d[d["size"] == n].world.unique() for n in sorted(d["size"].unique())]
        boots = []
        for _ in range(5):                       # bootstrap by society, within each size: parameter uncertainty
            pick = np.concatenate([rng.choice(ws, len(ws)) for ws in strata])
            idx = np.concatenate([rows_of[w] for w in pick])
            boots.append(fit_glm(X[idx], y[idx], link))
        models[name] = dict(link=link, features=names, coef=b.round(5).tolist(),
                            coef_draws=np.round(boots, 5).tolist(), rows=int(len(y)))
    # offer level relative to the searcher's target level, by regime (best offer held)
    g = job[job.offer == 1]
    models["offer_level"] = {str(m): [round(float((g[g.M == m].level_vs_target == d).mean()), 4) for d in (-1, 0, 1, 2)]
                             for m in range(3)}
    # market tightness the twin meets: quantiles of theta, by regime (jobs, flats) or programme (study places)
    q = np.linspace(0.025, 0.975, 20)
    models["theta"] = {
        "job": {str(m): np.quantile(job[job.M == m].theta, q).round(5).tolist() for m in range(3)},
        "admission": {str(p): np.quantile(adm[adm.prog == p].theta, q).round(5).tolist() for p in (sp.MASTER, sp.PHD)},
        "flat": {f"{h}_{m}": np.quantile(flat[(flat.tier_to == h) & (flat.M == m)].theta, q).round(5).tolist()
                 if ((flat.tier_to == h) & (flat.M == m)).sum() > 20 else
                 np.quantile(flat[flat.tier_to == h].theta, q).round(5).tolist() for h in range(3) for m in range(3)},
    }
    models["coverage"] = {
        "job_rows_by_edu_regime": {f"{sp.EDU_NAMES[e]}|{['normal', 'inflation', 'recession'][m]}": int(((job.edu == e) & (job.M == m)).sum())
                                   for e in range(4) for m in range(3)},
        "admission_rows_by_programme": {"master": int((adm.prog == sp.MASTER).sum()), "phd": int((adm.prog == sp.PHD).sum())},
        "flat_rows_by_target_tier": {str(h): int((flat_e.tier_to == h).sum()) for h in range(3)},
        "rows_by_society_size": {str(n): {"job": int((job["size"] == n).sum()), "admission": int((adm["size"] == n).sum()),
                                          "flat": int((flat_e["size"] == n).sum())} for n in sorted(job["size"].unique())},
    }
    return models


def validate(models, tests):
    out = {}
    for test, tb in tests.items():
        r = {}
        for name, key, design, ycol, flt in (("job_offer", "job", job_design, "offer", None),
                                             ("admission", "adm", adm_design, "admitted", None),
                                             ("flat", "flat", flat_design, "got", "eligible")):
            d = tb[key]
            if flt:
                d = d[d[flt] == 1]
            if len(d) == 0:
                continue
            m = models[name]
            r[name] = calibration_report(design(d), d[ycol].values.astype(float), np.array(m["coef"]), m["link"])
        out[test] = r
    return out


def direction_checks(models):
    c = {n: dict(zip(models[n]["features"], models[n]["coef"])) for n in ("job_offer", "admission", "flat")}
    j, a, f = c["job_offer"], c["admission"], c["flat"]
    checks = {
        "job: a master's helps more than a bachelor's": j["edu_master"] > j["edu_bachelor"],
        "job: better grades help": j["grades"] > 0,
        "job: a long spell out of work hurts": j["u_years"] < 0,
        "job: more applications help": j["log_apps"] > 0,
        "job: more openings per application help": j["log_theta"] > 0,
        "admission: better grades help": a["grades"] > 0,
        "admission: more places per applicant help": a["log_theta"] > 0,
        "flat: higher income relative to rent helps": f["log_income_over_rent"] > 0,
        "flat: more vacancies per applicant help": f["log_theta"] > 0,
    }
    return {k: bool(v) for k, v in checks.items()}


def main():
    t0 = time.time()
    lines = []

    def log(msg):
        lines.append(msg)
        print(msg, flush=True)

    prof = load_profiles()
    log("1. calibrating to Swiss targets")
    kn, moments = calibrate(prof, log)
    log(f"   calibrated moments ({CAL_SIZE} people): { {k: round(float(v), 4) if isinstance(v, float) else v for k, v in moments.items()} }")
    keys = ("unemployment_rate", "median_unemployment_months", "bachelor_to_master_within_24m", "master_to_phd_within_24m")
    moments_by_size = {}
    for n, w in TRAIN_SIZES.items():
        m = evaluate(prof, kn, n_worlds=max(2, 16000 // n), seed=99, n_people=n)
        moments_by_size[str(n)] = {k: round(float(m[k]), 4) for k in keys}
        log(f"   same knobs, {n} people: {moments_by_size[str(n)]}")
    log(f"2. simulating the calibrated world, 120 months, societies of {list(TRAIN_SIZES)} people "
        f"({sum(n * w for n, w in TRAIN_SIZES.items()):,} people in all)")
    tb = simulate(prof, kn, TRAIN_SIZES, seed=2025)
    log(f"   logged rows: { {k: len(v) for k, v in tb.items()} }")
    log("3. fitting market models")
    models = fit_markets(tb, np.random.default_rng(0))
    log("4. validating on new societies")
    tests = {f"new_societies_{n}_people": simulate(prof, kn, {n: w}, seed=31 + i) for i, (n, w) in enumerate(TEST_SIZES.items())}
    tests["different_persona_mix_1000_people"] = simulate(prof, kn, {1000: 10}, seed=41, mix="shifted")
    val = validate(models, tests)
    checks = direction_checks(models)
    target_check = {k: dict(target=TARGETS[k]["value"], band=TARGETS[k].get("band"),
                            world=round(moments[k], 4) if isinstance(moments.get(k), float) else None,
                            within_band=bool(TARGETS[k].get("band") and TARGETS[k]["band"][0] <= moments[k] <= TARGETS[k]["band"][1]),
                            use=TARGETS[k]["use"], source=TARGETS[k]["source"])
                    for k in TARGETS if k in moments}
    world_model = dict(
        description="Learned and assumed pieces of P(s'|s,a) for the Financial Twin MDP. Load with twin_game_theory.world.transition.TransitionModel.",
        built=time.strftime("%Y-%m-%d %H:%M"),
        knobs=asdict(kn), training_society_sizes=TRAIN_SIZES, calibration_society_size=CAL_SIZE,
        moments_by_society_size=moments_by_size, markets=models,
        swiss_targets=target_check, validation=val, direction_checks=checks,
        sources=sp.SOURCES,
    )
    (OUT / "world_model.json").write_text(json.dumps(world_model, indent=1, default=float) + "\n")
    log(f"   validation: {json.dumps(val, indent=1)}")
    log(f"   direction checks all pass: {all(checks.values())}")
    log(f"done in {time.time() - t0:.0f}s -> {OUT / 'world_model.json'}")
    (OUT / "build_log.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
