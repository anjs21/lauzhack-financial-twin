#!/usr/bin/env python3
"""Toy job market: can a transition function learned from a small simulated society be trusted?

A cohort of graduates searches for jobs. Openings are limited, every application lands on a
random opening, each opening offers the job to its best-scoring applicant, and the person
accepts or rejects the offer. Agents follow two cheap rule personas (relaxed, driven).

From the logs we fit P(offer | own state, effort, market) and ask four questions:
  1. Does it fit held-out worlds of the same kind?
  2. Does a model fitted on 100 people transfer to 1,000 people?        (scale)
  3. Does it transfer when most people behave differently?               (Lucas critique)
  4. How many simulated episodes are needed?                             (cost)

Receiving an offer is the transition; accepting it is the agent's action, so it is not fitted.
All numbers are toy assumptions. CPU only: numpy + scipy. Runs in about a minute.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy import optimize
from scipy.special import expit

OUT = Path(__file__).resolve().parent / "results"
SEED = 20261121
SEARCH, EMP, STUDY = 0, 1, 2

# World (assumed; in the real build these few numbers are calibrated to Swiss statistics)
W = dict(
    T=60,                                  # months per episode
    open_rate={0: 0.10, 1: 0.04},          # openings per month per cohort member: normal, recession
    sep_rate={0: 0.008, 1: 0.02},          # monthly job loss: normal, recession
    p_nr=0.03, p_rn=0.10,                  # regime switching normal->recession, recession->normal
    msc_months=24,
    score=dict(edu=0.8, exp=0.03, udur=-0.06, skill=0.6, noise=0.6),  # employer's ranking
    wage_base={0: 5200.0, 1: 5900.0}, wage_sd=0.15, wage_recession=0.95,
)
# Personas: (relaxed, driven)
PERSONA = dict(p_msc=(0.25, 0.55), effort_logit=(-1.5, 1.0), effort_udur=(0.10, 0.05),
               res0=(1.00, 1.10), res_decay=(0.03, 0.015))
MIXES = {"A": 0.3, "B": 0.8}               # share of driven agents


def episode(N, share_driven, rng, rows):
    k = (rng.random(N) < share_driven).astype(int)
    P = {key: np.asarray(v)[k] for key, v in PERSONA.items()}
    skill = rng.standard_normal(N)
    edu = np.zeros(N, int)
    status = np.where(rng.random(N) < P["p_msc"], STUDY, SEARCH)
    study_left = np.where(status == STUDY, W["msc_months"], 0)
    exp_m, udur = np.zeros(N), np.zeros(N)
    base = np.array([W["wage_base"][0], W["wage_base"][1]])
    M = 0
    for t in range(W["T"]):
        if M == 0 and rng.random() < W["p_nr"]:
            M = 1
        elif M == 1 and rng.random() < W["p_rn"]:
            M = 0
        st = status == STUDY
        grad = st & (study_left == 1)
        study_left[st] -= 1
        status[grad], edu[grad] = SEARCH, 1
        lose = (status == EMP) & (rng.random(N) < W["sep_rate"][M])
        status[lose], udur[lose] = SEARCH, 0
        exp_m[status == EMP] += 1

        s = np.flatnonzero(status == SEARCH)
        if len(s) == 0:
            continue
        p_high = expit(P["effort_logit"][s] + P["effort_udur"][s] * udur[s])
        apps = np.where(rng.random(len(s)) < p_high, 3, 1)
        V, A = rng.poisson(W["open_rate"][M] * N), int(apps.sum())
        got, best = np.zeros(len(s), bool), np.zeros(len(s))
        if V > 0:
            who = np.repeat(np.arange(len(s)), apps)
            where = rng.integers(0, V, size=A)
            c = W["score"]
            sc = (c["edu"] * edu[s] + c["exp"] * np.minimum(exp_m[s], 36)
                  + c["udur"] * np.minimum(udur[s], 24) + c["skill"] * skill[s])[who]
            sc = sc + c["noise"] * rng.standard_normal(A)
            order = np.lexsort((-sc, where))
            _, first = np.unique(where[order], return_index=True)
            win = who[order[first]]
            w = base[edu[s][win]] * np.exp(W["wage_sd"] * rng.standard_normal(len(win)))
            w *= W["wage_recession"] if M else 1.0
            np.maximum.at(best, win, w)
            got[win] = True
        rows.append(np.column_stack([
            edu[s], exp_m[s], udur[s], apps, np.full(len(s), V / max(A, 1)),
            np.full(len(s), M), np.full(len(s), edu[s].mean()), got, np.full(len(s), t)]))
        res = P["res0"][s] * np.exp(-P["res_decay"][s] * udur[s]) * base[edu[s]]
        acc = got & (best >= res)
        status[s[acc]], udur[s[acc]] = EMP, 0
        udur[s[~acc]] += 1


def world(N, mix, episodes, seed):
    rng, rows = np.random.default_rng(seed), []
    for _ in range(episodes):
        episode(N, MIXES[mix], rng, rows)
    return np.vstack(rows)


COLS = ["edu", "exp", "udur", "apps", "theta", "recession", "share_msc", "offer", "t"]
FEATURES = {
    "constant": [],
    "individual_only": ["edu", "exp", "udur", "log_apps", "recession"],
    "+tightness": ["edu", "exp", "udur", "log_apps", "recession", "log_theta"],
    "+tightness+composition": ["edu", "exp", "udur", "log_apps", "recession", "log_theta", "share_msc"],
}


def design(D, feats):
    f = dict(edu=D[:, 0], exp=np.minimum(D[:, 1], 36) / 12, udur=np.minimum(D[:, 2], 24) / 12,
             log_apps=np.log(D[:, 3]), log_theta=np.log(np.maximum(D[:, 4], 1e-3)),
             recession=D[:, 5], share_msc=D[:, 6])
    return np.column_stack([np.ones(len(D))] + [f[k] for k in feats])


def cloglog(eta):
    """P(at least one offer) = 1 - exp(-exp(eta)): the natural link for 'several random draws'."""
    return -np.expm1(-np.exp(np.clip(eta, -30, 4)))


def fit(D, feats):
    X, y = design(D, feats), D[:, 7]

    def nll(b):
        p = np.clip(cloglog(X @ b), 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))
    b0 = np.zeros(X.shape[1])
    b0[0] = np.log(-np.log1p(-y.mean()))
    return optimize.minimize(nll, b0, method="L-BFGS-B").x


def evaluate(D, b, feats):
    X, y = design(D, feats), D[:, 7]
    p = np.clip(cloglog(X @ b), 1e-9, 1 - 1e-9)
    ll = -np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))
    bins = np.quantile(p, np.linspace(0, 1, 11))
    idx = np.clip(np.searchsorted(bins, p, side="right") - 1, 0, 9)
    ece = sum(abs(p[idx == i].mean() - y[idx == i].mean()) * (idx == i).mean() for i in range(10) if (idx == i).any())
    # aggregate path: monthly finding rate, model vs world, over the cohort's months
    t = D[:, 8].astype(int)
    path_err = np.mean([abs(p[t == m].mean() - y[t == m].mean()) for m in np.unique(t)])
    return dict(log_loss=round(float(ll), 4), ece_pts=round(100 * float(ece), 2),
                mean_pred_pts=round(100 * float(p.mean()), 2), mean_obs_pts=round(100 * float(y.mean()), 2),
                monthly_rate_abs_err_pts=round(100 * float(path_err), 2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train = world(100, "A", 300, SEED)
    sim_seconds = time.time() - t0
    agent_months = 300 * 100 * W["T"]
    tests = {
        "same_world_new_seeds (N=100, mix A)": world(100, "A", 300, SEED + 1),
        "scale_10x (N=1000, mix A)": world(1000, "A", 30, SEED + 2),
        "behaviour_change (N=100, mix B)": world(100, "B", 300, SEED + 3),
        "scale+behaviour (N=1000, mix B)": world(1000, "B", 30, SEED + 4),
    }
    report = dict(train_rows=len(train), train_offer_rate_pts=round(100 * train[:, 7].mean(), 2),
                  sim_speed_agent_months_per_s=round(agent_months / sim_seconds), models={})
    for name, feats in FEATURES.items():
        b = fit(train, feats)
        report["models"][name] = dict(
            coefficients=dict(zip(["intercept"] + feats, np.round(b, 3).tolist())),
            **{k: evaluate(D, b, feats) for k, D in tests.items()})
    # Oracle: a model fitted directly on the behaviour-change world, as the floor for that test
    feats = FEATURES["+tightness+composition"]
    D = tests["behaviour_change (N=100, mix B)"]
    report["oracle_behaviour_change_log_loss"] = evaluate(D, fit(world(100, "B", 300, SEED + 5), feats), feats)["log_loss"]
    # Learning curve: how many 100-person episodes are needed?
    curve = []
    for n_ep in (5, 20, 50, 150, 300):
        b = fit(train[: int(len(train) * n_ep / 300)], feats)
        curve.append(dict(episodes=n_ep, coef_log_theta=round(float(b[6]), 3), coef_edu=round(float(b[1]), 3),
                          **evaluate(tests["behaviour_change (N=100, mix B)"], b, feats)))
    report["learning_curve_+tightness+composition_on_behaviour_change"] = curve
    report["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "toy_market.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
