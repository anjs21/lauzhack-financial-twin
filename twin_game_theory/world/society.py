"""The simulated world: many 200-person societies side by side, three markets, four personas.

Each month, in order:
  1. economy regime and chance events (job loss, promotion, family, graduation, retirement)
  2. every person decides (persona scores -> softmax; 10% explorers pick any allowed option)
  3. decisions are executed (accept/decline offers, stop, apply, hours changes)
  4. markets clear: jobs (monthly), study places (July), flats (monthly)
  5. money moves and spell counters update
Everything a market decides is logged, so the transition model can be learned from it.
Cash here only drives behaviour (runway); the exact ledger lives in transition.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import spec as sp

# Persona weights: money, safety, leisure, career, inertia, home. apps = job applications a month.
TRAITS = ["money", "safety", "leisure", "career", "inertia", "home"]
PERSONAS = {
    "driven":   dict(w=[1.0, 0.5, 0.3, 1.5, 0.8, 1.0], apps=3),
    "anxious":  dict(w=[1.0, 2.0, 0.5, 0.5, 1.5, 0.8], apps=2),
    "relaxed":  dict(w=[0.6, 0.5, 1.2, 0.3, 2.0, 1.0], apps=1),
    "balanced": dict(w=[1.0, 1.0, 0.8, 0.8, 1.2, 1.0], apps=2),
}
MIX = {"reference": [0.25, 0.25, 0.25, 0.25], "shifted": [0.50, 0.20, 0.10, 0.20]}


@dataclass
class Knobs:
    openings: float = 0.008        # job openings a month per labour-force member (calibrated)
    b_master: float = 1.0          # preference for a master's (calibrated)
    phd_seats: float = 0.010       # PhD positions a year per person (calibrated)
    b_phd: float = -1.0            # preference for a PhD (assumed)
    master_seats: float = 0.25     # master places a year per person: ample, Swiss bachelors have access (assumed)
    flat_turnover: float = 0.093 / 12   # flats re-let a month per flat: Swiss moving rate 9.3%/yr (BFS)
    open_mult: tuple = (1.0, 0.85, 0.5)  # openings by regime normal/inflation/recession (assumed)
    explore: float = 0.10          # share of decisions taken at random, for coverage
    explore_employed: float = 0.03 # lower for workers' reviews, so random quits do not swamp real job loss
    student_share: float = 0.08    # students over-sampled (data: 3%) so the study market has applicants
    score: dict = field(default_factory=lambda: dict(edu=0.5, grades=0.5, u=-0.6, k=0.15, noise=0.8))
    offer_delta: dict = field(default_factory=lambda: {0: (0.15, 0.65, 0.20), 1: (0.15, 0.65, 0.20), 2: (0.30, 0.60, 0.10)})


def sample_choice(sc, explore, rng):
    """Softmax over finite scores; explorers pick uniformly among allowed options."""
    allowed = np.isfinite(sc)
    z = np.where(allowed, sc, -1e9)
    p = np.exp(z - z.max(1, keepdims=True)) * allowed
    p /= p.sum(1, keepdims=True)
    p[explore] = allowed[explore] / allowed[explore].sum(1, keepdims=True)
    return (p.cumsum(1) > rng.random(len(sc))[:, None]).argmax(1)


def load_profiles():
    d = sp.ROOT / "Data"
    prof = pd.read_csv(d / "synthetic-profiles.csv")
    assets = pd.read_csv(d / "synthetic-assets.csv")
    inv = assets[assets.asset_type == "investment_portfolio"].groupby("profile_id").value_chf.sum()
    prof["inv0"] = prof.profile_id.map(inv).fillna(0.0)
    s001 = pd.read_csv(d / "scenario-assumptions.csv")
    tui = s001[s001.parameter == "annual_tuition_chf"].set_index("profile_id").value
    prof["tuition"] = prof.profile_id.map(tui).fillna(sp.TUITION_DEFAULT)
    return prof


def initial_state(prof: pd.DataFrame, rng) -> dict:
    """Map profile rows to MDP states (spec: 'Start state from the data')."""
    status = prof.employment_status.values
    Wmap = {"employed": sp.E100, "self_employed": sp.E100, "part_time": sp.E60, "student": sp.S,
            "unemployed": sp.N, "retired": sp.N}
    W = np.array([Wmap[s] for s in status])
    inc = prof.reference_monthly_net_income_chf.values
    fte = np.where(status == "part_time", inc / 0.6, np.where(status == "retired", inc / sp.PENSION_SHARE, inc))
    k = 1 + np.abs(fte[:, None] - np.array(sp.PAY)[None, :]).argmin(1)
    k = np.where(status == "student", 1, np.where(status == "unemployed", 2, k))
    p_tert = np.array([0.25, 0.40, 0.55, 0.70, 0.80])[k - 1]           # assumed: degree share rises with pay
    tert = rng.random(len(W)) < p_tert
    deg = rng.choice([sp.BACHELOR, sp.MASTER, sp.PHD], size=len(W), p=[0.45, 0.45, 0.10])
    edu = np.where(status == "student", sp.NONE, np.where(tert, deg, sp.NONE))
    hh = prof.household_type.values
    H = np.select([hh == "shared_housing", hh == "single"], [0, 1], 2)
    F = np.select([hh == "couple", hh == "family"], [sp.COUPLE, sp.FAMILY], sp.SINGLE)
    return dict(W=W, k=k, edu=edu, prog=np.where(W == sp.S, sp.BACHELOR, sp.NONE), H=H, F=F,
                u=np.where(status == "unemployed", rng.integers(0, 12, len(W)), 0))


class World:
    def __init__(self, prof, n_worlds, n_players, knobs: Knobs, mix="reference", seed=0):
        self.kn, self.rng = knobs, np.random.default_rng(seed)
        rng = self.rng
        self.nw, self.np_ = n_worlds, n_players
        A = n_worlds * n_players
        stu = (prof.employment_status == "student").values
        p = np.where(stu, knobs.student_share / stu.sum(), (1 - knobs.student_share) / (~stu).sum())
        idx = rng.choice(len(prof), size=A, p=p)
        P = prof.iloc[idx].reset_index(drop=True)
        self.wid = np.repeat(np.arange(n_worlds), n_players)
        st = initial_state(P, rng)
        for key, v in st.items():
            setattr(self, key, v.astype(int))
        self.age0 = P.age_at_start.values.astype(float)
        self.city = P.city.values
        self.rent_city = np.array([sp.CITY_RENT[c] for c in self.city])
        self.living = np.array([sp.CITY_LIVING[c] for c in self.city])
        self.tuition = P.tuition.values
        self.C = P.opening_cash_chf.values.astype(float)
        self.I = P.inv0.values.astype(float)
        self.L = np.ones(A, int)
        self.pend = np.zeros(A, int)
        self.offer_k = np.zeros(A, int)
        self.grades = rng.standard_normal(A)
        self.persona = rng.choice(4, size=A, p=MIX[mix])
        base = np.array([PERSONAS[n]["w"] for n in PERSONAS])
        self.wt = base[self.persona] * np.exp(0.2 * rng.standard_normal((A, len(TRAITS))))   # +-20% spread
        self.apps = np.array([PERSONAS[n]["apps"] for n in PERSONAS])[self.persona]
        self.review = rng.integers(1, 13, A)
        self.searching = self.W == sp.N
        self.searching &= self.age0 < sp.RETIRE_AGE
        self.event = np.zeros(A, bool)
        self.force_up = np.zeros(A, bool)
        self.M = np.zeros(n_worlds, int)
        self.grad_t = np.full((A, 4), -10_000)        # month of finishing bachelor/master/phd
        self.start_t = np.full((A, 4), -10_000)       # month of starting master/phd
        self.logs = {"job": [], "adm": [], "flat": []}
        self.mom = dict(unemp=0.0, lf=0.0, n_hires=0.0, n_search=0.0, tier_moves=0.0, person_months=0.0)
        self.grad_check = []                          # (prog, searching at +12 months)

    # ---------------------------------------------------------------- helpers
    def income(self, age):
        pay = np.array(sp.PAY)[self.k - 1]
        frac = np.select([self.W == sp.E60, self.W == sp.E80, self.W == sp.E100], [0.6, 0.8, 1.0], 0.0)
        inc = pay * frac
        stu = np.where(self.prog == sp.PHD, sp.PHD_PAY, sp.STUDENT_PAY[sp.BACHELOR])
        inc = np.where(self.W == sp.S, stu, inc)
        nw = np.where(age >= sp.RETIRE_AGE, sp.PENSION_SHARE * pay, sp.BENEFIT)
        return np.where(self.W == sp.N, nw, inc)

    def rent(self, H):
        return sp.RENT_BASE * self.rent_city * np.array(sp.TIER_MULT)[H]

    def outgoings(self, inc):
        living = (self.living + sp.LIVING_INCOME_SHARE * inc - np.where(self.W == sp.S, sp.STUDENT_LIVING_DISCOUNT, 0))
        return self.rent(self.H) + living * np.array(sp.LIFESTYLE_MULT)[self.L] + (self.F == sp.FAMILY) * sp.CHILD_COST

    # ---------------------------------------------------------------- one month
    def step(self, t, log=True):
        rng, kn, A = self.rng, self.kn, len(self.W)
        cal = t % 12 + 1
        age = self.age0 + t / 12
        # 1. economy and chance events
        r = rng.random(self.nw)
        M = self.M
        self.M = np.select([(M == 0) & (r < 0.005), (M == 0) & (r < 0.010), (M == 1) & (r < 0.04), (M == 2) & (r < 0.06)],
                           [sp.INFLATION, sp.RECESSION, sp.NORMAL, sp.NORMAL], M)
        Mw = self.M[self.wid]
        self.event[:] = False
        working = np.isin(self.W, [sp.E60, sp.E80, sp.E100])
        retire = (age >= sp.RETIRE_AGE) & (self.W != sp.N)
        self.W[retire], self.pend[retire] = sp.N, sp.P_NONE
        lose = working & ~retire & (rng.random(A) < np.array([sp.P_JOB_LOSS[m] for m in range(3)])[Mw])
        self.W[lose], self.u[lose], self.event[lose] = sp.N, 0, True
        prom = working & ~lose & (rng.random(A) < np.array([sp.P_PROMOTION[m] for m in range(3)])[Mw])
        self.k[prom] = np.minimum(self.k[prom] + 1, 5)
        x = rng.random(A)
        partner = (self.F == sp.SINGLE) & (age < sp.PARTNER_MAX_AGE) & (x < sp.P_PARTNER)
        child = (self.F == sp.COUPLE) & (age < sp.CHILD_MAX_AGE) & (x < sp.P_CHILD)
        sep = (self.F == sp.COUPLE) & ~child & (x > 1 - sp.P_SEPARATION)
        self.F[partner], self.F[child], self.F[sep] = sp.COUPLE, sp.FAMILY, sp.SINGLE
        self.force_up |= child & (self.H == 0)
        self.event |= child
        self.C += (rng.random(A) < sp.P_WINDFALL) * rng.uniform(sp.WINDFALL_LO, sp.WINDFALL_HI, A)
        stu = self.W == sp.S
        end = stu & (rng.random(A) < np.array([0, sp.P_STUDY_END[1], sp.P_STUDY_END[2], sp.P_STUDY_END[3]])[self.prog])
        for pr in (sp.BACHELOR, sp.MASTER, sp.PHD):
            self.grad_t[end & (self.prog == pr), pr] = t
        self.edu[end] = np.maximum(self.edu[end], self.prog[end])
        self.W[end], self.prog[end], self.u[end], self.event[end] = sp.N, sp.NONE, 0, True

        # 2. decisions
        inc = self.income(age)
        out = self.outgoings(inc)
        runway = self.C / np.maximum(out, 1.0)
        status = self.decide_status(t, cal, age, inc, runway)
        house = self.decide_housing(cal, inc, runway)
        # spending and assets: monthly habits (bookkeeping actions)
        wt = self.wt
        cut = (runway < 1) & (self.L > 0) & (rng.random(A) < 0.3 * wt[:, 1] / 2)
        up = (runway > 12) & (self.L < 2) & (rng.random(A) < 0.02 * wt[:, 2])
        self.L = self.L - cut + up
        buf = sp.BUFFER_MONTHS * out
        inv = (self.C > 2 * buf) & (rng.random(A) < 0.1)
        self.I[inv] += self.C[inv] - buf[inv]
        self.C[inv] = buf[inv]
        sell = (self.C < out) & (self.I > 0)
        amt = np.where(sell, np.minimum(self.I, buf - self.C), 0)
        self.C += amt
        self.I -= amt

        # 3. execute status decisions
        acc = status == sp.ST["accept"]
        was_N = self.W == sp.N
        job = acc & (self.pend == sp.P_JOB_OFFER)
        self.W[job], self.k[job], self.u[job], self.prog[job] = sp.E100, self.offer_k[job], 0, sp.NONE
        for pend, pr in ((sp.P_ADMIT_MASTER, sp.MASTER), (sp.P_ADMIT_PHD, sp.PHD)):
            m = acc & (self.pend == pend)
            self.W[m], self.prog[m] = sp.S, pr
            self.start_t[m, pr] = t
        hires_from_N = (job & was_N).sum()
        decline = (status == sp.ST["stay"]) & np.isin(self.pend, [sp.P_JOB_OFFER, sp.P_ADMIT_MASTER, sp.P_ADMIT_PHD])
        self.pend[acc | decline] = sp.P_NONE
        stop = status == sp.ST["stop"]
        self.W[stop], self.u[stop] = sp.N, 0
        self.pend[status == sp.ST["apply_master"]] = sp.P_APPLIED_MASTER
        self.pend[status == sp.ST["apply_phd"]] = sp.P_APPLIED_PHD
        mn = (status == sp.ST["minus"]) & (rng.random(A) < sp.P_FEWER_HOURS_OK)
        pl = (status == sp.ST["plus"]) & np.isin(self.W, [sp.E60, sp.E80]) & (rng.random(A) < sp.P_MORE_HOURS_OK)
        self.W[mn] = np.where(self.W[mn] == sp.E100, sp.E80, sp.E60)
        self.W[pl] = np.where(self.W[pl] == sp.E60, sp.E80, sp.E100)

        # 4. markets
        search = (status == sp.ST["plus"]) & np.isin(self.W, [sp.N, sp.S]) & (age < sp.RETIRE_AGE)
        self.job_market(t, search, runway, Mw, log)
        self.admissions(t, log)
        self.flat_market(t, house, inc, Mw, log)

        # 5. money and counters
        inc = self.income(age)
        out = self.outgoings(inc)
        bills = (rng.random(A) < (sp.BILL_P_FEB_AUG if cal in (2, 8) else sp.BILL_P_OTHER)) * rng.uniform(sp.BILL_LO, sp.BILL_HI, A)
        fees = np.where((self.W == sp.S) & (self.prog != sp.PHD) & (cal in (2, 9)), self.tuition / 2, 0)
        self.C += inc - out - bills - fees + (sp.NOISE_BASE + sp.NOISE_SHARE * inc) * rng.standard_normal(A)
        mu, sd = np.array([sp.RETURNS[m] for m in range(3)]).T
        self.I *= 1 + (mu[self.M] / 12 + sd[self.M] / np.sqrt(12) * rng.standard_normal(self.nw))[self.wid]
        nw_ = (self.W == sp.N) & (age < sp.RETIRE_AGE)
        self.u = np.where(nw_, np.minimum(self.u + 1, 36), 0)
        self.searching = search
        if t >= 12:
            unemp = (self.W == sp.N) & search
            working = np.isin(self.W, [sp.E60, sp.E80, sp.E100])
            self.mom["unemp"] += unemp.sum()
            self.mom["lf"] += working.sum() + unemp.sum()
            self.mom["n_search"] += ((status == sp.ST["plus"]) & (self.W == sp.N)).sum()
            self.mom["n_hires"] += hires_from_N
            self.mom["person_months"] += A
        for pr in (sp.BACHELOR, sp.MASTER):     # graduates one year on: still searching for work?
            m = self.grad_t[:, pr] == t - 12
            if m.any():
                self.grad_check.append(np.column_stack([np.full(m.sum(), pr), self.W[m] == sp.S,
                                                        (self.W[m] == sp.N) & self.searching[m],
                                                        np.isin(self.W[m], [sp.E60, sp.E80, sp.E100])]))

    # ---------------------------------------------------------------- decisions
    def decide_status(self, t, cal, age, inc, runway):
        A, kn, rng = len(self.W), self.kn, self.rng
        w = self.wt
        pay = np.array(sp.PAY)
        free_now = np.array([sp.FREE_TIME[s] for s in range(5)])[self.W]
        kt = np.maximum(np.array([sp.ENTRY_LEVEL[e] for e in range(4)])[self.edu], self.k - (self.u >= 12))
        low = (runway < 3).astype(float)

        def U(money, leisure=0.0, career=0.0, change=1.0):
            return w[:, 0] * money + w[:, 1] * money * low + w[:, 2] * leisure + w[:, 3] * career - w[:, 4] * change

        sc = np.full((A, len(sp.STATUS)), -np.inf)
        stay = np.zeros(A)
        young = age < sp.RETIRE_AGE
        window = cal in sp.APPLY_MONTHS
        free_pend = self.pend == sp.P_NONE
        isN, isS = (self.W == sp.N) & young, self.W == sp.S
        isE = np.isin(self.W, [sp.E60, sp.E80, sp.E100])
        review = (self.review == cal) | self.event
        # who decides this month
        offer = np.isin(self.pend, [sp.P_JOB_OFFER, sp.P_ADMIT_MASTER, sp.P_ADMIT_PHD])
        decides = offer | isN | ((isE | isS) & review)
        sc[decides, sp.ST["stay"]] = np.where(isN & self.searching, -w[:, 4], 0.0)[decides]   # stopping a search is a change
        # search / more hours
        m = decides & (isN | isS) & ~offer & free_pend          # a pending study application blocks searching
        gain = (pay[kt - 1] - inc) / 1000
        s_plus = U(0.5 * gain, 0.5 * (0 - free_now) / 1000, 0.2 + 0.25 * np.minimum(self.u, 24) / 12,
                   change=(~self.searching).astype(float))
        s_plus = s_plus - 3.0 * isS                               # assumed: dropping out forfeits the degree
        sc[m, sp.ST["plus"]] = s_plus[m]
        frac_next = {sp.E60: sp.E80, sp.E80: sp.E100}
        for cur, nxt in frac_next.items():
            mm = decides & (self.W == cur) & ~offer
            sc[mm, sp.ST["plus"]] = U(0.2 * pay[self.k - 1] / 1000, (sp.FREE_TIME[nxt] - sp.FREE_TIME[cur]) / 1000)[mm]
        for cur, nxt in {sp.E100: sp.E80, sp.E80: sp.E60}.items():
            mm = decides & (self.W == cur) & ~offer
            sc[mm, sp.ST["minus"]] = U(-0.2 * pay[self.k - 1] / 1000, (sp.FREE_TIME[nxt] - sp.FREE_TIME[cur]) / 1000)[mm]
        mm = decides & isE & ~offer & young
        early = 1 + np.maximum(0, age - 55) / 5
        sc[mm, sp.ST["stop"]] = U((sp.BENEFIT - inc) / 1000, (2000 - free_now) / 1000 * early, -0.5)[mm]
        # applications to study
        # apply when a spell without work starts (e.g. just graduated) or at the yearly review
        stud_ok = window & (age <= sp.STUDY_MAX_AGE) & free_pend & ~offer & (isN | isE) & (review | (isN & (self.u <= 1)))
        lvl_m = np.maximum(sp.ENTRY_LEVEL[sp.MASTER], kt)
        lvl_p = np.maximum(sp.ENTRY_LEVEL[sp.PHD], kt)
        am = decides & stud_ok & (self.edu == sp.BACHELOR)
        s_am = U(0.5 * (sp.STUDENT_PAY[sp.MASTER] - inc - self.tuition / 12) / 1000, (0 - free_now) / 1000,
                 (pay[lvl_m - 1] - pay[kt - 1]) / 1000 + kn.b_master)
        sc[am, sp.ST["apply_master"]] = s_am[am]
        ap = decides & stud_ok & (self.edu == sp.MASTER)
        s_ap = U(0.5 * (sp.PHD_PAY - inc) / 1000, (0 - free_now) / 1000, (pay[lvl_p - 1] - pay[kt - 1]) / 1000 + kn.b_phd)
        sc[ap, sp.ST["apply_phd"]] = s_ap[ap]
        # accepting offers
        jo = self.pend == sp.P_JOB_OFFER
        ok = np.maximum(self.offer_k, 1)
        s_acc = U((pay[ok - 1] - inc) / 1000, (0 - free_now) / 1000,
                  (pay[ok - 1] - pay[kt - 1]) / 1000 + 0.25 * np.minimum(self.u, 24) / 12, change=0.0)
        sc[jo, sp.ST["accept"]] = s_acc[jo]
        for pend, s_x in ((sp.P_ADMIT_MASTER, s_am), (sp.P_ADMIT_PHD, s_ap)):
            mm = self.pend == pend
            sc[mm, sp.ST["accept"]] = (s_x + w[:, 4])[mm]      # they applied: no extra change cost
        # sample: softmax, or uniform among allowed options for explorers
        sc[~np.isfinite(sc).any(1), sp.ST["stay"]] = 0.0
        choice = sample_choice(sc, decides & (rng.random(A) < np.where(isE & ~offer, kn.explore_employed, kn.explore)), rng)
        return np.where(decides, choice, sp.ST["stay"])

    def decide_housing(self, cal, inc, runway):
        A, rng, w = len(self.W), self.rng, self.wt
        review = (self.review == cal) | self.force_up
        rents = np.stack([self.rent(np.full(A, h)) for h in range(3)], 1)
        hv = np.array([[np.nan if v is None else v for v in row] for row in sp.HOME_VALUE])
        cur = np.nan_to_num(hv[self.F, self.H], nan=-3000.0)        # a family in a shared flat is badly off
        sc = np.full((A, 3), -np.inf)          # options: down, stay, up
        sc[:, 1] = 0.0
        low = (runway < 2).astype(float)
        for j, d in ((0, -1), (2, 1)):
            h2 = self.H + d
            ok = (h2 >= 0) & (h2 <= 2)
            h2c = np.clip(h2, 0, 2)
            home = (hv[self.F, h2c] - cur) / 1000
            ok &= np.isfinite(home)
            save = -(rents[np.arange(A), h2c] - rents[np.arange(A), self.H]) / 1000
            afford = np.where((rents[np.arange(A), h2c] > 0.45 * inc) & (self.W != sp.S) & (d > 0), -3.0, 0.0)
            s = w[:, 5] * np.nan_to_num(home) + w[:, 0] * save + w[:, 1] * save * low + afford - w[:, 4] - 0.17
            sc[ok, j] = s[ok]
        sc[self.force_up, 0] = -np.inf
        sc[self.force_up & np.isfinite(sc[:, 2]), 1] = -np.inf
        choice = sample_choice(sc, review & (rng.random(A) < self.kn.explore), rng) - 1
        return np.where(review, choice, 0)

    # ---------------------------------------------------------------- markets
    def job_market(self, t, search, runway, Mw, log):
        rng, kn = self.rng, self.kn
        s = np.flatnonzero(search)
        working = np.isin(self.W, [sp.E60, sp.E80, sp.E100])
        lf = np.bincount(self.wid, weights=working | search, minlength=self.nw)
        V = rng.poisson(kn.openings * lf * np.array(kn.open_mult)[self.M])
        if len(s) == 0:
            return
        apps = self.apps[s] + (runway[s] < 2)
        A_w = np.bincount(self.wid[s], weights=apps, minlength=self.nw)
        theta = V / np.maximum(A_w, 1)
        who = np.repeat(np.arange(len(s)), apps)
        w_of = self.wid[s][who]
        ok = V[w_of] > 0
        who, w_of = who[ok], w_of[ok]
        got, best, n_off = np.zeros(len(s), bool), np.zeros(len(s), int), np.zeros(len(s), int)
        kt = np.maximum(np.array([sp.ENTRY_LEVEL[e] for e in range(4)])[self.edu[s]], self.k[s] - (self.u[s] >= 12))
        if len(who):
            slot = np.concatenate([[0], np.cumsum(V)])[w_of] + (rng.random(len(who)) * V[w_of]).astype(int)
            c = kn.score
            g = s[who]
            score = (c["edu"] * self.edu[g] + c["grades"] * self.grades[g] + c["u"] * np.minimum(self.u[g], 24) / 12
                     + c["k"] * (self.k[g] - 1) + c["noise"] * rng.standard_normal(len(who)))
            order = np.lexsort((-score, slot))
            _, first = np.unique(slot[order], return_index=True)
            win = who[order[first]]
            probs = np.array([kn.offer_delta[m] for m in range(3)])[self.M[self.wid[s[win]]]]
            delta = (probs.cumsum(1) > rng.random(len(win))[:, None]).argmax(1) - 1
            lvl = np.clip(kt[win] + delta, 1, 5)
            np.maximum.at(best, win, lvl)
            np.add.at(n_off, win, 1)
            got[win] = True
        self.pend[s[got]], self.offer_k[s[got]] = sp.P_JOB_OFFER, best[got]
        if log and t >= 12:
            self.logs["job"].append(np.column_stack([
                self.wid[s], np.full(len(s), t), Mw[s], self.edu[s], self.grades[s], self.u[s], self.k[s],
                apps, theta[self.wid[s]], got, np.where(got, best - kt, 0), n_off, self.W[s] == sp.S]))

    def admissions(self, t, log):
        rng, kn = self.rng, self.kn
        for pend, pr, c, admit in ((sp.P_APPLIED_MASTER, sp.MASTER, kn.master_seats, sp.P_ADMIT_MASTER),
                                   (sp.P_APPLIED_PHD, sp.PHD, kn.phd_seats, sp.P_ADMIT_PHD)):
            a = np.flatnonzero(self.pend == pend)
            seats = rng.poisson(c * self.np_ / 12, self.nw)      # c = places a year per person; decided monthly
            if len(a) == 0:
                continue
            n_app = np.bincount(self.wid[a], minlength=self.nw)
            score = self.grades[a] + 0.5 * rng.standard_normal(len(a))
            order = np.lexsort((-score, self.wid[a]))
            wa = self.wid[a][order]
            start = np.concatenate([[0], np.cumsum(n_app)])[wa]
            rank = np.arange(len(a)) - start
            yes = np.zeros(len(a), bool)
            yes[order] = rank < seats[wa]
            self.pend[a] = np.where(yes, admit, sp.P_NONE)
            if log and t >= 12:
                theta = seats / np.maximum(n_app, 1)
                self.logs["adm"].append(np.column_stack([
                    self.wid[a], np.full(len(a), t), np.full(len(a), pr), self.grades[a], self.edu[a],
                    theta[self.wid[a]], yes]))

    def flat_market(self, t, house, inc, Mw, log):
        rng, kn = self.rng, self.kn
        a = np.flatnonzero(house != 0)
        if len(a) == 0:
            return
        target = self.H[a] + house[a]
        new_rent = sp.RENT_BASE * self.rent_city[a] * np.array(sp.TIER_MULT)[target]
        ratio = np.log(np.maximum(inc[a], 1) / new_rent)
        eligible = (new_rent <= inc[a] / 3) | (self.C[a] >= 12 * new_rent) | (self.W[a] == sp.S)
        grp = self.wid[a] * 3 + target
        stock = np.bincount(self.wid * 3 + self.H, minlength=self.nw * 3)
        V = rng.poisson(kn.flat_turnover * np.maximum(stock, 1))
        score = np.where(eligible, ratio + 0.5 * rng.standard_normal(len(a)), -np.inf)
        n_app = np.bincount(grp, minlength=self.nw * 3)
        order = np.lexsort((-score, grp))
        g = grp[order]
        start = np.concatenate([[0], np.cumsum(n_app)])[g]
        rank = np.arange(len(a)) - start
        yes = np.zeros(len(a), bool)
        yes[order] = (rank < V[g]) & np.isfinite(score[order])
        mv = a[yes]
        self.H[mv] = target[yes]
        self.C[mv] -= sp.MOVE_COST
        self.force_up[mv] = False
        self.mom["tier_moves"] += len(mv) if t >= 12 else 0
        if log and t >= 12:
            theta = V / np.maximum(n_app, 1)
            self.logs["flat"].append(np.column_stack([
                self.wid[a], np.full(len(a), t), Mw[a], self.H[a] - np.where(yes, house[a], 0), target, ratio,
                eligible, theta[grp], yes]))

    # ---------------------------------------------------------------- run
    def run(self, T=120, log=True):
        for t in range(T):
            self.step(t, log)
        return self

    def tables(self):
        cols = {"job": ["world", "t", "M", "edu", "grades", "u", "k", "apps", "theta", "offer", "level_vs_target",
                        "n_offers", "student"],
                "adm": ["world", "t", "prog", "grades", "edu", "theta", "admitted"],
                "flat": ["world", "t", "M", "tier_from", "tier_to", "log_income_over_rent", "eligible", "theta", "got"]}
        return {k: pd.DataFrame(np.vstack(v) if v else np.empty((0, len(cols[k]))), columns=cols[k])
                for k, v in self.logs.items()}

    def moments(self, T=120):
        m = self.mom
        h = m["n_hires"] / max(m["n_search"], 1)
        out = dict(unemployment_rate=m["unemp"] / max(m["lf"], 1),
                   monthly_job_finding=h,
                   median_unemployment_months=float(np.log(2) / -np.log(1 - h)) if 0 < h < 1 else np.nan,
                   tier_moves_per_year=12 * m["tier_moves"] / max(m["person_months"], 1))
        for pr, nxt, key in ((sp.BACHELOR, sp.MASTER, "bachelor_to_master_within_24m"),
                             (sp.MASTER, sp.PHD, "master_to_phd_within_24m")):
            g = self.grad_t[:, pr]
            elig = (g >= 0) & (g <= T - 25) & (self.age0 + g / 12 <= 35)     # typical graduates, as in the statistic
            s = self.start_t[:, nxt]
            out[key] = float(((s > g) & (s <= g + 24))[elig].mean()) if elig.any() else np.nan
            out[f"n_{key}"] = int(elig.sum())
        if self.grad_check:
            gc = np.vstack(self.grad_check)
            for pr, key in ((sp.BACHELOR, "graduate_unemployment_1y_bachelor"), (sp.MASTER, "graduate_unemployment_1y_master")):
                r = gc[(gc[:, 0] == pr) & (gc[:, 1] == 0)]
                lf = r[:, 2].sum() + r[:, 3].sum()
                out[key] = float(r[:, 2].sum() / lf) if lf else np.nan     # ILO-style: searching / (working + searching)
        return out
