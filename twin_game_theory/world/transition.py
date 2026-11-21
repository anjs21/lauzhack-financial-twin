"""P(s' | s, a) for the Financial Twin MDP.

    model = TransitionModel()                     # loads world_model.json next to this file
    model.allowed(s)                              # which actions are allowed in s
    model.outcome_probs(s, a)                     # exact probabilities of every discrete next situation
    s2, ledger = model.sample(s, a, rng)          # one random next state, including cash and investments

One month, in the spec's order: apply the action -> economy and chance events -> market outcomes
(job offers, study places, flats) -> money. sample() and outcome_probs() walk the same list of stages,
so the sampler and the probabilities always agree. Market stages use the models learned from the
simulated world; everything else is a prior (spec) or exact bookkeeping.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.special import expit

from . import spec as sp

DISCRETE = ("W", "prog", "edu", "k", "u", "pend", "offer_k", "H", "F", "L", "M")


@dataclass(frozen=True)
class State:
    t: int = 0                    # months since January 2025
    age: float = 20.0
    W: int = sp.S                 # work status: S, E60, E80, E100, N
    prog: int = sp.BACHELOR       # programme while studying
    edu: int = sp.NONE            # highest degree
    k: int = 1                    # pay level 1..5
    u: int = 0                    # months in the current spell without work
    pend: int = sp.P_NONE         # pending application or offer
    offer_k: int = 0              # pay level of a pending job offer
    H: int = 1                    # housing tier: shared, small, family flat
    F: int = sp.SINGLE            # household
    L: int = 1                    # lifestyle: lean, standard, comfortable
    M: int = sp.NORMAL            # economy
    C: float = 8000.0             # cash, CHF
    I: float = 0.0                # investments, CHF
    leave: int = 0                # months of parental leave left
    pay_index: float = 1.0        # pay growth since January 2025
    cost_index: float = 1.0       # cost growth since January 2025
    breach: bool = False          # credit limit breached (episode ends)
    # fixed context
    city: str = "Lausanne"
    grades: float = 0.0           # ability, standardised
    tuition: float = sp.TUITION_DEFAULT
    apps: int = 2                 # job applications a month when searching (persona habit)
    loan_payment: float = 0.0
    loan_months: int = 0


@dataclass(frozen=True)
class Action:
    status: str = "stay"          # stop, minus, stay, plus, apply_master, apply_phd, accept
    housing: int = 0              # -1, 0, +1 tier
    spending: int = 0             # -1, 0, +1 lifestyle tier
    assets: int = 0               # -1 sell to refill the buffer, 0 hold, +1 invest above it


class TransitionModel:
    def __init__(self, path=None):
        path = path or Path(__file__).resolve().parent / "world_model.json"
        self.wm = json.loads(open(path).read())
        self.mk = self.wm["markets"]

    # ------------------------------------------------------------------ allowed actions
    def allowed(self, s: State) -> dict:
        working = s.W in (sp.E60, sp.E80, sp.E100)
        young = s.age < sp.RETIRE_AGE
        offer = s.pend in (sp.P_JOB_OFFER, sp.P_ADMIT_MASTER, sp.P_ADMIT_PHD)
        free = s.pend == sp.P_NONE
        st = ["stay"]
        if offer:
            st.append("accept")
        if not offer and ((s.W == sp.N and young and free) or s.W == sp.S or s.W in (sp.E60, sp.E80)):
            st.append("plus")
        if s.W in (sp.E80, sp.E100):
            st.append("minus")
        if working and young:
            st.append("stop")
        can_study = free and s.age <= sp.STUDY_MAX_AGE and (s.W == sp.N or working)
        if can_study and s.edu == sp.BACHELOR:
            st.append("apply_master")
        if can_study and s.edu == sp.MASTER:
            st.append("apply_phd")
        housing = [0] + [d for d in (-1, 1) if 0 <= s.H + d <= 2 and not (s.F == sp.FAMILY and s.H + d == 0)]
        spending = [0] + [d for d in (-1, 1) if 0 <= s.L + d <= 2]
        assets = [0, 1, -1] if s.I > 0 else [0, 1]
        return dict(status=st, housing=sorted(housing), spending=sorted(spending), assets=assets)

    # ------------------------------------------------------------------ learned market pieces
    def _coef(self, name, draw):
        m = self.mk[name]
        return np.array(m["coef"] if draw is None else m["coef_draws"][draw % len(m["coef_draws"])]), m["link"]

    def _p(self, name, x_without_theta, thetas, draw, theta_floor):
        """Probability averaged over the market tightness the world produced in this situation."""
        b, link = self._coef(name, draw)
        th = np.log(np.maximum(np.asarray(thetas), theta_floor))
        names = self.mk[name]["features"]
        X = np.tile(x_without_theta, (len(th), 1))
        X[:, names.index("log_theta")] = th
        eta = X @ b
        p = -np.expm1(-np.exp(np.clip(eta, -30, 4))) if link == "cloglog" else expit(eta)
        return float(p.mean())

    def p_job_offer(self, d, draw=None):
        x = np.array([1.0, d["edu"] == 1, d["edu"] == 2, d["edu"] == 3, d["grades"], min(d["u"], 24) / 12,
                      d["k"] - 1, np.log(d["apps"]), 0.0, d["M"] == 1, d["M"] == 2, d["W"] == sp.S], float)
        return self._p("job_offer", x, self.mk["theta"]["job"][str(d["M"])], draw, 1e-3)

    def p_admission(self, d, prog, draw=None):
        x = np.array([1.0, prog == sp.PHD, d["grades"], 0.0])
        return self._p("admission", x, self.mk["theta"]["admission"][str(prog)], draw, 1e-2)

    def p_flat(self, d, target, income, draw=None):
        new_rent = sp.rent(d["city"], target) * d["cost_index"]
        if not (new_rent <= income / 3 or d["C"] >= 12 * new_rent or d["W"] == sp.S):
            return 0.0                                 # landlord's income rule
        x = np.array([1.0, target == 1, target == 2, target < d["H"], np.log(max(income, 1.0) / new_rent), 0.0])
        return self._p("flat", x, self.mk["theta"]["flat"][f"{target}_{d['M']}"], draw, 1e-2)

    # ------------------------------------------------------------------ stages
    def _stages(self, s: State, a: Action, draw):
        """List of stage functions. Each maps a partial next state d to [(prob, updates), ...]."""
        income0 = self.income(asdict(s))

        def action(d):
            up = {"L": int(np.clip(d["L"] + a.spending, 0, 2))}
            if a.status == "accept" and d["pend"] == sp.P_JOB_OFFER:
                up.update(W=sp.E100, k=d["offer_k"], u=0, prog=sp.NONE, pend=sp.P_NONE, offer_k=0)
            elif a.status == "accept" and d["pend"] in (sp.P_ADMIT_MASTER, sp.P_ADMIT_PHD):
                up.update(W=sp.S, prog=sp.MASTER if d["pend"] == sp.P_ADMIT_MASTER else sp.PHD, pend=sp.P_NONE)
            elif a.status == "stay" and d["pend"] in (sp.P_JOB_OFFER, sp.P_ADMIT_MASTER, sp.P_ADMIT_PHD):
                up.update(pend=sp.P_NONE, offer_k=0)                       # decline
            elif a.status == "stop":
                up.update(W=sp.N, u=0)
            elif a.status == "apply_master":
                up.update(pend=sp.P_APPLIED_MASTER)
            elif a.status == "apply_phd":
                up.update(pend=sp.P_APPLIED_PHD)
            return [(1.0, up)]

        def hours(d):
            if a.status == "minus" and d["W"] in (sp.E80, sp.E100):
                nxt = sp.E80 if d["W"] == sp.E100 else sp.E60
                return [(sp.P_FEWER_HOURS_OK, {"W": nxt}), (1 - sp.P_FEWER_HOURS_OK, {})]
            if a.status == "plus" and d["W"] in (sp.E60, sp.E80):
                nxt = sp.E80 if d["W"] == sp.E60 else sp.E100
                return [(sp.P_MORE_HOURS_OK, {"W": nxt}), (1 - sp.P_MORE_HOURS_OK, {})]
            return [(1.0, {})]

        def economy(d):
            moves = sp.ECONOMY[d["M"]]
            return [(p, {"M": m}) for m, p in moves.items()] + [(1 - sum(moves.values()), {})]

        def work_events(d):
            if d["W"] == sp.N or d["W"] == sp.S:
                return [(1.0, {})]
            if s.age >= sp.RETIRE_AGE:
                return [(1.0, {"W": sp.N})]                                  # retirement
            pl, pp = sp.P_JOB_LOSS[d["M"]], sp.P_PROMOTION[d["M"]]
            out = [(pl, {"W": sp.N, "u": 0}), (1 - pl - pp, {})]
            if d["k"] < 5:
                out.append((pp, {"k": d["k"] + 1}))
            else:
                out[1] = (1 - pl, {})
            return out

        def family(d):
            if d["F"] == sp.SINGLE and s.age < sp.PARTNER_MAX_AGE:
                return [(sp.P_PARTNER, {"F": sp.COUPLE}), (1 - sp.P_PARTNER, {})]
            if d["F"] == sp.COUPLE:
                pc = sp.P_CHILD if s.age < sp.CHILD_MAX_AGE else 0.0
                return [(pc, {"F": sp.FAMILY, "leave": sp.LEAVE_MONTHS}), (sp.P_SEPARATION, {"F": sp.SINGLE}),
                        (1 - pc - sp.P_SEPARATION, {})]
            return [(1.0, {})]

        def graduation(d):
            if d["W"] != sp.S or d["prog"] == sp.NONE:
                return [(1.0, {})]
            p = sp.P_STUDY_END[d["prog"]]
            return [(p, {"W": sp.N, "edu": max(d["edu"], d["prog"]), "prog": sp.NONE, "u": 0}), (1 - p, {})]

        def jobs(d):
            searching = a.status == "plus" and s.W in (sp.N, sp.S) and s.age < sp.RETIRE_AGE and d["W"] in (sp.N, sp.S)
            if not searching:
                return [(1.0, {})]
            p = self.p_job_offer(d, draw)
            kt = sp.target_level(d["edu"], d["k"], d["u"])
            levels = defaultdict(float)
            for delta, q in zip((-1, 0, 1, 2), self.mk["offer_level"][str(d["M"])]):
                levels[int(np.clip(kt + delta, 1, 5))] += q
            z = sum(levels.values()) or 1.0
            return [(p * q / z, {"pend": sp.P_JOB_OFFER, "offer_k": lvl}) for lvl, q in levels.items()] + [(1 - p, {})]

        def admissions(d):
            if d["pend"] not in (sp.P_APPLIED_MASTER, sp.P_APPLIED_PHD):
                return [(1.0, {})]
            prog = sp.MASTER if d["pend"] == sp.P_APPLIED_MASTER else sp.PHD
            p = self.p_admission(d, prog, draw)
            admit = sp.P_ADMIT_MASTER if prog == sp.MASTER else sp.P_ADMIT_PHD
            return [(p, {"pend": admit}), (1 - p, {"pend": sp.P_NONE})]

        def flats(d):
            target = d["H"] + a.housing
            if a.housing == 0 or not 0 <= target <= 2:
                return [(1.0, {})]
            p = self.p_flat(d, target, income0, draw)
            return [(p, {"H": target, "_moved": True}), (1 - p, {})]

        return [action, hours, economy, work_events, family, graduation, jobs, admissions, flats]

    # ------------------------------------------------------------------ public API
    def outcome_probs(self, s: State, a: Action, draw=None, top=None) -> dict:
        """Exact distribution of the discrete part of s' (W', prog', edu', k', u', pend', offer_k', H', F', L', M')."""
        branches = [(1.0, asdict(s))]
        for stage in self._stages(s, a, draw):
            nxt = []
            for p, d in branches:
                for q, up in stage(d):
                    if q > 0:
                        nxt.append((p * q, {**d, **up}))
            branches = nxt
        out = defaultdict(float)
        for p, d in branches:
            d["u"] = self._next_u(d, s)
            out[tuple(d[k] for k in DISCRETE)] += p
        items = sorted(out.items(), key=lambda kv: -kv[1])
        return dict(items[:top] if top else items)

    def sample(self, s: State, a: Action, rng: np.random.Generator, draw=None):
        """One draw of s' ~ P(.|s,a). `draw` picks one bootstrap parameter set (model uncertainty)."""
        d = asdict(s)
        for stage in self._stages(s, a, draw):
            opts = stage(d)
            i = rng.choice(len(opts), p=np.array([p for p, _ in opts]) / sum(p for p, _ in opts))
            d.update(opts[i][1])
        return self._money(s, a, d, rng)

    # ------------------------------------------------------------------ money (exact bookkeeping)
    @staticmethod
    def income(d):
        pay = sp.PAY[d["k"] - 1] * d["pay_index"]
        if d["W"] in sp.FRAC:
            inc = pay * sp.FRAC[d["W"]] * (sp.LEAVE_PAY if d.get("leave", 0) > 0 else 1.0)
        elif d["W"] == sp.S:
            inc = (sp.PHD_PAY if d["prog"] == sp.PHD else sp.STUDENT_PAY[sp.BACHELOR]) * d["pay_index"]
        else:
            inc = sp.PENSION_SHARE * pay if d["age"] >= sp.RETIRE_AGE else sp.BENEFIT * d["pay_index"]
        return inc

    @staticmethod
    def _next_u(d, s):
        return min(d["u"] + 1, 36) if d["W"] == sp.N and s.age < sp.RETIRE_AGE else 0

    def _money(self, s, a, d, rng):
        cal = s.t % 12 + 1
        g_cost = sp.COST_GROWTH_INFL if d["M"] == sp.INFLATION else sp.COST_GROWTH
        d["pay_index"] = s.pay_index * (1 + sp.PAY_GROWTH) ** (1 / 12)
        d["cost_index"] = s.cost_index * (1 + g_cost) ** (1 / 12)
        ci = d["cost_index"]
        inc = self.income(d)
        led = {"income": inc}
        led["bonus"] = (inc * rng.uniform(sp.BONUS_LO, sp.BONUS_HI)
                        if cal == 12 and d["W"] in sp.FRAC and rng.random() < sp.BONUS_P else 0.0)
        led["rent"] = -sp.rent(d["city"], d["H"]) * ci
        stu = sp.STUDENT_LIVING_DISCOUNT if d["W"] == sp.S else 0.0
        led["living"] = -(sp.CITY_LIVING[d["city"]] * ci + sp.LIVING_INCOME_SHARE * inc - stu) * sp.LIFESTYLE_MULT[d["L"]]
        led["child"] = -sp.CHILD_COST * ci if d["F"] == sp.FAMILY else 0.0
        led["tuition"] = (-d["tuition"] / 2 * ci if d["W"] == sp.S and d["prog"] in (sp.BACHELOR, sp.MASTER)
                          and cal in (2, 9) else 0.0)
        led["loan"] = -s.loan_payment if s.loan_months > 0 else 0.0
        p_bill = sp.BILL_P_FEB_AUG if cal in (2, 8) else sp.BILL_P_OTHER
        led["bills"] = -rng.uniform(sp.BILL_LO, sp.BILL_HI) * ci if rng.random() < p_bill else 0.0
        led["noise"] = rng.normal(0, sp.NOISE_BASE + sp.NOISE_SHARE * inc)
        led["moving"] = -sp.MOVE_COST * ci if d.pop("_moved", False) else 0.0
        led["windfall"] = rng.uniform(sp.WINDFALL_LO, sp.WINDFALL_HI) if rng.random() < sp.P_WINDFALL else 0.0
        led["credit_interest"] = sp.CREDIT_RATE / 12 * min(s.C, 0.0)
        outgo = -(led["rent"] + led["living"] + led["child"] + led["loan"])
        buffer = sp.BUFFER_MONTHS * outgo
        led["invest"] = -max(0.0, s.C - buffer) if a.assets == 1 else 0.0
        led["sell"] = min(s.I, max(0.0, buffer - s.C)) if a.assets == -1 else 0.0
        mu, sd = sp.RETURNS[d["M"]]
        r = rng.normal(mu / 12, sd / np.sqrt(12))
        led["market_return"] = (s.I - led["sell"] - led["invest"]) * r       # booked on investments, not cash
        cash_flows = sum(v for k, v in led.items() if k != "market_return")
        d["C"] = s.C + cash_flows
        d["I"] = (s.I + -led["invest"] - led["sell"]) * (1 + r)
        d["u"] = self._next_u(d, s)
        d["t"], d["age"] = s.t + 1, s.age + 1 / 12
        d["leave"] = max(0, d["leave"] - 1)
        d["breach"] = s.breach or d["C"] < -sp.CREDIT_LIMIT_MONTHS * max(inc, 1.0)
        d["loan_months"] = max(0, s.loan_months - 1)
        return State(**d), led


def alexia() -> State:
    """Alexia (P00001): 20, finishing a bachelor's in Lausanne, CHF 8,000 cash, rent CHF 900 (shared-flat tier)."""
    return State(t=0, age=20.0, W=sp.S, prog=sp.BACHELOR, edu=sp.NONE, k=1, H=0, F=sp.SINGLE, C=8000.0, city="Lausanne")


if __name__ == "__main__":
    m = TransitionModel()
    s = replace(alexia(), W=sp.N, prog=sp.NONE, edu=sp.BACHELOR, u=1)
    for act in (Action(status="plus"), Action(status="apply_master"), Action(status="plus", housing=1)):
        print(act)
        for key, p in m.outcome_probs(s, act, top=5).items():
            print(f"   {p:.3f}  " + ", ".join(f"{k}={v}" for k, v in zip(DISCRETE, key)))
