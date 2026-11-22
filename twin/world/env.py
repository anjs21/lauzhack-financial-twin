"""The world: one month of a person's life, for many paths at once.

    s' ~ T(s, a): apply the action -> draw events and the economy -> draw the month's net
    flow from the learned network -> update the books (exact).

Job offers, master's admission and flat moves go through the learned market model (market.py,
fitted on a simulated Swiss society in twin_game_theory/world) unless Config(markets=False).

Every random component draws from its own seeded stream (one fixed-size draw per stream
and month), so two runs with the same seed see the same luck (common random numbers) and a
stream can be "shared" across paths to switch that source of spread off.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from .. import config as C
from ..data import load_priors
from .flow import SCALE, FlowModel, features
from .market import BACHELOR, DEFAULTS as MARKET_DEFAULTS, EDUCATION, MASTER, NONE, MarketModel, draw_education, pay_level
from .redraw import IncomeModel, RentModel, income_kind, load as load_redraw, shift_rank

STREAMS = {"traits": 1, "model": 2, "policy": 3, "events": 4, "flow": 5, "market": 6, "person": 7}


class Streams:
    """Seeded random draws per (stream, month); a shared stream gives every path the same draw."""

    def __init__(self, seed: int, n: int, shared=()):
        self.seed, self.n, self.shared = seed, n, set(shared)

    def _size(self, name):
        return 1 if name in self.shared else self.n

    def uniform(self, name, t, k=1):
        u = np.random.default_rng([self.seed, STREAMS[name], t]).random((k, self._size(name)))
        return np.broadcast_to(u, (k, self.n))

    def normal(self, name, t, k=1):
        z = np.random.default_rng([self.seed, STREAMS[name], t, 1]).standard_normal((k, self._size(name)))
        return np.broadcast_to(z, (k, self.n))


@dataclass
class State:
    person: np.ndarray        # row of the person in the people table
    age: np.ndarray
    mode: np.ndarray          # 0 student, 1 part-time, 2 full-time, 3 not working
    tau: np.ndarray           # months in the current mode
    rank: np.ndarray          # pay rank in (0, 1): carries earning power across modes
    income: np.ndarray        # net_income, nominal CHF a month (student income and pensions included)
    wage_adjustment: np.ndarray   # contractual difference from mode/rank pay, at opening pay index
    pensioner: np.ndarray
    self_employed: np.ndarray
    search: np.ndarray        # 0 not looking, 1 looking for part-time, 2 for full-time work
    study_left: np.ndarray    # months left in the master's
    delayed: np.ndarray
    degree: np.ndarray        # already did (or is doing) the master's: one per life
    tuition: np.ndarray       # CHF a year if this person studies
    tier: np.ndarray          # 0 shared flat, 1 small flat, 2 family flat
    rent_z: np.ndarray        # the person's place in the rent distribution (kept across moves)
    rent: np.ndarray          # nominal CHF a month
    city: np.ndarray
    household: np.ndarray     # 0 single, 1 couple, 2 family
    child_cost: np.ndarray    # a child born during the simulation (data families are already in the flows)
    lifestyle: np.ndarray     # 0 lean, 1 standard, 2 comfortable
    buffer_months: np.ndarray # the persona's cash buffer target, months of outgoings
    econ: np.ndarray          # 0 normal, 1 inflation, 2 recession
    P: np.ndarray             # price index (1 in January 2025)
    G: np.ndarray             # pay index
    cash: np.ndarray
    inv: np.ndarray
    debt: np.ndarray
    loan_pay: np.ndarray
    loan_rate: np.ndarray
    member: np.ndarray        # ensemble member of this path (model uncertainty)
    offset: np.ndarray        # person offset in real CHF thousands, drawn once per path
    side_job: np.ndarray      # a student working a job next to the course (ends with the course)
    edu: np.ndarray           # highest degree: 0 none, 1 bachelor, 2 master, 3 PhD (market model)
    grades: np.ndarray        # grades in standard deviations from the average (market model)
    flat_search: np.ndarray   # open flat search: -1 smaller, 0 none, +1 bigger (until found or the next review)
    t: int = 0

    def copy(self):
        return State(**{f.name: (getattr(self, f.name).copy() if f.name != "t" else self.t) for f in fields(self)})

    @property
    def n(self):
        return len(self.cash)


@dataclass
class Config:
    events: bool = True        # job loss, offers, promotions, family, economy switches, windfalls
    loans: bool = True
    credit_interest: bool = True
    auto_topup: bool = True    # standing order: cash below zero -> sell investments back to a month of outgoings
    markets: bool = True       # offers, admission, flats from the learned market model; False: priors.yaml rates,
                               # admission certain, moves immediate. Markets only run with events on.


class World:
    def __init__(self, flow: FlowModel, income: IncomeModel, rent: RentModel, priors: dict | None = None,
                 config: Config | None = None):
        self.flow, self.income_model, self.rent_model = flow, income, rent
        self.pr = priors or load_priors()
        self.cfg = config or Config()
        lc = self.pr["living_costs"]
        self.city_living = np.array([lc["city_median"][c] for c in C.CITIES], float)
        self.living_share = lc["share_of_income"]
        self.typical_fees = lc["typical_student_fees_per_year"]
        self.lifestyle_mult = np.array(self.pr["lifestyle"]["multipliers"], float)
        eco = self.pr["economy"]
        self.cost_growth = np.array([eco["cost_growth_per_year"][e] for e in C.ECONOMY])
        self.ret = np.array([eco["returns_per_year"][e] for e in C.ECONOMY])
        j = self.pr["jobs"]
        self.p_loss = np.array([j["loss_per_month"][e] for e in C.ECONOMY])
        self.p_offer = np.array([j["offer_per_month"][e] for e in C.ECONOMY])
        self.p_promo = np.array([j["promotion_per_month"][e] for e in C.ECONOMY])
        self.mk = {**MARKET_DEFAULTS, **self.pr.get("markets", {})}
        self.market = MarketModel.load(C.ROOT / self.mk["model"]) if self.cfg.markets else None

    @classmethod
    def load(cls, path=C.ARTIFACTS, priors=None, config=None):
        income, rent = load_redraw(path / "redraw.json")
        return cls(FlowModel.load(path / "flow"), income, rent, priors, config)

    # ------------------------------------------------------------------------------ setup
    def initial_state(self, people: pd.DataFrame, n_paths: int, streams: Streams, buffer_months=3.0) -> State:
        """Repeat each person n_paths times. `people` has the columns of data.profiles()."""
        rep = lambda col: np.repeat(people[col].values, n_paths)  # noqa: E731
        N = len(people) * n_paths
        mode, pens = rep("mode").astype(int), rep("pensioner").astype(bool)
        kind = income_kind(mode, pens)
        income0 = rep("income0").astype(float)
        rank = self.income_model.rank(kind, income0)
        tier, city = rep("tier").astype(int), rep("city_idx").astype(int)
        u = streams.uniform("model", 0, 3)
        z = streams.normal("model", 0, 1)
        b, var = self.flow.offset_for(people.index.values)
        levels = np.array(self.pr["masters"]["tuition_levels_per_year"], float)
        tuition = rep("tuition").astype(float)
        tuition = np.where(np.isnan(tuition), levels[(u[1] * len(levels)).astype(int).clip(0, len(levels) - 1)], tuition)
        debt, rate = rep("debt0").astype(float), rep("rate0").astype(float)
        term = self.pr["loans"]["term_months"]
        i = rate / 12
        with np.errstate(divide="ignore", invalid="ignore"):
            pay = np.where(i > 0, debt * i / (1 - (1 + i) ** -term), debt / term)
        if not self.cfg.loans:
            debt, pay = np.zeros(N), np.zeros(N)
        # Months left in the current course and whether it is a master's come from the profile.
        # Tables without these columns fall back to a course ending at the opening review.
        opt = lambda col, default: (rep(col) if col in people else np.full(N, default))  # noqa: E731
        study_left = np.where(mode == C.STUDENT, np.maximum(opt("study_left0", 1).astype(int), 1), 0)
        degree = (mode == C.STUDENT) & opt("masters0", False).astype(bool)
        # Market attributes the ledger lacks. A stated education is kept; otherwise students have no
        # degree yet and everyone else draws one per path from the assumed prior (who they are).
        given = people["education"] if "education" in people.columns else pd.Series(None, index=people.index)
        given = np.repeat(given.map(lambda e: EDUCATION[e] if isinstance(e, str) else -1).values, n_paths)
        up = streams.uniform("person", 0, 2)
        drawn = draw_education(pay_level(rank), up[0], up[1], self.mk["education_unknown"])
        edu = np.where(given >= 0, given, np.where(mode == C.STUDENT, NONE, drawn)).astype(int)
        grades = people["grades"] if "grades" in people.columns else pd.Series(np.nan, index=people.index)
        grades = np.repeat(grades.astype(float).fillna(self.mk["grades"]).values, n_paths)
        edu = np.where(degree & (given < 0), BACHELOR, edu)    # already doing a master's: holds a bachelor's
        return State(
            person=np.repeat(np.arange(len(people)), n_paths), age=rep("age").astype(float), mode=mode,
            tau=np.zeros(N, int), wage_adjustment=income0 - self.income_model.quantile(kind, rank), rank=rank, income=income0, pensioner=pens,
            self_employed=rep("self_employed").astype(bool), search=np.zeros(N, int),
            study_left=study_left,
            delayed=np.zeros(N, bool), degree=degree, tuition=tuition, tier=tier,
            rent_z=self.rent_model.z_of(rep("rent0"), tier, city), rent=rep("rent0").astype(float), city=city,
            household=rep("household").astype(int), child_cost=np.zeros(N, bool), lifestyle=np.ones(N, int),
            buffer_months=np.broadcast_to(np.asarray(buffer_months, float), (N,)).copy(),
            econ=np.zeros(N, int), P=np.ones(N), G=np.ones(N), cash=rep("cash0").astype(float),
            inv=rep("inv0").astype(float), debt=debt, loan_pay=np.where(debt > 0, pay, 0.0), loan_rate=rate,
            member=(u[0] * len(self.flow.members)).astype(int).clip(0, len(self.flow.members) - 1),
            offset=np.repeat(b, n_paths) + np.repeat(np.sqrt(var), n_paths) * z[0],
            edu=edu, grades=grades, flat_search=np.zeros(N, int), side_job=np.zeros(N, bool), t=0)

    # --------------------------------------------------------------------------- helpers
    def living_base(self, s: State):
        """Non-rent living costs at standard lifestyle, nominal CHF a month."""
        return (self.city_living[s.city] + self.living_share * s.income / s.P) * s.P

    def outgoings(self, s: State):
        """Regular monthly outgoings: rent, living costs at the chosen lifestyle, child, loan."""
        child = self.pr["household"]["child_cost_per_month"] * s.P * s.child_cost
        return s.rent + self.living_base(s) * self.lifestyle_mult[s.lifestyle] + child + s.loan_pay

    def masks(self, s: State):
        """Valid options per head: status (N, 5), housing/spending/assets (N, 3) for -1, 0, +1."""
        N = s.n
        work = np.isin(s.mode, [C.PART_TIME, C.FULL_TIME])
        young_enough = s.age < self.pr["jobs"]["forced_retirement_age"]
        status = np.ones((N, 5), bool)
        # A master's can start once any current course is in its final month.
        status[:, 1] = ~s.degree & (s.age <= self.pr["masters"]["max_start_age"]) & ~((s.mode == C.STUDENT) & (s.study_left > 1))
        status[:, 2] = (s.mode != C.PART_TIME) & (s.search != 1) & young_enough
        status[:, 3] = (s.mode != C.FULL_TIME) & (s.search != 2) & young_enough
        status[:, 4] = work
        min_tier = np.where(s.household == C.FAMILY, 1, 0)
        housing = np.stack([s.tier > min_tier, np.ones(N, bool), s.tier < 2], 1)
        spending = np.stack([s.lifestyle > 0, np.ones(N, bool), s.lifestyle < 2], 1)
        buffer = s.buffer_months * self.outgoings(s)
        assets = np.stack([(s.inv > 0) & (s.cash < buffer), np.ones(N, bool), s.cash > buffer], 1)
        if not C.annual_review(s.t):          # every voluntary choice waits for the January review
            status[:, 1:] = False
            housing[:, [0, 2]] = False
            spending[:, [0, 2]] = False
            assets[:, [0, 2]] = False
        return status, housing, spending, assets

    def _set_income(self, s: State):
        base = self.income_model.quantile(income_kind(s.mode, s.pensioner), s.rank)
        s.income = np.maximum(0, base + s.wage_adjustment) * s.G

    def set_salary(self, s, salary):
        """Preserve a quoted salary even outside the empirical pay range."""
        kind = income_kind(s.mode, s.pensioner)
        real = np.broadcast_to(salary, s.income.shape) / s.G
        s.rank = self.income_model.rank(kind, real)
        s.wage_adjustment = real - self.income_model.quantile(kind, s.rank)
        self._set_income(s)

    def accept_offer(self, s: State, offer) -> np.ndarray:
        """Apply an accepted offer (work mode, net CHF a month[, alongside studies]); return the rows that took the job.

        Alongside studies, a current student keeps studying and earns the salary until the course ends;
        their pay rank is untouched, so the graduate job that follows is priced as before. Everyone
        else takes the job; a student who does so leaves the course, and an unfinished master's no
        longer counts as done.
        """
        mode, salary = offer[0], float(offer[1])
        if mode not in (C.PART_TIME, C.FULL_TIME) or not np.isfinite(salary) or salary < 0:
            raise ValueError("Specify a paid work mode and a non-negative finite monthly salary.")
        studying = s.mode == C.STUDENT
        side = studying & (len(offer) > 2 and bool(offer[2]))
        job = ~side
        s.degree[job & studying] = False
        s.side_job = side.copy()
        s.mode[job], s.search[job], s.tau[job], s.study_left[job] = mode, 0, 0, 0
        s.pensioner[job], s.self_employed[job] = False, False
        kind = income_kind(s.mode, s.pensioner)
        real = np.full(s.n, salary) / s.G
        s.rank[job] = self.income_model.rank(kind[job], real[job])
        s.wage_adjustment = real - self.income_model.quantile(kind, s.rank)
        self._set_income(s)
        return job

    def _move(self, s: State, rows, new_tier):
        s.tier[rows] = new_tier[rows]
        s.rent[rows] = self.rent_model.rent(s.tier[rows], s.city[rows], s.rent_z[rows]) * s.P[rows]

    # ------------------------------------------------------------------------------ step
    def step(self, s: State, action: np.ndarray, streams: Streams, offer=None):
        """Advance one month. action: (N, 4) = status 0..4, housing/spending/assets as -1/0/+1."""
        s = s.copy()
        t, N, pr = s.t, s.n, self.pr
        info = {k: np.zeros(N, bool) for k in ["job_loss", "offer", "graduated", "partner", "child",
                                              "separation", "forced_move", "windfall", "moved", "status_change",
                                              "admission_refused", "side_job"]}
        markets = self.market is not None and self.cfg.events
        previous_mode = s.mode.copy()
        cash0 = s.cash.copy()
        st_mask, h_mask, sp_mask, a_mask = self.masks(s)
        a = np.asarray(action, int)
        status = np.where(st_mask[np.arange(N), a[:, 0]], a[:, 0], 0)
        housing = np.where(h_mask[np.arange(N), a[:, 1] + 1], a[:, 1], 0)
        spending = np.where(sp_mask[np.arange(N), a[:, 2] + 1], a[:, 2], 0)
        asset = np.where(a_mask[np.arange(N), a[:, 3] + 1], a[:, 3], 0)
        ev = streams.uniform("events", t, 16)
        noise = streams.normal("events", t, 2)
        moving = np.zeros(N)

        # 1. the action ------------------------------------------------------------------
        rank_noise = pr["jobs"]["rank_noise_on_change"]
        m = status == 1                                             # start a master's
        if markets:                                                 # ... if a place is offered
            admitted = ev[10] < self.market.p_admission(s.grades, s.member)
            info["admission_refused"] = m & ~admitted
            m &= admitted
        s.mode[m], s.tau[m], s.study_left[m], s.search[m], s.pensioner[m] = C.STUDENT, 0, pr["masters"]["months"], 0, False
        s.delayed[m], s.degree[m] = False, True
        for opt, target in ((2, C.PART_TIME), (3, C.FULL_TIME)):
            switch = (status == opt) & np.isin(s.mode, [C.PART_TIME, C.FULL_TIME])   # same job, other hours:
            s.mode[switch], s.tau[switch] = target, 0                                 # pay rank unchanged
            look = (status == opt) & np.isin(s.mode, [C.STUDENT, C.NOT_WORKING])     # needs an offer
            s.search[look] = opt - 1
        leave = status == 4
        s.mode[leave], s.tau[leave], s.search[leave] = C.NOT_WORKING, 0, 0
        s.pensioner[leave] = s.age[leave] >= pr["retirement"]["pension_age"]
        if offer is not None:
            if not C.annual_review(t) or np.any(s.age >= pr["jobs"]["forced_retirement_age"]):
                raise ValueError("An offer requires a January review before forced retirement.")
            job = self.accept_offer(s, offer)
            status[:] = np.where(job, 2 if offer[0] == C.PART_TIME else 3, 0)
            info["side_job"] = ~job
        else:
            s.wage_adjustment[s.mode != previous_mode] = 0
        self._set_income(s)
        action_mode, action_pension = s.mode.copy(), s.pensioner.copy()
        info["status_change"] = status > 0

        if markets:
            # A housing choice opens a flat search that lasts until a flat is found or the next
            # review replaces it (0 cancels). Each month the flat market decides, after the
            # landlord's income rule.
            if C.annual_review(t):
                s.flat_search = housing.copy()
            min_tier = np.where(s.household == C.FAMILY, 1, 0)
            new_tier = s.tier + s.flat_search
            s.flat_search[(new_tier < min_tier) | (new_tier > 2)] = 0
            new_tier = np.clip(s.tier + s.flat_search, 0, 2)
            new_rent = self.rent_model.rent(new_tier, s.city, s.rent_z) * s.P
            p_flat = self.market.p_flat(new_tier, s.tier, s.income, new_rent, s.cash, s.mode == C.STUDENT,
                                        s.econ, s.member)
            moved = (s.flat_search != 0) & (ev[11] < p_flat)
            s.flat_search[moved] = 0
        else:
            new_tier = s.tier + housing
            moved = housing != 0
        self._move(s, moved, new_tier)
        moving += moved * pr["housing"]["moving_cost"] * s.P
        info["moved"] = moved
        s.lifestyle = s.lifestyle + spending

        buffer = s.buffer_months * self.outgoings(s)
        x = np.where(asset == 1, np.maximum(s.cash - buffer, 0), 0.0) \
            - np.where(asset == -1, np.minimum(s.inv, np.maximum(buffer - s.cash, 0)), 0.0)
        s.cash, s.inv = s.cash - x, s.inv + x                     # transfers: net worth unchanged

        # 2. events and the economy --------------------------------------------------------
        oneoff = np.zeros(N)
        if self.cfg.events:
            e = pr["economy"]["switch_per_month"]
            u = ev[0]
            new = s.econ.copy()
            nrm = s.econ == 0
            new[nrm & (u < e["normal"]["inflation"])] = 1
            new[nrm & (u >= e["normal"]["inflation"]) & (u < e["normal"]["inflation"] + e["normal"]["recession"])] = 2
            new[(s.econ == 1) & (u < e["inflation"]["normal"])] = 0
            new[(s.econ == 2) & (u < e["recession"]["normal"])] = 0
            s.econ = new

            working = np.isin(s.mode, [C.PART_TIME, C.FULL_TIME])
            lost = working & (ev[1] < self.p_loss[s.econ])
            pension_age = pr["retirement"]["pension_age"]
            s.search[lost] = np.where(s.age[lost] < pension_age, np.where(s.mode[lost] == C.FULL_TIME, 2, 1), 0)
            s.mode[lost], s.tau[lost] = C.NOT_WORKING, 0
            s.pensioner[lost] = s.age[lost] >= pension_age
            info["job_loss"] = lost

            looking = (s.search > 0) & np.isin(s.mode, [C.STUDENT, C.NOT_WORKING]) & ~lost
            if markets:
                months_out = np.where(s.mode == C.NOT_WORKING, s.tau, 0)
                p_offer = self.market.p_offer(s.edu, s.grades, months_out, pay_level(s.rank),
                                              self.mk["applications_per_month"], s.econ, s.mode == C.STUDENT, s.member)
            else:
                p_offer = self.p_offer[s.econ]
            got = looking & (ev[2] < p_offer)
            s.degree[got & (s.mode == C.STUDENT)] = False       # left the course for the job: no master's done
            s.mode[got] = np.where(s.search[got] == 1, C.PART_TIME, C.FULL_TIME)
            s.tau[got], s.search[got], s.study_left[got], s.pensioner[got] = 0, 0, 0, False
            s.rank[got] = shift_rank(s.rank[got], rank_noise * noise[0][got])   # a new job: rank moves a little
            info["offer"] = got

            working = np.isin(s.mode, [C.PART_TIME, C.FULL_TIME])
            promo = working & ~got & (offer is None) & (ev[3] < self.p_promo[s.econ])
            s.rank[promo] = shift_rank(s.rank[promo], pr["jobs"]["promotion_rank_step"])

            hh = pr["household"]
            partner = (s.household == C.SINGLE) & (s.age < hh["partner_max_age"]) & (ev[4] < hh["partner_per_month"])
            child = (s.household == C.COUPLE) & (s.age < hh["child_max_age"]) & (ev[5] < hh["child_per_month"])
            sep = (s.household == C.COUPLE) & ~child & (ev[6] < hh["separation_per_month"])
            s.household[partner], s.household[child], s.household[sep] = C.COUPLE, C.FAMILY, C.SINGLE
            s.child_cost[child] = True
            leave_cost = hh["parental_leave_months"] * (1 - hh["parental_leave_pay"]) * s.income
            oneoff += np.where(child & working, leave_cost, 0.0)
            info["partner"], info["child"], info["separation"] = partner, child, sep

            w = pr["windfall"]
            win = ev[7] < w["per_month"]
            oneoff -= np.where(win, (w["size"][0] + (w["size"][1] - w["size"][0]) * ev[8]) * s.P, 0.0)
            info["windfall"] = win

        # The master's runs its course; pensions start at 65; work stops at 70.
        studying = s.mode == C.STUDENT
        s.study_left[studying] -= 1
        done = studying & (s.study_left <= 0)
        late = done & s.degree & ~s.delayed & (ev[9] < pr["masters"]["delay_probability"])
        s.study_left[late], s.delayed[late] = pr["masters"]["delay_months"], True
        grad = done & ~late
        to_pension = (s.mode == C.NOT_WORKING) & ~s.pensioner & (s.age >= pr["retirement"]["pension_age"])
        s.pensioner[to_pension], s.search[to_pension] = True, 0
        forced = np.isin(s.mode, [C.PART_TIME, C.FULL_TIME]) & (s.age >= pr["jobs"]["forced_retirement_age"])
        s.mode[forced], s.tau[forced], s.pensioner[forced] = C.NOT_WORKING, 0, True

        crowded = (s.household == C.FAMILY) & (s.tier == 0)        # a child in a shared flat forces a move
        self._move(s, crowded, np.ones(N, int))
        s.flat_search[crowded] = 0
        moving += crowded * pr["housing"]["moving_cost"] * s.P
        info["forced_move"] = crowded
        s.wage_adjustment[(s.mode != action_mode) | (s.pensioner != action_pension)] = 0
        self._set_income(s)

        # 3. the month's net flow from the network ---------------------------------------------
        draws = streams.uniform("flow", t, 1)[0]
        z = streams.normal("flow", t, 1)[0]
        moy = (C.START_MONTH - 1 + t) % 12 + 1
        # The data have no working students (student pay tops out near CHF 2,000), so a student with a
        # side job is costed as a part-timer on the same pay and rent, plus their tuition below.
        flow_mode = np.where(s.side_job & (s.mode == C.STUDENT), C.PART_TIME, s.mode)
        X = features(flow_mode, s.self_employed, s.pensioner, s.income / s.P, s.rent / s.P, s.city, np.full(N, moy))
        flow = (self.flow.sample(X, s.member, draws, z) + s.offset) * SCALE * s.P
        lifestyle_adj = -(self.lifestyle_mult[s.lifestyle] - 1) * self.living_base(s)
        child_cost = pr["household"]["child_cost_per_month"] * s.P * s.child_cost
        # Student flows already contain typical fees; a side-job student's part-time flows do not.
        fees_in_flow = np.where(s.side_job, 0.0, self.typical_fees)
        tuition = np.where(s.mode == C.STUDENT, np.maximum(s.tuition - fees_in_flow, 0) / 12 * s.P, 0.0)
        interest = np.minimum(cash0, 0) * pr["credit"]["rate_per_year"] / 12 if self.cfg.credit_interest else 0.0

        # 4. the books (exact) --------------------------------------------------------------------
        loan_interest = s.debt * s.loan_rate / 12
        loan_payment = np.minimum(s.loan_pay, s.debt + loan_interest)
        s.debt = np.maximum(s.debt + loan_interest - loan_payment, 0.0)
        s.loan_pay = np.where(s.debt > 0, s.loan_pay, 0.0)
        net = flow + lifestyle_adj - child_cost - tuition - oneoff - moving - loan_payment + interest
        s.cash = s.cash + net
        mu, sd = self.ret[s.econ, 0] / 12, self.ret[s.econ, 1] / np.sqrt(12)
        r = mu + sd * streams.normal("market", t, 1)[0]
        s.inv = s.inv * (1 + r)
        topup = np.zeros(N)
        if self.cfg.auto_topup:            # not a decision: a standing order that runs any month
            need = np.where(s.cash < 0, self.outgoings(s) - s.cash, 0.0)
            topup = np.minimum(np.maximum(s.inv, 0.0), need)
            s.cash, s.inv = s.cash + topup, s.inv - topup

        # 5. clocks ------------------------------------------------------------------------------
        g = pr["drift"]["pay_growth_per_year"]
        s.P = s.P * (1 + self.cost_growth[s.econ]) ** (1 / 12)
        s.G = s.G * (1 + g) ** (1 / 12)
        s.rent = s.rent * (1 + self.cost_growth[s.econ]) ** (1 / 12)
        s.age = s.age + 1 / 12
        s.mode[grad], s.tau[grad] = C.NOT_WORKING, 0
        s.search[grad & (s.search == 0)] = 2
        earned = grad & s.degree
        s.edu[grad] = np.maximum(s.edu[grad], np.where(s.degree[grad], MASTER, BACHELOR))
        s.rank[earned] = np.clip(s.rank[earned] + pr["masters"]["degree_rank_shift"], 1e-4, 1 - 1e-4)
        info["graduated"] = grad
        s.wage_adjustment[grad] = 0
        s.side_job &= s.mode == C.STUDENT                      # the side job ends with the course
        s.tau = s.tau + 1
        s.t = t + 1
        self._set_income(s)
        limit = np.maximum(pr["credit"]["limit_months_of_pay"] * s.income, pr["credit"]["limit_floor"])
        info.update(flow=flow, net=net, transfer=x, topup=topup, loan_payment=loan_payment, loan_interest=loan_interest,
                    returns=r, action=np.stack([status, housing, spending, asset], 1),
                    below_zero=s.cash < 0, credit_breach=s.cash < -limit)
        return s, info
