"""Rule persona (MVP policy): choose like this person, at random, with memory of the path.

For every head the policy builds logits from trait-weighted rules plus inertia, masks
invalid options, divides by the person's temperature and samples. Nothing is maximised.

January is the annual review for every voluntary decision: work, housing, lifestyle and
investing. External events (job loss, bills, markets, family) still happen monthly.
"""
from __future__ import annotations

import numpy as np

from .. import config as C
from ..world.env import State, Streams, World
from .traits import EVENTS, TRAIT, Intention, buffer_months

HEADS = ["status", "housing", "spending", "assets"]
N_OPTS = {"status": 5, "housing": 3, "spending": 3, "assets": 3}
NEG = -1e9
# Lifestyle odds were set per month; decided once a year they get log(12) so the yearly rate
# stays the same. Investing or selling to the buffer becomes a likely annual rebalance.
YEARLY = float(np.log(12))
ANNUAL_REBALANCE = 1.5


def softmax_masked(logits, mask, temperature):
    z = np.where(mask, logits / temperature[:, None], NEG)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z) * mask
    return p / p.sum(1, keepdims=True)


def sample(p, u):
    return (u[:, None] > np.cumsum(p, 1)[:, :-1]).sum(1)


class RulePolicy:
    name = "rule persona"

    def __init__(self, world: World, traits: np.ndarray, temperature, goals: dict, intentions=(), people_index=None):
        """traits (N, 8); goals: per-path arrays high_target, high_cash_only, med_target (nan if none)."""
        self.w, self.tr = world, traits
        self.temp = np.broadcast_to(np.asarray(temperature, float), (len(traits),)).copy()
        self.goals = goals
        self.intentions = [i if isinstance(i, Intention) else Intention.parse(i) for i in intentions]
        ids = [it.id for it in self.intentions]
        self._after = []
        for it in self.intentions:
            if it.after is not None and it.after not in ids:
                raise ValueError(f"intention {it.id!r} comes after unknown id {it.after!r}")
            self._after.append(None if it.after is None else ids.index(it.after))

    @staticmethod
    def buffer_months(traits):
        return buffer_months(traits)

    def reset(self, s: State, streams: Streams):
        N = s.n
        self.last_move = np.full(N, -99)
        self.last_status = np.full(N, -99)
        self.last_job_loss = np.full(N, -99)
        K = len(self.intentions)
        self.fired = np.zeros((K, N), bool)
        self.switches = np.zeros(N, int)
        # Plan bookkeeping, per intention and path: month carried out (-1 = not yet), chosen this
        # month, ever due (time and conditions met), ever due and possible, ever due but not possible.
        self.done_at = np.full((K, N), -1)
        self.chosen = np.zeros((K, N), bool)
        self.was_due = np.zeros((K, N), bool)
        self.was_open = np.zeros((K, N), bool)
        self.tried = np.zeros((K, N), bool)          # chosen at least once (the world may still say no)
        self.searching = np.zeros((K, N), bool)      # a housing intention's flat search is still open
        self.blocked = np.zeros((K, N), bool)
        self.last_event = {e: np.full(N, -999) for e in EVENTS}

    # ------------------------------------------------------------------ features of the state
    def context(self, s: State):
        t = s.t
        out_m = np.maximum(self.w.outgoings(s), 1.0)
        cash_m = s.cash / out_m
        liquid_m = (s.cash + s.inv) / out_m
        g = self.goals
        nw = s.cash + s.inv - s.debt
        high_have = np.where(g["high_cash_only"], s.cash, nw)
        pressure = np.zeros(s.n)
        for target, have, deadline in ((g["high_target"], high_have, C.T_HIGH_GOAL),
                                       (g["high_target"] + np.nan_to_num(g["med_target"]), nw, C.T_MED_GOAL)):
            left = deadline - t
            if left <= 0:
                continue
            gap = np.clip((target - have) / np.maximum(target, 1), 0, 1)
            gap = np.where(np.isnan(target), 0, gap)
            pressure = np.maximum(pressure, gap * (1.0 if left <= 24 else 0.5))
        return dict(
            cash_m=cash_m, liquid_m=liquid_m,
            cash_pressure=np.clip((2 - cash_m) / 2, 0, 1),
            comfort=np.clip((liquid_m - 6) / 18, 0, 1),
            goal_pressure=pressure,
            retire_ready=np.clip(liquid_m / 60, 0, 1),
            recent_move=(t - self.last_move) < 24,
            recent_loss=(t - self.last_job_loss) < 6,
        )

    # --------------------------------------------------------------------------- logits
    def logits(self, s: State, c: dict):
        T = lambda name: self.tr[:, TRAIT[name]]  # noqa: E731
        N, age = s.n, s.age
        mode, cp, gp = s.mode, c["cash_pressure"], c["goal_pressure"] * T("goal_focus")
        st = np.zeros((N, 5))
        st[:, 0] = 1.0 + 3.0 * T("inertia")
        st[:, 1] = -3.0 + 4.0 * T("career_drive") - 0.15 * np.maximum(age - 28, 0) - 2.0 * cp
        from_work = np.isin(mode, [C.PART_TIME, C.FULL_TIME])
        idle = np.isin(mode, [C.STUDENT, C.NOT_WORKING])
        old = age >= self.w.pr["retirement"]["pension_age"]
        st[:, 2] = np.where(mode == C.FULL_TIME,
                            -3.5 + 3.5 * T("free_time") - 2.0 * cp - 1.0 * gp + 1.0 * (age >= 60),
                            -0.5 + 2.0 * T("free_time") + 1.0 * cp + 1.0 * c["recent_loss"])
        st[:, 3] = np.where(mode == C.PART_TIME,
                            -2.5 + 2.5 * T("career_drive") + 2.5 * cp + 1.5 * gp - 1.5 * T("free_time"),
                            np.where(mode == C.STUDENT,
                                     -3.0 + 1.5 * T("career_drive") + 3.0 * cp,
                                     1.0 + 2.0 * T("career_drive") + 2.0 * cp - 1.5 * T("free_time")
                                     + 1.5 * c["recent_loss"]))
        st[idle & old, 2:4] = -5.0
        st[:, 4] = (-5.0 + 3.0 * T("free_time") + (age >= 58) * (1.0 + 3.0 * c["retire_ready"])
                    + 3.0 * old + 2.0 * (age >= 67) - 2.0 * cp)
        st[~from_work, 4] = NEG

        need_bigger = ((s.household == C.FAMILY) & (s.tier < 2)) * 2.5 + ((s.household == C.COUPLE) & (s.tier < 2)) * 1.0
        ho = np.zeros((N, 3))
        ho[:, 1] = 1.5 + 3.0 * T("inertia") + 1.5 * c["recent_move"]
        ho[:, 0] = -3.0 + 4.0 * cp + 2.0 * gp + 1.0 * T("frugality")
        ho[:, 2] = -3.5 + 2.5 * T("housing_aspiration") + 2.0 * c["comfort"] + need_bigger - 3.0 * cp - 1.5 * gp

        sp = np.zeros((N, 3))
        sp[:, 1] = 3.5 + 2.0 * T("inertia")
        sp[:, 0] = YEARLY - 2.0 + 4.0 * cp + 2.5 * gp + 1.0 * T("frugality")
        sp[:, 2] = (YEARLY - 2.5 + 1.5 * (1 - T("frugality")) + 1.5 * T("impatience") + 1.5 * c["comfort"]
                    - 4.0 * cp - 1.5 * gp)

        buffer = np.maximum(s.buffer_months * self.w.outgoings(s), 1.0)
        below = np.clip(1 - s.cash / buffer, 0, 1)               # 0 at the buffer, 1 at zero cash
        asst = np.zeros((N, 3))
        asst[:, 1] = 1.0 + 1.0 * T("inertia")
        asst[:, 2] = ANNUAL_REBALANCE - 0.5 + 3.0 * T("risk_tolerance") + 1.0 * (s.cash > 2 * buffer)
        asst[:, 0] = ANNUAL_REBALANCE - 1.5 + 6.0 * below + 1.0 * (1 - T("risk_tolerance"))
        return {"status": st, "housing": ho, "spending": sp, "assets": asst}

    # ----------------------------------------------------------------------------- acting
    def decision_points(self, s: State):
        return np.full(s.n, C.annual_review(s.t))

    def act(self, s: State, streams: Streams):
        """Sample (N, 4) actions; also return the probabilities they were drawn from."""
        c = self.context(s)
        valid = dict(zip(HEADS, self.w.masks(s)))
        masks = dict(valid)
        lg = self.logits(s, c)
        decide = self.decision_points(s)
        self.last_decide = decide
        for h in ("status", "housing"):            # outside a decision point: continue / stay
            keep = np.zeros_like(masks[h])
            keep[:, 0 if h == "status" else 1] = True
            masks[h] = np.where(decide[:, None], masks[h], keep)
        u = streams.uniform("policy", s.t, 4 + len(self.intentions))
        probs, choice = {}, {}
        for i, h in enumerate(HEADS):
            probs[h] = softmax_masked(lg[h], masks[h], self.temp)
            choice[h] = sample(probs[h], u[i])
        self._apply_intentions(s, c, valid, decide, choice, probs, u[4:])
        action = np.stack([choice["status"], choice["housing"] - 1, choice["spending"] - 1, choice["assets"] - 1], 1)
        return action, probs

    def _due(self, k, it, s, c, decide):
        """Paths where intention k applies this month (timing and conditions), open option or not."""
        t = s.t
        if it.when is not None and not it.windowed:
            ok = np.full(s.n, t == C.next_review(C.month_index(it.when)))  # falls due at a January review
        else:
            start = 0 if it.when is None else C.next_review(C.month_index(it.when))
            ok = decide & (t >= start)
            if it.windowed:
                ok &= t <= C.month_index(it.until)
        if it.min_age is not None:
            ok &= s.age >= it.min_age
        if it.max_age is not None:
            ok &= s.age <= it.max_age
        if it.cash_months_below is not None:
            ok &= c["cash_m"] < it.cash_months_below
        if it.cash_months_above is not None:
            ok &= c["cash_m"] > it.cash_months_above
        if self._after[k] is not None:
            done = self.done_at[self._after[k]]
            ok &= (done >= 0) & (t - done >= it.after_months)
        if it.event is not None:
            ok &= t - self.last_event[it.event] <= it.within_months
        if it.once:
            ok &= ~self.fired[k]
        return ok

    def _apply_intentions(self, s, c, valid, decide, choice, probs, u):
        """Stated plans fall due at the first January on or after the requested date."""
        for k, it in enumerate(self.intentions):
            idx = it.option if it.head == "status" else it.option + 1
            due = self._due(k, it, s, c, decide)
            possible = valid[it.head][:, idx]
            ok = due & possible
            act_now = ok & (u[k] < it.prob)
            choice[it.head] = np.where(act_now, idx, choice[it.head])
            self.chosen[k] = act_now
            self.tried[k] |= act_now
            if it.head == "housing":                 # a January choice opens or replaces the flat search
                self.searching[k] = np.where(decide, act_now, self.searching[k])
            self.was_due[k] |= due
            self.was_open[k] |= ok
            self.blocked[k] |= due & ~possible
            if it.once:
                # Considered once; acted on with probability prob. A windowed plan the world turns
                # down (no admission, no flat) is retried at the next review; see observe().
                self.fired[k] |= (ok & ~act_now) if it.windowed else ok
            onehot = np.eye(N_OPTS[it.head])[idx]
            probs[it.head] = np.where(ok[:, None], it.prob * onehot + (1 - it.prob) * probs[it.head], probs[it.head])

    def observe(self, info: dict, s: State):
        """Update the path's memory after the world moved."""
        t = s.t - 1
        moved = info["moved"] | info["forced_move"]           # moves that happened, not flat searches
        self.last_move = np.where(moved, t, self.last_move)
        changed = info["status_change"] | info["offer"] | info["job_loss"] | info["graduated"]
        self.last_status = np.where(changed, t, self.last_status)
        self.switches += changed
        self.last_job_loss = np.where(info["job_loss"], t, self.last_job_loss)
        for e in EVENTS:
            self.last_event[e] = np.where(info[e], t, self.last_event[e])
        a = info["action"]
        col = {h: j for j, h in enumerate(HEADS)}
        refused = info.get("admission_refused", np.zeros(s.n, bool))
        for k, it in enumerate(self.intentions):
            # Carried out = it really happened: a move once the flat search finds a flat (that can
            # take months), a master's once admitted, anything else once the world applied it.
            if it.head == "housing" and it.option != 0:
                done = self.searching[k] & info["moved"]
                self.searching[k] &= ~info["moved"]
            else:
                done = self.chosen[k] & (a[:, col[it.head]] == it.option)
                if it.head == "status" and it.option == 1:
                    done &= ~refused
            done &= self.done_at[k] < 0
            self.done_at[k] = np.where(done, t, self.done_at[k])
            if it.once and it.windowed:
                self.fired[k] |= done
        self.chosen[:] = False
