"""Foresight: the same persona, nudged toward outcomes it values. A knob f from 0 to 1.

    python -m twin foresight                     # ~5 min: look ahead, distil, compare on unseen profiles
    python -m twin explore P00001 --policy foresight --foresight 0.5

At a decision point the persona's own probabilities pi(o) are tilted by how much each option
is worth to this persona over the next five years:

    pi_f(o)  proportional to  pi(o) * exp( f * (U(o) - max U) / CHF 10,000 )

This is the exact optimum of  f * utility - KL(pi_f || pi)  for that decision, so f = 0 is
the person as they tend to decide, and f = 1 makes an option worth CHF 10k more e times as
likely. U(o) is estimated by simulating the option in the learned world model, 60 months
ahead with the persona deciding afterwards; every option of a decision point sees the same
luck, so only the choice differs. The tilted probabilities at ~2,000 decision points, for
f = 0, 0.25, 0.5 and 1, are distilled into the neural persona with an extra input f.
U is the spec's preference table (lifestyle, home, free time, months below zero, moves,
goals met on time) plus net worth at the end of the look-ahead.

(A first version trained f with policy gradient over whole lives; with one reward per
ten-year life spread over ~480 sampled choices it learned nothing measurable, so it was
replaced by this look-ahead.)
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from .. import config as C
from ..world.env import STREAMS, State, Streams, World
from .net import N_IN, SIZES, NeuralPolicy, PersonaNet, kl_loss, load_net, split_ids, teacher_data
from .rules import HEADS, RulePolicy, sample, softmax_masked
from .traits import from_profile

PATH = C.ARTIFACTS / "foresight_net.pt"
LEVELS = (0.0, 0.25, 0.5, 1.0)
SCALE = 10_000.0          # CHF-equivalent that multiplies an option's odds by e at f = 1
PREFS = {   # the spec's default preference table, CHF-equivalent a month
    "lifestyle": [-300.0, 0.0, 200.0],
    "home": [[-700.0, 0.0, 300.0], [-1500.0, -500.0, 0.0], [-2000.0, -1200.0, 0.0]],   # household x tier
    "free_time": {C.STUDENT: 0.0, C.PART_TIME: 1500.0, C.FULL_TIME: 0.0, C.NOT_WORKING: 2000.0},
    "below_zero": -2000.0, "change": -300.0, "high_goal_share": 1.0, "medium_goal_share": 0.5,
}
FREE = np.array([PREFS["free_time"][m] for m in range(4)])
HOME = np.array(PREFS["home"])
LIFE = np.array(PREFS["lifestyle"])
# Option slots evaluated at a decision point: status 0-4, housing -1/0/+1, spending -1/0/+1.
SLOTS = [("status", j) for j in range(5)] + [("housing", j) for j in range(3)] + [("spending", j) for j in range(3)]
HEAD_SLOTS = {"status": slice(0, 5), "housing": slice(5, 8), "spending": slice(8, 11)}
OFF = {"status": 0, "housing": 5, "spending": 8, "assets": 11}


def month_utility(lifestyle, household, tier, mode, cash, changed):
    """Persona utility of one month, CHF-equivalent (vectorised over paths)."""
    return (LIFE[lifestyle] + HOME[household, tier] + FREE[mode]
            + PREFS["below_zero"] * (cash < 0) + PREFS["change"] * changed)


def goal_value(goals, t, cash, nw):
    """Value of goals met at their deadline (month index t is the month just finished)."""
    v = np.zeros(len(cash))
    if t == C.T_HIGH_GOAL:
        have = np.where(goals["high_cash_only"], cash, nw)
        v += PREFS["high_goal_share"] * goals["high_target"] * (have >= goals["high_target"])
    if t == C.T_MED_GOAL:
        med = np.nan_to_num(goals["med_target"])
        v += PREFS["medium_goal_share"] * med * ((nw >= goals["high_target"] + med) & (med > 0))
    return v


def life_utility(paths) -> np.ndarray:
    """Persona utility of each sampled life (explore.Paths), CHF-equivalent."""
    from ..explore import goals_per_path
    goals = goals_per_path(paths.people, paths.n_paths)
    N, T = paths.action.shape[:2]
    u = np.zeros(N)
    for t in range(T):
        k = t + 1
        changed = (paths.action[:, t, 0] > 0) | (paths.action[:, t, 1] != 0)
        u += month_utility(paths.lifestyle[:, k], paths.household[:, k], paths.tier[:, k], paths.mode[:, k],
                           paths.cash[:, k], changed)
        if t in (C.T_HIGH_GOAL, C.T_MED_GOAL):
            u += goal_value(goals, t, paths.cash[:, k], paths.nw[:, k]) * ~paths.credit_breach[:, :k].any(1)
    return u + paths.nw[:, -1]


class GroupStreams(Streams):
    """Streams whose draws are shared by consecutive groups of `g` paths, except `own` streams."""

    def __init__(self, seed, n, g, own=()):
        super().__init__(seed, n)
        self.g, self.own = g, set(own)

    def _draw(self, name, t, k, normal):
        size = self.n if name in self.own else self.n // self.g
        key = [self.seed, STREAMS[name], t] + ([1] if normal else [])
        r = np.random.default_rng(key)
        x = r.standard_normal((k, size)) if normal else r.random((k, size))
        return x if name in self.own else np.repeat(x, self.g, axis=1)

    def uniform(self, name, t, k=1):
        return self._draw(name, t, k, False)

    def normal(self, name, t, k=1):
        return self._draw(name, t, k, True)


# ------------------------------------------------------------------------------ look-ahead
def subset(s: State, rows) -> State:
    out = s.copy()
    for f in out.__dataclass_fields__:
        if f != "t":
            setattr(out, f, getattr(s, f)[rows])
    return out


def option_values(world, ctx, points, K=6, horizon=60, log=print):
    """Q (n_points, 11): mean persona utility of each option slot over the next `horizon` months."""
    people, traits, temp = ctx["people"], ctx["traits"], ctx["temperature"]
    Q = np.full((len(points), len(SLOTS)), np.nan)
    by_k = {}
    for j, (i, k) in enumerate(points):
        by_k.setdefault(k, []).append(j)
    G = len(SLOTS)
    for n, (k, js) in enumerate(sorted(by_k.items())):
        lives = np.array([points[j][0] for j in js])
        rows = np.repeat(lives, K * G)                        # point, rollout, slot
        s = subset(ctx["full"][k], rows)
        N = len(rows)
        streams = GroupStreams(C.SEED + 7 + k, N, G)          # every slot of a rollout shares all draws
        person = s.person
        goals = {"high_target": people.high_target.values[person].astype(float),
                 "high_cash_only": people.high_goal.values[person] == "Emergency savings",
                 "med_target": people.med_target.values[person].astype(float)}
        pol = RulePolicy(world, traits[rows], temp[rows], goals)
        pol.reset(s, streams)
        slot = np.tile(np.arange(G), len(js) * K)
        a = np.zeros((N, 4), int)
        a[slot < 5, 0] = slot[slot < 5]
        hs = (slot >= 5) & (slot < 8)
        a[hs, 1] = slot[hs] - 6
        a[slot >= 8, 2] = slot[slot >= 8] - 9
        util = np.zeros(N)
        breached = np.zeros(N, bool)
        end = min(k + horizon, C.HORIZON)
        for t in range(k, end):
            if t > k:
                a, _ = pol.act(s, streams)
            s, info = world.step(s, a, streams)
            pol.observe(info, s)
            changed = (info["action"][:, 0] > 0) | (info["action"][:, 1] != 0)
            util += month_utility(s.lifestyle, s.household, s.tier, s.mode, s.cash, changed)
            breached |= info["credit_breach"]
            util += goal_value(goals, t, s.cash, s.cash + s.inv - s.debt) * ~breached
        util += s.cash + s.inv - s.debt
        Q[js] = util.reshape(len(js), K, G).mean(1)
        if n % 30 == 0:
            log(f"  look-ahead: month {k}, {len(js)} decision points, {N} simulated lives")
    return Q


def tilted(P_point, M_point, q, f):
    """Tilt the persona's probabilities at one decision point (14-vector) toward valuable options."""
    out = P_point.copy()
    for h, sl in HEAD_SLOTS.items():
        o = OFF[h]
        n = SIZES[HEADS.index(h)]
        v = M_point[o:o + n].astype(bool)
        if v.sum() <= 1:
            continue
        d = np.where(v, q[sl] - np.nanmax(np.where(v, q[sl], -np.inf)), 0.0) / SCALE
        p = P_point[o:o + n] * np.exp(f * d) * v
        out[o:o + n] = p / p.sum()
    return out


# ------------------------------------------------------------------------------- policy
def foresight_net_from_clone(clone: PersonaNet) -> PersonaNet:
    net = PersonaNet(n_in=N_IN + 1)
    sd = clone.state_dict()
    w = sd["inp.0.weight"]
    sd["inp.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], 1)], 1)    # f starts with no effect
    net.load_state_dict(sd)
    return net


def load_foresight(path=PATH):
    net = PersonaNet(n_in=N_IN + 1)
    net.load_state_dict(torch.load(path, weights_only=True))
    net.eval()
    return net


class ForesightPolicy(NeuralPolicy):
    """The neural persona with the foresight knob set to `foresight` (0 = as they tend to decide)."""
    name = "foresight persona"
    foresight = 0.5

    def __init__(self, world, traits, temperature, goals, intentions=(), net=None, foresight=None):
        super().__init__(world, traits, temperature, goals, intentions, net=net or load_foresight())
        self.f = self.foresight if foresight is None else foresight

    def inputs(self, s, c, decide):
        x = super().inputs(s, c, decide)
        return np.hstack([x, np.full((len(x), 1), self.f, np.float32)])


def at_level(f):
    """A ForesightPolicy class fixed at level f (for explore.simulate's policy_cls)."""
    return type(f"Foresight{int(round(100 * f))}", (ForesightPolicy,), {"foresight": f, "name": f"foresight {f:g}"})


class AnchoredForesightPolicy(ForesightPolicy):
    """The rule persona's own probabilities, tilted by what the foresight net learned at level f:

        pi_f(o)  proportional to  pi_rules(o) * exp( net_f(o) - net_0(o) )

    The net's f = 0 output stands in for its learned copy of the persona, so only its foresight
    adjustment is applied. f = 0 is the rule persona exactly; the unanchored net can drift from
    the rules at f = 0 on rare choices (e.g. part-time back to full-time) that compound over a life.
    """
    name = "anchored foresight persona"

    def reset(self, s, streams):
        super().reset(s, streams)
        self.h0 = None

    def act(self, s: State, streams: Streams):
        c = self.context(s)
        valid, masks, decide = self.clock_masks(s)
        self.last_decide = decide
        x = self.inputs(s, c, decide)
        x0 = x.copy()
        x0[:, -1] = 0.0
        with torch.no_grad():
            at_f, self.h = self.net(torch.as_tensor(x)[:, None, :], self.h)
            at_0, self.h0 = self.net(torch.as_tensor(x0)[:, None, :], self.h0)
        rules = self.logits(s, c)
        u = streams.uniform("policy", s.t, 4 + len(self.intentions))
        probs, choice = {}, {}
        for i, h in enumerate(HEADS):
            p = softmax_masked(rules[h], masks[h], self.temp)
            tilt = (at_f[i][:, 0] - at_0[i][:, 0]).numpy().astype(float)
            q = p * np.exp(tilt - tilt.max(1, keepdims=True)) * masks[h]
            probs[h] = q / q.sum(1, keepdims=True)
            choice[h] = sample(probs[h], u[i])
        self._apply_intentions(s, c, valid, decide, choice, probs, u[4:])
        action = np.stack([choice["status"], choice["housing"] - 1, choice["spending"] - 1, choice["assets"] - 1], 1)
        return action, probs


def anchored_at_level(f):
    """An AnchoredForesightPolicy class fixed at level f (for explore.simulate's policy_cls)."""
    return type(f"AnchoredForesight{int(round(100 * f))}", (AnchoredForesightPolicy,),
                {"foresight": f, "name": f"anchored foresight {f:g}"})


# ------------------------------------------------------------------------------ training
def train(n_points=2000, K=6, horizon=60, epochs=15, weight=10.0, n_paths=6, log=print):
    from .llm import pick_points
    world = World.load()
    train_ids, test_ids = split_ids()
    log(f"teacher lives for {len(train_ids)} profiles ...")
    X, M, P, ctx = teacher_data(world, train_ids, n_paths, keep_full=True)
    points = pick_points(ctx, n_points, seed=C.SEED + 3)
    log(f"look-ahead at {len(points)} decision points: {len(SLOTS)} options x {K} rollouts x {horizon} months ...")
    Q = option_values(world, ctx, points, K, horizon, log)
    Xs, Ps, Ws = [], [], []
    shift = {}
    for f in LEVELS:
        Pf, Wf = P.copy(), np.ones(P.shape[:2], np.float32)
        tv = []
        for (i, k), q in zip(points, Q):
            Pf[i, k] = tilted(P[i, k], M[i, k], q, f)
            Wf[i, k] = weight
            tv.append(0.5 * np.abs(Pf[i, k] - P[i, k])[:11].sum())
        shift[f] = float(np.mean(tv))
        Xs.append(np.concatenate([X, np.full(X.shape[:2] + (1,), f, np.float32)], -1))
        Ps.append(Pf)
        Ws.append(Wf)
    log("average change of the persona's choice probabilities at decision points: "
        + ", ".join(f"f={f:g}: {v:.3f}" for f, v in shift.items()))
    Xa, Ma, Pa, Wa = (torch.as_tensor(np.concatenate(a)) for a in (Xs, [M] * len(LEVELS), Ps, Ws))
    Xv, Mv, Pv = teacher_data(world, test_ids, n_paths, seed=C.SEED + 1)
    Xv = torch.as_tensor(np.concatenate([Xv, np.zeros(Xv.shape[:2] + (1,), np.float32)], -1))
    Mv, Pv = torch.as_tensor(Mv), torch.as_tensor(Pv)
    torch.manual_seed(C.SEED)
    net = foresight_net_from_clone(load_net())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(C.SEED)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xa), generator=gen)
        for b in range(0, len(Xa), 64):
            idx = perm[b:b + 64]
            loss = kl_loss(net(Xa[idx])[0], Ma[idx], Pa[idx], Wa[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 5 == 4:
            net.eval()
            with torch.no_grad():
                v = kl_loss(net(Xv)[0], Mv, Pv).item()
            log(f"  epoch {ep + 1}: training KL {loss.item():.4f}, held-out KL at f=0 (vs the persona) {v:.4f}")
    net.eval()
    torch.save(net.state_dict(), PATH)
    q_spread = np.nanmax(Q, 1) - np.nanmin(Q, 1)
    return {"points": len(points), "rollouts": K, "horizon": horizon, "scale_chf": SCALE,
            "mean_prob_change": shift, "median_option_value_range_chf": float(np.median(q_spread))}


def compare(ids, n_paths=500, levels=(0.0, 0.5, 1.0), log=print):
    """Rule persona and the foresight persona at several levels, same profiles and seeds."""
    from ..explore import goal_met, lasting_shortfall, person_table, simulate, yearly_action_rates
    world = World.load()
    rows = []
    for pid in ids:
        people = person_table(pid)
        persona = from_profile(people.iloc[0])
        row = {"profile": pid, "status": people.iloc[0]["status"], "age": float(people.iloc[0]["age"]), "levels": {}}
        for name, cls in [("rules", RulePolicy)] + [(f"f={lv:g}", at_level(lv)) for lv in levels]:
            p = simulate(world, people, persona, n_paths, seed=5, policy_cls=cls)
            r = yearly_action_rates(p)
            row["levels"][name] = {"high_goal_odds": float(goal_met(p)[0].mean()),
                                   "shortfall_odds": float(p.below_zero.any(1).mean()),
                                   "lasting_shortfall_odds": float(lasting_shortfall(p).mean()),
                                   "median_final_net_worth": float(np.median(p.nw[:, -1])),
                                   "mean_life_utility": float(life_utility(p).mean()),
                                   "status_changes_per_year": r["status change"],
                                   "lifestyle_changes_per_year": r["lifestyle change"],
                                   "housing_moves_per_year": r["housing move"]}
        rows.append(row)
        log(f"  {pid} ({row['status']}, {row['age']:.0f}): " + " | ".join(
            f"{k}: goal {v['high_goal_odds']:.2f}, short {v['shortfall_odds']:.2f}, NW {v['median_final_net_worth'] / 1e3:,.0f}k,"
            f" U {v['mean_life_utility'] / 1e3:,.0f}k" for k, v in row["levels"].items()))
    return rows


def main(n_points=2000, log=print, **_):
    t0 = time.time()
    info = train(n_points=n_points, log=log)
    _, test_ids = split_ids()
    log("comparison on profiles the persona never saw (500 lives each):")
    ids = list(test_ids[:5]) + ["P00001", "P00003"]
    rows = compare(ids, log=log)
    res = {**info, "levels": LEVELS, "held_out": rows, "seconds": round(time.time() - t0, 1)}
    (C.ARTIFACTS / "foresight.json").write_text(json.dumps(res, indent=1))
    log(f"saved {PATH} ({res['seconds']:.0f}s)")
    return res
