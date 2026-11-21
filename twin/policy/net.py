"""Neural persona (stretch): one GRU network for all personas, cloned from a teacher.

    python -m twin clone              # generate teacher lives, train, compare, save

The teacher is the rule persona; LLM persona demonstrations can be added to the same
dataset later (same inputs, same probability targets). The student sees the state, its own
path so far (GRU memory), the persona's traits and the decision clock, and outputs one
probability per option for status, housing, spending and assets. It is trained on the
teacher's full probabilities (KL divergence), not on sampled actions, and is judged on
personas built from profiles it never saw.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from torch import nn

from .. import config as C
from ..data import profiles
from ..world.env import State, Streams, World
from .rules import HEADS, N_OPTS, RulePolicy, sample
from .traits import TRAITS, Persona, from_profile, sample_traits

SIZES = [N_OPTS[h] for h in HEADS]
N_IN = 47


def policy_inputs(w: World, s: State, c: dict, decide, traits, temperature) -> np.ndarray:
    N = s.n
    X = np.zeros((N, N_IN), np.float32)
    r = np.arange(N)
    X[r, s.mode] = 1                                                   # 0-3
    X[r, 4 + s.search] = 1                                             # 4-6
    X[:, 7], X[:, 8] = s.degree, s.pensioner
    X[:, 9] = np.log1p(s.income / s.P / 1000)
    X[:, 10] = np.log(np.maximum(s.rent / s.P, 50) / 1000)
    X[r, 11 + s.tier] = 1                                              # 11-13
    X[r, 14 + s.household] = 1                                         # 14-16
    X[r, 17 + s.lifestyle] = 1                                         # 17-19
    X[r, 20 + s.econ] = 1                                              # 20-22
    X[:, 23] = np.clip(c["cash_m"], -6, 24) / 12
    X[:, 24] = np.clip(c["liquid_m"], -6, 120) / 60
    for j, k in enumerate(["cash_pressure", "comfort", "goal_pressure", "retire_ready", "recent_move", "recent_loss"]):
        X[:, 25 + j] = c[k]                                            # 25-30
    X[:, 31] = s.age / 100
    X[:, 32] = np.minimum(s.tau, 120) / 120
    X[:, 33] = max(C.T_HIGH_GOAL - s.t, 0) / 60
    X[:, 34] = max(C.T_MED_GOAL - s.t, 0) / 96
    X[:, 35] = decide
    buffer = np.maximum(s.buffer_months * w.outgoings(s), 1.0)
    X[:, 36], X[:, 37] = s.cash > buffer, s.cash > 2 * buffer
    X[:, 38:46] = traits
    X[:, 46] = temperature
    return X


class PersonaNet(nn.Module):
    def __init__(self, hidden=32, n_in=N_IN):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU())
        self.gru = nn.GRU(64, hidden, batch_first=True)
        self.heads = nn.ModuleList([nn.Linear(hidden + 64, k) for k in SIZES])

    def forward(self, x, h=None):
        """x (B, T, N_IN) -> list of logits (B, T, k) per head, new hidden state."""
        e = self.inp(x)
        o, h = self.gru(e, h)
        z = torch.cat([o, e], -1)
        return [head(z) for head in self.heads], h


def masked_log_softmax(logits, mask):
    return torch.log_softmax(logits.masked_fill(~mask, -1e9), -1)


class NeuralPolicy(RulePolicy):
    """Same decision clock, masks and intentions as the rule persona; choices from the network."""
    name = "neural persona"

    def __init__(self, world, traits, temperature, goals, intentions=(), net: PersonaNet | None = None):
        super().__init__(world, traits, temperature, goals, intentions)
        self.net = net or load_net()

    def reset(self, s, streams):
        super().reset(s, streams)
        self.h = None

    def inputs(self, s, c, decide):
        return policy_inputs(self.w, s, c, decide, self.tr, self.temp)

    def clock_masks(self, s):
        """Valid options, and the options open this month under the decision clock."""
        valid = dict(zip(HEADS, self.w.masks(s)))
        masks = dict(valid)
        decide = self.decision_points(s)
        for h in ("status", "housing"):
            keep = np.zeros_like(masks[h])
            keep[:, 0 if h == "status" else 1] = True
            masks[h] = np.where(decide[:, None], masks[h], keep)
        return valid, masks, decide

    def act(self, s: State, streams: Streams):
        c = self.context(s)
        valid, masks, decide = self.clock_masks(s)
        self.last_decide = decide
        x = torch.as_tensor(self.inputs(s, c, decide))[:, None, :]
        with torch.no_grad():
            logits, self.h = self.net(x, self.h)
        u = streams.uniform("policy", s.t, 4 + len(self.intentions))
        probs, choice = {}, {}
        for i, h in enumerate(HEADS):
            lp = masked_log_softmax(logits[i][:, 0], torch.as_tensor(masks[h]))
            probs[h] = lp.exp().numpy().astype(float) * masks[h]
            probs[h] /= probs[h].sum(1, keepdims=True)
            choice[h] = sample(probs[h], u[i])
        self._apply_intentions(s, c, valid, decide, choice, probs, u[4:])
        action = np.stack([choice["status"], choice["housing"] - 1, choice["spending"] - 1, choice["assets"] - 1], 1)
        return action, probs


# --------------------------------------------------------------------------------- training
class Recorder(RulePolicy):
    """The rule persona, recording what the student needs: inputs, masks, probabilities."""

    def act(self, s, streams):
        action, probs = super().act(s, streams)
        c = self.context(s)
        valid = dict(zip(HEADS, self.w.masks(s)))
        for h in ("status", "housing"):
            keep = np.zeros_like(valid[h])
            keep[:, 0 if h == "status" else 1] = True
            valid[h] = np.where(self.last_decide[:, None], valid[h], keep)
        self.log.append((policy_inputs(self.w, s, c, self.last_decide, self.tr, self.temp),
                         np.concatenate([valid[h] for h in HEADS], 1),
                         np.concatenate([probs[h] for h in HEADS], 1).astype(np.float32)))
        if self.keep_states:          # readable snapshot for LLM prompts
            snap = {k: getattr(s, k).copy() for k in SNAP_FIELDS}
            snap.update(t=s.t, decide=self.last_decide.copy(), outgoings=self.w.outgoings(s),
                        recent_loss=c["recent_loss"].copy(), recent_move=c["recent_move"].copy(),
                        valid={h: valid[h].copy() for h in HEADS})
            self.states.append(snap)
        if self.keep_full:            # full world state, to restart simulations from this month
            self.full.append(s.copy())
        return action, probs


SNAP_FIELDS = ["person", "age", "mode", "tau", "income", "pensioner", "search", "degree", "tier", "rent", "city",
               "household", "child_cost", "lifestyle", "buffer_months", "econ", "cash", "inv", "debt", "loan_pay"]


def teacher_data(world: World, ids, n_paths=6, seed=C.SEED, months=C.HORIZON, keep_states=False, keep_full=False):
    """Rule-persona lives for varied personas around each profile's defaults.

    With keep_states, also returns the readable states and personas (for LLM prompts)."""
    from ..explore import goals_per_path
    prof = profiles().loc[ids]
    N = len(prof) * n_paths
    streams = Streams(seed, N)
    z = streams.normal("traits", 0, 8)
    traits = np.vstack([sample_traits(Persona(from_profile(r).traits, spread=0.8), n_paths, z[:, i * n_paths:(i + 1) * n_paths])
                        for i, (_, r) in enumerate(prof.iterrows())])
    temp = 0.7 + 0.8 * streams.uniform("traits", 1, 1)[0]
    s = world.initial_state(prof, n_paths, streams, buffer_months=RulePolicy.buffer_months(traits))
    pol = Recorder(world, traits, temp, goals_per_path(prof, n_paths))
    pol.reset(s, streams)
    pol.log, pol.states, pol.keep_states, pol.full, pol.keep_full = [], [], keep_states or keep_full, [], keep_full
    for _ in range(months):
        a, _ = pol.act(s, streams)
        s, info = world.step(s, a, streams)
        pol.observe(info, s)
    X = np.stack([x for x, _, _ in pol.log], 1)
    M = np.stack([m for _, m, _ in pol.log], 1)
    P = np.stack([p for _, _, p in pol.log], 1)
    if keep_states or keep_full:
        return X, M, P, {"states": pol.states, "full": pol.full, "traits": traits, "temperature": temp, "people": prof}
    return X, M, P


def split_heads(t):
    out, i = [], 0
    for k in SIZES:
        out.append(t[..., i:i + k])
        i += k
    return out


def kl_loss(logits, M, P, W=None):
    """Weighted mean over steps of sum over heads KL(teacher || student), on valid options."""
    total = 0.0
    for lg, m, p in zip(logits, split_heads(M), split_heads(P)):
        lq = masked_log_softmax(lg, m)
        lp = torch.log(p.clamp_min(1e-9))
        kl = (p * (lp - lq) * m).sum(-1)
        total = total + (kl.mean() if W is None else (kl * W).sum() / W.sum())
    return total


def train_net(X, M, P, Xv, Mv, Pv, epochs=80, batch=64, lr=2e-3, seed=C.SEED, log=print, W=None):
    """W (lives x months): weight of each step's target; LLM answers get more than rule targets."""
    torch.manual_seed(seed)
    net = PersonaNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    X, M, P = torch.as_tensor(X), torch.as_tensor(M), torch.as_tensor(P)
    W = torch.ones(X.shape[:2]) if W is None else torch.as_tensor(W, dtype=torch.float32)
    Xv, Mv, Pv = torch.as_tensor(Xv), torch.as_tensor(Mv), torch.as_tensor(Pv)
    gen = torch.Generator().manual_seed(seed)
    best, state = float("inf"), None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(X), generator=gen)
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            loss = kl_loss(net(X[idx])[0], M[idx], P[idx], W[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = kl_loss(net(Xv)[0], Mv, Pv).item()
        if v < best:
            best, state = v, {k: t.clone() for k, t in net.state_dict().items()}
        if ep % 10 == 9:
            log(f"  epoch {ep + 1}: held-out KL {v:.4f}")
    net.load_state_dict(state)
    net.eval()
    return net, best


def per_head_kl(net, X, M, P):
    with torch.no_grad():
        logits = net(torch.as_tensor(X))[0]
    out = {}
    for h, lg, m, p in zip(HEADS, logits, split_heads(torch.as_tensor(M)), split_heads(torch.as_tensor(P))):
        lq = masked_log_softmax(lg, m)
        out[h] = float((p * (torch.log(p.clamp_min(1e-9)) - lq) * m).sum(-1).mean())
    return out


def load_net(path=C.ARTIFACTS / "persona_net.pt"):
    net = PersonaNet()
    net.load_state_dict(torch.load(path, weights_only=True))
    net.eval()
    return net


def split_ids(n_profiles=400):
    """Profiles for teacher lives: (training ids, held-out ids). Shared by clone, LLM demos and foresight."""
    ids = np.random.default_rng(C.SEED).permutation(profiles().index.values)[:n_profiles]
    n_test = n_profiles // 5
    return ids[n_test:], ids[:n_test]


def main(n_profiles=400, n_paths=6, use_llm=False, llm_weight=10.0, llm_sharpen=(1.0, 2.0, 2.0, 1.0), log=print):
    t0 = time.time()
    if use_llm:
        from .llm import DEMOS
        if not DEMOS.exists():
            raise SystemExit(f"No LLM answers yet ({DEMOS}). Run `python -m twin llm-demos --calls 300` first.")
    world = World.load()
    prof = profiles()
    train_ids, test_ids = split_ids(n_profiles)
    ids = np.concatenate([test_ids, train_ids])
    log(f"teacher lives: {n_profiles} profiles x {n_paths} personas x {C.HORIZON} months ...")
    X, M, P = teacher_data(world, train_ids, n_paths)
    Xv, Mv, Pv = teacher_data(world, test_ids, n_paths, seed=C.SEED + 1)
    W, n_llm = None, 0
    if use_llm:
        from .llm import apply_demos
        P, W, n_llm = apply_demos(P, M, train_ids, n_paths, weight=llm_weight, power=llm_sharpen)
        log(f"LLM persona answers: {n_llm} decision points, weight {llm_weight:g} each, sharpened x{llm_sharpen} (status, housing, spending, assets)")
    log(f"training on {len(X)} lives, testing on {len(Xv)} lives of unseen profiles ...")
    net, best = train_net(X, M, P, Xv, Mv, Pv, log=log, W=W)
    torch.save(net.state_dict(), C.ARTIFACTS / "persona_net.pt")
    heads = per_head_kl(net, Xv, Mv, Pv)

    # Whole lives: rule persona vs its clone on unseen profiles, same seeds.
    from ..explore import goal_met, simulate, yearly_action_rates
    compare = []
    for pid in ids[:5]:
        people = prof.loc[[pid]]
        persona = from_profile(people.iloc[0])
        a = simulate(world, people, persona, 1000, seed=5)
        b = simulate(world, people, persona, 1000, seed=5, policy_cls=NeuralPolicy)
        ra, rb = yearly_action_rates(a), yearly_action_rates(b)
        compare.append({"profile": pid, "status": people.iloc[0]["status"], "age": float(people.iloc[0]["age"]),
                        "high_goal_odds": [float(goal_met(a)[0].mean()), float(goal_met(b)[0].mean())],
                        "median_final_net_worth": [float(np.median(a.nw[:, -1])), float(np.median(b.nw[:, -1]))],
                        "status_changes_per_year": [ra["status change"], rb["status change"]],
                        "lifestyle_changes_per_year": [ra["lifestyle change"], rb["lifestyle change"]]})
    res = {"held_out_kl_total": best, "held_out_kl_by_head": heads, "lives_rule_vs_neural": compare,
           "llm_decision_points": n_llm, "llm_weight": llm_weight if use_llm else None,
           "llm_sharpen": llm_sharpen if use_llm else None,
           "train_lives": int(len(X)), "test_lives": int(len(Xv)), "seconds": round(time.time() - t0, 1)}
    (C.ARTIFACTS / "persona_net.json").write_text(json.dumps(res, indent=1))
    log(f"held-out KL by head: {({k: round(v, 4) for k, v in heads.items()})}")
    for r in compare:
        log(f"  {r['profile']} ({r['status']}, {r['age']:.0f}): high goal {r['high_goal_odds'][0]:.2f} rule vs "
            f"{r['high_goal_odds'][1]:.2f} neural | median NW {r['median_final_net_worth'][0]:,.0f} vs "
            f"{r['median_final_net_worth'][1]:,.0f} | status changes/yr {r['status_changes_per_year'][0]:.3f} vs "
            f"{r['status_changes_per_year'][1]:.3f}")
    log(f"saved {C.ARTIFACTS / 'persona_net.pt'} ({res['seconds']:.0f}s)")
    return res
