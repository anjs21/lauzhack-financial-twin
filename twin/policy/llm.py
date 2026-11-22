"""LLM persona demonstrations: an LLM (OpenAI API) role-plays the person at decision points.

    export OPENAI_API_KEY=sk-...                 # read from the environment, never stored
    export OPENAI_MODEL=gpt-4o-mini              # optional; or --model
    python -m twin llm-demos --calls 300         # 300 simulated reviews plus 60 ledger snapshots
    python -m twin llm-demos --dry-run           # print one prompt, no API call
    python -m twin clone --llm                   # rule targets everywhere, LLM answers where asked

The LLM is asked how *this person* would most likely act (habits included), not what is
best, and answers with a probability per option. Its answers replace the rule persona's
targets at those decision points when the neural persona is cloned, with extra weight, so
the network learns common-sense reactions the hand-written rules do not have. Only the
synthetic profile and simulated state are sent; nothing about the user.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import config as C
from ..data import indices, ledger, profiles
from .net import Recorder, SIZES, teacher_data
from .rules import HEADS
from .traits import QUESTIONS, TRAITS, from_profile

DEMOS = C.ARTIFACTS / "llm_demos.json"
CACHE = C.ARTIFACTS / "llm_cache.jsonl"
DEFAULT_MODEL = "gpt-4o-mini"
JAN_2026 = C.month_index("2026-01")
DEFAULT_JAN_2026_CALLS = 60
OFFSET = [0, sum(SIZES[:1]), sum(SIZES[:2]), sum(SIZES[:3])]

SYSTEM = ("You role-play one specific person deciding about their own work, home, spending and savings. "
          "You are not a financial adviser: say how this person would most likely act this month, including "
          "imperfect habits, inertia and emotions. Answer only with JSON.")
# Assumed base rates (not from the data), stated so the LLM does not hedge toward change.
BASE_RATES = (
    "Calibration: real people rarely change anything at a review. For a typical person, the chance of acting at one "
    "yearly review is about: changing work situation 3-8%, moving home 5-10%, changing lifestyle level 5-10%; the "
    "rest is keeping things as they are. Go clearly above these only if this person's situation or character pushes "
    "hard (cash running out, a job just lost, a goal deadline close, a strong trait). Do not spread probability evenly "
    "to look balanced: if you think they would keep things as they are, put 0.9 or more on that option.")

OPTION_TEXT = {
    "status": {0: "continue as now", 1: "start a full-time master's (24 months, pays tuition, small student income)",
               2: "accept part-time work", 3: "accept full-time work", 4: "leave paid work (quit, break or retire)"},
    "housing": {-1: "move one step cheaper (smaller flat or flatshare), costs CHF 2,000 to move", 0: "stay",
                1: "move one step bigger, costs CHF 2,000 to move"},
    "spending": {-1: "cut lifestyle (about 10% less living costs)", 0: "keep lifestyle",
                 1: "raise lifestyle (about 10% more living costs)"},
    "assets": {-1: "sell investments to refill the cash buffer", 0: "leave savings as they are",
               1: "invest the cash above the buffer"},
}
MODE_TEXT = {C.STUDENT: "a full-time student", C.PART_TIME: "working part-time", C.FULL_TIME: "working full-time",
             C.NOT_WORKING: "not working"}
TIER_TEXT = ["a shared flat", "a small flat", "a family-size flat"]
LIFESTYLE_TEXT = ["lean", "standard", "comfortable"]


def level(v):
    return ("strongly agree" if v >= 0.75 else "agree" if v >= 0.55 else "slightly agree" if v >= 0.35
            else "disagree")


def chf(x):
    return f"CHF {x:,.0f}".replace(",", "'")


def option_value(head, idx):
    return idx if head == "status" else idx - 1


def prompt_for(ctx, i, k):
    """Prompt for life i at step k of the teacher data in ctx (see net.teacher_data)."""
    st = ctx["states"][k]
    row = ctx["people"].iloc[int(st["person"][i])]
    tr = ctx["traits"][i]
    t = int(st["t"])
    out = max(float(st["outgoings"][i]), 1.0)
    mode = int(st["mode"][i])
    work = MODE_TEXT[mode] + (" (retired, on a pension)" if mode == C.NOT_WORKING and st["pensioner"][i] else "")
    if st["search"][i]:
        work += f", looking for {'part-time' if st['search'][i] == 1 else 'full-time'} work"
    nw = st["cash"][i] + st["inv"][i] - st["debt"][i]
    goals = [f"High-priority goal: {row['high_goal']} of {chf(row['high_target'])} by December 2029 "
             f"(now {chf(st['cash'][i] if row['high_goal'] == 'Emergency savings' else nw)})."]
    if isinstance(row["med_goal"], str):
        goals.append(f"Medium goal: {row['med_goal']} of {chf(row['med_target'])} by December 2032 on top of it.")
    recent = [w for flag, w in ((st["recent_loss"][i], "you lost your job in the last six months"),
                                (st["recent_move"][i], "you moved in the last two years"),
                                (st["child_cost"][i], "you had a child recently")) if flag]
    lines = [
        f"You are {row['name']}, {st['age'][i]:.0f}, living in {row['city']} in {TIER_TEXT[int(st['tier'][i])]}, "
        f"{['single', 'in a couple', 'in a family with children'][int(st['household'][i])]}. It is {C.month_label(t)}.",
        f"Work: {work}, " + (f"for {int(st['tau'][i])} months so far" if t > 0 else "since before 2025")
        + f". Take-home income {chf(st['income'][i])} a month."
        + (" You already have (or are doing) a master's." if st["degree"][i] else ""),
        f"Money: cash {chf(st['cash'][i])} ({st['cash'][i] / out:.1f} months of outgoings), investments "
        f"{chf(st['inv'][i])}, debt {chf(st['debt'][i])}. Monthly outgoings {chf(out)} including rent "
        f"{chf(st['rent'][i])}; lifestyle is {LIFESTYLE_TEXT[int(st['lifestyle'][i])]}. "
        f"Your cash buffer target is {st['buffer_months'][i]:.1f} months.",
        f"Economy: {C.ECONOMY[int(st['econ'][i])]}." + (f" Recently {'; '.join(recent)}." if recent else ""),
        *goals,
        "How you are: " + "; ".join(f"you {level(v)} that \"{QUESTIONS[name]}\"" for name, v in zip(TRAITS, tr)) + ".",
        "This is one of your decision points (annual January review). Options you have:",
    ]
    if "cash_flow" in ctx:
        lines.insert(-1, f"Your average monthly net cash flow in 2025 was {chf(ctx['cash_flow'][i])}. "
                     "Work, household, investments and debt use the opening profile assumptions.")
    for h in HEADS:
        valid = [j for j in range(len(OPTION_TEXT[h])) if st["valid"][h][i, j]]
        lines.append(f"- {h}: " + "; ".join(f"\"{option_value(h, j)}\" = {OPTION_TEXT[h][option_value(h, j)]}"
                                            for j in valid))
    lines.append(BASE_RATES)
    lines.append('Reply as JSON: {"status": {"<option>": probability, ...}, "housing": {...}, "spending": {...}, '
                 '"assets": {...}, "why": "one short sentence"}. Probabilities per group sum to 1; '
                 "use only the options listed.")
    return "\n".join(lines)


def parse(answer: dict, valid: dict):
    """LLM JSON -> 14 probabilities (status 5, housing 3, spending 3, assets 3), or None if unusable."""
    out = np.zeros(sum(SIZES), np.float32)
    for h, off in zip(HEADS, OFFSET):
        got = answer.get(h)
        if not isinstance(got, dict):
            return None
        p = np.zeros(len(valid[h]), float)
        for key, val in got.items():
            try:
                j = int(str(key).strip().split()[0]) + (0 if h == "status" else 1)
                p[j] = max(float(val), 0.0)
            except (ValueError, IndexError):
                continue
        p = p * valid[h]
        if p.sum() <= 0:
            return None
        p = p / p.sum()
        p = (p + 2e-3 * valid[h]) / (1 + 2e-3 * valid[h].sum())     # keep every valid option possible
        out[off:off + len(p)] = p
    return out


class Client:
    """Thin wrapper over the OpenAI SDK with a disk cache keyed by (model, prompt)."""

    def __init__(self, model=None, cache=CACHE, sdk_client=None):
        self.model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        self.cache_path = cache
        self.cache = {}
        if cache.exists():
            for line in cache.read_text().splitlines():
                d = json.loads(line)
                self.cache[d["key"]] = d["answer"]
        self._sdk = sdk_client
        self.calls = 0
        self.lock = threading.Lock()

    @property
    def sdk(self):
        if self._sdk is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise SystemExit("Set OPENAI_API_KEY in your environment first (export OPENAI_API_KEY=...).")
            from openai import OpenAI
            self._sdk = OpenAI(max_retries=5, timeout=60)
        return self._sdk

    def key(self, prompt):
        return hashlib.sha256(f"{self.model}\n{SYSTEM}\n{prompt}".encode()).hexdigest()

    def ask(self, prompt):
        k = self.key(prompt)
        if k in self.cache:
            return self.cache[k]
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        try:
            try:
                r = self.sdk.chat.completions.create(model=self.model, messages=msgs, temperature=0.2,
                                                     response_format={"type": "json_object"})
            except Exception as e:          # some models only accept the default temperature
                if "temperature" not in str(e):
                    raise
                r = self.sdk.chat.completions.create(model=self.model, messages=msgs,
                                                     response_format={"type": "json_object"})
        except SystemExit:
            raise
        except Exception as e:
            raise SystemExit(explain(e, self.model)) from e
        with self.lock:
            self.calls += 1
        try:
            answer = json.loads(r.choices[0].message.content)
        except (json.JSONDecodeError, TypeError):
            answer = {}
        with self.lock:
            self.cache[k] = answer
            with self.cache_path.open("a") as f:
                f.write(json.dumps({"key": k, "answer": answer}) + "\n")
        return answer


def explain(e, model):
    """Turn an OpenAI error into a message that says what to do."""
    name, text = type(e).__name__, str(e)
    if name == "AuthenticationError" or "401" in text:
        return "OpenAI rejected the key (401). Check OPENAI_API_KEY in the same terminal: echo ${OPENAI_API_KEY:0:7}"
    if "insufficient_quota" in text:
        return "The OpenAI account behind this key has no credit or quota left (429 insufficient_quota)."
    if name == "NotFoundError" or "model_not_found" in text or "does not exist" in text:
        return f"Model '{model}' is not available to this key. Pick another: export OPENAI_MODEL=gpt-4o-mini (or --model)."
    if name in ("APIConnectionError", "APITimeoutError"):
        return f"Could not reach the OpenAI API ({name}). Check the network or a proxy / OPENAI_BASE_URL."
    if name == "RateLimitError":
        return "OpenAI rate limit hit even after retries. Rerun with fewer workers: --workers 2 (answers so far are cached)."
    return f"OpenAI call failed: {name}: {text[:300]}"


def pick_points(ctx, n, seed=C.SEED):
    """Decision points to ask about; month 0 (everyone's first review) is capped at 20%."""
    pts = [(i, k) for k, st in enumerate(ctx["states"]) for i in np.flatnonzero(st["decide"])]
    rng = np.random.default_rng(seed)
    first = [p for p in pts if p[1] == 0]
    later = [p for p in pts if p[1] > 0]
    n0 = min(len(first), n // 5)
    take = [first[j] for j in rng.choice(len(first), n0, replace=False)] if n0 else []
    take += [later[j] for j in rng.choice(len(later), min(len(later), n - n0), replace=False)]
    return take


def january_snapshots(world, train_ids, count):
    """Build ledger-backed histories through the opening of January 2026."""
    from ..explore import goals_per_path
    from ..world.env import Streams

    history = ledger().loc[lambda df: df["t"].between(0, JAN_2026)]
    cash = history.pivot(index="profile_id", columns="t", values="opening_cash_chf")
    cash = cash.reindex(index=train_ids, columns=range(JAN_2026 + 1)).dropna()
    if cash.empty:
        raise SystemExit("No training profiles have complete cash histories through January 2026.")
    rng = np.random.default_rng(C.SEED)
    ids = rng.choice(cash.index, min(count, len(cash)), replace=False)
    people = profiles().loc[ids]
    traits = np.stack([from_profile(row).vector() for _, row in people.iterrows()])
    streams = Streams(C.SEED, len(people))
    state = world.initial_state(people, 1, streams, buffer_months=Recorder.buffer_months(traits))
    policy = Recorder(world, traits, 1.0, goals_per_path(people, 1))
    policy.reset(state, streams)
    policy.log, policy.states = [], []
    policy.keep_states, policy.keep_full = True, False
    for t, opening_cash in enumerate(cash.loc[ids].to_numpy().T):
        state.t = t
        state.age = people["age"].to_numpy() + t / 12
        state.tau.fill(t)
        state.G, state.P = (np.full(state.n, index) for index in indices(t, world.pr))
        state.income = people["income0"].to_numpy() * state.G
        state.rent = people["rent0"].to_numpy() * state.P
        state.cash = opening_cash
        policy.act(state, streams)
    features = np.stack([x for x, _, _ in policy.log], axis=1)
    cash_flow = history.loc[history["t"] < JAN_2026].groupby("profile_id")["net_transactions_chf"].mean()
    ctx = {"states": policy.states, "people": people, "traits": traits, "cash_flow": cash_flow.loc[ids].to_numpy()}
    return features, ctx


def collect(calls=300, n_profiles=400, n_paths=6, model=None, workers=8, client=None, log=print, dry_run=False,
            out=None, jan_2026_calls=DEFAULT_JAN_2026_CALLS):
    """Label simulated decisions and January 2026 ledger snapshots from training profiles."""
    from ..world.env import World
    from .net import split_ids

    if min(calls, jan_2026_calls) < 0 or calls + jan_2026_calls == 0:
        raise SystemExit("Use nonnegative demo counts with at least one decision point.")
    train_ids, _ = split_ids(n_profiles)
    world = World.load()
    records = []
    if jan_2026_calls:
        features, ctx = january_snapshots(world, train_ids, 1 if dry_run else jan_2026_calls)
        for i, pid in enumerate(ctx["people"].index):
            records.append((ctx, i, JAN_2026, {
                "profile_id": pid, "month": JAN_2026, "features": features[i].tolist(),
            }, None))
    if calls and not (dry_run and records):
        log(f"rule-persona lives for {len(train_ids)} training profiles x {n_paths} ...")
        _, _, P, ctx = teacher_data(world, train_ids, n_paths, keep_states=True)
        records.extend((ctx, i, k, {"life": int(i), "month": k}, P[i, k]) for i, k in pick_points(ctx, calls))
    if dry_run:
        ctx, i, k, _, _ = records[0]
        print(prompt_for(ctx, i, k))
        return None
    client = client or Client(model)
    prompts = [prompt_for(ctx, i, k) for ctx, i, k, _, _ in records]
    cached = sum(client.key(p) in client.cache for p in prompts)
    log(f"asking {client.model} at {len(records)} decision points ({cached} cached, "
        f"about {(len(prompts) - cached) * 900 / 1e6:.2f}M tokens) ...")
    t0 = time.time()
    first = next((p for p in prompts if client.key(p) not in client.cache), None)
    if first is not None:                  # one call up front: a bad key or model fails fast and clearly
        client.ask(first)
        log("first answer received; asking the rest ...")
    with ThreadPoolExecutor(workers) as ex:
        answers = list(ex.map(client.ask, prompts))
    demos, snapshots, bad = [], [], 0
    for (ctx, i, k, record, rule_probs), ans in zip(records, answers):
        valid = {h: ctx["states"][k]["valid"][h][i] for h in HEADS}
        p = parse(ans, valid)
        if p is None:
            bad += 1
            continue
        record.update(probs=p.round(5).tolist(), why=str(ans.get("why", ""))[:300])
        if rule_probs is None:
            record["mask"] = np.concatenate([valid[h] for h in HEADS]).tolist()
            snapshots.append(record)
        else:
            record["rule_probs"] = rule_probs.round(5).tolist()
            demos.append(record)
    meta = {"decision_clock": "annual_january_v1", "model": client.model, "n_profiles": n_profiles,
            "n_paths": n_paths, "seed": C.SEED,
            "train_ids_hash": hashlib.sha256(",".join(train_ids).encode()).hexdigest()[:16],
            "asked": len(records), "usable": len(demos) + len(snapshots), "unusable": bad,
            "jan_2026_usable": len(snapshots), "new_calls": client.calls, "seconds": round(time.time() - t0, 1)}
    out = out or DEMOS
    out.write_text(json.dumps({"meta": meta, "demos": demos, "jan_2026_snapshots": snapshots}, indent=1))
    log(f"{len(demos)} simulated and {len(snapshots)} ledger answers ({bad} unusable) "
        f"from {client.calls} new calls -> {out}")
    if demos:
        llm = change_rates([d["probs"] for d in demos])
        sharp = change_rates([sharpen(d["probs"], 2.0) for d in demos])
        rules = change_rates([d["rule_probs"] for d in demos])
        log("chance of acting at simulated decision points (LLM / LLM sharpened x2 / rule persona):")
        for h in HEADS:
            log(f"  {h:9s} {llm[h]:.3f} / {sharp[h]:.3f} / {rules[h]:.3f}")
    return meta


def sharpen(p14, power):
    """Raise each head's probabilities to `power` and renormalise: LLMs hedge, this undoes some of it.

    power: one number, or one per head (status, housing, spending, assets)."""
    out = np.array(p14, np.float64)
    powers = np.broadcast_to(np.asarray(power, float), (len(SIZES),))
    for off, n, pw in zip(OFFSET, SIZES, powers):
        q = out[off:off + n] ** pw
        out[off:off + n] = q / q.sum() if q.sum() > 0 else out[off:off + n]
    return out.astype(np.float32)


def change_rates(probs):
    """Per head: probability of doing something (not continue / stay / keep / hold)."""
    probs = np.asarray(probs)
    keep = {"status": 0, "housing": 6, "spending": 9, "assets": 12}
    return {h: float(1 - probs[:, keep[h]].mean()) for h in HEADS}


def apply_demos(P, M, train_ids, n_paths, weight=10.0, path=DEMOS, power=(1.0, 2.0, 2.0, 1.0)):
    """Put saved LLM answers (sharpened) into the clone's targets; returns (P, W, count)."""
    d = json.loads(path.read_text())
    meta = d["meta"]
    if meta.get("decision_clock") != "annual_january_v1":
        raise SystemExit("The saved LLM labels describe trajectories from an older decision clock. Regenerate labels before cloning with --llm.")
    if meta["n_paths"] != n_paths or meta["train_ids_hash"] != hashlib.sha256(",".join(train_ids).encode()).hexdigest()[:16]:
        raise SystemExit("llm_demos.json was made for other teacher lives; rerun `python -m twin llm-demos`.")
    P0, P = P, P.copy()
    W = np.ones(P.shape[:2], np.float32)
    used = 0
    for demo in d["demos"]:
        i, k = demo["life"], demo["month"]
        # The answer belongs to one simulated situation; if the teacher lives changed (new rules,
        # new decision months), the rule targets there differ and the answer no longer fits.
        if not np.allclose(P0[i, k], demo["rule_probs"], atol=1e-3):
            continue
        P[i, k] = sharpen(demo["probs"], power) * M[i, k]
        W[i, k] = weight
        used += 1
    if used < len(d["demos"]):
        print(f"skipped {len(d['demos']) - used} of {len(d['demos'])} LLM answers whose situations changed; "
              "rerun `python -m twin llm-demos` to ask about the current decision points")
    return P, W, used


def append_january_demos(X, M, P, W, weight, power, path=DEMOS):
    """Add ledger histories with supervision only at the January 2026 review."""
    snapshots = json.loads(path.read_text()).get("jan_2026_snapshots", [])
    if not snapshots:
        return X, M, P, W, 0

    n = len(snapshots)
    months, inputs = X.shape[1:]
    features = np.zeros((n, months, inputs), dtype=X.dtype)
    masks = np.ones((n, months, M.shape[-1]), dtype=M.dtype)
    probs = np.zeros((n, months, P.shape[-1]), dtype=P.dtype)
    weights = np.zeros((n, months), dtype=W.dtype)
    for i, demo in enumerate(snapshots):
        k = demo["month"]
        features[i, :k + 1] = demo["features"]
        masks[i, k] = demo["mask"]
        probs[i, k] = sharpen(demo["probs"], power) * masks[i, k]
        weights[i, k] = weight
    return (np.concatenate([X, features]), np.concatenate([M, masks]), np.concatenate([P, probs]),
            np.concatenate([W, weights]), n)
