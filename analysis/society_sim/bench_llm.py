#!/usr/bin/env python3
"""How much does a language-model agent cost, and does a small model choose sensibly?

Runs on one GPU with vLLM. Two decision modes:
  choice  : prompt -> 1 token; read the probability of each option letter from the logprobs.
            One forward pass gives the whole policy pi(a|s); no text is generated.
  reason  : prompt -> answer letter + one-sentence reason (up to 48 tokens), for narrated paths.

Sanity probes (choice mode):
  runway  : same persona and offer, cash runway 0.5..12 months -> P(accept a low offer) should fall.
  persona : 'driven' vs 'relaxed' persona on the same states -> P(search hard) should differ.

Usage: python bench_llm.py MODEL [N_PROMPTS]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from vllm import LLM, SamplingParams

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
OUT = Path(__file__).resolve().parent / "results"
rng = np.random.default_rng(0)

PERSONAS = {
    "driven": "You are ambitious and career-focused. You dislike being without work and will put in a lot of effort to get ahead.",
    "relaxed": "You value free time and low stress. You are patient about your career and do not like to rush decisions.",
    "anxious": "You worry a lot about money and want security above everything else.",
    "balanced": "You try to balance career, money and personal life sensibly.",
}
SYSTEM = ("You are role-playing a fictional person living in Switzerland. Stay in character. "
          "Choose the option this person would most likely pick. Answer with a single letter.")


def search_prompt(persona, age, edu, months_searching, runway, offers_last_month):
    return (f"{PERSONAS[persona]}\n\nYour situation: you are {age}, you hold a {edu} degree and live in Lausanne. "
            f"You have been looking for a job for {months_searching} months. Your savings cover {runway} months of costs. "
            f"Last month {offers_last_month}% of job seekers you know got an offer.\n\n"
            "This month, what do you do?\nA) Apply to many jobs (high effort)\nB) Apply to a few jobs (low effort)\n"
            "C) Stop searching and start a two-year master's\nD) Take a break from searching this month\n\nAnswer:")


def offer_prompt(persona, months_searching, runway, offer, target):
    return (f"{PERSONAS[persona]}\n\nYour situation: you are 23, you hold a bachelor's degree and live in Lausanne. "
            f"You have been looking for a job for {months_searching} months. Your savings cover {runway} months of costs. "
            f"You hoped for about CHF {target:,} a month after tax.\n\n"
            f"You received a job offer paying CHF {offer:,} a month after tax.\n"
            "A) Accept the offer\nB) Reject it and keep searching\n\nAnswer:")


def main():
    t0 = time.time()
    # Triton attention avoids FlashInfer's JIT build, whose headers clash with the pip CUDA toolkit
    llm = LLM(model=MODEL, gpu_memory_utilization=0.85, max_model_len=1024, enable_prefix_caching=True, seed=0,
              attention_backend="TRITON_ATTN")
    load_s = time.time() - t0
    tok = llm.get_tokenizer()
    letters = ["A", "B", "C", "D"]
    ids = [tok.encode(x, add_special_tokens=False)[0] for x in letters]

    def chat(user):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def choose(prompts, k):
        outs = llm.generate([chat(p) for p in prompts], SamplingParams(max_tokens=1, logprobs=20, temperature=1.0),
                            use_tqdm=False)
        P, valid = np.zeros((len(prompts), k)), np.zeros(len(prompts))
        for i, o in enumerate(outs):
            lp = o.outputs[0].logprobs[0]
            v = np.array([np.exp(lp[j].logprob) if j in lp else 0.0 for j in ids[:k]])
            valid[i] = v.sum()
            P[i] = v / max(v.sum(), 1e-12)
        return P, valid

    # --- throughput, choice mode
    states = [dict(persona=rng.choice(list(PERSONAS)), age=int(rng.integers(21, 30)),
                   edu=rng.choice(["bachelor's", "master's"]), months_searching=int(rng.integers(0, 18)),
                   runway=float(rng.choice([0.5, 1, 2, 3, 6, 12])), offers_last_month=int(rng.integers(5, 40)))
              for _ in range(N)]
    prompts = [search_prompt(**s) for s in states]
    n_tok = np.mean([len(tok.encode(chat(p))) for p in prompts[:200]])
    choose(prompts[:64], 4)  # warm-up
    t = time.time()
    P, valid = choose(prompts, 4)
    choice_s = time.time() - t

    # --- throughput, reason mode
    M = min(N, 1024)
    t = time.time()
    outs = llm.generate([chat(p.replace("Answer:", "Answer with the letter, then one short sentence explaining why.")) for p in prompts[:M]],
                        SamplingParams(max_tokens=48, temperature=0.7), use_tqdm=False)
    reason_s = time.time() - t
    reason_ok = np.mean([o.outputs[0].text.strip()[:1] in letters for o in outs])

    # --- sanity probe 1: cash runway vs accepting a low offer
    runways = [0.5, 1, 2, 3, 6, 12]
    probe = {}
    for persona in PERSONAS:
        pr = [offer_prompt(persona, m, r, 4600, 5500) for m in (1, 6) for r in runways]
        pa, _ = choose(pr, 2)
        probe[persona] = {f"searching_{m}m": dict(zip(map(str, runways), np.round(pa[i * 6:(i + 1) * 6, 0], 3).tolist()))
                          for i, m in enumerate((1, 6))}
    mono = []
    for persona in probe:
        for key, row in probe[persona].items():
            v = list(row.values())
            mono.append(float(np.corrcoef(np.argsort(np.argsort(v)), np.arange(len(v)))[0, 1]))

    # --- sanity probe 2: persona effect on high-effort search, same states
    base = [dict(age=24, edu="bachelor's", months_searching=int(m), runway=float(r), offers_last_month=20)
            for m in (1, 4, 8, 12) for r in (1, 3, 6, 12)]
    eff = {p: float(choose([search_prompt(persona=p, **s) for s in base], 4)[0][:, 0].mean()) for p in PERSONAS}

    res = dict(
        model=MODEL, gpu_load_seconds=round(load_s, 1), prompt_tokens_mean=round(float(n_tok)),
        choice_mode=dict(prompts=N, seconds=round(choice_s, 2), decisions_per_s=round(N / choice_s),
                         mean_prob_mass_on_valid_letters=round(float(valid.mean()), 3),
                         mean_policy=np.round(P.mean(0), 3).tolist()),
        reason_mode=dict(prompts=M, seconds=round(reason_s, 2), decisions_per_s=round(M / reason_s),
                         answer_parse_rate=round(float(reason_ok), 3), example=outs[0].outputs[0].text.strip()[:200]),
        probe_runway_P_accept_low_offer=probe,
        probe_runway_rank_corr_mean=round(float(np.mean(mono)), 3),
        probe_runway_note="rank correlation of P(accept) with runway; sensible behaviour is negative",
        probe_persona_P_high_effort=dict((k, round(v, 3)) for k, v in eff.items()),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"bench_{MODEL.split('/')[-1]}.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
