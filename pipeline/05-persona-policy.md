# 5. The persona: who makes the decisions

**In short.** The policy is a *behavioural twin* of the person: it chooses the way this person tends
to choose, at random, with memory of their own path. It is never trained to be optimal. The data
holds no decisions, so the policy is built from traits, the person's own answers and, since the
planner agent, their own plan ([06](06-planner-agent.md)). A neural copy, LLM demonstrations and a
"foresight" knob were built on top; the table at the end says which one runs where.

## The rule persona (the default)

[`twin/policy/rules.py`](../twin/policy/rules.py). At each January review, for each of the four
choices:

1. Build a score (logit) for every option from trait-weighted rules plus inertia.
2. Remove the options the masks forbid.
3. Divide by the person's temperature (higher = more random) and take a softmax.
4. **Sample** one option. Nothing is maximised.

**Eight traits**, each in [0, 1]: risk tolerance, frugality, career drive, wish for free time,
housing aspiration, inertia, impatience and goal focus.

* **Defaults come from the profile.** For example, risk tolerance is 0.3 + 0.5 × the share of
  liquid wealth that is invested; career drive is higher under 30 and for students and lower over 55;
  an "Education flexibility" goal raises career drive.
* **A questionnaire can override them.** One statement per trait, answered 1–5, blended in with
  weight 0.7 (for example "I save first and spend what is left" for frugality).
* **Each simulated life draws its own traits** around those values (logit-normal, sd 0.4). The
  spread in the results therefore includes not knowing the person exactly.

**What the rules look at.** Cash in months of outgoings, cash pressure (cash below two months),
comfort (more than six months of liquid savings), goal pressure (gap to the goal, more weight in the
last two years), readiness to retire, a move in the last two years, a job lost in the last six
months. Two examples:

* Start a master's: −3 + 4 × career drive − 0.15 per year over 28 − 2 × cash pressure.
* Move bigger: −3.5 + 2.5 × housing aspiration + 2 × comfort + (2.5 for a family, 1 for a couple,
  below a family flat) − 3 × cash pressure − 1.5 × goal pressure.

The person's **cash buffer** target is 6 − 4 × risk tolerance months of outgoings: 6 for the most
cautious, 2 for the boldest. The assets choice invests above it and sells to refill it.

**Stated intentions.** A person can say "start a master's in January 2026, 60% sure" or "move
somewhere cheaper if cash falls below two months". When such an intention is due and possible, it
overrides the sampled choice with its stated probability. The planner agent's plans compile into
exactly these intentions.

## Is the behaviour plausible?

There is no recorded decision to fit or test against, so we check rates and directions instead.
Rule persona, 240 random profiles × 50 lives each, re-run for these docs:

| Rate per person and year | All | Employed | Students | Unemployed | Assumed typical rate* |
|---|---|---|---|---|---|
| Changes work situation | 4.1% | 5.3% | 5.4% | 10.8% | 3–8% |
| Starts a master's | 0.7% | 0.8% | 2.6% | 0.5% | |
| Chooses a bigger or cheaper flat | 1.6% | 1.3% | 2.8% | 3.2% | 5–10% |
| Changes lifestyle level | 10.9% | 10.6% | 10.7% | 13.5% | 5–10% |
| Moves money between cash and investments | 78% | 80% | 69% | 70% | (a yearly rebalance) |

\* The base rates stated to the LLM persona ([`policy/llm.py`](../twin/policy/llm.py)); they are
assumptions, not data. Work changes are in range and lifestyle changes slightly above it. Housing
changes are well below it, and below the Swiss moving rate of 9.3% a year; part of the gap is that
the twin only models moves to another size of flat ([09](09-limitations.md)).

Tests check the directions: frugal personas upgrade their lifestyle less, intentions fire with their
stated probability, status and housing change only at decision points, masks are respected.

## Built on top of the rule persona

### Neural persona: a GRU clone

[`policy/net.py`](../twin/policy/net.py). One network for all personas: a 32-unit GRU reads the
monthly path (47 inputs: state, traits, temperature, decision clock) and outputs one probability per
option for each of the four choices. It is trained on the rule persona's **full probabilities**
(KL divergence, not sampled actions) from 1,920 lives and tested on 480 lives of profiles it never
saw.

* Held-out KL per decision: 0.0018 status, 0.0007 housing, 0.0008 spending, 0.0010 assets.
* Whole lives on five unseen profiles: goal odds within about 3 points of the rule persona, but status
  changes more often in some (0.08 a year against 0.02 for one 42-year-old employee). Small
  differences on rare choices compound over ten years.

### LLM persona demonstrations

[`policy/llm.py`](../twin/policy/llm.py). An LLM role-plays the person at decision points (situation,
goals and traits in words) and answers how *this person* would likely act, as probabilities. Only
synthetic profiles and simulated states are sent.

* 300 simulated reviews were labelled by gpt-4o-mini and used as targets with 10× weight in the
  clone. LLMs hedge, so housing and spending answers are sharpened (squared and renormalised).
* 60 real January 2026 ledger snapshots (cash history through January 2026) were labelled by
  Qwen3-4B-Instruct on vLLM. A GRU cloned from them (`jan_2026_proxy_gru.pt`, held-out KL 0.031 on 9
  profiles) is what `--policy neural` loads by default.

### Foresight: the same person, a little more deliberate

[`policy/foresight.py`](../twin/policy/foresight.py). A knob $f \in [0, 1]$ tilts the persona's own
probabilities towards options worth more to *this persona* over the next five years:

$$\pi_f(o) \propto \pi(o)\, \exp\!\big(f \cdot (U(o) - \max U) / \text{CHF } 10{,}000\big)$$

This is the exact solution of "maximise $f \cdot U - \mathrm{KL}(\pi_f \,\|\, \pi)$", so the person
stays random and close to their habits. $U(o)$ comes from simulating each option 60 months ahead in
the world model at 2,000 decision points, with the same luck for every option. $U$ uses the spec's
preference table: lifestyle, home, free time, months below zero, changes, goals met, net worth.

* Mean change in choice probabilities: 0.10 at f = 0.25, 0.14 at f = 0.5, 0.18 at f = 1.
* On held-out profiles, more foresight helps most where trouble is likely. A 67-year-old retiree's
  goal odds rise from 2% to 21% and the chance of running short falls from 99% to 81% (f = 1). An
  unemployed 34-year-old's chance of running short falls from 56% to 8%.
* The **anchored** version used by the web UI applies only the learned tilt to the rule persona's
  own probabilities: $\pi_{\text{rules}}(o) \cdot \exp(\text{net}_f(o) - \text{net}_0(o))$. At f = 0 it
  is exactly the rule persona. The unanchored network had drifted from the rules even at f = 0 (an
  unemployed person's chance of running short was 38% instead of 56%).

The clone and foresight were trained before the market models were added, and the gpt-4o-mini labels
describe an older decision clock. They need retraining before their numbers are quoted.

## Which policy runs where

| Where | Policy |
|---|---|
| `python -m twin explore` (default) | Rule persona |
| `--policy neural` | GRU from the January 2026 Qwen labels |
| `--policy foresight --foresight 0.5` | Foresight network at that level |
| Web UI, "habitual" | Rule persona |
| Web UI, "deliberate" | Anchored foresight, f = 0.5 |
| With a plan (CLI or web) | Any of the above, with the plan's steps on top ([06](06-planner-agent.md)) |

Reproduce: `python -m twin clone` (~2 min), `python -m twin foresight` (~5 min),
`python -m twin clone-snapshots`.
