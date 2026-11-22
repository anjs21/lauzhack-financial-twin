# 8. What we tried, and why we kept what we have

**In short.** Most parts of the pipeline replaced something simpler that we built first. Each entry
says what we tried, what happened and what we kept. The evidence is in the files linked; nothing
here is from memory alone.

## The transition function

| We tried | What happened | We kept |
|---|---|---|
| **Replaying the recorded ledger** plus an intervention delta ([`analyze_mdp.py`](../analysis/analyze_mdp.py)) | Good for auditing and quick sensitivity (+CHF 500 a month prevents 14 of 61 shortfalls), but flows are fixed, nothing is uncertain and no action can change the path | As the data audit and first sensitivity check |
| **A deterministic causal model** of work vs master's for Alexia ([`causal-model/`](../causal-model/README.md)) | Clear mechanisms and an exact cash decomposition, but one profile, no uncertainty, and a fixed job-search delay | As a teaching example; its mechanism order (study → graduation → job → income → cash) is in the world step |
| **A hand-calibrated stochastic simulator** ([`action_importance.py`](../analysis/action_importance/action_importance.py)) | Recovers the ledger's rules and backtests well (61.2 vs 61 shortfalls) | As the **baseline** every learned model must beat, and as the action and state screen ([02](02-state-and-actions.md)) |
| **One network for the whole step** | Rejected before building: nobody switches in the data, so it would learn that actions do nothing, and it would add error to exact accounting | A network for the flow only, exact accounting around it |
| **The calendar year as a network input** | Rejected: ten-year rollouts would ask about years the network never saw | Real terms, with pay and price indices outside the network (a test enforces it) |
| **A flow network without work mode × month inputs** | Under-fits the rare spikes: students' fees and the December bonus (noted in [`flow.py`](../twin/world/flow.py); the numbers were not saved) | Mode × month inputs |
| **Fixed market rates**: 20% offer chance a month, admission certain, moves at once | Not from any source, and blind to who else is competing | Learned markets ([04](04-markets.md)); fixed rates remain as `--markets priors` for comparison |

## Markets

| We tried | What happened | We kept |
|---|---|---|
| A **neural network** on the ledger, a **diffusion model**, an **LLM as the transition function** | No transitions to train on; diffusion needs path data and its probabilities cannot be inspected; an LLM is not calibrated, not reproducible and slow | Small regression models on a simulated society |
| **Fitting on history only** (Markowitz-style) | Overfits the past | History only as calibration targets |
| **LinkedIn career paths** | Against the terms of service, privacy problems, and biased towards successes | Official SECO and BFS statistics |
| A market model of **one's own state only** | 7.5 points calibration error after a change in behaviour (the Lucas critique) | Own state **plus market tightness θ**: 1.1 points |
| Adding **the share of master's holders** to θ | No gain (1.1 points) | θ alone |
| Training on **200-person societies only** | Failed in 2,000-person societies (3.3 and 4.3 points for jobs and flats); small markets are lumpy, and the PhD share is 6.7% at 200 people vs 16.6% at 2,000 | Training on 200, 1,000 and 2,000 together |
| **Small LLMs as the society's agents** ([`bench_llm.py`](../analysis/society_sim/bench_llm.py), Qwen3 1.7B and 8B on one B200) | Fast enough (690–1,530 decisions a second) but nearly deterministic per persona, and acceptance of a low offer *rose* with savings instead of falling; the 8B model accepted every low offer | A utility-based softmax: transparent, tunable, responds to savings by construction |
| **Two adapters** for the markets in `twin/` (`markets.py` and `market.py`, built in parallel) | Same idea, duplicated | Merged into [`market.py`](../twin/world/market.py), with a test that it equals the reference model exactly |

## The persona

| We tried | What happened | We kept |
|---|---|---|
| **Monthly decisions** for spending and investing (work and housing were already yearly) | Two decision clocks in one model; the commit that changed it does not record a further reason | One clock: every voluntary choice at the **January review**. Lifestyle scores got +log 12 so the yearly rate of lifestyle changes stayed the same. |
| **A GRU clone** of the rule persona | Matches per decision (KL ≤ 0.002), but small differences on rare choices compound over ten years: one profile changed status 3× as often | Kept as an option (`--policy neural`) |
| **LLM demonstrations** (gpt-4o-mini, 300 reviews) mixed into the clone | LLMs hedge towards change, so answers needed explicit base rates in the prompt and sharpening; the labels went stale when the decision clock changed | A Qwen3-4B proxy on 60 January 2026 ledger snapshots is the neural default; the older labels need regenerating |
| **Foresight by policy gradient** over whole lives | One reward per ten-year life spread over ~480 sampled choices: it learned nothing measurable | A look-ahead tilt with a closed-form solution, distilled into the network |
| **Unanchored foresight** at f = 0 | Drifted from the rule persona (an unemployed person ran short in 38% of lives instead of 56%) | **Anchored** foresight: only the learned tilt is applied to the rule persona; the web UI's "deliberate" mode uses it |
| **LLM agents living whole lives** (Qwen3-8B, 500 lives × 120 months, about 23,000 decisions in 10 minutes on an L4; [`logs/`](../logs/)) | Runs, with 99.9% of probability on valid answers; the code and an analysis of the lives are not in the repository | Not part of the pipeline |

## Planning

| We tried | What happened | We kept |
|---|---|---|
| **Plans from the GRU persona** cloned from LLM decisions | It knows how people like this one decide, not what *this* person intends | The **planner agent**: the person's own plan on top of the habit model ([06](06-planner-agent.md)) |
| **One fixed intention per plan** (`--plan '{...}'`) | Cannot say "after the master's", "if I lose my job" or "retry until a flat is found" | Steps with windows, `after`, events and conditions |
| **Letting the agent rewrite the whole plan on a nudge** | A small model kept steps the person had just dropped (the master's) | Edits only: remove ids, change fields |
| **Trusting the LLM's output as written** | Repeated slips: no-op steps, `after: "promotion"` naming an event, `invest_surplus` used as an emergency fund, cash fallbacks that never fire | Validation, automatic repairs reported back, and a diagnosis of *when and why* money ran short |
| **Letting the agent sign at once** | It could sign a plan it had never seen simulated | Signing only after at least one simulation |
| **Counting a step as done when chosen** | A chosen move with no flat found, or a refused master's, looked carried out | Done only when the world applied it |

## Data choices we made

* **Opening students finish their course at the first review.** The data does not say how far along
  they are. They can choose a master's at that review.
* **Loans repay over 60 months.** The ledger has no repayments; a schedule is needed or debt is free.
* **Education is drawn from a prior when unknown.** The markets need it and the data does not have it.
* **The student credit limit stays at 3 × pay (floor CHF 5,000).** A student on CHF 1,200 a month
  against about CHF 2,700 of rent and living costs crosses it in the first year
  ([`twin_game_theory/README.md`](../twin_game_theory/README.md), section 5.5). This is still open
  ([09](09-limitations.md)).
