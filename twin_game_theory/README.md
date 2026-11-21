# Learning the transition function P(s' | s, a) from a simulated Swiss society

The Financial Twin models a life as a Markov decision process (MDP): each month,

$$x_{t+1} \sim P(\,\cdot \mid x_t, u_t), \qquad u_t = \pi_{\text{twin}}(x_t).$$

The team split the work four ways: the state and actions (the MDP spec), the policy (a behavioural twin of the user, not trained), the **transition function P**, and the presentation. This folder is the transition function: what we tried, what we built, how well it works, and what is still wrong.

**Quick start**

```bash
python -m twin_game_theory.world.test_transition    # 7 checks, about 30 s
python -m twin_game_theory.world.build              # rebuild world_model.json, about 2-3 min
```

Then open [`world/transition_tutorial.ipynb`](world/transition_tutorial.ipynb) with the kernel "Python (lauzhack-sim)" and run all (about 45 s).

```python
from twin_game_theory.world.transition import TransitionModel, State, Action, alexia
m = TransitionModel()                          # loads world/world_model.json
s = alexia()                                   # the challenge's example person
m.allowed(s)                                   # which actions are allowed in s
m.outcome_probs(s, Action("plus"), top=5)      # exact P(next discrete situation | s, a)
s2, ledger = m.sample(s, Action("plus"), rng)  # one random next month, with cash flows
```

---

## 1. The problem: we have no transition data

The challenge data are monthly cash flows of anonymous people. They tell us what money looks like in a given situation: pay, rent, living costs, bills, bonuses. They never show a **transition**: nobody is seen getting a job, being admitted to a master's, or finding a flat. So the most important parts of P cannot be read off the data:

| What P needs | In the data? |
|---|---|
| Pay, rent, living costs, bills, bonus, noise | Yes, fitted on the ledger |
| P(job offer \| degree, time out of work, economy, ...) | No |
| P(admitted to a master's or PhD \| grades, ...) | No |
| P(getting a bigger or smaller flat \| income, ...) | No |
| Job loss, promotion, partner, child, economy | No, priors from the MDP spec |

## 2. Ideas we considered and dropped

| Idea | Why we dropped it |
|---|---|
| Neural network trained on the ledger | Nothing to train on: no transitions are observed. |
| Fit on history only (Markowitz-style) | Overfits the past, the same failure as Markowitz portfolios. We use history only as calibration targets. |
| Black–Litterman-style blend of data and beliefs | Good instinct, and in a sense what we do: Swiss statistics are the "market", our assumptions are the "views", and the simulator combines them. |
| Diffusion model | Generates realistic-looking paths, but needs path data we don't have, and its probabilities can't be inspected. |
| An LLM as the transition function | Not a calibrated probability, not reproducible, and slow. |
| LinkedIn profiles | Scraping breaks the terms of service and raises privacy issues; career paths are also biased towards successes. |
| Every agent a small LLM | Benchmarked on the cluster (section 6). Fast enough, but the choices were nearly deterministic and ignored savings, so we used a utility-based persona policy instead. |

## 3. The key insight: one person's chance depends on everyone else

A job offer depends on who else is applying. So the true transition is

$$P(s_i' \mid s_i, a_i, \underbrace{s_{-i}, a_{-i}}_{\text{everyone else}})$$

and not $P(s_i' \mid s_i, a_i)$. [`analysis/society_sim/MATH.md`](../analysis/society_sim/MATH.md) shows this with three people and two openings. With the same Alexia, her chance of an offer is 9/16. It is 15/32 if Chloé applies twice, and 7/9 with three openings.

**Mean field.** "Everyone else" can be summarised by one number, the market **tightness** θ = openings per application. Given θ, each person's chance depends only on their own state. Leaving θ out breaks the model as soon as behaviour changes, which is the Lucas critique.

The toy experiment ([`analysis/society_sim/toy_market.py`](../analysis/society_sim/toy_market.py), results in `results/toy_market.json`) tested this. We trained on 100-person worlds and tested on 10× larger worlds and on a different behaviour mix:

| Model | Calibration error after a behaviour change (pts) |
|---|---|
| Constant rate | 2.1 |
| Own state only | **7.5** (breaks) |
| Own state + tightness θ | 1.1 |
| + share of master's holders | 1.1 (adds nothing) |

The coefficients on log θ and log(applications) came out close to 1, as the theory of independent applications predicts.

## 4. What we built

### 4.1 A simulated society ([`world/society.py`](world/society.py))

Agents live month by month in the MDP and compete in **three markets with one law**: limited places, applicants ranked, and the best wins.

| Market | Places | Ranking |
|---|---|---|
| Jobs | Openings each month ∝ labour force, × 0.85 in inflation, × 0.5 in recession | Degree, grades, time out of work, pay level, noise |
| Study places | Master's: ample. PhD: about 0.012 a year per person, the scarce one | Grades |
| Flats | 9.3% re-let a year (Swiss moving rate) | Landlord rule: rent ≤ income / 3, or 12 months of rent in savings, or a student |

**How agents decide** (the population's policy, *not* the user's twin): a softmax over a utility with five parts: money, safety (weighted more when savings are low), leisure, career and inertia. There are four archetypes, driven, anxious, relaxed and balanced, each varied ±20%, plus a few random explorers. They only generate realistic market traffic. The user's twin plugs in separately (section 7).

### 4.2 Changes to the MDP spec

The original actions were status (5 options), housing, spending and assets. To make the markets work we added:

- **State:** `prog` (programme while studying), `edu` (highest degree), `u` (months without work), `pend` (a pending application or offer), `offer_k` (the offered pay level), `leave`, price indices and `breach` (credit limit hit). Context fields: city, grades, tuition, applications a month, loan.
- **Status actions:** stop / minus / stay / plus / **apply_master** / **apply_phd** / **accept**. "plus" while not working means *search*; "accept" exists only with an offer pending.

### 4.3 Calibrated to Swiss statistics ([`world/swiss_targets.json`](world/swiss_targets.json))

Each target has its source (SECO, BFS, SNSF) and a URL. We tune five knobs, by coordinate search with common random numbers, until the world matches the targets in 1,000-person societies:

| Knob | Value | Meaning |
|---|---|---|
| openings | 0.0129 | job openings a month per labour-force member |
| b_master | 16 | preference for a master's |
| phd_seats | 0.012 | PhD positions a year per person |
| b_phd | 1.0 | preference for a PhD |
| flat_turnover | 0.0078 | flats re-let a month (Swiss statistic, not tuned) |

### 4.4 Learning P from the simulation ([`world/build.py`](world/build.py))

1. **Simulate** societies of 200, 1,000 and 2,000 people (90,000 people in all) for 120 months each. Mixing sizes stops the model from learning one market size (section 5.3).
2. **Log** every market attempt: 290k job-search months, 27k study applications, 145k flat applications.
3. **Fit** three small statistical models by maximum likelihood (L-BFGS with exact gradients):
   - **job offer:** complementary log-log on degree, grades, time out of work, pay level, log(applications), log θ, economy, student;
   - **admission:** logit on programme, grades, log θ;
   - **flat:** logit on target size, downsizing, log(income / rent), log θ.
4. **Remove θ for the user.** The twin doesn't see the market's tightness, so P averages over the θ values the world produced in that economy.
5. **Uncertainty:** 5 bootstrap refits, resampling whole societies within each size. `draw=i` uses refit *i*.

### 4.5 The transition function ([`world/transition.py`](world/transition.py))

`TransitionModel` applies one month in fixed stages: action, hours, economy, work events (job loss, promotion), family, graduation, jobs, admissions, flats, then money. Every stage returns a probability table, so the same code gives:

- **`outcome_probs(s, a)`:** the exact distribution over the next discrete situation (W, prog, edu, k, u, pend, offer_k, H, F, L, M);
- **`sample(s, a, rng)`:** one random next month, including cash C and investments I, plus a ledger of every flow (income, bonus, rent, living, child, tuition, loan, bills, noise, moving, windfall, credit interest, invest, sell, market return).

| Piece of P | Source |
|---|---|
| P(job offer), offered pay level | Learned from the simulated world |
| P(admitted to a master's / PhD) | Learned from the simulated world |
| P(getting a bigger or smaller flat) | Learned from the simulated world |
| Job loss, promotion, family, economy, investment returns | Priors from the MDP spec ([`world/spec.py`](world/spec.py)) |
| Hours changes approved, study durations, entry pay by degree | Our assumptions (`spec.py`) |
| Pay, rent, living costs, bills, bonus, noise | Fitted on the challenge ledger |
| Cash, investments, credit | Exact bookkeeping |

P is **stochastic**: the world is random and P returns a distribution, never one outcome.

## 5. Results

### 5.1 Against Swiss statistics (1,000-person societies)

| Statistic | World | Swiss | Within band? |
|---|---|---|---|
| Unemployment rate | 3.9% | 2.8% (SECO; ILO measure higher) | yes |
| Bachelor's → master's within 2 years | 84% | 87% (BFS) | yes |
| Master's → PhD within 2 years | 14% | 15% (assumed) | yes |
| Master's graduates unemployed after 1 year | 6.0% | 6.4% (BFS) | yes |
| **Median spell without work** | **3.5 months** | 6.5 (BFS) | **no** |
| **Bachelor's graduates (no master's) unemployed after 1 year** | **26%** | 4.9% (BFS, FH bachelor) | **no** |

### 5.2 Does the learned P transfer to societies it never saw?

Calibration error (ECE, percentage points) for job offers / admissions / flats:

| New societies | Error |
|---|---|
| 200 people | 1.6 / 0.4 / 1.4 |
| 1,000 people | 2.2 / 0.9 / 1.6 |
| 2,000 people | 2.6 / 1.5 / 1.3 |
| **5,000 people** (larger than any in training) | 3.0 / 1.8 / 2.4 |
| Different persona mix (1,000 people) | 1.5 / 0.7 / 1.9 |

All 9 direction checks pass: a master's helps more than a bachelor's, better grades help, a long spell out of work hurts, more openings help, higher income relative to rent helps, and so on.

### 5.3 Why we train on mixed sizes

Our first model trained on 200-person worlds only, and failed in 2,000-person worlds (errors of 3.3 points for jobs and 4.3 for flats). Small markets are lumpy. With the same knobs, the PhD share changes with society size:

| Society size | Unemployment | Median spell | Bachelor's → master's | Master's → PhD |
|---|---|---|---|---|
| 200 | 3.7% | 3.4 months | 86% | **6.7%** |
| 1,000 | 3.9% | 3.5 months | 84% | 13.7% |
| 2,000 | 3.2% | 2.9 months | 83% | 16.6% |

A 200-person society has about 2 PhD positions a year. Training on 200, 1,000 and 2,000 together fixed the transfer.

### 5.4 Alexia (the challenge's example: 20, bachelor's in Lausanne, CHF 8,000 saved)

Just after graduating, each month:

| If she... | Then |
|---|---|
| searches for a job | 24.6% chance of an offer (19.4% at pay level 1, 4.9% at level 2) |
| applies for a master's | 98% admitted |
| tries for a bigger flat | 0%: she fails the landlord's income rule |

P(job offer), normal economy vs recession:

| Degree | Normal | Recession |
|---|---|---|
| Bachelor's | 24.6% | 4.1% |
| Master's | 36.8% | 7.1% |
| PhD | 48.7% | 11.0% |

The 5 bootstrap refits put her offer chance between 24.3% and 24.9%. The sampler matches the exact probabilities (total variation 0.004 over 5,000 draws).

**One choice, two placeholder policies, 300 lives each, same random draws:**

| | Work first | Master's first |
|---|---|---|
| Median net wealth at 30 | CHF 224k | CHF 272k |
| 10th–90th percentile at 30 | CHF 45k to 443k | CHF −19k to 550k |
| Lowest median point | CHF −18k | CHF −60k |
| Holding a master's at 30 | 0% | 99% |
| Working at 30 | 97% | 96% |
| Hit the credit limit | 94% | 100% |

**Master's first pays more by 30, but with a deeper dip and a wider spread.** The placeholder policies only set the status action (search, apply, accept). They never touch housing, spending or assets.

### 5.5 A finding for the team: almost everyone hits the credit limit

Alexia's recorded student budget runs a deficit: CHF 1,200 of income against about CHF 2,700 of rent and living costs. The spec's credit limit is 3 × monthly income, only CHF 3,600 for a student. So she crosses it in her first year, whichever choice she makes. The model sets `breach`; the spec would end the episode there.

Decisions needed:
- the twin policy must use the spending and housing levers;
- the student credit limit (family support, student loans) probably needs revisiting.

## 6. LLM agents: benchmarked, not used

[`analysis/society_sim/bench_llm.py`](../analysis/society_sim/bench_llm.py) ran with vLLM on one B200.

| Model | Load time | Decisions / s | Probability mass on valid answers |
|---|---|---|---|
| Qwen3-1.7B | 63 s | 1,532 | 0.999 |
| Qwen3-8B | 95 s | 690 | 0.53 |

Speed is not the problem: 1,000 agents × 120 months would take about 1.5 minutes. **Behaviour is.** Each persona's choices were nearly deterministic: "relaxed" never accepted a low offer, "driven" always did. They also did not respond to savings correctly: acceptance should fall as savings grow, but rose. The 8B model accepted every low offer. A utility-based softmax is faster, transparent, tunable, and responds to savings by construction.

Notes for running vLLM on B200: set `attention_backend="TRITON_ATTN"` and `VLLM_USE_FLASHINFER_SAMPLER=0`, and put the venv's `bin` and `CUDA_HOME` on `PATH`.

## 7. How the pieces connect

```
MDP spec (state, actions)  ──►  TransitionModel.allowed(s)
Twin policy π(s) → Action  ──►  TransitionModel.sample(s, a) → s', ledger   (repeat 120 months)
                                TransitionModel.outcome_probs(s, a) → exact P(s' | s, a)
```

To plug in the twin: write any function `State -> Action`, then use it in place of `policy_factory(...)` in section 6 of the notebook and pass it to `rollout(policy)`.

## 8. Known limitations and next steps

| Issue | Likely cause | Next step |
|---|---|---|
| Spells without work too short (3.5 vs 6.5 months) | Openings calibrated to unemployment; one knob can't hit both targets | Separate knobs for openings and for hiring selectivity |
| Bachelor's-only graduates unemployed too often (26% vs 4.9%) | Our entry ranking penalises a bachelor's too hard; the BFS figure is for FH (applied) bachelor's | Separate university and FH bachelor's |
| 5,000-person job error 3.0 pts | Outside the training range | Add 5,000-person societies to training |
| Master's → PhD target is assumed (15%) | No verified BFS figure found | Find the BFS figure |
| Persona preferences, study durations, entry pay | Assumptions | Sensitivity analysis |
| Credit limit hit by most students | Student budget deficit vs a small limit | Team decision (section 5.5) |

Synthetic data and labelled assumptions only. This is not a forecast or financial advice.

## 9. Files

| Path | What |
|---|---|
| [`world/spec.py`](world/spec.py) | Every constant, tagged by source (data, swiss, spec, assumed) |
| [`world/society.py`](world/society.py) | The simulated society: personas, three markets, money |
| [`world/build.py`](world/build.py) | Calibrate → simulate → fit → validate → `world_model.json` |
| [`world/transition.py`](world/transition.py) | **The deliverable:** `State`, `Action`, `TransitionModel` |
| [`world/test_transition.py`](world/test_transition.py) | Probabilities sum to 1, sampler matches, money adds up, rules, directions, transfer |
| [`world/world_model.json`](world/world_model.json) | The built model: coefficients, bootstraps, θ quantiles, validation |
| [`world/build_log.txt`](world/build_log.txt) | Log of the last build |
| [`world/swiss_targets.json`](world/swiss_targets.json) | Swiss statistics with sources |
| [`world/transition_tutorial.ipynb`](world/transition_tutorial.ipynb) | The guided tour, executed |
| [`world/make_notebook.py`](world/make_notebook.py) | Regenerates the notebook: `python world/make_notebook.py world/transition_tutorial.ipynb` |
| [`../analysis/society_sim/MATH.md`](../analysis/society_sim/MATH.md) | The maths, with a three-person example |
| [`../analysis/society_sim/toy_market.py`](../analysis/society_sim/toy_market.py) | The toy experiment that tested the tightness idea |
| [`../analysis/society_sim/bench_llm.py`](../analysis/society_sim/bench_llm.py) | The LLM-agent benchmark |

**Environment:** the virtualenv `~/envs/lauzhack-sim` has numpy, scipy, pandas, matplotlib, vLLM, ipykernel and nbclient. The Jupyter kernel is "lauzhack-sim". On the cluster, use `sbatch --wrap` on partition `frida`, which doesn't allow interactive `srun`. Keep `TMPDIR` off `/tmp`, because the root disk is nearly full.
