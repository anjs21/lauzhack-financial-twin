# Financial Twin: learned world, sampled person

Explores the range of lives one person could lead from 2025-01 to 2034-12, given how they tend to
decide. Nothing is optimised: a **persona policy** samples each month's choices from the person's
traits and their own history, and a **world model** learned from the ledger samples what happens
next. Thousands of sampled lives show what is likely, what is possible and what drives it.

Design: [`analysis/mdp_spec/mdp-spec.html`](../analysis/mdp_spec/mdp-spec.html) (states, actions,
catalog coverage) and [`learning-plan.html`](../analysis/mdp_spec/learning-plan.html) (how it is built).

## Quick start

```bash
python -m twin train                     # ~3 min: fit the world model, run the acceptance tests
python -m twin explore P00003            # ~8 s: 2,000 lives + what-ifs -> twin/artifacts/reports/P00003.html
python -m twin explore --custom twin/examples/custom_person.json --answers twin/examples/answers.json \
    --plan '{"head": "status", "option": 1, "prob": 0.6, "when": "2026-01"}'
python -m twin clone                     # ~2 min, stretch: neural persona cloned from the rule persona
python -m twin explore P00003 --policy neural   # the same explorer driven by the neural persona
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q    # 20 checks, ~6 s
```

Needs numpy, pandas, scipy, PyTorch and PyYAML. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` only matters on
machines where another install (such as ROS) puts broken pytest plugins on the path.

## How one month works

`s' ~ T(s, a)` in [`world/env.py`](world/env.py), for many paths at once:

1. **Apply the action** (status 0-4, housing / spending / assets -1, 0, +1; invalid options are masked).
2. **Draw events and the economy**: job loss, job offers, promotions, the master's ending, partner,
   child, separation, windfalls, and switches between normal, inflation and recession.
3. **Draw the month's net flow** from the learned network.
4. **Update the books** exactly: cash, investments, loan principal and interest, credit interest.

| Part | Learned from | How |
|---|---|---|
| Monthly net flow | Ledger, 28,800 profile-months | Mixture density network (MLP 2x64, 5 Gaussians), 5-seed ensemble, per-person offset ([`world/flow.py`](world/flow.py)) |
| Income when the work mode changes | Take-home pay by mode in the profiles | The person keeps their pay rank in the new mode's distribution ([`world/redraw.py`](world/redraw.py)) |
| Rent after a move | Rent by city and housing tier (R² 0.66) | Log-rent model; the person keeps their place in the distribution |
| Pay and price drift | Ledger trends: +1.4% and +1.0% a year | Indices outside the network, which never sees the calendar year |
| Everything the data lacks | Nothing: assumptions | [`world/priors.yaml`](world/priors.yaml): event rates, economy, returns, offers, degree premium, lifestyle, credit |

The network works in real terms (deflated by the price index), so ten-year rollouts never ask it
to extrapolate past the two years it saw. Nobody changes job, flat or spending in the data, so an
action works by moving the person into a state other people already occupy; that is how the
synthetic data was generated and would need checking on real data.

## Acceptance tests (`twin/artifacts/validation.json`)

The hand-calibrated simulator from `analysis/action_importance` runs through the same tests.

| Test | Network | Baseline | Target |
|---|---|---|---|
| T1 CRPS of 2026 monthly flows, people held out of training | **409** | 421 | no worse than baseline |
| T2 end-2026 cash inside its 50% / 90% band (trained on 2025 only) | 0.48 / 0.89 | 0.51 / 0.89 | 0.50 / 0.90 |
| T3 profiles below zero in 2026 | 60.9 | 61.1 | 61 observed |
| End-2026 cash CRPS | **1,700** | 2,033 | lower is better |
| T4 patterns: December bonus, Feb/Aug bills, student fees, noise by pay, no carry-over, bills unrelated to pay | all pass | fails noise by pay | same as data |

## Persona policy

[`policy/rules.py`](policy/rules.py) builds logits from trait-weighted rules plus inertia, masks
invalid options, divides by the person's temperature and **samples**. Status and housing are
reconsidered at the January review; spending and assets every month.

* **Traits** ([`policy/traits.py`](policy/traits.py)): risk tolerance, frugality, career drive,
  wish for free time, housing aspiration, inertia, impatience and goal focus. Defaults come from
  the profile; a questionnaire (1-5 per trait) can override them; each path draws its own traits
  around them, which shows how much the answer depends on not knowing the person exactly.
* **Stated plans**: `{"head": "status", "option": 1, "prob": 0.6, "when": "2026-01"}`, or
  conditional ones such as `{"head": "housing", "option": -1, "prob": 0.7, "cash_months_below": 2}`.
* **Neural persona** ([`policy/net.py`](policy/net.py)): a GRU policy cloned from the rule
  persona's probabilities over 2,400 sampled lives, tested on unseen profiles
  (`twin/artifacts/persona_net.json`). `--policy neural` in the explorer.
* **LLM persona** ([`policy/llm.py`](policy/llm.py)): an OpenAI model role-plays the person at
  decision points of the clone's training lives (situation, goals and traits in words) and answers
  how *this person* would likely act, as probabilities. Those answers replace the rule targets at
  those points, with 10x weight, when the neural persona is cloned. Answers are cached on disk, so
  reruns cost nothing; only synthetic profiles and simulated states are sent.
* **Foresight** ([`policy/foresight.py`](policy/foresight.py)): the neural persona plus a knob
  f in [0, 1], trained by looking ahead in the world model: at ~2,000 decision points each option is simulated
  60 months ahead (same luck for every option) and the persona's probabilities are tilted by
  `exp(f x value gain / CHF 10k)`, the exact optimum of `f x utility - KL(to the persona)`, then distilled.
  f = 0 is the person as they tend to decide; higher f leans toward outcomes the persona values
  (the spec's preference table) while staying random and close to their habits. The explorer
  report shows f = 0, 0.5 and 1 side by side on the same random numbers.

### LLM persona and foresight

```bash
export OPENAI_API_KEY=sk-...             # read from the environment only; never written to files
export OPENAI_MODEL=gpt-4o-mini          # optional (default gpt-4o-mini)
python -m twin llm-demos --dry-run       # see one prompt, no call
python -m twin llm-demos --calls 300     # ~0.3M tokens; answers in twin/artifacts/llm_demos.json
python -m twin clone --llm               # retrain the neural persona with the LLM answers
python -m twin foresight                # ~5 min: look ahead, distil the foresight knob into the (new) clone
python -m twin explore P00001 --policy foresight --foresight 0.5
```

Retrain `foresight` after every `clone`, since foresight starts from the clone and keeps it as its reference.

## Explorer output

`python -m twin explore` writes an HTML report: fan charts of net worth, cash and income; work mode
by year; the most common life stories; turning points (associations, not causes); what-ifs that
rerun the same paths with one extra stated plan (common random numbers); and how much of the
spread comes from traits, choices, luck and model uncertainty.

## Tests ([`tests/test_twin.py`](../tests/test_twin.py))

Cash identity every month · transfers leave net worth unchanged · loans amortise and never go
negative · zero income is visible · higher tuition never lowers costs · negative returns shrink the
portfolio · same seed, same lives · a what-if changes nothing before its plan · masks respected ·
status and housing only at decision points · stated plans fire with their probability · frugal
personas upgrade less · world model passed its acceptance tests.

Synthetic data only. The explorer shows possibilities under stated assumptions; it is not a
forecast or financial advice.

## Live decision desk

Run from the repository root:

```bash
python -m streamlit run twin/desk.py
```

Install the optional UI dependency with `pip install -e '.[live]'` if necessary.
Select a profile and enter a part-time offer in **net CHF/month**. Preview compares
the quoted salary with the current plan on paired paths. Commit records that offer
for the next January review. Later job loss, workload changes and retirement can
end the contract. The annual salary drift assumption still applies.

Record each monthly closing cash balance in sequence. You can update investments,
rent, employment and pay at the same time. Actual cash and reported investment values
replace estimates for that month on every path. Omitted investments and employment
carry forward during the recorded period, scheduled loans amortize, and unreported
pay and rent follow normal drift. The optional comparable net-flow field enables
a shrunk estimate of recurring surplus. Leave it blank for balance-only updating.
A per-profile SQLite journal persists committed decisions and observations across
restarts. Set `TWIN_JOURNAL` to select a different file; the default is
`.twin-sessions.sqlite3` in the launch directory. Run the desk locally for one user.

The feasibility grid evaluates every entered salary with identical random streams.
It reports the lowest **tested** amount meeting the requested goal probability;
Monte Carlo sampling and model assumptions affect the result. Goal success requires
remaining within the credit limit up to the goal deadline. Paths continue after a
breach to show its consequences. The paired interval covers Monte Carlo uncertainty
in the change in goal probability. Portfolio, goal and wealth values are nominal CHF.

Voluntary status and housing choices share the January review, including dated
intentions. An off-cycle major plan takes effect the following January if eligible.
Spending and asset transfers remain monthly. A job offer from an ongoing search,
job loss, graduation or a forced family move can still change the state midyear.
Starting students are assumed to be finishing their current course and may elect
a master's at the opening review. Existing neural weights were fitted on the previous
review schedule. The annual eligibility rule applies to rules, neural and foresight
policies at execution time. Earlier cloning/foresight training metrics describe
the saved weights on their original training task.

## Chronological replay

`python -m twin replay` fits a 2025-only flow ensemble and scores January through
December 2026, revealing observations after each prediction. All three strategies
use flat ex-ante drift and the same 200 draws per person. The frozen strategy carries
its own balances, the second anchors to actual monthly cash, and the third also learns
a recurring flow correction. First-shortfall alerts are scored only while the person
has never breached during the evaluation period and started 2026 solvent. See
[the replay scores](artifacts/monthly_replay.json).
