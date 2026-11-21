# Causal financial transition model

This optional, standard-library-only model compares immediate work with a full-time
master's degree. It builds monthly transitions from explicit mechanisms rather
than fitting correlations in the synthetic profiles. It is a deterministic
transition model suitable for a future MDP environment, not a policy optimizer or
an empirically identified causal model. Existing starter examples are unchanged.

## Run

From the repository root:

```bash
python causal-model/run_causal_example.py
python causal-model/run_causal_example.py --sensitivity --export /tmp/causal-trace.csv
python -m unittest discover -s causal-model -p 'test_*.py'
```

Edit `causal-model/causal_assumptions.json` or supply `--assumptions path/to/config.json`.
Use `--months 24` for a two-year horizon. Month 1 starts at the profile's as-of
date. `--profile P00002` selects another profile with S001 inputs; profiles without
those inputs fail explicitly. Salary and cost defaults are Alexia-oriented teaching
assumptions and must be adapted for other profiles.

## Input provenance

- `Data/synthetic-profiles.csv`: profile identity, starting date, opening cash.
- `Data/scenario-assumptions.csv`: S001 study duration and annual tuition, with
  row IDs printed in the output. For Alexia these are 24 months and CHF 8,000/year.
- `causal_assumptions.json`: illustrative net salaries, living expenses,
  scholarship, employment delay, and reserve. These are assumptions, not causal
  estimates or official Swiss statistics. Explicit config parameters override
  CSV values; the CLI prints the complete effective parameter set.

The supplied cash-balance history is not reused as an employment baseline. Assets,
liabilities, returns, taxes, inflation and loan financing are not modelled here.
The older `starter-kit/scenario_config.json` is a separate teaching example and is not loaded.

## Mechanisms and timing

State contains month, cash, education status, completed study months, employment,
and months since graduation. It is immutable. `transition(state, action, assumptions)`
returns the next state and an accounting trace.

```text
action -> study progress -> graduation -> employment -> income -> cash
action -> tuition --------------------------------------------> cash
action -> living costs ---------------------------------------> cash
scholarship --------------------------------------------------> cash
```

- `work`: employment before enrollment, or graduate employment/job search after
  graduation. This illustrative model assumes immediate employment before study.
- `enroll`: allowed only before any study. It completes the first study month.
- `study`: allowed only while enrolled. It completes one further study month.
- Graduation occurs at the end of the final study month. Graduate earnings begin
  no earlier than the following month, after the configured job-search delay.
- Study earnings represent assumed part-time income, not a separately optimized
  hours decision. Pausing study and re-enrollment are not implemented.
- Annual tuition is due at the beginning of study months 1, 13, etc. A tuition
  scholarship is received immediately before tuition; it cannot exceed tuition.
  A final partial academic year is charged in full.
- Income and living expenses settle at month end. There are no daily transactions.
- Cash updates as `opening + scholarship - tuition + income - living costs`.
  Monetary amounts use Decimal, rounded to cents using ROUND_HALF_UP.

The employment mechanism is currently a deterministic delay, not a learned hiring
probability. Qualification changes assumed earnings and job-search timing only;
these relationships are hypotheses that can be varied, not proven degree effects.

## Funding and interpretation

The minimum includes opening cash, cash after upfront payments, and closing cash.
A tuition payment can therefore trigger a shortage even when month-end cash is
positive. Negative cash is an explicitly unfunded projection: no loan is silently
created, and the trace continues only to measure the funding requirement.

`extra_initial_funding` is the smallest additional starting cash needed to keep
that fixed path above `minimum_reserve`, given the model's payment timing. It is
not a borrowing recommendation and carries no assumed interest or repayment.
A path may have positive closing cash yet be infeasible without this funding.

Default 120-month results for Alexia:

| Policy | Closing cash | Minimum cash | Extra initial funding for zero reserve |
|---|---:|---:|---:|
| Work now | CHF 152,000 | CHF 8,000 | CHF 0 |
| Study now | CHF 133,600 | CHF -29,600 | CHF 29,600 |

These are conditional model outputs, not predictions or complete net worth.

## Interventions and sensitivity

Use `dataclasses.replace(assumptions, annual_scholarship=5000)` for an intervention
on tuition funding. It changes cash without altering study progress or salary.
The Python `simulate` function also accepts `enroll_month=12` to work for one year
before enrolling. The CLI compares work now with study now.

`--sensitivity` varies one parameter at a time: tuition, graduate income, and
employment delay. It reports the study-minus-work closing-cash difference and
funding requirement for each parameter choice. These are fixed-policy comparisons,
not policy re-optimization or calibrated probabilities. The reported mechanism
contributions are an exact additive cash-accounting decomposition, not estimated
causal mediation effects.

Tests cover accounting identities, scholarship intervention isolation, graduation
and job timing, annual tuition, intramonth shortfalls, funding top-ups, delayed
enrollment, invalid actions, CSV loading, and tuition monotonicity.

Next extensions can add borrowing with explicit limits and repayments, stochastic
employment shocks, selectable working hours, goals, and a reward/constraint layer.
