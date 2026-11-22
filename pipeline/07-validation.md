# 7. Validation: how well each part performs

**In short.** Every learned part is tested against something it was not fitted on: people held out
of training, a year held out of training, Swiss statistics, or societies it never saw. The
accounting is tested every month of every life. The flow network beats the hand-calibrated
simulator on held-out people and passes all four acceptance tests. The weakest evidence is for
behaviour and for anything beyond two years, because the data cannot test those (see the last
section and [09](09-limitations.md)).

For these docs we re-ran the acceptance tests and the test suite on 2026-11-22 with the current code;
both match the saved results exactly. Rows marked † were also computed that day, with
[`diagnostics.py`](diagnostics.py) (`python pipeline/diagnostics.py`, about 2 minutes).

## 1. The monthly flow network ([`twin/validate.py`](../twin/validate.py))

Every test is also run on the hand-calibrated simulator from
[`action_importance.py`](../analysis/action_importance/action_importance.py), the model the network
replaced.

| Test | What it checks | Network | Baseline | Target |
|---|---|---|---|---|
| **T1** | CRPS of 2026 monthly flows for people **held out of training** (5 folds by person) | **409** | 421 | ≤ baseline |
| **T2** | Trained on 2025 only: end-2026 cash inside its 50% / 90% band | 0.48 / 0.89 | 0.51 / 0.89 | 0.50 / 0.90 |
| **T3** | Rolled forward from December 2025: profiles that go below zero in 2026 | 60.9 (90%: 59–62) | 61.1 | 61 observed |
| | End-2026 cash CRPS | **1,700** | 2,033 | lower is better |
| **T4** | Simulated ledgers show the data's patterns | all 7 pass | 6 of 7 | |

CRPS (continuous ranked probability score) measures a whole predicted distribution against what
happened, in CHF. Lower is better; for a point forecast it is the absolute error.

**T1 by work status** (CRPS, network vs baseline): employed 401 vs 419, retired 298 vs 313,
part-time 244 vs 247, students 234 vs 237. The network is slightly worse for the self-employed
(1,004 vs 989) and the unemployed (226 vs 220).

**T4, the patterns** (data · network · baseline):

| Pattern | Data | Network | Baseline |
|---|---|---|---|
| Share of employees with a December bonus | 0.559 | 0.554 | 0.556 |
| Extra cost in February and August | CHF 362 | 349 | 369 |
| Student fee in September | CHF 831 | 847 | 860 |
| Noise by pay fifth (CHF) | 332 · 351 · 430 · 501 · 582 | 320 · 372 · 409 · 479 · 569 | 411 · 411 · 410 · 410 · 412 **(fails)** |
| Correlation between consecutive months | −0.026 | −0.038 | −0.027 |
| Bills per CHF 1,000 of pay | −8 | −4 | −1 |
| Mean flow by mode (student, part-time, full-time, not working) | −2,392 · 677 · 4,296 · −155 | −2,405 · 686 · 4,299 · −144 | −2,393 · 677 · 4,296 · −172 |

The baseline's one failure is the reason it was replaced: its noise does not grow with pay, so it
gets the spread of high earners wrong.

**Honest notes.**

* The end-2026 forecasts lean slightly low. The observed cash falls in the top tenth of the forecast
  158 times and in the bottom tenth 101 times, against 120 each if calibrated. The baseline leans
  the same way.
* T2 and T3 use the drift in `priors.yaml`, which was fitted on 2025 *and* 2026: a small look-ahead.
  The chronological replay below uses no drift at all and gets the same flow CRPS (410 vs 406), so the
  leak does not drive the result.

## 2. Monthly updating on 2026 ([`twin/replay.py`](../twin/replay.py))

Fit on 2025 with no drift, then predict each month of 2026 before revealing it
([`monthly_replay.json`](../twin/artifacts/monthly_replay.json)):

| Strategy | Flow CRPS | Mean closing-cash error (CHF) | Shortfall alerts (≥ 50%) | New shortfalls caught |
|---|---|---|---|---|
| Frozen: never look at actual cash | 410 | 1,355 | 29 | 25 of 28 |
| Anchor to the actual cash each month | 410 | **591** | 29 | **26 of 28** |
| Also learn a recurring correction | 410 | 592 | 29 | 26 of 28 |

Anchoring to actual balances more than halves the cash error. With it, 26 of the 29 alerts were real
and 26 of the 28 first shortfalls were caught.

## 3. The market models ([04](04-markets.md))

| Check | Result |
|---|---|
| Swiss statistics, 1,000-person societies | 4 of 6 within band; misses: spells without work too short (3.5 vs 6.5 months), bachelor's-only graduates unemployed too often (26% vs 4.9%) |
| Calibration on new societies of 200 to 2,000 people | under 3 points for job offers, admissions and flats |
| 5,000 people (outside training) | 3.0 / 1.8 / 2.4 points |
| A different persona mix | 1.5 / 0.7 / 1.9 points |
| Directions (degree, grades, time out, applications, openings, income/rent) | all 9 pass |
| Fitted coefficients on log θ and log(applications) | 0.91 and 0.94; theory predicts 1 |
| The twin's adapter equals the reference model | exact (test) |

**Effect on whole lives** †, 2,000 lives each, same random numbers, learned markets vs the fixed
rates they replaced:

| Person | Months without work before 65 | Goal met by 2029 | Credit limit breached | Median net worth 2034 |
|---|---|---|---|---|
| Alexia, 20, student | 6.5 vs 9.1 | 64% vs 56% | 24% vs 34% | CHF 628k vs 594k |
| Samir, 58, employed | 14.3 vs 14.2 | 96% vs 96% | 0% vs 0% | CHF 998k vs 993k |
| Mia, 27, employed | 3.3 vs 3.1 | 87% vs 87% | 8% vs 8% | CHF 355k vs 353k |

The markets matter for people who must *get* a job or a place, and barely at all for people who
already have one. That is the expected direction.

## 4. Where the spread comes from †

The width between the 10th and 90th percentile of 2034 net worth, with only one source of randomness
left on (CHF):

| Source | Alexia, student | Mia, employee |
|---|---|---|
| All sources | 659k | 343k |
| Luck: events, flows, markets | 569k | 315k |
| What they choose | 351k | 98k |
| What the model does not know (ensemble member, market refit) | 346k | 14k |
| Who they are (trait uncertainty) | 128k | 0.5k |

The widths do not add up, because the sources interact. For an established employee, luck dominates
and the model is confident. For a student about to enter the job market, the model's own uncertainty
is as large as her choices. The twin shows this so it is not mistaken for luck.

## 5. The persona ([05](05-persona-policy.md))

| Check | Result |
|---|---|
| Yearly rates †, 240 profiles × 50 lives | work changes 4.1%, flat changes 1.6%, lifestyle changes 10.9% a year |
| GRU clone vs rule persona, unseen profiles | KL 0.0018 (status), 0.0007 (housing), 0.0008 (spending), 0.0010 (assets) per decision |
| January 2026 Qwen proxy GRU | KL 0.031 on 9 held-out profiles (60 labels in all) |
| Foresight at f = 0.25 / 0.5 / 1 | mean change in choice probabilities 0.10 / 0.14 / 0.18 |
| Directions (tests) | frugal personas upgrade less; intentions fire with their probability |

There are no observed decisions, so none of this shows the persona decides like a real person.

## 6. The planner agent ([06](06-planner-agent.md))

A real Qwen3-8B run for Mia signed both versions in 2 rounds each. Version 2 changes her goal odds
by +5.7 points against her habits (Monte Carlo 95% interval +2.3 to +9.0). Fifteen tests cover the
plan semantics, the agent loop and the API.

## 7. The code ([`tests/`](../tests/)) †

**82 tests pass** in about 2 minutes: 75 in `tests/` and 7 in
`twin_game_theory/world/test_transition.py`.

| Group | Examples |
|---|---|
| Accounting | the cash identity every month; transfers leave net worth unchanged; loans repay and never go negative; negative returns shrink the portfolio; higher tuition never lowers costs; zero income stays visible |
| Reproducibility | same seed, same lives; a what-if changes nothing before its plan starts |
| Rules | masks respected; status and housing change only at decision points; the network never sees the calendar year |
| Markets | probabilities sum to 1; the sampler matches the exact probabilities; every move meets the landlord's rule; a flat search lasts until found or the next review |
| Planner and API | the 15 plan tests; authentication, validation, provider errors, the busy-laptop 429 |

Speed †: 2,000 lives × 120 months take 1.3–2.5 s on a CPU.

## What these tests do not show

* **No test of a change of situation against data.** The flow network is tested one year ahead for
  people whose situation stays the same. What happens after a job change or a move is built from
  comparing people, and the data holds no switches to check it against.
* **No test beyond two years.** Ten-year paths rest on the drift indices and the priors.
* **No test of behaviour.** The persona is checked for plausibility, not fitted to or tested against
  decisions.
* **The data is synthetic and simple.** The hand-calibrated simulator separates who runs short in
  2026 perfectly (AUC 1.0). Fitting this data well is easier than fitting real ledgers would be.

Reproduce: `python -m twin validate`, `python -m twin replay`,
`python -m twin_game_theory.world.build`,
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests twin_game_theory/world/test_transition.py -q`.
