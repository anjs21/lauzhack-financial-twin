# 4. Markets: job offers, study places and flats

**In short.** The data never shows anyone getting a job, a study place or a flat, so these
probabilities cannot be learned from it. Instead we simulate a small Swiss society where people
compete for a limited number of openings, places and flats, calibrate it to official Swiss
statistics, and fit three small regression models on what happens. The twin uses those models
every month. Full write-up: [`twin_game_theory/README.md`](../twin_game_theory/README.md).

## Why one person's chance depends on everyone else

Whether you get a job depends on who else applies. So the true transition is
$P(s_i' \mid s_i, a_i, s_{-i}, a_{-i})$, not $P(s_i' \mid s_i, a_i)$.
[`analysis/society_sim/MATH.md`](../analysis/society_sim/MATH.md) works this out by hand with three
people and two openings: Alexia's chance of an offer is 9/16. It becomes 15/32 if another applicant
applies twice and 7/9 with a third opening. Nothing about Alexia changed.

**Mean field.** In a large market, "everyone else" is summarised well by one number, the
**tightness** θ = openings per application. Given θ, a person's chance depends only on their own
state. With independent applications, theory gives

$$\log\big(-\log(1 - P)\big) = \beta_0 + \beta_a \log(\text{applications}) + \beta_\theta \log\theta + \beta^\top x, \qquad \beta_a \approx \beta_\theta \approx 1.$$

That is a complementary log-log regression.

**The toy test.** [`toy_market.py`](../analysis/society_sim/toy_market.py) trained on 100-person
worlds and tested on 10× larger worlds and on a different mix of behaviour
([`results/toy_market.json`](../analysis/society_sim/results/toy_market.json)):

| Model | Calibration error after a change in behaviour |
|---|---|
| Constant rate | 2.1 points |
| Own state only | **7.5 points**: it breaks |
| Own state + tightness θ | 1.1 points |
| + share of master's holders | 1.1 points: adds nothing |

A model without θ is right only for the behaviour it was trained on (the Lucas critique). With θ,
the fitted coefficients on log θ and log(applications) were 0.94 and 0.91, close to the theory's 1.

## The simulated society

[`twin_game_theory/world/society.py`](../twin_game_theory/world/society.py): agents live month by
month and compete in three markets under one rule: limited places, applicants ranked, the best wins.

| Market | Places | Ranking |
|---|---|---|
| Jobs | Openings each month in proportion to the labour force; × 0.85 in inflation, × 0.5 in a recession | Degree, grades, time out of work, pay level, noise |
| Study places | Master's: ample. PhD: about 0.012 a year per person | Grades |
| Flats | Vacancies at the Swiss moving rate | Landlord's rule: rent ≤ income / 3, or 12 months' rent saved, or a student |

The agents here are the *population* that creates market traffic: a softmax over a five-part
utility (money, safety, leisure, career, inertia) with four archetypes. They are not the user's
twin.

**Calibration.** Four knobs (job openings per worker, the preference for a master's, PhD places, the
preference for a PhD) are tuned by coordinate search, with common random numbers, until 1,000-person
societies match the targets; flat turnover is set from the Swiss moving rate. The targets are in
[`swiss_targets.json`](../twin_game_theory/world/swiss_targets.json). Every target has a source and
a URL:

| Statistic | Simulated | Swiss figure | Within band? |
|---|---|---|---|
| Unemployment rate | 3.9% | 2.8% (SECO registered; the ILO measure is higher) | yes |
| Bachelor's → master's within 2 years | 84% | 87% (BFS) | yes |
| Master's → PhD within 2 years | 14% | 15% (assumed: no verified figure) | yes |
| Master's graduates unemployed after 1 year | 6.0% | 6.4% (BFS) | yes |
| Median spell without work | **3.5 months** | 6.5 months (BFS) | **no** |
| Bachelor's graduates without a master's, unemployed after 1 year | **26%** | 4.9% (BFS, applied-science bachelor's) | **no** |

## The three fitted models

Societies of 200, 1,000 and 2,000 people (90,000 people, 120 months each) produce 290k job-search
months, 27k study applications and 206k flat applications. Three models are fitted by maximum
likelihood, with five bootstrap refits that resample whole societies:

| Model | Link | Inputs |
|---|---|---|
| Job offer while searching | complementary log-log | degree, grades, years out of work, pay level, log(applications), log θ, economy, student |
| Admitted to a master's | logit | programme, grades, log θ |
| Finding a flat | logit, after the landlord's rule | target size, downsizing, log(income / new rent), log θ |

The job-offer coefficients on log θ and log(applications) come out at **0.91 ± 0.01** and
**0.94 ± 0.02**, as the mean-field theory predicts.

**θ for the twin.** The person cannot see the market's tightness, so each probability is averaged
over the θ values the society produced in that economy. One person barely moves θ, so taking it
as given is consistent.

**Model uncertainty.** Each simulated life uses one of the five bootstrap refits, picked by the same
ensemble member as its flow network. Market uncertainty is therefore counted under "what the model
does not know".

## How it plugs into the twin

[`twin/world/market.py`](../twin/world/market.py) reads `world_model.json` and the world calls it
every month:

* An **open job search** gets an offer with the learned probability.
* **Starting a master's** needs admission (about 98% for average grades); a refused person can apply
  again next January.
* A **housing choice** opens a flat search. Each month the flat market decides, after the
  landlord's rule. The search lasts until a flat is found or the next review.

People in the data have no education or grades. Unless the user gives them, grades are average and
education is drawn per life from an assumed prior that rises with pay (25% to 80% tertiary from the
lowest to the highest pay fifth). A student earns a bachelor's on graduating, or a master's after
the master's.

`Config(markets=False)` (or `--markets priors`) switches back to fixed rates: a 20% monthly offer
rate (10% in a recession), admission always granted, moves at once. [07](07-validation.md) compares
the two.

## Does it transfer to societies it never saw?

Calibration error in percentage points, job offers / admissions / flats
([`build_log.txt`](../twin_game_theory/world/build_log.txt)):

| New societies | Error |
|---|---|
| 200 people | 1.6 / 0.4 / 1.4 |
| 1,000 people | 2.2 / 0.9 / 1.6 |
| 2,000 people | 2.6 / 1.5 / 1.3 |
| 5,000 people (larger than any in training) | 3.0 / 1.8 / 2.4 |
| A different mix of personas | 1.5 / 0.7 / 1.9 |

All nine direction checks pass: a master's helps more than a bachelor's, better grades help, a long
spell out of work hurts, more applications help, more openings help, higher income relative to rent
helps, and so on. `tests/test_market.py` checks that the twin's adapter gives exactly the same
probabilities as the reference `TransitionModel`, and that every simulated move meets the landlord's
rule.

Reproduce: `python -m twin_game_theory.world.build` (~2–3 min), then
`python -m twin_game_theory.world.test_transition` (7 checks, ~30 s).
