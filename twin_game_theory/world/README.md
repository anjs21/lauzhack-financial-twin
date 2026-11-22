# twin_game_theory/world: the transition function P(s' | s, a)

```python
from twin_game_theory.world.transition import TransitionModel, State, Action, alexia
m = TransitionModel()                         # loads world_model.json
s = alexia()                                  # or State(...)
m.allowed(s)                                  # allowed actions in s
m.outcome_probs(s, Action("plus"), top=5)     # exact probabilities of the next discrete situation
s2, ledger = m.sample(s, Action("plus"), rng) # one random next month, with cash and investments
```

**State:** the MDP spec's W, k, H, F, L, M, C, I, t, plus `prog` (programme while studying), `edu` (highest degree), `u` (months without work), `pend` (pending application or offer), `offer_k`, `leave`, and price indices.
**Actions:** `Action(status, housing, spending, assets)`, where `status` is one of stop / minus / stay / plus / apply_master / apply_phd / accept.

## Where each piece comes from

| Piece | Source |
|---|---|
| P(job offer), offered pay level | **Learned** from simulated societies of 200, 1,000 and 2,000 people (90,000 in all) competing for limited openings |
| P(admitted to a master's / PhD) | **Learned** from the same world (limited places, ranked by grades) |
| P(getting a bigger or smaller flat) | **Learned** (limited vacancies, landlords' income rule) |
| Job loss, promotion, family, economy, returns, hours changes | Priors (`spec.py`) |
| Pay, rent, living costs, bills, bonus, noise | Fitted on the challenge ledger |
| Cash, investments, credit | Exact bookkeeping |

The world is calibrated to Swiss statistics in `swiss_targets.json`. Rebuild with `python -m twin_game_theory.world.build` (about 2-3 minutes; `TRAIN_SIZES` in `build.py` sets the society sizes). Test with `python -m twin_game_theory.world.test_transition`.

## Status (see `world_model.json` and `build_log.txt`)

- **Matches Swiss figures** (calibrated in 1,000-person societies):
  - unemployment 3.9% (SECO 2.8%, inside the band allowing for the higher ILO measure);
  - bachelor's to master's 84% (BFS 87%);
  - master's graduates unemployed after one year 6.0% (BFS 6.4%).
- **Misses:**
  - median spell without work is 3.5 months, against 6.5 (BFS);
  - bachelor's graduates who don't go on to a master's are unemployed 26% of the time after a year, against 4.9% (BFS, FH bachelor).
- **Assumption:** master's to PhD is 14% against an assumed 15%; we found no verified BFS figure.
- **World size matters for the PhD market:** with the same knobs, master's to PhD is 6.7% in 200-person societies and 16.6% in 2,000-person ones (few, lumpy positions). Training mixes sizes for this reason.
- **Transfer:** calibration error under 3 points on new societies of 200, 1,000 and 2,000 people and on a different persona mix; 3.0 points for job offers in 5,000-person societies (outside the training range; tested with a 4-point bound).
- Synthetic data and labelled assumptions only; not a forecast or financial advice.
