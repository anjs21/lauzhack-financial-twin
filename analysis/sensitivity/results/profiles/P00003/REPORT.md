# Samir (P00003): sensitivity analysis of the current simulator

Seed: 20261122; baseline and individual stresses: 1000 paired paths; joint grid: 300 paired paths; horizon: 120 months.

## Baseline comparison

| Measure | Study intention | Work-search intention |
|---|---:|---:|
| Cash negative at any month end | 0.3% | 0.3% |
| Ever exceeds credit limit | 0.3% | 0.3% |
| High-priority goal at deadline | 96.1% | 96.1% |
| Median minimum cash (CHF) | 13,378 | 13,378 |
| Median terminal net worth (CHF) | 989,725 | 989,725 |
| Mean terminal net worth (CHF) | 1,009,223 | 1,009,223 |

Study minus work: shortfall +0.0 pp (conservative paired 95% Monte Carlo interval -0.5 to +0.5 pp); mean terminal net worth CHF +0 (paired t interval CHF +0 to +0).

At the opening review, 0.0% of study-intention paths selected the master’s action and 0.0% of work-intention paths selected the job-search action. Selection does not guarantee admission, employment, or completion.

## One-at-a-time changes to the study-intention baseline

| Change | Shortfall | Difference vs baseline (pp; paired 95% interval) | Goal | Mean final wealth change CHF (paired 95% interval) |
|---|---:|---:|---:|---:|
| Tuition CHF 2,500 | 0.3% | +0.0 (-0.5 to +0.5) | 96.1% | +0 (+0 to +0) |
| Tuition CHF 12,000 | 0.3% | +0.0 (-0.5 to +0.5) | 96.1% | +0 (+0 to +0) |
| Degree pay-rank shift 0.00 | 0.3% | +0.0 (-0.5 to +0.5) | 96.1% | +0 (+0 to +0) |
| Degree pay-rank shift 0.10 | 0.3% | +0.0 (-0.5 to +0.5) | 96.1% | +0 (+0 to +0) |
| Job-offer probability ×0.5 | 0.3% | +0.0 (-0.5 to +0.5) | 95.5% | -23,043 (-27,973 to -18,113) |
| Job-offer probability ×1.5 | 0.3% | +0.0 (-0.5 to +0.5) | 96.6% | +6,284 (+3,745 to +8,822) |
| Study-delay probability 0% | 0.3% | +0.0 (-0.5 to +0.5) | 96.1% | +0 (+0 to +0) |
| Study-delay probability 25% | 0.3% | +0.0 (-0.5 to +0.5) | 96.1% | +0 (+0 to +0) |
| Credit rate 4% | 0.3% | +0.0 (-0.5 to +0.5) | 96.1% | +6 (-2 to +14) |
| Credit rate 12% | 0.3% | +0.0 (-0.5 to +0.5) | 96.1% | -7 (-16 to +3) |
| Moving cost CHF 1,000 | 0.3% | +0.0 (-0.5 to +0.5) | 96.1% | +209 (+175 to +243) |
| Moving cost CHF 4,000 | 0.3% | +0.0 (-0.5 to +0.5) | 96.1% | -435 (-503 to -368) |
| All-regime price growth 0% | 0.3% | +0.0 (-0.5 to +0.5) | 96.0% | +26,010 (+24,562 to +27,457) |
| All-regime price growth 3% | 0.5% | +0.2 (-0.5 to +0.9) | 95.9% | -40,256 (-42,792 to -37,721) |

## Funding grid

| Added starting cash CHF | Negative-cash rate | Change vs baseline pp (paired 95% interval) |
|---:|---:|---:|
| 0 | 0.3% | +0.0 (-0.5 to +0.5) |
| 5,000 | 0.3% | +0.0 (-0.5 to +0.5) |
| 10,000 | 0.3% | +0.0 (-0.5 to +0.5) |
| 20,000 | 0.3% | +0.0 (-0.5 to +0.5) |
| 40,000 | 0.2% | -0.1 (-0.7 to +0.5) |

## Joint grid

See `joint_grid.csv` for 27 combinations of tuition, job-offer probability multiplier and degree pay-rank shift. Both plans are re-simulated in every combination. Numeric intervals are conditional Monte Carlo intervals, not evidence of causal effects.

## Interpretation and limitations

- Actions are intentions at the January review, not guaranteed admission or employment; actions may be unavailable or redundant for this profile, and later behaviour remains adaptive.
- Tuition baseline is the fixed stress-test assumption CHF 8,000, not necessarily the profile-specific tuition.
- The goal evaluation uses the simulator deadline of December 2029 for all profiles, regardless of the date in the source data.
- Tuition input applies to all months the model treats the person as a student, including any initial pre-master study. The flow model already includes typical fees; only the excess is charged separately.
- Offer factor multiplies the learned model probability and is clipped to [0,1]; it is an illustrative stress, not a calibrated economic scenario.
- Money is nominal CHF and recorded negative cash uses the model credit and automatic investment-top-up rules; credit approval is not modelled.
- Goal uses the current goal_met definition, including its credit-breach condition. Mean paired changes and percentile-based summaries are distinct.
- Personal residual offsets were fit on available history and may include information after the January 2025 start; not an out-of-time backtest.
- Output intervals describe Monte Carlo error conditional on model and assumptions, not causal effects or real-world confidence bounds.
- The funding grid re-simulates adaptive behaviour and is not a guaranteed minimum; probability may not be monotonic.
- With only a few hundred grid paths, small differences can be Monte Carlo noise; do not optimise based on them.

## Reproduce

```bash
python -B analysis/sensitivity/run_sensitivity.py --profile P00003 --paths 1000 --grid-paths 300 --seed 20261122
```
