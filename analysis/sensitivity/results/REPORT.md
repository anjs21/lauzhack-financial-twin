# Alexia: sensitivity analysis of the current simulator

Seed: 20261122; baseline and individual stresses: 1000 paired paths; joint grid: 300 paired paths; horizon: 120 months.

## Baseline comparison

| Measure | Study intention | Work-search intention |
|---|---:|---:|
| Cash negative at any month end | 99.6% | 18.3% |
| Ever exceeds credit limit | 99.6% | 10.0% |
| High-priority goal at deadline | 0.4% | 75.2% |
| Median minimum cash (CHF) | -30,428 | 269 |
| Median terminal net worth (CHF) | 653,028 | 649,202 |
| Mean terminal net worth (CHF) | 629,754 | 650,618 |

Study minus work: shortfall +81.3 pp (conservative paired 95% Monte Carlo interval +77.9 to +83.9 pp); mean terminal net worth CHF -20,864 (paired t interval CHF -38,797 to -2,931).

At the opening review, 100.0% of study-intention paths selected the master’s action and 100.0% of work-intention paths selected the job-search action. Selection does not guarantee admission, employment, or completion.

## One-at-a-time changes to the study-intention baseline

| Change | Shortfall | Difference vs baseline (pp; paired 95% interval) | Goal | Mean final wealth change CHF (paired 95% interval) |
|---|---:|---:|---:|---:|
| Tuition CHF 2,500 | 99.5% | -0.1 (-0.7 to +0.5) | 3.8% | +1,751 (-3,822 to +7,325) |
| Tuition CHF 12,000 | 99.7% | +0.1 (-0.5 to +0.7) | 0.4% | -1,503 (-6,262 to +3,256) |
| Degree pay-rank shift 0.00 | 99.6% | +0.0 (-0.5 to +0.5) | 0.4% | -116,632 (-123,132 to -110,131) |
| Degree pay-rank shift 0.10 | 99.6% | +0.0 (-0.5 to +0.5) | 0.4% | -54,815 (-58,545 to -51,084) |
| Job-offer probability ×0.5 | 99.7% | +0.1 (-0.5 to +0.7) | 0.3% | -18,883 (-27,906 to -9,860) |
| Job-offer probability ×1.5 | 99.6% | +0.0 (-0.5 to +0.5) | 0.4% | +4,544 (-2,106 to +11,193) |
| Study-delay probability 0% | 99.6% | +0.0 (-0.5 to +0.5) | 0.4% | +6,723 (+2,455 to +10,992) |
| Study-delay probability 25% | 99.6% | +0.0 (-0.5 to +0.5) | 0.4% | -10,303 (-14,826 to -5,781) |
| Credit rate 4% | 99.6% | +0.0 (-0.5 to +0.5) | 0.4% | -1,535 (-3,831 to +761) |
| Credit rate 12% | 99.6% | +0.0 (-0.5 to +0.5) | 0.4% | +2,378 (-249 to +5,004) |
| Moving cost CHF 1,000 | 99.6% | +0.0 (-0.5 to +0.5) | 0.4% | -93 (-1,545 to +1,359) |
| Moving cost CHF 4,000 | 99.6% | +0.0 (-0.5 to +0.5) | 0.4% | -26 (-1,827 to +1,775) |
| All-regime price growth 0% | 99.6% | +0.0 (-0.5 to +0.5) | 0.4% | +13,926 (+10,785 to +17,067) |
| All-regime price growth 3% | 99.6% | +0.0 (-0.5 to +0.5) | 0.3% | -28,063 (-31,450 to -24,676) |

## Funding grid

| Added starting cash CHF | Negative-cash rate | Change vs baseline pp (paired 95% interval) |
|---:|---:|---:|
| 0 | 99.6% | +0.0 (-0.5 to +0.5) |
| 5,000 | 99.6% | +0.0 (-0.5 to +0.5) |
| 10,000 | 97.0% | -2.6 (-4.0 to -1.2) |
| 20,000 | 85.1% | -14.5 (-17.2 to -11.7) |
| 40,000 | 7.2% | -92.4 (-94.1 to -89.8) |

## Joint grid

See `joint_grid.csv` for 27 combinations of tuition, job-offer probability multiplier and degree pay-rank shift. Both plans are re-simulated in every combination. Numeric intervals are conditional Monte Carlo intervals, not evidence of causal effects.

## Interpretation and limitations

- Actions are intentions at the January review, not guaranteed admission or employment; policies and later actions remain adaptive.
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
python -B analysis/sensitivity/run_sensitivity.py --paths 1000 --grid-paths 300 --seed 20261122
```
