# Yara (P00004): sensitivity analysis of the current simulator

Seed: 20261122; baseline and individual stresses: 1000 paired paths; joint grid: 300 paired paths; horizon: 120 months.

## Baseline comparison

| Measure | Study intention | Work-search intention |
|---|---:|---:|
| Cash negative at any month end | 82.6% | 3.4% |
| Ever exceeds credit limit | 66.4% | 2.9% |
| High-priority goal at deadline | 14.1% | 18.7% |
| Median minimum cash (CHF) | -8,037 | 13,134 |
| Median terminal net worth (CHF) | 437,587 | 439,469 |
| Mean terminal net worth (CHF) | 428,295 | 445,666 |

Study minus work: shortfall +79.2 pp (conservative paired 95% Monte Carlo interval +75.5 to +82.1 pp); mean terminal net worth CHF -17,371 (paired t interval CHF -29,642 to -5,100).

At the opening review, 100.0% of study-intention paths selected the master’s action and 0.0% of work-intention paths selected the job-search action. Selection does not guarantee admission, employment, or completion.

## One-at-a-time changes to the study-intention baseline

| Change | Shortfall | Difference vs baseline (pp; paired 95% interval) | Goal | Mean final wealth change CHF (paired 95% interval) |
|---|---:|---:|---:|---:|
| Tuition CHF 2,500 | 34.6% | -48.0 (-51.9 to -43.6) | 43.0% | +16,574 (+10,145 to +23,004) |
| Tuition CHF 12,000 | 91.5% | +8.9 (+6.3 to +11.4) | 2.5% | -1,386 (-7,094 to +4,322) |
| Degree pay-rank shift 0.00 | 83.3% | +0.7 (-0.2 to +1.6) | 5.1% | -150,863 (-156,779 to -144,946) |
| Degree pay-rank shift 0.10 | 83.2% | +0.6 (-0.3 to +1.4) | 8.5% | -75,342 (-79,113 to -71,571) |
| Job-offer probability ×0.5 | 87.6% | +5.0 (+3.1 to +6.9) | 9.5% | -26,021 (-34,332 to -17,711) |
| Job-offer probability ×1.5 | 80.5% | -2.1 (-3.5 to -0.7) | 17.3% | +11,416 (+5,525 to +17,307) |
| Study-delay probability 0% | 82.0% | -0.6 (-1.4 to +0.3) | 15.6% | +5,472 (+2,229 to +8,714) |
| Study-delay probability 25% | 83.6% | +1.0 (+0.0 to +2.0) | 11.1% | -8,970 (-13,705 to -4,236) |
| Credit rate 4% | 82.6% | +0.0 (-0.5 to +0.5) | 14.2% | +137 (-574 to +848) |
| Credit rate 12% | 82.6% | +0.0 (-0.5 to +0.5) | 14.1% | -579 (-775 to -383) |
| Moving cost CHF 1,000 | 82.6% | +0.0 (-0.5 to +0.5) | 14.2% | +159 (-534 to +852) |
| Moving cost CHF 4,000 | 82.8% | +0.2 (-0.5 to +0.9) | 13.9% | -1,107 (-1,336 to -877) |
| All-regime price growth 0% | 80.1% | -2.5 (-3.9 to -1.1) | 17.3% | +21,033 (+19,530 to +22,537) |
| All-regime price growth 3% | 86.4% | +3.8 (+1.9 to +5.7) | 9.1% | -35,829 (-38,429 to -33,228) |

## Funding grid

| Added starting cash CHF | Negative-cash rate | Change vs baseline pp (paired 95% interval) |
|---:|---:|---:|
| 0 | 82.6% | +0.0 (-0.5 to +0.5) |
| 5,000 | 63.5% | -19.1 (-22.0 to -16.0) |
| 10,000 | 38.9% | -43.7 (-47.3 to -39.6) |
| 20,000 | 12.6% | -70.0 (-73.2 to -66.1) |
| 40,000 | 3.9% | -78.7 (-81.5 to -75.1) |

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
python -B analysis/sensitivity/run_sensitivity.py --profile P00004 --paths 1000 --grid-paths 300 --seed 20261122
```
