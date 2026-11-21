# Action importance for the Financial Twin MDP

Answers one question: **which actions, shocks and state variables does the MDP need?**

`action_importance.py` recovers the rule behind the synthetic ledger and calibrates a stochastic monthly simulator from it. It back-tests that simulator on 2026 using 2025 data only. It then applies 29 candidate decisions and shocks to all 1,200 profiles over 2025-01..2034-12 (200 paths each, common random numbers) and ranks them by how often they materially change a profile's outcome.

It complements `../analyze_mdp.py`, which replays the recorded ledger, and keeps the accounting contract in `../mdp_design.json`.

```bash
python analysis/action_importance/action_importance.py   # ~2.5 min; numpy, pandas, scipy, scikit-learn
```

Every number that is not in the data sits in the dict `A` at the top of the script. Change it and re-run. `SENSITIVITY` holds the pessimistic and optimistic sets used for the robustness check.

`mdp-action-shortlist.html` is the written-up report: a standalone page you can open in any browser. It embeds a snapshot of `results/` taken on 2026-11-21 and does not update when you re-run the script.

## Outputs (`results/`)

| File | Contents |
|---|---|
| `calibration.json` | Transition parameters for the environment: pay drift, cost inflation, noise by status, one-off bills, December bonus, student fees, rent multipliers, income by status, plus all assumptions |
| `backtest.json` | Calibrate on 2025, predict 2026: 61.2 expected vs 61 recorded shortfall profiles; 50% and 90% bands cover 51% and 89% |
| `action_ranking.csv` | One row per intervention: eligible, better/worse counts, shortfalls fixed, goals gained/lost, mean deltas, tier, character |
| `action_by_status.csv` | The same effects split by employment status |
| `action_effects_by_profile.csv` | Per-profile deltas for every intervention |
| `baseline_by_profile.csv` | Shortfall risk, goal odds and net worth if nothing changes |
| `best_financial_lever_by_status.csv` | Which lever is best per profile (lowest risk, then goals, then wealth) |
| `state_importance.csv` | Permutation importance of starting-state variables for baseline outcomes |
| `sensitivity_tiers.csv` | Tiers under pessimistic, reference and optimistic assumptions |

## Headline results

- **Agent levers:** work (job or full-time), housing cost (cheaper flat, shared flat, cheaper city), cash buffer as a closed-loop rule (invest above it, sell below one month), and spending level.
- **User-chosen branches:** master's, business, 80% workload, early retirement, child, car. None of them pays off in cash terms, so the reward needs a preference term.
- **Shocks:** costs outpacing pay, rent rises, spells out of work, market drawdowns, emergency bills.
- **Drop:** early loan repayment, refinancing, new personal loans and rate rises each affect at most 1.9% of profiles.
- **State:** pay explains 77–94% of outcome differences, rent 4–12%, investments up to 12% (goals), cash about 3% (timing). Age, household, city and debt matter only for eligibility and rent.
- **Loopholes to close:** free overdraft (charge interest; 8%/yr is used here), certain pay rises and costless spending cuts (make them stochastic or costly), and students' do-nothing path of studying forever.

Synthetic data only. Results describe this simulator under the stated assumptions, not real-world effects.
