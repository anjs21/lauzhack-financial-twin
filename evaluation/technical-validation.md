# Technical and Financial Validation Checklist

**LauzHack 2026 | Suggested test guidance, not a mandated stack**

## 1. Reproducibility

- Can the same deterministic inputs reproduce the same results?
- Are random seeds and sampling assumptions recorded for stochastic methods?
- Are the time horizon, currency, and units visible?

## 2. Cash-flow identities

For a period, with all transfers and charges correctly classified:

$$
C_{t+1} = C_t + \sum_{j=1}^{N_t} CF_{j,t}
$$

Test that opening cash plus net transactions equals closing cash.

**Example:** CHF 8,000 opening cash + CHF 1,200 income − CHF 1,800 expenses = CHF 7,400 closing cash, absent other flows.

## 3. Asset and liability consistency

$$
NW_t = A_t - L_t
$$

A CHF 500 transfer from cash into investments should not reduce net worth by CHF 500, before fees and market movements.

Debt principal repayment should reduce cash and debt; interest is an expense.

## 4. Scenario isolation

- Change one assumption while keeping others fixed.
- Verify that the appropriate outputs change.
- Avoid accidental changes to unrelated scenario parameters.
- Document interactions when combining multiple interventions.

## 5. Boundary and stress cases

| Test | Expected handling |
|---|---|
| Zero income | Cash-flow and liquidity effects are visible |
| Zero interest rate | No divide-by-zero errors |
| Negative investment return | Portfolio value responds consistently |
| Higher tuition | All else equal, education costs do not fall |
| Loan paid off | Outstanding principal does not become negative without explanation |
| Missing input | Request clarification or apply a labelled assumption |
| Negative cash | Flag financing need or infeasibility |
| Multiple currencies | Apply explicit exchange-rate convention |
| Time-step change | Annual/monthly conversion is consistent |

## 6. Financial model integrity

Check for:
- Gross vs net income confusion.
- Double-counted tax or social contributions.
- Cash-to-investment transfers treated as consumption.
- Inflation applied inconsistently.
- Loan payments confused with principal repayment.
- Asset snapshots incorrectly interpreted as transaction histories.
- Synthetic data mistaken for verified Swiss population statistics.

## 7. Uncertainty methods

If using Monte Carlo, Bayesian models, or stochastic simulations:
- Specify distributions and dependencies.
- Explain the meaning of percentiles.
- Document sample counts and seeds.
- Avoid implying that modelled probabilities are known real-world probabilities.

Monte Carlo is an **optional uncertainty technique**, not a required forecasting component.

## 8. AI, game and agent validation

- LLM output agrees with the calculation engine.
- Game actions update financial state consistently.
- Agents cannot bypass declared constraints without explanation.
- Optimization results are checked for feasibility.
- Reinforcement-learning reward functions are disclosed.
- Quantum claims are benchmarked against a suitable classical baseline.

## 9. Evidence for judges

Show at least:
1. Input financial profile.
2. One intervention.
3. Assumptions.
4. Calculated comparison.
5. One sanity test or validation result.
6. Limitations.

This is guidance, not an additional official submission rule.
