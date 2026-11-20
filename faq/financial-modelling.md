# FAQ — Financial Modelling and Scenarios

**LauzHack 2026 | AI Financial Twin**

### 1. What is a financial state?
A structured representation of relevant cash, investments, debts, income, expenses, goals, and life circumstances at a particular time.

### 2. How does the state change?
Through a transition model:

$$
x_{t+1}=F(x_t,a_t,w_t;\theta)
$$

Here $a_t$ is a decision, $w_t$ external conditions, and $\theta$ model assumptions.

### 3. What is a baseline?
A reference scenario used to compare a changed decision or assumption.

### 4. What is a counterfactual?
A conditional comparison such as “What if Alexia studies rather than starts work?” It does not automatically establish a causal effect in real-world data.

### 5. How should we calculate cash?
A basic reconciliation is:

$$
C_{t+1}=C_t+\sum_j CF_{j,t}
$$

Cash inflows and outflows must be classified consistently.

### 6. What is net worth?

$$
NW_t=A_t-L_t
$$

Assets minus liabilities. Net worth is not the same as immediately available cash.

### 7. Do investment transfers count as expenses?
They reduce cash but typically increase investment assets. Avoid treating the transfer as a reduction in total wealth before fees or market movements.

### 8. How do we model loans?
Track principal, periodic interest, new borrowing, payments, and fees separately. Ensure debt balances reconcile.

### 9. How should we model inflation?
State the inflation assumption and distinguish nominal from real values. Do not silently apply inflation to some expenses but not others.

### 10. How do we handle multiple currencies?
Declare a currency conversion convention, date, and exchange-rate source or assumption.

### 11. Can a scenario show negative cash?
Yes, but it should identify a funding gap, borrowing assumption, or infeasible state rather than ignoring it.

### 12. What does a life event change?
Potentially several variables: tuition, income, living costs, employment timing, housing, location, or liabilities.

### 13. How many scenarios are required?
The scenario catalogue is optional inspiration. Demonstrate meaningful comparisons and calculations rather than counting scenario labels.

### 14. What is sensitivity analysis?
Change one or more assumptions and inspect the resulting change in outcomes.

### 15. Is Monte Carlo forecasting?
Not necessarily. Monte Carlo samples assumptions to characterize conditional outcome distributions. Its results depend on the chosen probability model.

### 16. Can we optimize decisions?
Yes, under declared objectives and constraints. Multiple objectives may conflict.

### 17. Is the scenario with the most money always best?
No. Liquidity, education, family, risk, flexibility, and personal goals may matter.

### 18. Can we claim our model is accurate?
Only to the extent supported by validation evidence. Financial accounting correctness does not prove future predictive accuracy.

### 19. How do we validate?
Test cash reconciliation, units, boundary cases, scenario isolation, debt accounting, and reproducibility.

### 20. Where are the detailed equations?
See `docs/scenario-modelling.md` and `evaluation/technical-validation.md`.
