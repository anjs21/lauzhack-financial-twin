
# Scenario Modelling — Mathematical Framework and Scenario Library

**UBS × LauzHack 2026 | AI Financial Twin Challenge**

Advanced technical reference for financial state modelling, counterfactual simulation, scenario exploration, optimization, uncertainty, and decision support.

---

## 1. What Is Financial Twin Scenario Modelling?

A Financial Twin is a stateful, assumption-explicit computational representation of an individual's financial situation.

It enables users to explore the consequences of decisions under different assumptions and constraints.

**A Financial Twin is not a forecasting product.**

The fundamental question is:

> What would happen to my financial situation if I made a particular decision under a specified set of assumptions?

This differs from:

> What will my financial future be?

A Financial Twin may incorporate forecasting algorithms, statistical estimation, Monte Carlo simulation, machine learning, reinforcement learning, optimization, or other computational methods.

However, these methods are components of a possible implementation. None individually defines a Financial Twin.

### Core Capabilities

A Financial Twin may support:

1. Representation of a financial state
2. Modelling of financial relationships
3. Simulation of decisions and life events
4. Comparison of alternative scenarios
5. Analysis of constraints and trade-offs
6. Exploration of uncertainty
7. Optimization of decision strategies
8. Explanation of outcomes
9. Updating the representation when information changes
10. Transparent and reproducible calculations

The objective is not to produce one supposedly correct future.

It is to make financial possibilities explorable.

---

## 2. Mathematical Definition of a Financial Twin

Let time be discrete:

$$
t = 0,1,\ldots,T
$$

The interval between two periods may represent a day, month, quarter, or year.

The selected time step must remain consistent throughout the model.

Define the financial state vector:

$$
x_t =
(C_t,I_t,D_t,Y_t,E_t,H_t,G_t,L_t,J_t,R_t,Z_t)
$$

Where:

| Variable | Definition |
|---|---|
| $C_t$ | Liquid cash and savings |
| $I_t$ | Investment holdings |
| $D_t$ | Outstanding debt |
| $Y_t$ | Income streams |
| $E_t$ | Expenditure streams |
| $H_t$ | Housing and property position |
| $G_t$ | Financial goals and progress |
| $L_t$ | Location, currency, and tax context |
| $J_t$ | Employment and education status |
| $R_t$ | Risk preferences and financial constraints |
| $Z_t$ | Additional state, including dependents, insurance, and pensions |

The system evolves according to a transition function:

$$
x_{t+1}=F(x_t,a_t,w_t;\theta)
$$

Where:

- $x_t$ is the financial state at time $t$.
- $a_t$ represents a decision or action.
- $w_t$ represents external conditions.
- $\theta$ represents modelling assumptions.

Examples of decisions include:

- Pursuing a master's degree
- Accepting employment
- Relocating
- Purchasing property
- Increasing investment contributions
- Taking a career break
- Starting a business
- Repaying debt

Examples of external conditions include:

- Interest rates
- Inflation
- Investment returns
- Exchange rates
- Employment-market conditions
- Regulatory changes

The model must distinguish user-controlled decisions from external conditions.

### Observation Model

The available data may not reveal the complete financial state.

An observation model can be represented as:

$$
o_t=H(x_t,d_t)
$$

Where:

- $o_t$ represents observed financial information.
- $d_t$ represents available data.
- $H$ maps the underlying state to observable information.

A robust implementation should retain information about data provenance, timestamps, units, and assumptions.

---

## 3. Cash-Flow Modelling

Cash flow describes money entering and leaving the financial system.

For a given period:

$$
CF_t =
Y_t^{net}
+
O_t^{in}
-
E_t
-
T_t
-
DS_t
-
IC_t
-
K_t
+
IW_t
$$

Where:

| Variable | Definition |
|---|---|
| $Y_t^{net}$ | Net employment and other regular income |
| $O_t^{in}$ | Other cash inflows |
| $E_t$ | Living and operating expenses |
| $T_t$ | Taxes not already included in net income |
| $DS_t$ | Debt service |
| $IC_t$ | Investment contributions |
| $K_t$ | One-time costs |
| $IW_t$ | Investment withdrawals |

The liquid cash balance evolves according to:

$$
C_{t+1}=C_t+CF_t+R_t^{cash}
$$

Where $R_t^{cash}$ represents interest earned on cash during the period.

### Example

Assume the following synthetic monthly values:

| Component | CHF |
|---|---:|
| Net income | 4,500 |
| Living expenses | 2,700 |
| Debt service | 200 |
| Investment contributions | 500 |
| Other costs | 100 |

Then:

$$
CF_t=4500-2700-200-500-100
$$

$$
CF_t=1000
$$

The monthly cash surplus is CHF 1,000.

If opening cash is CHF 8,000 and no cash interest is assumed:

$$
C_{t+1}=8000+1000=9000
$$

### Important Accounting Rules

- Do not deduct taxes twice.
- Distinguish gross income from net income.
- Investment transfers must not be counted as expenditure that destroys wealth.
- Debt principal repayments reduce both cash and liabilities.
- All financial values must use consistent currencies and time periods.
- Negative cash balances must be explained through borrowing, withdrawals, or infeasibility.

---

## 4. Savings and Liquidity

Savings represent accumulated financial resources.

Liquidity represents the availability of funds to meet financial obligations.

A simplified savings model is:

$$
S_{t+1}=S_t+Y_t-E_t
$$

A model with interest and additional transfers can be represented as:

$$
S_{t+1}=S_t(1+r_t)+A_t-W_t
$$

Where:

- $r_t$ is the periodic interest rate.
- $A_t$ is an additional deposit.
- $W_t$ is a withdrawal.

### Liquidity Constraints

A user may require a minimum emergency reserve:

$$
C_t \geq C_{\min}
$$

Where $C_{\min}$ is a user-defined minimum cash balance.

A scenario violating this constraint should be identified as financially infeasible under the specified assumptions.

---

## 5. Investment Modelling

An investment portfolio may contain several assets.

Let:

- $q_{k,t}$ represent the quantity of asset $k$.
- $P_{k,t}$ represent its price.
- $b_{k,t}$ represent purchases.
- $s_{k,t}$ represent sales.

The asset quantity evolves as:

$$
q_{k,t+1}=q_{k,t}+b_{k,t}-s_{k,t}
$$

The total investment portfolio value is:

$$
I_t=\sum_{k=1}^{K}q_{k,t}P_{k,t}
$$

### Simplified Investment Account

Assuming contributions and withdrawals occur at the end of the period:

$$
I_{t+1}=I_t(1+r_t)+A_t-W_t-F_t
$$

Where:

- $r_t$ is the assumed investment return.
- $A_t$ is the investment contribution.
- $W_t$ is the withdrawal.
- $F_t$ represents fees.

### Compound Growth

For a constant periodic return:

$$
FV=PV(1+r)^n
$$

Where:

- $FV$ is future value.
- $PV$ is present value.
- $r$ is the assumed periodic return.
- $n$ is the number of periods.

For periodic contributions made at the end of each period:

$$
FV=PV(1+r)^n+A\frac{(1+r)^n-1}{r}
$$

For $r=0$:

$$
FV=PV+nA
$$

These are conditional calculations, not guaranteed investment outcomes.

---

## 6. Debt and Mortgage Modelling

Debt evolves according to:

$$
D_{t+1}=D_t(1+i_t)+B_t-P_t
$$

Where:

- $D_t$ is outstanding debt.
- $i_t$ is the periodic interest rate.
- $B_t$ is new borrowing.
- $P_t$ is the total payment, including interest and principal.

For a fixed-rate amortizing loan:

$$
A=\frac{D_0i}{1-(1+i)^{-n}}
$$

Where:

- $A$ is the periodic payment.
- $D_0$ is initial principal.
- $i$ is the periodic interest rate.
- $n$ is the number of payments.

For zero interest:

$$
A=\frac{D_0}{n}
$$

A housing scenario may also need to include:

- Down payment
- Property transaction costs
- Mortgage affordability
- Maintenance
- Insurance
- Taxes
- Property valuation
- Amortization
- Alternative rental costs

Mortgage structures and legal requirements vary by jurisdiction.

---

## 7. Net Worth

Net worth measures the difference between assets and liabilities.

$$
NW_t=A_t-L_t
$$

A more detailed representation is:

$$
NW_t=C_t+I_t+H_t+O_t-D_t-L_t^{other}
$$

Where:

- $C_t$ is cash.
- $I_t$ is investments.
- $H_t$ is property value.
- $O_t$ is other assets.
- $D_t$ is debt.
- $L_t^{other}$ is other liabilities.

A Financial Twin should distinguish net worth from liquidity.

For example, buying property may reduce available cash without necessarily reducing net worth by the same amount.

---

## 8. Inflation and Purchasing Power

Nominal financial values do not directly represent purchasing power.

Inflation-adjusted value can be calculated using:

$$
V_t^{real}=V_t^{nominal}\frac{CPI_0}{CPI_t}
$$

Where $CPI_t$ is the consumer price index.

Under an assumed constant annual inflation rate:

$$
CPI_t=CPI_0(1+\pi)^t
$$

Therefore:

$$
V_t^{real}=\frac{V_t^{nominal}}{(1+\pi)^t}
$$

The inflation assumption should be disclosed.

---

## 9. Life-Event Modelling

Life events should be represented as changes to the financial state.

Examples include:

- Graduation
- Employment
- Unemployment
- Further education
- Relocation
- Marriage
- Parenthood
- Property purchase
- Entrepreneurship
- Retirement

A life event may affect several variables simultaneously.

For example, a master's degree may affect:

1. Tuition costs
2. Employment income
3. Working hours
4. Living expenses
5. Graduation timing
6. Future employment opportunities

The model should represent these effects separately rather than arbitrarily adjusting a final savings figure.

### Event Transition

A general event-based transition is:

$$
x_{t+1}=F_e(x_t,a_t,\theta_e)
$$

Where $F_e$ represents the transition associated with event $e$.

Events may be conditional on dates, eligibility, available resources, or other constraints.

---

## 10. Counterfactual Scenario Analysis

Counterfactual analysis compares outcomes under alternative decisions.

Let:

$$
x_T^{(0)}
$$

represent the outcome under a baseline decision.

Let:

$$
x_T^{(1)}
$$

represent the outcome under an alternative decision.

The scenario difference is:

$$
\Delta x_T=x_T^{(1)}-x_T^{(0)}
$$

For a selected financial measure $M$:

$$
\Delta M_T=M(x_T^{(1)})-M(x_T^{(0)})
$$

For example:

- Baseline: Start working immediately.
- Alternative: Pursue a master's degree.

The model can compare liquidity, net worth, income, and progress toward financial goals.

A modelled difference should not automatically be described as a proven causal effect.

---

## 11. Uncertainty and Sensitivity Analysis

Financial decisions depend on assumptions that may change.

Relevant variables include:

- Income growth
- Employment timing
- Tuition fees
- Living costs
- Inflation
- Investment returns
- Interest rates
- Exchange rates
- Unexpected expenses

### Sensitivity Analysis

Let the outcome measure be:

$$
M=f(\theta_1,\theta_2,\ldots,\theta_n)
$$

The sensitivity to parameter $\theta_i$ may be examined through:

$$
\frac{\partial M}{\partial\theta_i}
$$

Numerical finite differences can also be used.

### Scenario Ranges

Students may define alternative parameter sets:

- Lower-cost scenario
- Reference scenario
- Higher-cost scenario
- Adverse economic scenario

These represent conditional possibilities, not predictions.

---

## 12. Monte Carlo Simulation

Monte Carlo simulation uses repeated sampling from specified probability distributions.

It may help explore uncertainty within a Financial Twin.

It does not define a Financial Twin and is not synonymous with forecasting.

Let:

$$
w^{(k)}\sim P(w)
$$

represent the $k$-th sampled external-condition path.

For each sample:

$$
x_{t+1}^{(k)}=F(x_t^{(k)},a_t,w_t^{(k)};\theta)
$$

The resulting simulated outcomes may be used to calculate:

- Conditional outcome distributions
- Liquidity shortfall frequencies
- Quantiles
- Tail risks
- Sensitivities

For an outcome $M$, the sample mean is:

$$
\bar{M}=\frac{1}{N}\sum_{k=1}^{N}M^{(k)}
$$

The results depend on the assumed distributions, dependencies, and simulation design.

Students should report these assumptions and avoid presenting simulated probabilities as objectively known future probabilities.

---

## 13. Optimization and Decision Support

A Financial Twin may explore decisions subject to constraints.

A general optimization problem is:

$$
\max_{\{a_t\}}\sum_{t=0}^{T}\beta^t u(x_t,a_t)
$$

Where:

- $a_t$ represents decisions.
- $\beta$ is a discount factor.
- $u$ is a user-defined utility function.

Potential objectives include:

- Maximizing available liquidity
- Reducing debt burden
- Reaching an education goal
- Preserving financial flexibility
- Improving long-term financial resilience

### Multi-Objective Optimization

$$
\max_a\sum_{m=1}^{M}\lambda_m f_m(a)
$$

Where $\lambda_m$ represents the weight assigned to objective $m$.

These weights express preferences and value judgments.

They should not be hidden from users.

Students may also explore Pareto frontiers instead of reducing all objectives to one score.

---

## 14. Financial Constraints

Financial decisions may be subject to constraints.

### Minimum Liquidity

$$
C_t\geq C_{\min}
$$

### Debt-Service Ratio

$$
\frac{DS_t}{Y_t}\leq d_{\max}
$$

### Budget Constraint

$$
E_t+DS_t+IC_t\leq Y_t+C_t^{available}
$$

### Goal Deadline

$$
t_{goal}\leq T_{goal}
$$

These thresholds are illustrative model parameters.

They are not universal recommendations or UBS lending policies.

---

## 15. Alternative Modelling Paradigms

Participants may use any appropriate mathematical or computational approach.

### Deterministic Simulation

Evaluates financial outcomes under fixed assumptions.

### Discrete-Event Simulation

Models events such as graduation, employment, relocation, or loan changes.

### System Dynamics

Explores feedback relationships between financial variables.

### Agent-Based Modelling

Models interacting individuals, institutions, employers, markets, or households.

### Stochastic Simulation

Explores uncertainty using explicitly defined distributions or scenario trees.

### Causal Modelling

Examines interventions using causal assumptions.

A key distinction is:

$$
P(Y\mid X)\neq P(Y\mid do(X))
$$

Observing a relationship is not the same as identifying the effect of an intervention.

### Reinforcement Learning

Explores sequential decisions through interaction with a simulated environment.

### Bayesian Methods

Represent and update uncertainty using probability models.

### Optimization

Searches for feasible decisions under specified constraints.

### Hybrid Quantum-Classical Methods

Investigates selected computational subproblems using quantum and classical components.

No practical quantum advantage should be assumed without evidence.

---

## 16. Alexia — Illustrative Client

Alexia is a fictional 20-year-old student living in Lausanne.

Initial liquid savings:

**CHF 8,000**

She is considering three decisions:

1. Start working immediately
2. Pursue a master's degree
3. Move abroad

The simulation horizon is age 20 to age 30.

Each scenario may affect:

- Employment income
- Education expenses
- Living costs
- Savings
- Liquidity
- Career progression
- Financial goals

The starting balance alone is insufficient to calculate ten-year outcomes.

Students must specify the remaining financial assumptions before generating numerical results.

---

## 17. Scenario Library — 120 Interventions

The following scenarios are research and prototyping prompts.

They are not precomputed forecasts or validated financial recommendations.

### Education and Skills

| ID | Scenario | Intervention |
|---|---|---|
| S001 | Full-time master's | Enrol for two years instead of starting work |
| S002 | Part-time master's | Work 60% while studying |
| S003 | Scholarship | Receive an education grant |
| S004 | Tuition increase | Increase annual tuition |
| S005 | Internship year | Complete a paid internship |
| S006 | Reskilling | Pay for a professional bootcamp |
| S007 | Doctoral studies | Enter a funded doctoral programme |
| S008 | Study abroad | Spend one semester overseas |
| S009 | Student loan | Finance tuition through borrowing |
| S010 | Delayed graduation | Graduate one year later |

### Career and Employment

| ID | Scenario | Intervention |
|---|---|---|
| S011 | Early employment | Start working instead of studying |
| S012 | Salary negotiation | Negotiate a higher starting salary |
| S013 | Promotion | Receive a promotion after two years |
| S014 | Unemployment | Experience a six-month employment gap |
| S015 | Freelancing | Replace salaried work with freelancing |
| S016 | Career switch | Change professional field |
| S017 | Four-day week | Reduce working hours to 80% |
| S018 | Equity compensation | Accept equity-heavy remuneration |
| S019 | Contract employment | Move to a fixed-term contract |
| S020 | Second income | Add part-time employment |

### Relocation and Geography

| ID | Scenario | Intervention |
|---|---|---|
| S021 | Move to Zurich | Relocate from Lausanne |
| S022 | Move to Geneva | Change city and cost structure |
| S023 | Move to Berlin | Relocate to Germany |
| S024 | Move to London | Relocate to the United Kingdom |
| S025 | Move to New York | Relocate to the United States |
| S026 | Cross-border employment | Work in another jurisdiction |
| S027 | Currency exposure | Change income and expense currencies |
| S028 | Return to Switzerland | Relocate after working abroad |
| S029 | Remote work | Work remotely from another country |
| S030 | Relocation support | Receive employer relocation assistance |

### Housing and Property

| ID | Scenario | Intervention |
|---|---|---|
| S031 | Rent alone | Move from shared housing |
| S032 | Shared apartment | Reduce housing expenses |
| S033 | Buy first property | Purchase a home |
| S034 | Mortgage-rate change | Change financing costs |
| S035 | Larger down payment | Increase initial equity |
| S036 | Property maintenance | Increase ongoing property costs |
| S037 | Property price change | Alter purchase price |
| S038 | Rent increase | Increase recurring rent |
| S039 | Housing relocation | Move to a cheaper property |
| S040 | Early mortgage repayment | Accelerate debt repayment |

### Spending and Lifestyle

| ID | Scenario | Intervention |
|---|---|---|
| S041 | Transport choice | Replace car ownership with public transport |
| S042 | Car purchase | Purchase and maintain a vehicle |
| S043 | Travel spending | Increase annual travel budget |
| S044 | Lifestyle inflation | Increase discretionary spending |
| S045 | Budget reduction | Reduce recurring expenses |
| S046 | Insurance change | Adjust insurance coverage and cost |
| S047 | Subscription reduction | Cancel recurring subscriptions |
| S048 | Healthcare expense | Introduce an unexpected medical cost |
| S049 | Large purchase | Finance a major one-time purchase |
| S050 | Spending discipline | Implement a monthly spending cap |

### Savings and Liquidity

| ID | Scenario | Intervention |
|---|---|---|
| S051 | Emergency fund | Build a minimum cash reserve |
| S052 | Higher savings rate | Increase monthly saving |
| S053 | Savings interruption | Stop contributions temporarily |
| S054 | Cash interest | Change savings-account interest |
| S055 | Liquidity shock | Introduce a sudden expense |
| S056 | Cash allocation | Change cash versus investment allocation |
| S057 | Goal-based saving | Save for a defined purchase |
| S058 | Irregular income | Introduce variable monthly income |
| S059 | Family support | Receive or provide financial support |
| S060 | Savings withdrawal | Use accumulated savings |

### Investments and Allocation

| ID | Scenario | Intervention |
|---|---|---|
| S061 | Start investing | Begin periodic investments |
| S062 | Increase contributions | Raise monthly investment amounts |
| S063 | Portfolio reallocation | Change asset allocation |
| S064 | Investment fees | Change annual fees |
| S065 | Market downturn | Apply an adverse market scenario |
| S066 | Market recovery | Explore recovery assumptions |
| S067 | Diversification | Change portfolio concentration |
| S068 | Investment withdrawal | Withdraw part of a portfolio |
| S069 | Risk preference | Change investment constraints |
| S070 | Investment horizon | Extend or shorten the horizon |

### Debt and Credit

| ID | Scenario | Intervention |
|---|---|---|
| S071 | Student debt | Take an education loan |
| S072 | Debt repayment | Accelerate loan repayment |
| S073 | Interest-rate increase | Raise borrowing costs |
| S074 | Debt consolidation | Combine existing loans |
| S075 | Credit-card balance | Introduce revolving debt |
| S076 | Personal loan | Finance a large expense |
| S077 | Loan refinancing | Change financing terms |
| S078 | Payment holiday | Temporarily defer payments |
| S079 | New borrowing | Increase debt principal |
| S080 | Debt-free target | Set a debt-elimination goal |

### Family and Life Events

| ID | Scenario | Intervention |
|---|---|---|
| S081 | Marriage | Combine household finances |
| S082 | Parenthood | Add childcare and household costs |
| S083 | Parental leave | Reduce employment income temporarily |
| S084 | Supporting parents | Add recurring family support |
| S085 | Household separation | Split household finances |
| S086 | Inheritance | Receive a one-time asset transfer |
| S087 | Care responsibilities | Reduce work hours |
| S088 | Household relocation | Move with dependents |
| S089 | Education for children | Create an education savings goal |
| S090 | Household income change | Change partner income |

### Retirement and Pensions

| ID | Scenario | Intervention |
|---|---|---|
| S091 | Early retirement | Retire before the reference age |
| S092 | Later retirement | Extend employment |
| S093 | Pension contributions | Change contribution amounts |
| S094 | Career interruption | Introduce pension contribution gaps |
| S095 | Retirement spending | Change retirement expenses |
| S096 | Partial retirement | Reduce working hours gradually |
| S097 | Retirement relocation | Change country of residence |
| S098 | Retirement drawdown | Change withdrawal strategy |
| S099 | Pension uncertainty | Explore alternative pension assumptions |
| S100 | Retirement goal | Set a target financial reserve |

### Macroeconomic and Systemic Shocks

| ID | Scenario | Intervention |
|---|---|---|
| S101 | Inflation shock | Increase living-cost inflation |
| S102 | Interest-rate shock | Increase borrowing rates |
| S103 | Currency depreciation | Change exchange-rate assumptions |
| S104 | Recession | Introduce income and employment stress |
| S105 | Market crash | Apply a severe investment loss |
| S106 | Energy-price shock | Increase utility expenses |
| S107 | Tax change | Modify tax assumptions |
| S108 | Wage stagnation | Remove salary growth |
| S109 | Housing-cost shock | Increase housing expenses |
| S110 | Combined stress | Apply multiple adverse conditions |

### Entrepreneurship and Innovation

| ID | Scenario | Intervention |
|---|---|---|
| S111 | Start a business | Leave employment to launch a venture |
| S112 | Side business | Add entrepreneurial income |
| S113 | Startup investment | Invest personal savings in a venture |
| S114 | Business loan | Finance operations with borrowing |
| S115 | Revenue delay | Delay expected business revenue |
| S116 | Business failure | Model venture closure |
| S117 | Business growth | Increase revenue and operating costs |
| S118 | Co-founder | Share ownership and costs |
| S119 | Funding round | Introduce external financing |
| S120 | Business sale | Model an exit transaction |

---

## 18. Combining Multiple Scenarios

Financial decisions rarely occur in isolation.

Students may combine multiple interventions.

For example:

$$
S001+S023+S111
$$

Could represent:

- Pursuing a master's degree
- Relocating to Berlin
- Starting a business

The combined model should account for interactions between events.

Scenario combinations should not be treated as independent when their financial effects overlap.

---

## 19. Verification and Validation

Every Financial Twin should be tested for internal consistency.

### Accounting Validation

Asset transfers should preserve net worth before applicable fees and taxes.

### Unit Validation

Check consistency between:

- CHF and other currencies
- Monthly and annual values
- Gross and net income
- Nominal and real values

### Boundary Testing

Consider:

- Zero income
- Zero interest
- Negative investment returns
- Missing data
- Negative liquidity
- Unexpected expenses
- Early loan repayment

### Sensitivity Testing

A higher tuition cost should not increase available cash when all other assumptions remain unchanged.

### Reproducibility

Identical deterministic inputs should produce identical results.

Stochastic simulations should document random seeds, distributions, and the number of simulations.

### Explainability

Users should be able to distinguish:

- Observed information
- Assumptions
- Calculated outcomes
- Uncertainty
- Recommendations, if any

---

## 20. Scenario Reporting Template

Each scenario should document:

| Field | Description |
|---|---|
| Scenario ID | Unique identifier |
| Question | User's financial question |
| Initial state | Starting financial profile |
| Intervention | Decision or event |
| Baseline | Comparison scenario |
| Time horizon | Simulation duration |
| Currency | Financial units |
| Jurisdiction | Relevant country or tax context |
| Assumptions | Explicit parameters |
| Model | Selected computational method |
| Constraints | Financial or personal limitations |
| Outputs | Calculated financial measures |
| Uncertainty | Sensitivity or stochastic assumptions |
| Validation | Checks performed |
| Limitations | Known model limitations |

A good Financial Twin should make the reasoning behind its outputs understandable.

---

## 21. Public Data and Research Resources

### Swiss Financial and Economic Data

- Swiss Federal Statistical Office: https://www.bfs.admin.ch/
- Swiss National Bank: https://www.snb.ch/
- Swiss Open Government Data: https://opendata.swiss/

### International Data

- World Bank: https://data.worldbank.org/
- FRED: https://fred.stlouisfed.org/
- Federal Reserve Survey of Consumer Finances: https://www.federalreserve.gov/econres/scfindex.htm

### Digital Twin Research

- National Academies: https://www.nationalacademies.org/read/26894/chapter/4
- NASA Digital Twin Paradigm: https://ntrs.nasa.gov/api/citations/20120008178/downloads/20120008178.pdf

### Computational Resources

- NumPy: https://numpy.org/
- SciPy: https://scipy.org/
- SimPy: https://simpy.readthedocs.io/
- PyTorch: https://pytorch.org/
- Scikit-learn: https://scikit-learn.org/
- Google OR-Tools: https://developers.google.com/optimization
- Gymnasium: https://gymnasium.farama.org/
- PennyLane: https://pennylane.ai/
- Qiskit: https://www.ibm.com/quantum/qiskit

Participants are responsible for checking licensing, data provenance, and applicability to their chosen financial scenarios.

---

## 22. Responsible Development

Participants should:

- Use synthetic or appropriately licensed public data.
- Never use real UBS client data.
- Avoid publishing credentials or API keys.
- Document assumptions and limitations.
- Distinguish scenario outputs from predictions.
- Validate financial calculations.
- Communicate uncertainty.
- Avoid unsupported personalized financial advice.
- Respect applicable intellectual-property and data-usage requirements.

---

## Final Principle

**A Financial Twin should not tell people that it knows their future.**

It should provide a transparent computational environment in which they can explore decisions, understand constraints, compare possibilities, and make more informed choices.

**Don't predict the future. Make it explorable.**


