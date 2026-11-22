# 2. State and actions: why these variables and these four choices

**In short.** One step is one month, 120 steps from January 2025 to December 2034. The state holds
the situation (work, pay, home, rent, household, lifestyle, economy), the money (cash and
investments), and the clocks needed for the next month to depend on the state alone. There are four
choices: work, housing, spending and investing. Both lists come from measuring what moves outcomes
in the data, not from taste.

## The process

$$x_{t+1} \sim P(\,\cdot \mid x_t, u_t), \qquad u_t \sim \pi(\,\cdot \mid x_t, h_t, z)$$

$P$ is the world ([03](03-transition-function.md)). $\pi$ is the persona: it sees the state, the
person's own path so far $h_t$ and their traits $z$ ([05](05-persona-policy.md)). Goals are checked
at fixed deadlines: the high-priority goal in December 2029, the medium one in December 2032.

## How we chose the state

Before writing the model, [`action_importance.py`](../analysis/action_importance/action_importance.py)
simulated every profile with a hand-calibrated simulator and asked which starting variables explain
the outcomes (grouped permutation importance of a random forest, 5-fold cross-validation;
[`state_importance.csv`](../analysis/action_importance/results/state_importance.csv)):

| Variable | Share of explained differences in outcomes |
|---|---|
| Pay | 77–94% (largest for net worth) |
| Rent | 4–12% (largest for the chance of running short) |
| Investments | up to 12% (for goals) |
| Cash | about 3% (it decides *when* trouble starts) |
| Age, household, city, debt, loan rate | each under 1%: they matter through eligibility and rent, not directly |

So pay and rent must be **continuous amounts** in the state, not just categories. Household, city
and age stay because they decide what is allowed (a family needs at least a small flat, a master's
only up to 45, pension from 65) and what rent and living costs are.

## The state, grouped by why it is there

**The situation and the money** (the nine variables and two clocks of the
[MDP spec](../analysis/mdp_spec/mdp-spec.html)):

| Variable | Values | Why |
|---|---|---|
| Work mode W | student, part-time, full-time, not working; flags for self-employed and pensioner | Decides pay and which choices exist |
| Net income Y | CHF a month, through a pay rank kept across modes | 77–94% of outcome differences |
| Housing tier H | shared flat, small flat, family flat | The housing choice moves along it |
| Rent R | CHF a month | 4–12% of outcome differences; two people in one tier pay different rents |
| Household F | single, couple, family | Gates the housing tier; a child adds costs |
| Lifestyle L | lean, standard, comfortable | Living costs × 0.9, 1.0, 1.1 |
| Economy M | normal, inflation, recession | Moves returns, cost growth and job risk for everyone at once |
| Cash C | CHF, may go below zero | Below zero is a credit line at 8% a year |
| Investments I | CHF ≥ 0 | One portfolio, returns depend on M |
| Month t | 0 … 119 | Age, December bonus, February and August bills, deadlines |
| Months in mode τ | 0, 1, 2, … | Ends the master's; time out of work matters for job offers |

**Clocks and pending things**, so that next month needs nothing that is not in the state (the
Markov property):

| Variable | Why it must be in the state |
|---|---|
| `search` (looking for part- or full-time work) | An offer can arrive in any month after the choice |
| `study_left`, `delayed` | Months left in the course; a master's can run 6 months late once |
| `degree` | A master's happens at most once per life |
| `flat_search` | A flat search stays open until a flat is found or the next review |
| `side_job` | A student working alongside studies; ends with the course |
| `debt`, `loan_pay`, `loan_rate` | The loan repays on a fixed schedule |
| `P`, `G` | Price and pay indices: drift happens outside the learned network |

**Fixed for each simulated life** (drawn once at the start): the person's pay rank and place in
the rent distribution, their cash-flow offset, the ensemble member that path uses, education and
grades (for the markets), tuition, the persona's cash-buffer target, and the city.

**Derived, never stored:** net worth, monthly outgoings, credit limit, goal progress. They are
computed from the state each time, so two copies of the same balance can never disagree.

## Left out on purpose

85 of the challenge's 120 scenarios run in this MDP: 51 inside a sampled life (an action, an event
or a built-in rule) and 34 as a what-if with one parameter changed. The other 35 each need a new
state variable:

| Extension | Would add | Scenarios | Why it waits |
|---|---|---|---|
| Abroad and currency | country, currency | 10 | The data is Swiss and in CHF only |
| Home ownership | a fourth tier, home value, mortgage | 7 | No property or mortgages in the data |
| Separate debt choices | repay, refinance, new loans | 6 | Changed at most 2% of outcomes in the action screen (below) |
| Self-employment as its own mode | business cash | 5 | Only a flag in the data |
| Pension pots | locked pillar 2 / 3a balance | 4 | Not in the data |
| Other equity | equity pay, rebalancing, tax | 3 | Pay is net and the portfolio is one asset |

## The four choices

| Head | Options | What it does |
|---|---|---|
| **Status** | continue · start a master's · accept part-time · accept full-time · leave paid work | A master's is 24 months of study and tuition, then a higher pay rank. From work, changing hours is immediate; from study or not working, it starts a job search and the job comes when an offer does. |
| **Housing** | one tier cheaper · stay · one tier bigger | Opens a flat search; the new rent is drawn for the new tier at the person's usual rank; CHF 2,000 to move |
| **Spending** | lifestyle down · keep · up | Living costs −10% or +10% |
| **Assets** | sell to refill the cash buffer · hold · invest the cash above the buffer | Moves money between cash and investments; net worth is unchanged |

That is 5 × 3 × 3 × 3 = 135 joint actions. **Masks** remove the options that make no sense: a
master's only once per life and up to age 45; no part-time "switch" from part-time; leaving work
only from work; no flat below the household's minimum; selling only with investments and cash below
the buffer.

**When choices are made.** Every voluntary choice waits for the **January review**, the yearly
moment when people take stock. Events (job loss, offers from an open search, bills, markets,
family) still happen in any month. People rarely change much at a review, so most of the
probability sits on "keep things as they are".

## How we chose the actions: the action screen

The same analysis applied 29 candidate decisions and shocks to all 1,200 profiles (200 paths each,
same random numbers with and without) and counted how often each one materially changed a profile's
outcome: a probability moving by 10 points or more, or net worth by 10% or more
([`action_ranking.csv`](../analysis/action_importance/results/action_ranking.csv)).

| Candidate | Profiles materially changed | Kept as |
|---|---|---|
| Move to a flat 20% cheaper | 52% | Housing −1 |
| Move to the cheapest city | 44% | A what-if (city is context) |
| Move into a shared flat (singles) | 26% of all, 66% of singles | Housing −1 |
| Invest above a buffer, sell below one month | 22% | Assets ±1 plus the automatic top-up |
| Cut living costs 10% | 21% | Spending −1 |
| Full-time master's | 26% of all, 100% of eligible | Status 1 |
| Take a full-time job (students, unemployed) | all 76 eligible | Status 3 |
| Part-time to full-time | all 120 eligible | Status 3 |
| Cut workload to 80% | 66% | Status 2 |
| Have a child, retire early, start a business, buy a car | 13–76% | Events, status 4, or what-ifs |
| Take a CHF 20k loan | 1.9% | **Dropped** |
| Repay debt from cash | 0.9% | **Dropped** (loans repay on schedule) |
| Refinance 2 points cheaper | 0.1% | **Dropped** |
| Loan rate +2 points | 0.25% | **Dropped** |

The screen also found four **loopholes** that an MDP must close, or any decision rule will exploit
them:

| Loophole | How the MDP closes it |
|---|---|
| Free overdraft | Credit costs 8% a year, limit max(3 × monthly pay, CHF 5,000); a breach is flagged and fails the goals |
| A pay rise you can simply choose | Promotions are random events, never an action |
| Spending cuts that cost nothing | Lifestyle has a value to the persona, and inertia makes cuts rare |
| Students who study forever and never work | A master's lasts 24 months, once per life; an opening student's course ends at the first review |

Reproduce: `python analysis/action_importance/action_importance.py` (~2.5 min). The scenario
mapping is in [`analysis/mdp_spec/mdp-spec.html`](../analysis/mdp_spec/mdp-spec.html).
