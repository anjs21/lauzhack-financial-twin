# 1. The data: what it tells us and what it cannot

**In short.** The challenge gives 1,200 fictional people and their monthly cash for two years. It
shows very clearly what a month of money looks like *in a given situation*. It never shows anyone
*changing* situation: nobody takes a job, starts a degree, moves or has a child. Everything else in
the design follows from that one fact.

## What is in it

| File | Rows | What we use it for |
|---|---|---|
| `synthetic-profiles.csv` | 1,200 | Age, city, work status, household, reference pay and rent, opening cash |
| `synthetic-cash-balances.csv` | 28,800 | Opening cash, net transactions and closing cash for every person and month, 2025-01 to 2026-12 |
| `synthetic-assets.csv` | 2,399 | Opening investment portfolio |
| `synthetic-liabilities.csv` | 283 | Loan principal and interest rate (one loan per person at most) |
| `synthetic-goals.csv` | 1,500 | One high-priority goal per person (due Dec 2029); 300 people also have a medium one (due Dec 2032) |
| `scenario-assumptions.csv` | 240 | Master's study plans (scenario S001): 24 months, tuition CHF 1,200–8,000 a year |
| `scenario-catalog.csv` | 120 | The challenge's scenario ideas; we map each one to the MDP in [02](02-state-and-actions.md) |

Who is in it: 682 employed, 106 self-employed, 120 part-time, 216 retired, 39 students and 37
unemployed. Opening cash has a median of CHF 53k; monthly net flow per person has a median of
CHF 2,239.

## We checked the data before using it

[`analysis/analyze_mdp.py`](../analysis/analyze_mdp.py) audits every file. All checks pass:
closing cash = opening cash + net transactions in every row, months are continuous, each person has
exactly 24 consecutive months, ids are unique, every row links to a profile, and the "negative
balance" flag matches the sign of closing cash.

## The rules hidden in the ledger

[`analysis/action_importance/action_importance.py`](../analysis/action_importance/action_importance.py)
recovers how the ledger was generated (results in
[`calibration.json`](../analysis/action_importance/results/calibration.json)):

| Pattern | Measured value |
|---|---|
| Net flow ≈ pay − rent − other living costs − bills + bonus + noise | pay explains 94% of the regular-month flow |
| December bonus (employed and self-employed) | in 56% of years, 0.23–1.01 × monthly pay |
| One-off bills | 37% of Februaries and Augusts, 4.5% of other months; CHF 450–1,780 |
| Student fees | CHF 1,051 in February, CHF 846 in September |
| Month-to-month noise | grows with pay (CHF 330 to 580 a month from the lowest to the highest pay fifth) |
| Carry-over between months | none: correlation −0.03 between consecutive regular months |
| Bills and pay | unrelated |
| Drift | pay +1.42% a year, costs +1.01% a year |

These patterns become the acceptance tests the learned model has to pass ([07](07-validation.md)).

## What the associations say (and do not say)

[`analysis/state-correlations`](../analysis/state-correlations/results/REPORT.md) ran 181 pairwise
tests with Benjamini–Hochberg correction, one observation per person. Two results shaped the model:

* Pay and mean monthly net flow are almost the same thing (Spearman 0.97). Pay must be in the state
  as a number, not only through the work mode.
* Age and pay look related when everyone is pooled (Spearman −0.26), but inside each work status
  the association disappears (all |ρ| ≤ 0.11, all q > 0.45). The work-mode mix explains the pooled
  pattern. So the model conditions pay on the work mode, not on age.

## What the data cannot tell us

| Needed for a transition function | In the data? |
|---|---|
| What a month of money looks like for a given situation | **Yes**, 28,800 examples |
| What happens to pay when someone changes work mode | No, but other people in the new mode can be compared |
| What happens to rent when someone moves | No, but rents by city and size can be compared |
| The chance of a job offer, a study place or a flat | **No** |
| The chance of losing a job, a promotion, a partner, a child | **No** |
| Anyone's decisions | **No**: not one decision is recorded |
| Gross pay, taxes, expense categories, education, grades | No |

Two more facts matter later:

* Every high-priority goal amount equals max(CHF 10,000, 1.6 × opening cash), for all 1,200
  people. Goals are generated from cash, so they carry no information of their own.
* The 283 loans have no scheduled repayments in the ledger. We assume a 60-month repayment at the
  stated rate.

## What this means for the design

1. The learned part can only learn **what a month looks like in a situation**. An action works by
   moving the person into a situation other people are already in ([03](03-transition-function.md)).
2. Anything about **changing** situation needs another source: a simulated society for markets
   ([04](04-markets.md)) and stated priors for the rest.
3. A simple hand-calibrated simulator of the ledger stays as the **floor**: the learned model must
   beat it ([07](07-validation.md)).
4. Training and testing are **split by person**, never by month, so nobody is in both.
5. Decisions cannot be fitted to data. The policy is a behaviour model built from traits, the
   person's answers and their own plan ([05](05-persona-policy.md), [06](06-planner-agent.md)).

Reproduce: `python analysis/analyze_mdp.py` (seconds), `python analysis/action_importance/action_importance.py` (~2.5 min),
`python analysis/state-correlations/analyze_state_pairs.py`.
