# Financial Twin: model assumptions

Every assumption the simulator, the persona, the what-if branches and the assistant rely on, with its
value and where it comes from. Numbers are taken from the code and configuration as of this document;
most live in [`twin/world/priors.yaml`](../twin/world/priors.yaml), where they can be changed.

**Source tags**

| Tag | Meaning |
|---|---|
| **Data** | Measured on the challenge's synthetic data (profiles and monthly ledger) |
| **Swiss** | Calibrated to an official Swiss statistic (SECO, BFS); see [`swiss_targets.json`](../twin_game_theory/world/swiss_targets.json) |
| **Learned** | Fitted by a model (on the ledger, or on simulated societies) |
| **Spec** | Default from the challenge's MDP specification |
| **Assumed** | Our own choice, to be tuned or replaced with better evidence |

This is a research prototype on synthetic data. It shows possibilities under these assumptions; it is
not a forecast or financial advice.

---

## 1. Time, scope and runs

| Assumption | Value | Source |
|---|---|---|
| Simulated period | January 2025 to December 2034 (120 months), whatever today's date is | Spec |
| Balances, age and income are given as of | January 2025 | Spec |
| Voluntary decisions (work, home, spending, investing) | Only at the **January review** each year; events and cash flows evolve monthly | Spec |
| Main ("high") goal deadline | December 2029 | Spec |
| Medium goal deadline (data profiles only) | December 2032 | Spec |
| Lives simulated per run in the web app | 500 | Assumed |
| Random seed | Fixed (20261121), so the same inputs give the same results | Assumed |
| Paired comparisons | Every path and branch uses the same random draws, so differences come from the choices, not from luck | Assumed |

## 2. The data the model learns from

| Assumption | Value | Source |
|---|---|---|
| Profiles | 1,200 synthetic people: 682 employed, 216 retired, 120 part-time, 106 self-employed, 39 students, 37 unemployed | Data |
| Ledger | Monthly cash balances and flows, January 2025 to December 2026 (28,800 person-months) | Data |
| No transitions are observed | Nobody in the data changes job, studies or flat, so those chances come from priors or the learned markets (section 10) | Data |
| Self-employed people | Treated as full-time workers (with their own cash-flow pattern) | Assumed |
| Household types | `shared_housing` = single in a shared flat; `single` = single in a small flat; `couple` and `family` = family-size flat | Assumed |
| Cities | 12 supported: Basel, Bern, Biel, Fribourg, Geneva, Lausanne, Lucerne, Lugano, Neuchâtel, St Gallen, Winterthur, Zurich | Data |

## 3. Income and pay

| Assumption | Value | Source |
|---|---|---|
| Pay depends on | Work mode only (student, part-time, full-time, benefit, pension), not on age, city or rent | Data |
| Earning power | Each person keeps a pay **rank** within the pay distribution; changing work mode reads the new pay at the same rank, loosened by noise (0.30) | Learned, assumed noise |
| Pay growth | 1.42% a year | Data |
| Promotions | 0.7% chance a month (0.3% in a recession), each a step of 0.25 standard deviations in pay rank | Spec |
| A quoted job offer | Pays exactly the quoted salary, which then grows with pay growth and promotions | Assumed |
| Changing hours in the same job (part-time ↔ full-time) | Keeps the pay rank | Assumed |
| Not working under 65 | Benefit income, from the data's benefit distribution | Data |
| Not working at 65 or older | Pension, from the data's pension distribution | Data |
| December bonus, February/August bills, student fees | Learned from the ledger's monthly pattern (part of the cash-flow model below) | Learned |

## 4. Living costs, rent and cash flow

| Assumption | Value | Source |
|---|---|---|
| Cash-flow model | A neural network (mixture of 5 Gaussians) predicts each month's net flow from work mode, self-employment, pension, income, rent, city and calendar month; never the calendar year | Learned |
| Model uncertainty | 5 independently trained networks; each simulated life uses one of them | Learned |
| Personal spending habits | A per-person offset (shrunk average residual) carries what the inputs cannot see | Learned |
| Non-rent living costs (standard lifestyle) | City median plus 1.6% of income; e.g. Zurich CHF 1,647, Geneva 1,610, Lausanne 1,542, Fribourg 1,338 a month | Data |
| Lifestyle levels | Lean ×0.9, standard ×1.0, comfortable ×1.1 of living costs | Spec |
| Cost growth (prices, rent) | 1.01% a year (3% a year during inflation) | Data / Spec |
| Rent | log rent = level + flat size + city + personal residual (R² 0.66); a person keeps their residual when moving | Learned |
| Child costs | CHF 1,800 a month for a child born during the simulation (data families already have them in their flows) | Spec |
| Loans | Repaid in equal instalments over 60 months (the data gives only principal and rate) | Assumed |
| Working student (part-time job alongside studies) | Costed like a part-timer on the same pay and rent, plus full tuition; the data has no working students (student pay tops out near CHF 2,000) | Assumed |

## 5. Economy and investments

| Assumption | Value | Source |
|---|---|---|
| Starting economy | Normal | Spec |
| Switching (per month) | Normal → inflation 0.5%, normal → recession 0.5%; inflation → normal 4%; recession → normal 6% | Spec |
| Investment returns (mean, standard deviation, per year) | Normal 6% ± 10%; inflation 1% ± 12%; recession −15% ± 25% (about 4% a year in the long run) | Spec |
| Returns | Drawn every month, independently for each simulated life | Spec |

## 6. Work and jobs

| Assumption | Value | Source |
|---|---|---|
| Job loss | 0.3% a month (1% in a recession), for people working | Spec |
| After losing a job | The person searches for the same kind of work (part- or full-time) | Assumed |
| Job offers while searching | From the learned job market (section 10) | Learned |
| Job offers without the learned market | 20% a month (10% in a recession); only used if markets are switched off | Assumed |
| Applications while searching | 2 a month | Assumed |
| Grades | Average (0) unless given | Assumed |
| Pension age | 65 | Spec |
| Forced retirement | Work stops at 70 | Spec |
| Leaving work at 65 or later | Counts as retirement (pension) | Spec |

## 7. Studies and education

| Assumption | Value | Source |
|---|---|---|
| Starting students | The current course ends at the first review (January 2025) unless *months until you finish* is given | Assumed |
| After graduating | Not working, searching for a full-time job | Assumed |
| A master's | 24 months | Data (S001) |
| Master's delays | 10% take 6 months longer | Assumed |
| Master's pay effect | +20 percentiles of pay rank on graduating | Assumed |
| Tuition | CHF 1,200 / 2,500 / 4,800 / 8,000 a year (drawn at random if unknown); charged beyond the typical student fees already in the flows (CHF 1,897 a year) | Data |
| Master's admission | From the learned admission model, about 98% | Learned |
| Who can start a master's | Up to age 45, once per life, not in the middle of another course | Spec / Assumed |
| Leaving a course early (for a job) | No master's is credited, so one can be started later | Assumed |
| Side job alongside studies | Pays the offer until the course ends, then ends; the pay rank (and graduate pay) is unchanged | Assumed |
| Highest degree when unknown | Drawn per life: share with a degree rises with pay quintile (25%, 40%, 55%, 70%, 80%); of those, 45% bachelor's, 45% master's, 10% PhD | Assumed |
| Students' degree | None yet (a master's student holds a bachelor's) | Assumed |

## 8. Housing

| Assumption | Value | Source |
|---|---|---|
| Flat sizes | Shared flat, small flat, family flat | Spec |
| Moving cost | CHF 2,000 | Spec |
| Finding a flat | From the learned flat market (section 10); a search stays open until a flat is found or the next review | Learned |
| Landlord's rule | New rent ≤ income / 3, or 12 months' rent in cash, or a student | Assumed (twin_game_theory) |
| Families | Cannot live in a shared flat; a child born there forces a move | Spec |

## 9. Household and family

| Assumption | Value | Source |
|---|---|---|
| Partner moves in | 0.4% a month, singles under 45 | Spec |
| Child | 0.7% a month, couples under 42 | Spec |
| Separation | 0.2% a month, couples | Spec |
| Parental leave | 4 months at 80% pay | Spec |
| Windfall (gift, inheritance) | 0.05% a month, CHF 5,000 to 50,000 | Spec |

## 10. Learned markets (from twin_game_theory)

The ledger never shows anyone getting a job, a study place or a flat. These three chances come from a
simulated Swiss society ([`twin_game_theory/world`](../twin_game_theory/world/README.md)): societies of
200, 1,000 and 2,000 people (90,000 in all) compete for limited openings, places and flats, and small
statistical models are fitted to the result.

| Assumption | Value | Source |
|---|---|---|
| Job offer chance depends on | Degree, grades, months out of work, pay level, applications, economy, being a student | Learned |
| Admission chance depends on | Programme, grades | Learned |
| Flat chance depends on | Target size, downsizing, income relative to the new rent | Learned |
| Market tightness | The person cannot see it, so each chance is averaged over the tightness the society produced in that economy | Assumed |
| Model uncertainty | 5 bootstrap refits; each life uses the refit matching its cash-flow network | Learned |
| Calibration targets | Unemployment 2.8% (SECO); bachelor's → master's within 2 years 87% (BFS); master's graduates unemployed after a year 6.4% (BFS); moving rate 9.3% a year | Swiss |
| Master's → PhD within 2 years | 15% | Assumed (no verified BFS figure) |
| Persona mix in the simulated society | Four archetypes (driven, anxious, relaxed, balanced), each varied ±20% | Assumed |
| **Known misses** | Median time out of work 3.5 months vs 6.5 (BFS); bachelor's-only graduates unemployed after a year 26% vs 4.9% (BFS, applied-university bachelor's) | Swiss (checked) |

## 11. Credit, cash and investing

| Assumption | Value | Source |
|---|---|---|
| Credit limit | 3 months of income, at least CHF 5,000 | Spec |
| Credit-limit breach | Cash below minus the limit | Spec |
| Interest on negative cash | 8% a year | Spec |
| Automatic top-up | If cash goes below zero, investments are sold to bring cash back to one month of outgoings (any month: a standing order, not a decision) | Assumed |
| Cash buffer the persona aims for | 2 to 6 months of outgoings: 6 for the most cautious, 2 for the most risk-tolerant | Assumed |
| Investing | Cash above the buffer can be invested; selling refills the buffer | Spec |

## 12. Goals and success

| Assumption | Value | Source |
|---|---|---|
| Main goal met | At December 2029, the target is reached **and** the credit limit was never breached before | Spec |
| Emergency-savings goal | Measured in cash | Spec |
| Other goals (housing deposit, education, career transition, retirement reserve) | Measured in net worth (cash + investments − debt) | Spec |
| Medium goal met (data profiles) | At December 2032, net worth ≥ main + medium targets, no breach before | Spec |

## 13. The persona (who decides)

| Assumption | Value | Source |
|---|---|---|
| Decision maker | A rule-based persona: at each January review it picks each choice with probabilities from its traits and situation (softmax) | Assumed |
| Traits (0 to 1) | Risk tolerance, frugality, career drive, free time, housing aspiration, inertia, impatience, goal focus | Spec |
| Default traits | Risk 0.5, frugality 0.5, career 0.5, free time 0.3, housing 0.4, inertia 0.7, impatience 0.4, goal focus 0.5; adjusted by the profile (investment share, age, students, part-timers, household, goals) | Assumed |
| Trait uncertainty | Each life draws traits around the persona's (0.4 on the logit scale), because we never know a person exactly | Assumed |
| Randomness of choices | Temperature 1.0 | Assumed |
| What pushes decisions | Cash pressure (under 2 months of cash), goal pressure (stronger within 2 years of the deadline), comfort (over 6 months of savings), a recent job loss or move | Assumed |
| Habit | People rarely change anything at a review; inertia weighs strongly toward "continue" | Assumed |
| Stated plans | A plan ("at January 2028, work full-time") is carried out with the stated probability where it is possible | Assumed |

## 14. "Decide deliberately"

| Assumption | Value | Source |
|---|---|---|
| Method | The persona's own probabilities, tilted toward options worth more over the next five years (anchored foresight, strength 0.5); at strength 0 it is exactly the persona | Assumed |
| What "worth more" means (CHF-equivalent a month) | Lifestyle −300 / 0 / +200; free time: part-time +1,500, not working +2,000; cash below zero −2,000; any change −300; home value by household and flat size; goals counted at their value when met on time; plus net worth at the end | Spec |
| Look-ahead | 60 months, 6 simulated futures per option, learned from 2,000 decision points | Assumed |
| Limitation | The foresight network was trained before the learned markets replaced the fixed job and flat rates | Known |

## 15. Scenarios, what-if branches and imposed events

| Assumption | Value | Source |
|---|---|---|
| Part-time offer | Taken at the January 2025 review at the quoted take-home pay; later decisions can change it | Assumed |
| A student taking the offer | Leaves the course, unless *Keep studying* is on (then it is a side job until the course ends) | Assumed |
| Spend less / cheaper home | A lifestyle cut / a one-step cheaper flat at the January 2025 review (the flat still has to be found) | Assumed |
| A branch | Its parent path with one more decision or event at a January review; identical to the parent before it | Assumed |
| A forced decision | Happens with certainty in every life where it is possible that January (the share is shown) | Assumed |
| A life event (lose my job, recession, child, partner, separation) | Imposed that month in every life where it can happen; everything that normally follows it (search, leave pay, forced move, recovery) is unchanged | Assumed |
| Timing | Events from January 2025 ("now"), decisions from January 2026; decisions come before events in the same month | Assumed |
| Depth | Up to 4 stacked decisions or events per branch | Assumed |
| Key moments | Decisions and events that happen in at least 10% of lives; the year shown is the median year it first happens | Assumed |

## 16. The assistant (LLM)

| Assumption | Value | Source |
|---|---|---|
| Provider and model | Together AI, `deepseek-ai/DeepSeek-V4.1-Flash` (set in `.env`) | Assumed |
| What it receives | Persona details without the name, result summaries, the list of paths, the conversation; never the monthly arrays | Assumed |
| What it may change | Only through its tools: persona facts and traits, scenario, decision style, branches, what the chart shows | Assumed |
| Translating words to settings | Amounts are monthly take-home CHF ("7k" = 7,000); rates are fractions; "now" means January 2025 | Assumed |
| Grounding | Instructed to use only the simulation's numbers; this is an instruction to the model, not a guarantee | Assumed |

## 17. Statistics and uncertainty

| Assumption | Value | Source |
|---|---|---|
| Fan chart | Percentiles of the simulated lives (5–95% and 25–75%), not confidence intervals | — |
| Sampling interval on differences | Covers Monte Carlo noise only, not model error | — |
| Covered by the spread | Trait uncertainty, cash-flow model uncertainty (5 networks), market model uncertainty (5 refits), random life events | — |
| Not covered | Whether the priors, the persona rules or the synthetic data describe real people | — |

## 18. Known limitations

- **Synthetic data.** Everything is learned from synthetic profiles and a two-year synthetic ledger; no real bank data.
- **No observed transitions.** Job changes, studies and moves rely on priors and on a simulated society, not on observed behaviour.
- **Job market misses.** Spells out of work are too short (3.5 vs 6.5 months), and bachelor's-only graduates are unemployed too often.
- **Assumed education.** The data does not say who holds a degree; it is drawn from an assumed prior.
- **One work mode at a time.** Studying and working only combine through the side-job option, which ends with the course.
- **Fixed start.** The timeline starts in January 2025 whatever today's date is.
- **Foresight network predates the learned markets.** Retrain with `python -m twin foresight` to update it.
