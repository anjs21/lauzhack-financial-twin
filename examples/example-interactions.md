
# Example Financial Twin Interactions

## Purpose

These conversations illustrate possible user experiences.

They are not mandatory interface designs.

All users and financial values are fictional.

## Interaction 1 — Education Decision

### User

"I'm 20, have CHF 8,000 in savings, and I'm unsure whether to start working or pursue a master's degree."

### Financial Twin

"I can help you explore both options.

To compare them, I would need assumptions about:

- Tuition and education costs
- Living expenses
- Potential employment income
- Study duration
- Part-time work
- Your preferred time horizon

Would you like to provide these values or explore an illustrative synthetic scenario?"

### User

"Use an illustrative scenario over ten years."

### Financial Twin

"I'll create two scenarios:

A. Start working immediately.

B. Pursue a two-year master's degree.

The model will compare cash flow, liquidity, savings, and financial goal progress.

All numerical outcomes will depend on the stated assumptions."

### Expected System Behaviour

The Financial Twin should:

1. Construct the scenarios.
2. Apply explicit financial assumptions.
3. Run the calculation engine.
4. Validate the outputs.
5. Display the differences.
6. Explain the assumptions and limitations.

---

## Interaction 2 — Scenario Modification

### User

"What if tuition increases by 20%?"

### Financial Twin

"I'll update the tuition assumption while keeping the other inputs unchanged.

This will allow us to isolate the effect of higher tuition on the master's scenario."

### Expected Behaviour

The system should recalculate the scenario rather than invent a new numerical result.

---

## Interaction 3 — Relocation

### User

"What happens if I move to London?"

### Financial Twin

"To explore that decision, we should consider salary, taxes, rent, transportation, relocation costs, currency, and savings goals.

Would you like to compare London with your current location?"

### Expected Behaviour

The system should identify relevant differences between jurisdictions and avoid assuming that salary alone determines financial attractiveness.

---

## Interaction 4 — Financial Constraint

### User

"I want to buy a home, but I don't want my emergency savings to fall below CHF 20,000."

### Financial Twin

"I'll treat CHF 20,000 as a minimum liquidity constraint.

We can explore purchase scenarios that respect that constraint and identify those that do not."

### Expected Behaviour

The model should distinguish feasible and infeasible scenarios.

---

## Interaction 5 — Uncertainty

### User

"Which option is guaranteed to make me wealthier?"

### Financial Twin

"No option can be guaranteed to produce the highest future wealth.

I can compare conditional outcomes under explicit assumptions and show which variables most affect the differences."

### Expected Behaviour

The system should avoid unsupported guarantees.

---

## Interaction 6 — Explainability

### User

"Why does the master's scenario show lower savings initially?"

### Financial Twin

"The model includes tuition expenses and a delayed start to full-time employment.

Those assumptions reduce available cash during the study period.

We can change tuition, part-time income, or employment timing to see how the outcome responds."

### Expected Behaviour

Explanations should correspond to actual model inputs and calculations.

---

## Interaction 7 — Game-Based Financial Twin

### Player

"I choose to move abroad."

### Game

"Your simulated location changes.

The financial model now applies the selected relocation costs, income assumptions, currency, and living expenses."

### Expected Behaviour

Game mechanics should remain linked to financial calculations.

---

## Interaction 8 — Reinforcement Learning

### Researcher

"Can an agent learn strategies for preserving liquidity while pursuing education?"

### Financial Twin Environment

"The environment defines:

- Financial states
- Available actions
- Transition rules
- Liquidity constraints
- Reward functions
- Scenario assumptions

The learning agent can then explore strategies through repeated simulated interaction."

### Expected Behaviour

The system should distinguish learned policies from verified financial recommendations.

---

## Final Principle

A Financial Twin should not merely produce convincing language.

Its explanations should be grounded in an explicit, reproducible computational model.

