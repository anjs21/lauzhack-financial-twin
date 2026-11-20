

# Financial Twin — Example Outputs and Visualizations

## Purpose

This document illustrates possible outputs from a Financial Twin.

All values are fictional and illustrative.

The examples are not forecasts or financial recommendations.

## 1. Scenario Comparison Table

A Financial Twin may compare alternatives using a table.

| Measure | Start Working | Master's Degree | Move Abroad |
|---|---|---|---|
| Initial savings | CHF 8,000 | CHF 8,000 | CHF 8,000 |
| Education costs | Scenario-dependent | Scenario-dependent | Scenario-dependent |
| Employment timing | Immediate | Delayed | Scenario-dependent |
| Relocation costs | None assumed | None assumed | Scenario-dependent |
| Ten-year savings | Calculated | Calculated | Calculated |
| Minimum liquidity | Calculated | Calculated | Calculated |
| Goal progress | Calculated | Calculated | Calculated |

The system should calculate outcomes from consistent assumptions.

## 2. Financial Timeline

Possible visualization:

- Horizontal axis: Time
- Vertical axis: Savings, net worth, or liquidity
- Lines: Alternative scenarios
- Markers: Life events
- Shaded areas: Conditional uncertainty ranges

Charts should distinguish observed values from simulated values.

## 3. Scenario Tree

A scenario tree can represent branching decisions.

Example:

Starting Point: Alexia, age 20

- Start working
  - Remain in Switzerland
  - Move abroad
- Pursue a master's degree
  - Work part-time
  - Study full-time
- Start a business
  - Self-finance
  - Seek external funding

Each branch may lead to different financial states.

## 4. Constraint Dashboard

Possible indicators:

- Minimum liquidity
- Debt-service ratio
- Monthly cash surplus
- Goal feasibility
- Funding shortfall
- Sensitivity to assumptions

Constraints should be clearly identified as user-defined or scenario-defined.

## 5. Sensitivity Visualization

A sensitivity chart may show how changing one assumption affects an outcome.

For example:

- Tuition increases
- Salary decreases
- Living costs rise
- Interest rates change
- Investment returns vary

The chart should identify the assumptions being changed.

## 6. Example JSON Output

The following is an illustrative output structure.

It is not a mandatory API schema.

```json
{
  "scenario_id": "S001",
  "user_id": "synthetic_alexia",
  "question": "What if I pursue a master's degree?",
  "currency": "CHF",
  "time_horizon_years": 10,
  "initial_state": {
    "age": 20,
    "liquid_savings": 8000
  },
  "baseline": {
    "decision": "start_working"
  },
  "alternative": {
    "decision": "pursue_masters",
    "study_duration_years": 2
  },
  "assumptions": {
    "tuition": "user_or_synthetic_input",
    "living_expenses": "user_or_synthetic_input",
    "employment_income": "user_or_synthetic_input",
    "salary_growth": "user_or_synthetic_input"
  },
  "outputs": {
    "baseline_savings": null,
    "alternative_savings": null,
    "minimum_liquidity": null,
    "scenario_difference": null
  },
  "validation": {
    "calculations_completed": false,
    "assumptions_documented": true
  },
  "limitations": [
    "Illustrative synthetic scenario",
    "Future employment outcomes are uncertain",
    "Not a financial forecast"
  ]
}
```

The null values are deliberate placeholders.

They should be populated only after the model has calculated valid results.

## 7. Explanation Output

A Financial Twin may generate an explanation containing:

1. The decision being explored
2. The baseline
3. The alternative
4. Key assumptions
5. Calculated differences
6. Main financial trade-offs
7. Relevant uncertainty
8. Model limitations

Explanations should be derived from model outputs.

## 8. Game-Based Output

A game-based Financial Twin may show:

- Current financial resources
- Available actions
- Life-event consequences
- Progress toward goals
- Financial constraints
- Alternative paths

The underlying accounting model should remain consistent.

## 9. Agent-Based Output

An agent-based model may produce:

- Agent states
- Interaction histories
- Financial transfers
- Scenario outcomes
- Emergent patterns
- Constraint violations

Results depend on the simulated environment and behavioural assumptions.

## 10. Optimization Output

An optimization-based Financial Twin may produce:

- Feasible decisions
- Constraint violations
- Objective values
- Pareto frontiers
- Sensitivity analysis
- Alternative strategies

Optimization outputs should not automatically be presented as personalized financial advice.

## 11. Reinforcement-Learning Output

Possible outputs include:

- Learned policies
- Cumulative simulated rewards
- State-action histories
- Constraint violations
- Comparison with baseline strategies
- Robustness tests

The reward function and environment assumptions should be disclosed.

## 12. Quantum-Classical Output

Possible research outputs include:

- Problem formulation
- Quantum circuit description
- Classical baseline
- Optimization results
- Runtime or resource comparison
- Experimental limitations

Quantum advantage should not be claimed without appropriate evidence.

## 13. Evaluation of Outputs

Students should consider:

- Are calculations reproducible?
- Are assumptions documented?
- Are units consistent?
- Are scenarios comparable?
- Are constraints respected?
- Are uncertainty and limitations visible?
- Can users understand why outcomes differ?

## Final Principle

A compelling Financial Twin does not simply display numbers.

It helps users understand the relationship between decisions, assumptions, constraints, and possible consequences.
