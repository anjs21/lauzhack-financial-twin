# Responsible AI, Financial Modelling, and Data Use

**LauzHack 2026 | Proposed evaluation guidance**

## Core principle

A Financial Twin should support **informed exploration**, not claim to predict a person's financial future or provide guaranteed personalized financial advice.

## Data protection

- Use fictional profiles and synthetic transaction records.
- Use public data only under applicable terms.
- Do not use real confidential or personal financial records.
- Do not commit API keys or credentials.
- Avoid including unnecessary identifiers.

## Transparency

Clearly identify:
- Synthetic starting values.
- Public historical observations.
- User-provided assumptions.
- Model-generated values.
- Conditional simulation outputs.
- Uncertainty and limitations.

## Calculation integrity

- Use auditable financial formulas or calculation tools.
- Verify LLM-generated explanations against model outputs.
- Check units, taxes, time steps, and currency conversions.
- Show when a scenario violates constraints.

## Fairness and accessibility

- Avoid presenting one financial goal as universally correct.
- Consider differing household, education, location, and employment situations.
- Communicate assumptions in accessible language.
- Do not assume public aggregates describe every individual.

## Uncertainty

- Describe what is known and what is assumed.
- Distinguish sensitivity analysis from probability estimates.
- Do not imply Monte Carlo outputs are guaranteed probabilities.
- Explain the limitations of learned models and historical data.

## Games and immersive experiences

- Ensure game mechanics correspond to the financial model.
- Do not reward misleading or financially impossible behaviour.
- Keep simulated trade-offs understandable.
- Make fictional and educational status clear.

## AI agents and reinforcement learning

- Define actions, constraints, and rewards.
- Test for invalid or unsafe actions.
- Explain policy limitations.
- Do not equate learned behaviour with professional financial advice.

## Third-party resources

- Check dataset and software licenses.
- Attribute providers where required.
- Do not assume free access permits redistribution.
- Document source dates and transformation steps.

## Suggested judge questions

1. What data is fictional, and what comes from external sources?
2. What assumptions drive the result?
3. Can the result be reproduced?
4. What happens if the assumptions change?
5. What limitations should a user understand?
6. What prevents the system from inventing unsupported claims?

This is evaluation guidance, not a legal or regulatory determination.
