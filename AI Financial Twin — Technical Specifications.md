# AI Financial Twin — Technical Specifications

Nov 22, 2026 · @Anooj Sathyan

## Overview

Parallel is an AI Financial Twin: it simulates thousands of possible lives for one fictional person from 2025 to 2034 and shows how a decision changes them. Instead of one forecast, it compares a baseline with an alternative (a master's, a part-time job, a bigger flat) and explains in plain words why the outcomes differ.

## How it works

Each simulated month follows the same four steps, run for many lives at once:

1. **The person decides.** A persona policy picks work, housing, spending and investing choices from the person's traits (risk tolerance, frugality, career drive and five more) and their own plan.
2. **The world reacts.** Random events happen: job loss or offers, promotions, a partner or child, recessions. Job offers, master's admission and flat searches come from a market model calibrated to Swiss SECO and BFS statistics.
3. **Money moves.** A neural network trained on 28,800 synthetic profile-months draws that month's net cash flow.
4. **The books update exactly.** Cash, investments, loans and interest are recomputed, and every identity is checked.

Baseline and alternative share the same random draws, so any difference comes from the decision, not from luck.

## Architecture and tech stack

&#91;embedded content: system architecture · 4 parts\]

The web app talks over an HTTPS tunnel to a FastAPI server on the team's laptop. That server runs the simulator (Python, NumPy, PyTorch) and calls an LLM through Together AI's OpenAI-compatible API. No model, key or persona data is deployed to the web host.

## Key features

- **Persona builder:** enter age, income, savings, debts, goals and preferences for a fictional person.
- **What-if comparison:** fan charts of net worth, cash and income, plus the chance of reaching the goal, baseline vs. alternative.
- **Plan in your own words:** an LLM planner turns "a master's next year, then a bigger flat" into yearly steps with fallbacks, tests them in 300 simulated lives and revises them when you say what changed.
- **Plain-language explanations:** the LLM explains only the numbers the simulator produced, and you can ask follow-up questions.
- **Uncertainty shown, not hidden:** the spread is split into traits, choices, luck and model uncertainty.

## Validation, privacy and limits

- **Tested:** 92 Python tests (cash identities, loans never negative, same seed gives same lives, plans fire as stated) and Playwright browser tests on desktop and mobile. The cash-flow network beats a hand-tuned baseline on held-out people (CRPS 409 vs. 421).
- **Private by design:** synthetic data only. The API needs an access key, sessions live in memory for one hour, and no chat is written to disk. The person's name is never sent to the LLM.
- **Limits:** results are possibilities under stated assumptions, not forecasts or financial advice. Assumptions the data cannot supply (event rates, returns, degree premium) are listed openly in one file.
