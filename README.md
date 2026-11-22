# AI Financial Twin — LauzHack 2026

**One choice. Many possible futures.**  
**Don't predict the future. Make it explorable.**

> **Challenge concept and framework:** Stefania Stan  
> **Event:** LauzHack 2026  
> **Status:** Educational hackathon challenge and reference materials

## The challenge

Life decisions can reshape a person's financial circumstances. Should a student pursue a master's degree or begin working? What changes when someone relocates, starts a business, buys a home, takes a career break, or faces an economic shock?

Build an **AI Financial Twin**: an interactive, computational representation of a fictional person's financial state that allows users to explore the *conditional consequences* of decisions, life events, and changing assumptions.

A Financial Twin is **not** a claim to forecast one certain future. It can use simulation, optimization, causal reasoning, probabilistic methods, machine learning, generative AI, game mechanics, or other techniques—but none of those methods alone defines the twin. The essential experience is to represent a financial state, change something meaningful, recompute consequences, compare alternatives, and explain the trade-offs.

## Illustrative starting point: Alexia

Alexia is a **fictional 20-year-old student in Lausanne**, with **CHF 8,000 in initial liquid savings**. She is considering three paths:

1. Start working.
2. Pursue a master's degree.
3. Move abroad.

Teams may use Alexia or create other entirely fictional profiles. No ten-year outcome is implied by the initial savings balance: meaningful comparisons require explicit assumptions about income, living costs, tuition, timing, taxes, and other factors.

## What to build

A compelling prototype should demonstrate:

- A clearly defined **fictional financial state** (such as income, expenses, savings, assets, liabilities, goals, and constraints).
- At least one **meaningful what-if intervention** or life-event decision.
- A **baseline versus alternative** or before/after comparison grounded in calculations.
- **Transparent assumptions**, units, limitations, and an explanation of why results differ.
- An interactive experience that helps a user explore financial possibilities.

Additional directions may include uncertainty and sensitivity analysis, multi-stage decisions, goal optimization, intelligent agents, serious games, immersive environments, or experimental computational approaches. The repository's 120-scenario library is **inspiration**, not a requirement to implement 120 scenarios.

## Freedom to innovate

There is **no prescribed programming language, AI framework, model, interface, or technology stack**. Teams may build a conversational assistant, mathematical simulator, decision game, virtual environment, agent-based ecosystem, causal laboratory, reinforcement-learning system, optimization engine, quantum-classical research prototype, or a combination.

**Monte Carlo simulation is optional**: it can explore uncertainty under assumed distributions, but it is not synonymous with forecasting and is not required for a Financial Twin.

## Repository guide

The [Parallel web app](web/README.md) adds a persona builder, scenario exploration,
and LLM explanations. Its React frontend deploys to Vercel; the FastAPI service and
simulation models stay on your laptop. See the setup guide for local startup,
provider configuration, and the HTTPS tunnel connection.

| Location | Purpose |
|---|---|
| [`pipeline/`](pipeline/) | How our Financial Twin works: data, states and actions, the transition function, the persona, validation, and what we tried |
| [`docs/`](docs/) | Digital-twin foundations, financial mathematics, scenario modelling, and technical approaches |
| [`examples/`](examples/) | Sample questions, interactions, experiences, and output structures |
| [`data/`](data/) | Fictional profiles, synthetic financial records, and scenario inputs |
| [`resources/`](resources/) | Optional downloadable public datasets, free technology tools, and usage guidance |
| [`evaluation/`](evaluation/) | **Proposed** judging rubric and technical-validation references, pending organizer approval |
| [`faq/`](faq/) | Frequently asked questions and troubleshooting |

Some linked documents may be added or revised as the challenge materials are finalized.

## Data and responsible modelling

- Use **synthetic or appropriately licensed public data**. Do not upload real confidential or identifiable personal financial records.
- Treat generated profiles and transaction histories as **fictional**, not representative Swiss population statistics unless separately calibrated and validated.
- Document sources, units, currencies, time periods, modelling assumptions, and data transformations.
- Validate cash-flow identities, assets and liabilities, constraints, and scenario transitions.
- Clearly distinguish **observations, assumptions, calculated outcomes, and uncertainty**.
- Do not present simulated outcomes as guaranteed forecasts or personalized financial advice.
- Review third-party data, model-weight, software, media, and API licenses independently.

## Authorship, attribution, and intellectual-property notice

**Original challenge concept, challenge framing, and authored supporting framework:** **Stefania Stan**.

The AI Financial Twin challenge as presented here—including its educational framing, structured scenario-exploration approach, explanatory documentation, curated example materials, and the original selection and arrangement of repository content—is attributed to Stefania Stan, **to the extent those elements are her original contributions**.

This attribution is intended to identify the originator of the challenge and to preserve clear credit. It does **not** assert exclusive ownership of general ideas, mathematical formulas, publicly known digital-twin concepts, independently developed techniques, or third-party materials. Legal ownership and permitted institutional use may be subject to applicable law, employment or contractual arrangements, and any relevant event agreements.

### Rights and permissions

**No repository-wide open-source or open-data license is granted by this README.** Unless a specific file states otherwise, original authored text and other copyright-protected original repository materials are provided for review and participation in this challenge; **all rights are reserved to the applicable rights holder(s)**.

Do not assume that publicly visible repository content is licensed for copying, redistribution, modification, commercialization, or reuse in other events or projects. Obtain appropriate permission from the relevant rights holder(s) before such reuse. Participants may refer to and use these materials as permitted by the challenge organizers and applicable agreements.

Third-party datasets, research, code, models, and assets retain **their own** licenses and attribution requirements. A link to a third-party resource does not transfer ownership or grant additional permissions.

**Student submissions:** Teams retain whatever rights they have under applicable law and the official event rules. This repository does not claim ownership of student-created code, designs, or submissions merely because they participate. Teams are responsible for rights in materials they incorporate.

For rights or attribution questions, contact the repository maintainer through the repository's available contact channels. Any formal licensing or ownership determination should be reviewed against relevant organizational and event policies before public release.

## Evaluation and official rules

The materials under [`evaluation/`](evaluation/) are **draft guidance** unless explicitly approved and published as official criteria. Submission deadlines, eligibility, team size, awards, and binding competition rules are determined by the event organizers and supersede illustrative guidance in this repository.

## Final principle

**Don't predict the future. Make it explorable.**
