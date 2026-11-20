

# AI and Technology Landscape — Financial Twin Challenge

**UBS × LauzHack 2026 | Advanced Technical Reference**

## 1. Purpose and Technology Freedom

The AI Financial Twin Challenge invites participants to explore how computational technologies can help individuals understand, simulate, and interact with the financial consequences of life decisions.

There is no prescribed architecture, programming language, algorithm, AI model, interface, or technology stack.

Participants may develop:

- AI-powered conversational systems
- Mathematical financial simulators
- Computer games and serious games
- Interactive decision laboratories
- Virtual worlds and 3D experiences
- Digital avatars and virtual humans
- Agent-based financial ecosystems
- Reinforcement-learning environments
- Multi-agent AI systems
- Optimization and planning engines
- Causal and counterfactual models
- Knowledge graphs and reasoning systems
- Hybrid quantum-classical prototypes
- Immersive AR/VR applications
- Novel human-computer interfaces
- Combinations of these approaches

The objective is not technological complexity for its own sake.

The objective is to create a credible, interactive, explainable environment for exploring financial decisions.

---

## 2. Financial Twin: A Capability-Based Definition

A Financial Twin is not defined by a specific technology.

It is defined by the capabilities it provides.

A Financial Twin may represent a financial situation, simulate changes, explore alternative decisions, and communicate their consequences.

### Core Capability Sets

| ID | Capability | Description |
|---|---|---|
| C01 | Financial state representation | Represent income, expenses, assets, liabilities, and goals |
| C02 | Personalization | Adapt the model to individual circumstances |
| C03 | State updating | Incorporate new information over time |
| C04 | Scenario creation | Define alternative decisions and conditions |
| C05 | Counterfactual exploration | Compare what-if decisions |
| C06 | Financial simulation | Calculate conditional financial outcomes |
| C07 | Life-event modelling | Represent education, employment, relocation, and other events |
| C08 | Constraint management | Apply budget, liquidity, and other constraints |
| C09 | Decision comparison | Compare financial and non-financial trade-offs |
| C10 | Uncertainty exploration | Examine sensitivity and alternative conditions |
| C11 | Goal tracking | Evaluate progress toward defined objectives |
| C12 | Explainability | Explain model inputs, assumptions, and outcomes |
| C13 | Interactivity | Allow users to modify decisions and assumptions |
| C14 | Adaptation | Update model behaviour when relevant information changes |
| C15 | Validation | Test financial consistency and model reliability |
| C16 | Transparency | Distinguish facts, assumptions, calculations, and uncertainty |
| C17 | Data provenance | Track the origin and transformation of information |
| C18 | Accessibility | Make complex financial concepts understandable |
| C19 | Reproducibility | Reproduce results under equivalent conditions |
| C20 | Responsible decision support | Support informed choices without implying certainty |

These capabilities may be implemented through different technological approaches.

No single technology is required to deliver all capabilities.

---

## 3. Capability Maturity Levels

Teams may demonstrate different levels of functionality.

### Level 1 — Representation

The system represents a fictional financial profile.

### Level 2 — Interaction

Users can inspect, update, and explore financial information.

### Level 3 — Simulation

Users can test decisions and calculate their conditional consequences.

### Level 4 — Comparison

Users can compare alternative scenarios, assumptions, and constraints.

### Level 5 — Adaptive Decision Support

The system can respond to changing information and support more sophisticated decision exploration.

### Level 6 — Intelligent and Immersive Exploration

The system may use AI agents, optimization, learning systems, game mechanics, or immersive environments to explore complex decisions.

These levels are illustrative capability categories, not official UBS judging levels.

A system's sophistication must be evaluated alongside reliability, explainability, and user value.

---

# PART I — COMPUTATIONAL AND MATHEMATICAL TECHNOLOGIES

## 4. Classical Computing

Classical computing provides the foundation for most financial simulation systems.

Possible technologies include:

- Python
- JavaScript and TypeScript
- Java
- C++
- C#
- Rust
- Julia
- R
- MATLAB
- SQL
- Spreadsheet engines
- WebAssembly

Possible applications include:

- Financial state calculations
- Cash-flow simulation
- Investment accounting
- Debt and mortgage calculations
- Scenario processing
- Data transformation
- Numerical optimization
- Visualization

### Useful Libraries

- NumPy: https://numpy.org/
- SciPy: https://scipy.org/
- Pandas: https://pandas.pydata.org/
- Polars: https://pola.rs/
- Julia: https://julialang.org/

---

## 5. Deterministic Simulation

Deterministic simulation calculates outcomes under fixed assumptions.

A general transition model is:

$$
x_{t+1}=F(x_t,a_t,w_t;\theta)
$$

Where:

- $x_t$ represents the financial state.
- $a_t$ represents decisions.
- $w_t$ represents external conditions.
- $\theta$ represents model assumptions.

Identical inputs should produce identical outputs.

Possible applications include:

- Education-cost comparisons
- Savings accumulation
- Loan repayment
- Housing affordability
- Career decisions
- Retirement cash-flow analysis

---

## 6. Discrete-Event Simulation

Discrete-event simulation represents systems that change when events occur.

Possible events include:

- Graduation
- Employment
- Promotion
- Unemployment
- Relocation
- Property purchase
- Marriage
- Parenthood
- Retirement

The model updates the financial state when an event occurs.

Useful technologies:

- SimPy
- AnyLogic
- Custom event engines

Reference:

https://simpy.readthedocs.io/

---

## 7. System Dynamics

System dynamics models feedback relationships and accumulation over time.

Potential financial feedback loops include:

- Income, savings, and investment accumulation
- Debt, interest, and repayment
- Education, income opportunities, and expenses
- Housing costs and financial flexibility
- Inflation and purchasing power

Possible tools:

- Vensim
- Stella
- AnyLogic
- Python-based system-dynamics libraries

System dynamics is particularly useful when interactions between variables matter.

---

## 8. Stochastic Simulation

Stochastic simulation introduces explicitly modelled uncertainty.

Examples include:

- Variable investment returns
- Uncertain employment timing
- Salary variability
- Unexpected expenses
- Interest-rate scenarios
- Exchange-rate changes

A stochastic state transition can be represented as:

$$
x_{t+1}=F(x_t,a_t,w_t;\theta)
$$

Where $w_t$ is drawn from a specified probability model.

The distribution and its assumptions must be documented.

---

## 9. Monte Carlo Simulation

Monte Carlo is a computational technique for repeated sampling.

It can support uncertainty analysis inside a Financial Twin.

It is not the definition of a Financial Twin.

It is also not inherently equivalent to forecasting.

Possible applications include:

- Conditional outcome distributions
- Liquidity shortfall analysis
- Sensitivity to investment returns
- Stress testing
- Financial resilience analysis
- Tail-risk exploration

Students should disclose sampling assumptions, dependencies, and limitations.

---

## 10. Optimization Technologies

Optimization can search for decisions that satisfy objectives and constraints.

Possible methods include:

- Linear programming
- Mixed-integer programming
- Nonlinear programming
- Dynamic programming
- Constraint programming
- Robust optimization
- Stochastic optimization
- Multi-objective optimization
- Evolutionary optimization
- Metaheuristics

Potential financial applications include:

- Budget allocation
- Savings strategies
- Debt repayment
- Goal prioritization
- Housing affordability
- Portfolio allocation
- Education financing
- Retirement planning

Useful technologies:

- Google OR-Tools
- SciPy Optimize
- CVXPY
- Pyomo
- JuMP

References:

https://developers.google.com/optimization

https://www.cvxpy.org/

https://pyomo.readthedocs.io/

https://jump.dev/

---

# PART II — ARTIFICIAL INTELLIGENCE

## 11. Generative AI and Large Language Models

Large language models may provide a natural-language interface.

Possible capabilities include:

- Understanding financial questions
- Extracting relevant parameters
- Asking clarifying questions
- Explaining scenario outcomes
- Summarizing trade-offs
- Generating user-friendly narratives
- Translating technical results into accessible language

Possible technologies include:

- OpenAI models
- Anthropic Claude
- Google Gemini
- Open-weight language models
- Local language models

Language models should not be treated as authoritative financial calculation engines.

Where calculations are required, they should be performed or verified using appropriate computational methods.

---

## 12. Tool-Augmented AI

An AI system may invoke external tools to perform specific tasks.

Examples include:

- Financial calculators
- Scenario engines
- Optimization solvers
- Data retrieval
- Currency conversion
- Visualization tools
- Validation services

A typical architecture may be:

User question

→ Language understanding

→ Structured parameters

→ Financial model

→ Scenario output

→ Explanation

Tool outputs should be validated before being presented as financial results.

---

## 13. Retrieval-Augmented Generation

Retrieval-augmented generation, or RAG, combines information retrieval with generative AI.

Potential uses include:

- Retrieving modelling assumptions
- Accessing public financial documentation
- Explaining financial terminology
- Accessing scenario definitions
- Providing references for economic indicators

Possible technologies include:

- Vector databases
- Semantic search
- Hybrid search
- Embedding models
- Document retrieval systems

RAG provides contextual information but does not automatically guarantee factual accuracy.

---

## 14. Machine Learning

Machine learning may support:

- Spending categorization
- Pattern recognition
- Income classification
- Expense estimation
- Financial behaviour analysis
- Anomaly detection
- Scenario classification
- Parameter estimation

Possible approaches include:

- Supervised learning
- Unsupervised learning
- Self-supervised learning
- Semi-supervised learning
- Transfer learning
- Online learning

Useful technologies:

- Scikit-learn
- PyTorch
- TensorFlow
- XGBoost
- LightGBM

Machine-learning performance should be evaluated using suitable validation methods.

---

## 15. Deep Learning

Deep learning may support complex representations and interactions.

Possible architectures include:

- Multilayer perceptrons
- Recurrent neural networks
- Transformers
- Graph neural networks
- Autoencoders
- Variational autoencoders
- Diffusion models
- Neural ordinary differential equations
- Neural operators

Potential applications include:

- Behaviour representation
- Multimodal interfaces
- Complex state modelling
- Representation learning
- Generative scenario environments

Deep learning is not automatically preferable to simpler financial models.

---

## 16. Reinforcement Learning

Reinforcement learning investigates sequential decision-making.

An agent interacts with an environment by choosing actions and receiving rewards.

A financial environment may contain:

- Income
- Expenses
- Savings
- Debt
- Financial goals
- Constraints
- Economic conditions

The agent may explore strategies such as:

- Saving
- Investing
- Repaying debt
- Changing spending
- Delaying a purchase
- Adjusting contributions

A Markov Decision Process may be represented as:

$$
MDP=(S,A,P,R,\gamma)
$$

Where:

- $S$ is the state space.
- $A$ is the action space.
- $P$ is the transition model.
- $R$ is the reward function.
- $\gamma$ is the discount factor.

Possible algorithms include:

- Q-learning
- Deep Q-Networks
- PPO
- SAC
- Actor-Critic
- Model-based reinforcement learning
- Offline reinforcement learning
- Multi-agent reinforcement learning

Useful technologies:

https://gymnasium.farama.org/

https://stable-baselines3.readthedocs.io/

https://docs.ray.io/en/latest/rllib/

Reward design and environmental assumptions are critical.

An RL agent can learn to exploit weaknesses in an unrealistic simulation.

---

## 17. Multi-Agent Systems

Multi-agent systems model multiple interacting decision-makers.

Possible agents include:

- Individuals
- Households
- Employers
- Banks
- Investors
- Landlords
- Governments
- Market participants

Possible applications include:

- Household financial interactions
- Employment-market simulations
- Housing-market dynamics
- Economic policy experiments
- Negotiation and resource allocation

Multi-agent systems may use rule-based behaviour, optimization, reinforcement learning, or language models.

Useful technologies:

- Mesa
- PettingZoo
- Ray RLlib
- Custom multi-agent environments

References:

https://mesa.readthedocs.io/

https://pettingzoo.farama.org/

---

## 18. Agentic AI

Agentic AI systems coordinate tools, memory, planning, and task execution.

Potential specialized agents include:

- Profile interpretation agent
- Scenario generation agent
- Financial calculation agent
- Constraint-checking agent
- Explanation agent
- Validation agent
- Visualization agent

Possible frameworks include:

- LangGraph
- Semantic Kernel
- AutoGen
- Custom orchestration systems

Agentic architectures should include explicit validation, security boundaries, and error handling.

---

## 19. Causal AI and Counterfactual Reasoning

Causal methods investigate the consequences of interventions.

They distinguish observing a relationship from changing a variable.

A central distinction is:

$$
P(Y\mid X)\neq P(Y\mid do(X))
$$

Potential applications include:

- Education decisions
- Career changes
- Relocation
- Spending interventions
- Financial policy changes

Possible approaches include:

- Structural causal models
- Directed acyclic graphs
- Counterfactual inference
- Causal discovery
- Causal effect estimation

Useful technologies:

- DoWhy
- EconML
- PyWhy

References:

https://www.pywhy.org/

https://www.pywhy.org/dowhy/

Causal conclusions require assumptions and appropriate evidence.

---

## 20. Bayesian AI

Bayesian methods represent uncertainty and update beliefs using evidence.

Possible applications include:

- Uncertain income assumptions
- Employment-transition modelling
- Parameter estimation
- Risk analysis
- Sequential information updating

Possible technologies:

- PyMC
- Stan
- NumPyro
- Pyro

References:

https://www.pymc.io/

https://mc-stan.org/

---

# PART III — COMPUTER GAMES AND IMMERSIVE TECHNOLOGIES

## 21. Computer Games as Financial Twins

Computer games can provide a natural interface for exploring financial decisions.

A Financial Twin could be implemented as a serious game in which the player navigates life events and experiences the financial consequences of choices.

Possible game mechanics include:

- Decision trees
- Branching narratives
- Resource management
- Financial strategy
- Time progression
- Life-event cards
- Scenario challenges
- Interactive goal setting
- Dynamic difficulty
- Feedback and rewards

Example:

A player controls Alexia, a 20-year-old student in Lausanne.

The player chooses between:

- Starting work
- Pursuing a master's degree
- Moving abroad

The game advances through simulated months or years.

Financial indicators change according to explicit modelling rules.

The player can restart, compare choices, and explore alternative paths.

The purpose is not to gamify financial advice.

It is to create an engaging environment for understanding consequences and trade-offs.

---

## 22. Game Engines

Game engines can support interactive Financial Twin experiences.

### Unity

Possible applications:

- 2D financial decision games
- 3D virtual environments
- Interactive avatars
- Scenario visualizations
- Mobile applications
- AR/VR experiences

Website:

https://unity.com/

### Unreal Engine

Possible applications:

- High-fidelity virtual worlds
- Interactive 3D environments
- Digital humans
- Real-time visualization
- Immersive simulations

Website:

https://www.unrealengine.com/

### Godot

Possible applications:

- Lightweight 2D and 3D games
- Open-source prototypes
- Interactive simulations
- Cross-platform experiences

Website:

https://godotengine.org/

### Web-Based Game Technologies

Possible options include:

- Phaser
- Three.js
- Babylon.js
- PlayCanvas
- WebGL
- WebGPU

These technologies allow browser-based experiences without requiring users to install a game engine.

---

## 23. Serious Games and Game-Based Learning

Serious games use game mechanics for education, training, or decision exploration.

Possible Financial Twin experiences include:

### Life Simulator

Players make education, career, housing, and investment decisions across a simulated lifetime.

### Financial Strategy Game

Players allocate limited financial resources while balancing competing goals.

### Scenario Escape Room

Players solve financial challenges using available information and constraints.

### Financial Sandbox

Players experiment with financial variables without fixed winning conditions.

### Branching Narrative

Players experience different life paths based on decisions.

### Multiplayer Household Simulation

Several players represent members of a household with shared and competing financial goals.

### Financial Resilience Game

Players respond to unexpected expenses, unemployment, or economic shocks.

### Entrepreneurial Simulation

Players manage personal finances while launching and growing a business.

Game mechanics should remain consistent with the underlying financial model.

---

## 24. Virtual Worlds and 3D Financial Environments

A Financial Twin could be represented as a virtual environment.

Examples include:

- A virtual financial city
- A digital representation of a household
- A career-path world
- An interactive timeline
- A virtual decision laboratory
- A financial ecosystem with interacting agents

Users could navigate environments, select decisions, and observe changes.

3D graphics are an interface choice, not proof that a system is a digital twin.

The underlying financial model remains essential.

---

## 25. Digital Avatars and Virtual Humans

Digital avatars may represent:

- The user
- A financial guide
- A future version of the user
- A household member
- A simulated adviser
- A fictional character

Possible technologies include:

- Animated 2D avatars
- 3D digital humans
- Conversational avatars
- Speech-driven characters
- Emotion-aware interfaces
- Virtual assistants

An avatar may improve engagement, but it should not create a false impression of financial certainty or professional authority.

---

## 26. Augmented and Virtual Reality

AR and VR may create immersive financial scenario experiences.

Possible applications include:

- Exploring future financial milestones
- Navigating a simulated household budget
- Visualizing assets and liabilities spatially
- Comparing life paths in virtual environments
- Interactive financial education

Possible technologies:

- Unity XR
- Unreal Engine XR
- WebXR
- OpenXR

Immersive experiences should remain accessible to users without specialized hardware where practical.

---

## 27. Gamification and Behavioural Design

Gamification can introduce:

- Progress indicators
- Milestones
- Challenges
- Feedback loops
- Achievements
- Interactive learning
- Goal visualization

Care is required when designing rewards.

For example, maximizing simulated wealth should not automatically be treated as the only desirable outcome.

Financial decisions involve preferences, risk, quality of life, and individual circumstances.

---

# PART IV — KNOWLEDGE, REASONING, AND INTELLIGENT SYSTEMS

## 28. Knowledge Graphs

Knowledge graphs represent relationships between entities.

Possible financial entities include:

- Person
- Employer
- Income
- Expense
- Asset
- Liability
- Goal
- Life event
- Jurisdiction
- Currency
- Financial product

A knowledge graph may help represent complex relationships and dependencies.

Possible technologies:

- Neo4j
- RDF
- SPARQL
- NetworkX
- Graph databases

References:

https://neo4j.com/

https://networkx.org/

---

## 29. Rule-Based and Expert Systems

Rule-based systems use explicit conditions and logic.

Possible applications include:

- Budget constraints
- Financial eligibility checks
- Scenario validation
- Goal feasibility
- Financial calculation rules

Possible technologies:

- Business rule engines
- Decision tables
- Logic programming
- Constraint solvers

Rule-based systems may be useful when traceability and reproducibility are important.

---

## 30. Planning and Search Algorithms

Planning algorithms can explore sequences of decisions.

Possible methods include:

- Breadth-first search
- Depth-first search
- A*
- Dynamic programming
- Monte Carlo Tree Search
- Model predictive control
- Constraint-based planning

Potential applications include:

- Career-path exploration
- Education financing
- Debt repayment sequences
- Goal scheduling
- Multi-stage financial decisions

These methods can support sequential decision exploration without requiring reinforcement learning.

---

## 31. Evolutionary and Swarm Intelligence

Possible methods include:

- Genetic algorithms
- Genetic programming
- Evolution strategies
- Particle swarm optimization
- Differential evolution
- Ant colony optimization

Potential applications include:

- Multi-objective financial planning
- Resource allocation
- Strategy exploration
- Constraint-heavy optimization

These approaches should be compared against appropriate alternatives.

---

# PART V — QUANTUM AND HYBRID COMPUTING

## 32. Quantum Computing

Quantum computing investigates computational methods based on quantum information.

Potential research areas relevant to financial systems include:

- Optimization
- Probability estimation
- Sampling
- Quantum machine learning
- Quantum reinforcement learning
- Quantum simulation

Quantum computing does not automatically improve financial predictions or simulations.

Practical advantages depend on the problem, algorithm, hardware, and comparison baseline.

---

## 33. Hybrid Quantum-Classical Computing

Hybrid approaches combine classical and quantum computational components.

A possible architecture is:

Classical financial environment

→ Classical data preparation

→ Quantum computational component

→ Classical optimization or interpretation

→ Financial scenario results

Possible research directions include:

- Variational quantum algorithms
- Quantum approximate optimization
- Quantum-enhanced machine learning
- Hybrid quantum-classical reinforcement learning
- Quantum-assisted sampling

Possible technologies:

- Qiskit
- PennyLane
- Cirq
- Amazon Braket

References:

https://www.ibm.com/quantum/qiskit

https://pennylane.ai/

https://quantumai.google/cirq

https://aws.amazon.com/braket/

Students should not claim quantum advantage without a suitable benchmark.

---

# PART VI — DATA, INTERFACES, AND SYSTEM ENGINEERING

## 34. Data Technologies

Possible data architectures include:

- Relational databases
- Document databases
- Graph databases
- Time-series databases
- Data lakes
- Event streams
- Vector databases
- In-memory data stores

Possible technologies:

- PostgreSQL
- SQLite
- DuckDB
- MongoDB
- Neo4j
- Redis
- Apache Kafka
- Parquet
- Apache Arrow

The choice depends on data volume, structure, latency, and system requirements.

---

## 35. Financial Data Standards and Interoperability

A Financial Twin may need consistent representations of:

- Currency
- Dates
- Financial transactions
- Assets
- Liabilities
- Financial goals
- Scenario parameters
- Model outputs

Relevant concepts include:

- ISO 4217 currency codes
- ISO 8601 dates
- JSON Schema
- OpenAPI
- Data dictionaries
- Versioned schemas
- Data lineage

Synthetic data should use clear units and documented conventions.

---

## 36. Visualization Technologies

Possible approaches include:

- Interactive line charts
- Scenario trees
- Sankey diagrams
- Network graphs
- Timelines
- Financial dashboards
- Heatmaps
- Sensitivity charts
- 3D environments
- Animated simulations

Possible technologies:

- Plotly
- D3.js
- ECharts
- Vega-Lite
- Matplotlib
- Streamlit
- React
- Three.js

The interface should help users distinguish observations, assumptions, simulated outcomes, and uncertainty.

---

## 37. Multimodal Interfaces

A Financial Twin may accept or produce multiple interaction types.

Possible inputs:

- Text
- Voice
- Images
- Structured financial forms
- Interactive sliders
- Gesture controls
- Game-controller input

Possible outputs:

- Text explanations
- Spoken responses
- Charts
- Animations
- 3D environments
- Interactive narratives
- Scenario reports

Multimodal interfaces should improve accessibility and comprehension rather than introduce unnecessary complexity.

---

## 38. Privacy-Preserving Technologies

Potential approaches include:

- Data minimization
- Pseudonymization
- Differential privacy
- Federated learning
- Secure multiparty computation
- Homomorphic encryption
- Trusted execution environments

These techniques have different threat models, costs, and limitations.

They do not replace the challenge requirement to avoid real UBS client data.

---

## 39. Model Validation and Verification

Financial Twin systems should be tested for:

- Accounting consistency
- Numerical correctness
- Reproducibility
- Unit consistency
- Scenario sensitivity
- Constraint satisfaction
- Model robustness
- Data provenance
- Explainability
- Security
- Appropriate uncertainty communication

Possible testing techniques include:

- Unit tests
- Property-based testing
- Integration tests
- Metamorphic testing
- Sensitivity analysis
- Stress testing
- Benchmark comparisons
- Human evaluation

---

## 40. Technology Selection Matrix

| Technology | Representation | Simulation | Decision exploration | Interaction |
|---|---|---|---|---|
| Classical financial models | High | High | Medium | Low |
| Optimization | Medium | Medium | High | Low |
| Generative AI | Medium | Low without tools | Medium | High |
| Machine learning | High for learned patterns | Depends on model | Medium | Medium |
| Reinforcement learning | High for defined environments | High with environment | High | Medium |
| Agent-based modelling | High for interacting agents | High | High | Medium |
| Game engines | High visually | High with implemented model | High | High |
| Serious games | High | High with implemented model | High | High |
| Knowledge graphs | High relationally | Low without dynamics | Medium | Medium |
| Causal models | High for causal structure | High under assumptions | High | Medium |
| Quantum computing | Task-dependent | Experimental | Experimental | Low |
| Hybrid architectures | Depends on components | Depends on components | Depends on components | Depends on interface |

These are qualitative indications, not benchmark results or technology rankings.

---

## 41. Example Solution Archetypes

### Archetype A — Conversational Financial Twin

LLM + structured financial profile + calculation engine + charts.

### Archetype B — Financial Decision Game

Game engine + financial state model + branching decisions + visual feedback.

### Archetype C — Immersive Financial World

3D environment + avatars + scenario simulation + interactive exploration.

### Archetype D — Multi-Agent Financial Ecosystem

Household agents + employer agents + economic environment + scenario engine.

### Archetype E — Reinforcement-Learning Financial Twin

Financial environment + policy learning + reward and constraint modelling.

### Archetype F — Causal Decision Laboratory

Causal graph + intervention model + counterfactual comparison.

### Archetype G — Optimization-Based Financial Twin

Financial state model + constraints + multi-objective optimization.

### Archetype H — Quantum-Classical Research Prototype

Classical financial model + experimental quantum component + classical benchmark.

### Archetype I — Knowledge-Driven Financial Twin

Knowledge graph + rules + financial simulation + explainable interface.

### Archetype J — Multimodal Financial Companion

Voice + text + visualization + financial model + interactive scenario exploration.

These archetypes are examples only.

Teams may invent entirely different architectures.

---

## 42. Final Design Principles

A strong Financial Twin should:

1. Represent a meaningful financial situation.
2. Support explicit changes to decisions or assumptions.
3. Produce internally consistent outcomes.
4. Allow meaningful comparisons.
5. Communicate limitations and uncertainty.
6. Preserve financial and computational transparency.
7. Support human judgment.
8. Avoid unsupported financial advice.
9. Use synthetic or appropriately licensed public data.
10. Provide an engaging and understandable experience.

The Financial Twin may be a game, a simulator, an AI assistant, an optimization engine, a virtual world, or a hybrid system.

The technology is the means.

The financial decision experience is the purpose.

---

## 43. Research and Learning Resources

### Digital Twins

- National Academies: https://www.nationalacademies.org/read/26894/chapter/4
- NASA Digital Twin Paradigm: https://ntrs.nasa.gov/api/citations/20120008178/downloads/20120008178.pdf

### Classical Simulation

- SciPy: https://scipy.org/
- SimPy: https://simpy.readthedocs.io/
- AnyLogic: https://www.anylogic.com/

### Artificial Intelligence

- PyTorch: https://pytorch.org/
- TensorFlow: https://www.tensorflow.org/
- Scikit-learn: https://scikit-learn.org/
- Hugging Face: https://huggingface.co/

### Reinforcement Learning

- Gymnasium: https://gymnasium.farama.org/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- Ray RLlib: https://docs.ray.io/en/latest/rllib/

### Computer Games

- Unity: https://unity.com/
- Unreal Engine: https://www.unrealengine.com/
- Godot: https://godotengine.org/
- Phaser: https://phaser.io/
- Three.js: https://threejs.org/
- Babylon.js: https://www.babylonjs.com/

### Optimization

- Google OR-Tools: https://developers.google.com/optimization
- CVXPY: https://www.cvxpy.org/
- Pyomo: https://pyomo.readthedocs.io/

### Quantum Computing

- IBM Qiskit: https://www.ibm.com/quantum/qiskit
- PennyLane: https://pennylane.ai/
- Cirq: https://quantumai.google/cirq

### Visualization

- Plotly: https://plotly.com/
- D3.js: https://d3js.org/
- Streamlit: https://streamlit.io/
- React: https://react.dev/

All external technologies are optional. Their inclusion does not imply UBS endorsement, guaranteed availability, or approval for use in UBS systems.

---

## 44. Responsible Development

Participants must use synthetic or appropriately licensed public data.

Real UBS client information must not be used.

Financial calculations should be verifiable.

Assumptions and limitations should be transparent.

Simulated results should not be presented as guaranteed future outcomes.

Students are responsible for respecting the licenses and security requirements of third-party technologies.

---

## Final Message

**A Financial Twin is not a prediction engine, a chatbot, or a dashboard alone.**

It is an interactive computational representation that helps people explore decisions and understand their possible consequences.

The implementation is open.

**Don't predict the future. Make it explorable.**
