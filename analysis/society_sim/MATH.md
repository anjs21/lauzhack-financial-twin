# Learning a transition function from a small simulated society

The maths behind `toy_market.py`, with a three-person example you can check by hand.

## 1. The problem

Each person is an MDP: state $s$, action $a$, next state $s' \sim P(s' \mid s, a)$.

The dataset never shows anyone change job, so it cannot tell us, for example,

$$P(\text{employed next month} \mid \text{searching},\ \text{effort } a).$$

Idea: **simulate a small society, watch people get jobs, and learn $P$ from that.**

## 2. Why other people matter

Getting a job depends on who else is applying. So the true transition is

$$P\big(s_i' \mid s_i, a_i,\ \underbrace{s_{-i}, a_{-i}}_{\text{everyone else}}\big),$$

not $P(s_i' \mid s_i, a_i)$. One person's transition is not a property of that person alone.

## 3. A three-person example

Two job openings ($V = 2$) and three people searching:

| Person | Degree | Applications $a_i$ |
|---|---|---|
| Alexia | bachelor | 2 |
| Ben | master | 1 |
| Chloé | bachelor | 1 |

Rules:
1. Each person sends a fixed number of applications $a_i$.
2. Each application goes to opening 1 or opening 2 with probability $1/2$, independently, so two of Alexia's applications can land on the same opening.
3. Each opening offers the job to its best applicant: master beats bachelor, and a tie is a coin flip.
4. A person gets an offer if they win at least one opening.

**Ben** always wins wherever he applies: $P = 1$.

**Alexia** (2 applications):
- **Both land on the same opening** (probability $\tfrac12$). Ben must be elsewhere ($\tfrac12$). Then she wins outright if Chloé is elsewhere, or on a coin flip if not: $\tfrac12 \cdot 1 + \tfrac12 \cdot \tfrac12 = \tfrac34$. Total for this case: $\tfrac12 \cdot \tfrac34 = \tfrac38$.
- **They land on different openings** (probability $\tfrac12$). Ben takes one of them; at the other she wins with probability $\tfrac34$, by the same argument.

$$P(\text{Alexia gets an offer}) = \tfrac12 \cdot \tfrac38 + \tfrac12 \cdot \tfrac34 = \tfrac{9}{16} \approx 0.56$$

**Chloé** (1 application): Ben must be elsewhere ($\tfrac12$). Alexia misses her opening with probability $\tfrac14$, otherwise it's a coin flip:

$$P(\text{Chloé gets an offer}) = \tfrac12\left(\tfrac14 \cdot 1 + \tfrac34 \cdot \tfrac12\right) = \tfrac{5}{16} \approx 0.31$$

**Change only the others, and Alexia's transition changes** (exact values, checked by enumerating every case):

| World | Alexia's $P(\text{offer})$ |
|---|---|
| As above | $9/16 = 0.56$ |
| Chloé applies twice | $15/32 = 0.47$ |
| Three openings instead of two | $7/9 = 0.78$ |
| One opening (Ben takes it) | $0$ |

## 4. Summarising "everyone else" in one number

We can't feed the transition function every other person's state. In a big market, **one number captures most of it, the tightness**:

$$\theta = \frac{V}{A} = \frac{\text{openings}}{\text{total applications}}.$$

Each opening receives about $\lambda = 1/\theta$ other applications, the actual number $K$ being roughly Poisson. In a fair lottery, one application wins with probability

$$q = \mathbb{E}\left[\frac{1}{1+K}\right] = \theta\left(1 - e^{-1/\theta}\right) \approx \theta \quad (\theta \text{ small}).$$

With $a$ applications:

$$P(\text{offer}) = 1 - (1-q)^a \approx 1 - e^{-a\theta}.$$

Taking logs twice gives a straight line:

$$\log\big(-\log(1 - P)\big) \approx 1 \cdot \log a + 1 \cdot \log \theta.$$

Rankings (degree, experience, months unemployed $x_i$) add a personal advantage $\beta^\top x_i$. So the model to learn is

$$\boxed{\ \log\big(-\log(1 - P_i)\big) = \beta_0 + \beta_a \log a_i + \beta_\theta \log \theta_t + \beta^\top x_i\ }$$

This is a GLM with the complementary log-log link. Theory predicts $\beta_a \approx 1$ and $\beta_\theta \approx 1$. The toy simulation fits $\beta_a = 0.92$ and $\beta_\theta = 0.93$.

## 5. Learning it

1. Simulate many months of a 100-person society.
2. Log one row per searching person per month: $(x_i,\ a_i,\ \theta_t,\ y_i)$, where $y_i = 1$ if they got an offer.
3. Maximise the likelihood:

$$\hat\beta = \arg\max_\beta \sum_{i,t} \Big[ y_{it} \log P_\beta + (1 - y_{it}) \log(1 - P_\beta) \Big].$$

Receiving an offer is the **transition**. Accepting it is the person's **action**, so it is not part of $P$.

## 6. The trap: forgetting $\theta$

Suppose we fit Alexia's chance without $\theta$ in the world where Chloé applies once: $\hat P = 0.56$. Now Chloé starts applying twice. The truth is $0.47$, but the model still says $0.56$.

A model that ignores the market is right only for the behaviour it was trained on. This is the Lucas critique. In the toy, that model overestimated hiring by 7 points; with $\theta$ included, the error is about 1 point.

## 7. Equilibrium

$\theta$ depends on how hard people search, and how hard they search depends on $\theta$. A consistent market is a fixed point:

$$\theta^* = \frac{V}{\sum_i a_i(\theta^*)}.$$

**Example:** $V = 2$, three people, each using the rule "apply twice if $\theta < 0.5$, otherwise once".

| Everyone applies | $A$ | $\theta$ | Does the rule agree? |
|---|---|---|---|
| once | 3 | $2/3$ | yes, $\theta \ge 0.5$ → once |
| twice | 6 | $1/3$ | yes, $\theta < 0.5$ → twice |

There are **two equilibria**: a calm market and a frantic one. Which one you get depends on where you start, which is why we find $\theta^*$ by simulating.

## 8. How the twin uses it

In a society of 100 people, one more person (the twin) barely moves $\theta$. So the twin takes $\theta^*$ as given and its job transition is

$$P(\text{offer} \mid s, a) = 1 - \exp\!\big(-\exp(\hat\beta_0 + \hat\beta_a \log a + \hat\beta_\theta \log \theta^*_M + \hat\beta^\top x)\big),$$

where $\theta^*_M$ is the equilibrium tightness in economy regime $M$. Cash still comes from the model fitted on the ledger.

What is learned and what is assumed:
- **The market rules** (limited openings, ranking): code.
- **Their scale** (openings, job loss): calibrated to Swiss statistics.
- **The behaviour** (who searches, how hard): personas.
- **$\hat\beta$**: learned from the simulation.
