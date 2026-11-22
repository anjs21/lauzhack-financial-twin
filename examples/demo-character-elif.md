# Demo character: Elif Demir

A fictional person for demonstrating the twin. The web app's built-in example is her profile
under the name **P0067** ("See an example"; regenerate with `python -m twin.plan.make_example`). Every number in the walkthrough comes from the
simulator: 500 lives, the fixed seed and the default decision style, unless noted otherwise.

## Who she is

Elif is 24 and in the last 14 months of a master's in data science in Zurich. She shares a flat
in Wiedikon for CHF 950 a month. A teaching-assistant job and some help from her parents bring
in about CHF 2,000 a month. She has CHF 9,000 saved and owes CHF 4,000 at 4% on the loan that
paid for her move and laptop.

Her goal is a **CHF 40,000 deposit by the end of 2029**, so that she and her partner Jonas can
stop renting one day. She is ambitious and tracks her savings closely, but she is not a
penny-pincher.

**The decision (January 2025):** a fintech start-up has offered her a 60% analyst role at
**CHF 3,200 a month** take-home. She can work it alongside her studies, or drop the master's and
take the job straight away. Her parents want her to finish the degree first. The start-up wants
an answer this month.

## Enter her in the app

**Build my persona**

| Field | Value |
|---|---|
| What should we call you? | Elif |
| Age at the start of 2025 | 24 |
| Where do you live? | Zurich |
| Your household | Sharing a home |
| Your work situation | Studying · Months until you finish **14** · ✔ It's a master's degree |
| Monthly take-home income | 2000 |
| Monthly rent | 950 |
| Cash & savings | 9000 |
| Investments | 0 |
| Outstanding debt | 4000 |
| Annual debt rate | 0.04 |
| Your main money goal | Housing deposit |
| Target by December 2029 | 40000 |

**What matters to you** (open *More preferences* for all eight sliders):

| Statement | Value |
|---|---|
| Moving up in my career or studies matters a lot to me. | 0.85 |
| I keep a close eye on my savings goals and deadlines. | 0.70 |
| I save first and spend what is left. | 0.60 |
| Living somewhere bigger or nicer is worth paying for. | 0.60 |
| I am comfortable holding investments that can lose value for a while. | 0.55 |
| I would rather enjoy money now than later. | 0.45 |
| I would trade some income for more free time. | 0.35 |
| Once things work, I rarely change them. | 0.30 |

Then choose **Go part-time**, enter **CHF 3200 / month**, leave **Keep studying** on, and click
**Explore this path**.

## The story the numbers tell

| Path | Deposit by 2029 | Credit-limit risk | Net worth 2034 (median) |
|---|---|---|---|
| Current path: study only | 56% | 43% | CHF 928K |
| **Take the job alongside her studies** | **98%** | **2%** | **CHF 1.13M** |
| Drop the master's for the job | 43% | 46% | CHF 164K |
| Drop the master's, deciding deliberately | 59% | 31% | CHF 221K |

1. **Studying alone is tight.** On CHF 2,000 a month in Zurich, 43% of her lives hit the credit
   limit before 2029. The job alongside her studies almost removes that risk.
2. **"Keep studying" is the real decision.** Switch it off and explore again. Without the
   degree, median net worth in 2034 falls from CHF 1.13M to CHF 164K. Most of those lives go back
   for a master's around 2028 (72%), but only about half ever finish one, typically around 2030.
3. **Key moments** on the alongside path: takes the job (2025, every life), finishes studies
   (2026, 94%), starts a full-time job (2026, 90%), partner moves in (2029, 38%), moves to a
   bigger home (2029, 58%), loses a job at some point (2030, 33%).
4. **Decide deliberately** helps most where habit hurts: on the drop-out path it lifts the
   deposit odds from 43% to 59%, mainly by moving her up to full-time by 2027 (62%).

## Branches to grow on the chart

Start from the **Part-time while studying** path and click a January dot:

| Branch | Deposit by 2029 | Credit risk | Net worth 2034 |
|---|---|---|---|
| Now (2025) → *Lose my job* | 47% | 53% | CHF 980K |
| 2026 → *A recession hits* | 82% | 18% | CHF 1.02M |
| 2027 → *Move somewhere bigger* | 98% | 2% | CHF 1.08M |
| 2027 → *Partner moves in*, then 2030 → *Have a child* | 98% | 2% | CHF 956K |

The lesson: early shocks threaten her goal, while late ones mostly cost long-run wealth. Losing
the job in her first month leaves her worse off than never taking it (47% against 56%), because
in the model the offer replaces her TA income. By 2027 she is graduated and employed, so moving,
a partner or a child barely touch the 2029 deposit.

## Things to say to the assistant

Each line shows the tool the assistant should use.

- *"What if I lose the start-up job right away?"* → a branch: lose my job, January 2025.
- *"And what if a recession hits in 2026 on top of that?"* → a branch from that branch.
- *"Jonas is moving in with me in 2027."* → a branch: partner moves in, 2027.
- *"Actually I'm 25 and the TA job pays 2,300 now."* → updates the persona and re-runs every life.
- *"Show me net worth for the drop-out path."* → highlights the path and switches the chart.
- *"Why does finishing the degree matter so much?"* → a plain answer, no tool.

## Her plan, in her own words

**Make my own plan:**

> I'll keep the start-up job at 60% while I finish my master's, then go full-time there. Jonas
> and I want to move in together in 2027 and find something bigger once we both earn properly.
> I want the deposit by 2030. If money gets tight I'd rather cut spending than touch
> investments.

**Something changed? (from 2026):**

> The start-up is cutting everyone to 40% next year, so I'll look for a full-time job elsewhere
> after graduating instead.

## For the API (curl or tests)

```json
{
  "person": {"name": "Elif", "age": 24, "city": "Zurich", "status": "student",
             "household_type": "shared_housing", "income": 2000, "rent": 950, "cash": 9000,
             "investments": 0, "debt": 4000, "rate": 0.04, "high_goal": "Housing deposit",
             "high_target": 40000, "study_months_left": 14, "in_masters": true},
  "traits": {"risk_tolerance": 0.55, "frugality": 0.6, "career_drive": 0.85, "free_time": 0.35,
             "housing_aspiration": 0.6, "inertia": 0.3, "impatience": 0.45, "goal_focus": 0.7},
  "scenario": {"kind": "part_time", "salary": 3200, "alongside_studies": true},
  "n_paths": 500
}
```

Elif, Jonas and the start-up are fictional. This is not financial advice.
