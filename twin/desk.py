"""Start with: python -m streamlit run twin/desk.py

A yearly desk: check in once a year with December figures, decide at the January review,
and see the spread of possible lives as a fan chart with the reasons behind it.
"""
import json
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from twin import config as C
from twin import report
from twin.data import profiles
from twin.explore import compare, fan, simulate, summarize
from twin.live import branch, outcomes, read_journal, record, salary_search
from twin.policy.traits import from_profile
from twin.world.env import World

st.set_page_config(page_title="Financial Twin", layout="wide")
st.title("Financial Twin")
db = Path(os.environ.get("TWIN_JOURNAL", ".twin-sessions.sqlite3"))
ids = profiles().index.tolist()
pid = st.selectbox("Fictional profile", ids, index=ids.index("P00003"))
people = profiles().loc[[pid]]
person = people.iloc[0]
journal = read_journal(db, pid)
current = journal.current_month
review = C.next_review(current)
world = st.cache_resource(World.load)()
paths_count = st.sidebar.selectbox("Simulation paths", [200, 500, 2000], index=1)
free_time = st.sidebar.slider("Value of free time", 0., 1., float(from_profile(person).traits["free_time"]), key=pid)
persona = from_profile(person)
persona.traits["free_time"] = free_time
seed = C.SEED
MONTHS = pd.to_datetime([C.month_label(t) for t in range(C.HORIZON + 1)])


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, float) and np.isnan(x):
        return None
    return x


@st.cache_data(max_entries=24, show_spinner="Simulating possible lives…")
def project(profile_id, revision, preference, n, salary, _journal, _world):
    rows = profiles().loc[[profile_id]]
    p = from_profile(rows.iloc[0])
    p.traits["free_time"] = preference
    base = simulate(_world, rows, p, n, seed, timeline=_journal.timeline)
    out = outcomes(base)
    as_of = _journal.current_month
    out.update(cash=float(base.cash[0, as_of]), invested=float(base.inv[0, as_of]),
               debt=float(base.debt[0, as_of]), income=float(base.income[0, as_of]),
               fan_nw=fan(base.nw), fan_cash=fan(base.cash))
    summary = summarize(base)
    out["summary"] = jsonable({"person": {k: (None if pd.isna(v) else v) for k, v in rows.iloc[0].items()},
                               "persona": {"traits": p.traits, "temperature": p.temperature, "spread": p.spread,
                                           "intentions": [], "policy": "rule persona"},
                               "n_paths": n, "seed": seed, **summary})
    if salary is None:
        return out, None
    alt_paths = simulate(_world, rows, p, n, seed, timeline=branch(_journal, C.PART_TIME, salary))
    delta = compare(base, alt_paths)
    delta["part_time_months"] = float(((alt_paths.mode[:, 1:] == C.PART_TIME).sum(1)
                                       - (base.mode[:, 1:] == C.PART_TIME).sum(1)).mean())
    delta["median_nw"] = np.median(alt_paths.nw, 0)
    delta["median_cash"] = np.median(alt_paths.cash, 0)
    return out, delta


def fan_chart(f, measure, offer_median=None, checkins=(), goal=None):
    """Fan chart: 5-95% and 25-75% of paths, median, check-ins, offer preview, goal marker."""
    df = pd.DataFrame({"date": MONTHS, "p5": f[0], "p25": f[1], "p50": f[2], "p75": f[3], "p95": f[4]})
    x = alt.X("date:T", title=None, axis=alt.Axis(format="%Y", tickCount="year"))
    y = alt.Y("p5:Q", title=f"{measure}, CHF", axis=alt.Axis(format="~s"))
    base = alt.Chart(df).encode(x=x)
    hover = alt.selection_point(nearest=True, on="pointerover", fields=["date"], empty=False)
    layers = [
        base.mark_area(color="#86b6ef", opacity=0.35).encode(y=y, y2="p95:Q"),
        base.mark_area(color="#2a78d6", opacity=0.45).encode(y="p25:Q", y2="p75:Q"),
        base.mark_line(color="#1c5cab", strokeWidth=2.5).encode(y="p50:Q"),
        base.mark_rule(color="#888").encode(
            opacity=alt.condition(hover, alt.value(0.7), alt.value(0)),
            tooltip=[alt.Tooltip("date:T", title="Month", format="%b %Y")]
            + [alt.Tooltip(f"{q}:Q", title=t, format=",.0f") for q, t in
               (("p95", "95th pct"), ("p75", "75th"), ("p50", "median"), ("p25", "25th"), ("p5", "5th"))]
        ).add_params(hover),
    ]
    if offer_median is not None:
        od = pd.DataFrame({"date": MONTHS, "offer": offer_median})
        layers.append(alt.Chart(od).mark_line(color="#eb6834", strokeWidth=2, strokeDash=[6, 4])
                      .encode(x=x, y="offer:Q", tooltip=[alt.Tooltip("offer:Q", title="Median with offer", format=",.0f")]))
    if len(checkins):
        cd = pd.DataFrame({"date": MONTHS[[t + 1 for t, _ in checkins]], "value": [v for _, v in checkins]})
        layers.append(alt.Chart(cd).mark_point(color="black", filled=True, size=60)
                      .encode(x=x, y="value:Q", tooltip=[alt.Tooltip("date:T", title="Check-in", format="%b %Y"),
                                                         alt.Tooltip("value:Q", title="Recorded", format=",.0f")]))
    if goal is not None:
        gd = pd.DataFrame({"date": [MONTHS[C.T_HIGH_GOAL + 1]], "target": [goal]})
        layers.append(alt.Chart(gd).mark_point(shape="diamond", size=140, color="black", filled=False, strokeWidth=2)
                      .encode(x=x, y="target:Q", tooltip=[alt.Tooltip("target:Q", title="Goal target", format=",.0f")]))
    return alt.layer(*layers).properties(height=360)


# Include the full observed history in the cache key, including journals at different paths.
key = (journal.revision, repr(journal.timeline))
baseline, _ = project(pid, key, free_time, paths_count, None, journal, world)
as_of = f"start of {C.month_label(current)}" if current < C.HORIZON else "end of 2034"
st.subheader(f"{person['name']} · {as_of}")
available = baseline["cash"] if person.high_goal == "Emergency savings" else baseline["cash"] + baseline["invested"] - baseline["debt"]
gap = max(0, person.high_target - available)
st.write(f"Cash CHF {baseline['cash']:,.0f} · Investments CHF {baseline['invested']:,.0f} · "
         f"Debt CHF {baseline['debt']:,.0f} · Monthly take-home CHF {baseline['income']:,.0f}")
st.write(f"{person.high_goal} · CHF {person.high_target:,.0f} by December 2029. Current gap CHF {gap:,.0f}. "
         "Goal success requires staying within the credit limit through the deadline.")
a, b, c = st.columns(3)
a.metric("Goal success", f"{baseline['goal']:.1%}")
b.metric("Credit limit exceeded by 2034", f"{baseline['breach']:.1%}")
c.metric("Median wealth in 2034", f"CHF {baseline['wealth']:,.0f}")
st.caption("Conditional estimates from the rule persona. Every voluntary decision (work, housing, lifestyle, "
           "investing) is made once a year at the January review; if cash falls below zero in between, "
           "investments are sold automatically back to one month of outgoings.")

committed = journal.timeline.offers.get(review)
can_offer = review < C.HORIZON and person.age + review / 12 < world.pr["jobs"]["forced_retirement_age"] and not committed
salary, delta = None, None
if committed:
    st.success(f"Committed for {C.month_label(review)}: monthly net pay CHF {committed[1]:,.0f}.")
if can_offer:
    st.subheader(f"Part-time offer for the January {C.month_label(review)[:4]} review")
    salary = st.number_input("Offered monthly take-home pay (CHF)", min_value=1., value=5000., step=100.)
    _, delta = project(pid, key, free_time, paths_count, salary, journal, world)

# ----------------------------------------------------------------------------- the chart
st.subheader("How the money could move")
measure = st.radio("Show", ["Net worth", "Cash"], horizontal=True, label_visibility="collapsed")
f = baseline["fan_nw"] if measure == "Net worth" else baseline["fan_cash"]
obs = journal.timeline.observations
checkins = [(t, f[2][t + 1]) for t in sorted(obs)]       # all paths sit on the check-in figures there
goal_on_this_chart = (person.high_goal == "Emergency savings") == (measure == "Cash")
offer_line = None if delta is None else (delta["median_nw"] if measure == "Net worth" else delta["median_cash"])
st.altair_chart(fan_chart(f, measure, offer_line, checkins, person.high_target if goal_on_this_chart else None),
                width="stretch")
legend = "Light band: 5–95% of paths · dark band: 25–75% · line: median"
legend += " · black dots: your year-end check-ins" if checkins else ""
legend += " · diamond: high goal, Dec 2029" if goal_on_this_chart else ""
legend += " · dashed orange: median if you take the offer" if offer_line is not None else ""
st.caption(legend)

if can_offer and delta is not None:
    lo, hi = delta["goal_change_mc95_pp"]
    wealth_change = delta["median_final_net_worth"][1] - baseline["wealth"]
    st.write(f"With the offer, goal success is {delta['high_goal_odds'][1]:.1%} "
             f"({delta['goal_change_pp']:+.1f} percentage points; approximate 95% paired sampling interval "
             f"{lo:+.1f} to {hi:+.1f}). Median wealth changes by CHF {wealth_change:+,.0f}.")
    st.write(f"The offer changes expected time in part-time work by {delta['part_time_months']:+.1f} months. "
             f"Credit-limit breaches change from {baseline['breach']:.1%} to {delta['credit_breach_odds'][1]:.1%}.")
    st.caption("The interval covers simulation sampling error under the chosen assumptions.")
    if st.button("Commit this offer"):
        try:
            record(db, journal, review, "offer", [C.PART_TIME, salary])
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    if review <= C.T_HIGH_GOAL:
        with st.form("search"):
            st.subheader("What salary would preserve my goal?")
            grid = st.text_input("Monthly take-home offers to test (CHF)", "3000, 4000, 5000, 6000, 7000")
            target = st.slider("Target goal success (%)", 1, 100, 90)
            search = st.form_submit_button("Compare salary offers")
        if search:
            try:
                salaries = [float(v.strip()) for v in grid.split(",")]
                if len(salaries) > 30:
                    raise ValueError("Compare up to 30 offers at a time.")
                minimum, result = salary_search(world, people, persona, journal, salaries, target/100, paths_count, seed)
                st.write(f"Lowest tested qualifying salary: CHF {minimum:,.0f}/month." if minimum is not None
                         else "No tested salary reaches this threshold.")
                st.dataframe(pd.DataFrame(result).rename(columns={"goal":"Goal success (fraction)", "breach":"Credit breach (fraction)",
                             "shortfall":"Negative cash (fraction)", "wealth":"Median final wealth (CHF)", "salary":"Salary (CHF/month)"}), hide_index=True)
            except ValueError as exc:
                st.error(str(exc))

# -------------------------------------------------------------------------- explanation
sm = baseline["summary"]
st.subheader("Why it comes out this way")
left, right = st.columns(2)
with left:
    st.markdown("**The most common ways the ten years go**")
    st.dataframe(pd.DataFrame([{"Story": s["story"], "Share": f"{s['share']:.0%}",
                                "From": s["typical_change_year"] or "—",
                                "Median wealth 2034": f"CHF {s['median_final_net_worth']:,.0f}",
                                "Goal met": f"{s['high_goal_odds']:.0%}"} for s in sm["stories"]]), hide_index=True)
with right:
    st.markdown("**What separates the better paths from the worse**")
    st.dataframe(pd.DataFrame([{"On the path": t["feature"], "Share": f"{t['share']:.0%}",
                                "Median wealth with": f"CHF {t['median_nw_with']:,.0f}",
                                "without": f"CHF {t['median_nw_without']:,.0f}",
                                "Goal with / without": f"{t['high_goal_with']:.0%} / {t['high_goal_without']:.0%}"}
                               for t in sm["turning_points"]]), hide_index=True)
    st.caption("Associations across sampled lives, not causes.")
val_path = C.ARTIFACTS / "validation.json"
html = report.render(sm, json.loads(val_path.read_text()) if val_path.exists() else None)
st.download_button("Download the full report (HTML)", html, file_name=f"{pid}-twin-report.html", mime="text/html")

# ------------------------------------------------------------------------ year-end check-in
if current < C.HORIZON:
    checkin = journal.next_checkin
    year = C.month_label(checkin)[:4]
    with st.form("observed", clear_on_submit=True):
        st.subheader(f"Year-end check-in for {year}")
        st.caption(f"Once a year: enter your figures at the end of December {year}. The months in between are "
                   "simulated; your figures then replace the simulated ones.")
        cash = st.number_input("Cash at 31 December (CHF)", value=None)
        assets = st.number_input("Investments at 31 December (optional CHF)", value=None, min_value=0.)
        income = st.number_input("Monthly take-home income for next year (optional CHF)", value=None, min_value=0.)
        rent = st.number_input("Monthly rent for next year (optional CHF)", value=None, min_value=1.)
        status = st.selectbox("Employment status at year end", ["Carry forward", *C.STATUS_TO_MODE])
        flow = st.number_input("Average monthly net flow this year (optional CHF)", value=None,
                               help="Income less normal living costs, excluding investment transfers, loan payments and "
                                    "one-off scenario costs. Lets the twin learn your typical surplus.")
        save = st.form_submit_button("Save year and update projections")
    if save:
        try:
            if cash is None:
                raise ValueError("Enter cash at 31 December.")
            fields = dict(cash=cash, investments=assets, rent=rent, income=income, avg_net_flow=flow,
                          status=None if status == "Carry forward" else status)
            record(db, journal, checkin, "observation", {k: v for k, v in fields.items() if v is not None})
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
if journal.timeline.observations or journal.timeline.offers:
    with st.expander("Saved timeline"):
        for t in sorted(set(journal.timeline.observations) | set(journal.timeline.offers)):
            st.write(C.month_label(t), journal.timeline.offers.get(t, ""), journal.timeline.observations.get(t, ""))
