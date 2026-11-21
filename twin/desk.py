"""Start with: python -m streamlit run twin/desk.py"""
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from twin import config as C
from twin.data import profiles
from twin.explore import compare, simulate
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


@st.cache_data(max_entries=24, show_spinner="Comparing possible futures…")
def project(profile_id, revision, preference, n, salary, _journal, _world):
    rows = profiles().loc[[profile_id]]
    p = from_profile(rows.iloc[0])
    p.traits["free_time"] = preference
    base = simulate(_world, rows, p, n, seed, timeline=_journal.timeline)
    out = outcomes(base)
    as_of = _journal.current_month
    out.update(cash=float(base.cash[0, as_of]), invested=float(base.inv[0, as_of]),
               debt=float(base.debt[0, as_of]), income=float(base.income[0, as_of]))
    if salary is None:
        return out, None
    alt = simulate(_world, rows, p, n, seed, timeline=branch(_journal, C.PART_TIME, salary))
    delta = compare(base, alt)
    delta["part_time_months"] = float(((alt.mode[:, 1:] == C.PART_TIME).sum(1)
                                       - (base.mode[:, 1:] == C.PART_TIME).sum(1)).mean())
    return out, delta


# Include the full observed history in the cache key, including journals at different paths.
key = (journal.revision, repr(journal.timeline))
baseline, _ = project(pid, key, free_time, paths_count, None, journal, world)
st.subheader(f"{person['name']} · {C.month_label(current) if current < C.HORIZON else 'End of 2034'}")
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
st.caption("Conditional estimates from the rule persona. Voluntary work and housing decisions take place in January.")
committed = journal.timeline.offers.get(review)
can_offer = review < C.HORIZON and person.age + review / 12 < world.pr["jobs"]["forced_retirement_age"] and not committed
if committed:
    st.success(f"Committed for {C.month_label(review)}: monthly net pay CHF {committed[1]:,.0f}.")
if can_offer:
    st.subheader(f"Part-time offer for {C.month_label(review)}")
    salary = st.number_input("Offered monthly take-home pay (CHF)", min_value=1., value=5000., step=100.)
    _, alt = project(pid, key, free_time, paths_count, salary, journal, world)
    lo, hi = alt["goal_change_mc95_pp"]
    wealth_change = alt["median_final_net_worth"][1] - baseline["wealth"]
    st.write(f"With the offer, goal success is {alt['high_goal_odds'][1]:.1%} "
             f"({alt['goal_change_pp']:+.1f} percentage points; approximate 95% paired sampling interval "
             f"{lo:+.1f} to {hi:+.1f}). Median wealth changes by CHF {wealth_change:+,.0f}.")
    st.write(f"The offer changes expected time in part-time work by {alt['part_time_months']:+.1f} months. "
             f"Credit-limit breaches change from {baseline['breach']:.1%} to {alt['credit_breach_odds'][1]:.1%}.")
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
if current < C.HORIZON:
    with st.form("observed", clear_on_submit=True):
        st.subheader(f"Record actuals for {C.month_label(current)}")
        cash = st.number_input("Closing cash (CHF)", value=None)
        rent = st.number_input("Updated monthly rent for next month (optional CHF)", value=None, min_value=1.)
        assets = st.number_input("Closing investments (optional CHF)", value=None, min_value=0.)
        income = st.number_input("Updated take-home income for next month (optional CHF)", value=None, min_value=0.)
        status = st.selectbox("Employment status at close", ["Carry forward", *C.STATUS_TO_MODE])
        flow = st.number_input("Comparable monthly net flow (optional CHF)", value=None,
                               help="Income less normal living costs, excluding investment transfers, loan payments and exceptional scenario costs. Enables a recurring-surplus adjustment.")
        save = st.form_submit_button("Save month and update projections")
    if save:
        try:
            if cash is None:
                raise ValueError("Enter closing cash.")
            fields = dict(cash=cash, rent=rent, investments=assets, net_flow=flow, income=income,
                          status=None if status == "Carry forward" else status)
            record(db, journal, current, "observation", {k:v for k,v in fields.items() if v is not None})
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
if journal.timeline.observations or journal.timeline.offers:
    with st.expander("Saved timeline"):
        for t in sorted(set(journal.timeline.observations) | set(journal.timeline.offers)):
            st.write(C.month_label(t), journal.timeline.offers.get(t, ""), journal.timeline.observations.get(t, ""))
