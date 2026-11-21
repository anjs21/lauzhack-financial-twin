"""Audit bundled synthetic data and measure descriptive MDP action relevance.

Run from any directory: python analysis/analyze_mdp.py
Uses only the Python standard library. Writes CSV/JSON alongside this script.
Sensitivity replays hold the recorded monthly net flows fixed. They are not
causal estimates, forecasts, policy evaluations, or fitted MDP transitions.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "results"


def read(name):
    with (ROOT / "Data" / f"{name}.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantiles(values, digits=2):
    values = sorted(values)
    def q(p):
        x = (len(values) - 1) * p
        lo, hi = math.floor(x), math.ceil(x)
        return round(values[lo] + (values[hi] - values[lo]) * (x - lo), digits)
    return {"min": q(0), "p25": q(.25), "median": q(.5), "p75": q(.75), "max": q(1)}


def main():
    OUT.mkdir(exist_ok=True)
    names = ["synthetic-profiles", "synthetic-accounts", "synthetic-assets",
             "synthetic-liabilities", "synthetic-goals", "synthetic-cash-balances",
             "scenario-assumptions", "scenario-catalog"]
    data = {name: read(name) for name in names}
    profiles = data["synthetic-profiles"]
    ids = {p["profile_id"] for p in profiles}
    assert len(ids) == len(profiles), "Duplicate profile IDs"
    history, investments, loans, goals = defaultdict(list), defaultdict(float), defaultdict(list), defaultdict(list)
    for name, rows in data.items():
        if rows and "profile_id" in rows[0]:
            assert all(r["profile_id"] in ids for r in rows), f"Orphan profile in {name}"
    for row in data["synthetic-cash-balances"]:
        history[row["profile_id"]].append(row)
        assert Decimal(row["opening_cash_chf"]) + Decimal(row["net_transactions_chf"]) == Decimal(row["closing_cash_chf"])
        assert (Decimal(row["closing_cash_chf"]) < 0) == (row["status"] == "negative_balance_requires_financing")
    for row in data["synthetic-assets"]:
        if row["asset_type"] == "investment_portfolio":
            investments[row["profile_id"]] += float(row["value_chf"])
    for row in data["synthetic-liabilities"]:
        loans[row["profile_id"]].append(row)
    for row in data["synthetic-goals"]:
        goals[row["profile_id"]].append(row)

    features, replays = [], []
    goal_formula_matches = 0
    for p in profiles:
        pid = p["profile_id"]
        rows = sorted(history[pid], key=lambda r: r["month"])
        assert len(rows) == 24 and len({r["month"] for r in rows}) == 24
        assert [r["month"] for r in rows] == [f"{y}-{m:02}" for y in (2025, 2026) for m in range(1, 13)]
        assert Decimal(rows[0]["opening_cash_chf"]) == Decimal(p["opening_cash_chf"])
        assert all(Decimal(a["closing_cash_chf"]) == Decimal(b["opening_cash_chf"]) for a, b in zip(rows, rows[1:]))
        net = [float(r["net_transactions_chf"]) for r in rows]
        closing = [float(r["closing_cash_chf"]) for r in rows]
        opening = float(p["opening_cash_chf"])
        shortfall = min(closing) < 0
        funding_gap = max(0, -min(closing))
        income, rent = float(p["reference_monthly_net_income_chf"]), float(p["reference_monthly_rent_chf"])
        principal = sum(float(r["opening_principal_chf"]) for r in loans[pid])
        interest = sum(float(r["opening_principal_chf"]) * float(r["annual_interest_rate"]) / 12 for r in loans[pid])
        high = next(g for g in goals[pid] if g["priority"] == "high")
        goal_formula_matches += abs(float(high["target_amount_chf"]) - round(max(10000, 1.6 * opening), 2)) < .011
        # A diagnostic from the JANUARY 2025 snapshot to DECEMBER 2029: 60 months.
        # Does not consume goals, allocate investments, model debt, or predict income.
        required_saving = max(0, (float(high["target_amount_chf"]) - opening) / 60)
        f = dict(profile_id=pid, employment_status=p["employment_status"], age_at_start=int(p["age_at_start"]),
                 household_type=p["household_type"], opening_cash_chf=opening,
                 opening_investments_chf=investments[pid], opening_debt_chf=principal,
                 reference_income_chf=income, reference_rent_chf=rent,
                 rent_to_reference_income=round(rent / income, 4),
                 mean_monthly_net_chf=round(mean(net), 2), sd_monthly_net_chf=round(stdev(net), 2),
                 negative_net_months=sum(x < 0 for x in net), negative_cash_months=sum(x < 0 for x in closing),
                 first_negative_cash_month=next((r["month"] for r in rows if float(r["closing_cash_chf"]) < 0), ""),
                 history_min_closing_cash_chf=min(closing), history_final_cash_chf=closing[-1],
                 observed_cash_shortfall=shortfall, negative_mean_net=mean(net) < 0,
                 max_recorded_funding_gap_chf=round(funding_gap, 2),
                 constant_monthly_uplift_to_avoid_shortfall_chf=round(max([0] + [-c / (i+1) for i, c in enumerate(closing)]), 2),
                 debt_first_month_interest_assuming_apr_div_12_chf=round(interest, 2),
                 high_priority_goal_chf=float(high["target_amount_chf"]),
                 goal_required_monthly_cash_saving_60m_chf=round(required_saving, 2),
                 history_mean_below_goal_saving_rate=mean(net) < required_saving)
        features.append(f)

        # Single-lever replays; all unmodified monthly flows are held fixed.
        modifications = [("baseline", 0, 0, "Recorded month-end cash; no behavioural change")]
        for amount in (250, 500, 1000):
            modifications.append((f"monthly_net_plus_{amount}", amount, 0,
                                  "Assumed extra net income OR expense reduction from month 1; no implementation cost"))
        for pct in (.1, .2):
            modifications.append((f"rent_reduction_{int(pct*100)}pct", rent*pct, 0,
                                  "Assumed recurring rent saving; no move cost, deposit, eligibility or income change"))
        modifications.append(("liquidate_all_opening_investments", 0, investments[pid],
                              "Convert opening investment snapshot to cash at month 0; zero fees/returns; wealth unchanged"))
        for name, monthly, lump, note in modifications:
            altered = [c + lump + monthly*(i+1) for i, c in enumerate(closing)]
            replays.append(dict(profile_id=pid, intervention=name, applicability=True,
                                baseline_shortfall=shortfall, replay_shortfall=min(altered) < -1e-8,
                                replay_funding_gap_chf=round(max(0, -min(altered)), 2),
                                final_cash_delta_chf=round(altered[-1] - closing[-1], 2), assumptions=note))
        if loans[pid]:
            # The ledger explicitly excludes amortization. Add a 60-month schedule
            # with nominal APR/12, no origination fees, applied after month's flows.
            payment = 0
            for loan in loans[pid]:
                d, i = float(loan["opening_principal_chf"]), float(loan["annual_interest_rate"])/12
                payment += d*i/(1-(1+i)**-60) if i else d/60
            f["assumed_60m_debt_payment_chf"] = round(payment, 2)
            altered = [c-payment*(i+1) for i, c in enumerate(closing)]
            replays.append(dict(profile_id=pid, intervention="add_assumed_60m_debt_payments", applicability=True,
                                baseline_shortfall=shortfall, replay_shortfall=min(altered) < -1e-8,
                                replay_funding_gap_chf=round(max(0, -min(altered)), 2),
                                final_cash_delta_chf=round(altered[-1]-closing[-1], 2),
                                assumptions="Loan profiles only; hypothetical 60-month repayment, nominal APR/12; recorded cash excludes repayment"))
        else:
            f["assumed_60m_debt_payment_chf"] = 0

    cohorts = []
    for status in sorted({f["employment_status"] for f in features}):
        fs = [f for f in features if f["employment_status"] == status]
        n = len(fs)
        cohorts.append(dict(employment_status=status, profiles=n,
                            negative_mean_cash_flow=sum(f["negative_mean_net"] for f in fs),
                            cash_shortfall_profiles=sum(f["observed_cash_shortfall"] for f in fs),
                            shortfall_pct=round(100*sum(f["observed_cash_shortfall"] for f in fs)/n, 1),
                            median_monthly_net_chf=round(median(f["mean_monthly_net_chf"] for f in fs), 2),
                            median_rent_to_reference_income=round(median(f["rent_to_reference_income"] for f in fs), 3),
                            debt_profiles=sum(f["opening_debt_chf"]>0 for f in fs)))
    sensitivity = []
    for name in dict.fromkeys(r["intervention"] for r in replays):
        rs = [r for r in replays if r["intervention"] == name]
        sensitivity.append(dict(intervention=name, applicable_profiles=len(rs),
                                baseline_shortfall_profiles=sum(r["baseline_shortfall"] for r in rs),
                                replay_shortfall_profiles=sum(r["replay_shortfall"] for r in rs),
                                shortfalls_prevented=sum(r["baseline_shortfall"] and not r["replay_shortfall"] for r in rs),
                                new_shortfalls=sum(not r["baseline_shortfall"] and r["replay_shortfall"] for r in rs),
                                median_final_cash_delta_chf=round(median(r["final_cash_delta_chf"] for r in rs), 2),
                                total_funding_gap_chf=round(sum(r["replay_funding_gap_chf"] for r in rs), 2),
                                assumptions=rs[0]["assumptions"]))

    # Analyst classification, not learned labels. Scenarios frequently combine
    # decisions, exogenous outcomes, parameters and objectives.
    roles = {}
    for numbers, role in [
        ([3,10,13,14,28,35,37,39,40,48,50,57,58,65,75,87,89,90,96,100,101,102,103,104,105,106,107,108,109,110,113,114,115], "exogenous_outcome_or_shock"),
        ([4,29,30,46,56,66,98], "scenario_parameter"),
        ([51,54,60], "goal_or_preference"), ([79], "constraint"),
        ([59,78,94], "derived_state_or_constraint_event")]:
        for number in numbers:
            roles[number] = role
    # One primary implementation family per controllable catalogue entry.
    families = {}
    for family, numbers in {
        "education_and_career": [1,2,5,6,7,8,11,12,15,16,17,18,19,20,44,83,86,112],
        "budget_housing_and_household": [31,32,41,43,45,49,53,81,82,84,88],
        "relocation": [21,22,23,24,25,26,27,95],
        "major_purchase": [33,34,36,38,42,47,85],
        "liquidity_and_investment_allocation": [52,55,61,62,63,64,67,68,69,70,97],
        "debt_and_financing": [9,71,72,73,74,76,77,80,119],
        "retirement_and_pension": [91,92,93,99],
        "business": [111,116,117,118,120],
    }.items():
        for number in numbers:
            families[number] = family
    catalogue = []
    for r in data["scenario-catalog"]:
        number = int(r["id"][1:])
        role = roles.get(number, "controllable_action")
        assert number in roles or number in families, f"Unmapped scenario {number}"
        catalogue.append(dict(scenario_id=r["id"], category=r["category"], scenario=r["scenario"],
                              proposed_mdp_role=role, primary_action_family=families.get(number, ""),
                              classification_basis="Manual modelling judgment; outcome of an attempted action may be stochastic"))

    short = [f for f in features if f["observed_cash_shortfall"]]
    summary = dict(
        scope="All 1,200 bundled synthetic profiles; descriptive retrospective analysis, not population inference",
        history_start="2025-01", history_end="2026-12", snapshot_date="2025-01-01",
        source_rows={k:len(v) for k,v in data.items()},
        validations={"cash_identities_pass": True, "monthly_continuity_pass": True,
                     "unique_profile_ids_pass": True, "profile_foreign_keys_pass": True,
                     "24_consecutive_months_each_pass": True, "negative_balance_flags_pass": True},
        metrics={"profiles":len(features), "cash_shortfall_profiles":len(short),
                 "negative_mean_net_profiles":sum(f["negative_mean_net"] for f in features),
                 "any_negative_net_month_profiles":sum(f["negative_net_months"]>0 for f in features),
                 "rent_above_40pct_reference_income_profiles":sum(float(p["reference_monthly_rent_chf"])/float(p["reference_monthly_net_income_chf"])>.4 for p in profiles),
                 "shortfall_profiles_with_investments":sum(f["opening_investments_chf"]>0 for f in short),
                 "shortfall_profiles_whose_opening_investments_cover_recorded_gap":sum(f["opening_investments_chf"]>=f["max_recorded_funding_gap_chf"] for f in short),
                 "debt_profiles":sum(f["opening_debt_chf"]>0 for f in features),
                 "high_priority_goals_matching_max_10000_1point6_opening_cash":goal_formula_matches,
                 "mean_history_net_below_60m_goal_saving_rate_profiles":sum(f["history_mean_below_goal_saving_rate"] for f in features),
                 "multiple_goal_profiles":sum(len(gs)>1 for gs in goals.values())},
        distributions={"opening_cash_chf":quantiles([f["opening_cash_chf"] for f in features]),
                       "monthly_net_chf_per_profile":quantiles([f["mean_monthly_net_chf"] for f in features]),
                       "funding_gap_chf_among_shortfall_profiles":quantiles([f["max_recorded_funding_gap_chf"] for f in short]),
                       "constant_monthly_uplift_needed_among_shortfall_profiles":quantiles([f["constant_monthly_uplift_to_avoid_shortfall_chf"] for f in short]),
                       "loan_nominal_apr":quantiles([float(r["annual_interest_rate"]) for r in data["synthetic-liabilities"]], digits=4),
                       "assumed_60m_payment_chf_among_borrowers":quantiles([f["assumed_60m_debt_payment_chf"] for f in features if f["opening_debt_chf"]>0])},
        goal_counts=dict(Counter(r["goal_name"] for r in data["synthetic-goals"])),
        catalogue_role_counts=dict(Counter(r["proposed_mdp_role"] for r in catalogue)),
        limitations=[
            "No recorded actions, action propensities, employment transitions, market returns or causal labels; cannot estimate P(s_next|s, action).",
            "Reference income is not actual monthly income; net transactions do not identify gross income or expense categories.",
            "Each replay holds recorded flows fixed and assumes the intervention is available immediately; it is not a deployable policy.",
            "Entire 24-month record used descriptively; these results are not out-of-sample validation. Split time and profiles before fitting or evaluating a policy.",
            "Investment liquidation uses opening snapshots with zero return/fees and assumed immediate accessibility. It increases cash, not net worth.",
            "Loans have zero scheduled payments in source; 60-month amortization and nominal APR/12 are explicit analyst assumptions.",
            "All high-priority goal amounts are mechanically generated from opening cash; goal-derived feature importance would be misleading.",
            "A cash balance goal is not necessarily a purchase. Multiple goals may overlap; do not double count cash as independently allocated to each.",
            "All samples are synthetic. Counts characterize this file, not the Swiss population. Month-end cash omits intramonth funding needs.",
        ])
    write_csv("profile_features.csv", features)
    write_csv("employment_cohorts.csv", cohorts)
    write_csv("sensitivity_summary.csv", sensitivity)
    write_csv("sensitivity_by_profile.csv", replays)
    write_csv("scenario_mdp_mapping.csv", catalogue)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({"metrics":summary["metrics"], "cohorts":cohorts, "sensitivity":sensitivity,
                      "catalogue_roles":summary["catalogue_role_counts"]}, indent=2))


if __name__ == "__main__":
    main()
