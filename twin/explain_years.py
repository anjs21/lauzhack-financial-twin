"""Explain complete world rollouts with signed annual accounting and paired plans."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import html
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from . import config as C
from .explore import default_what_ifs, person_table, simulate
from .policy.rules import RulePolicy
from .policy.traits import from_profile
from .world.env import CASH_COMPONENTS, World

LABELS = {
    'learned_flow': 'Learned operating net flow', 'personal_offset': 'Personal residual offset',
    'lifestyle_adjustment': 'Lifestyle adjustment', 'childcare': 'Additional childcare',
    'tuition': 'Tuition adjustment above learned student fees', 'events': 'Net one-off events',
    'moving': 'Moving costs', 'loan_payment': 'Loan payments (principal + interest)',
    'credit_interest': 'Credit interest', 'investment_transfer': 'Net transfers from investments to cash',
    'investment_gain': 'Investment gains/losses', 'loan_interest_expense': 'Loan interest expense',
}
WEALTH_COMPONENTS = tuple(k for k in CASH_COMPONENTS if k not in ('investment_transfer', 'loan_payment')) + ('investment_gain', 'loan_interest_expense')


def mean_interval(values):
    """Monte Carlo t interval for an average over independent simulated paths."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError('At least two simulated paths are required')
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    margin = float(student_t.ppf(.975, len(x)-1) * se)
    return dict(estimate=mean, standard_error=se, lower=mean-margin, upper=mean+margin, n=len(x))


def probability_interval(values):
    """95% Wilson interval for Monte Carlo frequency; no real-world calibration."""
    x = np.asarray(values, dtype=bool)
    n = len(x)
    if not n:
        raise ValueError('At least one path is required')
    phat = float(x.mean()); z = 1.959963984540054
    denom=1+z*z/n
    centre=(phat+z*z/(2*n))/denom
    margin=z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/denom
    return dict(estimate=phat, lower=float(max(0,centre-margin)), upper=float(min(1,centre+margin)), n=n)


def paired_probability_interval(base, alternative):
    """Conservative paired difference CI using Bonferroni Wilson discordance bounds."""
    from scipy.stats import norm
    b=np.asarray(base, dtype=bool); a=np.asarray(alternative, dtype=bool)
    if b.shape != a.shape or b.ndim != 1 or len(b) == 0:
        raise ValueError('Paired outcomes must have matching nonempty shapes')
    def wilson(x):
        n=len(x); p=float(x.mean()); z=float(norm.ppf(.9875)); denom=1+z*z/n
        center=(p+z*z/(2*n))/denom
        margin=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
        return max(0,center-margin), min(1,center+margin)
    plus=a & ~b; minus=b & ~a
    lp,up=wilson(plus); lm,um=wilson(minus)
    return dict(estimate=float(plus.mean()-minus.mean()), lower=float(max(-1,lp-um)),
                upper=float(min(1,up-lm)), n=len(b), method='Bonferroni Wilson bounds on paired discordance rates')


def quantiles(values):
    return dict(zip(('p05', 'p50', 'p95'), map(float, np.percentile(values, [5, 50, 95]))))


def summarize_trajectory(paths):
    if paths.accounting is None:
        raise ValueError('Simulate with record_accounting=True')
    a = paths.accounting
    cash_delta = np.diff(paths.cash, axis=1)
    nw_delta = np.diff(paths.nw, axis=1)
    cash_error = cash_delta - sum(a[k] for k in CASH_COMPONENTS)
    nw_error = nw_delta - sum(a[k] for k in WEALTH_COMPONENTS)
    if not np.allclose(cash_error, 0, atol=1e-6) or not np.allclose(nw_error, 0, atol=1e-6):
        raise ValueError('Monthly accounting failed to reconcile')
    horizon = cash_delta.shape[1]
    rows = []
    for start in range(0, horizon, 12):
        end = min(start+12, horizon)
        cash_components = {k: float(a[k][:, start:end].sum(1).mean()) for k in CASH_COMPONENTS}
        nw_components = {k: float(a[k][:, start:end].sum(1).mean()) for k in WEALTH_COMPONENTS}
        opening, closing = float(paths.cash[:, start].mean()), float(paths.cash[:, end].mean())
        rows.append(dict(
            year=C.START_YEAR+start//12, months=end-start,
            mean_opening_cash=opening, mean_closing_cash=closing,
            mean_cash_change=closing-opening, cash_components=cash_components,
            cash_reconciliation_error=closing-opening-sum(cash_components.values()),
            mean_opening_net_worth=float(paths.nw[:, start].mean()),
            mean_closing_net_worth=float(paths.nw[:, end].mean()),
            mean_net_worth_change=float((paths.nw[:, end]-paths.nw[:, start]).mean()),
            net_worth_components=nw_components,
            closing_cash_quantiles=quantiles(paths.cash[:, end]),
            closing_net_worth_quantiles=quantiles(paths.nw[:, end]),
            annual_operating_flow_quantiles=quantiles((a['learned_flow'][:, start:end]+a['personal_offset'][:, start:end]).sum(1)),
            shortfall_probability=float(paths.below_zero[:, start:end].any(1).mean()),
            shortfall_mc_interval=probability_interval(paths.below_zero[:, start:end].any(1)),
            mean_closing_cash_mc_interval=mean_interval(paths.cash[:, end]),
        ))
    below = paths.below_zero
    hit = below.any(1)
    first = below.argmax(1)+1
    histogram = {C.month_label(t): int(((first==t+1) & hit).sum()) for t in range(horizon)}
    histogram = {k: v for k, v in histogram.items() if v}
    p = paths.people.iloc[0]
    high = None
    if horizon > C.T_HIGH_GOAL:
        available = paths.cash[:, C.T_HIGH_GOAL+1] if p.high_goal == 'Emergency savings' else paths.nw[:, C.T_HIGH_GOAL+1]
        high = float((available >= p.high_target).mean())
    med = None
    if horizon > C.T_MED_GOAL and pd.notna(p.med_target):
        med = float((paths.nw[:, C.T_MED_GOAL+1] >= p.high_target+p.med_target).mean())
    minimum = paths.cash.min(1)
    return dict(
        years=rows, horizon_months=horizon,
        mean_final_net_worth=float(paths.nw[:, -1].mean()), final_net_worth_quantiles=quantiles(paths.nw[:, -1]),
        minimum_cash_quantiles=quantiles(minimum),
        maximum_negative_cash_depth_quantiles=quantiles(np.maximum(-minimum, 0)),
        shortfall_probability=float(hit.mean()),
        shortfall_mc_interval=probability_interval(hit),
        mean_final_net_worth_mc_interval=mean_interval(paths.nw[:, -1]),
        first_shortfall_month_counts=histogram, no_shortfall_paths=int((~hit).sum()),
        conditional_first_shortfall_month=quantiles(first[hit]) if hit.any() else None,
        high_goal_probability=high, medium_goal_probability=med,
        high_goal_mc_interval=probability_interval(available >= p.high_target) if high is not None else None,
        medium_goal_mc_interval=probability_interval(paths.nw[:, C.T_MED_GOAL+1] >= p.high_target+p.med_target) if med is not None else None,
        reconciliation=dict(max_monthly_cash_error=float(np.abs(cash_error).max()),
                            max_monthly_net_worth_error=float(np.abs(nw_error).max())),
        cash_fan=np.percentile(paths.cash, [5, 50, 95], axis=0).tolist(),
        net_worth_fan=np.percentile(paths.nw, [5, 50, 95], axis=0).tolist(),
    )


def compare_trajectories(base, alternative, base_summary=None):
    b = base_summary or summarize_trajectory(base)
    a = summarize_trajectory(alternative)
    annual = []
    for i, (br, ar) in enumerate(zip(b['years'], a['years'])):
        end = min((i+1)*12, base.below_zero.shape[1])
        delta = alternative.nw[:, end]-base.nw[:, end]
        annual.append(dict(year=br['year'], mean_net_worth_difference=float(delta.mean()),
                           mean_net_worth_difference_mc_interval=mean_interval(delta),
                           paired_net_worth_difference_quantiles=quantiles(delta),
                           cumulative_mean_cash_difference=float((alternative.cash[:, end]-base.cash[:, end]).mean()),
                           annual_cash_change_difference=ar['mean_cash_change']-br['mean_cash_change'],
                           cash_component_differences={k: ar['cash_components'][k]-br['cash_components'][k] for k in CASH_COMPONENTS},
                           net_worth_component_differences={k: ar['net_worth_components'][k]-br['net_worth_components'][k] for k in WEALTH_COMPONENTS}))
    monthly_mean_delta=(alternative.nw-base.nw).mean(0)
    negative=np.flatnonzero(monthly_mean_delta[1:] < -1e-6)+1
    catch_up=None
    if len(negative):
        last_negative=int(negative[-1])
        if last_negative < len(monthly_mean_delta)-1:
            catch_up=C.month_label(last_negative)  # balance index k corresponds to month k-1
    return dict(summary=a, years=annual,
                mean_final_net_worth_difference=a['mean_final_net_worth']-b['mean_final_net_worth'],
                shortfall_change_percentage_points=100*(a['shortfall_probability']-b['shortfall_probability']),
                paired_final_wealth_difference_mc_interval=mean_interval(alternative.nw[:, -1]-base.nw[:, -1]),
                paired_shortfall_difference_mc_interval=paired_probability_interval(base.below_zero.any(1), alternative.below_zero.any(1)),
                sustained_mean_net_worth_catch_up=catch_up,
                catch_up_definition='After falling behind, first subsequent month end with mean net worth no lower through the end of the simulated horizon. Null if no prior deficit or no sustained recovery.',
                interpretation='Paired model rollouts with common random streams. Persona decisions adapt after the plan; differences include subsequent responses. Accounting contributions are not empirical causal effects.')


def explain_local_path(paths):
    """Explain one actual trajectory selected by terminal net-worth median proximity."""
    if paths.accounting is None or len(paths.people) != 1:
        raise ValueError('A single profile with recorded accounting is required')
    terminal = paths.nw[:, -1]
    index = int(np.argmin(np.abs(terminal-np.median(terminal))))
    years = []
    for start in range(0, paths.action.shape[1], 12):
        end = min(start+12, paths.action.shape[1])
        components = {k: float(paths.accounting[k][index, start:end].sum()) for k in CASH_COMPONENTS}
        wealth = {k: float(paths.accounting[k][index, start:end].sum()) for k in WEALTH_COMPONENTS}
        actions = []
        for t in range(start, end):
            raw = paths.action[index, t]
            changes = []
            if raw[0] != 0:
                changes.append(C.STATUS_OPTIONS[int(raw[0])])
            for i, name in [(1, 'housing'), (2, 'spending'), (3, 'assets')]:
                if raw[i] != 0:
                    changes.append(f'{name} {int(raw[i]):+d}')
            events = [name.replace('_', ' ') for name, values in paths.events.items() if values[index, t]]
            if changes or events:
                actions.append(dict(month=C.month_label(t), actions=changes, events=events))
        opening = float(paths.cash[index, start]); closing = float(paths.cash[index, end])
        years.append(dict(year=C.START_YEAR+start//12, opening_cash=opening, closing_cash=closing,
            cash_change=closing-opening, cash_components=components,
            cash_reconciliation_error=closing-opening-sum(components.values()),
            opening_net_worth=float(paths.nw[index, start]), closing_net_worth=float(paths.nw[index, end]),
            net_worth_components=wealth, actions_and_events=actions,
            first_negative_month=next((C.month_label(t) for t in range(start,end) if paths.below_zero[index,t]), None)))
    return dict(path_index=index, selection='One actual sampled path whose terminal net worth is closest to the sample median; it need not be median in earlier years.',
                years=years, limitations=['Accounting is exact for this simulated path, but does not explain the neural network’s learned feature importance.',
                'Actions shown are the world-accepted action codes. Existing event logging is selective; event cash amounts are included in the accounting.',
                'The selected path is illustrative and changes with seed and sample size; it is not the most likely or representative history at every year.'])


def explain_years(profile_id, n_paths=1000, years=10, seed=C.SEED, persona=None,
                  world=None, plans=None, with_sensitivity=False, policy_cls=RulePolicy):
    for value, label, low, high in [(n_paths, 'paths', 2, 100000), (years, 'years', 1, 10)]:
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f'{label} must be an integer between {low} and {high}')
    world=world or World.load()
    people=person_table(profile_id)
    persona=persona or from_profile(people.iloc[0])
    horizon=years*12
    def run(w, per, ps=people):
        return simulate(w, ps, per, n_paths, seed, horizon=horizon, policy_cls=policy_cls, record_accounting=True)
    base=run(world, persona)
    b=summarize_trajectory(base)
    local = explain_local_path(base)
    from .global_explain import global_importance
    global_explanation = global_importance(world.flow, seed=seed)
    from .explanation_methods import compute_methods
    methods = compute_methods(world.flow, base, local["path_index"], seed)
    alternatives=plans if plans is not None else default_what_ifs(people.iloc[0], persona)
    comparisons={}
    for name, alt in alternatives.items():
        comparisons[name]=compare_trajectories(base, run(world, alt), b)
        comparisons[name]['intentions']=[vars(i) if not isinstance(i, dict) else i for i in alt.intentions]
    sensitivity=[]
    if with_sensitivity:
        for name, key, value in [
            ('No assumed degree pay-rank premium', ('masters', 'degree_rank_shift'), 0.0),
            ('Job-offer probabilities halved', ('jobs', 'offer_per_month'), {k: v/2 for k,v in world.pr['jobs']['offer_per_month'].items()}),
            ('Credit rate +4 percentage points', ('credit', 'rate_per_year'), world.pr['credit']['rate_per_year']+.04),
        ]:
            priors=deepcopy(world.pr)
            priors[key[0]][key[1]]=value
            variant=World(world.flow, world.income_model, world.rent_model, priors, world.cfg)
            variant_paths=run(variant, persona)
            sm=summarize_trajectory(variant_paths)
            sensitivity.append(dict(name=name, parameter='.'.join(key), baseline_value=world.pr[key[0]][key[1]], alternative_value=value,
                                    mean_final_net_worth_difference=sm['mean_final_net_worth']-b['mean_final_net_worth'],
                                    paired_wealth_mc_interval=mean_interval(variant_paths.nw[:,-1]-base.nw[:,-1]),
                                    shortfall_change_percentage_points=100*(sm['shortfall_probability']-b['shortfall_probability'])))
    return dict(profile_id=profile_id, name=str(people.iloc[0]['name']), n_paths=n_paths, seed=seed,
                persona=dict(traits=persona.traits, temperature=persona.temperature, spread=persona.spread, intentions=[vars(i) if not isinstance(i, dict) else i for i in persona.intentions]),
                policy=policy_cls.name, horizon_years=years, medium_target=float(people.iloc[0].med_target) if pd.notna(people.iloc[0].med_target) else None, baseline=b, comparisons=comparisons,
                sensitivity=sensitivity, local_explanation=local, global_explanation=global_explanation, explanation_methods=methods, priors=deepcopy(world.pr), component_labels=LABELS,
                provenance=dict(flow='Saved MDN ensemble, sampled monthly net flow',
                                personal_offset='Saved per-person residual fit; may use the complete supplied 2025–2026 history',
                                events='Explicit world priors, not identified intervention effects',
                                starting_state='January 2025 synthetic snapshot'),
                limitations=[
                    'This explains the existing simulator, not a real-world forecast or an out-of-time backtest. Saved personal offsets can include information after January 2025.',
                    'The model learns total operating net flow; it does not separately observe salary, food, transport, or every expense category.',
                    'Tuition is the simulator adjustment above typical student fees already represented by learned flow; it is not the full tuition bill.',
                    'Means are used for additive accounting. Quantiles come from complete sampled trajectories; marginal medians need not reconcile or describe one actual path.',
                    'Common random numbers align random inputs; action effects and cross-state transport remain assumptions. Downstream actions and event outcomes may diverge.',
                    'Negative cash follows existing simulator credit-interest rules. Negative-balance depth is not a computed initial funding top-up: extra funding can change later actions and costs.',
                    'Within-month balances and actual credit approval are not modelled. Funding dates and depths are monthly path diagnostics.',
                    'Goal checks use existing simulator conventions: emergency goals use cash; other high goals use net worth, medium goals require the combined high-plus-medium target. Unreached deadlines are null, not failure.',
                    'Monte Carlo intervals quantify finite simulation sampling error conditional on this model and assumed independent paths. They do not cover misspecification, causal assumptions, or future real-world outcomes.',
                    'Sensitivity varies one assumption at a time with the same responsive persona. It is not optimization or a causal effect estimate.',
                ])


def summary_text(result):
    b=result['baseline']
    lines=[f"Multi-year explanation — {result['name']} ({result['profile_id']})",
           f"{result['horizon_years']} years, {result['n_paths']} paths, seed {result['seed']}, {result['policy']}",
           f"Mean terminal net worth CHF {b['mean_final_net_worth']:,.0f}; ever negative cash {b['shortfall_probability']:.1%}",
           'Annual means (cash components reconcile):']
    for row in b['years']:
        lines.append(f"  {row['year']}: opening CHF {row['mean_opening_cash']:,.0f}, change CHF {row['mean_cash_change']:+,.0f}, closing CHF {row['mean_closing_cash']:,.0f}")
    for name, alt in result['comparisons'].items():
        lines.append(f"  {name}: mean terminal wealth difference CHF {alt['mean_final_net_worth_difference']:+,.0f}; negative-cash change {alt['shortfall_change_percentage_points']:+.1f} pp")
    lines.append(f"Maximum monthly cash reconciliation error: {b['reconciliation']['max_monthly_cash_error']:.3g} CHF")
    return '\n'.join(lines)


def render(result):
    """Render the focused interactive explanation dashboard."""
    from .explanation_report import render as render_dashboard
    return render_dashboard(result)
