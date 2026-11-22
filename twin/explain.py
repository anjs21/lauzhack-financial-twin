"""Local explanations of the saved flow kernel, not empirical causal effects.

Exact grouped Shapley values over income, rent and season (8 model evaluations).
Work mode, flags, city and opening cash remain fixed. No retraining is required.
"""
from __future__ import annotations

import html
from itertools import combinations
from math import factorial

import numpy as np
from scipy.special import ndtr
from scipy.optimize import brentq

from . import config as C
from .data import profiles
from .world.flow import SCALE, features

GROUPS = ('income', 'rent', 'calendar_month')


def exact_shapley(values, groups=GROUPS):
    """All coalitions must be present. Interactions are shared across their inputs."""
    n = len(groups)
    out = {}
    for group in groups:
        others = [g for g in groups if g != group]
        result = np.zeros_like(np.asarray(values[frozenset()], dtype=float))
        for k in range(n):
            weight = factorial(k) * factorial(n-k-1) / factorial(n)
            for subset in combinations(others, k):
                coalition = frozenset(subset)
                result += weight * (np.asarray(values[coalition | {group}]) - values[coalition])
        out[group] = result
    return out


def distribution(model, context, opening_cash, reserve=0):
    X = features([context['mode']], [context['self_employed']], [context['pensioner']],
                 [context['income']], [context['rent']], [context['city_idx']], [context['calendar_month']])
    weights, means, sigmas = [], [], []
    member_means, within_variances = [], []
    for i in range(len(model.members)):
        w, mu, sigma = model.params(X, i)
        w, mu, sigma = w[0], mu[0] * SCALE, sigma[0] * SCALE
        m = float(w @ mu)
        member_means.append(m)
        within_variances.append(float(w @ (sigma**2 + mu**2) - m**2))
        weights.extend(w / len(model.members))
        means.extend(mu)
        sigmas.extend(sigma)
    w, mu, sigma = map(np.asarray, (weights, means, sigmas))
    # Renormalization removes float32 softmax accumulation error.
    w = w / w.sum()
    cdf = lambda x: float(w @ ndtr((x-mu)/sigma))
    lower, upper = float(np.min(mu-12*sigma)), float(np.max(mu+12*sigma))
    quantile = lambda q: float(brentq(lambda x: cdf(x)-q, lower, upper))
    within = max(0, float(np.mean(within_variances)))
    between = float(np.var(member_means))
    return {
        'mean_flow_chf': float(w @ mu),
        'flow_p05_chf': quantile(.05), 'flow_p50_chf': quantile(.5), 'flow_p95_chf': quantile(.95),
        'negative_flow_probability': cdf(0),
        'reserve_breach_probability': cdf(reserve-opening_cash),
        'expected_closing_cash_chf': float(opening_cash + w @ mu),
        'within_member_variance_chf2': within,
        'between_member_mean_variance_chf2': between,
        'member_mean_flow_chf': member_means,
    }


def explain_profile(model, profile_id, month=1, reserve=0, alternate_rent=None):
    if isinstance(month, bool) or not isinstance(month, int) or not 1 <= month <= 12:
        raise ValueError('month must be an integer from 1 to 12')
    if not np.isfinite(reserve) or reserve < 0:
        raise ValueError('reserve must be finite and nonnegative')
    if alternate_rent is not None and (not np.isfinite(alternate_rent) or alternate_rent < 0):
        raise ValueError('alternate rent must be finite and nonnegative')
    table = profiles()
    if profile_id not in table.index:
        raise ValueError(f'Unknown profile: {profile_id}')
    p = table.loc[profile_id]
    # Exclude the explained person from the reference cohort.
    peers = table[(table.status == p.status) & (table.index != profile_id)]
    local = peers[peers.city == p.city]
    cohort = local if len(local) >= 10 else peers
    context = dict(mode=int(p['mode']), self_employed=bool(p.self_employed), pensioner=bool(p.pensioner),
                   city_idx=int(p.city_idx), income=float(p.income0), rent=float(p.rent0), calendar_month=month)
    reference = dict(context, income=float(cohort.income0.median()), rent=float(cohort.rent0.median()), calendar_month=1)
    metrics = ('mean_flow_chf', 'negative_flow_probability', 'reserve_breach_probability')
    coalitions, predictions = {}, {}
    for k in range(len(GROUPS)+1):
        for subset in combinations(GROUPS, k):
            ctx = dict(reference, **{g: context[g] for g in subset})
            prediction = distribution(model, ctx, float(p.cash0), reserve)
            predictions[frozenset(subset)] = prediction
            coalitions[frozenset(subset)] = np.array([prediction[m] for m in metrics])
    contributions = exact_shapley(coalitions)
    baseline, target = predictions[frozenset()], predictions[frozenset(GROUPS)]
    rows = [dict(group=g, **{metric: float(value) for metric, value in zip(metrics, contributions[g])}) for g in GROUPS]
    residual = {m: float(target[m] - baseline[m] - sum(r[m] for r in rows)) for m in metrics}
    warnings = []
    if len(cohort) < 30:
        warnings.append(f'Small reference cohort: {len(cohort)} profiles.')
    for label, column in [('income', 'income0'), ('rent', 'rent0')]:
        if not cohort[column].min() <= context[label] <= cohort[column].max():
            warnings.append(f'{label} lies outside the reference cohort range.')
    if context['rent'] < 50:
        warnings.append('The existing flow encoder clips rent to CHF 50; predictions below that threshold are identical.')
    alternative = None
    if alternate_rent is not None:
        changed = dict(context, rent=float(alternate_rent))
        alt = distribution(model, changed, float(p.cash0), reserve)
        alternative = dict(rent_chf=float(alternate_rent), prediction=alt,
                           mean_flow_change_chf=alt['mean_flow_chf']-target['mean_flow_chf'],
                           reserve_breach_change_percentage_points=100*(alt['reserve_breach_probability']-target['reserve_breach_probability']),
                           interpretation='Model response with rent changed and other inputs fixed; not an identified moving effect. Excludes moving costs and deposits.')
        if not cohort.rent0.min() <= alternate_rent <= cohort.rent0.max():
            warnings.append('Alternative rent lies outside the reference cohort range.')
        if alternate_rent < 50:
            warnings.append('Alternative rent is clipped to CHF 50 by the existing encoder.')
    return dict(
        profile_id=profile_id, name=str(p['name']), status=str(p.status), city=str(p.city),
        context=context, opening_cash_chf=float(p.cash0), reserve_chf=float(reserve),
        reference=dict(context=reference, cohort_size=len(cohort), selection='same employment status and city' if len(local)>=10 else 'same employment status; city held fixed', excluded_profile=profile_id),
        prediction=target, reference_prediction=baseline, attributions=rows, reconciliation_error=residual,
        alternative=alternative, warnings=warnings,
        method='Exact baseline-replacement Shapley over income, rent, and calendar month; recomputes mode-month features for each coalition.',
        scope='Population flow kernel in base-price CHF; not a full world transition or a dated historical forecast.',
        exclusions=['Per-person residual offset and offset uncertainty', 'Pay/price drift', 'Tuition and event costs', 'Investments, debt, and credit', 'Persona choices'],
        limitations=[
            'Attributions explain a prediction relative to this explicit reference; they are not causal effects.',
            'Hybrid reference/input combinations may be weakly supported even when individual features are in range.',
            'Intervals describe the ensemble mixture, not a guarantee of calibrated coverage for this profile.',
            'Between-member disagreement is not proof of out-of-distribution detection; agreement is not proof of support.',
            'The reference month is January, with employment flags and city held fixed. Changing this reference changes attributions.',
            'This is a saved final model explanation, not a held-out evaluation. Cohort exclusion does not remove the profile from model training.',
            'Age and household are not direct features of the fitted flow network; their policy or redraw effects are outside this explanation.',
        ])


def text_summary(result):
    p, b = result['prediction'], result['reference_prediction']
    lines = [f"Flow explanation — {result['name']} ({result['profile_id']})",
             f"Expected monthly flow: CHF {p['mean_flow_chf']:,.2f}; 5–95%: CHF {p['flow_p05_chf']:,.2f} to {p['flow_p95_chf']:,.2f}",
             f"Negative flow: {p['negative_flow_probability']:.1%}; closing cash below reserve: {p['reserve_breach_probability']:.1%}",
             f"Reference expected flow: CHF {b['mean_flow_chf']:,.2f} ({result['reference']['cohort_size']} peers; January)",
             'Contributions to expected flow difference:']
    lines += [f"  {r['group']}: CHF {r['mean_flow_chf']:+,.2f}" for r in result['attributions']]
    lines.append(f"  Reconciliation error: CHF {result['reconciliation_error']['mean_flow_chf']:.6f}")
    if result['alternative']:
        a = result['alternative']
        lines.append(f"Rent CHF {a['rent_chf']:,.2f}: predicted flow changes CHF {a['mean_flow_change_chf']:+,.2f}; reserve-breach change {a['reserve_breach_change_percentage_points']:+.2f} percentage points")
    lines += ['Scope: '+result['scope'], 'Personal history offsets and full-world costs are excluded.',
              'Model associations and assumptions; not established causal effects.']
    lines += ['Note: '+w for w in result['warnings']]
    return '\n'.join(lines)


def render(result):
    """Self-contained explanation report, with every reference and exclusion visible."""
    esc = lambda value: html.escape(str(value), quote=True)
    p, b = result['prediction'], result['reference_prediction']
    rows = ''.join(f"<tr><td>{esc(r['group'])}</td><td>{r['mean_flow_chf']:+,.2f}</td><td>{100*r['negative_flow_probability']:+.2f}</td><td>{100*r['reserve_breach_probability']:+.2f}</td></tr>" for r in result['attributions'])
    notes = ''.join(f'<li>{esc(x)}</li>' for x in result['limitations']+result['warnings'])
    excluded = ', '.join(result['exclusions'])
    alt = ''
    if result['alternative']:
        a = result['alternative']
        alt = f"<h2>Rent what-if</h2><p>Rent CHF {a['rent_chf']:,.2f}: predicted flow changes CHF {a['mean_flow_change_chf']:+,.2f}; reserve-breach probability changes {a['reserve_breach_change_percentage_points']:+.2f} percentage points.</p><p>{esc(a['interpretation'])}</p>"
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Flow explanation: {esc(result['name'])}</title>
<style>body{{font:16px/1.6 system-ui;margin:40px auto;padding:0 20px;max-width:960px;color:#172535}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}pre{{white-space:pre-wrap;background:#f3f5f8;padding:16px}}h2{{margin-top:32px}}.scroll{{overflow:auto}}</style>
<h1>Why this predicted cash flow?</h1><p>{esc(result['name'])} · {esc(result['profile_id'])} · {esc(result['status'])} · {esc(result['city'])}</p>
<pre>{esc(text_summary(result))}</pre><h2>Reference and attribution</h2>
<p>{esc(result['method'])} Reference: {esc(result['reference']['selection'])}, excluding this profile. Reference income CHF {result['reference']['context']['income']:,.2f}, rent CHF {result['reference']['context']['rent']:,.2f}, calendar month January. Employment and city stay fixed.</p>
<div class="scroll"><table><tr><th>Input group</th><th>Flow contribution (CHF)</th><th>Negative-flow contribution (pp)</th><th>Reserve-breach contribution (pp)</th></tr>{rows}</table></div>
<p>Reference flow CHF {b['mean_flow_chf']:,.2f} + contributions = predicted flow CHF {p['mean_flow_chf']:,.2f}. Contributions share interactions; they are not standalone causal effects.</p>
<h2>Uncertainty</h2><p>Average within-member variance: {p['within_member_variance_chf2']:,.2f} CHF². Between-member mean variance: {p['between_member_mean_variance_chf2']:,.2f} CHF². These describe the fitted ensemble and do not include all uncertainty in the world model.</p>
{alt}<h2>Scope and limitations</h2><p>Excluded: {esc(excluded)}.</p><ul>{notes}</ul></html>'''
