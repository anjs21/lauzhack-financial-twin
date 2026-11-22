"""Prediction explanation contracts; these tests do not retrain the world model."""
from itertools import combinations

import numpy as np
import pytest
from scipy.special import ndtr

from twin.explain import GROUPS, distribution, exact_shapley, explain_profile, render


class SimpleModel:
    members = [0, 1]

    def params(self, X, m):
        # Known additive Gaussian kernels in CHF thousands.
        mean = 2*X[:, 6] - X[:, 7] + m
        return np.ones((len(X), 1)), mean[:, None], np.full((len(X), 1), .5)


def test_exact_shapley_splits_interaction_and_reconciles():
    values = {}
    for k in range(4):
        for subset in combinations(GROUPS, k):
            subset = frozenset(subset)
            value = 10 + 3*('income' in subset) - 2*('rent' in subset)
            value += 8*({'income', 'rent'} <= subset)
            values[subset] = np.array([value])
    attr = exact_shapley(values)
    assert attr['income'][0] == pytest.approx(7)
    assert attr['rent'][0] == pytest.approx(2)
    assert attr['calendar_month'][0] == 0
    assert sum(v[0] for v in attr.values()) == pytest.approx(values[frozenset(GROUPS)][0]-10)


def test_gaussian_mixture_moments_probability_and_units():
    context = dict(mode=0, self_employed=False, pensioner=False, income=0,
                   rent=1000, city_idx=0, calendar_month=1)
    pred = distribution(SimpleModel(), context, opening_cash=0)
    assert pred['mean_flow_chf'] == pytest.approx(500)
    assert pred['negative_flow_probability'] == pytest.approx((.5 + ndtr(-2))/2)
    assert pred['reserve_breach_probability'] == pred['negative_flow_probability']
    assert pred['within_member_variance_chf2'] == pytest.approx(250000)
    assert pred['between_member_mean_variance_chf2'] == pytest.approx(250000)
    assert pred['flow_p05_chf'] < pred['flow_p50_chf'] < pred['flow_p95_chf']
    assert pred['flow_p50_chf'] == pytest.approx(500)


def test_profile_reference_exclusion_attribution_and_rent_response():
    result = explain_profile(SimpleModel(), 'P00001', month=12, alternate_rent=700)
    assert result['reference']['excluded_profile'] == 'P00001'
    assert all(abs(v)<1e-6 for v in result['reconciliation_error'].values())
    assert result['alternative']['mean_flow_change_chf'] > 0
    # Mock kernel ignores seasonal features, so attribution must also ignore them.
    assert result['attributions'][2]['mean_flow_chf'] == pytest.approx(0)
    assert result['prediction']['expected_closing_cash_chf'] == pytest.approx(8000+result['prediction']['mean_flow_chf'])
    assert 'not a full world transition' in result['scope']


def test_invalid_inputs_and_escaped_report():
    for kw in ({'month': 0}, {'month': 13}, {'reserve': -1}, {'alternate_rent': float('nan')}):
        with pytest.raises(ValueError):
            explain_profile(SimpleModel(), 'P00001', **kw)
    with pytest.raises(ValueError):
        explain_profile(SimpleModel(), 'unknown')
    result = explain_profile(SimpleModel(), 'P00001')
    result['name'] = '<script>bad()</script>'
    page = render(result)
    assert '<script>bad()</script>' not in page
    assert '&lt;script&gt;' in page
