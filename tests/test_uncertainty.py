import numpy as np
import pytest
from twin.explain_years import mean_interval, probability_interval, paired_probability_interval
from twin.explanation_methods import annual_predict, local_year
from test_explanation_methods import Model, context


def test_monte_carlo_intervals_cover_known_values_and_boundary():
    assert mean_interval([4,4,4])['lower'] == 4
    assert mean_interval([-1,0,1])['lower'] < 0 < mean_interval([-1,0,1])['upper']
    zero=probability_interval([False]*100)
    assert zero['lower']==pytest.approx(0, abs=1e-15) and 0<zero['upper']<.05
    full=probability_interval([True]*100)
    assert full['upper']==pytest.approx(1) and full['lower']>.95
    with pytest.raises(ValueError): mean_interval([1])


def test_lime_repeats_and_member_ranges_are_reproducible():
    c=context(); ref={'income':np.full(12,1500.),'rent':np.full(12,1000.)}
    r=local_year(Model(),c,ref,seed=123,samples=300)
    assert len(r['lime']['repeat_fits'])==5
    assert r['lime']['income_weight_range_chf'][0] <= r['lime']['weights_chf_per_standard_deviation']['income_multiplier'] <= r['lime']['income_weight_range_chf'][1]
    assert r['shapley']['member_income_range_chf']==pytest.approx([r['shapley']['income_chf']]*2)
    assert r==local_year(Model(),c,ref,seed=123,samples=300)


def test_paired_binary_interval_retains_uncertainty_when_no_discordance():
    interval=paired_probability_interval([False]*100,[False]*100)
    assert interval['lower'] < 0 < interval['upper']
    changed=paired_probability_interval([False]*100,[True]*100)
    assert changed['estimate']==1 and changed['lower']<1 and changed['upper']==1
