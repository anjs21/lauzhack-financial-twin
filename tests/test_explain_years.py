"""Annual accounting and paired-rollout explanation contracts."""
from dataclasses import replace

import numpy as np
import pytest

from twin.data import profiles
from twin.explore import simulate
from twin.explain_years import WEALTH_COMPONENTS, compare_trajectories, explain_years, render, summarize_trajectory
from twin.policy.traits import Intention, from_profile
from twin.world.env import CASH_COMPONENTS, World


@pytest.fixture(scope='module')
def world():
    return World.load()


def test_accounting_trace_does_not_change_lives(world):
    person=profiles().loc[['P00013']]
    persona=from_profile(person.iloc[0])
    plain=simulate(world,person,persona,n_paths=24,seed=15,horizon=36)
    traced=simulate(world,person,persona,n_paths=24,seed=15,horizon=36,record_accounting=True)
    for key in ('cash','inv','debt','action'):
        assert np.array_equal(getattr(plain,key),getattr(traced,key))
    assert plain.accounting is None
    assert np.allclose(np.diff(traced.cash,axis=1),sum(traced.accounting[k] for k in CASH_COMPONENTS),atol=1e-6)
    assert np.allclose(np.diff(traced.nw,axis=1),sum(traced.accounting[k] for k in WEALTH_COMPONENTS),atol=1e-6)
    assert traced.accounting['loan_payment'].sum() < 0
    assert traced.accounting['loan_interest_expense'].sum() < 0


def test_annual_means_reconcile_quantiles_and_goal_deadlines(world):
    person=profiles().loc[['P00001']]
    paths=simulate(world,person,from_profile(person.iloc[0]),n_paths=30,seed=3,horizon=24,record_accounting=True)
    result=summarize_trajectory(paths)
    assert result['high_goal_probability'] is None
    assert result['medium_goal_probability'] is None
    for row in result['years']:
        assert sum(row['cash_components'].values()) == pytest.approx(row['mean_cash_change'],abs=1e-6)
        assert sum(row['net_worth_components'].values()) == pytest.approx(row['mean_net_worth_change'],abs=1e-6)
    assert sum(result['first_shortfall_month_counts'].values())+result['no_shortfall_paths']==30
    expected=np.percentile((paths.accounting['learned_flow'][:,:12]+paths.accounting['personal_offset'][:,:12]).sum(1),50)
    assert result['years'][0]['annual_operating_flow_quantiles']['p50']==pytest.approx(expected)


def test_plan_isolated_until_start_and_differences_reconcile(world):
    person=profiles().loc[['P00003']]
    persona=from_profile(person.iloc[0])
    alternative=replace(persona,intentions=[Intention('status',2,1,when='2026-01')])
    base=simulate(world,person,persona,n_paths=40,seed=42,horizon=36,record_accounting=True)
    alt=simulate(world,person,alternative,n_paths=40,seed=42,horizon=36,record_accounting=True)
    assert np.array_equal(base.cash[:,:13],alt.cash[:,:13])
    comparison=compare_trajectories(base,alt)
    assert comparison['years'][0]['mean_net_worth_difference']==0
    for row in comparison['years']:
        assert sum(row['cash_component_differences'].values())==pytest.approx(row['annual_cash_change_difference'],abs=1e-6)
    total=sum(sum(row['net_worth_component_differences'].values()) for row in comparison['years'])
    assert total==pytest.approx(comparison['mean_final_net_worth_difference'],abs=1e-6)


def test_report_has_scope_and_safe_payload(world):
    result=explain_years('P00001',n_paths=12,years=1,seed=6,world=world,plans={})
    result['name']='</script><script>bad()</script>'
    page=render(result)
    assert '<script>bad()</script>' not in page
    assert '&lt;script&gt;' in page
    assert all(f'id="{name}"' in page for name in ('pdp', 'shapley', 'lime', 'feature'))
    assert 'id="pdp-feature"' not in page
    assert 'not an estimate of the initial funding' in page
    assert result['baseline']['reconciliation']['max_monthly_cash_error'] < 1e-6
    for kw in ({'years':0},{'n_paths':1},{'years':11}):
        with pytest.raises(ValueError):
            explain_years('P00001',world=world,**kw)
