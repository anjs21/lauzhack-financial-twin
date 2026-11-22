from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from twin.global_explain import grouped_permutation, ensemble_nll
from twin.explain_years import explain_local_path, render
from twin.world.env import CASH_COMPONENTS


class FakeModel:
    members = [1]
    def params(self, X, member):
        # A normal model in CHF thousands that relies only on encoded income.
        return np.ones((len(X),1)), X[:,6,None], np.ones((len(X),1)) * 0.2


def frame():
    rows=[]
    for i in range(8):
        for t in range(12):
            rows.append(dict(profile_id=f'P{i}', t=t, moy=t+1, mode=i%2,
                self_employed=False, pensioner=False, city_idx=i%3,
                income_real=1000+2000*i+t, rent_real=700+50*i,
                flow_real=1000*np.log1p((1000+2000*i+t)/1000)))
    return pd.DataFrame(rows)


def test_income_permutation_preserves_donor_trajectory_and_degrades_nll():
    data=frame()
    original=ensemble_nll(FakeModel(),data).mean()
    shuffled=grouped_permutation(data,'income',np.random.default_rng(5))
    assert ensemble_nll(FakeModel(),shuffled).mean() > original
    for _, group in shuffled.groupby('profile_id'):
        assert np.allclose(np.diff(group.income_real), 1)
    assert np.array_equal(data.income_real, frame().income_real)


def test_season_and_mode_permutations_keep_valid_joint_inputs():
    data=frame()
    shuffled=grouped_permutation(data,'season',np.random.default_rng(7))
    assert all(set(g.moy)==set(range(1,13)) for _,g in shuffled.groupby('profile_id'))
    shuffled=grouped_permutation(data,'employment',np.random.default_rng(7))
    assert all(g['mode'].nunique()==1 for _,g in shuffled.groupby('profile_id'))


def test_local_path_selection_and_accounting():
    n=3; horizon=12
    cash=np.array([np.linspace(10,20,13),np.linspace(10,30,13),np.linspace(10,50,13)])
    a={k:np.zeros((n,horizon)) for k in CASH_COMPONENTS}
    a.update(investment_gain=np.zeros((n,horizon)),loan_interest_expense=np.zeros((n,horizon)))
    a['learned_flow'][:]=np.diff(cash)
    paths=SimpleNamespace(people=[1],cash=cash,nw=cash,accounting=a,action=np.zeros((n,horizon,4),int),events={},below_zero=cash[:,1:]<0)
    result=explain_local_path(paths)
    assert result['path_index']==1
    assert result['years'][0]['closing_cash']==30
    assert result['years'][0]['cash_reconciliation_error']==pytest.approx(0)
    assert result['years'][0]['actions_and_events']==[]
