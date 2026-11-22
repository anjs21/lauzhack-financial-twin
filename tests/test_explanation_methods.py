import numpy as np
import pytest
from twin.explanation_methods import annual_predict, local_year


class Model:
    def mean_var(self, X):
        # Exactly linear in the encoded log inputs, with an interaction.
        mean=2*X[:,6]-X[:,7]+0.4*X[:,6]*X[:,7]
        return mean, np.ones(len(X))


def context():
    return dict(mode=np.zeros(12),self_employed=np.zeros(12),pensioner=np.zeros(12),
                income_real=np.full(12,2000.),rent_real=np.full(12,1200.),city_idx=np.zeros(12),
                month=np.arange(1,13),price_index=np.ones(12))


def test_shapley_reconciles_and_lime_is_locally_faithful():
    c=context()
    ref=dict(income=np.full(12,1500.),rent=np.full(12,1000.))
    result=local_year(Model(),c,ref,seed=123,samples=1200)
    s=result['shapley']; l=result['lime']
    assert s['reference_annual_flow_chf']+s['income_chf']+s['rent_chf']==pytest.approx(s['predicted_annual_flow_chf'],abs=1e-6)
    assert s['income_chf'] > 0 and s['rent_chf'] < 0
    assert l['weighted_local_r2'] > 0.95
    assert l['local_fitted_annual_chf']==pytest.approx(s['predicted_annual_flow_chf'],abs=1000)
    assert l['weights_chf_per_standard_deviation']['income_multiplier'] > 0
    assert l['weights_chf_per_standard_deviation']['rent_multiplier'] < 0
    assert result == local_year(Model(),c,ref,seed=123,samples=1200)


def test_annual_predict_applies_price_index_and_rejects_shape():
    c=context()
    x=annual_predict(Model(),c,c['income_real'],c['rent_real'])[0]
    c['price_index']*=2
    assert annual_predict(Model(),c,c['income_real'],c['rent_real'])[0]==pytest.approx(2*x)
    with pytest.raises(ValueError):
        annual_predict(Model(),c,np.zeros(11),np.zeros(11))


def test_pdp_uses_each_observed_year_and_synthetic_model(monkeypatch):
    import pandas as pd
    from twin import explanation_methods as methods
    rows=[]
    for year in (2025, 2026):
        for person in range(5):
            for month in range(1,13):
                rows.append(dict(profile_id=f'P{person}',t=(year-2025)*12+month-1,
                    mode=0,self_employed=False,pensioner=False,city_idx=0,moy=month,
                    income_real=1000+500*person,rent_real=800+100*person))
    data=pd.DataFrame(rows)
    monkeypatch.setattr(methods, 'profile_months', lambda: data)
    result=methods.partial_dependence(Model(),points=4)
    assert len(result['curves'])==10
    assert {c['feature'] for c in result['curves']} == {'income','rent','employment','city','season'}
    for curve in result['curves']:
        assert curve['rows']==60 and curve['profiles']==5
        if curve['feature'] in ('income', 'rent'):
            diff=np.diff(curve['mean_monthly_flow_chf'])
            assert np.all(diff>0) if curve['feature']=='income' else np.all(diff<0)
        else:
            assert len(curve['category_labels']) == len(curve['grid_chf'])
            assert np.allclose(curve['mean_monthly_flow_chf'], curve['mean_monthly_flow_chf'][0])


def test_categorical_local_uses_valid_reference_and_detects_effect():
    from twin.explanation_methods import categorical_local
    class CategoryModel:
        def mean_var(self, X):
            return 2*X[:,0]+3*X[:,1]+X[:,8], np.ones(len(X))
    c=context()
    ref={k:np.copy(v) for k,v in c.items()}
    ref['mode']=np.ones(12)
    result=categorical_local(CategoryModel(),c,ref,'employment',seed=9,samples=500)
    assert result['shapley_chf']==pytest.approx(-12000)
    assert result['lime_weight_chf'] < 0
    assert result['lime_r2'] > .95
    assert result['shapley_member_range_chf']==pytest.approx([-12000,-12000])
    assert result==categorical_local(CategoryModel(),c,ref,'employment',seed=9,samples=500)
