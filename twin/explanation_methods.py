"""Shapley, LIME and PDP explanations of the fitted flow kernel.

Local target: sum of ensemble-mean operating flows conditional on the recorded
states of one path in a year. Global target: mean flow over historical profiles.
"""
from __future__ import annotations
import numpy as np
import torch
from . import config as C
from .data import profile_months
from .world.flow import features, frame_features, SCALE


def _mean(model, X):
    return np.asarray(model.mean_var(np.asarray(X, np.float32))[0], dtype=float) * SCALE


class _MemberModel:
    def __init__(self, model, member):
        self.model, self.member = model, member

    def mean_var(self, X):
        w, mu, sigma = self.model.params(X, self.member)
        mean = (w * mu).sum(axis=1)
        return mean, (w * (sigma*sigma + mu*mu)).sum(axis=1) - mean*mean


def member_models(model):
    """Individual fitted networks; fallback for a non-ensemble test model."""
    return [_MemberModel(model, i) for i in range(len(model.members))] if hasattr(model, 'params') and hasattr(model, 'members') else [model]


def annual_predict(model, context, income, rent):
    """Expected nominal annual flow for one or more 12-month input vectors."""
    income, rent = np.atleast_2d(income), np.atleast_2d(rent)
    if income.shape != rent.shape or income.shape[1] != len(context['mode']):
        raise ValueError('Income and rent must match path-month dimensions')
    n, months = income.shape
    tiled = lambda key: np.tile(context[key], n)
    X = features(tiled('mode'), tiled('self_employed'), tiled('pensioner'),
                 np.maximum(income.ravel(),0), np.maximum(rent.ravel(),0),
                 tiled('city_idx'), tiled('month'))
    return (_mean(model, X).reshape(n, months) * context['price_index']).sum(axis=1)


def local_year(model, context, reference, seed, samples=1200):
    """Two-group exact Shapley plus LIME of annual expected flow at selected path.

    LIME features are relative income/rent multipliers centred at 1. Its built-in
    normal neighborhood is centred at the explained instance, weighted by LIME's
    exponential kernel, and fitted with its default Ridge surrogate.
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as e:
        raise ImportError('Multi-year Shapley/LIME needs LIME: pip install -r twin/requirements-explain.txt') from e
    income=np.asarray(context['income_real']); rent=np.asarray(context['rent_real'])
    ri=np.asarray(reference['income']); rr=np.asarray(reference['rent'])
    def predict(a,b): return float(annual_predict(model, context, a, b)[0])
    v00=predict(ri,rr); v10=predict(income,rr); v01=predict(ri,rent); v11=predict(income,rent)
    shap_income=((v10-v00)+(v11-v01))/2
    shap_rent=((v01-v00)+(v11-v10))/2
    # Deterministic synthetic local perturbation scales (15% income, 15% rent).
    # These configure LIME's neighbourhood; they do not represent a data distribution.
    training=np.array([[.85,1.0],[1.15,1.0],[1.0,.85],[1.0,1.15]])
    def predictor(x):
        return annual_predict(model, context, np.maximum(x[:,0,None],0)*income,
                               np.maximum(x[:,1,None],0)*rent)
    fits=[]
    for repeat in range(5):
        explainer=LimeTabularExplainer(training, mode='regression', feature_names=['income_multiplier','rent_multiplier'],
            discretize_continuous=False, sample_around_instance=True, random_state=seed+repeat, feature_selection='none')
        fit=explainer.explain_instance(np.ones(2), predictor, num_features=2, num_samples=samples)
        weights={['income_multiplier','rent_multiplier'][idx]:float(value) for idx,value in fit.local_exp[1]}
        fits.append(dict(weights=weights, score=float(fit.score), fitted=float(np.asarray(fit.local_pred).reshape(-1)[0]), seed=seed+repeat))
    lime=fits[0]
    member_attributions=[]
    for member in member_models(model):
        def pm(a,b): return float(annual_predict(member,context,a,b)[0])
        z00,z10,z01,z11=pm(ri,rr),pm(income,rr),pm(ri,rent),pm(income,rent)
        member_attributions.append(dict(income_chf=((z10-z00)+(z11-z01))/2,
                                        rent_chf=((z01-z00)+(z11-z10))/2,
                                        expected_annual_flow_chf=z11))
    weights=lime['weights']
    fitted=lime['fitted']
    return dict(shapley=dict(reference_annual_flow_chf=v00,predicted_annual_flow_chf=v11,
                    income_chf=shap_income,rent_chf=shap_rent,
                    member_attributions=member_attributions,
                    member_income_range_chf=[min(x["income_chf"] for x in member_attributions),max(x["income_chf"] for x in member_attributions)],
                    member_rent_range_chf=[min(x["rent_chf"] for x in member_attributions),max(x["rent_chf"] for x in member_attributions)],
                    residual_chf=v11-v00-shap_income-shap_rent),
        lime=dict(weights_chf_per_standard_deviation=weights, local_fitted_annual_chf=fitted,
                  actual_annual_prediction_chf=v11, weighted_local_r2=lime['score'], repeat_fits=fits,
                  income_weight_range_chf=[min(x['weights']['income_multiplier'] for x in fits),max(x['weights']['income_multiplier'] for x in fits)],
                  rent_weight_range_chf=[min(x['weights']['rent_multiplier'] for x in fits),max(x['weights']['rent_multiplier'] for x in fits)],
                  feature_scales=explainer.scaler.scale_.tolist(),samples=samples,seed=seed,
                  kernel_width=float(np.sqrt(2)*.75)),
        reference_income_real_chf=ri.tolist(),reference_rent_real_chf=rr.tolist(),
        context_income_real_chf=income.tolist(),context_rent_real_chf=rent.tolist())



def categorical_local(model, context, reference_context, group, seed, samples=1200):
    """Single-group conditional Shapley and binary-category LIME.

    Other path inputs remain fixed, so these are separate one-player games;
    they must not be added to the income/rent two-player decomposition.
    """
    from lime.lime_tabular import LimeTabularExplainer
    keys={'employment':('mode','self_employed','pensioner'), 'city':('city_idx',), 'season':('month',)}[group]
    income=np.asarray(context['income_real']); rent=np.asarray(context['rent_real'])
    nmonths=len(income)
    def predict_switch(x, member=model):
        x=np.asarray(x).reshape(-1)
        n=len(x)
        tile=lambda key: np.tile(np.asarray(context[key]),(n,1))
        fields={key:tile(key) for key in ('mode','self_employed','pensioner','city_idx','month')}
        for key in keys:
            ref=np.broadcast_to(reference_context[key], (n,nmonths))
            fields[key]=np.where(x[:,None]>=.5,fields[key],ref)
        X=features(fields['mode'].ravel(),fields['self_employed'].ravel(),fields['pensioner'].ravel(),
                   np.tile(income,n),np.tile(rent,n),fields['city_idx'].ravel(),fields['month'].ravel())
        return (_mean(member,X).reshape(n,nmonths)*context['price_index']).sum(1)
    base,actual=predict_switch([0,1])
    members=[]
    for member in member_models(model):
        v=predict_switch([0,1],member)
        members.append(float(v[1]-v[0]))
    fits=[]
    for repeat in range(5):
        exp=LimeTabularExplainer(np.array([[0],[1],[0],[1]]),mode='regression',
            feature_names=[group],categorical_features=[0],categorical_names={0:['reference','path']},
            discretize_continuous=False,random_state=seed+repeat,feature_selection='none')
        fit=exp.explain_instance(np.array([1.]),lambda x:predict_switch(x[:,0]),num_features=1,num_samples=samples)
        fits.append(dict(weight=float(dict(fit.local_exp[1])[0]),score=float(fit.score),seed=seed+repeat,
                         fitted=float(np.asarray(fit.local_pred).reshape(-1)[0])))
    return dict(reference_flow_chf=float(base),predicted_flow_chf=float(actual),
        shapley_chf=float(actual-base),shapley_member_range_chf=[min(members),max(members)],
        member_contributions_chf=members,lime_weight_chf=fits[0]['weight'],
        lime_weight_range_chf=[min(x['weight'] for x in fits),max(x['weight'] for x in fits)],
        lime_r2=fits[0]['score'],lime_fits=fits,
        reference={key:np.asarray(reference_context[key]).tolist() for key in keys},
        selected={key:np.asarray(context[key]).tolist() for key in keys})

def local_explanations(model, paths, path_index, seed):
    from .data import profiles
    p=profiles()
    # Cohort medians by work mode, fixed for each month; use aggregate fallback.
    med=p.groupby('mode')[['income0','rent0']].median()
    result=[]
    for start in range(0,paths.flow_context['mode'].shape[1],12):
        end=min(start+12,paths.flow_context['mode'].shape[1])
        context={k:v[path_index,start:end] for k,v in paths.flow_context.items()}
        reference={'income':np.array([med.loc[int(m),'income0'] for m in context['mode']]),
                   'rent':np.array([med.loc[int(m),'rent0'] for m in context['mode']])}
        r=local_year(model,context,reference,seed+start)
        person=paths.people.iloc[0]
        reference_context=dict(mode=np.full(end-start,int(person['mode'])),
            self_employed=np.full(end-start,bool(person['self_employed'])),
            pensioner=np.full(end-start,bool(person['pensioner'])),
            city_idx=np.full(end-start,int(p.city_idx.mode().iloc[0])),
            month=np.ones(end-start,dtype=int))
        r['categorical']={group:categorical_local(model,context,reference_context,group,seed+start+100*i)
                          for i,group in enumerate(('employment','city','season'))}
        r['year']=C.START_YEAR+start//12
        r['months']=end-start
        # This is the already sampled flow in that path (not the ensemble mean).
        r['sampled_learned_flow_chf']=float(paths.accounting['learned_flow'][path_index,start:end].sum())
        result.append(r)
    return result


def partial_dependence(model, points=9):
    """Observed-year average predictions; each profile appears once per month."""
    df=profile_months()
    outputs=[]
    for year in (2025,2026):
        frame=df[(df.t//12+C.START_YEAR)==year].copy()
        for feature,column in [('income','income_real'),('rent','rent_real'),
                               ('employment',None),('city','city_idx'),('season','moy')]:
            if feature in ('income','rent'):
                grid=np.unique(np.quantile(frame[column],np.linspace(.05,.95,points)))
                labels=None
            elif feature == 'employment':
                grid=np.arange(6)
                labels=['Student','Part-time','Employed','Self-employed','Unemployed','Retired']
            elif feature == 'city':
                grid=np.arange(len(C.CITIES)); labels=C.CITIES
            else:
                grid=np.arange(1,13); labels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            values=[]; member_values=[]
            for x in grid:
                altered=frame.copy()
                if feature == 'employment':
                    mode,self_employed,pensioner=[(0,False,False),(1,False,False),(2,False,False),
                                                (2,True,False),(3,False,False),(3,False,True)][int(x)]
                    altered['mode']=mode; altered['self_employed']=self_employed; altered['pensioner']=pensioner
                else:
                    altered[column]=float(x) if labels is None else int(x)
                X=frame_features(altered)
                members=[float(_mean(member,X).mean()) for member in member_models(model)]
                values.append(float(np.mean(members))); member_values.append(members)
            outputs.append(dict(year=year,feature=feature,grid_chf=grid.tolist(),category_labels=labels,
                                mean_monthly_flow_chf=values, member_mean_flow_chf=member_values,
                                member_min_chf=[min(v) for v in member_values],
                                member_max_chf=[max(v) for v in member_values], rows=len(frame),profiles=frame.profile_id.nunique()))
    return dict(curves=outputs,method='One-way partial dependence, equal profile-month weighting, ensemble mean, real 2025 CHF',
        limitations=['The saved model is evaluated in-sample on synthetic data; this is model behavior, not a real-world causal effect.',
                     'Holding other features fixed can generate unrealistic combinations when features are correlated; the plot is not a conditional estimate.',
                     'Curves cover 2025–2026 feature contexts, not a claim that the model is calibrated for later years.',
                     'The shaded minimum–maximum across ensemble members measures disagreement between the saved networks, not a confidence interval or total predictive uncertainty.',
                     'Personal offsets, events, financing, and policy responses are excluded.'])


def compute_methods(model,paths,path_index,seed):
    old=torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        return dict(global_pdp=partial_dependence(model),
                    local=local_explanations(model,paths,path_index,seed))
    finally:
        torch.set_num_threads(old)
