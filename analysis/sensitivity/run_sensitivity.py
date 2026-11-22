"""Reproducible paired sensitivity analysis of the current Financial Twin.

Run from repository root: python -B analysis/sensitivity/run_sensitivity.py
This is a model-conditional stress test; no causality or real-world calibration is claimed.
"""
from __future__ import annotations
import argparse
from copy import copy, deepcopy
from dataclasses import replace
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t, norm
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from twin import config as C
from twin.explore import goal_met, person_table, simulate
from twin.policy.traits import Intention, from_profile
from twin.world.env import World

OUT = Path(__file__).resolve().parent / 'results'


class ScaledOfferMarket:
    """Assumption: multiply the learned offer probability, clipping to [0,1]."""
    def __init__(self, market, factor):
        self.market, self.factor = market, factor

    def p_offer(self, *args, **kwargs):
        return np.clip(self.market.p_offer(*args, **kwargs) * self.factor, 0, 1)

    def __getattr__(self, name):
        return getattr(self.market, name)


def paired_mean(values):
    a=np.asarray(values,float)
    if len(a)<2: raise ValueError('At least two paired paths required')
    mean=float(a.mean()); se=float(a.std(ddof=1)/np.sqrt(len(a)))
    margin=float(t.ppf(.975,len(a)-1)*se)
    return {'mean':mean,'lower':mean-margin,'upper':mean+margin,'se':se}


def paired_binary(a,b):
    """Conservative simultaneous Wilson bounds for p(alt only)-p(base only)."""
    a,b=np.asarray(a,bool),np.asarray(b,bool)
    if a.shape != b.shape or a.ndim != 1 or not len(a): raise ValueError('Invalid paired outcomes')
    def interval(x):
        p=float(x.mean()); n=len(x); z=norm.ppf(.9875); d=1+z*z/n
        m=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
        return max(0,m-h),min(1,m+h)
    plus=b & ~a; minus=a & ~b
    lp,up=interval(plus); lm,um=interval(minus)
    return {'change_pp':100*float(plus.mean()-minus.mean()), 'lower_pp':100*max(-1,lp-um),
            'upper_pp':100*min(1,up-lm), 'discordant_plus':int(plus.sum()),'discordant_minus':int(minus.sum())}


def metrics(paths):
    high,_=goal_met(paths)
    return {'shortfall':paths.below_zero.any(1), 'breach':paths.credit_breach.any(1),
            'goal':high, 'wealth':paths.nw[:,-1], 'min_cash':paths.cash.min(1),
            'ever_graduated':paths.events['graduated'].any(1),
            'first_review_selected_masters':paths.action[:,0,0]==1,
            'first_review_selected_work_search':paths.action[:,0,0]==3}


def summary(m):
    return dict(shortfall_pct=100*float(m['shortfall'].mean()), breach_pct=100*float(m['breach'].mean()),
                goal_pct=100*float(m['goal'].mean()), median_min_cash_chf=float(np.median(m['min_cash'])),
                median_final_wealth_chf=float(np.median(m['wealth'])), mean_final_wealth_chf=float(m['wealth'].mean()),
                first_review_masters_pct=100*float(m['first_review_selected_masters'].mean()),
                first_review_work_search_pct=100*float(m['first_review_selected_work_search'].mean()))


def pair(base,alt):
    return dict(shortfall=paired_binary(base['shortfall'],alt['shortfall']),
                breach=paired_binary(base['breach'],alt['breach']),
                goal=paired_binary(base['goal'],alt['goal']),
                mean_wealth_chf=paired_mean(alt['wealth']-base['wealth']))


def make_world(original, degree_shift=None, offer_factor=1.0, credit_rate=None, delay=None, moving=None, inflation=None):
    pr=deepcopy(original.pr)
    for key,value in [('degree_rank_shift',degree_shift),('delay_probability',delay)]:
        if value is not None: pr['masters'][key]=value
    if credit_rate is not None: pr['credit']['rate_per_year']=credit_rate
    if moving is not None: pr['housing']['moving_cost']=moving
    if inflation is not None:
        for key in pr['economy']['cost_growth_per_year']: pr['economy']['cost_growth_per_year'][key]=inflation
    world=World(original.flow,original.income_model,original.rent_model,pr,copy(original.cfg))
    if world.market is None: raise ValueError('This stress analysis requires the learned market model')
    world.market=ScaledOfferMarket(world.market,offer_factor)
    return world


def analyse(n_paths, grid_paths, seed, profile_id="P00001"):
    if min(n_paths,grid_paths)<2: raise ValueError('At least two paths required')
    world=World.load(); person=person_table(profile_id)
    persona=from_profile(person.iloc[0])
    study=replace(persona,intentions=[Intention('status',1,1.0,when='2025-01')])
    work=replace(persona,intentions=[Intention('status',3,1.0,when='2025-01')])
    def run(w,p,n,tuition=8000,extra_cash=0):
        pp=person.copy(); pp['tuition']=tuition; pp['cash0']+=extra_cash
        return metrics(simulate(w,pp,p,n_paths=n,seed=seed))
    baseline_world=make_world(world)
    print('Running paired baseline...',flush=True)
    baseline=run(baseline_world,study,n_paths)
    baseline_work=run(baseline_world,work,n_paths)
    cases=[
        ('Tuition CHF 2,500',{'tuition':2500}),('Tuition CHF 12,000',{'tuition':12000}),
        ('Degree pay-rank shift 0.00',{'degree_shift':0.0}),('Degree pay-rank shift 0.10',{'degree_shift':0.10}),
        ('Job-offer probability ×0.5',{'offer_factor':0.5}),('Job-offer probability ×1.5',{'offer_factor':1.5}),
        ('Study-delay probability 0%',{'delay':0.0}),('Study-delay probability 25%',{'delay':0.25}),
        ('Credit rate 4%',{'credit_rate':0.04}),('Credit rate 12%',{'credit_rate':0.12}),
        ('Moving cost CHF 1,000',{'moving':1000}),('Moving cost CHF 4,000',{'moving':4000}),
        ('All-regime price growth 0%',{'inflation':0.0}),('All-regime price growth 3%',{'inflation':0.03}),
    ]
    tests=[]
    for name, setting in cases:
        print('Running:',name,flush=True)
        w=make_world(world,**{k:v for k,v in setting.items() if k!='tuition'})
        v=run(w,study,n_paths,tuition=setting.get('tuition',8000))
        tests.append(dict(name=name,setting=setting,summary=summary(v),difference_from_study_baseline=pair(baseline,v)))
    grid=[]
    for tuition in (2500,8000,12000):
        for factor in (.5,1.0,1.5):
            for degree in (0.,.10,.20):
                print(f'Grid: tuition={tuition}, offer multiplier={factor}, rank shift={degree}',flush=True)
                w=make_world(world,degree_shift=degree,offer_factor=factor)
                m=run(w,study,grid_paths,tuition)
                job=run(w,work,grid_paths,tuition)
                grid.append(dict(tuition=tuition,offer_factor=factor,degree_rank_shift=degree,
                                 study=summary(m),work=summary(job),study_minus_work=pair(job,m)))
    funding=[]
    for extra in (0,5000,10000,20000,40000):
        print(f'Funding grid: +CHF {extra}',flush=True)
        m=baseline if extra==0 else run(baseline_world,study,n_paths,extra_cash=extra)
        funding.append(dict(extra_initial_cash_chf=extra,summary=summary(m),difference_from_baseline=pair(baseline,m)))
    return dict(profile=profile_id, profile_name=str(person.iloc[0]['name']),
        goal_name=str(person.iloc[0]['high_goal']), goal_target_chf=float(person.iloc[0]['high_target']), seed=seed,paths=n_paths,grid_paths=grid_paths,horizon_months=C.HORIZON,
        markets='learned, job offer probability multiplied and clipped',
        plans={'study':'January 2025 intention to start a master’s, subject to mask and admission',
               'work':'January 2025 intention to seek full-time work, subject to mask and job offers'},
        assumptions={'baseline_tuition_chf':8000,'baseline_degree_rank_shift':world.pr['masters']['degree_rank_shift'],
                     'baseline_credit_rate':world.pr['credit']['rate_per_year'],
                     'baseline_delay_probability':world.pr['masters']['delay_probability'],
                     'baseline_moving_cost_chf':world.pr['housing']['moving_cost'],
                     'offer_probability_multipliers':[0.5,1.0,1.5]},
        baseline_study=summary(baseline),baseline_work=summary(baseline_work),
        study_minus_work=pair(baseline_work,baseline),one_at_a_time=tests,grid=grid,funding=funding,
        caveats=[
            'Actions are intentions at the January review, not guaranteed admission or employment; actions may be unavailable or redundant for this profile, and later behaviour remains adaptive.',
            'Tuition baseline is the fixed stress-test assumption CHF 8,000, not necessarily the profile-specific tuition.',
            'The goal evaluation uses the simulator deadline of December 2029 for all profiles, regardless of the date in the source data.',
            'Tuition input applies to all months the model treats the person as a student, including any initial pre-master study. The flow model already includes typical fees; only the excess is charged separately.',
            'Offer factor multiplies the learned model probability and is clipped to [0,1]; it is an illustrative stress, not a calibrated economic scenario.',
            'Money is nominal CHF and recorded negative cash uses the model credit and automatic investment-top-up rules; credit approval is not modelled.',
            'Goal uses the current goal_met definition, including its credit-breach condition. Mean paired changes and percentile-based summaries are distinct.',
            'Personal residual offsets were fit on available history and may include information after the January 2025 start; not an out-of-time backtest.',
            'Output intervals describe Monte Carlo error conditional on model and assumptions, not causal effects or real-world confidence bounds.',
            'The funding grid re-simulates adaptive behaviour and is not a guaranteed minimum; probability may not be monotonic.',
            'With only a few hundred grid paths, small differences can be Monte Carlo noise; do not optimise based on them.',
        ])


def write_results(result, out):
    OUT = out
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'sensitivity.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (OUT/'one_at_a_time.csv').open('w',newline='') as f:
        fields=['case','shortfall_pct','goal_pct','median_min_cash_chf','median_final_wealth_chf',
                'shortfall_change_pp','shortfall_ci_low_pp','shortfall_ci_high_pp','mean_wealth_change_chf',
                'mean_wealth_ci_low_chf','mean_wealth_ci_high_chf']
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader()
        for v in result['one_at_a_time']:
            d=v['difference_from_study_baseline']; sm=v['summary']
            wr.writerow(dict(case=v['name'],**{k:sm[k] for k in fields[1:5]},
                shortfall_change_pp=d['shortfall']['change_pp'],shortfall_ci_low_pp=d['shortfall']['lower_pp'],
                shortfall_ci_high_pp=d['shortfall']['upper_pp'],mean_wealth_change_chf=d['mean_wealth_chf']['mean'],
                mean_wealth_ci_low_chf=d['mean_wealth_chf']['lower'],mean_wealth_ci_high_chf=d['mean_wealth_chf']['upper']))
    with (OUT/'joint_grid.csv').open('w',newline='') as f:
        fields=['tuition','offer_factor','degree_rank_shift','study_shortfall_pct','work_shortfall_pct',
                'study_goal_pct','work_goal_pct','study_minus_work_shortfall_pp','shortfall_ci_low_pp','shortfall_ci_high_pp',
                'study_minus_work_mean_wealth_chf','wealth_ci_low_chf','wealth_ci_high_chf']
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader()
        for v in result['grid']:
            d=v['study_minus_work'];wr.writerow(dict(tuition=v['tuition'],offer_factor=v['offer_factor'],
                degree_rank_shift=v['degree_rank_shift'],study_shortfall_pct=v['study']['shortfall_pct'],
                work_shortfall_pct=v['work']['shortfall_pct'],study_goal_pct=v['study']['goal_pct'],work_goal_pct=v['work']['goal_pct'],
                study_minus_work_shortfall_pp=d['shortfall']['change_pp'],shortfall_ci_low_pp=d['shortfall']['lower_pp'],
                shortfall_ci_high_pp=d['shortfall']['upper_pp'],study_minus_work_mean_wealth_chf=d['mean_wealth_chf']['mean'],
                wealth_ci_low_chf=d['mean_wealth_chf']['lower'],wealth_ci_high_chf=d['mean_wealth_chf']['upper']))
    b=result['baseline_study']; w=result['baseline_work']; diff=result['study_minus_work']
    lines=[f"# {result['profile_name']} ({result['profile']}): sensitivity analysis of the current simulator",'',
       f"Seed: {result['seed']}; baseline and individual stresses: {result['paths']} paired paths; joint grid: {result['grid_paths']} paired paths; horizon: 120 months.", '',
       '## Baseline comparison','', '| Measure | Study intention | Work-search intention |','|---|---:|---:|',
       f"| Cash negative at any month end | {b['shortfall_pct']:.1f}% | {w['shortfall_pct']:.1f}% |",
       f"| Ever exceeds credit limit | {b['breach_pct']:.1f}% | {w['breach_pct']:.1f}% |",
       f"| High-priority goal at deadline | {b['goal_pct']:.1f}% | {w['goal_pct']:.1f}% |",
       f"| Median minimum cash (CHF) | {b['median_min_cash_chf']:,.0f} | {w['median_min_cash_chf']:,.0f} |",
       f"| Median terminal net worth (CHF) | {b['median_final_wealth_chf']:,.0f} | {w['median_final_wealth_chf']:,.0f} |",
       f"| Mean terminal net worth (CHF) | {b['mean_final_wealth_chf']:,.0f} | {w['mean_final_wealth_chf']:,.0f} |",'',
       f"Study minus work: shortfall {diff['shortfall']['change_pp']:+.1f} pp (conservative paired 95% Monte Carlo interval {diff['shortfall']['lower_pp']:+.1f} to {diff['shortfall']['upper_pp']:+.1f} pp); mean terminal net worth CHF {diff['mean_wealth_chf']['mean']:+,.0f} (paired t interval CHF {diff['mean_wealth_chf']['lower']:+,.0f} to {diff['mean_wealth_chf']['upper']:+,.0f}).",'',
       f"At the opening review, {b['first_review_masters_pct']:.1f}% of study-intention paths selected the master’s action and {w['first_review_work_search_pct']:.1f}% of work-intention paths selected the job-search action. Selection does not guarantee admission, employment, or completion.",'',
       '## One-at-a-time changes to the study-intention baseline','', '| Change | Shortfall | Difference vs baseline (pp; paired 95% interval) | Goal | Mean final wealth change CHF (paired 95% interval) |', '|---|---:|---:|---:|---:|']
    for x in result['one_at_a_time']:
        d=x['difference_from_study_baseline']; q=d['shortfall']; z=d['mean_wealth_chf']; v=x['summary']
        lines.append(f"| {x['name']} | {v['shortfall_pct']:.1f}% | {q['change_pp']:+.1f} ({q['lower_pp']:+.1f} to {q['upper_pp']:+.1f}) | {v['goal_pct']:.1f}% | {z['mean']:+,.0f} ({z['lower']:+,.0f} to {z['upper']:+,.0f}) |")
    lines+=['','## Funding grid','', '| Added starting cash CHF | Negative-cash rate | Change vs baseline pp (paired 95% interval) |', '|---:|---:|---:|']
    for x in result['funding']:
        q=x['difference_from_baseline']['shortfall']
        lines.append(f"| {x['extra_initial_cash_chf']:,.0f} | {x['summary']['shortfall_pct']:.1f}% | {q['change_pp']:+.1f} ({q['lower_pp']:+.1f} to {q['upper_pp']:+.1f}) |")
    lines+=['','## Joint grid','', 'See `joint_grid.csv` for 27 combinations of tuition, job-offer probability multiplier and degree pay-rank shift. Both plans are re-simulated in every combination. Numeric intervals are conditional Monte Carlo intervals, not evidence of causal effects.','','## Interpretation and limitations','']
    lines+=['- '+v for v in result['caveats']]
    lines+=['','## Reproduce','','```bash','python -B analysis/sensitivity/run_sensitivity.py --profile '+result['profile']+' --paths '+str(result['paths'])+' --grid-paths '+str(result['grid_paths'])+' --seed '+str(result['seed']),'```','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',default='P00001',help='Synthetic profile ID (default P00001)')
    parser.add_argument('--paths',type=int,default=1000)
    parser.add_argument('--grid-paths',type=int,default=300)
    parser.add_argument('--seed',type=int,default=20261122)
    args=parser.parse_args()
    torch.set_num_threads(1)
    
    try:
        person_table(args.profile)
    except KeyError:
        parser.error(f'Unknown profile ID: {args.profile}')
    result=analyse(args.paths,args.grid_paths,args.seed,args.profile)
    out=OUT/'profiles'/args.profile
    write_results(result,out)
    print('Results:',out,flush=True)


if __name__=='__main__': main()
