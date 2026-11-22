"""Explain an LLM plan, the simulator policy executing it, and resulting outcomes.

Default: reproduce the bundled, real Qwen-generated Mia example, without a network call.
See README.md for using a saved profile plan and optional live planner experiments.
"""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import norm
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis' / 'sensitivity'))
from run_sensitivity import make_world, metrics, pair, summary
from twin import config as C, llm_provider
from twin.explore import goals_per_path, person_table, simulate
from twin.plan.agent import PlannerAgent
from twin.plan.evaluate import evaluate, with_plan
from twin.plan.schema import LifePlan, OPTIONS
from twin.policy.rules import HEADS, RulePolicy
from twin.policy.traits import from_profile
from twin.world.env import Streams, World

OUT = ROOT / 'analysis' / 'policy-explanations' / 'results'
LABELS = {'status':['Continue','Start master’s','Part-time','Full-time','Stop working'],
          'housing':['Cheaper','Stay','Bigger'], 'spending':['Cut','Keep','Raise'],
          'assets':['Sell to buffer','Hold','Invest surplus']}


def read_input(profile, plan_path, version):
    if profile == 'example':
        raw = json.loads((ROOT / 'web/src/example.json').read_text())
        people = person_table(custom={**raw['person'], 'study_months_left': 1, 'in_masters': False})
        persona = from_profile(people.iloc[0]); persona.traits.update(raw['traits'])
        path = plan_path or ROOT / 'web/src/example-plan.json'
        source = 'Bundled Mia example; LLM-generated plan, matched to web/src/example.json.'
    else:
        people = person_table(profile); persona = from_profile(people.iloc[0])
        path = plan_path or C.ARTIFACTS / 'plans' / f'{profile}.json'
        source = 'Saved plan for the selected synthetic profile; verify the plan was generated for this profile.'
    raw_plan = json.loads(path.read_text())
    entries = raw_plan.get('history') or raw_plan.get('versions') or ([raw_plan['latest']] if raw_plan.get('latest') else [])
    entries = [e for e in entries if isinstance(e, dict) and isinstance(e.get('plan'), dict)]
    if not entries: raise ValueError('Plan file has no full version containing a plan; provide a saved plan or example-plan.json')
    if version < 1 or version > len(entries): raise ValueError(f'Plan version must be between 1 and {len(entries)}')
    entry = entries[version - 1]
    plan, errors = LifePlan.from_dict(entry['plan'])
    if errors: raise ValueError('Invalid saved plan: ' + '; '.join(errors))
    return people, persona, plan, entry, entries, {'file': str(path), 'description': source,
            'model': raw_plan.get('model'), 'version': version, 'profile': profile, 'as_of': entry.get('as_of')}


def policy_at_opening(world, people, persona, plan, cash_values, seed):
    """Exact one-step probabilities conditional on fixed opening state, traits, memory and plan."""
    stream = Streams(seed, 1)
    traits = np.array([persona.vector()])
    original = world.initial_state(people, 1, stream, buffer_months=RulePolicy.buffer_months(traits))
    out = []
    for cash in cash_values:
        state = original.copy(); state.cash[:] = cash
        views = {}
        for name, p in [('habits', persona), ('with_plan', with_plan(persona, plan))]:
            policy = RulePolicy(world, traits, p.temperature, goals_per_path(people, 1), p.intentions)
            policy.reset(state, stream)
            context = policy.context(state)
            masks = dict(zip(HEADS, world.masks(state)))
            _, probabilities = policy.act(state, stream)
            views[name] = {'cash_pressure': float(context['cash_pressure'][0]),
                           'cash_months': float(context['cash_m'][0]),
                           'probabilities': {h: [float(x) for x in probabilities[h][0]] for h in HEADS},
                           'available': {h: [bool(x) for x in masks[h][0]] for h in HEADS}}
        out.append({'cash_chf':float(cash), **views})
    a,b = out[0]['with_plan']['probabilities'],out[1]['with_plan']['probabilities']
    changes={}
    for h in HEADS:
        pa,pb=np.array(a[h]),np.array(b[h])
        changes[h]={'change_pp':(100*(pb-pa)).tolist()}
    return {'month':'2025-01','fixed_traits':persona.traits,'cash_comparisons':out,'change_low_to_high':changes,
            'labels':LABELS, 'note':'Exact conditional one-step policy probabilities, not simulated frequencies or LLM probabilities. Derived context and action masks are recomputed; opening memory and other state are fixed. The low-cash input is hypothetical.'}


def sampled_plan_comparison(world, people, persona, wishes, low, high, samples, n_paths, seed):
    if not llm_provider.ready(): raise RuntimeError(llm_provider.SETUP)
    if samples < 2: raise ValueError('Use at least 2 live samples per setting')
    agent=PlannerAgent(world,n_paths=n_paths,seed=seed)
    def wilson(count, confidence=.95):
        z=norm.ppf((1+confidence)/2); p=count/samples; den=1+z*z/samples
        mid=(p+z*z/(2*samples))/den
        half=z*np.sqrt(p*(1-p)/samples+z*z/(4*samples*samples))/den
        return {'count':count,'frequency':p,'lower':max(0.,float(mid-half)),'upper':min(1.,float(mid+half))}
    results=[]
    for label,cash in [('low_cash',low),('baseline_cash',high)] :
        p=people.copy(); p['cash0']=cash
        plans=[]
        for i in range(samples):
            print(f'Live LLM {label} sample {i+1}/{samples} (up to {agent.max_rounds} LLM rounds)',flush=True)
            r=agent.draft(p,persona,wishes)
            plans.append({'signed':r.signed,'plan':r.plan.to_dict()})
        options=sorted({f'{s["head"]}:{s["option"]}' for p0 in plans for s in p0['plan']['steps']})
        frequencies={o:wilson(sum(any(f'{s["head"]}:{s["option"]}'==o for s in p0['plan']['steps']) for p0 in plans)) for o in options}
        results.append({'setting':label,'cash_chf':cash,'plans':plans,'step_presence_frequency':frequencies})
    effects={}
    options=sorted(set(results[0]['step_presence_frequency']) | set(results[1]['step_presence_frequency']))
    for option in options:
        counts=[g['step_presence_frequency'].get(option,{}).get('count',0) for g in results]
        a,b=[wilson(count,.975) for count in counts]
        effects[option]={'change_pp':100*(b['frequency']-a['frequency']),
                         'lower_pp':100*max(-1.,b['lower']-a['upper']),
                         'upper_pp':100*min(1.,b['upper']-a['lower'])}
    return {'model':llm_provider.model(),'samples_per_setting':samples,'wishes':wishes,'simulation_paths_per_draft':n_paths,'simulation_seed':seed,'groups':results,'presence_effects':effects,
            'note':'Frequencies are presence of a step anywhere in a generated plan, not monthly action probabilities. Repeated independently, full planner including simulation feedback. Wilson intervals describe Monte Carlo uncertainty for each frequency; combined conservative 97.5% Wilson bounds yield a 95% interval for independent-condition frequency differences. Small samples have high uncertainty. No causal claims.'}


def analyse(people, persona, plan, entry, entries, provenance, n, seed, low_cash, live_samples, wishes):
    world=World.load()
    high_cash=float(people.iloc[0]['cash0'])
    # Fixed mean trait and opening memory for this one-state local explanation.
    local=policy_at_opening(world,people,persona,plan,[low_cash,high_cash],seed)
    print('Simulating plan execution and habits...',flush=True)
    ev,_,planned=evaluate(world,people,persona,plan,n_paths=n,seed=seed)
    habits=simulate(world,people,persona,n_paths=n,seed=seed)
    pmet,hmet=metrics(planned),metrics(habits)
    stress=[]
    for factor in (0.5,1.5):
        print(f'Simulating offer probability ×{factor}...',flush=True)
        w=make_world(world,offer_factor=factor)
        _,_,paths=evaluate(w,people,persona,plan,n_paths=n,seed=seed)
        m=metrics(paths)
        stress.append({'offer_factor':factor,'summary':summary(m),'difference_vs_plan_baseline':pair(pmet,m)})
    live=None
    if live_samples: live=sampled_plan_comparison(world,people,persona,wishes,low_cash,high_cash,live_samples,min(n,300),seed)
    history=[{'version':i+1,'kind':v.get('kind'),'said':v.get('said'), 'signed':v.get('signed'),
              'steps':[{'id':s.get('id'),'head':s.get('head'),'option':s.get('option')} for s in v['plan'].get('steps',[])]}
             for i,v in enumerate(entries)]
    return {'provenance':provenance,'person':{'name':str(people.iloc[0]['name']), 'initial_cash_chf':high_cash,
             'high_goal':str(people.iloc[0]['high_goal']),'high_target_chf':float(people.iloc[0]['high_target'])},
            'paths':n,'seed':seed,'plan':plan.to_dict(),'planner_history':history,'planner_claims':{'signed':entry.get('signed'),
             'rounds':entry.get('rounds',[]),'changes':entry.get('changes')},'live_planner_experiment':live,
            'execution':{'opening_state_intervention':local,'step_report':ev['steps'],
                         'off_plan_changes_per_life':ev['off_plan_changes_per_life']},
            'outcomes':{'plan':summary(pmet),'habits':summary(hmet),'plan_minus_habits':pair(hmet,pmet),
                        'offer_stress':stress},
            'limitations':['The saved plan is LLM-generated; its narrative and stated reasons are self-reports, not verified causal explanations.',
             'The saved draft and revision respond to different messages and are not repeated samples of one state.',
             'The opening intervention explains the rule policy conditioned on one state and the fixed saved plan, not the LLM; masks and triggers can change.',
             'Exact one-step probabilities omit uncertainty about traits, state and model. Plan-step reports count different events over all paths and are not mutually exclusive.',
             'Plan versus habits also changes spontaneity and multiple planned intentions, so it does not isolate individual steps or identify causal effects.',
             'Financial uncertainty intervals quantify finite-path simulation error conditional on the model, not real-world forecast uncertainty.',
             'Plan and baseline simulate same random seed, but path-dependent behaviour and events may diverge.',
             'Saved plans revised as of a later date are simulated from January 2025; the results are counterfactual, not a reconstruction of what was known then.',
             'All amounts nominal CHF; goal uses the simulator fixed December 2029 deadline and credit-breach rule.',
             'The offer stress changes the simulated world with the plan fixed; it does not ask the LLM to replan.']}


def render(r):
    e=lambda x:html.escape(str(x),quote=True)
    f=lambda n: f'{n:,.1f}'
    plan=r['plan']; person=r['person']; loc=r['execution']['opening_state_intervention']; vals=loc['cash_comparisons']; outcomes=r['outcomes']; d=outcomes['plan_minus_habits']
    blocks=['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Three-level policy explanation</title><style>body{font:16px/1.55 system-ui,sans-serif;color:#183249;background:#f3f7f6;margin:0}main{max-width:1050px;margin:auto;padding:32px 20px 60px}.card{background:#fff;border:1px solid #dce8e7;border-radius:12px;padding:24px;margin:18px 0}h1,h2{line-height:1.2}h1{font-size:2.3rem}h2{margin-top:0}.eyebrow{color:#177077;text-transform:uppercase;letter-spacing:.12em;font-weight:700;font-size:.75rem}.muted{color:#526c7d}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;min-width:550px;font-size:.9rem}td,th{border-bottom:1px solid #e3eaea;text-align:left;padding:9px 8px;vertical-align:top}th{color:#526c7d}.note{background:#edf7f5;border-left:3px solid #16757b;padding:12px 16px}.bar{display:inline-block;background:#1a8283;height:9px;border-radius:3px}code{font-size:.9em}li{margin:6px 0}</style></head><body><main>',
       '<div class="eyebrow">Financial twin / explainability</div><h1>From plan to outcomes</h1>',
       '<p class="muted">'+e(person['name'])+' · '+str(r['paths'])+' paired simulated paths · seed '+str(r['seed'])+' · January 2025 to December 2034</p>',
       '<section class="card"><div class="eyebrow">Level 1 / LLM planner</div><h2>What did the LLM plan?</h2>',
       '<p><strong>Source:</strong> '+e(r['provenance']['description'])+' Model: '+e(r['provenance']['model'] or 'not recorded')+'. Version '+str(r['provenance']['version'])+'.</p>',
       '<p class="note">This is one saved LLM-generated plan. The narrative and reasons are the model’s statements, not evidence of its internal reasoning. The saved versions respond to different user messages; they are not repeated samples of the same input.</p>',
       '<h3>Plan narrative</h3><p>'+e(plan['narrative'])+'</p><p>Spontaneity setting: '+f(100*plan['spontaneity'])+'% (applies to unplanned work and home actions).</p>',
       '<div class="table"><table><tr><th>Step</th><th>Action</th><th>Timing / condition</th><th>Commitment</th></tr>']
    for s in plan['steps']:
        cond=[]
        for k in ('start','until','after','event','cash_months_below','cash_months_above','min_age','max_age'):
            if k in s:cond.append(k.replace('_',' ')+': '+str(s[k]))
        blocks.append('<tr><td>'+e(s['label'])+'</td><td>'+e(s['head']+' / '+s['option'])+'</td><td>'+e('; '.join(cond) or 'No additional condition')+'</td><td>'+f(100*s['commitment'])+'%</td></tr>')
    blocks.append('</table></div><h3>Saved plan history</h3><ul>')
    for h in r['planner_history']:
        blocks.append('<li>Version '+str(h['version'])+' · '+e(h.get('kind') or 'unknown')+' · User input: '+e(h.get('said') or 'not recorded')+'. Steps: '+e(', '.join(x['head']+':'+x['option'] for x in h['steps']))+'.</li>')
    blocks.append('</ul>')
    if r['live_planner_experiment'] is None:
        blocks.append('<p class="muted">Repeated live LLM comparisons have not been run. They require provider credentials and the optional <code>--live-samples</code> command. The plan history above is not an experiment.</p>')
    else:
        live=r['live_planner_experiment'];blocks.append('<h3>Repeated live planner experiment</h3><p>'+e(live['note'])+'</p><div class="table"><table><tr><th>Setting</th><th>Plan-step presence frequency</th></tr>')
        for g in live['groups']:
            blocks.append('<tr><td>'+e(g['setting'])+' (CHF '+f(g['cash_chf'])+')</td><td>'+e('; '.join(k+': '+f(v['frequency']*100)+'% (95% '+f(v['lower']*100)+'–'+f(v['upper']*100)+'%)' for k,v in g['step_presence_frequency'].items()))+'</td></tr>')
        blocks.append('</table></div><h4>Change in plan-step presence · baseline cash minus low cash</h4><div class="table"><table><tr><th>Step</th><th>Change</th><th>Conservative 95% interval</th></tr>')
        for option,q in live['presence_effects'].items():
            blocks.append('<tr><td>'+e(option)+'</td><td>'+f(q['change_pp'])+' pp</td><td>'+f(q['lower_pp'])+' to '+f(q['upper_pp'])+' pp</td></tr>')
        blocks.append('</table></div>')
    blocks.append('</section><section class="card"><div class="eyebrow">Level 2 / simulator execution</div><h2>How is the plan carried out?</h2><p>The fixed saved plan is combined with the rule-based persona policy. These results use the current simulator, so they may differ from the plan’s original saved evaluation.</p><div class="table"><table><tr><th>Plan step</th><th>Trigger arose</th><th>Carried out</th><th>Due but never available</th><th>Typical year</th></tr>')
    for s in r['execution']['step_report']:
        blocks.append('<tr><td>'+e(s['label'])+'</td><td>'+f(s['trigger_share']*100)+'%</td><td>'+f(s['taken_share']*100)+'%</td><td>'+f(s['blocked_share']*100)+'%</td><td>'+e(s.get('typical_year') or '—')+'</td></tr>')
    blocks.append('</table></div><p class="muted">Shares are percentages of all paths. Events can overlap and do not form mutually exclusive categories. A commitment is conditional on the step being due and available; it is not the overall completion rate.</p><h3>Opening-state cash intervention</h3><p>At January 2025, compare the same opening state and fixed mean traits with cash set to CHF '+f(vals[0]['cash_chf'])+' versus CHF '+f(vals[1]['cash_chf'])+'. Derived inputs, plan triggers and action masks are recalculated. The table shows direct rule-policy probabilities with the plan applied, not LLM response frequencies.</p>')
    for head in HEADS:
        blocks.append('<h4>'+e(head.title())+'</h4><div class="table"><table><tr><th>Action</th><th>Low cash · plan</th><th>Starting cash · plan</th><th>Change · plan</th><th>Habits: low / starting cash</th><th>Availability (low / start)</th></tr>')
        for i,label in enumerate(LABELS[head]):
            a=vals[0]['with_plan']['probabilities'][head][i]*100;b=vals[1]['with_plan']['probabilities'][head][i]*100
            available=[v['with_plan']['available'][head][i] for v in vals]
            blocks.append('<tr><td>'+e(label)+'</td><td>'+f(a)+'%</td><td>'+f(b)+'%</td><td>'+f(b-a)+' pp</td><td>'+f(vals[0]['habits']['probabilities'][head][i]*100)+'% / '+f(vals[1]['habits']['probabilities'][head][i]*100)+'%</td><td>'+('Yes' if available[0] else 'No')+' / '+('Yes' if available[1] else 'No')+'</td></tr>')
        blocks.append('</table></div>')
    blocks.append('</section><section class="card"><div class="eyebrow">Level 3 / financial outcomes</div><h2>What happens financially?</h2><p>Compare the fixed plan to the same persona’s habits without the plan, using paired random draws. The plan changes multiple intentions and spontaneity; this comparison does not isolate a single step.</p><div class="table"><table><tr><th>Outcome</th><th>Plan</th><th>Habits</th><th>Plan minus habits · 95% Monte Carlo interval</th></tr>')
    for label,key,skey in [('Ever negative cash','shortfall','shortfall_pct'),('Credit limit breached','breach','breach_pct'),('High-priority goal met','goal','goal_pct')]:
        q=d[key];blocks.append('<tr><td>'+label+'</td><td>'+f(outcomes['plan'][skey])+'%</td><td>'+f(outcomes['habits'][skey])+'%</td><td>'+f(q['change_pp'])+' pp ('+f(q['lower_pp'])+' to '+f(q['upper_pp'])+')</td></tr>')
    q=d['mean_wealth_chf']; blocks.append('<tr><td>Mean final net worth</td><td>CHF '+f(outcomes['plan']['mean_final_wealth_chf'])+'</td><td>CHF '+f(outcomes['habits']['mean_final_wealth_chf'])+'</td><td>CHF '+f(q['mean'])+' ('+f(q['lower'])+' to '+f(q['upper'])+')</td></tr></table></div><p class="muted">The goal is '+e(person['high_goal'])+' (CHF '+f(person['high_target_chf'])+'), evaluated in December 2029 under the simulator’s credit-breach rule.</p><h3>World stress with the plan held fixed</h3><div class="table"><table><tr><th>Offer probability multiplier</th><th>Ever negative cash</th><th>Change vs plan baseline · 95% interval</th><th>Mean final wealth change · 95% interval</th></tr>')
    for x in outcomes['offer_stress']:
        z=x['difference_vs_plan_baseline'];q=z['shortfall'];w=z['mean_wealth_chf'];blocks.append('<tr><td>×'+str(x['offer_factor'])+'</td><td>'+f(x['summary']['shortfall_pct'])+'%</td><td>'+f(q['change_pp'])+' pp ('+f(q['lower_pp'])+' to '+f(q['upper_pp'])+')</td><td>CHF '+f(w['mean'])+' ('+f(w['lower'])+' to '+f(w['upper'])+')</td></tr>')
    blocks.append('</table></div><p class="muted">This changes the simulated environment without asking the LLM to generate a new plan.</p></section><section class="card"><h2>What this does and does not show</h2><ul>')
    blocks.extend('<li>'+e(x)+'</li>' for x in r['limitations']);blocks.append('</ul></section></main></body></html>')
    return ''.join(blocks)


def render_live(result):
    """Self-contained report focused on the repeated live planner experiment."""
    live=result['live_planner_experiment']
    if live is None: return None
    e=lambda x:html.escape(str(x),quote=True)
    signed=[sum(bool(p['signed']) for p in g['plans']) for g in live['groups']]
    body=['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Live LLM planner comparison</title><style>body{font:16px/1.6 system-ui,sans-serif;color:#19324a;background:#f3f7f6;margin:0}main{max-width:1050px;margin:auto;padding:35px 20px}.card{background:white;border:1px solid #dce8e7;border-radius:12px;padding:24px;margin:18px 0}h1{font-size:2.2rem;line-height:1.2}h2{font-size:1.4rem}p,li{color:#526b7e}.note{background:#edf7f5;border-left:3px solid #16757b;padding:12px 16px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;min-width:600px}th,td{text-align:left;border-bottom:1px solid #e3eaea;padding:10px}th{color:#526b7e}small{color:#526b7e}</style></head><body><main><h1>How does starting cash change the LLM’s plans?</h1>',
       '<p>Live planner experiment · '+e(result['person']['name'])+' · '+e(live['model'])+'</p><section class="card"><h2>Experiment</h2><p>Repeated drafts at each cash setting, with the same profile and planning wishes. Each draft can include simulation feedback and revision. The saved Mia plan in the other report is not one of these drafts.</p>',
       '<p><strong>Wishes:</strong> '+e(live.get('wishes','Not recorded'))+'</p><p><strong>Cash settings:</strong> CHF '+f"{live['groups'][0]['cash_chf']:,.0f}"+' and CHF '+f"{live['groups'][1]['cash_chf']:,.0f}"+' · <strong>Samples:</strong> '+str(live['samples_per_setting'])+' per setting · <strong>Simulation:</strong> '+str(live.get('simulation_paths_per_draft','not recorded'))+' paths per draft, same seed '+e(live.get('simulation_seed','not recorded'))+'.</p>',
       '<p>Planner marked '+str(signed[0])+'/'+str(live['samples_per_setting'])+' low-cash drafts and '+str(signed[1])+'/'+str(live['samples_per_setting'])+' baseline-cash drafts ready after feedback.</p>',
       '<p class="note"><strong>Sampling uncertainty:</strong> small sample sizes give wide uncertainty intervals. Counts describe whether a step appears anywhere in a plan; a conditional fallback counts even when it is unlikely to occur. These are not probabilities of actions taken or evidence of causation.</p></section><section class="card"><h2>Plan-step presence</h2><p>The difference is baseline-cash frequency minus low-cash frequency. The conservative interval describes finite-sampling uncertainty under the planner settings; it does not cover model or prompt uncertainty.</p><div class="scroll"><table><tr><th>Plan step</th><th>Low cash</th><th>Baseline cash</th><th>Difference</th><th>95% interval</th></tr>']
    n=live['samples_per_setting']
    for option,q in live['presence_effects'].items():
        cs=[g['step_presence_frequency'].get(option,{}).get('count',0) for g in live['groups']]
        body.append('<tr><td>'+e(option.replace(':',' / ').replace('_',' '))+'</td><td>'+str(cs[0])+'/'+str(n)+'</td><td>'+str(cs[1])+'/'+str(n)+'</td><td>'+f"{q['change_pp']:+.1f}"+' pp</td><td>'+f"{q['lower_pp']:+.1f}"+' to '+f"{q['upper_pp']:+.1f}"+' pp</td></tr>')
    body.append('</table></div><p class="note">Intervals are calculated for each step separately; they are not adjusted for examining multiple steps. Identical counts do not prove equal underlying probabilities.</p></section><section class="card"><h2>What was sampled?</h2><p>The same model and wishes were used at both cash values. The comparison changes starting cash in the planner’s prompt and in the simulator feedback. The model may choose different timing, conditions and commitments even when a step’s action type appears in both groups.</p><p>The output does not establish what a real person would choose. Results may change with the model, prompt, reference scenario or simulation assumptions.</p></section></main></body></html>')
    return ''.join(body)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',default='example',help='example (Mia), or synthetic profile ID')
    p.add_argument('--plan-file',type=Path,help='Plan history JSON (default bundled example or saved profile plan)')
    p.add_argument('--version',type=int,default=0,help='1-based full plan version; default latest')
    p.add_argument('--paths',type=int,default=1000);p.add_argument('--seed',type=int,default=20261122)
    p.add_argument('--low-cash',type=float,default=3000);p.add_argument('--live-samples',type=int,default=0)
    p.add_argument('--wishes',default='',help='Identical planning wishes for optional live experiments')
    p.add_argument('--out',type=Path,default=OUT)
    args=p.parse_args()
    if args.paths<2 or not np.isfinite(args.low_cash) or args.low_cash<0 or args.live_samples<0: p.error('At least two paths; nonnegative finite low cash; nonnegative live-samples required')
    if args.live_samples and not llm_provider.ready(): p.error(llm_provider.SETUP)
    if args.live_samples == 1: p.error('Use at least 2 live samples per setting')
    torch.set_num_threads(1)
    # Read latest by default, without accepting a mismatched custom plan silently.
    try:
        path=args.plan_file or (ROOT/'web/src/example-plan.json' if args.profile=='example' else C.ARTIFACTS/'plans'/f'{args.profile}.json')
        raw=json.loads(path.read_text()); versions=raw.get('history') or raw.get('versions') or ([raw['latest']] if raw.get('latest') else [])
        valid=[x for x in versions if isinstance(x,dict) and isinstance(x.get('plan'),dict)]
        version=args.version or len(valid)
        people,persona,plan,entry,entries,prov=read_input(args.profile,path,version)
    except (FileNotFoundError,KeyError,ValueError,json.JSONDecodeError) as exc: p.error(str(exc))
    res=analyse(people,persona,plan,entry,entries,prov,args.paths,args.seed,args.low_cash,args.live_samples,args.wishes)
    args.out.mkdir(parents=True,exist_ok=True)
    name='mia' if args.profile=='example' else args.profile
    (args.out/f'{name}_explanation.json').write_text(json.dumps(res,indent=2,allow_nan=False)+'\n')
    (args.out/f'{name}_explanation.html').write_text(render(res))
    if res['live_planner_experiment']:
        (args.out/f'{name}_live_planner.html').write_text(render_live(res))
    print('Report:',args.out/f'{name}_explanation.html',flush=True)

if __name__=='__main__': main()
