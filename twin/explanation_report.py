"""Focused interactive global and local explanation charts."""
from __future__ import annotations
import html
import json


def render(result):
    escape=lambda x: html.escape(str(x), quote=True)
    data=json.dumps(result['explanation_methods'], allow_nan=False).replace('<', '\\u003c')
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{escape(result['name'])}: model explanations</title><style>
:root{{color-scheme:light dark;--bg:#f3f6fa;--surface:#fff;--fg:#172639;--muted:#53657b;--border:#dce4ef;--accent:#245faf;--line:#d6dfeb}}@media(prefers-color-scheme:dark){{:root{{--bg:#101827;--surface:#172337;--fg:#edf3fb;--muted:#b0c0d4;--border:#33445e;--accent:#8bb9ff;--line:#40516c}}}}
*{{box-sizing:border-box}}body{{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif;margin:0}}main{{max-width:1180px;margin:auto;padding:36px 24px 64px}}h1,h2{{line-height:1.2}}h1{{font-size:2rem;margin:8px 0}}h2{{font-size:1.2rem;margin:0}}p{{margin:8px 0}}.muted{{color:var(--muted)}}.eyebrow{{color:var(--accent);font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em}}.toolbar{{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0}}label{{display:grid;gap:4px;color:var(--muted);font-size:.85rem}}select{{padding:9px 35px 9px 10px;background:var(--surface);color:var(--fg);border:1px solid var(--border);border-radius:7px;font:inherit}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}section{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px;min-width:0}}.chart{{width:100%;min-height:230px}}svg{{display:block;width:100%;height:auto}}svg text{{fill:var(--muted);font:11px system-ui,sans-serif}}svg .axis{{stroke:var(--line)}}.legend{{font-size:.83rem;color:var(--muted)}}.result{{font-variant-numeric:tabular-nums;font-weight:600}}details{{margin-top:20px}}@media(max-width:760px){{.grid{{grid-template-columns:1fr}}main{{padding:24px 14px}}}}
</style></head><body><main><header><div class="eyebrow">Financial twin · explanation explorer</div><h1>{escape(result['name'])}: model explanations</h1><p class="muted">Select a feature to update the global and local explanations.</p></header>
<div class="toolbar"><label for="feature">Feature<select id="feature"><option value="income">Income</option><option value="rent">Rent</option><option value="employment">Employment status</option><option value="city">City</option><option value="season">Calendar month</option></select></label></div>
<div class="grid">
<section><div class="eyebrow">Global · observed contexts</div><h2>Partial dependence</h2><p class="muted">Average predicted monthly flow as the selected feature varies.</p><div id="pdp" class="chart"></div><p class="legend">Blue: 2025 contexts · orange: 2026 contexts. Shading or whiskers: min–max across fitted networks, not a confidence interval. Continuous inputs use CHF/month; categories are discrete. Flow is CHF/month in 2025 prices.</p></section>
<section><div class="eyebrow">Local · simulated horizon</div><h2>Shapley contribution over time</h2><p class="muted">Annual expected-flow contribution relative to the feature reference.</p><div id="shapley" class="chart"></div><p class="legend">Shading and whiskers: min–max across five fitted networks, not a confidence interval. Units: CHF/year. Hover, tap, or focus a point to see its exact range. One selected simulated path.</p></section>
<section><div class="eyebrow">Local · simulated horizon</div><h2>LIME weight over time</h2><p class="muted">Annual local surrogate weight for the selected input.</p><div id="lime" class="chart"></div><p class="legend">Shading and whiskers: range across five perturbation seeds, not a confidence interval. Income and rent: CHF per neighborhood standard deviation; categorical features: CHF for the path/reference match. Hover, tap, or focus a point for its exact range and local R².</p></section>
</div><details><summary>How to interpret these plots</summary><p>PDPs average predictions over observed profile-months; holding other inputs fixed may create implausible combinations. The selected local path is closest to the median terminal net worth and is not necessarily typical in every year. PDP curves compare the observed 2025 and 2026 contexts. The PDP is not a ten-year forecast.</p><p>Income and rent use a two-feature Shapley game with mode-specific medians. Employment, city, and month use separate one-feature Shapley comparisons while other inputs remain fixed; those categorical contributions cannot be summed with the income/rent values. Employment compares the path with the initial status, city with the most common profile city, and each month with January. Categorical LIME uses only these two settings, so a high local R² is expected and does not validate the model. LIME approximates expected flow near that path; its weights use a different scale and should not be compared directly with Shapley. Local results hold the later path fixed and exclude personal residuals and other cash movements.</p><p>Model-member and perturbation ranges show limited sensitivity, not predictive confidence intervals. The model was trained on synthetic data; these plots show model behaviour rather than established causal effects.</p><p>The retained JSON output contains simulation summaries, including uncertainty about simulated outcomes. Negative-balance depth is not an estimate of the initial funding required.</p></details>
</main><script id="data" type="application/json">{data}</script><script>{JS}</script></body></html>'''


JS=r'''
const D=JSON.parse(document.getElementById('data').textContent),ns='http://www.w3.org/2000/svg';
const feature=document.getElementById('feature');
function el(tag,attrs={{}},text){{const e=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e}}
function svg(label){{return el('svg',{{viewBox:'0 0 520 245',role:'img','aria-label':label}})}}
function label(v){{return Math.round(v).toLocaleString('en-US')}}
function precise(v){{return v.toLocaleString('en-US',{{minimumFractionDigits:2,maximumFractionDigits:2}})}}
function extent(vals){{let lo=Math.min(...vals),hi=Math.max(...vals);if(hi===lo){{lo-=1;hi+=1}}const pad=(hi-lo)*.1;return [lo-pad,hi+pad]}}
function lineChart(id,series,xName,yName){{
 const host=document.getElementById(id),out=svg(xName+' versus '+yName),xs=series.flatMap(s=>s.x),ys=series.flatMap(s=>[...s.low,...s.high]),[xmin,xmax]=extent(xs),[ymin,ymax]=extent(ys);
 const x=v=>65+(v-xmin)/(xmax-xmin)*430,y=v=>195-(v-ymin)/(ymax-ymin)*170;
 const detail=el('text',{{x:65,y:12,'font-size':10,'aria-live':'polite'}},'');out.append(detail);
 for(let i=0;i<=4;i++){{let xv=xmin+(xmax-xmin)*i/4,yv=ymin+(ymax-ymin)*i/4;out.append(el('line',{{x1:65,x2:495,y1:y(yv),y2:y(yv),class:'axis'}}));out.append(el('text',{{x:59,y:y(yv)+4,'text-anchor':'end'}},label(yv)));if(!xName.includes('year'))out.append(el('text',{{x:x(xv),y:213,'text-anchor':'middle'}},label(xv)))}}
 series.forEach(s=>{{let band=s.x.map((v,i)=>(i?'L':'M')+x(v)+','+y(s.low[i])).join('')+s.x.map((v,i)=>[x(v),y(s.high[i])]).reverse().map(a=>'L'+a.join(',')).join('')+'Z';out.append(el('path',{{d:band,fill:s.color,opacity:.15}}));let line=s.x.map((v,i)=>(i?'L':'M')+x(v)+','+y(s.y[i])).join('');out.append(el('path',{{d:line,fill:'none',stroke:s.color,'stroke-width':2.5}}));s.x.forEach((v,i)=>{{if(s.whiskers){{out.append(el('line',{{x1:x(v),x2:x(v),y1:y(s.low[i]),y2:y(s.high[i]),stroke:s.color,'stroke-width':1.5}}));for(const edge of [s.low[i],s.high[i]])out.append(el('line',{{x1:x(v)-4,x2:x(v)+4,y1:y(edge),y2:y(edge),stroke:s.color,'stroke-width':1.5}}))}}const message=s.name+' · '+label(v)+': '+precise(s.y[i])+' · range '+precise(s.low[i])+' to '+precise(s.high[i])+(s.scores?' · R² '+s.scores[i].toFixed(3):'');const point=el('circle',{{cx:x(v),cy:y(s.y[i]),r:4,fill:s.color,tabindex:0,role:'img','aria-label':message}});point.append(el('title',{{}},message));for(const event of ['mouseenter','focus','click'])point.addEventListener(event,()=>{{detail.textContent=message}});out.append(point)}})}});
 if(xName.includes('year'))series[0].x.forEach(v=>out.append(el('text',{{x:x(v),y:213,'text-anchor':'middle'}},String(v))));
 out.append(el('text',{{x:280,y:240,'text-anchor':'middle'}},xName));out.append(el('text',{{x:14,y:105,transform:'rotate(-90 14 105)','text-anchor':'middle'}},yName));host.replaceChildren(out)
}}
function categoryChart(curves, name){{
 const out=svg(name+' comparison of expected monthly flow'), xs=curves[0].x, labels=curves[0].labels, vals=curves.flatMap(c=>[...c.low,...c.high]), [lo,hi]=extent(vals), y=v=>195-(v-lo)/(hi-lo)*170, spacing=415/Math.max(xs.length,1), x=(i,j)=>75+(i+.5)*spacing+(j?3:-3);
 for(let j=0;j<=4;j++){{const v=lo+(hi-lo)*j/4;out.append(el('line',{{x1:65,x2:495,y1:y(v),y2:y(v),class:'axis'}}));out.append(el('text',{{x:59,y:y(v)+4,'text-anchor':'end'}},label(v)))}}
 curves.forEach((c,j)=>c.x.forEach((v,i)=>{{const pos=x(i,j);out.append(el('line',{{x1:pos,x2:pos,y1:y(c.low[i]),y2:y(c.high[i]),stroke:c.color,'stroke-width':2}}));const point=el('circle',{{cx:pos,cy:y(c.y[i]),r:3,fill:c.color}});point.append(el('title',{{}},c.name+' '+labels[i]+': CHF '+label(c.y[i])+'; range '+label(c.low[i])+' to '+label(c.high[i])));out.append(point)}}));
 labels.forEach((v,i)=>{{const pos=x(i,0);out.append(el('text',{{x:pos,y:208,transform:'rotate(-40 '+pos+' 208)','text-anchor':'end'}},v))}});document.getElementById('pdp').replaceChildren(out)
}}

function renderPDP(){{const f=feature.value;
 const curves=D.global_pdp.curves.filter(c=>c.feature===f).map((c,i)=>({{name:String(c.year),x:c.grid_chf,labels:c.category_labels,y:c.mean_monthly_flow_chf,low:c.member_min_chf,high:c.member_max_chf,color:i?'#d46b36':'#3273c3'}}));
 if(curves[0].labels)categoryChart(curves,f);else lineChart('pdp',curves,f+' · CHF/month','Expected flow · CHF/month');
}}
function updateLocal(){{const f=feature.value,rows=D.local;
 const local=rows.map(r=>{{const c=r.categorical[f],s=r.shapley,l=r.lime;return {{year:r.year,shap:c?c.shapley_chf:s[f+'_chf'],shapRange:c?c.shapley_member_range_chf:s['member_'+f+'_range_chf'],lime:c?c.lime_weight_chf:l.weights_chf_per_standard_deviation[f+'_multiplier'],limeRange:c?c.lime_weight_range_chf:l[f+'_weight_range_chf'],score:c?c.lime_r2:l.weighted_local_r2}}}});
 lineChart('shapley',[{{name:'Shapley',whiskers:true,x:local.map(x=>x.year),y:local.map(x=>x.shap),low:local.map(x=>x.shapRange[0]),high:local.map(x=>x.shapRange[1]),color:'#3273c3'}}],'Simulated year','Shapley · CHF/year');
 lineChart('lime',[{{name:'LIME',whiskers:true,x:local.map(x=>x.year),y:local.map(x=>x.lime),low:local.map(x=>x.limeRange[0]),high:local.map(x=>x.limeRange[1]),color:'#b55232',scores:local.map(x=>x.score)}}],'Simulated year',rows[0].categorical[f]?'LIME · CHF':'LIME · CHF per std');

}}
feature.addEventListener('change',()=>{{renderPDP();updateLocal()}});renderPDP();updateLocal();
'''.replace('{{','{').replace('}}','}')
