"""One person's explorer report as a standalone HTML page (charts drawn in the browser)."""
from __future__ import annotations

import json

from . import config as C

CSS = r"""
:root{color-scheme:light;--bg:#f9f9f7;--surface:#fcfcfb;--fg:#0b0b0b;--fg2:#52514e;--muted:#6f6d68;--grid:#e1e0d9;--axis:#c3c2b7;
--line:rgba(11,11,11,.10);--hover:rgba(11,11,11,.04);--accent:#1c5cab;--tint:#eef4fc;
--band-outer:#cde2fb;--band-inner:#86b6ef;--median:#1c5cab;--alt:#eb6834;--bar:#2a78d6;
--m0:#2a78d6;--m1:#eb6834;--m2:#1baf7a;--m3:#eda100;--good:#0ca30c;--critical:#d03b3b;
--font-display:"Instrument Sans","Inter",system-ui,-apple-system,"Segoe UI",sans-serif;
--font-body:"Inter",system-ui,-apple-system,"Segoe UI",sans-serif;--font-mono:"JetBrains Mono",ui-monospace,Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#0d0d0d;--surface:#1a1a19;--fg:#f0efec;--fg2:#c3c2b7;
--muted:#9b9a93;--grid:#2c2c2a;--axis:#4a4a46;--line:rgba(255,255,255,.10);--hover:rgba(255,255,255,.05);--accent:#86b6ef;--tint:#14243a;
--band-outer:#0d366b;--band-inner:#184f95;--median:#86b6ef;--alt:#d95926;--bar:#3987e5;--m0:#3987e5;--m1:#d95926;--m2:#199e70;--m3:#c98500}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#0d0d0d;--surface:#1a1a19;--fg:#f0efec;--fg2:#c3c2b7;--muted:#9b9a93;--grid:#2c2c2a;--axis:#4a4a46;
--line:rgba(255,255,255,.10);--hover:rgba(255,255,255,.05);--accent:#86b6ef;--tint:#14243a;--band-outer:#0d366b;--band-inner:#184f95;
--median:#86b6ef;--alt:#d95926;--bar:#3987e5;--m0:#3987e5;--m1:#d95926;--m2:#199e70;--m3:#c98500}
*{box-sizing:border-box}[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--fg);font:400 15px/1.6 var(--font-body)}
.page{max-width:1000px;margin:0 auto;padding:40px max(16px,env(safe-area-inset-right,0px)) 60px max(16px,env(safe-area-inset-left,0px))}
h1,h2,h3{font-family:var(--font-display);margin:0;text-wrap:balance}
h1{font-size:clamp(1.9rem,5.5vw,2.6rem);line-height:1.08;letter-spacing:-.02em}h2{font-size:1.35rem;line-height:1.25}
p{margin:0;max-width:72ch}.eyebrow{font:500 .72rem/1.3 var(--font-mono);text-transform:uppercase;letter-spacing:.09em;color:var(--fg2)}
.small{font-size:.86rem;color:var(--fg2)}.num{font-variant-numeric:tabular-nums}
header{display:grid;gap:14px;padding-bottom:28px}
section{display:grid;gap:14px;padding:30px 0;border-top:1px solid var(--line);min-width:0}section>*{min-width:0}
.tiles{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px 16px;display:grid;gap:4px;align-content:start}
.tile .v{font:600 1.75rem/1.1 var(--font-body)}.tile .l{font-size:.84rem;color:var(--fg2);line-height:1.4}
.facts{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:.88rem;color:var(--fg2)}.facts b{color:var(--fg);font-weight:600}
.card{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px 16px}
.tools{display:flex;flex-wrap:wrap;gap:8px;align-items:center;justify-content:space-between}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}
.seg button{font:500 .82rem/1 var(--font-body);color:var(--fg2);background:transparent;border:0;padding:8px 12px;cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--tint);color:var(--accent)}
.seg button:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:.82rem;color:var(--fg2)}.legend span{display:inline-flex;gap:6px;align-items:center}
.sw{width:12px;height:12px;border-radius:3px;display:inline-block}
.chart svg{display:block;width:100%;height:auto;overflow:visible}
.chart text{font:400 11px var(--font-body);fill:var(--muted)}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:.87rem}th,td{padding:8px 12px;text-align:left;vertical-align:top;border-bottom:1px solid var(--grid)}
thead th{font:600 .72rem/1.3 var(--font-body);color:var(--fg2);text-transform:uppercase;letter-spacing:.05em;white-space:nowrap}
tbody tr:last-child td{border-bottom:0}td.r,th.r{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.bars{display:grid;gap:8px}.barrow{display:grid;grid-template-columns:15rem minmax(0,1fr) 6rem;gap:12px;align-items:center;font-size:.88rem}
.track{height:10px;border-radius:999px;background:var(--grid);overflow:hidden}.fill{height:100%;border-radius:999px;background:var(--bar)}
.tt{position:fixed;z-index:9;pointer-events:none;background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:8px 10px;
box-shadow:0 6px 24px rgba(0,0,0,.16);font-size:.8rem;line-height:1.4;display:grid;gap:2px;min-width:150px}
.tt b{font-family:var(--font-display)}.tt .row{display:flex;justify-content:space-between;gap:12px}.tt .row span:first-child{color:var(--fg2)}
footer{border-top:1px solid var(--line);padding-top:18px;font-size:.8rem;color:var(--fg2);display:grid;gap:6px}
@media (max-width:760px){.tiles{grid-template-columns:repeat(2,minmax(0,1fr))}.barrow{grid-template-columns:minmax(0,1fr) 5rem}.barrow .track{grid-column:1/-1;grid-row:2}}
"""

JS = r"""
const D=JSON.parse(document.getElementById('data').textContent);
const tt=document.getElementById('tt');const fmt=v=>(Math.abs(v)>=1e6?(v/1e6).toFixed(2)+'M':Math.abs(v)>=1e3?(v/1e3).toFixed(0)+'k':v.toFixed(0));
const chf=v=>'CHF '+Math.round(v).toLocaleString('en-US').replace(/,/g,"'");
const el=(t,a={},txt)=>{const e=document.createElementNS('http://www.w3.org/2000/svg',t);for(const k in a)e.setAttribute(k,a[k]);if(txt!=null)e.textContent=txt;return e};
function showTT(x,y,title,rows){tt.replaceChildren();const h=document.createElement('b');h.textContent=title;tt.append(h);
 for(const[k,v]of rows){const r=document.createElement('div');r.className='row';const a=document.createElement('span');a.textContent=k;const b=document.createElement('span');b.textContent=v;r.append(a,b);tt.append(r)}
 tt.hidden=false;const w=tt.offsetWidth,hh=tt.offsetHeight;tt.style.left=Math.min(window.innerWidth-w-8,Math.max(8,x+14))+'px';tt.style.top=Math.max(8,y-hh-10)+'px'}
function hideTT(){tt.hidden=true}
function fanChart(host,key){
 const W=920,H=330,L=64,R=16,T=12,B=30;const f=D[key];const n=f[0].length;
 const lo=Math.min(0,...f[0]),hi=Math.max(...f[4],key==='fan_net_worth'&&D.person.high_target?D.person.high_target:0);
 const x=i=>L+(W-L-R)*i/(n-1),y=v=>T+(H-T-B)*(1-(v-lo)/(hi-lo||1));
 const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':host.dataset.label});
 for(let k=0;k<=4;k++){const v=lo+(hi-lo)*k/4;svg.append(el('line',{x1:L,x2:W-R,y1:y(v),y2:y(v),stroke:'var(--grid)'}));svg.append(el('text',{x:L-8,y:y(v)+4,'text-anchor':'end'},fmt(v)))}
 if(lo<0)svg.append(el('line',{x1:L,x2:W-R,y1:y(0),y2:y(0),stroke:'var(--axis)'}));
 for(let yr=0;yr<=10;yr+=2){svg.append(el('text',{x:x(yr*12),y:H-8,'text-anchor':'middle'},String(2025+yr)))}
 const band=(a,b,fill)=>{let d='M'+x(0)+','+y(f[a][0]);for(let i=1;i<n;i++)d+='L'+x(i)+','+y(f[a][i]);for(let i=n-1;i>=0;i--)d+='L'+x(i)+','+y(f[b][i]);svg.append(el('path',{d:d+'Z',fill}))};
 band(0,4,'var(--band-outer)');band(1,3,'var(--band-inner)');
 let d='M'+x(0)+','+y(f[2][0]);for(let i=1;i<n;i++)d+='L'+x(i)+','+y(f[2][i]);svg.append(el('path',{d,fill:'none',stroke:'var(--median)','stroke-width':2}));
 if(key==='fan_net_worth'&&D.person.high_target){const gx=x(60),gy=y(D.person.high_target);
  svg.append(el('line',{x1:gx,x2:gx,y1:T,y2:H-B,stroke:'var(--axis)','stroke-dasharray':'4 4'}));
  svg.append(el('circle',{cx:gx,cy:gy,r:5,fill:'var(--surface)',stroke:'var(--fg)','stroke-width':2}));
  svg.append(el('text',{x:gx+8,y:gy-8},'high goal, Dec 2029'))}
 const cross=el('line',{y1:T,y2:H-B,stroke:'var(--fg2)','stroke-width':1,visibility:'hidden'});svg.append(cross);
 const hit=el('rect',{x:L,y:T,width:W-L-R,height:H-T-B,fill:'transparent'});svg.append(hit);
 hit.addEventListener('mousemove',e=>{const r=svg.getBoundingClientRect();const px=(e.clientX-r.left)/r.width*W;const i=Math.max(0,Math.min(n-1,Math.round((px-L)/((W-L-R)/(n-1)))));
  cross.setAttribute('x1',x(i));cross.setAttribute('x2',x(i));cross.setAttribute('visibility','visible');
  showTT(e.clientX,e.clientY,D.months[i],[['95th pct',chf(f[4][i])],['75th',chf(f[3][i])],['median',chf(f[2][i])],['25th',chf(f[1][i])],['5th',chf(f[0][i])]])});
 hit.addEventListener('mouseleave',()=>{cross.setAttribute('visibility','hidden');hideTT()});
 host.replaceChildren(svg)}
const fh=document.getElementById('fan');let fanKey='fan_net_worth';fanChart(fh,fanKey);
document.querySelectorAll('#fan-tabs button').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('#fan-tabs button').forEach(o=>o.setAttribute('aria-pressed',o===b));fanKey=b.dataset.key;fh.dataset.label=b.textContent+' fan chart';fanChart(fh,fanKey)}));
(function modes(){const host=document.getElementById('modes');const names=['full-time','part-time','studying','not working'];const cols=['var(--m0)','var(--m1)','var(--m2)','var(--m3)'];
 const W=920,H=220,L=40,B=26,T=8;const n=11;const bw=(W-L)/n;const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':'Work mode share by year'});
 for(const v of[0,.5,1]){const yy=T+(H-T-B)*(1-v);svg.append(el('line',{x1:L,x2:W,y1:yy,y2:yy,stroke:'var(--grid)'}));svg.append(el('text',{x:L-6,y:yy+4,'text-anchor':'end'},(v*100)+'%'))}
 for(let i=0;i<n;i++){let acc=0;const g=el('g');for(let m=0;m<4;m++){const v=D.mode_share_by_year[names[m]][i];if(v<=0)continue;const h=(H-T-B)*v;
  g.append(el('rect',{x:L+i*bw+4,y:T+(H-T-B)*(1-acc)-h+1,width:bw-8,height:Math.max(0,h-2),rx:2,fill:cols[m]}));acc+=v}
  g.append(el('text',{x:L+i*bw+bw/2,y:H-8,'text-anchor':'middle'},String(2025+i)));
  const hit=el('rect',{x:L+i*bw,y:T,width:bw,height:H-T-B,fill:'transparent'});
  hit.addEventListener('mousemove',e=>showTT(e.clientX,e.clientY,'Start of '+(2025+i),names.map(k=>[k,(D.mode_share_by_year[k][i]*100).toFixed(0)+'%'])));
  hit.addEventListener('mouseleave',hideTT);g.append(hit);svg.append(g)}
 host.replaceChildren(svg)})();
"""


def pct(x):
    return "—" if x is None else f"{100 * x:.0f}%"


def chf(x):
    return f"CHF {x:,.0f}".replace(",", "'")


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def describe(i: dict) -> str:
    """A stated plan in words."""
    words = {"status": C.STATUS_OPTIONS, "housing": {-1: "move somewhere cheaper", 0: "stay", 1: "move somewhere bigger"},
             "spending": {-1: "cut lifestyle", 0: "keep lifestyle", 1: "raise lifestyle"},
             "assets": {-1: "sell investments to the buffer", 0: "hold", 1: "invest above the buffer"}}
    what = words[i["head"]][i["option"]]
    when = [f"in {i['when']}"] if i.get("when") else []
    if i.get("when") and i["head"] in ("status", "housing"):
        when = [f"at the {C.month_label(C.next_review(C.month_index(i['when'])))} review"]
    when += [f"from age {i['min_age']:g}"] if i.get("min_age") is not None else []
    when += [f"if cash < {i['cash_months_below']:g} months"] if i.get("cash_months_below") is not None else []
    when += [f"if cash > {i['cash_months_above']:g} months"] if i.get("cash_months_above") is not None else []
    return f"{what} {' '.join(when)}, {100 * i.get('prob', 1):.0f}% likely".replace("  ", " ")


def render(res: dict, validation: dict | None = None) -> str:
    p = res["person"]
    name = p.get("name") or "This person"
    fw = res["final_net_worth"]
    data = {k: res[k] for k in ("months", "fan_net_worth", "fan_cash", "fan_income", "mode_share_by_year")}
    data["person"] = {"high_target": p.get("high_target")}
    rows_st = "".join(
        f"<tr><td>{esc(s['story'])}</td><td class='r'>{pct(s['share'])}</td><td class='r'>{s['typical_change_year'] or '—'}</td>"
        f"<td class='r'>{chf(s['median_final_net_worth'])}</td><td class='r'>{pct(s['high_goal_odds'])}</td></tr>" for s in res["stories"])
    rows_tp = "".join(
        f"<tr><td>{esc(t['feature'])}</td><td class='r'>{pct(t['share'])}</td><td class='r'>{chf(t['median_nw_with'])}</td>"
        f"<td class='r'>{chf(t['median_nw_without'])}</td><td class='r'>{pct(t['high_goal_with'])} vs {pct(t['high_goal_without'])}</td></tr>"
        for t in res["turning_points"])
    def interval(v):
        bounds = v.get("goal_change_mc95_pp")
        if bounds is None:
            return "—"
        return f"{v['goal_change_pp']:+.1f} ({bounds[0]:+.1f} to {bounds[1]:+.1f})"

    wi = ""
    if res.get("what_ifs"):
        rows = "".join(
            f"<tr><td>{esc(k)}</td><td class='r'>{pct(v['high_goal_odds'][1])} <span class='small'>({pct(v['high_goal_odds'][0])})</span></td>"
            f"<td class='r'>{interval(v)}</td><td class='r'>{pct(v['credit_breach_odds'][1])}</td><td class='r'>{pct(v['shortfall_odds'][1])} <span class='small'>({pct(v['shortfall_odds'][0])})</span></td>"
            f"<td class='r'>{chf(v['median_final_net_worth'][1])} <span class='small'>({chf(v['median_final_net_worth'][0])})</span></td></tr>"
            for k, v in res["what_ifs"].items())
        wi = (f"<section><div class='eyebrow'>What-ifs</div><h2>Change one plan, keep the same luck</h2>"
              f"<p class='small'>Each row adds one stated plan and reruns the same {res['n_paths']:,} paths with the same random numbers, "
              f"so the difference comes from the plan alone. Brackets: as is.</p><div class='tablewrap'><table><thead><tr><th>Plan</th>"
              f"<th class='r'>High goal</th><th class='r'>Goal change, pp (paired 95% sampling interval)</th><th class='r'>Credit breach</th><th class='r'>Ever below zero</th><th class='r'>Median net worth 2034</th></tr></thead>"
              f"<tbody>{rows}</tbody></table></div></section>")
    fs = ""
    if res.get("foresight"):
        rows = "".join(
            f"<tr><td>{'as they tend to decide' if float(k) == 0 else f'foresight {k}'}</td><td class='r'>{pct(v['high_goal_odds'])}</td>"
            f"<td class='r'>{pct(v['shortfall_odds'])}</td><td class='r'>{pct(v['lasting_shortfall_odds'])}</td>"
            f"<td class='r'>{chf(v['median_final_net_worth'])}</td><td class='r'>{chf(v['median_life_utility'])}</td>"
            f"<td>{esc(v['top_story']['story'])} <span class='small'>({pct(v['top_story']['share'])})</span></td></tr>"
            for k, v in res["foresight"].items())
        fs = (f"<section><div class='eyebrow'>Foresight</div><h2>{esc(name)} as they are, and with foresight</h2>"
              f"<p class='small'>The neural persona with a foresight knob. At 0 it decides as {esc(name)} tends to; higher values "
              f"use the world model to lean toward outcomes this persona values (lifestyle, home, free time, goals, net worth), "
              f"while staying close to their habits and still random. Same {res['n_paths']:,} paths and random numbers in every row; "
              f"the gap between rows is what foresight would change.</p><div class='tablewrap'><table><thead><tr><th>Who decides</th>"
              f"<th class='r'>High goal</th><th class='r'>Ever below zero</th><th class='r'>3+ months</th><th class='r'>Median net worth 2034</th>"
              f"<th class='r'>Median life utility</th><th>Most common story</th></tr></thead><tbody>{rows}</tbody></table></div></section>")
    sp = ""
    if res.get("spread_sources"):
        s = res["spread_sources"]
        full = max(s["all sources"], 1)
        bars = "".join(f"<div class='barrow'><span>{esc(k)}</span><div class='track'><div class='fill' style='width:{min(100, 100 * v / full):.0f}%'></div></div>"
                       f"<span class='num r'>{chf(v)}</span></div>" for k, v in s.items())
        sp = (f"<section><div class='eyebrow'>Where the spread comes from</div><h2>Width of the 10–90% range of net worth in 2034</h2>"
              f"<p class='small'>Each bar keeps one source of randomness on and switches the others off. A small difference "
              f"early on also changes the person's later choices, so the bars include those knock-on effects and do not add up.</p>"
              f"<div class='card bars'>{bars}</div></section>")
    traits = "".join(f"<tr><td>{esc(k.replace('_', ' '))}</td><td class='r'>{v:.2f}</td></tr>" for k, v in res["persona"]["traits"].items())
    intents = "".join(f"<li>{esc(describe(i))}</li>" for i in res["persona"]["intentions"]) or "<li>none stated</li>"
    val = ""
    if validation:
        v = validation["verdict_network"]
        val = ("World model checked against the data: " + ", ".join(f"{k.split('_')[0]} {'pass' if ok else 'FAIL'}" for k, ok in v.items())
               + f" (held-out CRPS {validation['held_out_people']['monthly_flow_crps_network']:.0f} vs baseline "
               f"{validation['held_out_people']['monthly_flow_crps_baseline']:.0f}).")
    med = "" if res["medium_goal_odds"] is None else (
        f"<div class='tile'><div class='v num'>{pct(res['medium_goal_odds'])}</div><div class='l'>medium goal ({esc(p.get('med_goal'))}) by Dec 2032</div></div>")
    opening = p['cash0'] if p['high_goal'] == "Emergency savings" else p['cash0'] + p['inv0'] - p['debt0']
    gap = max(p['high_target'] - opening, 0)
    lasting = res.get("lasting_shortfall_odds", 0)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(name)}'s Possible Lives</title><style>{CSS}</style></head><body><div class="page">
<header><div class="eyebrow">Financial Twin explorer · {res['n_paths']:,} sampled lives · seed {res['seed']}</div>
<h1>{esc(name)}'s possible lives, 2025–2034</h1>
<p>Each path is one way the next ten years could go: {esc(name)} decides the way their persona tends to, at random, and a world
model learned from the ledger decides what happens. The spread is the point: it shows what is likely, what is possible, and what drives it.</p>
<div class="facts"><span><b>{p['age']:.0f}</b> years</span><span><b>{esc(p['status']).replace('_', ' ')}</b></span><span><b>{esc(p['household_type']).replace('_', ' ')}</b> in {esc(p['city'])}</span>
<span>income <b>{chf(p['income0'])}</b></span><span>rent <b>{chf(p['rent0'])}</b></span><span>cash <b>{chf(p['cash0'])}</b></span><span>investments <b>{chf(p['inv0'])}</b></span>
<span>high goal <b>{esc(p['high_goal'])}</b>, {chf(p['high_target'])}</span></div>
<div class="tiles">
<div class="tile"><div class="v num">{pct(res['high_goal_odds'])}</div><div class="l">high goal met by Dec 2029</div></div>{med}
<div class="tile"><div class="v num">{pct(res['shortfall_odds'])}</div><div class="l">ever below zero cash; {pct(lasting)} for 3+ months in a row</div></div>
<div class="tile"><div class="v num">{chf(fw['p50'])}</div><div class="l">median net worth in Dec 2034 (5–95%: {chf(fw['p5'])} to {chf(fw['p95'])})</div></div>
</div><p>The opening gap to the 2029 goal is {chf(gap)}. Goal success requires meeting the target and staying within the credit limit through its deadline. Across the full ten years, {pct(res['credit_breach_odds'])} of paths exceed their credit limit. Work and housing choices are reviewed each January.</p></header>
<section><div class="tools"><div><div class="eyebrow">Fan chart</div><h2>How the money could move</h2></div>
<div class="seg" id="fan-tabs" role="group" aria-label="Measure"><button aria-pressed="true" data-key="fan_net_worth">Net worth</button><button aria-pressed="false" data-key="fan_cash">Cash</button><button aria-pressed="false" data-key="fan_income">Income</button></div></div>
<div class="legend"><span><i class="sw" style="background:var(--band-outer)"></i>5–95% of paths</span><span><i class="sw" style="background:var(--band-inner)"></i>25–75%</span><span><i class="sw" style="background:var(--median);height:3px"></i>median</span></div>
<div class="card chart" id="fan" data-label="Net worth fan chart"></div></section>
<section><div class="eyebrow">Work</div><h2>What {esc(name)} is doing, year by year</h2>
<div class="legend"><span><i class="sw" style="background:var(--m0)"></i>full-time</span><span><i class="sw" style="background:var(--m1)"></i>part-time</span><span><i class="sw" style="background:var(--m2)"></i>studying</span><span><i class="sw" style="background:var(--m3)"></i>not working</span></div>
<div class="card chart" id="modes"></div></section>
<section><div class="eyebrow">Life stories</div><h2>The most common ways it goes</h2>
<div class="tablewrap"><table><thead><tr><th>Story</th><th class="r">Share</th><th class="r">From</th><th class="r">Median net worth 2034</th><th class="r">High goal</th></tr></thead><tbody>{rows_st}</tbody></table></div></section>
<section><div class="eyebrow">Turning points</div><h2>What separates the better paths from the worse</h2>
<div class="tablewrap"><table><thead><tr><th>On the path</th><th class="r">Share</th><th class="r">Median net worth with</th><th class="r">without</th><th class="r">High goal with vs without</th></tr></thead><tbody>{rows_tp}</tbody></table></div>
<p class="small">These are associations across sampled lives, not causes: paths that go part-time also tend to be the ones under pressure.</p></section>
{fs}{wi}{sp}
<section><div class="eyebrow">Persona</div><h2>How {esc(name)} tends to decide</h2>
<div class="tablewrap"><table><thead><tr><th>Trait</th><th class="r">Value (0–1)</th></tr></thead><tbody>{traits}</tbody></table></div>
<p class="small">Temperature {res['persona']['temperature']:.2f}; trait uncertainty {res['persona']['spread']:.2f} (logit sd). Stated plans:</p><ul class="small">{intents}</ul></section>
<footer><div>{esc(val)}</div><div>Synthetic data only. This explores possibilities under stated assumptions; it is not a forecast or financial advice.</div></footer>
</div><div class="tt" id="tt" hidden></div><script type="application/json" id="data">{json.dumps(data).replace('</', '<\\/')}</script><script>{JS}</script></body></html>"""


def write(res: dict, path, validation_path=C.ARTIFACTS / "validation.json"):
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else None
    path.write_text(render(res, validation))
    return path
