import { useEffect, useRef, useState } from 'react';
import type { Measure, PathNode } from './types';
import { money } from './types';

const axis = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 0 });

// Shown under the chart for the selected measure (definitions in docs/model-assumptions.md §11–12).
const MEASURES: Record<Measure, { label: string; note: string }> = {
  fan_net_worth: { label: 'Net worth', note: 'Net worth is everything owned minus everything owed: cash + investments − debt.' },
  fan_cash: { label: 'Cash', note: 'Cash is the money in the bank account: income left after rent, bills and spending. '
    + 'Money moved into investments leaves cash. Cash can go below zero as an overdraft at 8% a year, down to the credit limit '
    + '(3 months of income, at least CHF 5,000); when it does, investments are sold to top it back up.' },
};

type Props = {
  nodes: PathNode[];                       // every path in the tree (roots first)
  activeId: string;                        // the highlighted path: shaded range, decision points
  decisionMonths: number[];                // January reviews where the highlighted path can branch
  pick: number | null;                     // the review currently chosen for a new branch
  onPick: (month: number) => void;
  goal: { target: number; cashGoal: boolean };
  measure: Measure;                        // controlled, so the assistant can switch it
  onMeasure: (m: Measure) => void;
};

export default function Chart({ nodes, activeId, decisionMonths, pick, onPick, goal, measure, onMeasure }: Props) {
  const [month, setMonth] = useState(60);
  const plotRef = useRef<HTMLDivElement>(null);
  // The plot's box sets its size (CSS: fixed on narrow screens, the pane's height beside the text).
  const [{ width, height }, setSize] = useState({ width: 720, height: 250 });
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) => setSize({ width: entry.contentRect.width, height: Math.max(180, entry.contentRect.height) }));
    observer.observe(plotRef.current!);
    return () => observer.disconnect();
  }, []);
  const active = nodes.find(n => n.id === activeId) ?? nodes[0];
  const parent = nodes.find(n => n.id === active.parent);
  const fan = active.summary[measure];
  const median = (n: PathNode) => n.summary[measure][2];
  const showGoal = goal.cashGoal === (measure === 'fan_cash');
  const values = [...fan[0], ...fan[4], ...nodes.flatMap(median), ...(showGoal ? [goal.target] : [])];
  const low = Math.min(0, ...values), high = Math.max(1, ...values) * 1.05;
  const left = 40, right = 10, top = 10, bottom = 26;
  const last = fan[0].length - 1;
  const x = (i: number) => left + (i / last) * (width - left - right);
  const y = (v: number) => top + ((high - v) / (high - low)) * (height - top - bottom);
  const line = (arr: number[], from = 0) => arr.slice(from).map((v, i) => `${i ? 'L' : 'M'}${x(from + i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const area = (a: number[], b: number[]) => `${line(a)} ${b.map((_, i) => `L${x(b.length - 1 - i).toFixed(1)},${y(b[b.length - 1 - i]).toFixed(1)}`).join(' ')} Z`;
  const label = month === 120 ? 'End of 2034' : `Jan ${2025 + month / 12}`;
  const scrub = (clientX: number) => {
    const r = plotRef.current!.getBoundingClientRect();
    const t = Math.min(1, Math.max(0, (clientX - r.left - left) / (width - left - right)));
    setMonth(Math.round(t * 10) * 12);
  };
  const others = nodes.filter(n => n.id !== active.id);
  return <div className="chart">
    <div className="chart-head">
      <p className="readout"><span>{label}</span><b>CHF {money(fan[2][month])}</b><span style={{ color: active.color }}>{active.label}</span>
        {parent && month >= active.from && <span className="vs">vs CHF {money(median(parent)[month])} without it</span>}</p>
      <div className="toggle" role="group" aria-label="Chart measure">{(['fan_net_worth', 'fan_cash'] as const).map(key => <button key={key} aria-pressed={measure === key} title={MEASURES[key].note} onClick={() => onMeasure(key)}>{MEASURES[key].label}</button>)}</div>
    </div>
    <div className="plot" ref={plotRef} onPointerMove={e => scrub(e.clientX)}>
      <svg viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
        {[0, 1, 2].map(i => { const val = low + (high - low) * i / 2; return <g key={i}><line className="grid" x1={left} x2={width - right} y1={y(val)} y2={y(val)} /><text x={left - 8} y={y(val) + 4} textAnchor="end">{axis.format(val)}</text></g>; })}
        <path className="band-outer" d={area(fan[0], fan[4])} style={{ fill: active.color }} />
        <path className="band-inner" d={area(fan[1], fan[3])} style={{ fill: active.color }} />
        {showGoal && <line className="goal" x1={left} x2={x(60)} y1={y(goal.target)} y2={y(goal.target)} />}
        {others.map(n => <path key={n.id} className="branch-line" d={line(median(n), n.from)} style={{ stroke: n.color }} />)}
        <path className="median" d={line(fan[2])} style={{ stroke: active.color }} />
        {nodes.filter(n => n.parent).map(n => <circle key={n.id} className="fork" cx={x(n.from)} cy={y(median(n)[n.from])} r={n.id === active.id ? 5 : 3.5} style={{ fill: n.color }} />)}
        <line className="cursor" x1={x(month)} x2={x(month)} y1={top} y2={height - bottom} />
        <circle className="cursor-dot" cx={x(month)} cy={y(fan[2][month])} r="4" style={{ fill: active.color }} />
        {(width < 460 ? [0, 60, 120] : [0, 24, 48, 72, 96, 120]).map(i => <text key={i} x={x(i)} y={height - 6} textAnchor={i === 0 ? 'start' : i === 120 ? 'end' : 'middle'}>{i === 120 ? '2034' : 2025 + i / 12}</text>)}
      </svg>
      <input className="scrub" type="range" min="0" max="120" step="12" value={month} onChange={e => setMonth(Number(e.target.value))}
        aria-label="Explore a year" aria-valuetext={`${label}: median CHF ${money(fan[2][month])}`} />
      {decisionMonths.map(m => <button key={m} type="button" className={`decide ${pick === m ? 'picked' : ''}`}
        style={{ left: x(m), top: y(fan[2][m]), borderColor: active.color }} aria-pressed={pick === m}
        aria-label={`Decide in January ${2025 + m / 12}`} title={`Decide something new in January ${2025 + m / 12}`}
        onClick={() => onPick(m)} />)}
    </div>
    <p className="legend"><span><i className="key-dot" />January review: tap to decide</span><span><i className="key-band" style={{ background: active.color }} />Likely range</span>{showGoal && <span><i className="key-line goal" />Goal</span>}</p>
    <p className="measure-note">{MEASURES[measure].note}</p>
  </div>;
}
