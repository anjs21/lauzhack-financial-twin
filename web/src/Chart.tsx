import { useEffect, useRef, useState } from 'react';
import type { Evidence } from './types';
import { money } from './types';

const axis = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 0 });

export default function Chart({ evidence }: { evidence: Evidence }) {
  const [measure, setMeasure] = useState<'fan_net_worth' | 'fan_cash'>('fan_cash');
  const [month, setMonth] = useState(60);
  const plotRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(720);
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(plotRef.current!);
    return () => observer.disconnect();
  }, []);
  const fan = evidence.baseline[measure];
  const alternative = evidence.alternative?.[measure][2];
  const showGoal = (evidence.person.high_goal === 'Emergency savings') === (measure === 'fan_cash');
  const values = [...fan[0], ...fan[4], ...(alternative ?? []), ...(showGoal ? [evidence.person.high_target] : [])];
  const low = Math.min(0, ...values), high = Math.max(1, ...values) * 1.05;
  const height = 240, left = 40, right = 8, top = 8, bottom = 26;
  const x = (i: number) => left + (i / (fan[0].length - 1)) * (width - left - right);
  const y = (v: number) => top + ((high - v) / (high - low)) * (height - top - bottom);
  const line = (arr: number[]) => arr.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const area = (a: number[], b: number[]) => `${line(a)} ${b.map((_, i) => `L${x(b.length - 1 - i).toFixed(1)},${y(b[b.length - 1 - i]).toFixed(1)}`).join(' ')} Z`;
  const label = month === 120 ? 'End of 2034' : `Jan ${2025 + month / 12}`;
  // Hovering anywhere on the plot snaps the readout to the nearest year.
  const scrub = (clientX: number) => {
    const r = plotRef.current!.getBoundingClientRect();
    const t = Math.min(1, Math.max(0, (clientX - r.left - left) / (width - left - right)));
    setMonth(Math.round(t * 10) * 12);
  };
  return <div className="chart">
    <div className="chart-head">
      <p className="readout"><span>{label}</span><b>CHF {money(fan[2][month])}</b>{alternative && <span className="alt">CHF {money(alternative[month])} with your choice</span>}</p>
      <div className="toggle" role="group" aria-label="Chart measure">{(['fan_cash', 'fan_net_worth'] as const).map(key => <button key={key} aria-pressed={measure === key} onClick={() => setMeasure(key)}>{key === 'fan_cash' ? 'Cash' : 'Net worth'}</button>)}</div>
    </div>
    <div className="plot" ref={plotRef} onPointerMove={e => scrub(e.clientX)}>
      <svg viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
        {[0, 1, 2].map(i => { const val = low + (high - low) * i / 2; return <g key={i}><line className="grid" x1={left} x2={width - right} y1={y(val)} y2={y(val)} /><text x={left - 8} y={y(val) + 4} textAnchor="end">{axis.format(val)}</text></g>; })}
        <path className="band-outer" d={area(fan[0], fan[4])} />
        <path className="band-inner" d={area(fan[1], fan[3])} />
        {showGoal && <line className="goal" x1={left} x2={x(60)} y1={y(evidence.person.high_target)} y2={y(evidence.person.high_target)} />}
        <path className="median" d={line(fan[2])} />
        {alternative && <path className="alt-line" d={line(alternative)} />}
        <line className="cursor" x1={x(month)} x2={x(month)} y1={top} y2={height - bottom} />
        <circle className="cursor-dot" cx={x(month)} cy={y(fan[2][month])} r="4.5" />
        {(width < 460 ? [0, 60, 120] : [0, 24, 48, 72, 96, 120]).map(i => <text key={i} x={x(i)} y={height - 6} textAnchor={i === 0 ? 'start' : i === 120 ? 'end' : 'middle'}>{i === 120 ? '2034' : 2025 + i / 12}</text>)}
      </svg>
      <input className="scrub" type="range" min="0" max="120" step="12" value={month} onChange={e => setMonth(Number(e.target.value))}
        aria-label="Explore a year" aria-valuetext={`${label}: median CHF ${money(fan[2][month])}`} />
    </div>
    <div className="legend">
      <span><i className="key-line" />Current path</span>
      {alternative && <span><i className="key-line alt" />Your choice</span>}
      <span><i className="key-band" />Likely range</span>
      {showGoal && <span><i className="key-line goal" />Goal</span>}
    </div>
  </div>;
}
