import { useEffect, useRef, useState } from 'react';
import type { Evidence } from './types';
import { money } from './types';

export default function Chart({ evidence }: { evidence: Evidence }) {
  const [measure, setMeasure] = useState<'fan_net_worth' | 'fan_cash'>('fan_cash');
  const [month, setMonth] = useState(60);
  const chartRef = useRef<SVGSVGElement>(null);
  const [width, setWidth] = useState(720);
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(chartRef.current!);
    return () => observer.disconnect();
  }, []);
  const fan = evidence.baseline[measure];
  const alternative = evidence.alternative?.[measure][2];
  const showGoal = (evidence.person.high_goal === 'Emergency savings') === (measure === 'fan_cash');
  const values = [...fan[0], ...fan[4], ...(alternative ?? []), ...(showGoal ? [evidence.person.high_target] : [])];
  const low = Math.min(0, ...values), high = Math.max(1, ...values) * 1.08;
  const height = 265, left = 54, right = 30, top = 20, bottom = 32;
  const x = (i: number) => left + (i / (fan[0].length - 1)) * (width - left - right);
  const y = (v: number) => top + ((high - v) / (high - low)) * (height - top - bottom);
  const line = (arr: number[]) => arr.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const area = (a: number[], b: number[]) => `${line(a)} ${b.map((v, i) => `L${x(b.length - 1 - i).toFixed(1)},${y(b[b.length - 1 - i]).toFixed(1)}`).join(' ')} Z`;
  const label = month === 120 ? 'End of 2034' : `January ${2025 + month / 12}`;
  return <section className="card projection-card">
    <div className="section-heading"><div><span className="eyebrow">THE BIGGER PICTURE</span><h2>More than one possible future</h2></div>
      <div className="segmented" aria-label="Chart measure">{(['fan_cash', 'fan_net_worth'] as const).map(key => <button key={key} aria-pressed={measure === key} onClick={() => setMeasure(key)}>{key === 'fan_cash' ? 'Cash' : 'Net worth'}</button>)}</div>
    </div>
    <p className="muted chart-intro">A range of possibilities, with room for life to happen.</p>
    <div className="chart-readout"><span>{label}</span><strong>CHF {money(fan[2][month])}</strong><span>current path median</span>{alternative && <span className="alt-readout">CHF {money(alternative[month])} with your choice</span>}</div>
    <svg ref={chartRef} className="fan-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Possible ${measure === 'fan_cash' ? 'cash' : 'net worth'} from January 2025 to December 2034. Shaded ranges contain 50% and 90% of simulated paths. Use the year control below to inspect figures.`}>
      {[0, 1, 2, 3].map(i => { const val = low + (high - low) * i / 3; return <g key={i}><line className="gridline" x1={left} x2={width - right} y1={y(val)} y2={y(val)} /><text x={left - 10} y={y(val) + 4} textAnchor="end">{new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 0 }).format(val)}</text></g>; })}
      <path d={area(fan[0], fan[4])} fill="#e5eeea" />
      <path d={area(fan[1], fan[3])} fill="#c0d8cd" />
      {showGoal && <g><line x1={left} x2={x(60)} y1={y(evidence.person.high_target)} y2={y(evidence.person.high_target)} stroke="#87918a" strokeDasharray="3 5" /><circle cx={x(60)} cy={y(evidence.person.high_target)} r="4" fill="#7c887b" /><text x={left + 6} y={y(evidence.person.high_target) - 8}>Goal · CHF {money(evidence.person.high_target)}</text></g>}
      <path d={line(fan[2])} stroke="#285c50" strokeWidth="2.8" fill="none" />
      {alternative && <path d={line(alternative)} stroke="#b67836" strokeWidth="2.5" strokeDasharray="7 5" fill="none" />}
      <line x1={x(month)} x2={x(month)} y1={top} y2={height - bottom} stroke="#8a9d93" strokeDasharray="3 5" />
      <circle cx={x(month)} cy={y(fan[2][month])} r="5" fill="#285c50" stroke="white" strokeWidth="2" />
      {(width < 460 ? [0, 60, 120] : [0, 24, 48, 72, 96, 120]).map(i => <text key={i} x={x(i)} y={height - 8} textAnchor="middle">{i === 120 ? '2034' : 2025 + i / 12}</text>)}
    </svg>
    <div className="chart-control"><label htmlFor="chart-year">Explore a year</label><input id="chart-year" type="range" min="0" max="120" step="12" value={month} onChange={e => setMonth(Number(e.target.value))} aria-valuetext={label} /><span>{month === 120 ? 'End 2034' : 2025 + month / 12}</span></div>
    <div className="chart-legend"><span><i className="legend-line" />Current path</span>{alternative && <span><i className="legend-line alternative" />Your choice</span>}<span><i className="legend-band" />Middle 50% / 90% of paths</span></div>
    <p className="chart-note">At this point, 90% of current-path simulations fall between CHF {money(fan[0][month])} and CHF {money(fan[4][month])}. These are simulated ranges, not guarantees.</p>
  </section>;
}
