import { useState } from 'react';
import { LoaderCircle, PenLine, RotateCcw } from 'lucide-react';
import type { PlanStep, PlanVersion, StepResult } from './types';
import { compact, percent } from './types';

const EVENTS: Record<string, string> = {
  job_loss: 'if I lose my job', offer: 'once I get a job offer', graduated: 'once I finish studying',
  partner: 'if a partner moves in', child: 'if we have a child', separation: 'if we separate', windfall: 'if I receive a windfall',
};
const reviewYear = (month: string) => Number(month.slice(0, 4)) + (month.slice(5) === '01' ? 0 : 1);
const lives = (share: number) => share >= 0.995 ? 'every life' : `${percent(share)} of lives`;
const firmness = (c: number) => `${c >= 0.85 ? 'firm' : c >= 0.6 ? 'likely' : 'maybe'}, ${percent(c)} sure`;
export const contingent = (s: PlanStep) => s.event != null || s.cash_months_below != null;

function when(step: PlanStep, steps: PlanStep[]) {
  if (step.event) return EVENTS[step.event] ?? `after ${step.event}`;
  if (step.cash_months_below != null) return `if cash falls below ${step.cash_months_below} month${step.cash_months_below === 1 ? '' : 's'} of costs`;
  const parts: string[] = [];
  if (step.after) {
    const first = steps.find(s => s.id === step.after)?.label ?? step.after;
    const gap = step.after_months ? (step.after_months % 12 === 0 ? `${step.after_months / 12} year${step.after_months === 12 ? '' : 's'} ` : `${step.after_months} months `) : '';
    parts.push(`${gap}after “${first}”`);
  }
  const from = step.start ? reviewYear(step.start) : 2025;
  const to = step.until ? Number(step.until.slice(0, 4)) : step.start && !step.after && !step.repeat ? from : 2034;
  if (step.repeat) parts.push(`every year ${from}–${to}`);
  else if (!step.after || step.until || step.start) parts.push(from === to ? `in ${from}` : `${from}–${to}`);
  if (step.cash_months_above != null) parts.push(`with more than ${step.cash_months_above} months of costs in cash`);
  return parts.join(', ');
}

function flag(step: PlanStep, r: StepResult) {
  if (r.blocked_because && r.blocked_share >= 0.2) return `Not possible in ${percent(r.blocked_share)} of lives: ${r.blocked_because}`;
  if (r.turned_down_share >= 0.2) {
    const what = step.head === 'housing' ? 'no flat found' : step.option === 'start_masters' ? 'not admitted' : 'turned down';
    return `Tried, but ${what} in ${percent(r.turned_down_share)} of lives`;
  }
  return '';
}

function changesBetween(before: PlanVersion, after: PlanVersion) {
  const old = new Map(before.plan.steps.map(s => [s.id, s]));
  const now = new Map(after.plan.steps.map(s => [s.id, s]));
  const out: string[] = [];
  for (const [id, s] of old) if (!now.has(id)) out.push(`Dropped: ${s.label}`);
  for (const [id, s] of now) if (!old.has(id)) out.push(`Added: ${s.label}`);
  const groups: [string, (keyof PlanStep)[]][] = [['timing', ['start', 'until', 'after', 'after_months', 'repeat']], ['commitment', ['commitment']],
    ['trigger', ['event', 'within_months', 'cash_months_below', 'cash_months_above']], ['choice', ['head', 'option']]];
  for (const [id, s] of now) {
    const o = old.get(id);
    if (!o) continue;
    const what = groups.filter(([, keys]) => keys.some(k => o[k] !== s[k])).map(([name]) => name);
    if (what.length) out.push(`Changed ${what.join(' and ')}: ${s.label}`);
  }
  return out;
}

function Step({ step, result, steps }: { step: PlanStep; result?: StepResult; steps: PlanStep[] }) {
  const warning = result ? flag(step, result) : '';
  return <li className={contingent(step) ? 'fallback' : ''}>
    <span className="year">{result?.typical_year ?? '—'}</span>
    <span className="what"><b>{step.label}</b><small>{when(step, steps)} · {firmness(step.commitment)}</small>{warning && <small className="flag">{warning}</small>}</span>
    <span className="share">{result ? lives(result.taken_share) : ''}</span>
  </li>;
}

export function PlanCard({ version, previous, name, example }: { version: PlanVersion; previous?: PlanVersion; name: string; example: boolean }) {
  const results = new Map(version.evaluation.steps.map(r => [r.id, r]));
  const year = (s: PlanStep) => results.get(s.id)?.typical_year ?? 9999;
  const steps = version.plan.steps.filter(s => !contingent(s)).sort((a, b) => year(a) - year(b));
  const fallbacks = version.plan.steps.filter(contingent).sort((a, b) => year(a) - year(b));
  const c = version.comparison;
  const changes = previous ? changesBetween(previous, version) : [];
  return <div className="plan-card">
    <div className="plan-head">
      <h2>{name}’s plan</h2>
      <span className="plan-meta">{`${example ? 'Example · ' : ''}Version ${version.version} · ${version.signed ? 'signed by your twin' : 'draft, not signed'}`}</span>
    </div>
    <blockquote className="plan-narrative">{version.plan.narrative}</blockquote>
    <div className="stats plan-compare">
      <span>Goal by 2029<b>{percent(c.high_goal_odds[1])}</b><span className="was">{percent(c.high_goal_odds[0])} on habits alone</span></span>
      <span>Credit-limit risk<b>{percent(c.credit_breach_odds[1])}</b><span className="was">from {percent(c.credit_breach_odds[0])}</span></span>
      <span>Net worth 2034<b>CHF {compact(c.median_final_net_worth[1])}</b><span className="was">from {compact(c.median_final_net_worth[0])}</span></span>
    </div>
    {steps.length > 0 && <><h3>Steps <span>· year it usually happens, in {version.evaluation.n_lives} simulated lives</span></h3>
      <ol className="plan-steps">{steps.map(s => <Step key={s.id} step={s} result={results.get(s.id)} steps={version.plan.steps} />)}</ol></>}
    {fallbacks.length > 0 && <><h3>If things go wrong</h3>
      <ol className="plan-steps">{fallbacks.map(s => <Step key={s.id} step={s} result={results.get(s.id)} steps={version.plan.steps} />)}</ol></>}
    {version.kind === 'nudge' && <div className="plan-changes">
      <h3>What changed{version.as_of ? ` from ${version.as_of.slice(0, 4)}` : ''}</h3>
      <p className="said">“{version.said}”</p>
      {version.changes && <p>{version.changes}</p>}
      {changes.length > 0 && <ul>{changes.map(line => <li key={line}>{line}</li>)}</ul>}
    </div>}
    {!version.signed && <p className="note">Your twin did not sign this plan within its rounds: some steps may still clash with what the simulation shows.</p>}
    {version.plan.notes.length > 0 && <p className="note">Assumptions: {version.plan.notes.join(' ')}</p>}
  </div>;
}

const asOfOptions = Array.from({ length: 9 }, (_, i) => `${2026 + i}-01`);

// locked: something is running; outdated: the persona changed since the plan was made (only Start over helps).
export function NudgeForm({ busy, locked, outdated, onNudge, onReset }: { busy: boolean; locked: boolean; outdated: boolean; onNudge: (message: string, asOf: string | null) => void; onReset: () => void }) {
  const [message, setMessage] = useState('');
  const [asOf, setAsOf] = useState('');
  const disabled = locked || outdated;
  return <form className="plan-form" onSubmit={e => { e.preventDefault(); if (message.trim()) onNudge(message.trim(), asOf || null); }}>
    <label htmlFor="nudge">Something changed?</label>
    <textarea id="nudge" rows={2} maxLength={2000} value={message} disabled={disabled} onChange={e => setMessage(e.target.value)}
      placeholder="For example: I got a promotion, so no master’s. I’d still like a bigger flat around 2027." />
    <div className="plan-actions">
      <label className="from">From<select value={asOf} disabled={disabled} onChange={e => setAsOf(e.target.value)} aria-label="The change applies from">
        <option value="">the start</option>{asOfOptions.map(m => <option key={m} value={m}>January {m.slice(0, 4)}</option>)}
      </select></label>
      <button type="button" className="text-button" disabled={locked} onClick={onReset}><RotateCcw size={15} />Start over</button>
      <button className="primary small" disabled={disabled || !message.trim()}>{busy ? <><LoaderCircle className="spin" size={15} />Updating…</> : 'Update my plan'}</button>
    </div>
  </form>;
}

export function WishesForm({ busy, disabled, onPlan, onCancel, cancelLabel = 'Back to the example' }: { busy: boolean; disabled: boolean; onPlan: (wishes: string) => void; onCancel?: () => void; cancelLabel?: string }) {
  const [wishes, setWishes] = useState('');
  return <form className="plan-form" onSubmit={e => { e.preventDefault(); onPlan(wishes.trim()); }}>
    <label htmlFor="wishes">In your own words, what are your plans?</label>
    <textarea id="wishes" rows={3} maxLength={2000} value={wishes} disabled={disabled} onChange={e => setWishes(e.target.value)}
      placeholder="For example: a master’s next year, then full-time work and a bigger flat. If money gets tight I’d cut back rather than sell investments." />
    <p className="note">Your twin writes this as yearly steps for 2025–2034, with fallbacks, and tests it in 300 simulated lives. Leave it empty to let your twin guess from your persona. Your words and persona (not your name) go to your LLM provider.</p>
    <div className="plan-actions">
      {onCancel && <button type="button" className="text-button" disabled={disabled} onClick={onCancel}>{cancelLabel}</button>}
      <button className="primary small" disabled={disabled}>{busy ? <><LoaderCircle className="spin" size={15} />Planning…</> : <><PenLine size={15} />Make my plan</>}</button>
    </div>
  </form>;
}
