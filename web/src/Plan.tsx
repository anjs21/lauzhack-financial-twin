import { useState } from 'react';
import { ChevronDown, LoaderCircle, PenLine, RotateCcw } from 'lucide-react';
import type { PlanRound, PlanStep, PlanVersion, StepResult } from './types';
import { compact, money, percent, chfText, wholePercents } from './types';

const EVENTS: Record<string, string> = {
  job_loss: 'if I lose my job', offer: 'once I get a job offer', graduated: 'once I finish studying',
  partner: 'if a partner moves in', child: 'if we have a child', separation: 'if we separate', windfall: 'if I receive a windfall',
};
const reviewYear = (month: string) => Number(month.slice(0, 4)) + (month.slice(5) === '01' ? 0 : 1);
const lives = (share: number) => share >= 0.995 ? 'every life' : `${percent(share)} of lives`;
const firmness = (c: number) => `${c >= 0.85 ? 'firm' : c >= 0.6 ? 'likely' : 'maybe'}, ${percent(c)} sure`;
export const contingent = (s: PlanStep) => s.event != null || s.cash_months_below != null;

export function when(step: PlanStep, steps: PlanStep[]) {
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

/** What happened to a step across the simulated lives. The five parts don't overlap and add up to every life. */
function outcomeParts(step: PlanStep, r: StepResult) {
  const worldNo = step.head === 'housing' ? 'no flat found' : step.option === 'start_masters' ? 'not admitted' : 'no job offer came';
  const notDue = contingent(step) ? 'never needed' : step.after ? 'the step before never happened' : 'never came up';
  const parts = [
    { key: 'done', share: r.taken_share, text: 'did it' },
    { key: 'declined', share: r.declined_share, text: 'chose not to' },
    { key: 'refused', share: r.turned_down_share, text: `tried, ${worldNo}` },
    { key: 'blocked', share: r.blocked_share, text: 'not possible' },
    { key: 'idle', share: Math.max(0, 1 - r.trigger_share), text: notDue },
  ];
  const whole = wholePercents(parts.map(p => p.share));                   // rounded together: they add up to 100
  return parts.map((p, i) => ({ ...p, pct: whole[i] })).filter(p => p.pct > 0);
}

export function Outcome({ step, result }: { step: PlanStep; result: StepResult }) {
  const parts = outcomeParts(step, result);
  return <div className="outcome">
    <span className="outcome-bar" aria-hidden="true">{parts.map(p => <i key={p.key} className={p.key} style={{ flexGrow: p.pct }} />)}</span>
    <small>{parts.map(p => `${p.text} ${p.pct}%`).join(' · ')}</small>
    {result.blocked_because && <small className="flag">Not possible in {percent(result.blocked_share)}: {result.blocked_because}</small>}
  </div>;
}

// detailed: also the planner's reason and how the step played out across the lives.
function Step({ step, result, steps, detailed }: { step: PlanStep; result?: StepResult; steps: PlanStep[]; detailed: boolean }) {
  return <li className={contingent(step) ? 'fallback' : ''}>
    <span className="year">{result?.typical_year ?? '—'}</span>
    <span className="what">
      <b>{step.label}</b>
      <small>{when(step, steps)} · {firmness(step.commitment)}</small>
      {detailed && step.why && <span className="why">{chfText(step.why)}</span>}
      {detailed && result && <Outcome step={step} result={result} />}
    </span>
    <span className="share">{result ? lives(result.taken_share) : ''}</span>
  </li>;
}

/** The planner's text as the person reads it: step ids (snake_case, maybe quoted) become their labels. */
function plain(text: string, labels: Map<string, string>) {
  return chfText(text).replace(/'?\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b'?/g, (_, id: string) => `“${labels.get(id) ?? id.replace(/_/g, ' ')}”`);
}
/** A simulator warning: its first sentence (the rest is advice to the planner). */
const plainWarning = (text: string, labels: Map<string, string>) => plain(text.split(/(?<=[.?!])\s+(?=[A-Z])/)[0], labels);

function Rounds({ rounds, signed, labels, lives: n }: { rounds: PlanRound[]; signed: boolean; labels: Map<string, string>; lives: number }) {
  const last = rounds[rounds.length - 1];
  return <details className="plan-rounds">
    <summary>How your twin worked this out<span> · {rounds.length} round{rounds.length === 1 ? '' : 's'}, {signed ? `signed in round ${last.round}` : 'not signed'}</span></summary>
    <ol>{rounds.map(r => <li key={r.round}>
      <b>Round {r.round}</b>
      {r.errors ? <p className="muted">Its reply could not be used as a plan, so it was asked to fix it.</p>
        : <>
          {r.thought && <p>{plain(r.thought, labels)}</p>}
          <p className="muted">Tested in {n} lives: {r.warnings?.length ? 'the simulation flagged' : 'nothing flagged.'}</p>
          {!!r.warnings?.length && <ul>{r.warnings.map(w => <li key={w}>{plainWarning(w, labels)}</li>)}</ul>}
          {signed && r === last && <p className="muted">Your twin signed the plan.</p>}
        </>}
    </li>)}</ol>
  </details>;
}

export function PlanCard({ version, previous, name, example }: { version: PlanVersion; previous?: PlanVersion; name: string; example: boolean }) {
  const results = new Map(version.evaluation.steps.map(r => [r.id, r]));
  const year = (s: PlanStep) => results.get(s.id)?.typical_year ?? 9999;
  const steps = version.plan.steps.filter(s => !contingent(s)).sort((a, b) => year(a) - year(b));
  const fallbacks = version.plan.steps.filter(contingent).sort((a, b) => year(a) - year(b));
  const c = version.comparison;
  const changes = previous ? changesBetween(previous, version) : [];
  const labels = new Map([...(previous?.plan.steps ?? []), ...version.plan.steps].map(s => [s.id, s.label]));
  const trouble = version.evaluation.money_trouble;
  const mostly = trouble && Object.entries(trouble.while).sort((a, b) => b[1] - a[1])[0];
  const offPlan = version.evaluation.off_plan_changes_per_life;
  // The card shows the plan and how often each step happens; reasons and the planner's working are one tap away.
  const [detailed, setDetailed] = useState(false);
  const hasDetails = version.plan.steps.length > 0 || !!trouble || offPlan != null || version.plan.notes.length > 0 || !!version.rounds?.length;
  return <div className="plan-card">
    <div className="plan-head">
      <h2>{name}’s plan</h2>
      <span className="plan-meta">{`${example ? 'Example · ' : ''}Version ${version.version} · ${version.signed ? 'signed by your twin' : 'draft, not signed'}`}</span>
    </div>
    <blockquote className={`plan-narrative ${detailed ? '' : 'clamped'}`}>{chfText(version.plan.narrative)}</blockquote>
    <div className="stats plan-compare">
      <span>Goal by 2029<b>{percent(c.high_goal_odds[1])}</b><span className="was">{percent(c.high_goal_odds[0])} on habits alone</span></span>
      <span>Credit-limit risk<b>{percent(c.credit_breach_odds[1])}</b><span className="was">from {percent(c.credit_breach_odds[0])}</span></span>
      <span>Net worth 2034<b>CHF {compact(c.median_final_net_worth[1])}</b><span className="was">from {compact(c.median_final_net_worth[0])}</span></span>
    </div>
    {steps.length > 0 && <><h3>Steps <span>· in {version.evaluation.n_lives} simulated lives</span></h3>
      <ol className="plan-steps">{steps.map(s => <Step key={s.id} step={s} result={results.get(s.id)} steps={version.plan.steps} detailed={detailed} />)}</ol></>}
    {fallbacks.length > 0 && <><h3>If things go wrong</h3>
      <ol className="plan-steps">{fallbacks.map(s => <Step key={s.id} step={s} result={results.get(s.id)} steps={version.plan.steps} detailed={detailed} />)}</ol></>}
    {version.kind === 'nudge' && <div className="plan-changes">
      <h3>What changed{version.as_of ? ` from ${version.as_of.slice(0, 4)}` : ''}</h3>
      <p className="said">“{version.said}”</p>
      {version.changes && <p>{plain(version.changes, labels)}</p>}
      {changes.length > 0 && <ul>{changes.map(line => <li key={line}>{line}</li>)}</ul>}
    </div>}
    {!version.signed && <p className="note">Your twin did not sign this plan within its rounds: some steps may still clash with what the simulation shows.</p>}
    {hasDetails && <button type="button" className="text-button plan-more" aria-expanded={detailed} onClick={() => setDetailed(d => !d)}>
      <ChevronDown size={16} aria-hidden="true" />{detailed ? 'Hide details' : 'Show details: the full plan, reasons, and how each step played out'}
    </button>}
    {detailed && <>
    {version.plan.steps.length > 0 && <p className="outcome-key" aria-hidden="true">
      <span><i className="done" />did it</span><span><i className="declined" />chose not to</span><span><i className="refused" />world said no</span>
      <span><i className="blocked" />not possible</span><span><i className="idle" />never came up</span></p>}
    {(trouble || offPlan != null) && <div className="plan-findings">
      <h3>What the simulation showed</h3>
      <ul>
        {trouble && mostly && <li>In {percent(trouble.share_of_lives)} of lives cash runs past the credit limit, usually in {trouble.typical_year}, mostly while {mostly[0]} ({percent(mostly[1])}), on about CHF {money(trouble.income_then)} a month in {trouble.home_then}.</li>}
        {offPlan != null && <li>The steps set the big choices; habits decide the rest. Unplanned changes of work or home: about {offPlan.toFixed(1)} per life.</li>}
      </ul>
    </div>}
    {version.plan.notes.length > 0 && <div className="plan-findings">
      <h3>Your twin’s reasoning</h3>
      <ul>{version.plan.notes.map(n => <li key={n}>{plain(n, labels)}</li>)}</ul>
    </div>}
    {!!version.rounds?.length && <Rounds rounds={version.rounds} signed={version.signed} labels={labels} lives={version.evaluation.n_lives} />}
    </>}
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
