import type { PlanStep, StepResult } from './types';
import { compact, money, percent, chfText, wholePercents } from './types';
import { Outcome, when } from './Plan';

// The shape analysis/policy-explanations/run_explanations.py writes (web/src/example-explanation.json).
type Interval = { change_pp: number; lower_pp: number; upper_pp: number };
type Summary = { shortfall_pct: number; breach_pct: number; goal_pct: number; mean_final_wealth_chf: number; median_final_wealth_chf: number };
type View = { probabilities: Record<string, number[]>; available: Record<string, boolean[]> };
export type Explanation = {
  provenance: { model: string | null; version: number };
  person: { name: string; initial_cash_chf: number; high_goal: string; high_target_chf: number };
  paths: number;
  plan: { narrative: string; spontaneity: number; steps: PlanStep[] };
  planner_history: { version: number; kind: string | null; said: string | null; signed: boolean | null }[];
  execution: {
    step_report: StepResult[]; off_plan_changes_per_life: number;
    opening_state_intervention: { cash_comparisons: ({ cash_chf: number; habits: View; with_plan: View })[]; labels: Record<string, string[]> };
  };
  outcomes: {
    plan: Summary; habits: Summary;
    plan_minus_habits: { shortfall: Interval; breach: Interval; goal: Interval; mean_wealth_chf: { mean: number; lower: number; upper: number } };
    offer_stress: { offer_factor: number; summary: Summary; difference_vs_plan_baseline: { shortfall: Interval; goal: Interval; mean_wealth_chf: { mean: number; lower: number; upper: number } } }[];
  };
};

const HEADS: Record<string, string> = { status: 'Work and study', housing: 'Home', spending: 'Lifestyle', assets: 'Savings' };
const pct = (x: number) => `${x.toFixed(0)}%`;
const pp = (x: number) => `${x >= 0 ? '+' : '−'}${Math.abs(x).toFixed(1)} pts`;
const chf = (x: number) => `${x >= 0 ? '+' : '−'}CHF ${compact(Math.abs(x))}`;
const range = (lo: string, hi: string) => <span className="ci">95%: {lo} to {hi}</span>;

/** Three levels: what the LLM planned, how the simulated persona carries it out, and what follows financially. */
export default function Explained({ data }: { data: Explanation }) {
  const { plan, execution: ex, outcomes: o } = data;
  const results = new Map(ex.step_report.map(r => [r.id, r]));
  const [low, start] = ex.opening_state_intervention.cash_comparisons;
  const d = o.plan_minus_habits;
  return <article className="explained" aria-label="Plan explained">
    <header>
      <h2>From plan to outcomes</h2>
      <p className="muted">{data.person.name} · {money(data.paths)} paired simulated lives · plan version {data.provenance.version}{data.provenance.model ? ` by ${data.provenance.model}` : ''}</p>
    </header>

    <section>
      <span className="level">1 · The planner</span>
      <h3>What did the LLM plan?</h3>
      <blockquote>{chfText(plan.narrative)}</blockquote>
      <ul className="explained-steps">{plan.steps.map(s => <li key={s.id}>
        <b>{s.label}</b>
        <small>{when(s, plan.steps)} · {percent(s.commitment)} committed</small>
      </li>)}</ul>
      <p className="muted">Unplanned changes of work or home happen at {percent(plan.spontaneity)} of the usual rate. Versions: {data.planner_history.map(h => `${h.version} ${h.kind === 'nudge' ? '(after “' + h.said + '”)' : '(first draft)'}`).join(' · ')}.</p>
      <p className="note-box">The narrative and reasons are what the model wrote, not proof of how it reasoned.</p>
    </section>

    <section>
      <span className="level">2 · The simulated person</span>
      <h3>How is the plan carried out?</h3>
      <ul className="explained-steps">{plan.steps.map(s => { const r = results.get(s.id); return <li key={s.id}>
        <b>{s.label}{r?.typical_year ? <span className="muted"> · usually {r.typical_year}</span> : null}</b>
        {r && <Outcome step={s} result={r} />}
      </li>; })}</ul>
      <p className="muted">A commitment is the chance of following through when a step is due and possible, so it is not the share of lives where the step happens. Outside the steps, habits make about {ex.off_plan_changes_per_life.toFixed(1)} unplanned changes of work or home per life.</p>

      <h4>If January 2025 started with less cash</h4>
      <p className="muted">The same person and plan with CHF {money(low.cash_chf)} instead of CHF {money(start.cash_chf)}: the chance of each choice at that first review. Exact probabilities of the decision rules, not simulated frequencies.</p>
      {Object.keys(HEADS).map(head => {
        const labels = ex.opening_state_intervention.labels[head];
        // Each column is one choice's probabilities: rounded together, so a column adds up to 100%.
        const [a, b, ha, hb] = [low.with_plan, start.with_plan, low.habits, start.habits].map(v => wholePercents(v.probabilities[head]));
        const rows = labels.map((label, i) => ({ label, i, a: a[i], b: b[i], ha: ha[i], hb: hb[i],
          open: low.with_plan.available[head][i] || start.with_plan.available[head][i] }))
          .filter(r => r.open && Math.max(r.a, r.b, r.ha, r.hb) > 0);
        return <div className="table" key={head}><table>
          <thead><tr><th>{HEADS[head]}</th><th>CHF {compact(low.cash_chf)}</th><th>CHF {compact(start.cash_chf)}</th><th>Habits alone</th></tr></thead>
          <tbody>{rows.map(r => <tr key={r.i}><td>{r.label}</td><td>{r.a}%</td><td>{r.b}%</td><td className="muted">{r.ha}% / {r.hb}%</td></tr>)}</tbody>
        </table></div>;
      })}
    </section>

    <section>
      <span className="level">3 · The money</span>
      <h3>What happens financially?</h3>
      <p className="muted">The plan against the same person’s habits alone, on the same random draws.</p>
      <div className="table"><table>
        <thead><tr><th /><th>Plan</th><th>Habits</th><th>Difference</th></tr></thead>
        <tbody>
          <tr><td>{data.person.high_goal} of CHF {money(data.person.high_target_chf)} by 2029</td><td>{pct(o.plan.goal_pct)}</td><td>{pct(o.habits.goal_pct)}</td><td>{pp(d.goal.change_pp)} {range(pp(d.goal.lower_pp), pp(d.goal.upper_pp))}</td></tr>
          <tr><td>Hits the credit limit</td><td>{pct(o.plan.breach_pct)}</td><td>{pct(o.habits.breach_pct)}</td><td>{pp(d.breach.change_pp)} {range(pp(d.breach.lower_pp), pp(d.breach.upper_pp))}</td></tr>
          <tr><td>Cash ever below zero</td><td>{pct(o.plan.shortfall_pct)}</td><td>{pct(o.habits.shortfall_pct)}</td><td>{pp(d.shortfall.change_pp)} {range(pp(d.shortfall.lower_pp), pp(d.shortfall.upper_pp))}</td></tr>
          <tr><td>Net worth 2034 (mean)</td><td>CHF {compact(o.plan.mean_final_wealth_chf)}</td><td>CHF {compact(o.habits.mean_final_wealth_chf)}</td><td>{chf(d.mean_wealth_chf.mean)} {range(chf(d.mean_wealth_chf.lower), chf(d.mean_wealth_chf.upper))}</td></tr>
        </tbody>
      </table></div>

      <h4>If job offers came more or less often</h4>
      <p className="muted">The same plan in a harder or easier job market; the planner is not asked to change it.</p>
      <div className="table"><table>
        <thead><tr><th>Job offers</th><th>Goal by 2029</th><th>Change</th><th>Net worth 2034</th></tr></thead>
        <tbody>{o.offer_stress.map(s => <tr key={s.offer_factor}>
          <td>{s.offer_factor < 1 ? 'Half as often' : s.offer_factor > 1 ? '50% more often' : 'As usual'}</td>
          <td>{pct(s.summary.goal_pct)}</td>
          <td>{pp(s.difference_vs_plan_baseline.goal.change_pp)}</td>
          <td>{chf(s.difference_vs_plan_baseline.mean_wealth_chf.mean)}</td>
        </tr>)}</tbody>
      </table></div>
    </section>

  </article>;
}
