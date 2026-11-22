import { useEffect, useRef, useState } from 'react';
import { ArrowRight, ArrowUp, ChartSpline, Check, Compass, Home, Leaf, LoaderCircle, PenLine, RefreshCw, Sparkles, UserRound, Wallet, X } from 'lucide-react';
import Chart from './Chart';
import Dialog from './Dialog';
import Moments from './Moments';
import { DecisionSheet, Tree } from './Tree';
import PersonaBuilder from './PersonaBuilder';
import { NudgeForm, PlanCard, WishesForm } from './Plan';
import exampleData from './example.json';
import examplePlanData from './example-plan.json';
import { api, compact, MAX_DECISIONS, money, percent, REVIEW_MONTHS, scenarioNames, withScenarioDefaults, withStudyDefaults } from './types';
import type { BranchNode, ChatResult, ChoiceKey, Connection, Evidence, Measure, Message, PathNode, Person, PlanState, PlanVersion, Scenario } from './types';

const PALETTE = ['#10b981', '#a855f7', '#ec4899', '#06b6d4', '#eab308', '#6366f1', '#f97316', '#14b8a6'];
const scenarioLabel = (s: Scenario) => s.kind === 'part_time' ? `${s.alongside_studies ? 'Part-time while studying' : 'Part-time'} · CHF ${compact(s.salary)}`
  : s.kind === 'spend_less' ? 'Spend less' : s.kind === 'cheaper_home' ? 'Cheaper home' : 'Current path';
/** The two paths every run has: the current path, and the selected scenario branching from it in January 2025. */
const rootsOf = (ev: Evidence): PathNode[] => [
  { id: 'baseline', parent: null, root: 'baseline', label: 'Current path', color: 'var(--accent)', from: 0, decisions: [], summary: ev.baseline },
  ...(ev.alternative ? [{ id: 'alternative', parent: 'baseline', root: 'alternative' as const, label: scenarioLabel(ev.scenario), color: 'var(--alt)', from: 0, decisions: [], summary: ev.alternative }] : []),
];

const example: Evidence = { ...(exampleData as unknown as Evidence), person: withStudyDefaults(exampleData.person), scenario: withScenarioDefaults(exampleData.scenario as Scenario) };
const initialExplanation = `In this example, ${percent(example.baseline.high_goal_odds)} of simulated paths reach Mia’s CHF ${money(example.person.high_target)} emergency fund by December 2029, and ${percent(example.alternative!.high_goal_odds)} with the part-time offer.\n\nThe bigger difference comes later: median net worth at the end of 2034 falls from CHF ${money(example.baseline.final_net_worth.p50)} to CHF ${money(example.alternative!.final_net_worth.p50)}. How much would more time today matter to you?`;
const backToFullTime = example.alternative!.key_moments?.find(m => m.moment === 'Starts a full-time job');
const whenExplanation = backToFullTime ? `\n\nThe big decision comes first: every life cuts back to part-time in January 2025. Going back to full-time work is rarer and later, in ${percent(backToFullTime.share)} of lives, usually around ${backToFullTime.typical_year}.` : '';
const starterMessages: Message[] = [{ role: 'assistant', content: initialExplanation.replace('\n\nThe bigger difference', whenExplanation + '\n\nThe bigger difference') }];
const choices = [
  { kind: 'baseline', icon: Compass, title: 'Stay on my path' },
  { kind: 'part_time', icon: Leaf, title: 'Go part-time' },
  { kind: 'spend_less', icon: Wallet, title: 'Spend less' },
  { kind: 'cheaper_home', icon: Home, title: 'Cheaper home' },
] as const;
const suggestions = ['What is the biggest trade-off?', 'How uncertain is this?'];
// A real planner run for the example persona (python -m twin.plan.make_example): a draft, then one nudge.
const examplePlan = examplePlanData as unknown as { latest: PlanVersion; history: PlanVersion[] };
type OwnPlan = { id: string; versions: PlanVersion[]; person: Person; traits: Record<string, number> };
const POLL_MS = 1500, PLAN_TIMEOUT_MS = 6 * 60_000;
// Optional build-time workspace (VITE_TWIN_API_URL): the site connects to it without the Connect dialog.
const PRESET_URL = (import.meta.env.VITE_TWIN_API_URL ?? '').trim().replace(/\/+$/, '');

function ConnectionForm({ connection, onConnect }: { connection: Connection; onConnect: (c: Connection) => void }) {
  const [draft, setDraft] = useState(connection);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  async function connect(e: React.FormEvent) {
    e.preventDefault(); setBusy(true); setError('');
    try {
      const health = await api<{ llm_ready: boolean }>(draft, '/health');
      if (!health.llm_ready) throw new Error('Your laptop is reachable. Set TOGETHER_API_KEY and TOGETHER_MODEL there to enable the assistant, then reconnect.');
      onConnect(draft);
    } catch (err) { setError((err as Error).message); } finally { setBusy(false); }
  }
  return <form onSubmit={connect}>
    <div className="modal-heading"><h2>Connect workspace</h2><p>Your laptop runs the simulation and the assistant.</p></div>
    <label className="field">Workspace URL<input type="url" required placeholder="https://your-tunnel.example.com" value={draft.url} onChange={e => setDraft({ ...draft, url: e.target.value.trim() })} /></label>
    <label className="field">Workspace access key (if it has one)<input type="password" autoComplete="off" minLength={24} placeholder="Leave empty for an open workspace" value={draft.key} onChange={e => setDraft({ ...draft, key: e.target.value })} /></label>
    <p className="note">The access key stays in this tab; your LLM key stays on your laptop. Persona details, results and your messages are sent to your LLM provider.</p>
    <details className="setup-details"><summary>First time?</summary><ol><li>Start the FastAPI service on port 8000.</li><li>Expose it with an HTTPS tunnel and paste its URL above.</li><li>Add <code>{window.location.origin}</code> to <code>TWIN_ALLOWED_ORIGINS</code> and restart.</li><li>Use <code>TWIN_API_KEY</code> here, never your LLM key, or leave it empty if the API runs with <code>TWIN_OPEN=1</code>.</li></ol></details>
    {error && <p role="alert" className="error">{error}</p>}
    <button className="primary full-width" disabled={busy}>{busy ? <><LoaderCircle className="spin" size={17} />Connecting…</> : 'Connect workspace'}</button>
  </form>;
}

export default function App() {
  const [person, setPerson] = useState<Person>(example.person);
  const [traits, setTraits] = useState(example.traits);
  const [scenario, setScenario] = useState<Scenario>({ ...example.scenario, alongside_studies: true });
  const [deliberate, setDeliberate] = useState(false);
  const [evidence, setEvidence] = useState<Evidence>(example);
  const [messages, setMessages] = useState<Message[]>(starterMessages);
  const [session, setSession] = useState<string | null>(null);
  const [connection, setConnection] = useState<Connection>({ url: PRESET_URL || (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' ? 'http://127.0.0.1:8000' : ''), key: '' });
  const [connected, setConnected] = useState(!!PRESET_URL);
  const [exampleMode, setExampleMode] = useState(true);
  const [started, setStarted] = useState(false);          // landing screen until an exploration (or the example) starts
  const [edited, setEdited] = useState(false);
  const [modal, setModal] = useState<'persona' | 'connection' | 'method' | null>(null);
  const [busy, setBusy] = useState<'run' | 'chat' | 'plan' | null>(null);
  const [error, setError] = useState('');
  const [question, setQuestion] = useState('');
  const [ownPlan, setOwnPlan] = useState<OwnPlan | null>(null);
  const [composing, setComposing] = useState(false);       // example mode: the wishes form instead of the example plan
  const [followPlan, setFollowPlan] = useState(true);
  const [planProgress, setPlanProgress] = useState('');
  const [evidencePlan, setEvidencePlan] = useState<string | null>(null);   // plan version the results follow
  const planJob = useRef(0);
  const [branches, setBranches] = useState<PathNode[]>([]);
  const [activeId, setActiveId] = useState(example.alternative ? 'alternative' : 'baseline');
  const [pick, setPick] = useState<number | null>(null);
  const [growing, setGrowing] = useState(false);
  const [measure, setMeasure] = useState<Measure>('fan_cash');
  const chatEnd = useRef<HTMLDivElement>(null);
  const current = evidence.alternative ?? evidence.baseline;
  // Working alongside studies only applies to a student choosing the part-time offer.
  const canKeepStudying = person.status === 'student' && scenario.kind === 'part_time';
  const request = { ...scenario, alongside_studies: canKeepStudying && scenario.alongside_studies };
  const changed = JSON.stringify({ person, traits, scenario: request }) !== JSON.stringify({ person: evidence.person, traits: evidence.traits, scenario: evidence.scenario });
  const latestPlan = ownPlan?.versions[ownPlan.versions.length - 1];
  // A plan belongs to the persona it was made for; after an edit it is shown but no longer used.
  const planMatches = !!ownPlan && JSON.stringify({ person, traits }) === JSON.stringify({ person: ownPlan.person, traits: ownPlan.traits });
  const planKey = ownPlan && latestPlan && planMatches && followPlan ? `${ownPlan.id}:${latestPlan.version}` : null;
  const stale = changed || edited || deliberate !== (evidence.deliberate ?? false) || planKey !== evidencePlan;
  const nodes = [...rootsOf(evidence), ...branches];
  const active = nodes.find(n => n.id === activeId) ?? nodes[0];
  const lastDecision = Math.max(0, ...active.decisions.map(d => d.month));
  // January 2025 ("now") takes life events only; later Januaries take decisions too.
  const decisionMonths = stale || active.decisions.length >= MAX_DECISIONS ? []
    : [0, ...REVIEW_MONTHS].filter(m => active.decisions.length ? m > lastDecision : true);
  const canBranch = !exampleMode && !!session && !stale;
  const salaryInvalid = scenario.kind === 'part_time' && (!Number.isFinite(scenario.salary) || scenario.salary <= 0 || scenario.salary > 100000);

  useEffect(() => { if (messages.length > 1 || busy) chatEnd.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }, [messages.length, busy]);

  async function explore(withPlan?: { id: string; key: string } | null) {
    if (!connected) { setModal('connection'); return; }
    const plan = withPlan !== undefined ? withPlan : planKey ? { id: ownPlan!.id, key: planKey } : null;
    const what = request.alongside_studies ? 'working part-time alongside my studies' : scenarioNames[scenario.kind].toLowerCase();
    const question = !plan ? `Help me understand ${what}, how it affects my goal, when the big decisions in this life typically happen, and the trade-offs for what I value.`
      : scenario.kind === 'baseline' ? 'Help me understand my plan: how often its steps really happen and when, how it affects my goal, and the trade-offs for what I value.'
      : `Help me understand ${what} on top of my plan, how it affects my goal, when the big decisions typically happen, and the trade-offs for what I value.`;
    const wasStarted = started;
    setBusy('run'); setError(''); setStarted(true); setPick(null);
    try {
      const result = await api<{ session_id: string; evidence: Evidence; answer: string }>(connection, '/sessions', {
        person, traits, scenario: request, deliberate, n_paths: 500, question, ...(plan ? { plan_id: plan.id } : {}),
      });
      const previous = session;
      adopt(result.evidence); setSession(result.session_id); setExampleMode(false);
      setEvidencePlan(plan?.key ?? null);
      setMessages([{ role: 'assistant', content: result.answer }]); setQuestion('');
      if (previous) void api(connection, `/sessions/${previous}`, undefined, 'DELETE').catch(() => {});
    } catch (err) { setError((err as Error).message); if (!wasStarted) setStarted(false); } finally { setBusy(null); }
  }

  async function send(text = question) {
    if (!text.trim() || busy) return;
    if (!session || stale) { setError('Explore your current choices first, so the conversation uses the right results.'); return; }
    setBusy('chat'); setError('');
    try {
      const result = await api<ChatResult>(connection, `/sessions/${session}/messages`, { message: text.trim() });
      for (const action of result.actions ?? []) {
        if (action.type === 'rerun') adopt(action.evidence);
        else if (action.type === 'branch') addBranch(action.branch);
        else if (action.type === 'highlight') {
          setActiveId(action.path_id); setPick(null);
          if (action.measure) setMeasure(action.measure === 'net_worth' ? 'fan_net_worth' : 'fan_cash');
        }
      }
      setMessages(m => [...m, { role: 'user', content: text.trim() }, { role: 'assistant', content: result.answer, notes: result.notes }]);
      setQuestion('');
    } catch (err) { setError((err as Error).message); } finally { setBusy(null); }
  }

  // Planning runs on the laptop as a job (one to three minutes); poll it, then explore the new plan.
  async function runPlanJob(start: () => Promise<PlanState>, base: OwnPlan | null, snapshot: { person: Person; traits: Record<string, number> }) {
    const token = ++planJob.current;
    setBusy('plan'); setError(''); setPlanProgress('Starting');
    try {
      let state = await start();
      const id = base?.id ?? state.plan_id!;
      const deadline = Date.now() + PLAN_TIMEOUT_MS;
      while (state.status === 'working') {
        if (Date.now() > deadline) throw new Error('Planning is taking unusually long. Check the laptop, then try again.');
        await new Promise(resolve => setTimeout(resolve, POLL_MS));
        if (token !== planJob.current) return;
        state = await api<PlanState>(connection, `/plans/${id}`);
        setPlanProgress(state.progress);
      }
      if (token !== planJob.current) return;
      if (state.status === 'failed' || !state.latest) throw new Error(state.error ?? 'The planner did not produce a plan. Try again.');
      const latest = state.latest;
      setOwnPlan(base ? { ...base, versions: [...base.versions, latest] } : { id, versions: [latest], ...snapshot });
      setComposing(false); setFollowPlan(true); setPlanProgress(''); setBusy(null);
      await explore({ id, key: `${id}:${latest.version}` });
    } catch (err) { setError((err as Error).message); } finally {
      if (token === planJob.current) { setPlanProgress(''); setBusy(b => b === 'plan' ? null : b); }
    }
  }

  function makePlan(wishes: string) {
    if (!connected) { setModal('connection'); return; }
    if (ownPlan) void api(connection, `/plans/${ownPlan.id}`, undefined, 'DELETE').catch(() => {});
    void runPlanJob(() => api<PlanState>(connection, '/plans', { person, traits, wishes, n_paths: 300 }), null, { person, traits });
  }

  function nudgePlan(message: string, asOf: string | null) {
    if (!ownPlan) return;
    void runPlanJob(() => api<PlanState>(connection, `/plans/${ownPlan.id}/nudges`, { message, ...(asOf ? { as_of: asOf } : {}) }), ownPlan, ownPlan);
  }

  function resetPlan() {
    if (ownPlan) void api(connection, `/plans/${ownPlan.id}`, undefined, 'DELETE').catch(() => {});
    setOwnPlan(null); setComposing(true); setError('');
  }

  async function grow(choice: ChoiceKey) {
    if (!session || pick === null || growing) return;
    const from = active, month = pick;
    setGrowing(true); setError('');
    try {
      addBranch(await api<BranchNode>(connection, `/sessions/${session}/branches`, { parent: from.id, month, choice }));
    } catch (err) { setError((err as Error).message); } finally { setGrowing(false); }
  }

  /** New results (an exploration or the assistant's re-run): the controls follow them, and the tree restarts. */
  function adopt(ev: Evidence) {
    const next = { ...ev, person: withStudyDefaults(ev.person), scenario: withScenarioDefaults(ev.scenario) };
    setEvidence(next); setPerson(next.person); setTraits(next.traits); setDeliberate(!!next.deliberate);
    setScenario(s => ({ ...next.scenario, alongside_studies: next.scenario.alongside_studies || (next.person.status !== 'student' && s.alongside_studies) }));
    setBranches([]); setPick(null); setActiveId(next.alternative ? 'alternative' : 'baseline'); setEdited(false);
  }

  function addBranch(node: BranchNode) {
    setBranches(b => [...b, { ...node, color: PALETTE[b.length % PALETTE.length] }]);
    setActiveId(node.id); setPick(null);
  }

  function removeBranch(id: string) {
    const gone = new Set([id]);
    for (let grew = true; grew;) { grew = false; for (const n of branches) if (n.parent && gone.has(n.parent) && !gone.has(n.id)) { gone.add(n.id); grew = true; } }
    setBranches(b => b.filter(n => !gone.has(n.id)));
    if (session && !exampleMode) void api(connection, `/sessions/${session}/branches/${id}`, undefined, 'DELETE').catch(() => {});
    if (gone.has(activeId)) setActiveId(nodes.find(n => n.id === id)?.parent ?? 'baseline');
    setPick(null);
  }

  function savePersona(p: Person, t: Record<string, number>) {
    setPerson(p); setTraits(t); setEdited(true); setModal(null); setError('');
  }

  const goal = `${evidence.person.high_goal.toLowerCase()} of CHF ${money(evidence.person.high_target)}`;
  const loading = busy === 'run' && exampleMode;              // the first run: nothing of the person's own to show yet
  const progress = busy === 'plan' && <p className="plan-progress" role="status"><LoaderCircle className="spin" size={16} />{planProgress || 'Working'}<span>· usually one to three minutes</span></p>;
  const errorBox = error && <div className="error" role="alert"><span>{error}</span><button className="icon-button" onClick={() => setError('')} aria-label="Dismiss error"><X size={16} /></button></div>;
  const fine = <p className="fine">Simulated possibilities, not financial advice. <button onClick={() => setModal('method')}>How it works</button></p>;
  const controls = <>
    <div className="chips" role="group" aria-label="Possibilities">{choices.map(({ kind, icon: Icon, title }) =>
      <button key={kind} className="chip" aria-pressed={scenario.kind === kind} disabled={!!busy} onClick={() => { setScenario(s => ({ ...s, kind })); setError(''); }}>
        {scenario.kind === kind ? <Check size={16} /> : <Icon size={16} />}{title}
      </button>)}
    </div>
    <div className="explore-row">
      {scenario.kind === 'part_time' && <label className="salary">CHF<input aria-label="Part-time monthly take-home pay" type="number" min="1" max="100000" value={scenario.salary || ''} onChange={e => setScenario(s => ({ ...s, salary: Number(e.target.value) }))} disabled={!!busy} />/ month</label>}
      {canKeepStudying && <button type="button" role="switch" aria-checked={scenario.alongside_studies} className="switch" disabled={!!busy} onClick={() => { setScenario(s => ({ ...s, alongside_studies: !s.alongside_studies })); setError(''); }}
        title="On: work the part-time job next to your course, then look for a graduate job when it ends. Off: leave the course for the job.">
        <span className="track" aria-hidden="true"><span /></span>Keep studying
      </button>}
      {ownPlan && planMatches && <button type="button" role="switch" aria-checked={followPlan} className="switch" disabled={!!busy} onClick={() => { setFollowPlan(f => !f); setError(''); }}
        title="On: your twin follows your plan and decides the rest from habit. Off: habit alone.">
        <span className="track" aria-hidden="true"><span /></span>Follow my plan
      </button>}
      <button type="button" role="switch" aria-checked={deliberate} className="switch" disabled={!!busy} onClick={() => { setDeliberate(d => !d); setError(''); }}
        title="Your persona weighs each yearly choice by what it is worth over the next five years, instead of following habit.">
        <span className="track" aria-hidden="true"><span /></span>Decide deliberately
      </button>
      <button className="primary" disabled={!!busy || salaryInvalid} onClick={() => void explore()}>{busy === 'run' ? <><LoaderCircle className="spin" size={16} />Exploring…</> : 'Explore this path'}</button>
    </div>
  </>;

  return <div className={`app ${started ? 'split' : ''}`}>
    <header className="topbar">
      <button className="brand" onClick={() => { setStarted(false); setPick(null); }} aria-label="parallel, home">parallel</button>
      <div className="topbar-right">
        {exampleMode && started && <span className="tag">Example</span>}
        <button className="ghost" disabled={!!busy} onClick={() => setModal('connection')}><span className={`dot ${connected ? 'on' : ''}`} />{connected ? 'Connected' : 'Connect'}</button>
        <button className="avatar" disabled={!!busy} onClick={() => setModal('persona')} aria-label="Edit your persona">{person.name.slice(0, 1).toUpperCase()}</button>
      </div>
    </header>

    {!started ? <main id="main" className="home">
      <section className="hello">
        <h1><span className="gradient">Hi {person.name}</span></h1>
        <p>What would you like to explore?</p>
      </section>
      {controls}
      <div className="home-links">
        <button className="text-button" disabled={!!busy} onClick={() => setModal('persona')}><UserRound size={16} />Build my persona</button>
        {!ownPlan && !composing && <button className="text-button" disabled={!!busy} onClick={() => setComposing(true)}><PenLine size={16} />Make my own plan</button>}
        {exampleMode ? <button className="text-button" disabled={!!busy} onClick={() => setStarted(true)}><ChartSpline size={16} />See an example</button>
          : <button className="text-button" disabled={!!busy} onClick={() => setStarted(true)}><ArrowRight size={16} />Back to my results</button>}
      </div>
      {(composing || busy === 'plan') && <section className="plan home-plan" aria-label="Your plan">
        {progress}
        {composing && <WishesForm busy={busy === 'plan'} disabled={!!busy} onPlan={makePlan} onCancel={() => setComposing(false)} cancelLabel="Cancel" />}
      </section>}
      {errorBox}
      {fine}
    </main>

    : <main id="main" className="workspace">
      <div className="pane text-pane">
        <section className="controls" aria-label="Your choices">
          {controls}
          {stale && <p className="stale" role="status">Your choices have changed. Explore again to update.</p>}
        </section>

        {!loading && <>
          <section className={`summary ${stale ? 'dim' : ''}`} aria-label="Outlook">
            <h2 className="results-label">{evidence.person.name}’s outlook <span>· {exampleMode ? 'example' : stale ? 'previous run' : `${evidence.n_paths} paths`}{evidence.deliberate ? ' · deliberate' : ''}{evidence.plan ? ' · following your plan' : ''}</span></h2>
            <p className="headline"><b>{percent(current.high_goal_odds)}</b>chance of reaching {goal} by Dec 2029{evidence.alternative && <span className="was">{percent(evidence.baseline.high_goal_odds)} on your current path</span>}</p>
            <div className="stats">
              <span>Net worth 2034<b>CHF {compact(current.final_net_worth.p50)}</b>{evidence.alternative && <span className="was">from {compact(evidence.baseline.final_net_worth.p50)}</span>}</span>
              <span>Credit-limit risk<b>{percent(current.credit_breach_odds)}</b>{evidence.alternative && <span className="was">from {percent(evidence.baseline.credit_breach_odds)}</span>}</span>
            </div>
          </section>

          <section className="plan" aria-label="Your plan">
            {progress}
            {ownPlan && latestPlan ? <>
              <PlanCard version={latestPlan} previous={ownPlan.versions[ownPlan.versions.length - 2]} name={ownPlan.person.name} example={false} />
              {!planMatches && <p className="stale">You changed your persona after making this plan, so explorations no longer follow it. Start over to make a new one.</p>}
              <NudgeForm busy={busy === 'plan'} locked={!!busy} outdated={!planMatches} onNudge={nudgePlan} onReset={resetPlan} />
            </> : composing ? <>
              <h2 className="results-label">Your plan</h2>
              <WishesForm busy={busy === 'plan'} disabled={!!busy} onPlan={makePlan} onCancel={() => setComposing(false)} cancelLabel={exampleMode ? 'Back to the example' : 'Cancel'} />
            </> : exampleMode ? <>
              <PlanCard version={examplePlan.latest} previous={examplePlan.history[examplePlan.history.length - 2]} name={example.person.name} example />
              <button className="secondary plan-start" disabled={!!busy} onClick={() => setComposing(true)}><PenLine size={16} />Make my own plan</button>
            </> : <button className="text-button" disabled={!!busy} onClick={() => setComposing(true)}><PenLine size={16} />Make a long-term plan</button>}
          </section>

          <div className={stale ? 'dim' : ''}><Moments moments={active.summary.key_moments} label={active.label} color={active.color} /></div>
        </>}

        <section className="thread" aria-label="Conversation" aria-live="polite" aria-relevant="additions">
          {!loading && messages.map((m, i) => <div className={`msg ${m.role}`} key={i}>
            {m.role === 'assistant' && <span className="spark" aria-hidden="true"><Sparkles size={18} /></span>}
            <div className="msg-body">
              {!!m.notes?.length && <ul className="msg-notes" aria-label="Changes made">{m.notes.map(n => <li key={n}><RefreshCw size={13} aria-hidden="true" />{n}</li>)}</ul>}
              {m.content.split('\n\n').map((paragraph, j) => <p key={j}>{paragraph}</p>)}
            </div>
          </div>)}
          {busy && busy !== 'plan' && <div className="thinking" role="status"><LoaderCircle size={17} className="spin" />{busy === 'run' ? 'Exploring 500 paths…' : 'Thinking…'}</div>}
          <div className="thread-end" ref={chatEnd} />
        </section>

        <div className="dock">
          {errorBox}
          {!exampleMode && !stale && messages.length === 1 && !busy && <div className="suggestions">{suggestions.map(q => <button className="chip" key={q} onClick={() => void send(q)}>{q}</button>)}</div>}
          <form className="composer" onSubmit={e => { e.preventDefault(); void send(); }}>
            <label className="sr-only" htmlFor="question">Ask about your financial future</label>
            <textarea id="question" rows={1} maxLength={2000} value={question} disabled={exampleMode || stale || !!busy} onChange={e => setQuestion(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); void send(); } }} placeholder={exampleMode ? 'Connect to ask your own questions' : stale ? 'Explore your new choices first' : 'Ask about your future'} />
            {exampleMode
              ? <button type="button" className="primary small" disabled={!!busy} onClick={() => connected ? void explore() : setModal('connection')}>{connected ? 'Start exploring' : 'Connect my workspace'}</button>
              : <button type="submit" className="send" aria-label="Send message" disabled={!question.trim() || !!busy || stale}><ArrowUp size={20} /></button>}
          </form>
          {fine}
        </div>
      </div>

      <section className="pane canvas" aria-label="Your paths">
        {loading
          ? <div className="canvas-loading" aria-hidden="true"><LoaderCircle size={22} className="spin" />Simulating 500 lives, 2025 to 2034…</div>
          : <div className={stale ? 'dim' : ''}>
            <Chart nodes={nodes} activeId={active.id} decisionMonths={decisionMonths} pick={pick} onPick={m => setPick(p => p === m ? null : m)}
              measure={measure} onMeasure={setMeasure}
              goal={{ target: evidence.person.high_target, cashGoal: evidence.person.high_goal === 'Emergency savings' }} />
            {pick !== null && <DecisionSheet month={pick} from={active} busy={growing} canBranch={canBranch}
              reason={exampleMode ? 'This is an example. Connect your workspace to grow your own branches.' : 'Explore your current choices first.'}
              onChoose={c => void grow(c)} onClose={() => setPick(null)} onConnect={exampleMode ? () => setModal('connection') : undefined} />}
            <Tree nodes={nodes} activeId={active.id} onSelect={id => { setActiveId(id); setPick(null); }} onRemove={removeBranch} />
          </div>}
      </section>
    </main>}

    {modal === 'persona' && <Dialog title="Build your persona" wide onClose={() => setModal(null)}><PersonaBuilder person={person} traits={traits} onSave={savePersona} onClose={() => setModal(null)} /></Dialog>}
    {modal === 'connection' && <Dialog title="Connect workspace" onClose={() => setModal(null)}><ConnectionForm connection={connection} onConnect={c => { if (c.url !== connection.url || c.key !== connection.key) { setSession(null); setExampleMode(true); setEvidence(example); setMessages(starterMessages); setEdited(true); setOwnPlan(null); setEvidencePlan(null); planJob.current++; setBranches([]); setPick(null); setActiveId('alternative'); setStarted(false); } setConnection(c); setConnected(true); setModal(null); setError(''); }} /></Dialog>}
    {modal === 'method' && <Dialog title="How it works" onClose={() => setModal(null)}>
      <div className="modal-heading"><h2>How it works</h2></div>
      <div className="method-list">
        <div><h3>You set the starting point</h3><p>Balances, situation, goal and preferences shape the decisions your twin makes.</p></div>
        <div><h3>Your laptop simulates 500 paths</h3><p>January 2025 to December 2034. Choices happen each January; events and cash flows evolve monthly. A fixed historical start, not a forecast from today.</p></div>
        <div><h3>An assistant explains the results</h3><p>It receives persona details (not your name), the results and your messages. Change your choices and explore again for a new comparison.</p></div>
        <div><h3>Your plan</h3><p>Tell your twin your plans in your own words. An AI planner writes them as yearly steps with fallbacks, tests them in 300 simulated lives, and revises until the steps are possible. When something changes, tell it: it updates only what the change affects. Explorations then follow your plan; your twin decides the rest from habit.</p></div>
        <div><h3>Habit or deliberate</h3><p>By default your twin decides like people tend to: mostly sticking with what they do. “Decide deliberately” nudges every yearly choice toward what is worth more to you over the next five years.</p></div>
        <div><h3>Branch at any January</h3><p>Tap a dot on the chart to try a new decision that year. The branch shares every life with its parent up to that January, then follows the decision wherever it is possible. Branch again from a branch to build a tree.</p></div>
        <div><h3>What “goal success” means</h3><p>Reaching the target by December 2029 without exceeding the credit limit before then. Emergency savings counts cash; other goals use net worth.</p></div>
      </div>
      <p className="note">Research prototype on synthetic data. No bank connection. Sessions live in laptop memory for up to an hour.</p>
    </Dialog>}
  </div>;
}
