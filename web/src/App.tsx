import { useEffect, useRef, useState } from 'react';
import { Activity, BookOpen, Check, ChevronDown, CircleHelp, Compass, Heart, Home, Leaf, LoaderCircle, MessageCircle, PanelLeftClose, Plus, Send, Settings2, ShieldCheck, Sparkles, Target, Wallet, X } from 'lucide-react';
import Chart from './Chart';
import Dialog from './Dialog';
import PersonaBuilder from './PersonaBuilder';
import exampleData from './example.json';
import { api, money, percent, scenarioNames } from './types';
import type { Connection, Evidence, Message, Person, Scenario } from './types';

const example = exampleData as Evidence;
const initialExplanation = `Mia’s goal looks within reach on both paths. In this example, ${percent(example.baseline.high_goal_odds)} of current-path simulations reach CHF ${money(example.person.high_target)} in emergency savings by December 2029, compared with ${percent(example.alternative!.high_goal_odds)} after taking the part-time offer.\n\nThe bigger trade-off appears later: median net worth at the end of 2034 falls from CHF ${money(example.baseline.final_net_worth.p50)} to CHF ${money(example.alternative!.final_net_worth.p50)}. A small change in goal success does not mean the two futures are financially equivalent.\n\nThese are possibilities under the model’s assumptions. How much would having more time today matter to you?`;
const starterMessages: Message[] = [{ role: 'assistant', content: initialExplanation }];
const choices = [
  { kind: 'baseline', icon: Compass, title: 'Stay on my path', description: 'See where things could lead.' },
  { kind: 'part_time', icon: Leaf, title: 'Make room for life', description: 'Explore a part-time offer.' },
  { kind: 'spend_less', icon: Wallet, title: 'Spend a little less', description: 'Try a leaner lifestyle.' },
  { kind: 'cheaper_home', icon: Home, title: 'Rethink my home', description: 'Move one housing tier down.' },
] as const;

function ConnectionForm({ connection, onConnect }: { connection: Connection; onConnect: (c: Connection) => void }) {
  const [draft, setDraft] = useState(connection);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  async function connect(e: React.FormEvent) {
    e.preventDefault(); setBusy(true); setError('');
    try {
      const health = await api<{ llm_ready: boolean }>(draft, '/health');
      if (!health.llm_ready) throw new Error('Your laptop is reachable. Set OPENAI_API_KEY and OPENAI_MODEL there to enable the assistant, then reconnect.');
      onConnect(draft);
    } catch (err) { setError((err as Error).message); } finally { setBusy(false); }
  }
  return <form onSubmit={connect}>
    <div className="modal-heading"><span className="eyebrow">YOUR PRIVATE WORKSPACE</span><h2>Connect your twin.</h2><p>Your laptop runs the simulation and connects to the explanation service.</p></div>
    <label className="field">Workspace URL<input type="url" required placeholder="https://your-tunnel.example.com" value={draft.url} onChange={e => setDraft({ ...draft, url: e.target.value.trim() })} /></label>
    <label className="field">Workspace access key<input type="password" autoComplete="off" required minLength={24} placeholder="The TWIN_API_KEY on your laptop" value={draft.key} onChange={e => setDraft({ ...draft, key: e.target.value })} /></label>
    <div className="info-note"><ShieldCheck size={20} /><span>The access key stays in this tab’s memory. Your LLM provider key stays on your laptop. Relevant persona details, simulation summaries, and your messages are sent to that provider for explanations.</span></div>
    <details className="setup-details"><summary>Connecting for the first time?</summary><ol><li>Start the FastAPI service on your laptop at port 8000.</li><li>Expose it using an HTTPS tunnel, then enter its URL above.</li><li>Add <code>{window.location.origin}</code> to <code>TWIN_ALLOWED_ORIGINS</code> and restart FastAPI.</li><li>Use your laptop’s <code>TWIN_API_KEY</code> here, never your LLM provider key.</li></ol></details>
    {error && <p role="alert" className="error-message">{error}</p>}
    <button className="primary full-width" disabled={busy}>{busy ? <><LoaderCircle className="spin" size={17} />Connecting…</> : 'Connect workspace'}</button>
  </form>;
}

export default function App() {
  const [person, setPerson] = useState<Person>(example.person);
  const [traits, setTraits] = useState(example.traits);
  const [scenario, setScenario] = useState<Scenario>(example.scenario);
  const [evidence, setEvidence] = useState<Evidence>(example);
  const [messages, setMessages] = useState<Message[]>(starterMessages);
  const [session, setSession] = useState<string | null>(null);
  const [connection, setConnection] = useState<Connection>({ url: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' ? 'http://127.0.0.1:8000' : '', key: '' });
  const [connected, setConnected] = useState(false);
  const [exampleMode, setExampleMode] = useState(true);
  const [edited, setEdited] = useState(false);
  const [modal, setModal] = useState<'persona' | 'connection' | 'method' | null>(null);
  const [busy, setBusy] = useState<'run' | 'chat' | null>(null);
  const [error, setError] = useState('');
  const [question, setQuestion] = useState('');
  const [toast, setToast] = useState('');
  const [mobileNav, setMobileNav] = useState(false);
  const chatEnd = useRef<HTMLDivElement>(null);
  const scenarioRef = useRef<HTMLElement>(null);
  const chatRef = useRef<HTMLElement>(null);
  const current = evidence.alternative ?? evidence.baseline;
  const changed = JSON.stringify({ person, traits, scenario }) !== JSON.stringify({ person: evidence.person, traits: evidence.traits, scenario: evidence.scenario });
  const stale = changed || edited;

  useEffect(() => { if (messages.length > 1) chatEnd.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }, [messages.length]);
  useEffect(() => { if (!toast) return; const id = setTimeout(() => setToast(''), 4500); return () => clearTimeout(id); }, [toast]);

  async function explore() {
    if (!connected) { setModal('connection'); return; }
    setBusy('run'); setError('');
    try {
      const result = await api<{ session_id: string; evidence: Evidence; answer: string }>(connection, '/sessions', {
        person, traits, scenario, n_paths: 500,
        question: `Help me understand ${scenarioNames[scenario.kind].toLowerCase()}, how it affects my goal, and the trade-offs for what I value.`,
      });
      const previous = session;
      setEvidence(result.evidence); setSession(result.session_id); setExampleMode(false); setEdited(false);
      setMessages([{ role: 'assistant', content: result.answer }]); setQuestion('');
      if (previous) void api(connection, `/sessions/${previous}`, undefined, 'DELETE').catch(() => {});
      setToast('Your possibilities are ready. Let’s talk them through.');
    } catch (err) { setError((err as Error).message); } finally { setBusy(null); }
  }

  async function send(text = question) {
    if (!text.trim() || busy) return;
    if (!session || stale) { setError('Explore your current choices first, so the conversation uses the right evidence.'); return; }
    setBusy('chat'); setError('');
    try {
      const result = await api<{ answer: string }>(connection, `/sessions/${session}/messages`, { message: text.trim() });
      setMessages(m => [...m, { role: 'user', content: text.trim() }, { role: 'assistant', content: result.answer }]);
      setQuestion('');
    } catch (err) { setError((err as Error).message); } finally { setBusy(null); }
  }

  function savePersona(p: Person, t: Record<string, number>) {
    setPerson(p); setTraits(t); setEdited(true); setModal(null); setError('');
    setToast('Persona saved for this tab. Explore a path to update your results.');
  }
  function scrollTo(target: 'scenarios' | 'chat') { (target === 'scenarios' ? scenarioRef : chatRef).current?.scrollIntoView({ behavior: 'smooth', block: 'start' }); setMobileNav(false); }

  return <div className="app-shell">
    <a className="skip-link" href="#main">Skip to your overview</a>
    <aside className={`sidebar ${mobileNav ? 'open' : ''}`}>
      <a className="brand" href="#main" aria-label="Parallel home"><span className="brand-mark"><i /><i /></span>parallel<span className="brand-dot">.</span></a>
      <div className="workspace-label">YOUR FINANCIAL TWIN</div>
      <nav aria-label="Main navigation"><button className="nav-item selected" onClick={() => { window.scrollTo({ top: 0, behavior: 'smooth' }); setMobileNav(false); }}><Compass size={19} />My overview</button><button className="nav-item" disabled={!!busy} onClick={() => { setModal('persona'); setMobileNav(false); }}><Heart size={19} />My persona</button><button className="nav-item" onClick={() => scrollTo('scenarios')}><Leaf size={19} />Explore possibilities</button><button className="nav-item" onClick={() => scrollTo('chat')}><MessageCircle size={19} />My conversation</button></nav>
      <div className="sidebar-note"><div className="orb"><span /></div><h3>One life.<br />Many possibilities.</h3><p>You don’t need every answer to take your next step.</p></div>
      <div className="sidebar-bottom"><button className="nav-item" onClick={() => setModal('method')}><CircleHelp size={18} />How it works</button><button className="workspace-button" disabled={!!busy} onClick={() => setModal('connection')}><span className={`connection-dot ${connected ? 'online' : ''}`} /><span>{connected ? 'Workspace connected' : 'Connect workspace'}<small>{connected ? 'Powered by your laptop' : 'Start a live conversation'}</small></span><Settings2 size={16} /></button><span className="prototype-label">LAUZHACK 2026 · RESEARCH PROTOTYPE</span></div>
    </aside>
    <div className="workspace">
      <header className="topbar"><div><button className="icon-button mobile-menu" aria-label="Toggle navigation" aria-expanded={mobileNav} onClick={() => setMobileNav(!mobileNav)}>{mobileNav ? <X size={20} /> : <PanelLeftClose size={20} />}</button><span className="topbar-breadcrumb">Your space <span>/</span> <b>Overview</b></span></div><div className="topbar-right"><span className={`mode-badge ${exampleMode ? '' : 'live'}`}>{exampleMode ? 'Example walkthrough' : 'Live exploration'}</span><button className="avatar-button" disabled={!!busy} onClick={() => setModal('persona')} aria-label="Edit your persona">{person.name.slice(0, 1).toUpperCase()}</button></div></header>
      <main id="main">
        <div className="page-heading"><div><span className="eyebrow">A LITTLE CLARITY GOES A LONG WAY</span><h1>Your life. A little clearer.</h1><p>Explore what could change. Understand what matters to you.</p></div><button className="secondary create-persona" disabled={!!busy} onClick={() => setModal('persona')}><Plus size={18} />Build my persona</button></div>
        <section className="persona-strip" aria-label="Current persona"><div className="persona-avatar">{person.name.slice(0, 1).toUpperCase()}<span><Leaf size={12} /></span></div><div className="persona-summary"><strong>{person.name}’s financial twin</strong><span>{person.age} at start · {person.city} · {person.status.replaceAll('_', ' ')}</span></div><div className="persona-divider" /><div className="persona-priority"><span>WHAT MATTERS</span><strong><Heart size={14} />{traits.free_time >= .5 ? 'Time for life' : traits.career_drive >= .6 ? 'Room to grow' : 'A little more security'}</strong></div><div className="persona-goal"><span>WORKING TOWARDS</span><strong>{person.high_goal} · CHF {money(person.high_target)}</strong></div><button className="text-button" disabled={!!busy} onClick={() => setModal('persona')}>Edit persona <Settings2 size={15} /></button></section>
        {stale && <div className="change-notice" role="status"><Activity size={17} /><span>Your choices have changed. Explore again to update the results and conversation below.</span></div>}
        <div className="content-grid">
          <div className="exploration">
            <section className="scenario-section" ref={scenarioRef}>
              <div className="section-heading"><div><span className="eyebrow">START WITH A POSSIBILITY</span><h2>What’s on your mind?</h2></div><span className="small muted">You can always change your mind.</span></div>
              <div className="scenario-grid">{choices.map(({ kind, icon: Icon, title, description }) => <button className={`scenario-choice ${scenario.kind === kind ? 'active' : ''}`} aria-pressed={scenario.kind === kind} disabled={!!busy} key={kind} onClick={() => { setScenario(s => ({ ...s, kind })); setError(''); }}><span className="scenario-icon"><Icon size={20} />{scenario.kind === kind && <Check size={14} />}</span><strong>{title}</strong><span>{description}</span></button>)}</div>
              <div className="scenario-action">{scenario.kind === 'part_time' ? <label className="salary-field">Part-time take-home pay <span>CHF <input aria-label="Part-time monthly take-home pay" type="number" min="1" max="100000" value={scenario.salary || ''} onChange={e => setScenario(s => ({ ...s, salary: Number(e.target.value) }))} disabled={!!busy} /><small>/ month</small></span></label> : <p className="small muted">{scenario.kind === 'baseline' ? 'Explore the choices your persona might make over time.' : 'A change at the January 2025 review. Later decisions can evolve.'}</p>}<button className="primary" disabled={!!busy || (scenario.kind === 'part_time' && (!Number.isFinite(scenario.salary) || scenario.salary <= 0 || scenario.salary > 100000))} onClick={() => void explore()}>{busy === 'run' ? <><LoaderCircle className="spin" size={17} />Exploring your paths…</> : <><Sparkles size={17} />Explore this path</>}</button></div>
            </section>
            <div className="results-heading"><h2>{exampleMode ? 'A glimpse of Mia’s possibilities' : `${evidence.person.name}’s possibilities`}</h2><span>{exampleMode ? 'Example results' : stale ? 'Previous results' : 'Your results'} · {evidence.n_paths} paths</span></div>
            <div className="metric-grid"><article className="metric-card"><div><Target size={18} /><span>Goal within reach</span></div><strong>{percent(current.high_goal_odds)}<span>of paths</span></strong><p>{evidence.person.high_goal} by Dec 2029</p><small>{evidence.alternative ? `${percent(evidence.baseline.high_goal_odds)} on your current path` : 'Under the current assumptions'}</small></article><article className="metric-card"><div><Wallet size={18} /><span>Future net worth</span></div><strong><span>CHF</span> {money(current.final_net_worth.p50)}</strong><p>Median at the end of 2034</p><small>{evidence.alternative ? `CHF ${money(evidence.baseline.final_net_worth.p50)} on your current path` : 'Cash + investments − debt'}</small></article><article className="metric-card"><div><ShieldCheck size={18} /><span>Credit limit risk</span></div><strong>{percent(current.credit_breach_odds)}<span>of paths</span></strong><p>Exceed the limit by end of 2034</p><small>{evidence.alternative ? `${percent(evidence.baseline.credit_breach_odds)} on your current path` : 'Across the full ten-year horizon'}</small></article></div>
            <Chart evidence={evidence} />
            <section className="insight-card"><div className="insight-icon"><BookOpen size={20} /></div><div><span className="eyebrow">A NOTE ON THE NUMBERS</span><h3>Possibilities, with perspective.</h3><p>{evidence.comparison ? `The goal-success difference is ${evidence.comparison.goal_change_pp.toFixed(1)} percentage points. Its sampling interval runs from ${evidence.comparison.goal_change_mc95_pp[0].toFixed(1)} to ${evidence.comparison.goal_change_mc95_pp[1].toFixed(1)} points. That captures simulation noise, not every uncertainty in life.` : 'These paths combine your preferences with assumptions about income, living costs, markets, and life events. They help you ask better questions about your next step.'}</p><button className="text-button" onClick={() => setModal('method')}>Understand the assumptions <ChevronDown size={15} /></button></div></section>
          </div>
          <aside className="conversation card" ref={chatRef} aria-label="Financial twin conversation">
            <div className="conversation-header"><span className="assistant-avatar"><Sparkles size={20} /></span><div><h2>Let’s think it through</h2><span>Your space to ask, explore, and reflect.</span></div></div>
            <div className="conversation-context"><Leaf size={15} /><span>Exploring: {scenarioNames[evidence.scenario.kind]}</span></div>
            <div className="message-list" aria-live="polite" aria-relevant="additions"><div className="conversation-date">{exampleMode ? 'ILLUSTRATIVE CONVERSATION' : 'GROUNDED IN YOUR SIMULATION'}</div>{messages.map((m, i) => <div className={`message ${m.role}`} key={i}><div className="message-byline">{m.role === 'assistant' ? <><span className="mini-brand">Ⅱ</span>Parallel {exampleMode && <span className="example-label">Example script</span>}</> : 'You'}</div><div className="message-content">{m.content.split('\n\n').map((paragraph, j) => <p key={j}>{paragraph}</p>)}</div></div>)}{busy && <div className="thinking" role="status"><LoaderCircle size={17} className="spin" />{busy === 'run' ? 'Exploring 500 paths, then making sense of them…' : 'Thinking through your question…'}</div>}<div ref={chatEnd} /></div>
            {exampleMode ? <div className="conversation-invite"><span className="eyebrow">MAKE IT YOURS</span><p>This walkthrough uses a fictional persona and real simulation results. Connect your workspace to start a live LLM conversation.</p><button className="secondary full-width" disabled={!!busy} onClick={() => connected ? void explore() : setModal('connection')}>{connected ? 'Start my live exploration' : 'Connect my workspace'}</button></div> : <div className="suggested-questions"><span className="eyebrow">A GOOD QUESTION TO START WITH</span>{['What is the biggest trade-off?', 'How uncertain are these results?'].map(q => <button disabled={!!busy || stale} key={q} onClick={() => void send(q)}>{q}<Plus size={13} /></button>)}</div>}
            {error && <div className="error-message" role="alert">{error}<button className="icon-button" onClick={() => setError('')} aria-label="Dismiss error"><X size={16} /></button></div>}
            <form className="chat-composer" onSubmit={e => { e.preventDefault(); void send(); }}><label className="sr-only" htmlFor="question">Ask about your financial future</label><textarea id="question" rows={2} maxLength={2000} value={question} disabled={exampleMode || stale || !!busy} onChange={e => setQuestion(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); void send(); } }} placeholder={exampleMode ? 'Connect to start your conversation…' : stale ? 'Explore your updated choices first…' : 'What would you like to understand?'} /><button type="submit" aria-label="Send message" disabled={!question.trim() || !!busy || exampleMode || stale}><Send size={18} /></button></form>
            <p className="conversation-footnote">A thinking partner, not financial advice.</p>
          </aside>
        </div>
        <footer className="page-footer"><span><ShieldCheck size={14} />Built for thoughtful decisions.</span><button onClick={() => setModal('method')}>Synthetic model · 2025–2034 · How it works</button></footer>
      </main>
    </div>
    {modal === 'persona' && <Dialog title="Build your persona" wide onClose={() => setModal(null)}><PersonaBuilder person={person} traits={traits} onSave={savePersona} onClose={() => setModal(null)} /></Dialog>}
    {modal === 'connection' && <Dialog title="Connect workspace" onClose={() => setModal(null)}><ConnectionForm connection={connection} onConnect={c => { if (c.url !== connection.url || c.key !== connection.key) { setSession(null); setExampleMode(true); setEvidence(example); setMessages(starterMessages); setEdited(true); } setConnection(c); setConnected(true); setModal(null); setError(''); setToast('Connected. Choose a possibility and select Explore this path.'); }} /></Dialog>}
    {modal === 'method' && <Dialog title="How your financial twin works" onClose={() => setModal(null)}><div className="modal-heading"><span className="eyebrow">CLARITY, WITH CONTEXT</span><h2>A twin, not a crystal ball.</h2><p>Here’s what sits behind your conversation.</p></div><div className="method-list"><article><span><Heart size={21} /></span><div><h3>You set the starting point</h3><p>Your balances, life situation, goal, and preferences shape the decisions your twin might make. Persona details stay in this tab until you explore.</p></div></article><article><span><Activity size={21} /></span><div><h3>Your laptop explores possibilities</h3><p>A learned cash-flow model and a rule-based persona simulate 500 paths from January 2025 to December 2034. Choices happen each January; events and cash flows evolve monthly. This is a fixed historical starting point, not a forecast beginning today.</p></div></article><article><span><Sparkles size={21} /></span><div><h3>The assistant explains the evidence</h3><p>The LLM receives relevant persona details (without your name), results, and your conversation to explain trade-offs. Follow-up questions discuss that run. Change the controls and explore again for a new comparison.</p></div></article><article><span><Target size={21} /></span><div><h3>Success has a specific meaning</h3><p>Reach the target by December 2029 without exceeding the credit limit before then. Emergency savings is measured in cash; other goals use net worth. Later credit breaches still appear in the risk card.</p></div></article></div><div className="info-note"><ShieldCheck size={19} /><span>Research prototype trained on synthetic data. No live bank connection. Sessions stay in laptop memory for up to an hour and disappear on restart. This version uses one shared workspace access key.</span></div></Dialog>}
    {toast && <div className="toast" role="status"><Check size={17} />{toast}</div>}
  </div>;
}
