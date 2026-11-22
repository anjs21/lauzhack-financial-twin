import { LoaderCircle, X } from 'lucide-react';
import type { ChoiceKey, PathNode } from './types';
import { CHOICES, compact, percent, yearOf } from './types';

/** The paths as a tree: roots first, each branch indented under the path it grew from. */
export function Tree({ nodes, activeId, onSelect, onRemove }: {
  nodes: PathNode[]; activeId: string; onSelect: (id: string) => void; onRemove: (id: string) => void;
}) {
  const rows: { node: PathNode; depth: number }[] = [];
  const walk = (parent: string | null, depth: number) => nodes.filter(n => n.parent === parent).forEach(node => {
    rows.push({ node, depth }); walk(node.id, depth + 1);
  });
  walk(null, 0);
  return <section className="tree" aria-label="Paths">
    <h3>Paths</h3>
    <ul>{rows.map(({ node, depth }) => {
      const parent = nodes.find(n => n.id === node.parent);
      const nw = node.summary.final_net_worth.p50;
      const delta = parent ? nw - parent.summary.final_net_worth.p50 : null;
      return <li key={node.id} className={node.id === activeId ? 'active' : ''} style={{ paddingLeft: 4 + depth * 20 }}>
        <button className="tree-row" aria-pressed={node.id === activeId} onClick={() => onSelect(node.id)}>
          {depth > 0 && <span className="elbow" aria-hidden="true" />}
          <i style={{ background: node.color }} aria-hidden="true" />
          <span className="tree-label">{node.label}{node.applied !== undefined && node.applied < 0.95 &&
            <small> · possible in {percent(node.applied)} of lives</small>}</span>
          <span className="tree-stats">
            <span title="Chance of reaching the goal by Dec 2029">{percent(node.summary.high_goal_odds)}</span>
            <span title="Median net worth, end of 2034">CHF {compact(nw)}</span>
            {delta !== null && <span className={delta >= 0 ? 'up' : 'down'}>{delta >= 0 ? '+' : '−'}{compact(Math.abs(delta))}</span>}
          </span>
        </button>
        {node.decisions.length > 0 && <button className="icon-button tree-remove" aria-label={`Remove ${node.label}`} onClick={() => onRemove(node.id)}><X size={14} /></button>}
      </li>;
    })}</ul>
    <p className="tree-key">Goal chance · net worth 2034 · change from the path it branched from</p>
  </section>;
}

/** What would you do at this January review? One tap grows a branch. */
export function DecisionSheet({ month, from, busy, canBranch, reason, onChoose, onClose, onConnect }: {
  month: number; from: PathNode; busy: boolean; canBranch: boolean; reason: string;
  onChoose: (choice: ChoiceKey) => void; onClose: () => void; onConnect?: () => void;
}) {
  // January 2025 is the scenario's own review: only life events can be imposed then.
  const groups = month === 0 ? ['Life events'] as const : ['Work', 'Home', 'Money', 'Life events'] as const;
  return <section className="sheet" aria-label={`Decision in January ${yearOf(month)}`}>
    <div className="sheet-head">
      <h3>{month === 0 ? 'Now, in January 2025' : `In January ${yearOf(month)}`}, on <span style={{ color: from.color }}>{from.label}</span>, what if…</h3>
      <button className="icon-button" aria-label="Close" onClick={onClose}><X size={16} /></button>
    </div>
    {canBranch
      ? groups.map(g => <div className="sheet-group" key={g}><span>{g}</span><div className="chips">{CHOICES.filter(c => c.group === g).map(c =>
          <button key={c.key} className="chip" disabled={busy} onClick={() => onChoose(c.key)}>{c.label}</button>)}</div></div>)
      : <p className="note">{reason}{onConnect && <> <button className="text-button" onClick={onConnect}>Connect my workspace</button></>}</p>}
    {busy && <p className="thinking" role="status"><LoaderCircle size={16} className="spin" />Growing the branch…</p>}
    {canBranch && <p className="tree-key">Same lives up to January {yearOf(month)}; from there, this decision or event wherever it can happen.</p>}
  </section>;
}
