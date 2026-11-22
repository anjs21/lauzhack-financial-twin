import type { ReactNode } from 'react';
import source from '../../docs/model-assumptions.md?raw';

/** Inline Markdown: **bold**, *italic*, `code` and [links](...) shown as their text. */
function inline(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*|\[[^\]]+\]\([^)]+\))/g).filter(Boolean).map((part, i) =>
    part.startsWith('**') ? <b key={i}>{part.slice(2, -2)}</b>
      : part.startsWith('`') ? <code key={i}>{part.slice(1, -1)}</code>
      : part.startsWith('[') ? <span key={i}>{inline(part.slice(1, part.indexOf(']')))}</span>
      : part.startsWith('*') && part.length > 2 ? <i key={i}>{part.slice(1, -1)}</i>
      : part);
}

const cells = (row: string) => row.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());

/** The blocks of one section: paragraphs, tables and lists. */
function blocks(lines: string[]): ReactNode[] {
  const out: ReactNode[] = [];
  for (let i = 0; i < lines.length;) {
    const line = lines[i];
    if (!line.trim() || /^-{3,}$/.test(line.trim())) { i++; continue; }        // blank lines and rules
    if (line.startsWith('|')) {
      const rows: string[] = [];
      while (i < lines.length && lines[i].startsWith('|')) rows.push(lines[i++]);
      const [head, , ...body] = rows;
      out.push(<div className="table" key={i}><table>
        <thead><tr>{cells(head).map((c, j) => <th key={j}>{inline(c)}</th>)}</tr></thead>
        <tbody>{body.map((r, k) => <tr key={k}>{cells(r).map((c, j) => <td key={j}>{inline(c)}</td>)}</tr>)}</tbody>
      </table></div>);
    } else if (line.startsWith('- ')) {
      const items: string[] = [];
      while (i < lines.length && lines[i].startsWith('- ')) items.push(lines[i++].slice(2));
      out.push(<ul key={i}>{items.map((t, k) => <li key={k}>{inline(t)}</li>)}</ul>);
    } else {
      const para: string[] = [];
      while (i < lines.length && lines[i].trim() && !lines[i].startsWith('|') && !lines[i].startsWith('- ') && !lines[i].startsWith('#')) para.push(lines[i++]);
      out.push(<p key={i}>{inline(para.join(' '))}</p>);
    }
  }
  return out;
}

// docs/model-assumptions.md, split into its numbered sections; the text before them is the introduction.
const sections = source.split(/^## /m);
const intro = sections[0].split('\n').filter(l => !l.startsWith('# '));
const parts = sections.slice(1).map(s => { const [title, ...rest] = s.split('\n'); return { title: title.trim(), lines: rest }; });

export default function Assumptions() {
  return <div className="assumptions">
    <div className="modal-heading"><h2>Assumptions</h2></div>
    <div className="assumptions-intro">{blocks(intro)}</div>
    {parts.map(p => <details key={p.title}><summary>{p.title}</summary>{blocks(p.lines)}</details>)}
  </div>;
}
