import type { ReactNode } from 'react';
import { chfText } from './types';

// Figures worth spotting at a glance: CHF amounts, percentages and "6 in 10".
const FIGURE = /(CHF\s?\d[\d,]*(?:\.\d+)?|\d+(?:\.\d+)?%|\b\d+ (?:in|of|out of) \d+\b)/g;

function emphasize(text: string): ReactNode[] {
  return text.split(FIGURE).map((part, i) => i % 2 ? <b key={i}>{part}</b> : part);
}

const BULLET = /^\s*(?:[-•*]|\d+[.)])\s+/;

/**
 * An assistant reply, easy to scan: the first paragraph is the short answer, lines starting
 * with "- " become a list, and figures are bold. Plain paragraphs from older replies still work.
 */
export default function Answer({ text }: { text: string }) {
  const blocks = chfText(text).trim().split(/\n\s*\n/);
  return <>{blocks.map((block, i) => {
    const lines = block.split('\n').map(l => l.trim()).filter(Boolean);
    if (lines.length && lines.every(l => BULLET.test(l)))
      return <ul key={i} className="answer-list">{lines.map((l, j) => <li key={j}>{emphasize(l.replace(BULLET, ''))}</li>)}</ul>;
    // A paragraph followed directly by bullet lines (no blank line between them).
    const first = lines.findIndex(l => BULLET.test(l));
    if (first > 0 && lines.slice(first).every(l => BULLET.test(l)))
      return <div key={i}>
        <p className={i === 0 ? 'lead' : undefined}>{emphasize(lines.slice(0, first).join(' '))}</p>
        <ul className="answer-list">{lines.slice(first).map((l, j) => <li key={j}>{emphasize(l.replace(BULLET, ''))}</li>)}</ul>
      </div>;
    return <p key={i} className={i === 0 && blocks.length > 1 ? 'lead' : undefined}>{emphasize(lines.join(' '))}</p>;
  })}</>;
}
