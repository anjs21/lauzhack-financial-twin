import { useState } from 'react';
import type { Evidence } from './types';
import { percent } from './types';

const lives = (share: number) => share >= 0.995 ? 'every life' : `${percent(share)} of lives`;

export default function Moments({ evidence }: { evidence: Evidence }) {
  const [which, setWhich] = useState<'alternative' | 'baseline'>('alternative');
  if (!evidence.baseline.key_moments) return null;           // evidence from before key moments existed
  const hasAlt = !!evidence.alternative;
  const path = hasAlt && which === 'alternative' ? evidence.alternative! : evidence.baseline;
  const moments = path.key_moments ?? [];
  return <section className="moments" aria-label="Key moments">
    <div className="moments-head">
      <h3>Key moments</h3>
      {hasAlt && <div className="toggle" role="group" aria-label="Key moments for">{(['alternative', 'baseline'] as const).map(key =>
        <button key={key} aria-pressed={which === key} onClick={() => setWhich(key)}>{key === 'alternative' ? 'Your choice' : 'Current path'}</button>)}
      </div>}
    </div>
    {moments.length
      ? <ol>{moments.map(m => <li key={m.moment} className={m.kind}>
          <span className="year">{m.typical_year}</span>
          <span className="what"><i aria-hidden="true" />{m.moment}<span className="sr-only"> ({m.kind === 'event' ? 'life event' : 'decision'})</span></span>
          <span className="share">{lives(m.share)}</span>
        </li>)}</ol>
      : <p className="note">No big change happens in most lives.</p>}
    <p className="moments-key">Year it usually first happens · <span className="decision"><i />your decisions</span> <span className="event"><i />life events</span></p>
  </section>;
}
