import type { Moment } from './types';
import { percent } from './types';

const lives = (share: number) => share >= 0.995 ? 'every life' : `${percent(share)} of lives`;

/** Key moments of the highlighted path: when its big decisions and life events usually happen. */
export default function Moments({ moments, label, color }: { moments?: Moment[]; label: string; color: string }) {
  if (!moments) return null;                                 // evidence from before key moments existed
  return <section className="moments" aria-label="Key moments">
    <div className="moments-head"><h3>Key moments · <span style={{ color }}>{label}</span></h3></div>
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
