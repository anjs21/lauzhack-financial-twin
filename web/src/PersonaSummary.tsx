import { PenLine } from 'lucide-react';
import type { Person } from './types';
import { money } from './types';

const WORK: Record<Person['status'], string> = {
  employed: 'works full-time', part_time: 'works part-time', self_employed: 'self-employed',
  student: 'student', unemployed: 'not working right now', retired: 'retired',
};
const HOME: Record<Person['household_type'], string> = {
  single: 'lives alone', shared_housing: 'shares a flat', couple: 'lives with a partner', family: 'family with children',
};
// What each preference means at either end, as a viewer would say it.
const TRAIT: Record<string, [low: string, high: string]> = {
  career_drive: ['career is not the priority', 'ambitious about career and studies'],
  goal_focus: ['relaxed about savings goals', 'watches goals and deadlines closely'],
  frugality: ['spends first, saves what is left', 'saves first, spends what is left'],
  free_time: ['puts income before free time', 'would trade income for free time'],
  housing_aspiration: ['happy with a modest home', 'will pay for a nicer home'],
  risk_tolerance: ['avoids investment risk', 'comfortable with investment risk'],
  inertia: ['open to changing course', 'rarely changes what works'],
  impatience: ['patient with money', 'would rather enjoy money now'],
};

function situation(p: Person) {
  if (p.status !== 'student') return WORK[p.status];
  const left = p.study_months_left > 1 ? `, ${p.study_months_left} months left` : '';
  return `${p.in_masters ? 'master’s student' : 'student'}${left}`;
}

/** Who the person is, in one glance: the context viewers need before any numbers. */
export default function PersonaSummary({ person, traits, story, example, onEdit }: {
  person: Person; traits: Record<string, number>; story?: string; example: boolean; onEdit: () => void;
}) {
  const leanings = Object.entries(traits)
    .filter(([t, v]) => TRAIT[t] && Math.abs(v - 0.5) >= 0.15)
    .sort((a, b) => Math.abs(b[1] - 0.5) - Math.abs(a[1] - 0.5))
    .slice(0, 4)
    .map(([t, v]) => TRAIT[t][v > 0.5 ? 1 : 0]);
  const facts = [
    ['Takes home', `CHF ${money(person.income)} / month`],
    ['Rent', `CHF ${money(person.rent)} / month`],
    ['Cash', `CHF ${money(person.cash)}`],
    ...(person.investments > 0 ? [['Invested', `CHF ${money(person.investments)}`]] : []),
    ...(person.debt > 0 ? [['Owes', `CHF ${money(person.debt)}${person.rate > 0 ? ` at ${+(person.rate * 100).toFixed(1)}%` : ''}`]] : []),
  ];
  return <section className="persona" aria-label="Who this is">
    <div className="persona-head">
      <p className="persona-who"><b>{person.name}</b> · {person.age} · {person.city} · {situation(person)} · {HOME[person.household_type]}</p>
      {example && <span className="tag">Fictional</span>}
    </div>
    <dl className="persona-facts">{facts.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl>
    <p className="persona-goal">Goal: <b>CHF {money(person.high_target)} {person.high_goal.toLowerCase()}</b> by December 2029</p>
    {leanings.length > 0 && <p className="persona-traits">{leanings.join(' · ')}</p>}
    {story && <p className="persona-story">{story}</p>}
    <button className="text-button" onClick={onEdit}><PenLine size={14} />Edit this person</button>
  </section>;
}
