import { useState } from 'react';
import { Check, SlidersHorizontal } from 'lucide-react';
import type { Person } from './types';

export const cities = ['Basel', 'Bern', 'Biel', 'Fribourg', 'Geneva', 'Lausanne', 'Lucerne', 'Lugano', 'Neuchatel', 'St Gallen', 'Winterthur', 'Zurich'];
const traitsInfo: Record<string, [string, string, string]> = {
  free_time: ['Room for life', 'Income comes first', 'More time for myself'],
  risk_tolerance: ['Comfort with uncertainty', 'Prefer stability', 'Comfortable with risk'],
  frugality: ['Everyday spending', 'Enjoy the present', 'Save first'],
  career_drive: ['Career ambition', 'Content where I am', 'Ready for the next step'],
  housing_aspiration: ['My home', 'Keep it simple', 'More space matters'],
  inertia: ['Changing habits', 'Open to change', 'Prefer a familiar routine'],
  impatience: ['When to enjoy my money', 'Happy to wait', 'Enjoy it now'],
  goal_focus: ['Staying on track', 'Keep things flexible', 'Committed to my goal'],
};

export default function PersonaBuilder({ person, traits, onSave, onClose }: { person: Person; traits: Record<string, number>; onSave: (p: Person, t: Record<string, number>) => void; onClose: () => void }) {
  const [draft, setDraft] = useState<Person>({ ...person });
  const [values, setValues] = useState({ ...traits });
  const [step, setStep] = useState(0);
  const [advanced, setAdvanced] = useState(false);
  const set = (key: keyof Person, value: string | number | boolean) => setDraft(p => ({ ...p, [key]: value }));
  function number(key: keyof Person, label: string, min = 0, max = 10000000, increment = '1') {
    return <label className="field">{label}<div className="money-input"><span>{key === 'age' ? 'Years' : key === 'rate' ? 'Rate' : key === 'study_months_left' ? 'Months' : 'CHF'}</span><input type="number" required min={min} max={max} step={increment} value={draft[key] as number | string} onChange={e => set(key, e.target.value === '' ? '' : Number(e.target.value))} /></div></label>;
  }
  return <form onSubmit={e => { e.preventDefault(); if (step < 2) setStep(step + 1); else onSave(draft, values); }}>
    <div className="modal-heading"><h2>Your persona</h2></div>
    <div className="steps" aria-label="Persona steps">{['Your life', 'Your finances', 'What matters'].map((label, i) => <span key={label} className={i === step ? 'active' : i < step ? 'complete' : ''}><b>{i < step ? <Check size={14} /> : i + 1}</b>{label}</span>)}</div>
    {step === 0 && <div className="form-grid">
      <label className="field">What should we call you?<input required maxLength={60} value={draft.name} onChange={e => set('name', e.target.value)} placeholder="First name or a nickname" /></label>
      {number('age', 'Age at the start of 2025', 18, 90)}
      <label className="field">Where do you live?<select value={draft.city} onChange={e => set('city', e.target.value)}>{cities.map(c => <option key={c}>{c}</option>)}</select></label>
      <label className="field">Your household<select value={draft.household_type} onChange={e => set('household_type', e.target.value)}>{Object.entries({ single: 'Living on my own', shared_housing: 'Sharing a home', couple: 'Living with a partner', family: 'Family with children' }).map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select></label>
      <label className="field full">Your work situation<select value={draft.status} onChange={e => set('status', e.target.value)}>{Object.entries({ employed: 'Working full-time', part_time: 'Working part-time', self_employed: 'Self-employed', student: 'Studying', unemployed: 'Not currently working', retired: 'Retired' }).map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select></label>
      {draft.status === 'student' && <>
        {number('study_months_left', 'Months until you finish', 1, 72)}
        <label className="check">
          <input type="checkbox" checked={draft.in_masters} onChange={e => set('in_masters', e.target.checked)} />It’s a master’s degree
        </label>
      </>}
    </div>}
    {step === 1 && <div className="form-grid">
      <p className="full note">Figures for January 2025. Estimates are fine.</p>
      {number('income', 'Monthly take-home income', 0, 100000)}{number('rent', 'Monthly rent', 1, 30000)}
      {number('cash', 'Cash & savings')}{number('investments', 'Investments')}{number('debt', 'Outstanding debt')}{number('rate', 'Annual debt rate (0.05 = 5%)', 0, 0.5, '0.001')}
      <p className="full note">Other living costs are estimated from your city and situation.</p>
    </div>}
    {step === 2 && <>
      <div className="form-grid"><label className="field">Your main money goal<select value={draft.high_goal} onChange={e => set('high_goal', e.target.value)}>{['Emergency savings', 'Housing deposit', 'Education flexibility', 'Career transition', 'Retirement reserve'].map(g => <option key={g}>{g}</option>)}</select></label>{number('high_target', 'Target by December 2029', 1)}</div>
      <h3 className="values-title">What matters to you</h3>
      <div className="trait-list">{Object.entries(traitsInfo).slice(0, advanced ? 8 : 4).map(([key, [title, low, high]]) => <label key={key} className="trait"><span>{title}<b>{Math.round(values[key] * 100)} / 100</b></span><input type="range" min="0" max="1" step="0.05" value={values[key]} onChange={e => setValues(v => ({ ...v, [key]: Number(e.target.value) }))} /><span className="trait-extremes"><span>{low}</span><span>{high}</span></span></label>)}</div>
      <button className="text-button" type="button" onClick={() => setAdvanced(!advanced)}><SlidersHorizontal size={15} />{advanced ? 'Fewer' : 'More preferences'}</button>
    </>}
    <footer className="modal-actions"><button type="button" className="secondary" onClick={() => step ? setStep(step - 1) : onClose()}>{step ? 'Back' : 'Cancel'}</button><button className="primary" type="submit">{step === 2 ? 'Save my persona' : 'Continue'}</button></footer>
  </form>;
}
