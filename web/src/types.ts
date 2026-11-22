export type Person = {
  name: string; age: number; city: string; status: string; household_type: string;
  income: number; rent: number; cash: number; investments: number; debt: number;
  rate: number; high_goal: string; high_target: number;
  study_months_left: number; in_masters: boolean;
};
// Older evidence (and the bundled example) predates the study fields; append defaults in the API's key order.
export const withStudyDefaults = (p: Omit<Person, 'study_months_left' | 'in_masters'> & Partial<Person>): Person =>
  ({ ...p, study_months_left: p.study_months_left ?? 1, in_masters: p.in_masters ?? false });
export type Scenario = { kind: 'baseline' | 'part_time' | 'spend_less' | 'cheaper_home'; salary: number; alongside_studies: boolean };
export const withScenarioDefaults = (s: Omit<Scenario, 'alongside_studies'> & Partial<Scenario>): Scenario =>
  ({ ...s, alongside_studies: s.alongside_studies ?? false });
export type Moment = { moment: string; kind: 'decision' | 'event'; share: number; typical_year: number; share_by_goal_deadline: number };
export type Summary = {
  months: string[]; fan_cash: number[][]; fan_net_worth: number[][];
  high_goal_odds: number; shortfall_odds: number; credit_breach_odds: number;
  final_net_worth: { p5: number; p50: number; p95: number };
  stories?: { story: string; share: number; high_goal_odds: number }[];
  key_moments?: Moment[];
};

// Decisions the chart can branch on (keys match the API's CHOICES).
export const CHOICES = [
  { key: 'full_time', label: 'Work full-time', group: 'Work' },
  { key: 'part_time', label: 'Work part-time', group: 'Work' },
  { key: 'masters', label: 'Start a master’s', group: 'Work' },
  { key: 'stop_work', label: 'Stop working', group: 'Work' },
  { key: 'cheaper_home', label: 'Move somewhere cheaper', group: 'Home' },
  { key: 'bigger_home', label: 'Move somewhere bigger', group: 'Home' },
  { key: 'spend_less', label: 'Spend less', group: 'Money' },
  { key: 'spend_more', label: 'Spend more', group: 'Money' },
  { key: 'invest', label: 'Invest spare cash', group: 'Money' },
  // Life events happen to the person; they can be imposed from January 2025 ("now").
  { key: 'lose_job', label: 'Lose my job', group: 'Life events' },
  { key: 'recession', label: 'A recession hits', group: 'Life events' },
  { key: 'have_child', label: 'Have a child', group: 'Life events' },
  { key: 'partner', label: 'Partner moves in', group: 'Life events' },
  { key: 'separate', label: 'Separate', group: 'Life events' },
] as const;
export type ChoiceKey = typeof CHOICES[number]['key'];
export type Decision = { month: number; choice: ChoiceKey };
export type BranchResult = Summary & { applied_share: number; root: 'baseline' | 'alternative'; decisions: (Decision & { when: string; label: string })[] };
// One path in the tree: a root (current path or the selected scenario) or a branch grown from the chart.
export type PathNode = {
  id: string; parent: string | null; root: 'baseline' | 'alternative'; label: string; color: string;
  from: number; decisions: Decision[]; summary: Summary; applied?: number;
};
export const MAX_DECISIONS = 4;
export const REVIEW_MONTHS = [12, 24, 36, 48, 60, 72, 84, 96, 108];   // January 2026 .. January 2034
export const yearOf = (month: number) => 2025 + Math.floor(month / 12);
export type Evidence = {
  person: Person; traits: Record<string, number>; scenario: Scenario; n_paths: number; seed: number;
  baseline: Summary; alternative: Summary | null;
  comparison: { goal_change_pp: number; goal_change_mc95_pp: number[]; median_final_net_worth: number[] } | null;
  assumptions: string[];
  deliberate?: boolean;
};
export type Message = { role: 'user' | 'assistant'; content: string; notes?: string[] };
// A branch as the laptop API returns it (it assigns the id); the client adds a colour.
export type BranchNode = Omit<PathNode, 'color'>;
export type Measure = 'fan_cash' | 'fan_net_worth';
// Chart updates the assistant's tools ask for, applied in order.
export type ChatAction =
  | { type: 'rerun'; evidence: Evidence }
  | { type: 'branch'; branch: BranchNode }
  | { type: 'highlight'; path_id: string; measure: 'cash' | 'net_worth' | null };
export type ChatResult = { answer: string; actions?: ChatAction[]; notes?: string[] };
export type Connection = { url: string; key: string };
export const money = (value: number) => new Intl.NumberFormat('en-CH', { maximumFractionDigits: 0 }).format(value);
export const percent = (value: number) => `${Math.round(value * 100)}%`;
export const compact = (value: number) => new Intl.NumberFormat('en-CH', { notation: 'compact', maximumFractionDigits: Math.abs(value) >= 1e6 ? 1 : 0 }).format(value);
export const scenarioNames = { baseline: 'Staying on my path', part_time: 'Going part-time', spend_less: 'Spending less', cheaper_home: 'Moving to a cheaper home' };

export async function api<T>(connection: Connection, path: string, body?: unknown, method = body ? 'POST' : 'GET'): Promise<T> {
  const url = new URL(connection.url);
  if (url.username || url.password || url.search || url.hash || url.pathname !== '/') throw new Error('Use the base workspace URL, without a path or credentials.');
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && ['localhost', '127.0.0.1'].includes(url.hostname))) throw new Error('Use an HTTPS workspace URL (or localhost for local development).');
  if (window.location.protocol === 'https:' && url.protocol !== 'https:') throw new Error('This hosted page needs an HTTPS tunnel to your laptop.');
  let response: Response;
  try {
    response = await fetch(`${url.origin}${path}`, {
      method, headers: { Authorization: `Bearer ${connection.key}`, ...(body ? { 'Content-Type': 'application/json' } : {}) },
      body: body ? JSON.stringify(body) : undefined, signal: AbortSignal.timeout(120000), cache: 'no-store', redirect: 'error',
    });
  } catch {
    throw new Error('Your workspace could not be reached. Check that the laptop API and tunnel are running, and that this site is in TWIN_ALLOWED_ORIGINS.');
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Some persona details are invalid. Check your entries and try again.');
  return data as T;
}
