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
export type Summary = {
  months: string[]; fan_cash: number[][]; fan_net_worth: number[][];
  high_goal_odds: number; shortfall_odds: number; credit_breach_odds: number;
  final_net_worth: { p5: number; p50: number; p95: number };
  stories: { story: string; share: number; high_goal_odds: number }[];
  key_moments?: { moment: string; kind: 'decision' | 'event'; share: number; typical_year: number; share_by_goal_deadline: number }[];
};
export type Evidence = {
  person: Person; traits: Record<string, number>; scenario: Scenario; n_paths: number; seed: number;
  baseline: Summary; alternative: Summary | null;
  comparison: { goal_change_pp: number; goal_change_mc95_pp: number[]; median_final_net_worth: number[] } | null;
  assumptions: string[];
  deliberate?: boolean;
  plan?: { narrative: string; spontaneity: number; steps: string[]; carried_out: StepResult[] | null };
};

// Long-term plans (POST /plans, GET /plans/{id}): written by the planner agent, carried out by the simulator.
export type PlanStep = {
  id: string; label: string; head: 'status' | 'housing' | 'spending' | 'assets'; option: string;
  start?: string; until?: string; commitment: number; after?: string; after_months?: number;
  event?: string; within_months?: number; cash_months_below?: number; cash_months_above?: number;
  min_age?: number; max_age?: number; repeat?: boolean;
};
export type StepResult = {
  id: string; label: string; head: string; option: string; taken_share: number; typical_year: number | null;
  blocked_share: number; declined_share: number; turned_down_share: number; trigger_share: number; blocked_because?: string;
};
export type PlanVersion = {
  version: number; kind: 'draft' | 'nudge'; said: string; as_of: string | null; signed: boolean;
  plan: { narrative: string; spontaneity: number; notes: string[]; steps: PlanStep[] };
  evaluation: { n_lives: number; steps: StepResult[]; outcomes: { high_goal_odds: number; credit_breach_odds: number; final_net_worth: { p50: number } } };
  changes: string;
  comparison: { high_goal_odds: number[]; credit_breach_odds: number[]; median_final_net_worth: number[] };
};
export type PlanState = {
  plan_id?: string; status: 'working' | 'ready' | 'failed'; progress: string; error: string | null;
  latest: PlanVersion | null; versions: { version: number; kind: string; said: string; as_of: string | null }[];
};
export type Message = { role: 'user' | 'assistant'; content: string };
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
