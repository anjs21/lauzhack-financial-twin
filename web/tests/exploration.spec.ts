import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';

const example = JSON.parse(readFileSync(new URL('../src/example.json', import.meta.url), 'utf8'));

test('example, persona creation, and keyboard-accessible chart', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Hi P0067' })).toBeVisible();
  // The landing explains who this is before any numbers.
  const who = page.getByRole('region', { name: 'Who this is' });
  await expect(who).toContainText('master’s student, 14 months left');
  await expect(who).toContainText("CHF 40,000 housing deposit");
  await expect(who).toContainText('ambitious about career and studies');
  await expect(who).toContainText('The decision, January 2025');
  // A clean landing: only the first actions, no chart or conversation yet.
  await expect(page.getByRole('region', { name: 'Your paths' })).toHaveCount(0);
  await expect(page.getByLabel('Ask about your financial future')).toHaveCount(0);
  await page.getByRole('button', { name: 'See an example' }).click();
  await expect(page.getByText('Example', { exact: true })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Your paths' })).toBeVisible();
  await expect(page.getByLabel('Ask about your financial future')).toBeDisabled();
  // Net worth is the default measure; Cash comes with its definition.
  await expect(page.getByRole('button', { name: 'Net worth', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Cash', exact: true }).click();
  await expect(page.getByText('Cash is the money in the bank account', { exact: false })).toBeVisible();
  await page.getByRole('button', { name: 'Net worth', exact: true }).click();
  const moments = page.getByRole('region', { name: 'Key moments' });
  await expect(moments.getByText('Takes a job alongside studies')).toBeVisible();
  await expect(moments.getByText('every life')).toBeVisible();
  await page.getByRole('region', { name: 'Paths' }).getByRole('button', { name: /^Studies only/ }).click();
  await expect(moments.getByText('Takes a job alongside studies')).toBeHidden();
  const year = page.getByLabel('Explore a year');
  await year.focus(); await year.press('ArrowRight');
  await expect(year).toHaveValue('72');
  // The example has decision points too; growing a branch needs a connected workspace.
  await page.getByRole('button', { name: 'Decide in January 2028' }).click();
  await expect(page.getByRole('region', { name: 'Decision in January 2028' })).toContainText('Connect your workspace');
  await page.getByRole('button', { name: 'Close', exact: true }).click();
  await page.getByRole('button', { name: 'Edit your persona' }).click();
  const dialog = page.getByRole('dialog');
  await dialog.getByLabel('What should we call you?').fill('Alex');
  await dialog.getByLabel('Where do you live?').selectOption('Zurich');
  await dialog.getByRole('button', { name: 'Continue', exact: true }).click();
  await dialog.getByLabel('Monthly take-home income').fill('6100');
  await dialog.getByRole('button', { name: 'Continue', exact: true }).click();
  await dialog.getByLabel('Your main money goal').selectOption('Housing deposit');
  await dialog.getByLabel('Target by December 2029').fill('70000');
  await dialog.getByRole('button', { name: 'Save my persona' }).click();
  await expect(page.getByText('Your choices have changed.', { exact: false })).toBeVisible();
  // Results must remain explicitly attached to the old fictional persona until explored.
  await expect(page.getByRole('heading', { name: /P0067’s outlook/ })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
  await page.screenshot({ path: `/tmp/parallel-${test.info().project.name}.png`, fullPage: true });
  await page.getByRole('button', { name: 'parallel, home' }).click();
  await expect(page.getByRole('heading', { name: 'Hi Alex' })).toBeVisible();
  expect(errors).toEqual([]);
});

test('connect, explore, ask a grounded follow-up, invalidate stale chat', async ({ page }) => {
  const requests: unknown[] = [];
  await page.route('http://127.0.0.1:8000/**', async route => {
    const request = route.request();
    if (request.method() === 'OPTIONS') return route.fulfill({ status: 204, headers: { 'access-control-allow-origin': '*', 'access-control-allow-headers': 'authorization,content-type', 'access-control-allow-methods': 'GET,POST,DELETE' } });
    expect(request.headers().authorization).toBe('Bearer test-workspace-key-not-a-real-secret');
    let json: unknown;
    if (request.method() === 'DELETE') json = {};
    else if (request.url().endsWith('/health')) json = { status: 'ok', llm_ready: true };
    else if (request.url().endsWith('/messages')) {
      requests.push(request.postDataJSON());
      json = { answer: 'This explanation uses the same stored simulation. Would you like to compare another path?' };
    } else {
      requests.push(request.postDataJSON());
      const body = request.postDataJSON();
      json = { session_id: 'test-session', evidence: example, answer: 'Your simulation is ready. The goal estimate comes from 500 paths.' };
    }
    await route.fulfill({ json, headers: { 'access-control-allow-origin': '*' } });
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Connect', exact: true }).click();
  await page.getByRole('dialog').getByLabel('Workspace access key').fill('test-workspace-key-not-a-real-secret');
  await page.getByRole('dialog').getByRole('button', { name: 'Connect workspace', exact: true }).click();
  await expect(page.getByRole('dialog')).not.toBeVisible();
  await page.getByRole('button', { name: 'Explore this path', exact: true }).click();
  await expect(page.getByText('Example', { exact: true })).toBeHidden();
  await expect(page.getByText('Your simulation is ready.', { exact: false })).toBeVisible();
  await page.getByLabel('Ask about your financial future').fill('What matters most?');
  await page.getByRole('button', { name: 'Send message' }).click();
  await expect(page.getByText('This explanation uses the same stored simulation.', { exact: false })).toBeVisible();
  expect(requests).toHaveLength(2);
  expect(requests[0]).not.toHaveProperty('deliberate');
  expect(requests[1]).toEqual({ message: 'What matters most?' });
  await page.getByRole('button', { name: 'Spend less', exact: true }).click();
  await expect(page.getByLabel('Ask about your financial future')).toBeDisabled();
  await page.getByRole('button', { name: 'Go part-time', exact: true }).click();
  await expect(page.getByLabel('Ask about your financial future')).toBeEnabled();
  await expect(page.getByRole('switch', { name: 'Decide deliberately' })).toHaveCount(0);   // removed
  await page.getByRole('tab', { name: 'Plan explained' }).click();                          // no plan yet: offer to make one
  await expect(page.getByRole('button', { name: 'Make a long-term plan' }).last()).toBeVisible();
  await page.getByRole('tab', { name: 'Paths' }).click();
});

test('unreachable workspace has a recoverable error', async ({ page }) => {
  await page.route('http://127.0.0.1:8000/**', route => route.abort());
  await page.goto('/');
  await page.getByRole('button', { name: 'Explore this path', exact: true }).click();
  const dialog = page.getByRole('dialog');
  await dialog.getByLabel('Workspace access key').fill('test-workspace-key-not-a-real-secret');
  await dialog.getByRole('button', { name: 'Connect workspace', exact: true }).click();
  await expect(dialog.getByRole('alert')).toContainText('could not be reached');
  await expect(dialog.getByRole('button', { name: 'Connect workspace', exact: true })).toBeEnabled();
  await page.keyboard.press('Escape');
  await expect(dialog).not.toBeVisible();
});

test('a student can keep studying alongside the part-time offer', async ({ page }) => {
  const sessions: { scenario: { alongside_studies: boolean } }[] = [];
  await page.route('http://127.0.0.1:8000/**', async route => {
    const request = route.request();
    if (request.method() === 'OPTIONS') return route.fulfill({ status: 204, headers: { 'access-control-allow-origin': '*', 'access-control-allow-headers': 'authorization,content-type', 'access-control-allow-methods': 'GET,POST,DELETE' } });
    let json: unknown = {};
    if (request.url().endsWith('/health')) json = { status: 'ok', llm_ready: true };
    else if (request.method() === 'POST') {
      const body = request.postDataJSON();
      sessions.push(body);
      json = { session_id: 'student', evidence: { ...example, person: body.person, scenario: body.scenario }, answer: 'Ready.' };
    }
    await route.fulfill({ json, headers: { 'access-control-allow-origin': '*' } });
  });
  await page.goto('/');
  const keepStudying = page.getByRole('switch', { name: 'Keep studying' });
  await expect(keepStudying).toHaveAttribute('aria-checked', 'true');   // P0067 is a master's student
  const situation = async (status: string) => {
    await page.getByRole('button', { name: 'Edit this person', exact: true }).click();
    const dialog = page.getByRole('dialog');
    await dialog.getByLabel('Your work situation').selectOption(status);
    if (status === 'student') await dialog.getByLabel('Months until you finish').fill('24');
    for (const step of ['Continue', 'Continue']) await dialog.getByRole('button', { name: step, exact: true }).click();
    await dialog.getByRole('button', { name: 'Save my persona' }).click();
  };
  await situation('employed');
  await expect(keepStudying).toHaveCount(0);                     // only for students
  await situation('student');
  await expect(keepStudying).toHaveAttribute('aria-checked', 'true');
  await page.getByRole('button', { name: 'Connect', exact: true }).click();
  await page.getByRole('dialog').getByLabel('Workspace access key').fill('test-workspace-key-not-a-real-secret');
  await page.getByRole('dialog').getByRole('button', { name: 'Connect workspace', exact: true }).click();
  await page.getByRole('button', { name: 'Explore this path', exact: true }).click();
  await expect(page.getByText('Ready.')).toBeVisible();
  expect(sessions[0].scenario.alongside_studies).toBe(true);
  await expect(page.getByLabel('Ask about your financial future')).toBeEnabled();   // results match the request
  await keepStudying.click();
  await expect(page.getByLabel('Ask about your financial future')).toBeDisabled();
  await page.getByRole('button', { name: 'Stay on my path', exact: true }).click();
  await expect(keepStudying).toHaveCount(0);                     // only for the part-time offer
});

test('grow a tree of branches from decision points on the chart', async ({ page }) => {
  const bodies: { root: string; decisions: { month: number; choice: string }[] }[] = [];
  await page.route('http://127.0.0.1:8000/**', async route => {
    const request = route.request();
    if (request.method() === 'OPTIONS') return route.fulfill({ status: 204, headers: { 'access-control-allow-origin': '*', 'access-control-allow-headers': 'authorization,content-type', 'access-control-allow-methods': 'GET,POST,DELETE' } });
    let json: unknown = {};
    if (request.url().endsWith('/health')) json = { status: 'ok', llm_ready: true };
    else if (request.url().endsWith('/branches')) {
      const body = request.postDataJSON();
      bodies.push(body);
      const depth = body.parent === 'b1' ? 2 : 1;
      const label = { full_time: 'Work full-time', spend_less: 'Spend less', lose_job: 'Lose my job' }[body.choice as string] + ` · ${2025 + body.month / 12}`;
      const decisions = [...(depth === 2 ? [{ month: bodies[0].month, choice: bodies[0].choice }] : []), { month: body.month, choice: body.choice }];
      json = { id: `b${bodies.length}`, parent: body.parent, root: 'alternative', from: body.month, label, applied: depth === 1 ? 0.6 : 1,
               decisions, summary: { ...example.alternative, final_net_worth: { ...example.alternative.final_net_worth, p50: example.alternative.final_net_worth.p50 + 100000 * depth } } };
    } else if (request.method() === 'POST') json = { session_id: 'tree', evidence: example, answer: 'Ready.' };
    await route.fulfill({ json, headers: { 'access-control-allow-origin': '*' } });
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Connect', exact: true }).click();
  await page.getByRole('dialog').getByLabel('Workspace access key').fill('test-workspace-key-not-a-real-secret');
  await page.getByRole('dialog').getByRole('button', { name: 'Connect workspace', exact: true }).click();
  await page.getByRole('button', { name: 'Explore this path', exact: true }).click();
  await expect(page.getByText('Ready.')).toBeVisible();

  const tree = page.getByRole('region', { name: 'Paths' });
  await page.getByRole('button', { name: 'Decide in January 2027' }).click();
  await page.getByRole('region', { name: 'Decision in January 2027' }).getByRole('button', { name: 'Work full-time' }).click();
  const first = tree.getByRole('button', { name: /^Work full-time · 2027/ });
  await expect(first).toHaveAttribute('aria-pressed', 'true');
  await expect(first).toContainText('possible in 60% of lives');
  await expect(page.getByRole('region', { name: 'Key moments' })).toContainText('Work full-time · 2027');
  expect(bodies[0]).toEqual({ parent: 'alternative', month: 24, choice: 'full_time' });

  // Branch from the branch: only later reviews are offered, and the decisions stack.
  await expect(page.getByRole('button', { name: 'Decide in January 2026' })).toHaveCount(0);
  await page.getByRole('button', { name: 'Decide in January 2029' }).click();
  await page.getByRole('region', { name: 'Decision in January 2029' }).getByRole('button', { name: 'Spend less' }).click();
  await expect(tree.getByRole('button', { name: /^Spend less · 2029/ })).toHaveAttribute('aria-pressed', 'true');
  expect(bodies[1]).toEqual({ parent: 'b1', month: 48, choice: 'spend_less' });

  // "Now" (January 2025) takes life events only.
  await tree.getByRole('button', { name: /^Studies only/ }).click();
  await page.getByRole('button', { name: 'Decide in January 2025' }).click();
  const now = page.getByRole('region', { name: 'Decision in January 2025' });
  await expect(now).toContainText('Now, in January 2025');
  await expect(now.getByRole('button', { name: 'Work full-time' })).toHaveCount(0);
  await now.getByRole('button', { name: 'Lose my job' }).click();
  expect(bodies[2]).toEqual({ parent: 'baseline', month: 0, choice: 'lose_job' });

  // Removing a branch removes what grew from it.
  await tree.getByRole('button', { name: 'Remove Work full-time · 2027' }).click();
  await expect(tree.getByRole('button', { name: /^Spend less · 2029/ })).toHaveCount(0);
  await expect(tree.getByRole('button', { name: /^Job alongside studies/ })).toHaveAttribute('aria-pressed', 'false');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
});

test('the assistant changes the persona and grows branches from a chat message', async ({ page }) => {
  await page.route('http://127.0.0.1:8000/**', async route => {
    const request = route.request();
    if (request.method() === 'OPTIONS') return route.fulfill({ status: 204, headers: { 'access-control-allow-origin': '*', 'access-control-allow-headers': 'authorization,content-type', 'access-control-allow-methods': 'GET,POST,DELETE' } });
    let json: unknown = {};
    if (request.url().endsWith('/health')) json = { status: 'ok', llm_ready: true };
    else if (request.url().endsWith('/messages')) {
      const rerun = { ...example, person: { ...example.person, age: 31, income: 7000 } };
      const branch = { id: 'b1', parent: 'alternative', root: 'alternative', from: 36, label: 'Work full-time · 2028', applied: 0.95, decisions: [{ month: 36, choice: 'full_time' }],
                       summary: { ...example.alternative, high_goal_odds: 0.96 } };
      json = { answer: 'Done: you are now 31 on CHF 7,000, and the full-time branch is on the chart.',
               notes: ['Updated age 31; income CHF 7,000 · re-ran 500 lives', 'Grew a branch: Work full-time · 2028'],
               actions: [{ type: 'rerun', evidence: rerun }, { type: 'branch', branch }, { type: 'highlight', path_id: 'b1', measure: 'cash' }] };
    } else if (request.method() === 'POST') json = { session_id: 'chat', evidence: example, answer: 'Ready.' };
    await route.fulfill({ json, headers: { 'access-control-allow-origin': '*' } });
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Connect', exact: true }).click();
  await page.getByRole('dialog').getByLabel('Workspace access key').fill('test-workspace-key-not-a-real-secret');
  await page.getByRole('dialog').getByRole('button', { name: 'Connect workspace', exact: true }).click();
  await page.getByRole('button', { name: 'Explore this path', exact: true }).click();
  await expect(page.getByText('Ready.')).toBeVisible();
  await page.getByLabel('Ask about your financial future').fill('I am 31 and take home 7k. What if I go full-time in 2028?');
  await page.getByRole('button', { name: 'Send message' }).click();
  await expect(page.getByRole('list', { name: 'Changes made' })).toContainText('Updated age 31; income CHF 7,000');
  const tree = page.getByRole('region', { name: 'Paths' });
  await expect(tree.getByRole('button', { name: /^Work full-time · 2028/ })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('button', { name: 'Cash', exact: true })).toHaveAttribute('aria-pressed', 'true');   // the assistant switched the measure
  await expect(page.getByText('Your choices have changed', { exact: false })).toHaveCount(0);   // controls follow the new run
  await expect(page.getByLabel('Ask about your financial future')).toBeEnabled();
  await page.getByRole('button', { name: 'Edit your persona' }).click();
  await expect(page.getByRole('dialog').getByLabel('Age at the start of 2025')).toHaveValue('31');
});

test('assumptions dialog and the plan-explained tab', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await page.getByRole('button', { name: 'Assumptions', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Assumptions' });
  await dialog.getByText('1. Time, scope and runs').click();
  await expect(dialog.getByRole('cell', { name: /January 2025 to December 2034/ })).toBeVisible();
  await expect(dialog.getByText('14. The long-term plan (LLM planner)')).toBeVisible();
  await expect(dialog.getByText('Decide deliberately')).toHaveCount(0);
  await page.keyboard.press('Escape');

  await page.getByRole('button', { name: 'See an example' }).click();
  await page.getByRole('tab', { name: 'Plan explained' }).click();
  const explained = page.getByRole('article', { name: 'Plan explained' });
  await expect(explained.getByRole('heading', { name: 'From plan to outcomes' })).toBeVisible();
  for (const h of ['What did the LLM plan?', 'How is the plan carried out?', 'What happens financially?'])
    await expect(explained.getByRole('heading', { name: h })).toBeVisible();
  await expect(page.getByLabel('Explore a year')).toHaveCount(0);          // the chart is behind the other tab
  await page.getByRole('tab', { name: 'Paths' }).click();
  await expect(page.getByLabel('Explore a year')).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
  expect(errors).toEqual([]);
});

test('compare taking the job alongside the studies or instead of them', async ({ page }) => {
  const sessions: { scenario: { alongside_studies: boolean }; compare?: { kind: string; alongside_studies: boolean } }[] = [];
  await page.route('http://127.0.0.1:8000/**', async route => {
    const request = route.request();
    if (request.method() === 'OPTIONS') return route.fulfill({ status: 204, headers: { 'access-control-allow-origin': '*', 'access-control-allow-headers': 'authorization,content-type', 'access-control-allow-methods': 'GET,POST,DELETE' } });
    let json: unknown = {};
    if (request.url().endsWith('/health')) json = { status: 'ok', llm_ready: true };
    else if (request.method() === 'POST') {
      const body = request.postDataJSON();
      sessions.push(body);
      json = { session_id: 'cmp', evidence: { ...example, scenario: body.scenario, compare_scenario: body.compare ?? null, compare: body.compare ? example.compare : null }, answer: 'Compared.' };
    }
    await route.fulfill({ json, headers: { 'access-control-allow-origin': '*' } });
  });
  await page.goto('/');
  // Offline, the example already holds both choices.
  await page.getByRole('button', { name: /^Compare both choices/ }).click();
  const tree = page.getByRole('region', { name: 'Paths' });
  for (const path of [/^Studies only/, /^Job alongside studies/, /^Job instead of studies/]) await expect(tree.getByRole('button', { name: path })).toBeVisible();
  const instead = tree.getByRole('button', { name: /^Job instead of studies/ });
  await expect(instead).toContainText(`${Math.round(example.compare.high_goal_odds * 100)}%`);
  await instead.click();
  await expect(instead).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('switch', { name: 'Compare both' })).toHaveAttribute('aria-checked', 'true');

  // Connected, the button runs both scenarios in one exploration.
  await page.getByRole('button', { name: 'parallel, home' }).click();
  await page.getByRole('button', { name: 'Connect', exact: true }).click();
  await page.getByRole('dialog').getByLabel('Workspace access key').fill('test-workspace-key-not-a-real-secret');
  await page.getByRole('dialog').getByRole('button', { name: 'Connect workspace', exact: true }).click();
  await page.getByRole('button', { name: /^Compare both choices/ }).click();
  await expect(page.getByText('Compared.')).toBeVisible();
  expect(sessions[0].scenario.alongside_studies).toBe(true);
  expect(sessions[0].compare).toEqual({ kind: 'part_time', salary: 3200, alongside_studies: false });
  await expect(page.getByLabel('Ask about your financial future')).toBeEnabled();          // results match the controls
  await page.getByRole('switch', { name: 'Compare both' }).click();                        // turning it off needs a new run
  await expect(page.getByLabel('Ask about your financial future')).toBeDisabled();
});

test('exploring keeps the page where it is; only a chat question scrolls to the answer', async ({ page }) => {
  await page.route('http://127.0.0.1:8000/**', async route => {
    const request = route.request();
    if (request.method() === 'OPTIONS') return route.fulfill({ status: 204, headers: { 'access-control-allow-origin': '*', 'access-control-allow-headers': 'authorization,content-type', 'access-control-allow-methods': 'GET,POST,DELETE' } });
    let json: unknown = {};
    if (request.url().endsWith('/health')) json = { status: 'ok', llm_ready: true };
    else if (request.url().endsWith('/messages')) json = { answer: 'Answered.' };
    else if (request.method() === 'POST') { await new Promise(r => setTimeout(r, 400)); json = { session_id: 'scroll', evidence: example, answer: 'Explored.' }; }
    await route.fulfill({ json, headers: { 'access-control-allow-origin': '*' } });
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Connect', exact: true }).click();
  await page.getByRole('dialog').getByLabel('Workspace access key').fill('test-workspace-key-not-a-real-secret');
  await page.getByRole('dialog').getByRole('button', { name: 'Connect workspace', exact: true }).click();
  await page.getByRole('button', { name: 'Explore this path', exact: true }).click();
  await expect(page.getByText('Explored.')).toBeVisible();
  // Read the scroll of the page and of the text pane (the pane scrolls on desktop, the page on mobile).
  const scrolled = () => page.evaluate(() => window.scrollY + (document.querySelector('.text-pane')?.scrollTop ?? 0));
  await page.evaluate(() => { window.scrollTo(0, 0); document.querySelector('.text-pane')?.scrollTo(0, 0); });
  await page.getByRole('button', { name: 'Explore this path', exact: true }).click();
  await expect(page.getByText('Explored.')).toBeVisible();
  await page.waitForTimeout(600);
  expect(await scrolled()).toBe(0);
  await page.getByLabel('Ask about your financial future').fill('Why?');
  await page.getByRole('button', { name: 'Send message' }).click();
  await expect(page.getByText('Answered.')).toBeVisible();
  await page.waitForTimeout(600);
  expect(await scrolled()).toBeGreaterThan(0);
});
