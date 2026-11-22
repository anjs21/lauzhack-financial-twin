import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import { readFileSync } from 'node:fs';

const example = JSON.parse(readFileSync(new URL('../src/example.json', import.meta.url), 'utf8'));
const examplePlan = JSON.parse(readFileSync(new URL('../src/example-plan.json', import.meta.url), 'utf8'));
const [draft, nudged] = examplePlan.history;
const KEY = 'test-workspace-key-not-a-real-secret';

async function connect(page: Page) {
  await page.getByRole('button', { name: 'Connect', exact: true }).click();
  await page.getByRole('dialog').getByLabel('Workspace access key').fill(KEY);
  await page.getByRole('dialog').getByRole('button', { name: 'Connect workspace', exact: true }).click();
  await expect(page.getByRole('dialog')).not.toBeVisible();
}

test('the example plan shows its steps, how often they happen and what a nudge changed', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  const plan = page.getByRole('region', { name: 'Your plan' });
  await expect(plan.getByRole('heading', { name: 'Mia’s plan' })).toBeVisible();
  await expect(plan.getByText(/^Example · Version 2 · /)).toBeVisible();
  await expect(plan.getByText(nudged.plan.narrative)).toBeVisible();
  await expect(plan.getByRole('heading', { name: /What changed/ })).toBeVisible();
  await expect(plan.getByText(`“${nudged.said}”`)).toBeVisible();
  for (const step of nudged.plan.steps) await expect(plan.getByText(step.label, { exact: true })).toBeVisible();
  for (const step of draft.plan.steps.filter((s: { id: string }) => !nudged.plan.steps.some((n: { id: string }) => n.id === s.id)))
    await expect(plan.getByText(`Dropped: ${step.label}`)).toBeVisible();
  await plan.getByRole('button', { name: 'Make my own plan' }).click();
  await expect(plan.getByLabel('In your own words, what are your plans?')).toBeVisible();
  await plan.getByRole('button', { name: 'Back to the example' }).click();
  await expect(plan.getByRole('heading', { name: 'Mia’s plan' })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
  await page.screenshot({ path: `/tmp/parallel-plan-${test.info().project.name}.png`, fullPage: true });
  expect(errors).toEqual([]);
});

test('make a plan, follow it, nudge it, and stop following it', async ({ page }) => {
  const sessions: Record<string, unknown>[] = [];
  const planRequests: { url: string; body: Record<string, unknown> }[] = [];
  let polls = 0, version: typeof draft | null = null;
  await page.route('http://127.0.0.1:8000/**', async route => {
    const request = route.request();
    const url = request.url();
    if (request.method() === 'OPTIONS') return route.fulfill({ status: 204, headers: { 'access-control-allow-origin': '*', 'access-control-allow-headers': 'authorization,content-type', 'access-control-allow-methods': 'GET,POST,DELETE' } });
    expect(request.headers().authorization).toBe(`Bearer ${KEY}`);
    let json: unknown = {};
    let status = 200;
    if (url.endsWith('/health')) json = { status: 'ok', llm_ready: true };
    else if (request.method() === 'DELETE') json = { deleted: true };
    else if (url.endsWith('/plans') || url.endsWith('/nudges')) {
      planRequests.push({ url, body: request.postDataJSON() });
      polls = 0; status = 202;
      version = url.endsWith('/plans') ? draft : nudged;
      json = { plan_id: 'p1', status: 'working', progress: 'Starting', error: null, latest: null, versions: [] };
    } else if (url.endsWith('/plans/p1')) {
      polls += 1;
      json = polls < 2 ? { status: 'working', progress: 'Round 2 of up to 4: testing it in 300 lives', error: null, latest: null, versions: [] }
        : { status: 'ready', progress: '', error: null, latest: version, versions: [] };
    } else if (url.endsWith('/sessions')) {
      const body = request.postDataJSON();
      sessions.push(body);
      const plan = body.plan_id ? { narrative: version!.plan.narrative, spontaneity: .3, steps: [], carried_out: version!.evaluation.steps } : undefined;
      json = { session_id: `s${sessions.length}`, evidence: { ...example, scenario: body.scenario, deliberate: body.deliberate, plan }, answer: `Explained run ${sessions.length}.` };
    }
    await route.fulfill({ status, json, headers: { 'access-control-allow-origin': '*' } });
  });
  await page.goto('/');
  await connect(page);
  const plan = page.getByRole('region', { name: 'Your plan' });
  await plan.getByRole('button', { name: 'Make my own plan' }).click();
  await plan.getByLabel('In your own words, what are your plans?').fill(draft.said);
  await plan.getByRole('button', { name: 'Make my plan' }).click();
  await expect(plan.getByText('Round 2 of up to 4: testing it in 300 lives')).toBeVisible();
  await expect(plan.getByText(`Version 1 · ${draft.signed ? 'signed by your twin' : 'draft, not signed'}`)).toBeVisible();
  await expect(page.getByText('Explained run 1.')).toBeVisible();          // the new plan is explored right away
  expect(planRequests[0].body).toMatchObject({ wishes: draft.said, n_paths: 300 });
  expect(sessions[0].plan_id).toBe('p1');
  expect(String(sessions[0].question)).toContain('my plan');
  await expect(page.getByText('· following your plan', { exact: false })).toBeVisible();
  const follow = page.getByRole('switch', { name: 'Follow my plan' });
  await expect(follow).toHaveAttribute('aria-checked', 'true');
  await expect(page.getByLabel('Ask about your financial future')).toBeEnabled();

  await plan.getByLabel('Something changed?').fill(nudged.said);
  await plan.getByLabel('The change applies from').selectOption('2026-01');
  await plan.getByRole('button', { name: 'Update my plan' }).click();
  await expect(plan.getByText(`Version 2 · ${nudged.signed ? 'signed by your twin' : 'draft, not signed'}`)).toBeVisible();
  await expect(plan.getByRole('heading', { name: 'What changed from 2026' })).toBeVisible();
  await expect(page.getByText('Explained run 2.')).toBeVisible();
  expect(planRequests[1]).toMatchObject({ url: 'http://127.0.0.1:8000/plans/p1/nudges', body: { message: nudged.said, as_of: '2026-01' } });
  expect(sessions[1].plan_id).toBe('p1');

  await follow.click();                                                    // habits alone: results are stale until explored
  await expect(page.getByLabel('Ask about your financial future')).toBeDisabled();
  await page.getByRole('button', { name: 'Explore this path', exact: true }).click();
  await expect(page.getByText('Explained run 3.')).toBeVisible();
  expect(sessions[2].plan_id).toBeUndefined();
  await expect(page.getByText('· following your plan', { exact: false })).toBeHidden();

  await page.getByRole('button', { name: 'Edit your persona' }).click();   // a plan belongs to its persona
  const dialog = page.getByRole('dialog');
  await dialog.getByLabel('What should we call you?').fill('Alex');
  for (const step of ['Continue', 'Continue']) await dialog.getByRole('button', { name: step, exact: true }).click();
  await dialog.getByRole('button', { name: 'Save my persona' }).click();
  await expect(plan.getByText('You changed your persona after making this plan', { exact: false })).toBeVisible();
  await expect(follow).toHaveCount(0);
  await expect(plan.getByRole('button', { name: 'Update my plan' })).toBeDisabled();
  await plan.getByRole('button', { name: 'Start over' }).click();
  await expect(plan.getByLabel('In your own words, what are your plans?')).toBeVisible();
});

test('a failed planning job reports the error and keeps the form', async ({ page }) => {
  await page.route('http://127.0.0.1:8000/**', async route => {
    const request = route.request();
    if (request.method() === 'OPTIONS') return route.fulfill({ status: 204, headers: { 'access-control-allow-origin': '*', 'access-control-allow-headers': 'authorization,content-type', 'access-control-allow-methods': 'GET,POST,DELETE' } });
    const url = request.url();
    const json = url.endsWith('/health') ? { status: 'ok', llm_ready: true }
      : url.endsWith('/plans') ? { plan_id: 'bad', status: 'working', progress: 'Starting', error: null, latest: null, versions: [] }
      : { status: 'failed', progress: '', error: 'The planner did not produce a usable plan. Try again, or say more about your plans.', latest: null, versions: [] };
    await route.fulfill({ status: url.endsWith('/plans') ? 202 : 200, json, headers: { 'access-control-allow-origin': '*' } });
  });
  await page.goto('/');
  await connect(page);
  const plan = page.getByRole('region', { name: 'Your plan' });
  await plan.getByRole('button', { name: 'Make my own plan' }).click();
  await plan.getByRole('button', { name: 'Make my plan' }).click();
  await expect(page.getByRole('alert')).toContainText('did not produce a usable plan');
  await expect(plan.getByRole('button', { name: 'Make my plan' })).toBeEnabled();
});
