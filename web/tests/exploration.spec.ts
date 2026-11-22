import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';

const example = JSON.parse(readFileSync(new URL('../src/example.json', import.meta.url), 'utf8'));

test('example, persona creation, and keyboard-accessible chart', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Hi Mia' })).toBeVisible();
  await expect(page.getByText('Example', { exact: true })).toBeVisible();
  await expect(page.getByLabel('Ask about your financial future')).toBeDisabled();
  await page.getByRole('button', { name: 'Net worth', exact: true }).click();
  const moments = page.getByRole('region', { name: 'Key moments' });
  await expect(moments.getByText('Cuts back to part-time')).toBeVisible();
  await expect(moments.getByText('every life')).toBeVisible();
  await moments.getByRole('button', { name: 'Current path', exact: true }).click();
  await expect(moments.getByText('Cuts back to part-time')).toBeHidden();
  const year = page.getByLabel('Explore a year');
  await year.focus(); await year.press('ArrowRight');
  await expect(year).toHaveValue('72');
  await page.getByRole('button', { name: 'Build my persona', exact: true }).click();
  const dialog = page.getByRole('dialog');
  await dialog.getByLabel('What should we call you?').fill('Alex');
  await dialog.getByLabel('Where do you live?').selectOption('Zurich');
  await dialog.getByRole('button', { name: 'Continue', exact: true }).click();
  await dialog.getByLabel('Monthly take-home income').fill('6100');
  await dialog.getByRole('button', { name: 'Continue', exact: true }).click();
  await dialog.getByLabel('Your main money goal').selectOption('Housing deposit');
  await dialog.getByLabel('Target by December 2029').fill('70000');
  await dialog.getByRole('button', { name: 'Save my persona' }).click();
  await expect(page.getByRole('heading', { name: 'Hi Alex' })).toBeVisible();
  await expect(page.getByText('Your choices have changed.', { exact: false })).toBeVisible();
  // Results must remain explicitly attached to the old fictional persona until explored.
  await expect(page.getByRole('heading', { name: /Mia’s outlook/ })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
  await page.screenshot({ path: `/tmp/parallel-${test.info().project.name}.png`, fullPage: true });
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
      json = { session_id: 'test-session', evidence: { ...example, deliberate: body.deliberate }, answer: 'Your simulation is ready. The goal estimate comes from 500 paths.' };
    }
    await route.fulfill({ json, headers: { 'access-control-allow-origin': '*' } });
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Connect my workspace', exact: true }).click();
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
  expect((requests[0] as { deliberate: boolean }).deliberate).toBe(false);
  expect(requests[1]).toEqual({ message: 'What matters most?' });
  await page.getByRole('button', { name: 'Spend less', exact: true }).click();
  await expect(page.getByLabel('Ask about your financial future')).toBeDisabled();
  await page.getByRole('button', { name: 'Go part-time', exact: true }).click();
  await expect(page.getByLabel('Ask about your financial future')).toBeEnabled();
  // Switching to deliberate decisions makes the run stale until explored again, then is sent to the API.
  const toggle = page.getByRole('switch', { name: 'Decide deliberately' });
  await toggle.click();
  await expect(toggle).toHaveAttribute('aria-checked', 'true');
  await expect(page.getByLabel('Ask about your financial future')).toBeDisabled();
  await page.getByRole('button', { name: 'Explore this path', exact: true }).click();
  await expect(page.getByText('· deliberate', { exact: false })).toBeVisible();
  await expect(page.getByLabel('Ask about your financial future')).toBeEnabled();
  expect((requests[2] as { deliberate: boolean }).deliberate).toBe(true);
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
      json = { session_id: 'student', evidence: { ...example, person: body.person, scenario: body.scenario, deliberate: body.deliberate }, answer: 'Ready.' };
    }
    await route.fulfill({ json, headers: { 'access-control-allow-origin': '*' } });
  });
  await page.goto('/');
  const keepStudying = page.getByRole('switch', { name: 'Keep studying' });
  await expect(keepStudying).toHaveCount(0);                     // Mia is employed
  await page.getByRole('button', { name: 'Build my persona', exact: true }).click();
  const dialog = page.getByRole('dialog');
  await dialog.getByLabel('Your work situation').selectOption('student');
  await dialog.getByLabel('Months until you finish').fill('24');
  for (const step of ['Continue', 'Continue']) await dialog.getByRole('button', { name: step, exact: true }).click();
  await dialog.getByRole('button', { name: 'Save my persona' }).click();
  await expect(keepStudying).toHaveAttribute('aria-checked', 'true');
  await page.getByRole('button', { name: 'Connect my workspace', exact: true }).click();
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
