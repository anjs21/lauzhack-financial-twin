import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';

const example = JSON.parse(readFileSync(new URL('../src/example.json', import.meta.url), 'utf8'));

test('example, persona creation, and keyboard-accessible chart', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Your life. A little clearer.' })).toBeVisible();
  await expect(page.getByText('Example script')).toBeVisible();
  await expect(page.getByLabel('Ask about your financial future')).toBeDisabled();
  await page.getByRole('button', { name: 'Net worth', exact: true }).click();
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
  await expect(page.getByText('Alex’s financial twin')).toBeVisible();
  await expect(page.getByText('Your choices have changed.', { exact: false })).toBeVisible();
  // Results must remain explicitly attached to the old fictional persona until explored.
  await expect(page.getByRole('heading', { name: 'A glimpse of Mia’s possibilities' })).toBeVisible();
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
    if (request.url().endsWith('/health')) json = { status: 'ok', llm_ready: true };
    else if (request.url().endsWith('/messages')) {
      requests.push(request.postDataJSON());
      json = { answer: 'This explanation uses the same stored simulation. Would you like to compare another path?' };
    } else {
      requests.push(request.postDataJSON());
      json = { session_id: 'test-session', evidence: example, answer: 'Your simulation is ready. The goal estimate comes from 500 paths.' };
    }
    await route.fulfill({ json, headers: { 'access-control-allow-origin': '*' } });
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Connect my workspace', exact: true }).click();
  await page.getByRole('dialog').getByLabel('Workspace access key').fill('test-workspace-key-not-a-real-secret');
  await page.getByRole('dialog').getByRole('button', { name: 'Connect workspace', exact: true }).click();
  await expect(page.getByRole('dialog')).not.toBeVisible();
  await page.getByRole('button', { name: 'Explore this path', exact: true }).click();
  await expect(page.getByText('Live exploration', { exact: true })).toBeVisible();
  await expect(page.getByText('Your simulation is ready.', { exact: false })).toBeVisible();
  await page.getByLabel('Ask about your financial future').fill('What matters most?');
  await page.getByRole('button', { name: 'Send message' }).click();
  await expect(page.getByText('This explanation uses the same stored simulation.', { exact: false })).toBeVisible();
  expect(requests).toHaveLength(2);
  expect(requests[1]).toEqual({ message: 'What matters most?' });
  await page.getByRole('button', { name: 'Spend a little less', exact: false }).click();
  await expect(page.getByLabel('Ask about your financial future')).toBeDisabled();
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
