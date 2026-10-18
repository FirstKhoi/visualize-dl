import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { start, lab } from './browser.mjs';

let app;
before(async () => { app = await start(); });
after(() => app.close());

const slide = (page, key, value) => page.evaluate(([k, v]) => { const s = document.querySelector(`[data-labkey="${k}"]`); s.value = v; s.dispatchEvent(new Event('input', { bubbles: true })); }, [key, value]);
const picture = page => page.evaluate(() => document.getElementById('lab-canvas').toDataURL());

test('the lab measures a held-out test set beside the training set', async () => {
  const page = await app.open('#mlp');
  await page.click('[data-lab="step"]');
  const stats = await page.locator('.lab-stats').textContent();
  assert.match(stats, /train[^]*test/i);
  assert.match(await page.locator('.lab-out svg').getAttribute('aria-label'), /Train và Test$/);
  assert.deepEqual(page.errors, []);
  await page.context().close();
});

test('label noise flips training labels and leaves the test set alone', async () => {
  const page = await app.open('#mlp');
  const clean = await picture(page);
  await slide(page, 'noise', 4); // the largest noise level
  assert.notEqual(await picture(page), clean, 'the training points should change colour');
  await page.click('[data-labset="view"][data-val="test"]');
  const testNoisy = await picture(page);
  await slide(page, 'noise', 0);
  assert.equal(await picture(page), testNoisy, 'the test points and the fresh network do not depend on the noise');
  await page.context().close();
});

test('weight decay and batch size keep the run; label noise and width start a new one', async () => {
  const page = await app.open('#mlp');
  for (let i = 0; i < 3; i++) await page.click('[data-lab="step"]');
  await slide(page, 'decay', 3);
  await page.click('[data-labset="batch"][data-val="8"]');
  assert.equal(await lab.steps(page), 3);
  await page.click('[data-lab="step"]'); // a mini-batch step still counts as a step
  assert.equal(await lab.steps(page), 4);
  await slide(page, 'noise', 2);
  assert.equal(await lab.steps(page), 0);
  await page.click('[data-lab="step"]');
  await slide(page, 'H', 12);
  assert.equal(await lab.steps(page), 0);
  assert.deepEqual(page.errors, []);
  await page.context().close();
});
