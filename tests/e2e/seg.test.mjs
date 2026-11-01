import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { start } from './browser.mjs';
import { loadPage } from '../load.mjs';

let app;
before(async () => { app = await start(); });
after(() => app.close());

const link = 'unet.' + loadPage()('MODELS.unet.children').findIndex(k => k.live === 'seg');
const metric = async (page, key) => (await page.locator(`.seg-table [data-m="${key}"]`).textContent()).trim();
const slide = (page, id, v) => page.locator(id).evaluate((el, v) => { el.value = v; el.dispatchEvent(new Event('input', { bubbles: true })); }, v);

test('the U-Net output block scores a toy prediction: a perfect one, then one that predicts background everywhere', async () => {
  const page = await app.open('#' + link);
  assert.equal(await page.locator('section.live.seg').count(), 1);
  assert.equal(await metric(page, 'dice'), '1.00');
  await page.click('[data-segdo="none"]');
  assert.equal(await metric(page, 'dice'), '0.00');
  assert.equal(await metric(page, 'acc'), '98.7%');  // 4044 of 4096 pixels are background
  assert.equal(await metric(page, 'ce'), '0.068');
  assert.match(await metric(page, 'hd95'), /không xác định/);
  assert.deepEqual(page.errors, []);
  await page.context().close();
});

test('a 2 px shift costs a small lesion far more Dice than a large one, and a stray spot shows in HD95, not Dice', async () => {
  const page = await app.open('#' + link);
  await slide(page, '#seg-shift', 2);
  assert.equal(await metric(page, 'dice'), '0.69');
  await slide(page, '#seg-r', 16);
  assert.equal(await metric(page, 'dice'), '0.92');
  await slide(page, '#seg-shift', 0);
  await page.click('[data-segdo="blob"]');
  assert.equal(await metric(page, 'dice'), '0.99');
  assert.equal(await metric(page, 'hd95'), '17.7 px');
  assert.deepEqual(page.errors, []);
  await page.context().close();
});
