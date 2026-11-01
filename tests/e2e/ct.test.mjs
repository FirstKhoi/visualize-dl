import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { start } from './browser.mjs';
import { loadPage } from '../load.mjs';

let app;
before(async () => { app = await start(); });
after(() => app.close());

// the CT window block of the U-Net tab, found by name so the test follows the model file
const link = 'unet.' + loadPage()('MODELS.unet.children').findIndex(k => k.live === 'ct');
const row = async (page, tissue) => (await page.locator('.ct-table tr', { hasText: tissue }).first().textContent()).replace(/\s+/g, ' ');
const out = async (page, key) => (await page.locator(`#o-ct-${key}`).textContent()).trim();

test('the CT window block opens with the soft tissue preset, and a preset sets W, L and every gray value', async () => {
  const page = await app.open('#' + link);
  assert.equal(await page.locator('section.live.ct').count(), 1);
  assert.deepEqual([await out(page, 'W'), await out(page, 'L')], ['400', '40']);
  assert.match(await row(page, 'Mỡ'), /−100.*15%/); // (−100 − (−160)) / 400
  await page.click('[data-ctpreset="Xương"]');
  assert.deepEqual([await out(page, 'W'), await out(page, 'L')], ['1800', '400']);
  assert.match(await row(page, 'Mỡ'), /−100.*22%/); // (−100 − (−500)) / 1800 = 0.222
  assert.equal(await page.locator('[data-ctpreset="Xương"]').getAttribute('aria-pressed'), 'true');
  assert.deepEqual(page.errors, []);
  await page.context().close();
});

test('dragging L moves the window, and a click on the slice reads the HU there', async () => {
  const page = await app.open('#' + link);
  await page.locator('#ct-L').evaluate(el => { el.value = -600; el.dispatchEvent(new Event('input', { bubbles: true })); });
  assert.equal(await out(page, 'L'), '−600');
  assert.match(await row(page, 'Mô mềm'), /40.*100%/); // 40 > −600 + 200: white
  assert.equal(await page.locator('[data-ctpreset][aria-pressed="true"]').count(), 0, 'no preset matches W 400, L −600');
  const box = await page.locator('#ct-view').boundingBox();
  await page.mouse.click(box.x + box.width * 40.5 / 128, box.y + box.height * 62.5 / 128); // the middle of the left lung
  const read = await page.locator('.ct-read').textContent(), hu = +read.match(/(−?\d+) HU/)[1].replace('−', '-');
  assert.match(read, /Phổi/);
  assert.ok(hu > -900 && hu < -800, read); // −850 plus noise σ = 10
  assert.deepEqual(page.errors, []);
  await page.context().close();
});
