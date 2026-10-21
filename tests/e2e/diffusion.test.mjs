import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { start } from './browser.mjs';
import { loadPage } from '../load.mjs';

let app;
before(async () => { app = await start(); });
after(() => app.close());

const sd = loadPage()('MODELS.sd'), LINK = `#sd.${sd.children.findIndex(k => k.name === 'Bước khử nhiễu (scheduler)')}`;
const slide = (page, key, value) => page.evaluate(([k, v]) => { const s = document.querySelector(`[data-dif="${k}"]`); s.value = v; s.dispatchEvent(new Event('input', { bubbles: true })); }, [key, value]);
const reading = page => page.locator('.dif-out .reading').textContent();
const share = (text, prompt) => +text.match(new RegExp(`(\\d+)% điểm vào cụm "${prompt}"|(\\d+)% vào cụm "${prompt}"`)).slice(1).find(Boolean);

test('the scheduler block runs the toy diffusion: the prompt decides which clusters the points reach', async () => {
  const page = await app.open(LINK);
  assert.match(await reading(page), /Bước 50\/50/);
  assert.ok(share(await reading(page), 'mèo') >= 90, await reading(page)); // "mèo", guidance 7.5
  assert.equal(share(await reading(page), 'chó'), 0);
  await page.click('[data-dprompt=""]'); // the empty prompt: both kinds of clusters
  assert.ok(share(await reading(page), 'mèo') > 25 && share(await reading(page), 'chó') > 25, await reading(page));
  const painted = await page.locator('#diff-canvas').evaluate(c => c.getContext('2d').getImageData(0, 0, c.width, c.height).data.some((v, i) => i % 4 === 3 && v > 0));
  assert.ok(painted, 'the panel canvas should show the points');
  assert.deepEqual(page.errors, []);
  await page.context().close();
});

test('fewer steps reset the step slider, and scrubbing back shows the starting noise', async () => {
  const page = await app.open(LINK);
  await slide(page, 'N', 0); // 5 DDIM steps
  assert.equal(await page.locator('#o-dif-k').textContent(), '5/5');
  assert.equal(await page.locator('#dif-k').getAttribute('max'), '5');
  assert.match(await reading(page), /Bước 5\/5/);
  await slide(page, 'k', 0);
  assert.match(await reading(page), /Bước 0\/5 \(nhiễu ban đầu\): t = 801/);
  await page.context().close();
});
