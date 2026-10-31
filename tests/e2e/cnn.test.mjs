import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { start } from './browser.mjs';
import { loadPage } from '../load.mjs';

let app;
before(async () => { app = await start(); });
after(() => app.close());

// the stage blocks of the CNN tab, found by name so the test follows the model file
const kids = loadPage()('MODELS.cnn.children'), at = name => kids.findIndex(k => k.name === name);
const guess = async page => (await page.locator('.cnn-guess b').textContent()).trim();

// a stroke on the drawing pad, in pad fractions (0…1)
async function stroke(page, points) {
  const box = await page.locator('#cnn-pad').boundingBox(), xy = ([x, y]) => [box.x + x * box.width, box.y + y * box.height];
  await page.mouse.move(...xy(points[0]));
  await page.mouse.down();
  for (const p of points.slice(1)) await page.mouse.move(...xy(p), { steps: 4 });
  await page.mouse.up();
  await page.waitForTimeout(200);
}

test('the CNN tab opens on an MNIST test digit and shows what the network answers', async () => {
  const page = await app.open('#cnn');
  assert.equal(await page.locator('section.live.cnn').count(), 1);
  assert.match(await guess(page), /^\d$/);
  assert.equal(await page.locator('.cnn-out .drow').count(), 10);
  assert.deepEqual(page.errors, []);
  await page.context().close();
});

test('a sample digit goes through the trained network, which answers with its label', async () => {
  const page = await app.open('#cnn');
  for (const d of [3, 8]) {
    await page.click(`[data-cnnsample="${d}"]`);
    assert.equal(await guess(page), String(d));
  }
  await page.context().close();
});

test('drawing with the mouse: a vertical stroke reads as 1, a circle as 0', async () => {
  const page = await app.open('#cnn');
  await page.locator('#cnn-pad').scrollIntoViewIfNeeded();
  await page.click('[data-cnndo="clear"]');
  await stroke(page, [[0.5, 0.15], [0.5, 0.85]]);
  assert.equal(await guess(page), '1');
  await page.click('[data-cnndo="clear"]');
  await stroke(page, Array.from({ length: 25 }, (_, k) => [0.5 + 0.25 * Math.sin(k / 24 * 2 * Math.PI), 0.5 - 0.33 * Math.cos(k / 24 * 2 * Math.PI)]));
  assert.equal(await guess(page), '0');
  await page.context().close();
});

test('"Xoá" empties the pad and the network waits for a drawing', async () => {
  const page = await app.open('#cnn');
  await page.click('[data-cnndo="clear"]');
  assert.equal(await page.locator('.cnn-guess b').count(), 0);
  assert.match(await page.locator('.cnn-guess').textContent(), /Vẽ/);
  await page.context().close();
});

test('each conv and pool block shows its real feature maps, one per channel', async () => {
  for (const [name, channels] of [['Conv 3×3, 8 kênh', 8], ['MaxPool 2×2 (1)', 8], ['Conv 3×3, 16 kênh', 16], ['MaxPool 2×2 (2)', 16]]) {
    const page = await app.open(`#cnn.${at(name)}`);
    assert.equal(await page.locator('.panel h2').textContent(), name);
    assert.equal(await page.locator('.cnn-maps canvas').count(), channels, name);
    await page.context().close();
  }
});
