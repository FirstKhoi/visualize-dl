import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { start, lab } from './browser.mjs';

let app;
before(async () => { app = await start(); });
after(() => app.close());

const view = page => page.locator('#view canvas').first().screenshot();

test('the 3D view is still when nothing happens (makes the screenshot comparison below meaningful)', async () => {
  const page = await app.open('#mlp');
  const a = await view(page);
  await page.waitForTimeout(400);
  assert.ok(a.equals(await view(page)));
  await page.context().close();
});

test('the decision boundary is painted on the 3D block from the first view', async () => {
  const page = await app.open('#mlp');
  const first = await view(page);
  await page.click('[data-lab="reset"]'); // same seed, same network: the picture only changes if the first one was missing
  await page.waitForTimeout(300);
  assert.ok(first.equals(await view(page)), 'the 3D view changed after "Khởi tạo lại"');
  assert.deepEqual(page.errors, []);
  await page.context().close();
});

test('leaving the MLP stops its training run', async () => {
  const page = await app.open('#mlp');
  await page.click('[data-labset="data"][data-val="Xoắn ốc"]'); // never reaches loss < 0.001, so it can't stop by itself
  await page.click('[data-lab="run"]');
  await page.waitForTimeout(300);
  await page.click('#m-transformer');
  await page.waitForTimeout(800);
  await page.click('#m-mlp');
  await page.waitForTimeout(400);
  const a = await lab.steps(page);
  await page.waitForTimeout(800);
  assert.equal(await lab.steps(page), a, 'steps kept counting after coming back');
  assert.match(await lab.runLabel(page), /Chạy/);
  await page.context().close();
});

test('switching to dark mode repaints the boundary in dark colours', async () => {
  const page = await app.open('#mlp');
  const corner = () => page.evaluate(() => [...document.getElementById('lab-canvas').getContext('2d').getImageData(2, 2, 1, 1).data]);
  const light = await corner();
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.waitForTimeout(300);
  const dark = await corner();
  assert.ok(dark[0] + dark[1] + dark[2] < light[0] + light[1] + light[2] - 100, `corner ${light} → ${dark}`);
  await page.context().close();
});

const runOn = async (page, data) => {
  await page.click(`[data-labset="data"][data-val="${data}"]`);
  await page.click('[data-lab="run"]');
};

test('▶ after a run has solved the task starts a new run', async () => {
  const page = await app.open('#mlp');
  await runOn(page, 'Hai cụm'); // loss < 0.001 after about 30 steps, then the run stops by itself
  await page.waitForFunction(() => document.querySelector('[data-lab="run"]').textContent.includes('Chạy'), null, { timeout: 30000 });
  assert.ok(await lab.steps(page) > 0);
  // click and read in one go, before any animation frame can take a step
  const shown = await page.evaluate(() => { document.querySelector('[data-lab="run"]').click(); return document.querySelector('.lab-stats b').textContent; });
  assert.equal(shown, '0');
  await page.context().close();
});

test('the loss chart ends at the latest step when it shows only every k-th step', async () => {
  const page = await app.open('#mlp');
  await runOn(page, 'Xoắn ốc');
  await page.waitForFunction(() => +document.querySelector('.lab-stats b').textContent > 300, null, { timeout: 90000 });
  await page.click('[data-lab="run"]'); // pause
  let n = await lab.steps(page);
  while (n % Math.ceil((n + 1) / 300) === 0) { await page.click('[data-lab="step"]'); n = await lab.steps(page); } // land off the k-step grid
  const labels = await page.locator('.lab-out svg text').allTextContents();
  assert.ok(labels.includes(String(n)), `step ${n} is not on the axis: ${labels.join(' ')}`);
  await page.context().close();
});

test('training does not make screen readers re-read the panel', async () => {
  const page = await app.open('#mlp');
  await runOn(page, 'Xoắn ốc');
  await page.waitForTimeout(300);
  const announced = await page.evaluate(() => new Promise(done => {
    let n = 0;
    const region = node => (node.nodeType === 1 ? node : node.parentElement).closest('[aria-live]');
    const mo = new MutationObserver(ms => { for (const m of ms) { const r = region(m.target); if (r && r.getAttribute('aria-live') !== 'off') n++; } });
    mo.observe(document.getElementById('panel'), { subtree: true, childList: true, characterData: true });
    setTimeout(() => { mo.disconnect(); done(n); }, 1000);
  }));
  assert.equal(announced, 0, 'changes inside a live region during one second of training');
  await page.context().close();
});

test('the loss chart tooltip stays up while training redraws the chart', async () => {
  const page = await app.open('#mlp');
  await page.click('[data-labset="data"][data-val="Xoắn ốc"]');
  await page.click('[data-lab="step"]'); // the chart appears after the first step
  const chart = page.locator('.lab-out .chart svg');
  await chart.scrollIntoViewIfNeeded(); // while nothing redraws it; during a run it is replaced ~8 times a second
  const box = await chart.boundingBox();
  await page.evaluate(() => document.querySelector('[data-lab="run"]').click()); // start without scrolling the panel
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.waitForTimeout(600); // several redraws
  assert.ok(await page.locator('.lab-out .chart .tip').isVisible());
  await page.context().close();
});

test('"Bước +1" shows the loss of the updated weights', async () => {
  const page = await app.open('#mlp');
  const before = await lab.loss(page);
  await page.click('[data-lab="step"]');
  assert.equal(await lab.steps(page), 1);
  assert.notEqual(await lab.loss(page), before);
  await page.context().close();
});
