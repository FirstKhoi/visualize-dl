import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { start } from './browser.mjs';
import { loadPage } from '../load.mjs';

let app;
before(async () => { app = await start(); });
after(() => app.close());

// the blocks whose PyTorch code has the longest first lines, where a button in the code's corner covers the most
const MODELS = loadPage()('MODELS'), blocks = [];
for (const [m, root] of Object.entries(MODELS)) (function walk(nd, p) {
  if (nd.info?.code) blocks.push({ link: [m, ...p].join('.'), first: nd.info.code.split('\n')[0].length });
  (nd.children || []).forEach((k, i) => walk(k, [...p, i]));
})(root, []);
const longest = blocks.sort((a, b) => b.first - a.first).slice(0, 3).map(b => b.link);

test('the copy button sits outside the code it copies', async () => {
  for (const link of ['transformer', ...longest]) {
    for (const viewport of [{ width: 1280, height: 800 }, { width: 390, height: 844 }]) {
      const page = await app.open('#' + link, { viewport });
      const code = page.locator('.panel pre').last(), copy = page.locator('.panel [data-act="copy"]');
      await copy.scrollIntoViewIfNeeded();
      const a = await copy.boundingBox(), b = await code.boundingBox();
      const overlap = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x)) * Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
      assert.equal(overlap, 0, `#${link} at ${viewport.width}px: the button covers ${Math.round(overlap)} px² of the code`);
      await page.context().close();
    }
  }
});

test('the copy button copies the block\'s code', async () => {
  const page = await app.open('#transformer');
  await page.evaluate(() => { window.__copied = null; navigator.clipboard.writeText = t => { window.__copied = t; return Promise.resolve(); }; });
  await page.click('.panel [data-act="copy"]');
  assert.match(await page.evaluate(() => window.__copied), /nn\.Transformer\(d_model=512/);
  await page.context().close();
});

// found by opening every block at phone sizes: formulas inside a sentence cannot wrap, and the toy convolution's
// 8×8 output grid is wider than the panel of a phone held sideways
const PHONE = { width: 390, height: 844 }, SIDEWAYS = { width: 844, height: 390 };
test('on phones no formula or grid pushes the panel sideways', async () => {
  for (const [link, viewport, what] of [
    ['resnet.1.0', PHONE, 'stem conv: H_out = ⌊(224 + 2·3 − 7)/2⌋ + 1 in "Ví dụ số"'],
    ['densenet.8.0', PHONE, 'dense layer parameter sum in "Tham số"'],
    ['llm.2.4.1', SIDEWAYS, 'SiLU([−1.5, 0.3, 2]) in "Ví dụ số"'],
    ['densenet.8.0', SIDEWAYS, 'dense layer parameter sum in "Tham số"'],
    ['mlp', SIDEWAYS, '2H + H + H² + … in "Tham số"'],
    ['resnet.2.0.3', SIDEWAYS, 'the toy convolution\'s output grid']
  ]) {
    const page = await app.open('#' + link, { viewport, mathjax: true });
    const [scroll, width] = await page.evaluate(() => { const p = document.getElementById('panel'); return [p.scrollWidth, p.clientWidth]; });
    assert.ok(scroll <= width, `#${link} at ${viewport.width}×${viewport.height}: the panel scrolls ${scroll - width}px sideways (${what})`);
    assert.deepEqual(page.errors, [], link);
    await page.context().close();
  }
});
