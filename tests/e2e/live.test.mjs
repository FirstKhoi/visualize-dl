import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { start } from './browser.mjs';
import { loadPage } from '../load.mjs';

let app;
before(async () => { app = await start(); });
after(() => app.close());

// the first block carrying each kind of live number, with the link that opens it
const MODELS = loadPage()('MODELS'), first = new Map();
for (const [model, root] of Object.entries(MODELS)) (function walk(nd, path) {
  if (nd.live && !first.has(nd.live)) first.set(nd.live, { link: [model, ...path].join('.'), name: nd.name });
  (nd.children || []).forEach((k, i) => walk(k, [...path, i]));
})(root, []);

// which panel section each kind brings: the toy attention, the toy convolution, the lab, or none (weights painted in 3D only)
const SECTION = { conv: 'section.live.conv', boundary: 'section.lab', W1: null, W2: null, W3: null };

test('every kind of live block opens with its panel section and no errors', async () => {
  assert.ok(first.size >= 11, [...first.keys()].join(' '));
  for (const [key, { link, name }] of first) {
    const page = await app.open('#' + link);
    assert.equal(await page.locator('.panel h2').textContent(), name, link);
    const want = key in SECTION ? SECTION[key] : 'section.live:not(.conv)';
    for (const sel of ['section.live:not(.conv)', 'section.live.conv', 'section.lab'])
      assert.equal(await page.locator(sel).count(), sel === want ? 1 : 0, `${key} (${link}): ${sel}`);
    assert.deepEqual(page.errors, [], key);
    await page.context().close();
  }
});
