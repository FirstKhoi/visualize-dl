import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { start } from './browser.mjs';
import { loadPage } from '../load.mjs';

let app;
before(async () => { app = await start(); });
after(() => app.close());

test('picking a patch in the shifted-window demo updates what it attends to', async () => {
  // stage 1 → SW-MSA block → its attention: the demo shows there
  const at = loadPage()('[2, MODELS.swin.children[2].children.findIndex(k => k.name === "Khối Swin (SW-MSA)"), 1].join(".")');
  const page = await app.open('#swin.' + at);
  const reading = () => page.locator('.swin-out .reading').textContent();
  assert.match(await reading(), /Patch \(8, 4\).*16 patch.*8 patch.*32 patch/s); // default: a bottom-edge patch
  await page.locator('[data-sp="3,3"]').first().click();
  assert.match(await reading(), /Patch \(4, 4\).*64 patch.*không cần mặt nạ/s); // a centre patch reaches the whole grid
  assert.deepEqual(page.errors, []);
  await page.context().close();
});
