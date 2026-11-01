import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { start } from './browser.mjs';
import { loadPage } from '../load.mjs';

let app;
before(async () => { app = await start(); });
after(() => app.close());

const get = loadPage(), MODELS = get('MODELS'), navCode = get('navCode');
const h2 = page => page.locator('.panel h2').textContent();
const until = (page, name) => page.waitForFunction(n => document.querySelector('.panel h2')?.textContent === n, name, { timeout: 15000 });
const PHONE = { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true }, SIDEWAYS = { ...PHONE, viewport: { width: 844, height: 390 } };

test('without a link the page opens on a home page; a step of a learning path opens its block, and ⌂ comes back', async () => {
  const page = await app.open('');
  assert.ok(await page.locator('#home').isVisible());
  assert.equal(await page.locator('#home .path').count(), get('PATHS').length);
  assert.equal(await page.locator('#home .path .soon', { hasText: 'nnU-Net' }).count(), 1, 'a step still to come is listed, without a link');
  assert.equal(await page.locator('#home .path a', { hasText: 'nnU-Net' }).count(), 0);
  await page.click('#home a:text-is("Dice, cross-entropy và HD95")');
  await until(page, 'Conv 1×1 → 2 lớp');
  assert.ok(await page.locator('#home').isHidden());
  assert.match(page.url(), new RegExp(`#${navCode('unet/Conv 1×1 → 2 lớp').replace(/\./g, '\\.')}$`));
  await page.click('[data-home]');
  assert.ok(await page.locator('#home').isVisible());
  await page.goBack(); // the browser's back button returns to the block
  await page.waitForFunction(() => document.getElementById('home').hidden);
  assert.equal(await h2(page), 'Conv 1×1 → 2 lớp');
  assert.deepEqual(page.errors, []);
  await page.context().close();
});

test('old links keep opening the same blocks', async () => {
  for (const [link, name] of [['transformer.2.0.2', 'Scaled Dot-Product Attention'], ['unet.10', 'Conv 1×1 → 2 lớp'], ['llm.5', 'Lấy mẫu token kế'], ['transformer', 'Transformer']]) {
    const page = await app.open('#' + link);
    assert.equal(await h2(page), name, link);
    assert.ok(await page.locator('#home').isHidden(), link);
    await page.context().close();
  }
});

test('the tabs come in groups: the open model\'s group is marked, and a group shows its models', async () => {
  const page = await app.open('#unet');
  const shown = () => page.locator('#models button:visible').allTextContents();
  assert.equal(await page.locator('[data-group][aria-pressed="true"]').textContent(), 'Y tế');
  assert.deepEqual(await shown(), ['U-Net', 'CheXNet']);
  assert.equal(await page.locator('#m-unet').getAttribute('aria-pressed'), 'true');
  await page.click('[data-group]:text-is("Transformer")');
  assert.deepEqual(await shown(), ['Transformer', 'LLaMA 3', 'ViT', 'Swin-T']);
  await page.click('#m-llm');
  await until(page, 'LLaMA 3 8B');
  assert.equal(await page.locator('[data-group][aria-pressed="true"]').textContent(), 'Transformer');
  await page.click('[data-group]:text-is("Y tế")');
  await page.click('#models button:text-is("CheXNet")'); // a shortcut into DenseNet-121, still under "Y tế"
  await until(page, 'CheXNet (2017)');
  assert.equal(await page.locator('[data-group][aria-pressed="true"]').textContent(), 'Y tế');
  await page.click('[data-group]:text-is("Chuỗi")'); // a group of one opens its model
  await until(page, 'LSTM (mô hình ngôn ngữ)');
  assert.deepEqual(page.errors, []);
  await page.context().close();
});

test('on a phone the top bar fits: one grouped menu holds every model, and the search shows no keyboard shortcut', async () => {
  for (const opts of [PHONE, SIDEWAYS]) {
    const page = await app.open('#transformer', opts);
    const top = await page.locator('.top').evaluate(el => [el.scrollWidth, el.clientWidth]);
    assert.ok(top[0] <= top[1], `the top bar is ${top[0]} px wide in ${top[1]} px`);
    assert.ok(await page.locator('#model-pick').isVisible() && await page.locator('#groups').isHidden());
    const keys = await page.locator('#model-pick option').evaluateAll(os => os.map(o => o.value));
    for (const k of Object.keys(MODELS)) assert.ok(keys.includes(k), k);
    assert.doesNotMatch(await page.locator('#find-q').getAttribute('placeholder'), /F\)/);
    await page.selectOption('#model-pick', 'lstm');
    await until(page, 'LSTM (mô hình ngôn ngữ)');
    await page.context().close();
  }
  const desk = await app.open('#transformer');
  assert.match(await desk.locator('#find-q').getAttribute('placeholder'), /\((Ctrl F|⌘F)\)/); // a keyboard: the shortcut stays
  await desk.context().close();
});

test('on a phone no two block labels overlap at the root of any model', async () => {
  for (const opts of [PHONE, SIDEWAYS]) {
    const page = await app.open('#transformer', opts), bad = [];
    for (const key of Object.keys(MODELS)) {
      await page.evaluate(k => { location.hash = k; }, key);
      await until(page, MODELS[key].name);
      await page.waitForTimeout(700); // the labels settle once the camera has stopped
      const hits = await page.evaluate(() => {
        const r = [...document.querySelectorAll('#view .tag')].map(t => [t.textContent, t.getBoundingClientRect()]).filter(([, b]) => b.width > 0), out = [];
        for (let i = 0; i < r.length; i++) for (let j = i + 1; j < r.length; j++) {
          const [a, A] = r[i], [b, B] = r[j], w = Math.min(A.right, B.right) - Math.max(A.left, B.left), h = Math.min(A.bottom, B.bottom) - Math.max(A.top, B.top);
          if (w > 1 && h > 1) out.push(`${a} / ${b}`);
        }
        return out;
      });
      if (hits.length) bad.push(`${key} (${opts.viewport.width}×${opts.viewport.height}): ${hits.join('; ')}`);
    }
    assert.deepEqual(bad, []);
    await page.context().close();
  }
});
