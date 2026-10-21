import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { start } from './browser.mjs';
import { loadPage } from '../load.mjs';

let app;
before(async () => { app = await start(); });
after(() => app.close());

const llm = loadPage()('MODELS.llm'), at = llm.children.findIndex(k => k.name === 'Lấy mẫu token kế');
const LINK = `#llm.${at}`, topP = llm.children[at].children.findIndex(k => k.name === 'Top-p (nucleus)');
const slide = (page, key, value) => page.evaluate(([k, v]) => { const s = document.querySelector(`[data-smp="${k}"]`); s.value = v; s.dispatchEvent(new Event('input', { bubbles: true })); }, [key, value]);
const kept = page => page.locator('.dist .drow:not(.cut)').count();
const topPct = async page => parseFloat(await page.locator('.dist .drow .dp').first().textContent());

test('the sampling block shows the next-word distribution after the prompt', async () => {
  const page = await app.open(LINK);
  assert.match(await page.locator('.smp-prompt').textContent(), /Hôm nay trời rất/);
  assert.equal(await page.locator('.dist .drow').count(), 12);
  assert.equal(await kept(page), 5); // T = 0.7, top-p = 0.9
  assert.deepEqual(page.errors, []);
  await page.context().close();
});

test('a higher temperature flattens the distribution and a lower top-p cuts more words', async () => {
  const page = await app.open(LINK);
  const top = await topPct(page), keptBefore = await kept(page);
  await slide(page, 'T', 7); // T = 2
  assert.ok(await topPct(page) < top, 'the most likely word should lose probability');
  assert.ok(await kept(page) > keptBefore, 'more words should fit under top-p');
  await slide(page, 'p', 0); // top-p = 0.1
  assert.equal(await kept(page), 1);
  await page.context().close();
});

test('draws land only on words that survive top-p, and the panel counts them', async () => {
  const page = await app.open(LINK);
  await page.click('[data-smpdo="draw100"]');
  const rows = await page.locator('.dist .drow').evaluateAll(rs => rs.map(r => ({ cut: r.classList.contains('cut'), n: +r.querySelector('.dn').textContent || 0 })));
  assert.equal(rows.reduce((s, r) => s + r.n, 0), 100);
  assert.ok(rows.filter(r => r.cut).every(r => r.n === 0));
  const word = await page.locator('.smp-prompt b').textContent();
  assert.ok(rows.some((r, i) => !r.cut) && word.trim().length > 0, 'the last word drawn is shown after the prompt');
  await page.context().close();
});

test('the stages inside the sampling block run the same demo', async () => {
  const page = await app.open(`${LINK}.${topP}`);
  assert.equal(await page.locator('.panel h2').textContent(), 'Top-p (nucleus)');
  assert.equal(await page.locator('section.live.sample .dist .drow').count(), 12);
  await page.context().close();
});
