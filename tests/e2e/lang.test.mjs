import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { start } from './browser.mjs';

let app;
before(async () => { app = await start(); });
after(() => app.close());

const headings = page => page.locator('.panel h3').allTextContents();

test('the page starts in Vietnamese, and "EN" switches it to English, links included', async () => {
  const page = await app.open('#cnn');
  assert.equal(await page.evaluate(() => document.documentElement.lang), 'vi');
  assert.ok((await headings(page)).includes('Công thức'));
  await Promise.all([page.waitForNavigation(), page.click('[data-act="lang"]')]);
  await page.waitForSelector('.panel h2');
  assert.equal(await page.evaluate(() => document.documentElement.lang), 'en');
  assert.match(page.url(), /[?&]lang=en(&|#|$)/);
  assert.match(page.url(), /#cnn$/, 'the same block stays open');
  assert.equal(await page.locator('.panel h2').textContent(), 'Small CNN on MNIST');
  const h = await headings(page);
  assert.ok(h.includes('Formula') && h.includes('Parameters'), h.join(' | '));
  assert.match(await page.locator('.panel .feedback a').textContent(), /feedback/i);
  assert.match(await page.locator('#m-mlp').textContent(), /Training lab/);
  assert.deepEqual(page.errors, []);
  await page.context().close();
});

test('the choice is remembered, and ?lang=vi switches back', async () => {
  const page = await app.open('#transformer');
  await Promise.all([page.waitForNavigation(), page.click('[data-act="lang"]')]);
  await page.waitForSelector('.panel h2');
  await page.goto(page.url().replace(/\?lang=en/, '')); // no parameter: the remembered language
  await page.waitForSelector('.panel h2');
  assert.equal(await page.evaluate(() => document.documentElement.lang), 'en');
  await page.goto(page.url().split('?')[0].split('#')[0] + '?lang=vi#transformer');
  await page.waitForSelector('.panel h2');
  assert.equal(await page.evaluate(() => document.documentElement.lang), 'vi');
  await page.context().close();
});

test('in English, a block not translated yet shows its Vietnamese text and says so', async () => {
  const page = await app.open('?lang=en#vit');
  assert.equal(await page.locator('.panel .untranslated').count(), 1);
  assert.equal(await page.locator('.panel h2').textContent(), 'ViT-B/16');
  await page.context().close();
});

test('the English demos speak English: attention, sampling, the lab and the CNN', async () => {
  for (const [hash, expect] of [['transformer.2.0.2', /Live numbers/], ['llm.5', /Temperature/], ['mlp', /Training lab/], ['cnn', /The network sees/]]) {
    const page = await app.open(`?lang=en#${hash}`);
    assert.match(await page.locator('.panel').textContent(), expect, hash);
    assert.deepEqual(page.errors, [], hash);
    await page.context().close();
  }
});

test('the English tabs show no Vietnamese: panels, demos, 3D labels, legend, breadcrumbs', async () => {
  const VI = /[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]/i;
  // a root, a block with a demo or experiment, and a plain block in each translated tab
  for (const hash of ['transformer', 'transformer.2', 'transformer.2.0.2', 'transformer.2.0.2.2', 'transformer.2.1.1', 'llm', 'llm.2.1', 'llm.5', 'llm.5.3',
    'mlp', 'mlp.1', 'mlp.6', 'cnn', 'cnn.1', 'cnn.6', 'cnn.8']) {
    const page = await app.open(`?lang=en#${hash}`);
    const shown = await page.evaluate(() => {
      const panel = document.querySelector('.panel').cloneNode(true);
      panel.querySelectorAll('.xlinks').forEach(x => x.remove()); // links into the tabs not translated yet keep their names
      const attrs = el => [...el.querySelectorAll('[aria-label], [title]')].filter(e => e.id !== 'lang') // the way back to Vietnamese is labelled in Vietnamese
        .flatMap(e => [e.getAttribute('aria-label'), e.getAttribute('title')]);
      const out = [...document.querySelectorAll('.tag, .link-tag, #crumbs, #legend, .hint')].map(e => e.textContent);
      return [panel.textContent, ...attrs(panel), ...out, ...attrs(document.querySelector('.top')), document.title].filter(Boolean);
    });
    assert.deepEqual(shown.flatMap(s => s.split(/(?<=[.:])\s+/)).filter(s => VI.test(s)), [], hash);
    assert.deepEqual(page.errors, [], hash);
    await page.context().close();
  }
});
