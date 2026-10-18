import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { start } from './browser.mjs';
import { loadPage } from '../load.mjs';

let app;
before(async () => { app = await start(); });
after(() => app.close());

test('the gradient experiment charts tell screen readers what they plot', async () => {
  const lstm = loadPage()('MODELS.lstm.children.findIndex(k => k.exp === "rnn")'); // the LSTM stack hosts the time experiment
  for (const [link, unit] of [['resnet', 'lớp'], [`lstm.${lstm}`, 'bước']]) {
    const page = await app.open('#' + link);
    assert.match(await page.locator('.exp-out svg').getAttribute('aria-label'), new RegExp(`^Độ lớn gradient theo số ${unit}`), link);
    await page.context().close();
  }
});
