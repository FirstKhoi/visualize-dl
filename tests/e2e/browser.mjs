// Opens the page in headless Chromium without touching the network: three.js is served from node_modules
// (the version must match the import map), while MathJax, the old-browser import-map shim and the fonts are
// left out, since no test reads a formula. Reduced motion makes every transition instant.
import { chromium } from 'playwright';
import http from 'node:http';
import { once } from 'node:events';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json' };

async function serveThree(route) {
  const [, version, file] = new URL(route.request().url()).pathname.match(/^\/npm\/three@([^/]+)\/(.*)$/);
  const local = JSON.parse(await readFile(path.join(ROOT, 'node_modules/three/package.json'), 'utf8')).version;
  if (local !== version) throw new Error(`index.html imports three@${version} but node_modules has ${local}: update devDependencies`);
  await route.fulfill({ body: await readFile(path.join(ROOT, 'node_modules/three', file)), contentType: 'text/javascript' });
}

export async function start() {
  const server = http.createServer(async (req, res) => {
    const file = path.join(ROOT, decodeURIComponent(new URL(req.url, 'http://x').pathname));
    try {
      const body = await readFile(file.endsWith(path.sep) ? path.join(file, 'index.html') : file);
      res.writeHead(200, { 'content-type': TYPES[path.extname(file)] || 'text/html' }).end(body);
    } catch { res.writeHead(404).end(); }
  }).listen(0, '127.0.0.1');
  await once(server, 'listening');
  const base = `http://127.0.0.1:${server.address().port}/`;
  const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });

  return {
    // a fresh page at index.html#hash, settled; page.errors collects uncaught errors and console errors
    async open(hash = '', options = {}) {
      const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 }, reducedMotion: 'reduce', colorScheme: 'light', ...options });
      const page = await ctx.newPage();
      page.errors = [];
      page.on('pageerror', e => page.errors.push(e.message));
      page.on('console', m => { if (m.type() === 'error' || m.type() === 'assert') page.errors.push(m.text()); });
      await page.route(url => !url.href.startsWith(base), route => route.abort()); // nothing leaves the machine…
      await page.route('https://cdn.jsdelivr.net/npm/three@*/**', serveThree); // …except three.js, from node_modules
      await page.route(/cdn\.jsdelivr\.net\/npm\/(mathjax|es-module-shims)@/, route => route.fulfill({ body: '', contentType: 'text/javascript' }));
      await page.route('https://fonts.googleapis.com/**', route => route.fulfill({ body: '', contentType: 'text/css' }));
      await page.goto(base + hash);
      await page.waitForSelector('.panel h2');
      await page.waitForTimeout(600);
      return page;
    },
    async close() { await browser.close(); server.close(); }
  };
}

// what the lab panel shows
export const lab = {
  steps: async page => +(await page.locator('.lab-stats b').first().textContent()),
  loss: async page => (await page.locator('.lab-stats b').nth(1).textContent()).trim(),
  runLabel: async page => (await page.locator('[data-lab="run"]').textContent()).trim()
};
