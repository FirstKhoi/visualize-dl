// Opens every block of every model with real formulas, at several screen sizes, and reports what goes wrong in the panel:
//   1. text covered by a positioned element (a button, a tooltip…);
//   2. the panel scrolling sideways: something wider than it that no scroll box of its own contains;
//   3. text clipped by overflow: hidden or an ellipsis.
// A block that repeats (Stable Diffusion's 1003 blocks are mostly the same few, many times) is checked once per screen size.
// Needs what the browser tests need (npm ci && npx playwright install chromium).
//   node scripts/audit-layout.mjs                       every screen size, every model, Vietnamese
//   node scripts/audit-layout.mjs --size phone,phoneLand --model swin,cnn --lang en
//   --fonts  the real fonts from Google Fonts (needs the network); without it, fallback fonts, as in the tests
//   --all    every block, repeats included (about 4 times longer)
// Exits with 1 when it finds a problem.
import { parseArgs } from 'node:util';
import { start } from '../tests/e2e/browser.mjs';
import { loadPage } from '../tests/load.mjs';

const SIZES = { desktop: [1440, 900], laptop: [1280, 800], small: [1024, 700], phone: [390, 844], phoneLand: [844, 390] };
const { values: arg } = parseArgs({ options: { size: { type: 'string' }, model: { type: 'string' }, lang: { type: 'string', default: 'vi' }, fonts: { type: 'boolean' }, all: { type: 'boolean' } } });
const sizes = arg.size ? arg.size.split(',') : Object.keys(SIZES);
const MODELS = loadPage()('MODELS'), models = arg.model ? arg.model.split(',') : Object.keys(MODELS);
for (const k of [...sizes.filter(s => !SIZES[s]), ...models.filter(m => !MODELS[m])]) throw new Error(`unknown size or model: ${k}`);

// every block's link; a block with the same text and parts as one already listed shows the same panel, so it is left out
const seen = new Set(), links = models.flatMap(m => {
  const out = [];
  (function walk(nd, p) {
    const panel = JSON.stringify([nd.name, nd.cat, nd.shape, nd.repeat, nd.info, nd.live, nd.exp, nd.ref, (nd.children || []).map(k => k.name)]);
    if (arg.all || !seen.has(panel)) out.push([m, ...p].join('.'));
    seen.add(panel);
    (nd.children || []).forEach((k, i) => walk(k, [...p, i]));
  })(MODELS[m], []);
  return out;
});

// runs in the page: the problems of the panel as it is now
function problems() {
  const panel = document.getElementById('panel'), out = [];
  const label = el => { const c = String(el.className?.baseVal ?? el.className ?? '').split(' ')[0]; return el.tagName.toLowerCase() + (c ? '.' + c : ''); };
  const where = el => el.closest('section')?.querySelector('h3')?.textContent || 'header';
  const forEyes = el => !el.closest('mjx-assistive-mml'); // MathJax's copy of each formula for screen readers is clipped to nothing on purpose
  const box = (a, b) => ({ left: Math.max(a.left, b.left), right: Math.min(a.right, b.right), top: Math.max(a.top, b.top), bottom: Math.min(a.bottom, b.bottom) });
  const area = r => Math.max(0, r.right - r.left) * Math.max(0, r.bottom - r.top);
  const scrolls = el => getComputedStyle(el).overflowX !== 'visible' || getComputedStyle(el).overflowY !== 'visible';
  // what can be seen of a rect: cut by every scroll box inside the panel (not the panel itself: the parts below its fold count too)
  const visible = (r, el) => { for (let a = el.parentElement; a && a !== panel; a = a.parentElement) if (scrolls(a)) r = box(r, a.getBoundingClientRect()); return r; };

  // 1. positioned elements over text
  const texts = [], walk = document.createTreeWalker(panel, NodeFilter.SHOW_TEXT, { acceptNode: t => t.textContent.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT });
  for (let t; (t = walk.nextNode());) texts.push(t);
  const covers = [...panel.querySelectorAll('*')].filter(el => forEyes(el) && ['absolute', 'fixed', 'sticky'].includes(getComputedStyle(el).position)
    && el.offsetParent !== null && getComputedStyle(el).visibility !== 'hidden' && !el.hidden);
  for (const c of covers) {
    const cr = c.getBoundingClientRect();
    for (const t of texts) {
      if (c.contains(t)) continue;
      const range = document.createRange();
      range.selectNodeContents(t);
      if ([...range.getClientRects()].some(r => area(box(visible(r, t.parentElement), cr)) > 4)) {
        out.push(`${label(c)} "${c.textContent.trim().slice(0, 20)}" covers "${t.textContent.trim().slice(0, 40)}" (${where(c)})`);
        break;
      }
    }
  }
  // 2. the panel scrolls sideways; the culprits are the outermost elements past its edge that no scroll box of their own contains
  if (panel.scrollWidth > panel.clientWidth + 1) {
    const edge = panel.getBoundingClientRect().right - parseFloat(getComputedStyle(panel).paddingRight) + 1;
    const contained = el => { for (let a = el.parentElement; a && a !== panel; a = a.parentElement) if (getComputedStyle(a).overflowX !== 'visible') return true; return false; };
    const past = el => el.getBoundingClientRect().right > edge && !contained(el);
    const wide = [...panel.querySelectorAll('*')].filter(el => past(el) && (el.parentElement === panel || !past(el.parentElement)));
    const desc = el => el.tagName === 'MJX-CONTAINER' ? `math "${el.querySelector('mjx-assistive-mml')?.textContent.slice(0, 30)}"` : `${label(el)} "${el.textContent.trim().slice(0, 30)}"`;
    out.push(`the panel scrolls sideways by ${panel.scrollWidth - panel.clientWidth}px: ${[...new Set(wide.map(el => `${desc(el)} in ${where(el)}`))].slice(0, 3).join('; ')}`);
  }
  // 3. clipped text
  for (const el of panel.querySelectorAll('*')) {
    const cs = getComputedStyle(el);
    if (forEyes(el) && (cs.overflowX === 'hidden' || cs.textOverflow === 'ellipsis') && el.scrollWidth > el.clientWidth + 1 && el.textContent.trim())
      out.push(`${label(el)} clips its text "${el.textContent.trim().slice(0, 30)}" (${where(el)})`);
  }
  return out;
}

const app = await start();
let total = 0;
try {
  for (const size of sizes) {
    const [width, height] = SIZES[size];
    const page = await app.open(`?lang=${arg.lang}#${links[0]}`, { mathjax: true, fonts: arg.fonts, viewport: { width, height } });
    if (arg.fonts) await page.evaluate(() => document.fonts.ready);
    const found = new Map(); // problem → the blocks where it happens
    for (const link of links) {
      // a marker in the panel: once it is gone, the panel has been rebuilt for the new block (the block already open needs nothing)
      const moved = await page.evaluate(l => {
        if (location.hash.slice(1) === l) return false;
        document.getElementById('panel').append(Object.assign(document.createElement('i'), { id: 'audit-mark' }));
        location.hash = l;
        return true;
      }, link);
      if (moved) await page.waitForFunction(() => !document.getElementById('audit-mark'), null, { timeout: 15000 });
      await page.evaluate(() => window.MathJax.typesetPromise([document.getElementById('panel')]));
      for (const p of await page.evaluate(problems)) found.set(p, [...(found.get(p) || []), link]);
    }
    if (page.errors.length) found.set(`errors on the page: ${[...new Set(page.errors)].join(' | ')}`, ['']);
    console.log(`\n${size} ${width}×${height}, ${arg.lang}: ${links.length} blocks, ${found.size ? `${found.size} problem${found.size > 1 ? 's' : ''}` : 'no problems'}`);
    for (const [p, ls] of found) console.log(`  [${ls.length}×] ${p}   e.g. #${ls[0]}`);
    total += found.size;
    await page.context().close();
  }
} finally { await app.close(); }
process.exitCode = total ? 1 : 0;
