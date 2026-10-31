// Records media/demo.gif, the README's demo: open the Transformer down to its attention, draw a digit for the trained CNN,
// then train the MLP lab. Needs what the browser tests need, ffmpeg, and the network for the real fonts.
//   node scripts/record-demo.mjs [--lang vi]      English by default
// Chromium here draws WebGL in software, so the 3D runs at a few frames per second on a machine without a GPU, and
// the lab trains slower than on a real one; the GIF shows what it can.
import { parseArgs } from 'node:util';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { start } from '../tests/e2e/browser.mjs';

const { values: arg } = parseArgs({ options: { lang: { type: 'string', default: 'en' } } });
const OUT = fileURLToPath(new URL('../media/demo.gif', import.meta.url));
const dir = mkdtempSync(path.join(tmpdir(), 'demo-')), size = { width: 1280, height: 760 };

const app = await start();
const before = Date.now();
const page = await app.open(`?lang=${arg.lang}#transformer`, { mathjax: true, fonts: true, viewport: size, recordVideo: { dir, size }, reducedMotion: 'no-preference' });
await page.evaluate(() => document.fonts.ready);
const skip = (Date.now() - before) / 1000; // the video starts with the page loading: cut that
const pause = ms => page.waitForTimeout(ms);
const open = async k => { await page.click(`.panel button.kid[data-k="${k}"]`); await pause(2300); };

await pause(2000);
await open(2); // Encoder
await open(0); // Self-Attention
await open(2); // core attention: the live heatmaps
await pause(1000);

await page.click('#m-cnn');
await pause(2200);
await page.evaluate(() => document.getElementById('cnn-pad').scrollIntoView({ block: 'center', behavior: 'smooth' }));
await pause(900);
const pad = await page.locator('#cnn-pad').boundingBox(), at = ([x, y]) => [pad.x + x * pad.width, pad.y + y * pad.height];
// a "2": the top arc, the diagonal, the base
const stroke = [[0.3, 0.32], [0.38, 0.2], [0.52, 0.16], [0.66, 0.22], [0.7, 0.36], [0.62, 0.5], [0.48, 0.63], [0.32, 0.8], [0.5, 0.8], [0.72, 0.8]];
await page.mouse.move(...at(stroke[0]));
await page.mouse.down();
for (const p of stroke.slice(1)) await page.mouse.move(...at(p), { steps: 6 });
await page.mouse.up();
await pause(2500);

await page.click('#m-mlp');
await pause(2000);
await page.evaluate(() => { const p = document.getElementById('panel'); p.scrollTo({ top: p.querySelector('.lab-btns').offsetTop - 24, behavior: 'smooth' }); });
await pause(900);
await page.click('[data-lab="run"]');
await pause(5000);

const video = page.video();
await page.context().close();
await app.close();
const webm = await video.path();
const palette = 'fps=8,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle';
execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-ss', skip.toFixed(2), '-i', webm, '-vf', palette, '-loop', '0', OUT]);
rmSync(dir, { recursive: true });
console.log(`${OUT}: ${(statSync(OUT).size / 1e6).toFixed(1)} MB`);
