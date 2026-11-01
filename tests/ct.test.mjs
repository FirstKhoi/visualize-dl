import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadPage } from './load.mjs';

const CT = loadPage()('CT');
const near = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) <= tol, `${a} is not ${b}`);

test('a window maps [L − W/2, L + W/2] onto 0…1 and clips outside it (hand-computed)', () => {
  near(CT.gray(-600, 1500, -600), 0.5);          // the level is mid-gray
  near(CT.gray(-1000, 1500, -600), 350 / 1500);  // −1000 − (−1350) = 350
  near(CT.gray(40, 400, 40), 0.5);
  near(CT.gray(-100, 400, 40), 0.15);            // −100 − (−160) = 60, / 400
  assert.equal(CT.gray(1000, 400, 40), 1);       // above L + W/2 = 240: white
  assert.equal(CT.gray(-1000, 400, 40), 0);      // below L − W/2 = −160: black
  near(CT.gray(1000, 1800, 400), 1500 / 1800);   // 1000 − (−500) = 1500
});

test('the presets and the tissues are the values the panel cites', () => {
  assert.deepEqual(JSON.parse(JSON.stringify(CT.PRESETS)), { 'Phổi': { W: 1500, L: -600 }, 'Mô mềm': { W: 400, L: 40 }, 'Xương': { W: 1800, L: 400 } });
  assert.deepEqual([...CT.TISSUES].map(t => [...t]), [['Không khí', -1000], ['Phổi', -850], ['Mỡ', -100], ['Nước', 0], ['Mô mềm', 40], ['Xương', 1000]]);
});

test('without noise, each region of the synthetic slice holds its tissue value', () => {
  const s = CT.slice({ noise: 0 }), at = (x, y) => s.hu[y * s.N + x], name = (x, y) => CT.TISSUES[s.tissue[y * s.N + x]][0];
  assert.equal(s.N, 128);
  for (const [x, y, hu, tissue] of [[2, 2, -1000, 'Không khí'], [40, 62, -850, 'Phổi'], [8, 68, -100, 'Mỡ'], [92, 82, 0, 'Nước'], [64, 70, 40, 'Mô mềm'], [64, 100, 1000, 'Xương']]) {
    assert.equal(at(x, y), hu, `(${x}, ${y})`);
    assert.equal(name(x, y), tissue, `(${x}, ${y})`);
  }
});

test('the noise is the σ = 10 HU the panel states, and the same on every run (seeded)', () => {
  const clean = CT.slice({ noise: 0 }).hu, noisy = CT.slice().hu, d = noisy.map((v, i) => v - clean[i]);
  const mean = d.reduce((a, b) => a + b) / d.length, sd = Math.sqrt(d.reduce((a, b) => a + (b - mean) ** 2, 0) / d.length);
  assert.ok(Math.abs(mean) < 0.3 && Math.abs(sd - 10) < 0.3, `mean ${mean}, sd ${sd}`);
  assert.deepEqual([...CT.slice().hu.slice(0, 50)], [...noisy.slice(0, 50)]);
});

test('the contrast the panel quotes: fat vs soft tissue, lung vs air, in two windows', () => {
  const g = (name, { W, L }) => CT.gray(CT.TISSUES.find(t => t[0] === name)[1], W, L), soft = CT.PRESETS['Mô mềm'], lung = CT.PRESETS['Phổi'];
  near(g('Mô mềm', soft) - g('Mỡ', soft), 0.35);      // 140 HU / 400
  near(g('Mô mềm', lung) - g('Mỡ', lung), 140 / 1500); // 0.093
  assert.equal(g('Phổi', soft) - g('Không khí', soft), 0); // both clipped to black
  near(g('Phổi', lung) - g('Không khí', lung), 0.1);   // 150 HU / 1500
});
