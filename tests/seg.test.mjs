import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadPage } from './load.mjs';

const SEG = loadPage()('SEG');
const near = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) <= tol, `${a} is not ${b}`);
// a mask of size N×N from a list of [x, y] pixels, or from a w×h rectangle at (x, y)
const mask = (N, pts) => { const m = new Uint8Array(N * N); for (const [x, y] of pts) m[y * N + x] = 1; return m; };
const rect = (N, x0, y0, w, h) => mask(N, Array.from({ length: w * h }, (_, k) => [x0 + k % w, y0 + Math.floor(k / w)]));

test('Dice and IoU count overlap (hand-computed: a 4×4 square moved by one pixel)', () => {
  const G = rect(32, 10, 10, 4, 4), P = rect(32, 11, 10, 4, 4); // overlap 3×4 = 12
  near(SEG.dice(P, G), 24 / 32);
  near(SEG.iou(P, G), 12 / 20);
  assert.equal(SEG.dice(new Uint8Array(32 * 32), G), 0);                        // nothing predicted
  assert.ok(Number.isNaN(SEG.dice(new Uint8Array(32 * 32), new Uint8Array(32 * 32)))); // nothing to find: 0 / 0
});

test('cross-entropy and soft Dice from probabilities (hand-computed)', () => {
  const G = mask(2, [[0, 0]]); // 4 pixels, one of them lesion
  near(SEG.bce(new Float64Array(4).fill(0.5), G), Math.log(2));
  near(SEG.bce(Float64Array.from([0.9, 0.1, 0.1, 0.1]), G), -Math.log(0.9));
  near(SEG.softDice(new Float64Array(4).fill(0.5), G), 2 * 0.5 / (2 + 1));
});

test('the 95th percentile interpolates as NumPy does', () => {
  near(SEG.percentile([1, 2, 3, 4], 95), 3.85); // h = 0.95 · 3 = 2.85 → 3 + 0.85 · (4 − 3)
  near(SEG.percentile([0, 10], 50), 5);
  near(SEG.percentile([7], 95), 7);
  near(SEG.percentile([4, 0, 0, 0, 0], 95), 3.2); // sorts first
});

test('the boundary: mask pixels with a 4-neighbour outside the mask or the image', () => {
  assert.equal(SEG.boundary(rect(16, 5, 5, 3, 3), 16).length, 8);  // all but the center
  assert.equal(SEG.boundary(rect(16, 5, 5, 4, 4), 16).length, 12); // all but the 2×2 inside
  assert.equal(SEG.boundary(rect(16, 0, 0, 3, 3), 16).length, 8);  // the image edge counts as outside
  assert.deepEqual([...SEG.boundary(mask(16, [[3, 4]]), 16)].map(p => [...p]), [[3, 4]]);
});

test('HD95: the larger of the two directed 95th percentiles of boundary distances (hand-computed)', () => {
  near(SEG.hd95(mask(16, [[0, 0]]), mask(16, [[3, 4]]), 16), 5);
  near(SEG.hd95(rect(32, 13, 10, 2, 2), rect(32, 10, 10, 2, 2), 32), 3);
  // a 2×2 lesion found, plus one stray pixel 29 px away: Dice still 0.889, HD95 jumps
  const G = rect(64, 10, 10, 2, 2), P = mask(64, [[10, 10], [11, 10], [10, 11], [11, 11], [40, 10]]);
  near(SEG.dice(P, G), 8 / 9);
  near(SEG.hd95(P, G, 64), 0.8 * 29); // distances from P: [0, 0, 0, 0, 29] → 0 + 0.8 · 29; from G: all 0
  assert.ok(Number.isNaN(SEG.hd95(new Uint8Array(64 * 64), G, 64))); // nothing predicted: no boundary to measure
});

test('the scene of the panel: a small lesion, and a network that predicts background everywhere', () => {
  const s = SEG.scene({ r: 4, none: true }), share = s.G.reduce((a, b) => a + b) / s.G.length;
  assert.ok(share > 0.009 && share < 0.013, share); // about 1% of the image
  assert.ok(s.m.ce < 0.07 && s.m.acc > 0.98, JSON.stringify(s.m)); // low loss, 99% of pixels right…
  assert.equal(s.m.dice, 0);                                        // …and not one lesion pixel found
  const ok = SEG.scene({ r: 4 }).m, moved = SEG.scene({ r: 4, shift: 2 }).m, big = SEG.scene({ r: 16, shift: 2 }).m;
  assert.equal(ok.dice, 1);
  assert.ok(moved.dice < 0.75 && big.dice > 0.9, `${moved.dice} ${big.dice}`); // the same 2 px cost a small lesion far more
});
