import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { loadPage } from './load.mjs';

const get = loadPage(), CNN = get('CNN'), MNIST_CNN = get('MNIST_CNN');
const FIX = JSON.parse(readFileSync(new URL('./fixtures/mnist-cnn.json', import.meta.url)));
const argmax = a => a.indexOf(Math.max(...a));

test('the page runs the trained network exactly as training measured it: same logits on MNIST test images', () => {
  FIX.images.forEach((img, k) => {
    const out = CNN.run(img.map(v => v / 255));
    FIX.logits[k].forEach((want, d) => assert.ok(Math.abs(out.logits[d] - want) < 1e-9, `image ${k}, digit ${d}: ${out.logits[d]} vs ${want}`));
  });
});

test('every layer has the shape of the architecture: 8@26×26 → 8@13×13 → 16@11×11 → 16@5×5 → 10', () => {
  const out = CNN.run(new Array(784).fill(0));
  const shape = m => [m.length, m[0].length, m[0][0].length];
  assert.deepEqual([shape(out.c1), shape(out.p1), shape(out.c2), shape(out.p2), out.logits.length], [[8, 26, 26], [8, 13, 13], [16, 11, 11], [16, 5, 5], 10]);
  assert.ok(Math.abs(out.probs.reduce((a, b) => a + b) - 1) < 1e-12);
});

test('the exported weights are what training reports: 5 258 parameters, at least 98% on the MNIST test set', () => {
  // conv1 8·(1·9) + 8, conv2 16·(8·9) + 16, fc 10·400 + 10
  assert.equal(Object.values(MNIST_CNN.weights).reduce((n, w) => n + w.data.length, 0), 80 + 1168 + 4010);
  assert.ok(MNIST_CNN.test_acc >= 0.98, `${MNIST_CNN.test_acc}`);
});

test('the sample digits are MNIST test images the network gets right, one per digit', () => {
  assert.deepEqual([...MNIST_CNN.samples.map(s => s.label)], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]); // […]: arrays of the scripts' vm realm
  for (const s of MNIST_CNN.samples) assert.equal(argmax(CNN.run(CNN.decode(s.pixels)).logits), s.label);
});

// ---------- turning a drawing into an MNIST-like input ----------
const blank = n => Array.from({ length: n }, () => new Array(n).fill(0));
const paint = (img, top, left, h, w, v = 1) => { for (let i = top; i < top + h; i++) for (let j = left; j < left + w; j++) img[i][j] = v; return img; };
const rowsCols = x => { // the rows and columns of a 28×28 input that hold any ink
  const at = (i, j) => x[i * 28 + j], rows = [], cols = [];
  for (let i = 0; i < 28; i++) if ([...Array(28).keys()].some(j => at(i, j) > 0)) rows.push(i);
  for (let j = 0; j < 28; j++) if ([...Array(28).keys()].some(i => at(i, j) > 0)) cols.push(j);
  return { rows, cols };
};

test('an empty drawing becomes an empty input', () => {
  assert.deepEqual([...CNN.prepare(blank(140))], new Array(784).fill(0));
});

test('ink is scaled into a 20×20 box and centred, as MNIST digits were', () => {
  // a 40×40 square in the top-left corner of a 140 px canvas: scaled ×0.5 to 20×20, then centred, so rows and columns 4…23
  const { rows, cols } = rowsCols(CNN.prepare(paint(blank(140), 0, 0, 40, 40)));
  assert.deepEqual([rows[0], rows.at(-1), cols[0], cols.at(-1)], [4, 23, 4, 23]);
});

test('the aspect ratio is kept: a tall bar stays a tall bar', () => {
  // 100 tall, 20 wide → scale 20/100: 20 rows, 4 columns, centred at (14, 14): rows 4…23, columns 12…15
  const { rows, cols } = rowsCols(CNN.prepare(paint(blank(140), 10, 60, 100, 20)));
  assert.deepEqual([rows[0], rows.at(-1), cols[0], cols.at(-1)], [4, 23, 12, 15]);
});

test('centring uses the centre of mass, not the bounding box', () => {
  // an L: stem 100×20, foot 20×100, box 100×100 → scale 0.2. By hand, the mass sits at (13.56, 6.44) of the 20×20
  // patch, so it lands at rows 0…19 and columns 8…27 (mass at 13.56, 14.44); centring the box would put it at 10.44
  const img = paint(paint(blank(140), 10, 10, 100, 20), 90, 10, 20, 100);
  const x = CNN.prepare(img);
  let m = 0, ci = 0, cj = 0;
  for (let i = 0; i < 28; i++) for (let j = 0; j < 28; j++) { m += x[i * 28 + j]; ci += i * x[i * 28 + j]; cj += j * x[i * 28 + j]; }
  assert.ok(Math.abs(ci / m + 0.5 - 14) <= 0.5 && Math.abs(cj / m + 0.5 - 14) <= 0.5, `centre of mass at (${ci / m}, ${cj / m})`);
});
