import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadPage } from './load.mjs';

const TRAIN = loadPage()('TRAIN');
const ADAM = { act: 'relu', opt: 'adam', lr: 0.03 };
const flat = W => [...W.flat(2)]; // a plain array of this realm: the scripts' arrays have the vm context's prototypes

test('a step reports the loss and accuracy of the weights it leaves behind', () => {
  const net = TRAIN.init(8), D = TRAIN.makeData('Vòng tròn'), before = TRAIN.lossGrad(net, D, 'relu').loss;
  const r = TRAIN.step(net, D, ADAM), now = TRAIN.lossGrad(net, D, 'relu');
  assert.notEqual(r.loss, before);
  assert.ok(Math.abs(r.loss - now.loss) < 1e-12, `${r.loss} vs ${now.loss}`);
  assert.equal(r.acc, now.acc);
});

// a loss curve: `fall` steps going from 1 down to `floor`, then `flat` steps wobbling by ±`wobble` (relative)
const curve = (fall, floor, flat, wobble) => [...Array.from({ length: fall }, (_, i) => 1 - (1 - floor) * i / fall),
  ...Array.from({ length: flat }, (_, i) => floor * (1 + (i % 2 ? wobble : -wobble)))];
const data = (ones, n = 200) => ({ X: Array.from({ length: n }, () => [0, 0]), Y: Array.from({ length: n }, (_, i) => +(i < ones)) });

test('rounding noise on a flat loss is not called oscillation', () => {
  assert.notEqual(TRAIN.diagnose(curve(200, 0.33, 300, 1e-7), 0.865, data(100)), 'oscillating');
});

test('a loss bouncing by a few percent is called oscillation', () => {
  assert.equal(TRAIN.diagnose(curve(200, 0.33, 300, 0.03), 0.865, data(100)), 'oscillating');
});

test('a dead network is recognised on unbalanced data', () => {
  // 111 of 200 points in class 1: guessing p = 0.555 everywhere costs −(0.555 ln 0.555 + 0.445 ln 0.445) ≈ 0.6870 and scores 55.5%
  assert.equal(TRAIN.diagnose(curve(50, 0.68704, 300, 0), 0.555, data(111)), 'dead');
  assert.equal(TRAIN.diagnose(curve(50, Math.LN2, 300, 0), 0.5, data(100)), 'dead');
});

test('a network that learned something is not called dead', () => {
  assert.notEqual(TRAIN.diagnose(curve(50, 0.5, 300, 0), 0.555, data(111)), 'dead');
});

// the cases from the lab itself, trained for real
const train = (name, opts, H, steps) => {
  const net = TRAIN.init(H), D = TRAIN.makeData(name), h = [TRAIN.lossGrad(net, D, opts.act).loss];
  let acc = 0;
  for (let i = 0; i < steps; i++) {
    TRAIN.step(net, D, opts);
    const r = TRAIN.lossGrad(net, D, opts.act);
    h.push(r.loss); acc = r.acc;
  }
  return TRAIN.diagnose(h, acc, D);
};

test('lab: circle with Adam η = 0.3 stuck on a flat loss is not told it oscillates', () => {
  assert.notEqual(train('Vòng tròn', { act: 'relu', opt: 'adam', lr: 0.3 }, 8, 1000), 'oscillating');
});

test('lab: XOR with Adam η = 1 kills every ReLU and is told so', () => {
  assert.equal(train('XOR', { act: 'relu', opt: 'adam', lr: 1 }, 8, 800), 'dead');
});

// optimizer arithmetic, checked against the update rules themselves (guards rewrites of step)
const stepOnce = (opt, lr, net = TRAIN.init(5, 2), D = TRAIN.makeData('Hai cụm', 20)) => {
  const w0 = flat(net.W), g0 = flat(TRAIN.lossGrad(net, D, 'tanh').gW);
  TRAIN.step(net, D, { act: 'tanh', opt, lr });
  return { net, D, w0, g0, w1: flat(net.W) };
};
const close = (a, b, tol, what) => a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) <= tol, `${what}[${i}]: ${v} vs ${b[i]}`));

test('an SGD step moves every weight by −η × its gradient', () => {
  const { w0, g0, w1 } = stepOnce('sgd', 0.1);
  close(w1, w0.map((w, i) => w - 0.1 * g0[i]), 1e-12, 'w');
});

test("Adam's first step moves every weight by η against its gradient's sign", () => {
  // after one step m̂ = g and v̂ = g², so the update is η · g / (|g| + ε) with ε = 1e-8
  const { w0, g0, w1 } = stepOnce('adam', 0.01);
  close(w1, w0.map((w, i) => w - 0.01 * g0[i] / (Math.abs(g0[i]) + 1e-8)), 1e-12, 'w');
});

test('a momentum step adds 0.9 of the previous gradient to the new one', () => {
  const { net, D, w0, g0, w1 } = stepOnce('momentum', 0.05);
  close(w1, w0.map((w, i) => w - 0.05 * g0[i]), 1e-12, 'first step');
  const g1 = flat(TRAIN.lossGrad(net, D, 'tanh').gW);
  TRAIN.step(net, D, { act: 'tanh', opt: 'momentum', lr: 0.05 });
  close(flat(net.W), w1.map((w, i) => w - 0.05 * (0.9 * g0[i] + g1[i])), 1e-12, 'second step');
});

// the step rate of a run, whatever the display's refresh rate
const frames = (pace, hz, seconds, t0 = 0) => {
  let total = 0;
  for (let f = 0; f <= hz * seconds; f++) total += pace(t0 + f * 1000 / hz);
  return total;
};

test('a run takes the same number of steps per second at 60 Hz and at 144 Hz', () => {
  assert.equal(frames(TRAIN.pacer(240, 16), 60, 1), 240);
  assert.equal(frames(TRAIN.pacer(240, 16), 144, 1), 240);
});

test('after a stall the next frame owes no more than the cap', () => {
  const pace = TRAIN.pacer(240, 16);
  assert.equal(pace(0), 0);
  assert.equal(pace(5000), 16); // 5 s in a hidden tab would be 1 200 steps at once
  assert.equal(pace(5000 + 1000 / 60), 4);
});

test("the decision boundary's fast probability matches the network's forward pass", () => {
  for (const act of ['relu', 'tanh']) {
    const net = TRAIN.init(16, 3), p = TRAIN.prober(net, act);
    for (const [x, y] of [[0, 0], [0.7, -0.2], [-1.2, 1.1], [0.05, 0.9]]) {
      const one = { X: [[x, y]], Y: [1] }; // a one-point batch: its loss is −log p(class 1)
      assert.ok(Math.abs(p(x, y) - Math.exp(-TRAIN.lossGrad(net, one, act).loss)) < 1e-12, `${act} at ${x}, ${y}`);
    }
  }
});

test('the seeded random streams behind the data, the weights and the gradient experiments stay the same', () => {
  // values recorded before live.js shared one generator between TRAIN and GRAD
  const get = loadPage(), GRAD = get('GRAD');
  const res = GRAD.res({ L: 50, sigma: 0.9, beta: 0.2 }), rnn = GRAD.rnn({ T: 50, sigma: 1, f: 0.95 });
  // to 12 significant digits: Math.cos and friends differ in the last bit between platforms (macOS vs the Linux CI),
  // while any change to a seeded stream moves these numbers in the first digits
  const same = (got, want) => got.forEach((g, i) => assert.ok(Math.abs(g - want[i]) <= 1e-12 * Math.abs(want[i]), `[${i}] ${g} ≠ ${want[i]}`));
  same([res.plain[50], res.skip[50], rnn.plain[50], rnn.skip[50]], [0.014397718411632234, 2.942327557205068, 7.283670163079123e-9, 0.07694497527671315]);
  const net = TRAIN.init(8, 1), D = TRAIN.makeData('Vòng tròn');
  same([net.W[0][1][3], net.W[2][7][1], ...D.X[17]], [-0.2652991106673534, 0.725164236353931, 0.17346276161200289, 0.37538646178269686]);
  assert.equal(D.Y[17], 1);
});

// ---------- overfitting: a test set, label noise, weight decay, mini-batches ----------

test('label noise flips about that share of the labels and moves no point', () => {
  const clean = TRAIN.makeData('Vòng tròn'), noisy = TRAIN.makeData('Vòng tròn', 200, { flip: 0.2 });
  assert.deepEqual(flat(noisy.X), flat(clean.X));
  const flipped = clean.Y.filter((y, i) => y !== noisy.Y[i]).length;
  assert.ok(flipped >= 25 && flipped <= 55, `${flipped} of 200 flipped`); // 40 expected, ±2.6 standard deviations
});

test('more label noise only flips more labels', () => {
  const clean = TRAIN.makeData('XOR').Y, flips = p => new Set(TRAIN.makeData('XOR', 200, { flip: p }).Y.map((y, i) => y !== clean[i] ? i : -1).filter(i => i >= 0));
  const some = flips(0.1), more = flips(0.3);
  assert.ok(some.size > 0 && more.size > some.size);
  for (const i of some) assert.ok(more.has(i), `point ${i} flipped at 10% but not at 30%`);
});

test('a test set from another seed has different points', () => {
  assert.notDeepEqual(flat(TRAIN.makeData('Hai cụm', 200, { seed: 7 }).X), flat(TRAIN.makeData('Hai cụm').X));
});

test('evaluate gives the loss and accuracy that backprop measures, without the gradients', () => {
  for (const act of ['relu', 'tanh']) {
    const net = TRAIN.init(12, 4), D = TRAIN.makeData('Xoắn ốc'), e = TRAIN.evaluate(net, D, act), r = TRAIN.lossGrad(net, D, act);
    assert.ok(Math.abs(e.loss - r.loss) < 1e-12, act);
    assert.equal(e.acc, r.acc, act);
  }
});

test('weight decay pulls every weight towards 0 by η·λ·w on top of the step, and leaves biases alone', () => {
  for (const [opt, dir] of [['sgd', g => g], ['adam', g => g / (Math.abs(g) + 1e-8)]]) {
    const net = TRAIN.init(5, 2), D = TRAIN.makeData('Hai cụm', 20);
    net.b.forEach(v => v.fill(0.5)); // biases start at 0, where decaying them would change nothing
    const r = TRAIN.lossGrad(net, D, 'tanh');
    const w0 = flat(net.W), g = flat(r.gW), b0 = [...net.b.flat()], gb = [...r.gb.flat()];
    TRAIN.step(net, D, { act: 'tanh', opt, lr: 0.05, decay: 0.1 });
    close(flat(net.W), w0.map((w, i) => w - 0.05 * (dir(g[i]) + 0.1 * w)), 1e-12, `${opt} W`);
    close([...net.b.flat()], b0.map((b, i) => b - 0.05 * dir(gb[i])), 1e-12, `${opt} b`);
  }
});

// which b-point subsets of a small set an SGD step could have used: compare the step against each one
const subsetsOf = (n, b) => b === 0 ? [[]] : n < b ? [] : [...subsetsOf(n - 1, b), ...subsetsOf(n - 1, b - 1).map(s => [...s, n - 1])];
function usedPoints(net, D, b, lr) {
  const w0 = flat(net.W);
  const candidates = subsetsOf(D.X.length, b).map(s => ({ s, g: flat(TRAIN.lossGrad(net, { X: s.map(i => D.X[i]), Y: s.map(i => D.Y[i]) }, 'tanh').gW) }));
  TRAIN.step(net, D, { act: 'tanh', opt: 'sgd', lr, batch: b });
  const w1 = flat(net.W), hits = candidates.filter(({ g }) => w1.every((w, i) => Math.abs(w - (w0[i] - lr * g[i])) < 1e-12));
  assert.equal(hits.length, 1, `matches ${hits.length} subsets`);
  return hits[0].s;
}

test('a mini-batch step is a full step on b of the points', () => {
  const D = TRAIN.makeData('XOR', 6);
  assert.equal(usedPoints(TRAIN.init(4, 3), D, 2, 0.1).length, 2);
});

test('the mini-batches of one epoch use every point exactly once', () => {
  const net = TRAIN.init(4, 3), D = TRAIN.makeData('XOR', 6);
  const used = [0, 1, 2].flatMap(() => usedPoints(net, D, 2, 0.1));
  assert.deepEqual(used.sort(), [0, 1, 2, 3, 4, 5]);
});

test('a batch as large as the data is the full-batch step', () => {
  const D = TRAIN.makeData('Vòng tròn', 40), a = TRAIN.init(6, 5), b = TRAIN.init(6, 5);
  TRAIN.step(a, D, { ...ADAM, batch: 40 }); TRAIN.step(b, D, ADAM);
  assert.deepEqual(flat(a.W), flat(b.W));
});

// train loss falls from 1 to 0.1 over 600 steps; test loss falls to `low` at step 200, then moves linearly to `end`
const pair = (low, end) => {
  const train = Array.from({ length: 601 }, (_, i) => 1 - 0.9 * i / 600);
  const test = train.map((_, i) => i <= 200 ? 1 - (1 - low) * i / 200 : low + (end - low) * (i - 200) / 400);
  return { train, test };
};

test('test loss climbing back while train loss keeps falling is called overfitting', () => {
  const { train, test } = pair(0.3, 0.6);
  assert.equal(TRAIN.diagnose(train, 0.97, data(100), { h: test, acc: 0.85 }), 'overfitting');
});

test('test loss drifting up from an early low but staying below the train loss is not overfitting', () => {
  // noisy training labels keep the train loss near 0.3; the clean test set dipped to 0.18 early and settled at 0.26
  const train = Array.from({ length: 601 }, (_, i) => Math.max(0.3, 1 - i / 100));
  const test = train.map((_, i) => i <= 30 ? 1 - 0.82 * i / 30 : 0.26);
  assert.notEqual(TRAIN.diagnose(train, 0.88, data(100), { h: test, acc: 0.9 }), 'overfitting');
});

test('test loss that keeps falling with train loss is not overfitting', () => {
  const { train, test } = pair(0.3, 0.25);
  assert.notEqual(TRAIN.diagnose(train, 0.97, data(100), { h: test, acc: 0.92 }), 'overfitting');
});

// the lab's own case: the circle with 20% of its training labels flipped and 16 neurons
const overfit = decay => {
  const net = TRAIN.init(16), D = TRAIN.makeData('Vòng tròn', 200, { flip: 0.2 }), T = TRAIN.makeData('Vòng tròn', 200, { seed: 7 });
  const opts = { act: 'relu', opt: 'adam', lr: 0.03, decay }, h = [TRAIN.evaluate(net, D, 'relu').loss], t = [TRAIN.evaluate(net, T, 'relu').loss];
  let acc = 0, tacc = 0;
  for (let i = 0; i < 2000; i++) {
    ({ acc } = TRAIN.step(net, D, opts)); h.push(TRAIN.evaluate(net, D, 'relu').loss);
    const e = TRAIN.evaluate(net, T, 'relu'); t.push(e.loss); tacc = e.acc;
  }
  return TRAIN.diagnose(h, acc, D, { h: t, acc: tacc });
};

test('lab: 20% label noise and 16 neurons overfit, and the lab says so', () => {
  assert.equal(overfit(0), 'overfitting');
});

test('lab: the same run with weight decay 0.1 does not overfit', () => {
  assert.notEqual(overfit(0.1), 'overfitting');
});
