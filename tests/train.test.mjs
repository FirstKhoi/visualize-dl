import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadPage } from './load.mjs';

const TRAIN = loadPage()('TRAIN');
const ADAM = { act: 'relu', opt: 'adam', lr: 0.03 };
// the trainable state only, so a copy starts with no memory of earlier steps
const copy = net => ({ H: net.H, t: net.t, ...structuredClone({ W: net.W, b: net.b, m: net.m, v: net.v }) });
const flat = W => [...W.flat(2)]; // a plain array of this realm: the scripts' arrays have the vm context's prototypes

test('a step reports the loss and accuracy of the weights it leaves behind', () => {
  const net = TRAIN.init(8), D = TRAIN.makeData('Vòng tròn'), before = TRAIN.lossGrad(net, D, 'relu').loss;
  const r = TRAIN.step(net, D, ADAM), now = TRAIN.lossGrad(net, D, 'relu');
  assert.notEqual(r.loss, before);
  assert.equal(r.loss, now.loss);
  assert.equal(r.acc, now.acc);
});

test('consecutive steps move the weights exactly as steps taken from scratch', () => {
  const D = TRAIN.makeData('XOR'), other = TRAIN.makeData('Xoắn ốc'), a = TRAIN.init(6, 3);
  TRAIN.step(a, D, ADAM);
  const tanh = { ...ADAM, act: 'tanh' }; // each case changes one thing from the one before: data, activation, or optimizer
  for (const [data, opts] of [[D, ADAM], [other, ADAM], [other, tanh], [D, tanh], [D, { ...tanh, opt: 'sgd' }]]) {
    const b = copy(a);
    TRAIN.step(a, data, opts); TRAIN.step(b, data, opts);
    assert.deepEqual(flat(a.W), flat(b.W), `${opts.act}/${opts.opt}`);
  }
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
  assert.deepEqual([res.plain[50], res.skip[50], rnn.plain[50], rnn.skip[50]], [0.014397718411632234, 2.942327557205068, 7.283670163079123e-9, 0.07694497527671315]);
  const net = TRAIN.init(8, 1), D = TRAIN.makeData('Vòng tròn');
  assert.deepEqual([net.W[0][1][3], net.W[2][7][1], ...D.X[17], D.Y[17]], [-0.2652991106673534, 0.725164236353931, 0.17346276161200289, 0.37538646178269686, 1]);
});
