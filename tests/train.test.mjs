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
