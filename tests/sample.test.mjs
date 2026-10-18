import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadPage } from './load.mjs';

const SAMPLE = loadPage()('SAMPLE');
const near = (got, want, tol, what) => want.forEach((w, i) => assert.ok(Math.abs(got[i] - w) < tol, `${what}[${i}]: ${got[i]} vs ${w}`));
const Z = [2, 1, 0.1]; // the logits of the worked example in models/llm.js

test('temperature reshapes the softmax: sharper below 1, flatter above', () => {
  // e^(z/T) / Σ, by hand: T = 1 → 7.389, 2.718, 1.105 over 11.212; T = 0.5 → 54.60, 7.389, 1.221; T = 2 → 2.718, 1.649, 1.051
  near(SAMPLE.nucleus(Z, 1, 1).probs, [0.659, 0.242, 0.099], 1e-3, 'T = 1');
  near(SAMPLE.nucleus(Z, 0.5, 1).probs, [0.864, 0.117, 0.019], 1e-3, 'T = 0.5');
  near(SAMPLE.nucleus(Z, 2, 1).probs, [0.502, 0.304, 0.194], 1e-3, 'T = 2');
});

test('top-p keeps the fewest tokens whose probability reaches p, then renormalises', () => {
  // 0.659 < 0.9, 0.659 + 0.242 = 0.901 ≥ 0.9 → two tokens, 0.659 / 0.901 and 0.242 / 0.901
  const r = SAMPLE.nucleus(Z, 1, 0.9);
  assert.deepEqual([...r.keep], [true, true, false]);
  near(r.final, [0.731, 0.269, 0], 1e-3, 'p = 0.9');
  assert.deepEqual([...SAMPLE.nucleus(Z, 1, 0.5).final], [1, 0, 0]); // the top token alone already reaches 0.5
  assert.deepEqual([...SAMPLE.nucleus(Z, 1, 1).keep], [true, true, true]);
});

test('top-p works on the probabilities in order, wherever the tokens sit in the list', () => {
  const r = SAMPLE.nucleus([0.1, 2, 1], 1, 0.9);
  assert.deepEqual([...r.keep], [false, true, true]);
  near(r.final, [0, 0.731, 0.269], 1e-3, 'shuffled');
});

test('a draw lands on the token whose slice of the distribution holds u, never on a cut one', () => {
  const final = [0.731, 0.269, 0];
  assert.deepEqual([0, 0.5, 0.7309, 0.7311, 0.9999].map(u => SAMPLE.draw(final, u)), [0, 0, 0, 1, 1]);
});

test("the toy distribution's nonsense words are cut at the default T = 0.7, p = 0.9 and come back at T = 2, p = 1", () => {
  const nonsense = ['ngon', 'chạy', 'đẹp trai'].map(w => SAMPLE.TOKENS.indexOf(w));
  assert.ok(nonsense.every(i => i >= 0));
  const strict = SAMPLE.nucleus(SAMPLE.LOGITS, 0.7, 0.9), loose = SAMPLE.nucleus(SAMPLE.LOGITS, 2, 1);
  for (const i of nonsense) {
    assert.equal(strict.final[i], 0, SAMPLE.TOKENS[i]);
    assert.ok(loose.final[i] > 0.01, `${SAMPLE.TOKENS[i]}: ${loose.final[i]}`);
  }
});
