import { test } from 'node:test';
import assert from 'node:assert/strict';
import { run, pageScripts, loadPage } from './load.mjs';

test('the local scripts of index.html are found: shared code first, then the models', () => {
  const scripts = pageScripts();
  assert.deepEqual(scripts.slice(0, 2), ['blocks.js', 'live.js']);
  assert.ok(scripts.slice(2).length > 0 && scripts.slice(2).every(s => s.startsWith('models/')), scripts.join(', '));
});

test('a failing console.assert inside a script fails the load', () => {
  assert.throws(() => run([{ name: 'bad.js', code: "console.assert(1 + 1 === 3, 'arithmetic is broken')" }]), /arithmetic is broken/);
});

test("every script's own self-checks pass (toy attention, conv, backprop, parameter counts)", () => {
  const get = loadPage(); // throws on the first failing console.assert
  const models = pageScripts().filter(s => s.startsWith('models/'));
  assert.equal(Object.keys(get('MODELS')).length, models.length, 'each model file registers one model');
});

test('the MLP block states the real parameter count of the lab network', () => {
  const get = loadPage(), text = get('MODELS.mlp.info.params');
  const stated = Object.fromEntries([...text.matchAll(/\\\(H = (\d+)\\\) là (\d+)/g)].map(m => [m[1], +m[2]]));
  // 2→H, H→H, H→2 with biases: H = 8 → 16+8 + 64+8 + 16+2 = 114; H = 16 → 32+16 + 256+16 + 32+2 = 354
  assert.deepEqual(stated, { 8: 114, 16: 354 });
  for (const H of [8, 16]) {
    const net = get('TRAIN').init(H);
    assert.equal(net.W.flat(2).length + net.b.flat().length, stated[H], `H = ${H}`);
  }
});
