import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadPage } from './load.mjs';

const get = loadPage(), MODELS = get('MODELS'), GROUPS = get('GROUPS'), PATHS = get('PATHS'), navCode = get('navCode');

test('every model sits in exactly one group of the tab bar', () => {
  const placed = GROUPS.flatMap(([, items]) => items.filter(s => !s.includes('/')));
  assert.deepEqual([...placed].sort(), Object.keys(MODELS).sort());
});

test('a spec names a model, then children by name or tag, and becomes the link of that block', () => {
  assert.equal(navCode('transformer'), 'transformer');
  assert.equal(navCode('transformer/Encoder/Multi-Head Self-Attention/Scaled Dot-Product Attention'), 'transformer.2.0.2');
  assert.equal(navCode('densenet/CheXNet'), `densenet.${MODELS.densenet.children.length - 1}`); // by tag
  assert.equal(navCode('transformer/Nope'), null);
  assert.equal(navCode('nope'), null);
});

test('every step of every learning path opens a real block; steps still to come have no link', () => {
  for (const p of PATHS) {
    assert.ok(p.title && p.goal && p.steps.length >= 4, p.title);
    for (const [label, spec] of p.steps) if (spec) assert.ok(navCode(spec), `${p.title}: ${label} → ${spec}`);
  }
  const med = PATHS.find(p => p.title === 'Phân vùng ảnh y khoa');
  assert.deepEqual([...med.steps].map(s => s[0]).slice(-3), ['nnU-Net', 'Swin UNETR', 'MedSAM']); // to come, as the roadmap says
});

test('the medical group lists U-Net and CheXNet, the shortcut into DenseNet-121', () => {
  assert.deepEqual([...GROUPS.find(g => g[0] === 'Y tế')[1]], ['unet', 'densenet/CheXNet']);
});
