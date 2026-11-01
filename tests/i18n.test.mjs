import { test } from 'node:test';
import assert from 'node:assert/strict';
import { load, texts, shared, VI } from '../scripts/i18n.mjs';

const { get, MODELS, textId, EN } = load();
const DONE = ['transformer', 'llm', 'mlp', 'cnn']; // tabs fully in English; the others show their Vietnamese text, marked as untranslated

test('the English tabs translate everything they show, and so do concept names and paper titles', () => {
  for (const k of DONE) assert.deepEqual(texts(MODELS[k]).filter(s => !EN[textId(s)]), [], k);
  assert.deepEqual(shared(get).filter(s => !EN[textId(s)]), [], 'concepts and papers');
});

test('no stale translation: every entry belongs to a text the site still shows', () => {
  // an entry whose Vietnamese source was edited no longer matches; translate the new text and drop the old entry
  const ids = new Set([...Object.values(MODELS).flatMap(texts), ...shared(get)].map(textId));
  assert.deepEqual(Object.keys(EN).filter(id => !ids.has(id)), []);
});

test('two different texts never share an id', () => {
  const all = [...new Set([...Object.values(MODELS).flatMap(texts), ...shared(get)])], byId = new Map();
  for (const s of all) { const id = textId(s); assert.ok(!byId.has(id) || byId.get(id) === s, `${id}: "${byId.get(id)}" and "${s}"`); byId.set(id, s); }
});

test('every entry is English: none still holds the Vietnamese it was prefilled with', () => {
  assert.deepEqual(Object.entries(EN).filter(([, en]) => VI.test(en)).map(([id]) => id), []);
});

test('no TeX command inside \\text{…} in the English either (MathJax would print it as typed)', () => {
  const inText = /\\text(?:tt|rm|bf|it)?\{[^{}]*\\[^{}]*\}/;
  assert.deepEqual(Object.entries(EN).filter(([, en]) => inText.test(en)).map(([id, en]) => `${id}: ${en.match(inText)[0]}`), []);
});
