import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadPage } from './load.mjs';

const kids = loadPage()('MODELS.densenet.children'), chex = kids.at(-1);

test('CheXNet is a floating block added after DenseNet-121\'s own, so old #densenet.i links keep their blocks', () => {
  assert.equal(chex.tag, 'CheXNet');
  assert.equal(chex.float, true);
  assert.equal(kids.at(-2).name, 'Đầu phân loại');
  assert.deepEqual([...chex.children].map(k => k.name), ['Ảnh X-quang ngực', 'Đặc trưng DenseNet-121', 'Linear 1024 → 14', 'Sigmoid từng nhãn', 'BCE có trọng số']);
});

test('its numbers, computed by hand', () => {
  // torchvision densenet121: 7 978 856 parameters; the 1000-class head 1024 · 1000 + 1000 = 1 025 000; 14 labels: 1024 · 14 + 14 = 14 350
  assert.match(chex.info.params, /6\\,968\\,206/);
  assert.match(chex.children[1].info.params, /6\\,953\\,856/);
  assert.match(chex.children[2].info.params, /14\\,350/);
  // 13 positives in 1000 images: w+ = 987 / 1000, w− = 13 / 1000; −0.987 · ln 0.2 = 1.589, −0.013 · ln 0.8 = 0.0029
  const ex = chex.children[4].info.example;
  for (const s of ['0.987', '0.013', '1.589', '0.0029', '12.83']) assert.ok(ex.includes(s), s);
});
