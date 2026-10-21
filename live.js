// Seeded random numbers (mulberry32) and normal samples (Box–Muller), for everything below that needs noise.
const rng = s => () => { s = s + 0x6D2B79F5 | 0; let t = Math.imul(s ^ s >>> 15, 1 | s); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
const gauss = r => () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());

// Toy self-attention that really runs in the page. Weights are hand-picked so the numbers can be read:
// x has 5 named features, queries/keys live in 3 named dims, W_V = I (outputs stay in feature space), no PE, no W_O.
const TOY = (() => {
  const F = ['danh từ', 'động từ', 'tính từ', 'đại từ', 'con vật'];
  const KD = ['danh từ', 'con vật', 'động từ'];
  const VOCAB = {
    'mèo': [1, 0, 0, 0, 1], 'chuột': [1, 0, 0, 0, 1], 'cơm': [1, 0, 0, 0, 0],
    'đen': [0, 0, 1, 0, 0], 'đói': [0, 0, 1, 0, 0], 'đuổi': [0, 1, 0, 0, 0], 'ăn': [0, 1, 0, 0, 0], 'nó': [0, 0, 0, 1, 0]
  };
  // rows = features of x, cols = KD. Key: "what I am". Query: "what I look for".
  const WK = [[2, 0, 0], [0, 0, 2], [0, 0, 0], [0, 0, 0], [0, 2, 0]];
  const WQ = [[0, 0, 0.5], [1, 0.5, 0], [1, 0, 0], [0.5, 1.5, 0], [0, 0, 0]];
  const PRESETS = [['mèo', 'đen', 'đuổi', 'chuột'], ['mèo', 'ăn', 'cơm', 'nó', 'đói']];
  const mm = (A, B) => A.map(r => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));

  function run(words, { causal, scale }) {
    const X = words.map(w => VOCAB[w]), Q = mm(X, WQ), K = mm(X, WK), V = X;
    const S = Q.map(q => K.map(k => q.reduce((s, v, c) => s + v * k[c], 0)));
    const St = S.map(r => r.map(v => scale ? v / Math.sqrt(KD.length) : v));
    const Sm = St.map((r, i) => r.map((v, j) => causal && j > i ? -Infinity : v));
    const A = Sm.map(r => { const m = Math.max(...r), e = r.map(v => Math.exp(v - m)), z = e.reduce((a, b) => a + b); return e.map(v => v / z); });
    return { words, X, Q, K, V, S, St, Sm, A, O: mm(A, V) };
  }

  // self-check against values computed independently in Python
  const near = (a, b) => Math.abs(a - b) < 1e-3;
  const r1 = run(PRESETS[0], { causal: false, scale: true }), r2 = run(PRESETS[0], { causal: true, scale: true }), r3 = run(PRESETS[1], { causal: true, scale: true });
  console.assert(near(r1.A[2][0], 0.425) && near(r2.A[2][0], 0.739) && near(r3.A[3][0], 0.727) && near(r3.O[3][4], 0.727), 'toy attention mismatch');
  return { F, KD, VOCAB, WQ, WK, PRESETS, run };
})();

// Toy shifted windows (Swin) on an 8×8 patch grid: 4×4 windows, shift 2 (Swin-T stage 1: 56×56, M = 7, shift 3).
// Two patches may attend to each other when they share a region: the window (regular layer), or the shifted window
// in original coordinates (shifted layer). The cyclic roll packs several such regions into one batched window; the
// mask keeps them apart, so "shifted window" and "same rolled window + same mask label" are the same rule.
const SWIN = (() => {
  const G = 8, M = 4, S = M / 2;
  const band = (x, shifted) => shifted ? (x < S ? 0 : 1 + Math.floor((x - S) / M)) : Math.floor(x / M);
  const nb = shifted => shifted ? 1 + Math.ceil((G - S) / M) : G / M;
  const region = (i, j, shifted) => band(i, shifted) * nb(shifted) + band(j, shifted);
  const cells = Array.from({ length: G * G }, (_, k) => [Math.floor(k / G), k % G]);
  const sees = ([i, j], shifted) => cells.filter(([a, b]) => region(a, b, shifted) === region(i, j, shifted));
  // after a regular block then a shifted block, p has read the regular window of everything it saw in the second
  const reach = p => new Set(sees(p, true).flatMap(q => sees(q, false)).map(([a, b]) => a * G + b));
  // torch.roll(x, (-S, -S)): rolled cell (a, b) holds original patch src(a, b); win(p) = top-left of p's batched window
  const src = (a, b) => [(a + S) % G, (b + S) % G];
  const win = ([i, j]) => [Math.floor(((i - S + G) % G) / M) * M, Math.floor(((j - S + G) % G) / M) * M];
  // M²×M² mask of p's batched window: 0 = same region, −∞ otherwise (the real code adds −100)
  function mask(p) {
    const [a0, b0] = win(p), ids = [];
    for (let a = a0; a < a0 + M; a++) for (let b = b0; b < b0 + M; b++) ids.push(region(...src(a, b), true));
    return ids.map(x => ids.map(y => x === y ? 0 : -Infinity));
  }
  const zeros = p => mask(p).flat().filter(v => v === 0).length;
  // self-check: a bottom-edge patch sees 16 → 8 patches, reaches 32 after two blocks, and its rolled window mixes
  // two regions (128 of 256 pairs allowed); a centre patch reaches the whole grid and needs no mask
  console.assert(sees([7, 3], false).length === 16 && sees([7, 3], true).length === 8 && reach([7, 3]).size === 32 && zeros([7, 3]) === 128
    && sees([0, 0], true).length === 4 && zeros([0, 0]) === 64 && reach([3, 3]).size === 64 && zeros([3, 3]) === 256, 'toy shifted windows mismatch');
  return { G, M, S, region, sees, reach, src, win, mask };
})();

// Toy convolution on an 8×8 image (cross-correlation, like nn.Conv2d).
const TOYCONV = (() => {
  const N = 8, img = f => Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => f(i, j) ? 1 : 0));
  const IMAGES = {
    'Cạnh dọc': img((i, j) => j >= 4),
    'Ô vuông': img((i, j) => (i === 2 || i === 5) && j >= 2 && j <= 5 || (j === 2 || j === 5) && i >= 2 && i <= 5),
    'Chữ thập': img((i, j) => i === 3 || j === 3),
    'Đường chéo': img((i, j) => i === j)
  };
  const b = 1 / 9;
  const KERNELS = {
    'Sobel dọc': [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    'Sobel ngang': [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    'Làm mờ': [[b, b, b], [b, b, b], [b, b, b]],
    'Làm nét': [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    'Đồng nhất': [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
  };
  const outSize = (pad, stride) => Math.floor((N + 2 * pad - 3) / stride) + 1;
  // terms[i][j] = the 9 (x, w) pairs behind output (i, j); x = 0 where the window hangs over the padding
  function run(x, k, { pad, stride }) {
    const n = outSize(pad, stride), Y = [], terms = [];
    for (let i = 0; i < n; i++) {
      Y.push([]); terms.push([]);
      for (let j = 0; j < n; j++) {
        const t = [];
        for (let u = 0; u < 3; u++) for (let v = 0; v < 3; v++) {
          const r = stride * i + u - pad, c = stride * j + v - pad;
          t.push({ r, c, x: r >= 0 && r < N && c >= 0 && c < N ? x[r][c] : 0, w: k[u][v] });
        }
        Y[i].push(t.reduce((s, q) => s + q.x * q.w, 0)); terms[i].push(t);
      }
    }
    return { Y, terms, n };
  }
  const r = run(IMAGES['Cạnh dọc'], KERNELS['Sobel dọc'], { pad: 0, stride: 1 });
  console.assert(r.n === 6 && r.Y[0].join() === '0,0,4,4,0,0' && outSize(1, 2) === 4 && outSize(1, 1) === 8, 'toy conv mismatch');
  return { N, IMAGES, KERNELS, outSize, run };
})();

// Tiny MLP trained in the page: 2 → H → H → 2, softmax + cross-entropy, full batch or mini-batches, hand-written backprop.
const TRAIN = (() => {
  // each generator returns [x, y, label] in roughly [-1, 1]²
  const DATASETS = {
    'Hai cụm': (r, g, i) => { const c = i % 2; return [(c ? 0.45 : -0.45) + 0.28 * g(), (c ? 0.45 : -0.45) + 0.28 * g(), c]; },
    'Vòng tròn': (r, g, i) => { const c = i % 2, rad = c ? 0.45 * Math.sqrt(r()) : 0.7 + 0.25 * r(), a = 2 * Math.PI * r(); return [rad * Math.cos(a) + 0.04 * g(), rad * Math.sin(a) + 0.04 * g(), c]; },
    'XOR': (r) => { const x = 2 * r() - 1, y = 2 * r() - 1; return [x + (x > 0 ? 0.06 : -0.06), y + (y > 0 ? 0.06 : -0.06), x * y > 0 ? 1 : 0]; },
    // ~1.9 turns: 8 hidden units stall near 95%, 16 reach 100%
    'Xoắn ốc': (r, g, i) => { const c = i % 2, t = 0.25 + 2.6 * r(), a = 4.5 * t + c * Math.PI; return [0.36 * t * Math.cos(a) + 0.03 * g(), 0.36 * t * Math.sin(a) + 0.03 * g(), c]; }
  };
  // n points drawn from `seed`; with flip > 0 each label is swapped with that probability (label noise) and the points stay put
  function makeData(name, n = 200, { seed = 42, flip = 0 } = {}) {
    const r = rng(seed), g = gauss(r), coin = rng(seed + 1), X = [], Y = [];
    for (let i = 0; i < n; i++) { const [x, y, c] = DATASETS[name](r, g, i); X.push([x, y]); Y.push(coin() < flip ? 1 - c : c); }
    return { X, Y };
  }
  const mat = (a, b, f) => Array.from({ length: a }, () => Array.from({ length: b }, f));
  // He initialisation: weights ~ N(0, 2/fan_in), biases 0
  function init(H, seed = 1) {
    const g = gauss(rng(seed)), w = (a, b) => mat(a, b, () => g() * Math.sqrt(2 / a));
    const net = { H, seed, W: [w(2, H), w(H, H), w(H, 2)], b: [new Array(H).fill(0), new Array(H).fill(0), [0, 0]], t: 0 };
    resetOpt(net);
    return net;
  }
  function resetOpt(net) { // optimizer memory (momentum / Adam moments) has the same shapes as the parameters
    const z = () => ({ W: net.W.map(M => M.map(r => r.map(() => 0))), b: net.b.map(v => v.map(() => 0)) });
    net.m = z(); net.v = z(); net.t = 0;
  }
  const ACT = { relu: [z => z > 0 ? z : 0, z => z > 0 ? 1 : 0], tanh: [Math.tanh, z => 1 - Math.tanh(z) ** 2] };
  const affine = (a, W, b) => b.map((bj, j) => a.reduce((s, ai, i) => s + ai * W[i][j], bj));
  function forward(net, x, act) {
    const f = ACT[act][0], z1 = affine(x, net.W[0], net.b[0]), a1 = z1.map(f), z2 = affine(a1, net.W[1], net.b[1]), a2 = z2.map(f), z3 = affine(a2, net.W[2], net.b[2]);
    const m = Math.max(...z3), e = z3.map(v => Math.exp(v - m)), s = e[0] + e[1];
    return { x, z1, a1, z2, a2, p: [e[0] / s, e[1] / s] };
  }
  // loss, accuracy and gradients over the whole batch (backprop by the chain rule, layer by layer)
  function lossGrad(net, D, act) {
    const df = ACT[act][1], n = D.X.length, gW = net.W.map(M => M.map(r => r.map(() => 0))), gb = net.b.map(v => v.map(() => 0));
    let loss = 0, hit = 0;
    const back = (k, a, d) => { for (let i = 0; i < a.length; i++) for (let j = 0; j < d.length; j++) gW[k][i][j] += a[i] * d[j] / n; d.forEach((v, j) => gb[k][j] += v / n); };
    const thru = (k, d, z) => net.W[k].map((row, i) => row.reduce((s, w, j) => s + w * d[j], 0) * df(z[i])); // δ_prev = (δ Wᵀ) ⊙ f'(z)
    D.X.forEach((x, i) => {
      const c = forward(net, x, act), y = D.Y[i];
      loss -= Math.log(Math.max(c.p[y], 1e-12)) / n;
      if ((c.p[1] > 0.5 ? 1 : 0) === y) hit++;
      const d3 = [c.p[0] - (y === 0), c.p[1] - (y === 1)]; // softmax + cross-entropy: gradient = p − onehot(y)
      back(2, c.a2, d3);
      const d2 = thru(2, d3, c.z2); back(1, c.a1, d2);
      const d1 = thru(1, d2, c.z1); back(0, c.x, d1);
    });
    return { loss, acc: hit / n, gW, gb };
  }
  // mini-batches walk through a shuffled order of the points, reshuffled each epoch (a short last batch is skipped)
  function nextBatch(net, D, size) {
    const n = D.X.length, s = net.batches ??= { order: [], at: 0, coin: rng(net.seed + 99) };
    if (s.order.length !== n || s.at + size > n) { // a new epoch: Fisher–Yates shuffle
      s.order = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) { const j = Math.floor(s.coin() * (i + 1)); [s.order[i], s.order[j]] = [s.order[j], s.order[i]]; }
      s.at = 0;
    }
    const idx = s.order.slice(s.at, s.at += size);
    return { X: idx.map(i => D.X[i]), Y: idx.map(i => D.Y[i]) };
  }
  // one update from the gradient of a mini-batch (`batch` points; all of them when 0), with weight decay pulling every
  // weight (not the biases) towards 0 by η·λ·w after the optimizer's step (decoupled, as in AdamW);
  // returns the loss and accuracy of the new weights on the whole set, the ones the boundary is drawn from
  function step(net, D, { act, opt, lr, decay = 0, batch = 0 }) {
    const r = lossGrad(net, batch && batch < D.X.length ? nextBatch(net, D, batch) : D, act), b1 = 0.9, b2 = 0.999;
    net.t++;
    const upd = (P, G, M, V, i) => { // M[i], V[i]: the optimizer's memory for this parameter
      if (opt === 'sgd') return P - lr * G;
      if (opt === 'momentum') { M[i] = 0.9 * M[i] + G; return P - lr * M[i]; }
      M[i] = b1 * M[i] + (1 - b1) * G; V[i] = b2 * V[i] + (1 - b2) * G * G; // Adam
      return P - lr * (M[i] / (1 - b1 ** net.t)) / (Math.sqrt(V[i] / (1 - b2 ** net.t)) + 1e-8);
    };
    net.W.forEach((W, k) => W.forEach((row, i) => row.forEach((w, j) => { row[j] = upd(w, r.gW[k][i][j], net.m.W[k][i], net.v.W[k][i], j) - lr * decay * w; })));
    net.b.forEach((b, k) => b.forEach((w, j) => { b[j] = upd(w, r.gb[k][j], net.m.b[k], net.v.b[k], j); }));
    return evaluate(net, D, act);
  }
  // the class-1 margin z₁ − z₀ at one point, without the arrays forward() builds per point: the boundary picture
  // needs ~3 300 of them per redraw, and the lab measures the test set after every step
  function margin(net, act) {
    const f = ACT[act][0], H = net.H, [W1, W2, W3] = net.W, [b1, b2, b3] = net.b, a1 = new Float64Array(H), a2 = new Float64Array(H);
    return (x, y) => {
      for (let j = 0; j < H; j++) a1[j] = f(x * W1[0][j] + y * W1[1][j] + b1[j]);
      for (let j = 0; j < H; j++) { let s = b2[j]; for (let i = 0; i < H; i++) s += a1[i] * W2[i][j]; a2[j] = f(s); }
      let z0 = b3[0], z1 = b3[1];
      for (let i = 0; i < H; i++) { z0 += a2[i] * W3[i][0]; z1 += a2[i] * W3[i][1]; }
      return z1 - z0;
    };
  }
  const prober = (net, act) => { const m = margin(net, act); return (x, y) => 1 / (1 + Math.exp(-m(x, y))); }; // softmax over two logits = sigmoid of their difference
  // loss and accuracy over a whole set, forward only: the same numbers lossGrad gives
  function evaluate(net, D, act) {
    const m = margin(net, act), n = D.X.length;
    let loss = 0, hit = 0;
    D.X.forEach(([x, y], i) => {
      const d = m(x, y), py = 1 / (1 + Math.exp(D.Y[i] ? -d : d)); // p of the true label
      loss -= Math.log(Math.max(py, 1e-12)) / n;
      if ((d > 0 ? 1 : 0) === D.Y[i]) hit++;
    });
    return { loss, acc: hit / n };
  }
  // steps each animation frame owes so a run goes at `rate` steps per second whatever the display's refresh rate;
  // counted from the run's start (no drift), and after a stall (hidden tab, slow frame) a frame owes at most `most`
  function pacer(rate, most) {
    let t0 = null, given = 0;
    return now => {
      if (t0 === null) { t0 = now; return 0; }
      let k = Math.floor(rate * (now - t0) / 1000 + 1e-6) - given;
      if (k > most) { given += k - most; k = most; } // drop what a stall would owe instead of catching up in a burst
      given += k;
      return k;
    };
  }
  // reads the loss curve h (h[k] = loss after k steps) the way a person would; index.html turns the verdict into advice
  // test = { h, acc } of the held-out set, when there is one
  function diagnose(h, acc, D, test) {
    const n = h.length - 1, L0 = h[0], last = h.at(-1);
    // a dead net predicts the class shares for every point: its loss is their entropy (ln 2 when balanced), its accuracy the larger share
    const p = D.Y.reduce((s, y) => s + y, 0) / D.Y.length, xlx = q => q > 0 ? q * Math.log(q) : 0, flat = -xlx(p) - xlx(1 - p);
    if (!Number.isFinite(last) || last > 10) return 'diverged';
    if (n === 0) return 'idle';
    if (n >= 30 && Math.abs(last - flat) < 0.01 && acc <= Math.max(p, 1 - p)) return 'dead';
    if (test && overfits(h, test.h)) return 'overfitting';
    if (acc >= 0.99) return 'solved';
    const recent = h.slice(-40), ups = recent.filter((v, i) => i && v > recent[i - 1] * (1 + 1e-3)).length; // rises under 0.1% are rounding noise
    if (n >= 40 && ups > 12) return 'oscillating';
    if (n >= 300 && last > 0.9 * L0) return 'slow';
    if (n >= 400 && acc < 0.97 && h.at(-100) - last < 0.01) return 'plateau';
    return 'learning';
  }

  // the test loss has climbed well above its lowest point and well above the train loss, which kept falling since then
  // (with noisy training labels the train loss stays high, so a test loss drifting up from an early low is no sign)
  function overfits(h, t) {
    if (h.length < 101) return false;
    const k = t.indexOf(Math.min(...t)), last = t.at(-1);
    return k >= 0 && last > 1.2 * t[k] + 0.02 && last - h.at(-1) > 0.15 && h.at(-1) < h[k];
  }

  // self-check: backprop gradients must match finite differences (tanh: smooth everywhere)
  const D = makeData('XOR', 6), net = init(3, 5), { gW } = lossGrad(net, D, 'tanh');
  let worst = 0;
  for (const [k, i, j] of [[0, 0, 1], [1, 2, 0], [2, 1, 1], [0, 1, 2]]) {
    const w0 = net.W[k][i][j], e = 1e-5;
    net.W[k][i][j] = w0 + e; const lp = lossGrad(net, D, 'tanh').loss;
    net.W[k][i][j] = w0 - e; const lm = lossGrad(net, D, 'tanh').loss;
    net.W[k][i][j] = w0;
    worst = Math.max(worst, Math.abs((lp - lm) / (2 * e) - gW[k][i][j]) / Math.max(1e-8, Math.abs(gW[k][i][j])));
  }
  console.assert(worst < 1e-4, 'backprop gradient check failed', worst);
  return { DATASETS: Object.keys(DATASETS), makeData, init, resetOpt, step, lossGrad, evaluate, prober, pacer, diagnose, gradCheck: worst };
})();

// Gradient flowing backwards through many layers (or time steps), with fixed random 32×32 Jacobians.
const GRAD = (() => {
  const d = 32, MAX = 100;
  const rand = rng(7), normal = gauss(rand);
  const mat = () => Array.from({ length: d }, () => Array.from({ length: d }, () => normal() / Math.sqrt(d))); // E‖Wᵀg‖ ≈ ‖g‖
  const Ws = Array.from({ length: MAX }, mat), Wr = mat();
  const tanhD = Array.from({ length: MAX }, () => Array.from({ length: d }, () => 0.3 + 0.7 * rand())); // tanh' of each unit at each step
  const g0 = (v => { const z = Math.hypot(...v); return v.map(x => x / z); })(Array.from({ length: d }, normal));
  const mtv = (W, g) => W[0].map((_, j) => W.reduce((s, row, i) => s + row[j] * g[i], 0)); // Wᵀg
  const norm = g => Math.hypot(...g);
  // returns norms[k] for k = 0..L, k = layers (steps) between the loss and that point
  function walk(L, step) { const out = [1]; let g = g0; for (let k = 0; k < L; k++) { g = step(g, k); out.push(norm(g)); } return out; }
  const res = ({ L, sigma, beta }) => ({
    plain: walk(L, (g, k) => mtv(Ws[k], g).map(v => sigma * v)),
    skip: walk(L, (g, k) => { const w = mtv(Ws[k], g); return g.map((v, i) => v + beta * w[i]); })
  });
  const rnn = ({ T, sigma, f }) => ({
    plain: walk(T, (g, k) => mtv(Wr, g.map((v, i) => v * tanhD[k][i])).map(v => sigma * v)),
    skip: walk(T, g => g.map(v => f * v))
  });
  const a = res({ L: 20, sigma: 0.5, beta: 0 }), b = rnn({ T: 10, sigma: 1, f: 0.9 });
  console.assert(a.skip.every(v => Math.abs(v - 1) < 1e-9) && a.plain[20] < 1e-4 && Math.abs(b.skip[10] - 0.9 ** 10) < 1e-9, 'gradient experiment mismatch');
  return { MAX, res, rnn };
})();

// Next-token sampling on a toy distribution: hand-picked logits for the word after "Hôm nay trời rất", so the effect of
// temperature and top-p can be read. One token per word here; a real tokenizer often splits Vietnamese words further.
const SAMPLE = (() => {
  const PROMPT = 'Hôm nay trời rất';
  const TOKENS = ['đẹp', 'nóng', 'lạnh', 'nắng', 'oi', 'mát', 'âm u', 'trong', 'xanh', 'đẹp trai', 'ngon', 'chạy'];
  const LOGITS = [3.2, 2.9, 2.5, 2.2, 1.7, 1.4, 1.0, 0.6, 0.3, -0.6, -1.4, -2.0];
  // softmax(z / T); top-p keeps tokens, most likely first, while the probability before them is still < p
  // (the one that reaches p stays, as in the PyTorch code of the block), then renormalises what is kept
  function nucleus(z, T, p) {
    const m = Math.max(...z), e = z.map(v => Math.exp((v - m) / T)), s = e.reduce((a, b) => a + b, 0), probs = e.map(v => v / s);
    const keep = probs.map(() => false);
    let before = 0;
    for (const i of probs.map((_, i) => i).sort((a, b) => probs[b] - probs[a])) { if (before >= p) break; keep[i] = true; before += probs[i]; }
    const mass = probs.reduce((a, v, i) => a + (keep[i] ? v : 0), 0);
    return { probs, keep, mass, final: probs.map((v, i) => keep[i] ? v / mass : 0) };
  }
  // u uniform in [0, 1): the token whose slice of `final` holds u (the last kept one if rounding leaves a sliver at the end)
  function draw(final, u) {
    let c = 0;
    for (let i = 0; i < final.length; i++) { c += final[i]; if (final[i] > 0 && u < c) return i; }
    return final.findLastIndex(v => v > 0);
  }
  // self-check against the worked example of the "Top-p" block: logits [2, 1, 0.1], T = 1, p = 0.9 → [0.731, 0.269, 0]
  const ex = nucleus([2, 1, 0.1], 1, 0.9).final;
  console.assert(Math.abs(ex[0] - 0.731) < 1e-3 && Math.abs(ex[1] - 0.269) < 1e-3 && ex[2] === 0, 'top-p example mismatch');
  return { PROMPT, TOKENS, LOGITS, nucleus, draw };
})();

// Toy diffusion in 2D with Stable Diffusion's noise schedule. Each "image" is a point; the data are four Gaussian clusters,
// two per prompt. The noise prediction is exact (the posterior mean of the mixture, i.e. what a perfectly trained network
// would output), so the demo shows the sampler and the guidance, not a learned approximation.
const DIFF = (() => {
  const T = 1000, b0 = Math.sqrt(0.00085), b1 = Math.sqrt(0.012), AB = [];
  for (let i = 0, p = 1; i < T; i++) { p *= 1 - (b0 + (b1 - b0) * i / (T - 1)) ** 2; AB.push(p); }
  const S = 0.18, PROMPTS = { 'mèo': [[-1, 0.6], [-1, -0.6]], 'chó': [[1, 0.6], [1, -0.6]] };
  const ALL = Object.values(PROMPTS).flat();
  // E[x₀ | x_t] for an equal-weight mixture of N(μ, σ²I): responsibilities from x_t ~ N(√ᾱ μ, vI), then each component's mean
  function x0hat(x, ab, mus, s = S) {
    const sa = Math.sqrt(ab), v = ab * s * s + 1 - ab, k = sa * s * s / v;
    const lw = mus.map(m => -((x[0] - sa * m[0]) ** 2 + (x[1] - sa * m[1]) ** 2) / (2 * v)), top = Math.max(...lw);
    const w = lw.map(l => Math.exp(l - top)), z = w.reduce((a, b) => a + b);
    return [0, 1].map(d => mus.reduce((t, m, i) => t + w[i] / z * (m[d] + k * (x[d] - sa * m[d])), 0));
  }
  const epsOf = (x, x0, ab) => [0, 1].map(d => (x[d] - Math.sqrt(ab) * x0[d]) / Math.sqrt(1 - ab));
  // classifier-free guidance: ε_u + s (ε_c − ε_u); prompt null = the empty prompt (every cluster)
  function eps(x, ab, prompt, s) {
    const u = epsOf(x, x0hat(x, ab, ALL), ab);
    if (!prompt) return u;
    const c = epsOf(x, x0hat(x, ab, PROMPTS[prompt]), ab);
    return [0, 1].map(d => u[d] + s * (c[d] - u[d]));
  }
  // DDIM (η = 0): guess the clean point, then re-noise it to the earlier, less noisy time
  const ddim = (x, e, ab, abPrev) => [0, 1].map(d => Math.sqrt(abPrev) * (x[d] - Math.sqrt(1 - ab) * e[d]) / Math.sqrt(ab) + Math.sqrt(1 - abPrev) * e[d]);
  const timesteps = N => Array.from({ length: N }, (_, i) => (N - 1 - i) * Math.floor(T / N) + 1); // 50 → 981, 961, …, 1
  // frames[k] = the points after k steps; frames[0] is the seeded starting noise. The last step goes to ᾱ₀, as diffusers does.
  function sample({ n = 300, N = 50, prompt = 'mèo', s = 7.5, seed = 3 }) {
    const g = gauss(rng(seed)), ts = timesteps(N);
    let pts = Array.from({ length: n }, () => [g(), g()]);
    const frames = [pts];
    ts.forEach((t, i) => {
      const ab = AB[t], abPrev = i + 1 < N ? AB[ts[i + 1]] : AB[0];
      pts = pts.map(x => ddim(x, eps(x, ab, prompt, s), ab, abPrev));
      frames.push(pts);
    });
    return { frames, ts };
  }
  // the closest cluster centre and which prompt it belongs to
  const nearest = x => Object.entries(PROMPTS).flatMap(([p, ms]) => ms.map(m => ({ prompt: p, d: Math.hypot(x[0] - m[0], x[1] - m[1]) }))).reduce((a, b) => b.d < a.d ? b : a);
  console.assert(Math.abs(AB[999] - 0.00466) < 1e-5 && timesteps(50)[0] === 981, 'toy diffusion schedule mismatch');
  return { T, AB, S, PROMPTS, x0hat, epsOf, eps, ddim, timesteps, sample, nearest };
})();
