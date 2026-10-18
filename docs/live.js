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

// Tiny MLP trained in the page: 2 → H → H → 2, softmax + cross-entropy, full batch, hand-written backprop.
const TRAIN = (() => {
  const rng = s => () => { s = s + 0x6D2B79F5 | 0; let t = Math.imul(s ^ s >>> 15, 1 | s); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  const gauss = r => () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
  // each generator returns [x, y, label] in roughly [-1, 1]²
  const DATASETS = {
    'Hai cụm': (r, g, i) => { const c = i % 2; return [(c ? 0.45 : -0.45) + 0.28 * g(), (c ? 0.45 : -0.45) + 0.28 * g(), c]; },
    'Vòng tròn': (r, g, i) => { const c = i % 2, rad = c ? 0.45 * Math.sqrt(r()) : 0.7 + 0.25 * r(), a = 2 * Math.PI * r(); return [rad * Math.cos(a) + 0.04 * g(), rad * Math.sin(a) + 0.04 * g(), c]; },
    'XOR': (r) => { const x = 2 * r() - 1, y = 2 * r() - 1; return [x + (x > 0 ? 0.06 : -0.06), y + (y > 0 ? 0.06 : -0.06), x * y > 0 ? 1 : 0]; },
    // ~1.9 turns: 8 hidden units stall near 95%, 16 reach 100%
    'Xoắn ốc': (r, g, i) => { const c = i % 2, t = 0.25 + 2.6 * r(), a = 4.5 * t + c * Math.PI; return [0.36 * t * Math.cos(a) + 0.03 * g(), 0.36 * t * Math.sin(a) + 0.03 * g(), c]; }
  };
  function makeData(name, n = 200) {
    const r = rng(42), g = gauss(r), X = [], Y = [];
    for (let i = 0; i < n; i++) { const [x, y, c] = DATASETS[name](r, g, i); X.push([x, y]); Y.push(c); }
    return { X, Y };
  }
  const mat = (a, b, f) => Array.from({ length: a }, () => Array.from({ length: b }, f));
  // He initialisation: weights ~ N(0, 2/fan_in), biases 0
  function init(H, seed = 1) {
    const g = gauss(rng(seed)), w = (a, b) => mat(a, b, () => g() * Math.sqrt(2 / a));
    const net = { H, W: [w(2, H), w(H, H), w(H, 2)], b: [new Array(H).fill(0), new Array(H).fill(0), [0, 0]], t: 0 };
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
  // one update; the gradient at the new weights is computed right away and kept for the next step,
  // so the loss returned belongs to the weights now in net (the ones the boundary is drawn from) at no extra cost
  function step(net, D, { act, opt, lr }) {
    const r = net.next?.D === D && net.next.act === act ? net.next.r : lossGrad(net, D, act), b1 = 0.9, b2 = 0.999;
    net.t++;
    const upd = (P, G, M, V, i) => { // M[i], V[i]: the optimizer's memory for this parameter
      if (opt === 'sgd') return P - lr * G;
      if (opt === 'momentum') { M[i] = 0.9 * M[i] + G; return P - lr * M[i]; }
      M[i] = b1 * M[i] + (1 - b1) * G; V[i] = b2 * V[i] + (1 - b2) * G * G; // Adam
      return P - lr * (M[i] / (1 - b1 ** net.t)) / (Math.sqrt(V[i] / (1 - b2 ** net.t)) + 1e-8);
    };
    net.W.forEach((W, k) => W.forEach((row, i) => row.forEach((w, j) => { row[j] = upd(w, r.gW[k][i][j], net.m.W[k][i], net.v.W[k][i], j); })));
    net.b.forEach((b, k) => b.forEach((w, j) => { b[j] = upd(w, r.gb[k][j], net.m.b[k], net.v.b[k], j); }));
    net.next = { D, act, r: lossGrad(net, D, act) };
    return { loss: net.next.r.loss, acc: net.next.r.acc };
  }
  // p(class 1) as a function of (x, y), without the arrays forward() builds per point: the boundary picture needs ~3 300 of them
  function prober(net, act) {
    const f = ACT[act][0], H = net.H, [W1, W2, W3] = net.W, [b1, b2, b3] = net.b, a1 = new Float64Array(H), a2 = new Float64Array(H);
    return (x, y) => {
      for (let j = 0; j < H; j++) a1[j] = f(x * W1[0][j] + y * W1[1][j] + b1[j]);
      for (let j = 0; j < H; j++) { let s = b2[j]; for (let i = 0; i < H; i++) s += a1[i] * W2[i][j]; a2[j] = f(s); }
      let z0 = b3[0], z1 = b3[1];
      for (let i = 0; i < H; i++) { z0 += a2[i] * W3[i][0]; z1 += a2[i] * W3[i][1]; }
      return 1 / (1 + Math.exp(z0 - z1)); // softmax over two logits = sigmoid of their difference
    };
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
  function diagnose(h, acc, D) {
    const n = h.length - 1, L0 = h[0], last = h.at(-1);
    // a dead net predicts the class shares for every point: its loss is their entropy (ln 2 when balanced), its accuracy the larger share
    const p = D.Y.reduce((s, y) => s + y, 0) / D.Y.length, xlx = q => q > 0 ? q * Math.log(q) : 0, flat = -xlx(p) - xlx(1 - p);
    if (!Number.isFinite(last) || last > 10) return 'diverged';
    if (n === 0) return 'idle';
    if (n >= 30 && Math.abs(last - flat) < 0.01 && acc <= Math.max(p, 1 - p)) return 'dead';
    if (acc >= 0.99) return 'solved';
    const recent = h.slice(-40), ups = recent.filter((v, i) => i && v > recent[i - 1] * (1 + 1e-3)).length; // rises under 0.1% are rounding noise
    if (n >= 40 && ups > 12) return 'oscillating';
    if (n >= 300 && last > 0.9 * L0) return 'slow';
    if (n >= 400 && acc < 0.97 && h.at(-100) - last < 0.01) return 'plateau';
    return 'learning';
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
  return { DATASETS: Object.keys(DATASETS), makeData, init, resetOpt, step, lossGrad, prober, pacer, diagnose, gradCheck: worst };
})();

// Gradient flowing backwards through many layers (or time steps), with fixed random 32×32 Jacobians.
const GRAD = (() => {
  const d = 32, MAX = 100;
  const rand = (s => () => { s = s + 0x6D2B79F5 | 0; let t = Math.imul(s ^ s >>> 15, 1 | s); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; })(7);
  const gauss = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
  const mat = () => Array.from({ length: d }, () => Array.from({ length: d }, () => gauss() / Math.sqrt(d))); // E‖Wᵀg‖ ≈ ‖g‖
  const Ws = Array.from({ length: MAX }, mat), Wr = mat();
  const tanhD = Array.from({ length: MAX }, () => Array.from({ length: d }, () => 0.3 + 0.7 * rand())); // tanh' of each unit at each step
  const g0 = (v => { const z = Math.hypot(...v); return v.map(x => x / z); })(Array.from({ length: d }, gauss));
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
