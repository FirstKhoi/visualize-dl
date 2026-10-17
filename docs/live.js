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
