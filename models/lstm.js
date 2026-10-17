// LSTM language model, "medium" config of Zaremba et al. (2014) on Penn Treebank: 2 layers × 650, vocab 10 000.
(() => {
const V = 10000, E = 650, H = 650, T = 4; // T = time steps drawn (training unrolls 35)
const pGate = H * (E + H) + 2 * H, pLayer = 4 * pGate, pTotal = V * E + 2 * pLayer + H * V + V;
// self-check: PyTorch nn.LSTM has 4h(in + h) + 8h params per layer (two bias vectors)
console.assert(pLayer === 4 * H * (E + H) + 8 * H && pTotal === 19780400, 'LSTM param count mismatch');
const sub = k => '₀₁₂₃₄₅₆₇₈₉'[k];

const gate = (name, tag, sym, act, what, why, example) => n(name, 'linear', `${E + H} → ${H}`, {
  what, why, example,
  formula: [R`${sym}_t = ${act}\big(W_${sym}\,[h_{t-1};\ x_t] + b_${sym}\big), \qquad W_${sym} \in \mathbb{R}^{${H} \times ${E + H}}`],
  params: R`\(${H} \times ${E + H} + 2 \times ${H} = ${fmt(pGate)}\) (PyTorch giữ hai vector bias, \(b_{ih}\) và \(b_{hh}\)).`,
  code: `${sym} = torch.${act === '\\sigma' ? 'sigmoid' : 'tanh'}(W_i${sym} @ x_t + b_i${sym} + W_h${sym} @ h + b_h${sym})   # (B, ${H})`
}, { tag, size: [0.8, 1.2, 1.0] });

const CELL = n('Ô LSTM (một bước)', 'block', `x ${E}, h ${H}, c ${H} → h ${H}, c ${H}`, {
  what: R`Một bước thời gian: đọc input \(x_t\), trạng thái ẩn \(h_{t-1}\) và bộ nhớ \(c_{t-1}\); ra \(h_t\) và \(c_t\). Bốn cổng cùng đọc \([h_{t-1}; x_t]\): cổng quên quyết định xoá bớt gì khỏi bộ nhớ, cổng vào và ứng viên quyết định ghi thêm gì, cổng ra quyết định đưa phần nào của bộ nhớ ra ngoài.`,
  formula: [
    R`f_t = \sigma(W_f [h_{t-1}; x_t] + b_f), \quad i_t = \sigma(W_i [\cdot] + b_i), \quad g_t = \tanh(W_g [\cdot] + b_g), \quad o_t = \sigma(W_o [\cdot] + b_o)`,
    R`c_t = f_t \odot c_{t-1} + i_t \odot g_t`,
    R`h_t = o_t \odot \tanh(c_t)`,
    R`\frac{\partial c_t}{\partial c_{t-1}} = \mathrm{diag}(f_t) \quad (\text{chỉ tính đường đi thẳng qua } c)`
  ],
  why: R`RNN thường có \(h_t = \tanh(W h_{t-1} + U x_t)\): gradient lùi \(k\) bước bị nhân \(k\) lần với \(W\) và đạo hàm tanh, nên tắt dần hoặc bùng nổ. LSTM thêm bộ nhớ \(c\) được cập nhật bằng phép cộng: gradient đi dọc \(c\) chỉ bị nhân với cổng quên \(f\), không qua ma trận trọng số nào. Khi \(f \approx 1\), gradient đi qua hàng trăm bước gần như nguyên vẹn (bài gốc gọi là "constant error carousel"). Cùng ý tưởng với đường tắt của ResNet, nhưng chạy dọc thời gian thay vì dọc các lớp.`,
  example: R`Một chiều, \(c_{t-1} = 2\), \(f = 0.9\), \(i = 0.5\), \(g = -0.6\), \(o = 0.8\): \(c_t = 0.9 \cdot 2 + 0.5 \cdot (-0.6) = 1.5\), \(h_t = 0.8 \cdot \tanh(1.5) = 0.8 \cdot 0.905 = 0.724\).`,
  params: R`4 cổng × \(${fmt(pGate)}\) = \(${fmt(pLayer)}\). Mọi bước thời gian dùng chung đúng bộ trọng số này.`,
  code: `gates = self.ih(x_t) + self.hh(h)          # (B, 2600) = 4 cổng × 650
i, f, g, o = gates.chunk(4, dim=-1)          # thứ tự của PyTorch
c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
h = torch.sigmoid(o) * torch.tanh(c)`
}, { layout: 'manual', hint: '10 phần · mọi bước dùng chung', size: [1.6, 1.8, 1.4], ref: [ref('lstm'), ref('colah')], children: [
  { ...n('Nối [hₜ₋₁; xₜ]', 'shape', `${H} + ${E} → ${E + H}`, {
    what: R`Ghép trạng thái trước \(h_{t-1}\) và input hiện tại \(x_t\) thành một vector ${E + H} chiều. Cả 4 cổng đều đọc vector này.`,
    formula: [R`[h_{t-1};\ x_t] \in \mathbb{R}^{${E + H}}, \qquad W [h_{t-1}; x_t] = W_{h} h_{t-1} + W_{x} x_t`],
    why: R`PyTorch không ghép thật mà tính \(W_{ih} x_t + W_{hh} h_{t-1}\) rồi cộng, tương đương về toán. Lợi: phần \(W_{ih} x_t\) của cả chuỗi không phụ thuộc \(h\), nên tính song song một lần trước vòng lặp; chỉ phần \(W_{hh} h_{t-1}\) phải chạy tuần tự.`,
    code: `xw = x @ W_ih.T + b_ih            # cả chuỗi một lần: (B, n, 2600)
gates_t = xw[:, t] + h @ W_hh.T + b_hh`
  }, { tag: '[h; x]', lshape: `${E + H}`, size: [0.5, 1.2, 1.0] }), pos: [-5, 0, 0] },
  { ...gate('Cổng quên f', 'f (quên)', 'f', '\\sigma',
    R`Quyết định giữ lại bao nhiêu phần của bộ nhớ cũ \(c_{t-1}\): 1 là giữ nguyên, 0 là xoá sạch. Mỗi chiều trong ${H} chiều có cổng riêng.`,
    R`Bài 1997 chưa có cổng này; Gers và cộng sự thêm vào năm 2000 để mạng biết tự xoá bộ nhớ khi chuyển sang câu mới. Mẹo khởi tạo: đặt \(b_f = 1\) để lúc đầu \(f \approx \sigma(1) = 0.73\), mạng mặc định nhớ thay vì quên (Jozefowicz và cộng sự, 2015).`,
    R`Giữ \(f = 0.9\) suốt 20 bước thì thông tin còn lại \(0.9^{20} = 0.12\); với \(f = 0.99\) còn \(0.99^{20} = 0.82\). Chỉ cần logit \(2.2\) là đã có \(\sigma(2.2) = 0.90\).`), ref: [ref('gers'), ref('jozefowicz', 'bias cổng quên = 1')], pos: [-3, 0, 0] },
  { ...gate('Cổng vào i', 'i (vào)', 'i', '\\sigma',
    R`Quyết định ghi bao nhiêu phần của ứng viên \(g_t\) vào bộ nhớ.`,
    R`Tách "có ghi hay không" (\(i\)) khỏi "ghi cái gì" (\(g\)) cho phép mạng bỏ qua các token không quan trọng (\(i \approx 0\)) mà không làm hỏng bộ nhớ đang giữ.`), pos: [-1.5, 0, 0] },
  { ...gate('Ứng viên g', 'g (ứng viên)', 'g', '\\tanh',
    R`Nội dung mới có thể được ghi vào bộ nhớ ở bước này.`,
    R`Dùng tanh vì cần giá trị trong \((-1, 1)\), có cả âm lẫn dương, để vừa tăng vừa giảm được bộ nhớ. Ba cổng còn lại dùng sigmoid vì cần một tỉ lệ trong \((0, 1)\).`), pos: [0, 0, 0] },
  { ...gate('Cổng ra o', 'o (ra)', 'o', '\\sigma',
    R`Quyết định đưa bao nhiêu phần của bộ nhớ ra \(h_t\), thứ mà lớp trên và bước sau nhìn thấy.`,
    R`Bộ nhớ có thể giữ thông tin chưa cần dùng ngay (ví dụ chủ ngữ số nhiều, để chia động từ vài từ sau). Cổng ra giấu nó khỏi \(h_t\) cho tới khi cần.`), pos: [1.5, 0, 0] },
  { ...n('Quên bớt: f ⊙ cₜ₋₁', 'act', `${H}`, {
    what: 'Nhân từng phần tử bộ nhớ cũ với cổng quên.',
    formula: [R`\tilde c = f_t \odot c_{t-1}`],
    why: R`Đây là chỗ duy nhất bộ nhớ cũ bị biến đổi, và chỉ bằng một phép nhân từng phần tử với số trong \((0, 1)\). Vì vậy \(\partial c_t / \partial c_{t-1} = \mathrm{diag}(f_t)\): không có ma trận trọng số trên đường đi của gradient.`,
    code: `c = torch.sigmoid(f) * c`
  }, { tag: 'f ⊙ c', size: BALL, exp: 'rnn' }), pos: [-3, 2.6, 0] },
  { ...n('Ghi mới: i ⊙ g', 'act', `${H}`, {
    what: 'Nhân ứng viên với cổng vào: phần nội dung thật sự được ghi.',
    formula: [R`i_t \odot g_t`],
    code: `torch.sigmoid(i) * torch.tanh(g)`
  }, { tag: 'i ⊙ g', size: BALL }), pos: [-0.75, 1.4, 0] },
  { ...n('Cập nhật bộ nhớ cₜ', 'add', `${H}`, {
    what: 'Cộng phần còn lại của bộ nhớ cũ với phần ghi mới.',
    formula: [R`c_t = f_t \odot c_{t-1} + i_t \odot g_t`],
    why: 'Bộ nhớ được cập nhật bằng phép cộng, giống dòng residual của ResNet và Transformer: thông tin chảy dọc thời gian như residual chảy dọc các lớp, mỗi bước chỉ cộng thêm một phần hiệu chỉnh.',
    example: R`\(0.9 \cdot 2 + 0.5 \cdot (-0.6) = 1.8 - 0.3 = 1.5\).`,
    code: `c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)`
  }, { tag: '⊕ → cₜ', size: BALL }), pos: [-0.75, 2.6, 0] },
  { ...n('tanh(cₜ)', 'act', `${H}`, {
    what: R`Nén bộ nhớ về khoảng \((-1, 1)\) trước khi đưa ra ngoài.`,
    formula: [R`\tanh(c_t) = \frac{e^{c_t} - e^{-c_t}}{e^{c_t} + e^{-c_t}}`],
    why: R`Bản thân \(c_t\) không bị chặn (cộng dồn qua nhiều bước có thể rất lớn); \(h_t\) cần nằm trong khoảng cố định để các lớp sau ổn định.`,
    example: R`\(\tanh(1.5) = 0.905\).`,
    code: `torch.tanh(c)`
  }, { tag: 'tanh', size: [0.5, 0.8, 0.8] }), pos: [3, 1.4, 0] },
  { ...n('Output hₜ = o ⊙ tanh(cₜ)', 'act', `${H}`, {
    what: R`Trạng thái ẩn mới: phần của bộ nhớ được cổng ra cho phép đi qua. \(h_t\) vừa là output của bước này, vừa là input cho bước sau và cho lớp trên.`,
    formula: [R`h_t = o_t \odot \tanh(c_t)`],
    example: R`\(0.8 \cdot 0.905 = 0.724\).`,
    code: `h = torch.sigmoid(o) * torch.tanh(c)`
  }, { tag: 'o ⊙ tanh', size: BALL }), pos: [3, 0, 0] }
], links: [
  { a: { ext: 0, s: 'l', d: 1.3 }, b: { i: 0, s: 'l' }, label: 'hₜ₋₁' },
  { a: { ext: 0, s: 'd', d: 1.9 }, b: { i: 0, s: 'd' }, label: 'xₜ', at: 0.05 },
  ...[1, 2, 3, 4].map(g => ({ a: { i: 0, s: 'r' }, b: { i: g, s: 'd' }, kind: 'branch' })),
  { a: { ext: 5, s: 'l', d: 2.4 }, b: { i: 5, s: 'l' }, kind: 'short', label: 'cₜ₋₁' },
  { a: { i: 1, s: 't' }, b: { i: 5, s: 'd' } },
  { a: { i: 5, s: 'r' }, b: { i: 7, s: 'l' }, kind: 'short' },
  { a: { i: 2, s: 't' }, b: { i: 6, s: 'l' } },
  { a: { i: 3, s: 't' }, b: { i: 6, s: 'r' } },
  { a: { i: 6, s: 't' }, b: { i: 7, s: 'd' } },
  { a: { i: 7, s: 'r' }, b: { stub: 7, s: 'r', d: 4.65 }, kind: 'short', label: 'cₜ' },
  { a: { i: 7, s: 'r' }, b: { i: 8, s: 't' } },
  { a: { i: 8, s: 'd' }, b: { i: 9, s: 't' } },
  { a: { i: 4, s: 'r' }, b: { i: 9, s: 'l' } },
  { a: { i: 9, s: 'r' }, b: { stub: 9, s: 'r' }, label: 'hₜ' }
] });

// one LSTM layer unrolled over T steps: c runs along the top (residual colour), h below it, x enters from underneath
const UP = [0, 0.4, 0], DN = [0, -0.4, 0], X_OFF = [-0.55, 0, 0]; // x enters left of centre so it misses the step label
const timeLinks = [
  { a: { stub: 0, s: 'l', off: UP }, b: { i: 0, s: 'l', off: UP }, kind: 'short', label: 'c₀' },
  { a: { stub: 0, s: 'l', off: DN }, b: { i: 0, s: 'l', off: DN }, label: 'h₀' }
];
for (let k = 0; k < T; k++) {
  const last = k === T - 1;
  timeLinks.push({ a: { ext: k, s: 'd', d: 2.0, off: X_OFF }, b: { i: k, s: 'd', off: X_OFF }, label: `x${sub(k + 1)}`, at: 0.15 });
  timeLinks.push({ a: { i: k, s: 't' }, b: { stub: k, s: 't' }, label: k ? '' : 'lên lớp trên' });
  timeLinks.push({ a: { i: k, s: 'r', off: UP }, b: last ? { stub: k, s: 'r', off: UP } : { i: k + 1, s: 'l', off: UP }, kind: 'short', label: `c${sub(k + 1)}` });
  timeLinks.push({ a: { i: k, s: 'r', off: DN }, b: last ? { stub: k, s: 'r', off: DN } : { i: k + 1, s: 'l', off: DN }, label: `h${sub(k + 1)}` });
}

const LAYER = n('LSTM', 'block', `n×${E} → n×${H}`, {
  what: R`Một lớp LSTM chạy cùng một ô qua từng bước thời gian, mang theo hai trạng thái: \(h\) (trạng thái ẩn, cũng là output) và \(c\) (bộ nhớ). Ở đây vẽ ${T} bước; khi train, mạng trải ra 35 bước. Có 2 lớp chồng nhau: lớp 2 nhận chuỗi \(h\) của lớp 1 làm input.`,
  formula: [R`(h_t,\ c_t) = \mathrm{LSTMCell}(x_t,\ h_{t-1},\ c_{t-1}), \qquad t = 1, \dots, n`, R`\frac{\partial c_t}{\partial c_{t-k}} \approx \prod_{j=0}^{k-1} \mathrm{diag}(f_{t-j})`],
  why: R`Bước \(t\) cần \(h_{t-1}\), nên các bước phải chạy tuần tự, không song song theo thời gian được như attention: đây là lý do Transformer thay thế LSTM. Đổi lại, khi sinh văn bản, bộ nhớ có kích thước cố định (\(h\) và \(c\): \(2 \times ${H}\) số mỗi lớp) dù chuỗi dài bao nhiêu, trong khi KV cache của LLaMA phình ra theo từng token. Ưu điểm này là lý do các mô hình hồi quy kiểu Mamba được quan tâm trở lại. Train bằng lan truyền ngược qua thời gian (BPTT) có cắt: chỉ lùi 35 bước rồi dừng, còn \(h, c\) cuối được mang sang đoạn văn kế tiếp.`,
  params: R`Mỗi lớp \(${fmt(pLayer)}\), 2 lớp \(${fmt(2 * pLayer)}\).`,
  code: `lstm = nn.LSTM(input_size=650, hidden_size=650, num_layers=2, dropout=0.5, batch_first=True)
out, (h, c) = lstm(x, (h0, c0))   # out: (B, n, 650); h, c: (2, B, 650)`
}, { tag: 'LSTM', repeat: 2, hint: `${T} bước · 2 lớp`, size: [2.0, 2.2, 1.6], exp: 'rnn', ref: [ref('zaremba')],
  children: Array.from({ length: T }, (_, k) => ({ ...CELL, name: `Bước t = ${k + 1}`, tag: `t = ${k + 1}`, lshape: '', hint: '10 phần' })),
  links: timeLinks });

MODELS.lstm = n('LSTM (mô hình ngôn ngữ)', 'block', 'n token → token kế tiếp', {
  what: 'Mạng hồi quy có cổng (Hochreiter & Schmidhuber, 1997), ở đây dùng làm mô hình ngôn ngữ: đọc từng từ một, giữ một bộ nhớ chạy dọc câu, và đoán từ kế tiếp. Cùng nhiệm vụ với LLaMA 3 nhưng xử lý tuần tự thay vì nhìn cả chuỗi cùng lúc. Cấu hình "medium" của Zaremba và cộng sự (2014) trên Penn Treebank: 2 lớp, 650 chiều, từ điển 10 000 từ.',
  formula: [R`P(x_{t+1} \mid x_{\le t}) = \mathrm{softmax}\big(W\, h^{(2)}_t + b\big)`],
  why: 'Từ khoảng 2014 đến 2017, LSTM là lựa chọn mặc định cho dịch máy, nhận dạng giọng nói và mô hình ngôn ngữ. Toàn bộ "trí nhớ" về phần câu đã đọc phải nén vào hai vector 650 chiều mỗi lớp; Transformer bỏ giới hạn đó bằng cách cho mỗi token nhìn thẳng mọi token trước.',
  example: 'Perplexity trên tập test Penn Treebank (Zaremba 2014): bản medium 82.7, bản large (1500 chiều) 78.4. Perplexity 82.7 nghĩa là mô hình phân vân trung bình như khi chọn đều giữa khoảng 83 từ.',
  params: R`\(${fmt(pTotal)}\): embedding \(${fmt(V * E)}\), 2 lớp LSTM \(${fmt(2 * pLayer)}\), lớp ra \(${fmt(H * V + V)}\).`,
  code: `class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10000, 650)
        self.lstm = nn.LSTM(650, 650, num_layers=2, dropout=0.5, batch_first=True)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(650, 10000)
    def forward(self, ids, state=None):
        y, state = self.lstm(self.drop(self.emb(ids)), state)
        return self.out(self.drop(y)), state   # mang state sang đoạn văn kế`
}, { label: 'LSTM', noIn: true, outLabel: 'từ kế', ref: [ref('lstm'), ref('zaremba', 'cấu hình medium, Bảng 1')], children: [
  n('Token vào', 'io', 'n', {
    what: 'Câu đã được tách thành từ và đổi thành id. Penn Treebank dùng từ điển 10 000 từ thường gặp nhất; từ hiếm được thay bằng ⟨unk⟩.',
    why: 'Mô hình thời đó làm việc theo nguyên từ, chưa dùng subword như BPE, nên từ điển nhỏ và gặp từ lạ là chịu.',
    example: '"the cat sat on" → [1, 2341, 870, 21] (id minh hoạ); mô hình phải đoán "the" hoặc "a" tiếp theo.',
    code: `ids = torch.tensor([[1, 2341, 870, 21]])   # (B, n)`
  }, { tag: 'Token', size: [0.5, 1.4, 0.5] }),
  n('Embedding', 'embed', `n → n×${E}`, {
    what: R`Tra bảng \(E \in \mathbb{R}^{${V} \times ${E}}\). Không có mã hoá vị trí: thứ tự đã nằm sẵn trong việc LSTM đọc lần lượt từng từ.`,
    formula: [R`x_t = E[\mathrm{id}_t]`],
    why: 'Khác Transformer, mạng hồi quy không cần PE vì bước t luôn được xử lý sau bước t − 1.',
    params: R`\(${V} \times ${E} = ${fmt(V * E)}\).`,
    code: `x = self.emb(ids)   # (B, n, 650)`
  }, { size: [1.2, 1.5, 1.3] }),
  LAYER,
  n('Dropout', 'act', `n×${H}`, {
    what: 'Tắt ngẫu nhiên 50% phần tử, chỉ trên các kết nối đi giữa các lớp (embedding → lớp 1, lớp 1 → lớp 2, lớp 2 → lớp ra).',
    formula: [R`\mathrm{Dropout}(h)_c = \frac{m_c\, h_c}{1 - p}, \qquad p = 0.5`],
    why: R`Đây là phát hiện chính của Zaremba và cộng sự (2014): dropout không được đặt trên đường hồi quy \(h_{t-1} \to h_t\), vì khi đó bộ nhớ bị xoá ngẫu nhiên ở mỗi bước và mạng không nhớ được lâu. Đặt đúng chỗ, dropout giúp mạng lớn hơn không bị overfit.`,
    code: `y = self.drop(y)   # nn.Dropout(0.5)`
  }, { size: SLAB, ref: [ref('zaremba', 'mục 3'), ref('dropout')] }),
  n('Linear → từ điển', 'linear', `n×${H} → n×${V}`, {
    what: 'Chiếu trạng thái ẩn của lớp 2 thành 10 000 logit, mỗi từ một điểm.',
    formula: [R`z_t = W h^{(2)}_t + b, \qquad W \in \mathbb{R}^{${V} \times ${H}}`],
    params: R`\(${H} \times ${V} + ${V} = ${fmt(H * V + V)}\).`,
    code: `logits = self.out(y)   # (B, n, 10000)`
  }, { tag: 'Linear', size: [1.2, 1.6, 1.4] }),
  n('Softmax', 'act', `n×${V}`, {
    what: 'Đổi logit thành xác suất cho từ kế tiếp.',
    formula: [R`P(x_{t+1} = k) = \frac{e^{z_{t,k}}}{\sum_{j} e^{z_{t,j}}}`, R`\mathrm{PPL} = \exp\!\Big(-\frac{1}{n}\sum_{t} \log P(x_{t+1} \mid x_{\le t})\Big)`],
    why: 'Perplexity (PPL) là e mũ của loss cross-entropy trung bình: đơn vị đo chuẩn cho mô hình ngôn ngữ, càng thấp càng tốt.',
    example: R`Nếu trung bình mô hình gán xác suất \(1/83\) cho từ đúng thì loss \(= \ln 83 = 4.42\) và \(\mathrm{PPL} = 83\).`,
    code: `loss = F.cross_entropy(logits.view(-1, 10000), targets.view(-1))
ppl = loss.exp()`
  }, { size: SLAB })
] });
})();
