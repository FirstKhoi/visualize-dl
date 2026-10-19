// LLaMA 3 8B (Meta, 2024): decoder-only, Pre-norm, RMSNorm, RoPE, GQA, SwiGLU, no biases, untied LM head.
(() => {
const C = { d: 4096, h: 32, kvh: 8, dk: 128, dff: 14336, V: 128256, L: 32 };
const G = C.h / C.kvh; // query heads sharing one K/V head
const pAttn = C.d * C.d * 2 + 2 * C.d * C.kvh * C.dk, pFfn = 3 * C.d * C.dff, pLayer = pAttn + pFfn + 2 * C.d;
const pTotal = 2 * C.V * C.d + C.L * pLayer + C.d;
// self-check: must match HF transformers' LlamaForCausalLM parameter count for Meta-Llama-3-8B
console.assert(pTotal === 8030261248, 'LLaMA 3 8B param count mismatch');

const rmsNorm = s => n('RMSNorm', 'norm', `${s}×${C.d}`, {
  what: R`Chia vector của từng token cho căn trung bình bình phương (RMS) của chính nó, rồi nhân với \(\gamma\) học được. Giống LayerNorm nhưng không trừ trung bình và không có \(\beta\).`,
  formula: [R`\mathrm{RMS}(x) = \sqrt{\frac{1}{${C.d}}\sum_{c=1}^{${C.d}} x_c^2 + \epsilon}, \qquad y_c = \gamma_c\, \frac{x_c}{\mathrm{RMS}(x)}`],
  why: 'Bài RMSNorm (2019) cho thấy phần giúp ổn định là co giãn độ lớn của vector, còn dời tâm về 0 gần như không cần. Bỏ bước trừ trung bình thì rẻ hơn một chút mà chất lượng tương đương, nên hầu hết LLM hiện nay dùng RMSNorm.',
  example: R`\(x = [1, 2, 3, 6]\): \(\mathrm{RMS} = \sqrt{(1 + 4 + 9 + 36)/4} = 3.536\) → \([0.283,\ 0.566,\ 0.849,\ 1.697]\). Cùng input với ví dụ LayerNorm, nhưng ở đây giá trị nào cũng giữ dấu vì không bị trừ trung bình.`,
  params: R`\(\gamma\): \(${fmt(C.d)}\).`,
  code: `var = x.pow(2).mean(-1, keepdim=True)
y = self.weight * x * torch.rsqrt(var + 1e-5)`
}, { size: SLAB, ref: [ref('rmsnorm')] });

const sampleSoftmaxEx = R`Logit \([2,\ 1,\ 0.1]\): \(T = 1\) → \([0.659,\ 0.242,\ 0.099]\); \(T = 0.5\) → \([0.864,\ 0.117,\ 0.019]\); \(T = 2\) → \([0.502,\ 0.304,\ 0.194]\).`;

const attention = n('Grouped-Query Attention', 'attn', `n×${C.d} → n×${C.d}`, {
  what: `Self-attention có mặt nạ nhân quả như decoder của Transformer gốc, với 3 thay đổi. Vị trí được mã hoá bằng cách xoay Q và K (RoPE). Có ${C.h} đầu query nhưng chỉ ${C.kvh} đầu key/value, mỗi đầu K/V dùng chung cho ${G} đầu Q (GQA). K, V của các token cũ được giữ lại (KV cache) để khi sinh khỏi tính lại.`,
  formula: [R`\mathrm{head}_i = \mathrm{Attention}\big(\mathrm{RoPE}(Q_i),\ \mathrm{RoPE}(K_{g(i)}),\ V_{g(i)}\big), \qquad g(i) = \lfloor i / ${G} \rfloor`, R`\mathrm{Attn}(X) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_{${C.h}})\, W^O`],
  why: `Khi sinh văn bản, nút thắt thường không phải phép nhân mà là việc đọc KV cache từ bộ nhớ GPU ở mỗi bước. GQA cắt KV cache ${G} lần (${C.kvh} thay vì ${C.h} đầu K/V) mà chất lượng gần bằng attention đầy đủ.`,
  example: `KV cache mỗi token: 2 (K và V) × ${C.L} lớp × ${C.kvh} đầu × ${C.dk} chiều × 2 byte (bf16) = 131 072 byte = 128 KiB. Đủ 8192 token ngữ cảnh: 1 GiB cho mỗi chuỗi. Nếu không có GQA (${C.h} đầu K/V): 4 GiB.`,
  params: R`\(W^Q, W^O\): \(${C.d} \times ${C.d}\); \(W^K, W^V\): \(${C.d} \times ${C.kvh * C.dk}\); không bias. Tổng \(${fmt(pAttn)}\).`,
  code: `q = self.q_proj(x).view(B, n, 32, 128).transpose(1, 2)
k = self.k_proj(x).view(B, n, 8, 128).transpose(1, 2)
v = self.v_proj(x).view(B, n, 8, 128).transpose(1, 2)
q, k = apply_rotary_pos_emb(q, k, cos, sin)       # RoPE
k, v = cache.update(k, v, layer_idx)              # KV cache
k, v = repeat_kv(k, 4), repeat_kv(v, 4)           # 8 → 32 đầu
out = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask)
out = self.o_proj(out.transpose(1, 2).reshape(B, n, 4096))`
}, { tag: 'Attention (GQA)', size: [1.6, 1.8, 1.5], ref: [ref('gqa'), ref('mqa', 'Multi-Query và KV cache')], children: [
  n('Chiếu Q, K, V', 'linear', `n×${C.d} → Q ${C.h}×n×${C.dk} · K,V ${C.kvh}×n×${C.dk}`, {
    what: `Như Transformer gốc, nhưng K và V chỉ được chiếu ra ${C.kvh * C.dk} chiều (${C.kvh} đầu × ${C.dk}) thay vì ${C.d}, và không có bias. Kết quả được tách thành đầu ngay.`,
    formula: [R`Q = X W^Q,\ W^Q \in \mathbb{R}^{${C.d} \times ${C.d}}, \qquad K = X W^K,\ V = X W^V,\ W^{K,V} \in \mathbb{R}^{${C.d} \times ${C.kvh * C.dk}}`],
    why: 'Không lớp Linear nào trong LLaMA có bias. Bài PaLM báo cáo bỏ bias giúp train mô hình lớn ổn định hơn, và chất lượng gần như không đổi; LLaMA làm theo.',
    params: R`\(${C.d} \cdot ${C.d} + 2 \cdot ${C.d} \cdot ${C.kvh * C.dk} = ${fmt(C.d * C.d + 2 * C.d * C.kvh * C.dk)}\).`,
    code: `q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)   # nn.Linear(..., bias=False)`
  }, { tag: 'Q, K, V', lshape: 'Q 32 đầu · K,V 8 đầu', size: [1.2, 1.5, 1.3], ref: [ref('palm', 'mục 2: bỏ bias')] }),
  n('RoPE', 'embed', `Q ${C.h}×n×${C.dk} · K ${C.kvh}×n×${C.dk}`, {
    what: 'Xoay từng cặp chiều của q và k một góc tỉ lệ với vị trí của token. Không cộng gì vào embedding: vị trí chỉ được "khắc" vào Q và K ngay trước khi tính tích vô hướng. V không bị xoay.',
    formula: [R`\begin{bmatrix} q'_{2i} \\ q'_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{bmatrix} \begin{bmatrix} q_{2i} \\ q_{2i+1} \end{bmatrix}, \qquad \theta_i = 500000^{-2i/${C.dk}}`, R`\langle R_m\, q,\ R_n\, k \rangle = \langle q,\ R_{n-m}\, k \rangle`],
    why: R`Dòng thứ hai là lý do RoPE hay: tích vô hướng giữa query ở vị trí \(m\) và key ở vị trí \(n\) chỉ phụ thuộc khoảng cách \(n - m\), không phụ thuộc vị trí tuyệt đối. Cùng ý tưởng "kim đồng hồ" với PE sin/cos của Transformer gốc, nhưng áp vào Q, K bằng phép nhân thay vì cộng vào input. LLaMA 3 dùng cơ số 500 000 (LLaMA 2: 10 000) để các kim quay chậm hơn, hợp với ngữ cảnh dài.`,
    example: R`\(q = [1, 0]\) ở vị trí \(m = 1\) với \(\theta_0 = 1\): xoay 1 rad → \([\cos 1,\ \sin 1] = [0.540,\ 0.841]\). Cặp (q ở vị trí 5, k ở 3) và cặp (q ở 12, k ở 10) cho cùng tích vô hướng vì cùng cách nhau 2.`,
    params: 'Không có tham số.',
    code: `def rotate_half(x):
    x1, x2 = x[..., :64], x[..., 64:]
    return torch.cat((-x2, x1), dim=-1)
q = q * cos + rotate_half(q) * sin   # cos, sin: (n, 128) theo vị trí
k = k * cos + rotate_half(k) * sin
# HF ghép cặp chiều (i, i+64) thay vì (2i, 2i+1); trọng số đã được hoán vị tương ứng`
  }, { lshape: '', size: SLAB, ref: [ref('rope')] }),
  n('KV cache', 'io', `K,V ${C.kvh}×t×${C.dk} mỗi lớp`, {
    what: 'Bộ nhớ đệm giữ K, V của mọi token đã xử lý. Khi sinh token mới, chỉ tính q, k, v cho đúng token đó, nối k, v vào cache, rồi attention trên toàn bộ cache.',
    formula: [R`K_{1:t} = [K_{1:t-1};\ k_t], \qquad V_{1:t} = [V_{1:t-1};\ v_t]`, R`o_t = \mathrm{Attention}(q_t,\ K_{1:t},\ V_{1:t})`],
    why: R`Không có cache, bước sinh thứ \(t\) phải chạy lại cả \(t\) token, tổng chi phí tăng theo bình phương. Có cache, mỗi bước chỉ tốn chi phí của 1 token (cộng việc đọc cache). Giữ lại được vì mặt nạ nhân quả bảo đảm token cũ không bao giờ nhìn thấy token mới, nên K, V của chúng không bao giờ đổi.`,
    example: `Mỗi token chiếm 128 KiB cache (xem khối Attention). Prompt 1000 token thì bước đầu (prefill) tính cả 1000 token một lần; mỗi token sinh sau đó chỉ tính 1 token mới và đọc lại 1000+ token trong cache.`,
    code: `k = torch.cat([cache_k, k], dim=2)   # (B, 8, t, 128)
v = torch.cat([cache_v, v], dim=2)`
  }, { lshape: `${C.kvh}×t×${C.dk}`, size: [0.9, 1.5, 1.3] }),
  n('Lặp K, V cho nhóm', 'shape', `${C.kvh}×t×${C.dk} → ${C.h}×t×${C.dk}`, {
    what: `Mỗi đầu K/V được nhân bản ${G} lần để khớp ${C.h} đầu Q: đầu Q số 0–3 dùng chung K/V số 0, đầu 4–7 dùng K/V số 1, và cứ thế.`,
    formula: [R`K'_i = K_{\lfloor i / ${G} \rfloor}, \qquad V'_i = V_{\lfloor i / ${G} \rfloor}, \qquad i = 0, \dots, ${C.h - 1}`],
    why: `Chỉ nhân bản lúc tính, không lưu thêm gì: cache vẫn chỉ có ${C.kvh} đầu. GQA nằm giữa hai cực: Multi-Head (${C.h} đầu K/V, chất lượng cao, cache to) và Multi-Query (1 đầu K/V, cache nhỏ nhất, chất lượng giảm).`,
    code: `k = k[:, :, None].expand(B, 8, 4, t, 128).reshape(B, 32, t, 128)   # repeat_kv`
  }, { tag: 'Lặp K, V', size: [0.6, 1.5, 1.3] }),
  sdpa('masked', 'n', 't', { h: C.h, dk: C.dk, drop: 0 }),
  n('Nối các đầu', 'shape', `${C.h}×n×${C.dk} → n×${C.d}`, {
    what: `Đặt output của ${C.h} đầu cạnh nhau để trở lại ${C.d} chiều.`,
    formula: [R`\mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_{${C.h}}) \in \mathbb{R}^{n \times ${C.d}}`],
    code: `out = out.transpose(1, 2).reshape(B, n, 4096)`
  }, { tag: 'Nối đầu', size: [0.6, 1.5, 1.3] }),
  n('Chiếu ra Wᴼ', 'linear', `n×${C.d} → n×${C.d}`, {
    what: 'Một lớp Linear không bias trộn thông tin giữa các đầu.',
    formula: [R`\mathrm{Attn}(X) = \mathrm{Concat}(\dots)\, W^O, \qquad W^O \in \mathbb{R}^{${C.d} \times ${C.d}}`],
    params: R`\(${C.d} \cdot ${C.d} = ${fmt(C.d * C.d)}\).`,
    code: `out = self.o_proj(out)   # nn.Linear(4096, 4096, bias=False)`
  }, { tag: 'Wᴼ', size: [1.2, 1.5, 1.3] })
] });

const swiglu = n('SwiGLU FFN', 'linear', `n×${C.d} → n×${C.d}`, {
  what: `FFN có cổng: một nhánh tạo giá trị, một nhánh qua SiLU làm cổng, hai nhánh nhân từng phần tử với nhau, rồi nén về ${C.d} chiều.`,
  formula: [R`\mathrm{FFN}(x) = \big(\mathrm{SiLU}(x W_{\text{gate}}) \odot x W_{\text{up}}\big)\, W_{\text{down}}`, R`W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{${C.d} \times ${C.dff}}, \qquad W_{\text{down}} \in \mathbb{R}^{${C.dff} \times ${C.d}}`],
  why: `Có 3 ma trận thay vì 2 nên lớp ẩn được thu nhỏ lại (${C.dff} ≈ 3.5 × ${C.d} thay vì 4 ×) để tổng tham số tương đương. Shazeer (2020) thử nhiều biến thể có cổng (GLU), SwiGLU tốt hơn ReLU và GELU ở cùng chi phí. Chính bài báo nói thẳng là không giải thích được vì sao nó tốt hơn.`,
  params: R`\(3 \times ${C.d} \times ${C.dff} = ${fmt(pFfn)}\), khoảng 80% tham số của mỗi lớp.`,
  code: `y = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))`
}, { tag: 'FFN (SwiGLU)', size: [1.4, 1.7, 1.4], hint: '5 phần', ref: [ref('glu')],
  extra: [{ a: { gap: 0 }, b: { i: 4, s: 'l' }, kind: 'branch' }, { a: { i: 4, s: 'r' }, b: { i: 2, s: 't' }, kind: 'branch', label: 'u' }],
  children: [
    n('Nhánh cổng W_gate', 'linear', `n×${C.d} → n×${C.dff}`, {
      what: `Nhánh cổng: chiếu lên ${C.dff} chiều. Sau SiLU, mỗi chiều quyết định cho bao nhiêu phần của nhánh giá trị đi qua.`,
      formula: [R`g = x W_{\text{gate}}`],
      params: R`\(${C.d} \times ${C.dff} = ${fmt(C.d * C.dff)}\).`,
      code: `g = self.gate_proj(x)   # nn.Linear(4096, 14336, bias=False)`
    }, { tag: 'W_gate', size: [1.7, 1.7, 1.4] }),
    silu(`n×${C.dff}`, SLAB),
    n('Nhân cổng ⊙', 'act', `n×${C.dff}`, {
      what: 'Nhân từng phần tử: output của cổng nhân với nhánh giá trị.',
      formula: [R`h = \mathrm{SiLU}(g) \odot u`],
      why: 'Đây chính là "cổng": nhánh cổng gần 0 thì chặn, lớn thì cho nhánh giá trị đi qua. Vì output là tích của hai hàm của x, khối này biểu diễn được tương tác bậc hai giữa các đặc trưng, thứ mà một phi tuyến đơn lẻ kiểu ReLU(xW) phải dùng nhiều neuron hơn mới bắt chước được.',
      code: `h = F.silu(g) * u`
    }, { tag: '⊙', size: BALL }),
    n('Nhánh xuống W_down', 'linear', `n×${C.dff} → n×${C.d}`, {
      what: `Nén về lại ${C.d} chiều để cộng vào dòng residual.`,
      formula: [R`y = h\, W_{\text{down}}`],
      params: R`\(${C.dff} \times ${C.d} = ${fmt(C.d * C.dff)}\).`,
      code: `y = self.down_proj(h)   # nn.Linear(14336, 4096, bias=False)`
    }, { tag: 'W_down', size: [1.0, 1.5, 1.3] }),
    n('Nhánh giá trị W_up', 'linear', `n×${C.d} → n×${C.dff}`, {
      what: `Nhánh giá trị: chiếu lên ${C.dff} chiều, chạy song song với nhánh cổng.`,
      formula: [R`u = x W_{\text{up}}`],
      params: R`\(${C.d} \times ${C.dff} = ${fmt(C.d * C.dff)}\).`,
      code: `u = self.up_proj(x)   # nn.Linear(4096, 14336, bias=False)`
    }, { tag: 'W_up', float: true, size: [1.7, 1.2, 1.2] })
  ] });

const block = n('Decoder', 'block', `n×${C.d} → n×${C.d}`, {
  what: `Một lớp kiểu Pre-norm: chuẩn hoá, attention, cộng residual; chuẩn hoá, FFN, cộng residual. Lặp ${C.L} lần, mỗi lần trọng số riêng.`,
  formula: [R`h = x + \mathrm{Attn}\big(\mathrm{RMSNorm}(x)\big)`, R`x' = h + \mathrm{FFN}\big(\mathrm{RMSNorm}(h)\big)`],
  why: `Pre-norm để dòng residual x đi xuyên ${C.L} lớp mà không bị chuẩn hoá lại ở mỗi bước: tín hiệu và gradient đi thẳng từ đầu đến cuối, nên mạng sâu train ổn định mà không cần warmup cẩn thận như Post-LN của Transformer gốc. Đổi lại phải thêm một RMSNorm ở cuối mạng.`,
  params: R`Mỗi lớp \(${fmt(pLayer)}\): attention \(${fmt(pAttn)}\), FFN \(${fmt(pFfn)}\), 2 RMSNorm \(${fmt(2 * C.d)}\). Cả ${C.L} lớp khoảng ${(C.L * pLayer / 1e9).toFixed(2)}B.`,
  code: `h = x + self.self_attn(self.input_layernorm(x), position_ids=pos)
out = h + self.mlp(self.post_attention_layernorm(h))`
}, { repeat: C.L, hint: `6 phần · ${C.L} lớp`, size: [2.0, 2.2, 1.6], ref: [ref('llama3'), ref('xiong', 'vì sao Pre-norm')],
  children: [rmsNorm('n'), attention, addLeaf('n', C.d), rmsNorm('n'), swiglu, addLeaf('n', C.d)],
  skips: [[0, 2, 'x'], [3, 5, 'h']] });

const sampling = n('Lấy mẫu token kế', 'act', `logit ${C.V} → 1 token`, {
  what: `Biến ${fmt(C.V)} logit ở vị trí cuối thành đúng một token: chia cho nhiệt độ, softmax, cắt bỏ phần đuôi phân bố, rồi bốc thăm.`,
  formula: [R`x_{n+1} \sim \mathrm{TopP}_p\big(\mathrm{softmax}(z / T)\big)`],
  why: 'Luôn lấy token có xác suất cao nhất (greedy) dễ bị lặp từ và câu nhạt. Bốc thăm có kiểm soát giữ câu tự nhiên mà vẫn không lạc đề. Các tham số T và p chính là "temperature" và "top_p" bạn hay chỉnh khi gọi API.',
  code: `probs = (logits[:, -1] / 0.7).softmax(-1)
# top-p 0.9: giữ nhóm token nhỏ nhất có tổng xác suất >= 0.9
sorted_p, idx = probs.sort(descending=True)
keep = sorted_p.cumsum(-1) - sorted_p < 0.9
sorted_p[~keep] = 0
next_id = idx.gather(-1, torch.multinomial(sorted_p / sorted_p.sum(-1, keepdim=True), 1))`
}, { tag: 'Lấy mẫu', size: [1.1, 1.5, 1.3], children: [
  n('Chia nhiệt độ T', 'norm', `${C.V}`, {
    what: 'Chia mọi logit cho cùng một số T trước khi softmax.',
    formula: [R`z'_k = z_k / T`],
    why: R`\(T \lt 1\) kéo các logit xa nhau ra nên phân bố nhọn hơn (chắc chắn hơn); \(T > 1\) làm phẳng hơn (đa dạng hơn); \(T \to 0\) thành greedy. Thứ tự các token không đổi, chỉ độ chênh giữa chúng đổi.`,
    example: sampleSoftmaxEx,
    code: `z = logits[:, -1] / T`
  }, { tag: '÷ T', size: SLAB }),
  n('Softmax', 'act', `${C.V}`, {
    what: 'Đổi logit thành xác suất trên toàn bộ từ điển.',
    formula: [R`p_k = \frac{e^{z'_k}}{\sum_{j=1}^{${C.V}} e^{z'_j}}`],
    why: 'Khi train, loss là cross-entropy của token đúng ở mọi vị trí: cả chuỗi dài vài nghìn token cho vài nghìn bài tập "đoán token kế" trong một lần chạy.',
    example: sampleSoftmaxEx,
    code: `probs = z.softmax(-1)`
  }, { size: SLAB }),
  n('Top-p (nucleus)', 'norm', `${C.V}`, {
    what: R`Giữ nhóm token nhỏ nhất có tổng xác suất \(\ge p\), bỏ phần còn lại, rồi chuẩn hoá lại cho tổng bằng 1.`,
    formula: [R`V_p = \text{tập nhỏ nhất với} \sum_{k \in V_p} p_k \ge p`, R`p'_k = \frac{p_k}{\sum_{j \in V_p} p_j} \ (k \in V_p), \qquad p'_k = 0 \ (k \notin V_p)`],
    why: 'Khác top-k (luôn giữ đúng k token), top-p tự co giãn: khi mô hình chắc chắn thì chỉ còn vài token, khi phân vân thì giữ nhiều hơn. Phần đuôi dài hàng trăm nghìn token rất ít khả năng bị cắt hẳn, nên không bao giờ bốc trúng một token vô nghĩa.',
    example: R`\(p = [0.659,\ 0.242,\ 0.099]\), top-p \(0.9\): \(0.659 \lt 0.9\), \(0.659 + 0.242 = 0.901 \ge 0.9\) → giữ 2 token → \([0.731,\ 0.269,\ 0]\).`,
    code: `sorted_p, idx = probs.sort(descending=True)
keep = sorted_p.cumsum(-1) - sorted_p < 0.9
sorted_p[~keep] = 0`
  }, { tag: 'Top-p', size: SLAB, ref: [ref('topp')] }),
  n('Bốc thăm', 'io', '1 token', {
    what: 'Chọn ngẫu nhiên một token theo phân bố đã cắt, nối id đó vào cuối chuỗi, rồi chạy lại mạng cho bước sau.',
    formula: [R`x_{n+1} \sim \mathrm{Categorical}(p'), \qquad x_{1:n+1} = [x_{1:n};\ x_{n+1}]`],
    why: 'Nhờ KV cache, bước sau chỉ phải tính cho đúng 1 token mới thay vì cả chuỗi. Sinh dừng khi gặp token kết thúc hoặc đủ số token tối đa.',
    code: `next_id = idx.gather(-1, torch.multinomial(sorted_p / sorted_p.sum(-1, keepdim=True), 1))
ids = torch.cat([ids, next_id], dim=1)`
  }, { size: [0.5, 1.4, 0.5] })
] });

MODELS.llm = n('LLaMA 3 8B', 'block', `n token → token kế tiếp`, {
  what: 'Mô hình ngôn ngữ chỉ có decoder: đọc một chuỗi token, đoán token kế tiếp, nối token vừa đoán vào rồi lặp lại. Đây là kiến trúc của hầu hết LLM hiện nay. So với Transformer gốc: bỏ encoder và cross-attention, chuẩn hoá trước sublayer, LayerNorm thành RMSNorm, PE thành RoPE, ReLU thành SwiGLU, và các nhóm đầu Q dùng chung K, V (GQA).',
  formula: [R`P(x_1, \dots, x_n) = \prod_{t=1}^{n} P(x_t \mid x_{\lt t})`, R`\mathcal{L} = -\frac{1}{n}\sum_{t=1}^{n} \log P(x_t \mid x_{\lt t})`],
  why: 'Chỉ một mục tiêu đơn giản là đoán token kế tiếp, nhưng trên lượng văn bản khổng lồ (LLaMA 3 train trên hơn 15 nghìn tỷ token), là đủ để mô hình học ngữ pháp, kiến thức và cả suy luận. Không cần encoder vì văn bản thuần không có cặp nguồn–đích: mọi thứ chỉ là một chuỗi dài. GPT-2 (2019) dùng cùng khung này nhưng với LayerNorm, PE học được và GELU.',
  example: 'Prompt "Thủ đô của Pháp là" → xác suất cao nhất rơi vào token " Paris" → nối vào, chạy tiếp để sinh token sau (minh hoạ).',
  params: R`\(${fmt(pTotal)}\): embedding \(${fmt(C.V * C.d)}\), ${C.L} lớp × \(${fmt(pLayer)}\), LM head \(${fmt(C.V * C.d)}\) (không dùng chung với embedding). Cấu hình: \(d = ${C.d}\), ${C.h} đầu Q, ${C.kvh} đầu K/V, \(d_{\text{head}} = ${C.dk}\), FFN \(${fmt(C.dff)}\), từ điển \(${fmt(C.V)}\), ngữ cảnh 8192 token.`,
  code: `from transformers import AutoModelForCausalLM, AutoTokenizer
name = "meta-llama/Meta-Llama-3-8B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)
ids = tok("Thủ đô của Pháp là", return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=20, do_sample=True, temperature=0.7, top_p=0.9)`
}, { label: 'LLaMA 3', noIn: true, outLabel: 'token kế', ref: [ref('llama3')],
  extra: [{ a: { i: 5, s: 't' }, b: { i: 0, s: 't' }, kind: 'loop', label: 'nối token mới vào, chạy lại' }],
  children: [
    n('Token vào', 'io', 'n', {
      what: `Prompt đã được tokenizer cắt thành id. LLaMA 3 dùng BPE kiểu tiktoken với ${fmt(C.V).replace('\\,', ' ')} token, gấp 4 lần LLaMA 2 (32 000).`,
      why: `Từ điển lớn hơn thì cùng một đoạn văn thành ít token hơn (Meta báo cáo ít hơn tới 15% so với LLaMA 2): attention rẻ hơn và ngữ cảnh chứa được nhiều chữ hơn. Đổi lại, embedding và LM head to ra: mỗi cái hơn 525M tham số.`,
      example: 'Tiếng Việt có dấu thường tốn nhiều token hơn tiếng Anh cho cùng một ý, vì tokenizer học chủ yếu từ dữ liệu tiếng Anh.',
      code: `ids = tok("Thủ đô của Pháp là", return_tensors="pt").input_ids   # (1, n)`
    }, { tag: 'Token', size: [0.5, 1.4, 0.5] }),
    n('Token Embedding', 'embed', `n → n×${C.d}`, {
      what: R`Tra bảng \(E \in \mathbb{R}^{${C.V} \times ${C.d}}\). Không cộng PE ở đây: thông tin vị trí được đưa vào bên trong mỗi lớp attention bằng RoPE.`,
      formula: [R`e_i = E[x_i]`],
      why: R`Không nhân \(\sqrt{d}\) như Transformer gốc, vì không còn PE cộng vào cần cân độ lớn.`,
      params: R`\(${C.V} \times ${C.d} = ${fmt(C.V * C.d)}\).`,
      code: `emb = nn.Embedding(128256, 4096)
x = emb(ids)   # (1, n, 4096)`
    }, { tag: 'Embedding', size: [1.2, 1.5, 1.3] }),
    block,
    { ...rmsNorm('n'), name: 'RMSNorm cuối' },
    n('LM head', 'linear', `n×${C.d} → n×${C.V}`, {
      what: `Chiếu trạng thái cuối thành ${fmt(C.V).replace('\\,', ' ')} logit. Khi sinh, chỉ cần logit ở vị trí cuối cùng.`,
      formula: [R`z = h\, W_{\text{out}}, \qquad W_{\text{out}} \in \mathbb{R}^{${C.d} \times ${C.V}}`],
      why: 'LLaMA 3 8B không dùng chung ma trận này với embedding (khác Transformer gốc và GPT-2). Với mô hình 8B, thêm 525M tham số là chấp nhận được, đổi lại đầu vào và đầu ra được học tự do.',
      params: R`\(${C.d} \times ${C.V} = ${fmt(C.V * C.d)}\).`,
      code: `logits = model.lm_head(h[:, -1])   # (1, 128256)`
    }, { size: [1.2, 1.6, 1.4] }),
    sampling
  ] });
})();
