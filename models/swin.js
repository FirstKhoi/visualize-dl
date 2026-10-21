// Swin-T (Liu et al., 2021) as in torchvision swin_t: 4×4 patches, C = 96, depths {2, 2, 6, 2}, heads {3, 6, 12, 24}, window 7.
// The token grid stays 2D (shapes H×H×C, channels last). live: 'swin' paints the toy window grid of SWIN (live.js).
(() => {
const M = 7, SH = 3, T = M * M; // window, shift ⌊M/2⌋, tokens per window
const pBlock = (C, h) => 12 * C * C + 13 * C + (2 * M - 1) ** 2 * h; // 2 LayerNorm, qkv, proj, bias table, MLP ×4
const pMerge = C => 8 * C + 8 * C * C; // LayerNorm(4C) + Linear(4C → 2C, no bias)
const g = H => `${H}×${H}`;
const sci = v => { const e = Math.floor(Math.log10(v)); return R`${(v / 10 ** e).toFixed(2)} \cdot 10^{${e}}`; };
const step = [0.6, 1.5, 1.3]; // reshape-only steps

// shifted: second block of a pair. win: the grid holds more than one window (stages 1–3); the toy demo only fits there.
function windowAttention(C, h, H, shifted) {
  const win = H > M, sh = shifted && win, n7 = H / M, nW = n7 * n7, d = C / h, hw = H * H, W = `${nW}×${T}`;
  const demo = cfg => win ? { live: 'swin', swin: cfg } : {};
  const kids = [
    n('Q·Kᵀ', 'attn', `→ ${nW}×${h}×${T}×${T}`, {
      what: R`Trong mỗi cửa sổ và mỗi đầu, tích vô hướng giữa mọi cặp trong ${T} token: một ma trận điểm \(${T} \times ${T}\). Hai token ở hai cửa sổ khác nhau không bao giờ được so với nhau.`,
      formula: [R`S = QK^\top \in \mathbb{R}^{${T} \times ${T}}`],
      params: R`Không có tham số. \(${nW} \cdot ${h} \cdot ${T} \cdot ${T} \cdot ${d} = ${fmt(nW * h * T * T * d)}\) phép nhân-cộng cho cả lưới.`,
      code: `scores = q @ k.transpose(-2, -1)   # (B·${nW}, ${h}, ${T}, ${T})`
    }, { size: [1.2, 1.3, 1.3] }),
    n('Chia √dₖ', 'norm', `${nW}×${h}×${T}×${T}`, {
      what: R`Chia mọi điểm cho \(${sqrtTeX(d)}\), như mọi attention khác.`,
      formula: [R`\tilde S = S / \sqrt{d}, \qquad d = ${d}`],
      code: `scores = scores / math.sqrt(${d})`
    }, { tag: '÷ √d', size: SLAB }),
    n('Bias vị trí tương đối B', 'embed', `${nW}×${h}×${T}×${T}`, {
      what: R`Cộng vào điểm của cặp token \((i, j)\) một số chỉ phụ thuộc độ lệch vị trí giữa chúng trong cửa sổ. Độ lệch mỗi chiều nằm trong \([-${M - 1}, ${M - 1}]\), nên mỗi đầu chỉ cần một bảng \(${2 * M - 1} \times ${2 * M - 1} = ${(2 * M - 1) ** 2}\) số.`,
      formula: [R`B_{ij} = \hat B\big[r_i - r_j + ${M - 1},\ c_i - c_j + ${M - 1}\big]`, R`\hat B \in \mathbb{R}^{${2 * M - 1} \times ${2 * M - 1}}`,
        R`\text{idx} = (\Delta r + ${M - 1})\cdot ${2 * M - 1} + (\Delta c + ${M - 1}) \in \{0, \dots, ${(2 * M - 1) ** 2 - 1}\}`, R`\mathrm{Attention} = \mathrm{softmax}\big(QK^\top / \sqrt{d} + B\big)V`],
      why: `Swin không cộng vector vị trí vào token như ViT. Vị trí chỉ đi vào qua bias này, và vì bias phụ thuộc độ lệch chứ không phụ thuộc vị trí tuyệt đối, một cặp "cách nhau 2 hàng, 5 cột" được đối xử như nhau ở mọi chỗ trong ảnh: đúng giả định dịch chuyển của CNN mà ViT bỏ đi. Bảng 4 của bài gốc: không có thông tin vị trí 80.1% top-1, PE tuyệt đối như ViT 80.5%, bias tương đối 81.3%. Vì sao là bảng mà không học thẳng ma trận B? B cỡ ${T} × ${T} = ${fmt(T * T)} số mỗi đầu, và hai cặp cùng độ lệch lại có hai tham số riêng; bảng chỉ ${(2 * M - 1) ** 2} số, dùng chung cho mọi cửa sổ (mỗi đầu một bảng). Điểm yếu: bảng gắn chặt với cỡ cửa sổ 7. Muốn dùng cửa sổ lớn hơn cho ảnh lớn hơn thì phải nội suy bảng, và độ chính xác tụt mạnh: Swin V2 thay bảng bằng một MLP nhỏ (Log-CPB).`,
      example: R`Token \(i\) ở ô \((0, 0)\) và token \(j\) ở ô \((2, 5)\) của cửa sổ: \(B_{ij} = \hat B[0 - 2 + 6,\ 0 - 5 + 6] = \hat B[4, 1]\). Cặp ở \((3, 1)\) và \((5, 6)\) cũng lệch \((-2, -5)\), nên dùng đúng ô đó. Trong code, ô này là hàng \(\text{idx} = 4 \cdot 13 + 1 = 53\) của relative_position_bias_table.`,
      params: R`\(${(2 * M - 1) ** 2} \times ${h} = ${fmt((2 * M - 1) ** 2 * h)}\) cho mỗi khối.`,
      code: `bias = self.relative_position_bias_table[self.relative_position_index]   # (${T}·${T}, ${h})
scores = scores + bias.view(${T}, ${T}, ${h}).permute(2, 0, 1)`
    }, { tag: '+ B', size: SLAB, ref: [ref('swin', 'mục 3.2, công thức (4) và Bảng 4')] }),
    ...(sh ? [n('Mặt nạ', 'norm', `${nW}×${h}×${T}×${T}`, {
      what: 'Sau khi cuộn, vài cửa sổ gom các patch vốn nằm ở hai mép đối diện của lưới. Mặt nạ cộng −100 vào điểm của mọi cặp đến từ hai vùng khác nhau, nên sau softmax các cặp đó gần như bằng 0: mỗi patch vẫn chỉ chú ý trong đúng cửa sổ dịch của nó.',
      formula: [R`M_{ij} = \begin{cases} 0 & \text{cùng vùng} \\ -100 & \text{khác vùng} \end{cases}`, R`e^{-100} \approx 3.7 \cdot 10^{-44}`],
      why: 'Cách ngây thơ là đệm các cửa sổ nhỏ ở mép cho đủ 7×7: tốn thêm phép tính (bài gốc nêu ví dụ 2×2 → 3×3 cửa sổ, gấp 2.25 lần). Cuộn rồi che giữ nguyên số cửa sổ, và Bảng 5 của bài gốc đo cách này nhanh hơn cách đệm. Mặt nạ là hằng số, tính một lần, không có tham số.',
      example: R`Ở tầng này, trong ${nW} cửa sổ sau khi cuộn: ${(n7 - 1) ** 2} cửa sổ không cần che, ${2 * (n7 - 1)} cửa sổ ở hàng cuối và cột cuối gom 2 vùng, cửa sổ góc dưới phải gom 4 vùng.`,
      code: `attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)   # khác 0 = khác vùng
attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
scores = scores + attn_mask   # (${nW}, ${T}, ${T}), như nhau cho mọi ảnh và mọi đầu`
    }, { size: SLAB, ref: [ref('swin', 'mục 3.2, hình 4 và Bảng 5')], ...demo({ mask: true }) })] : []),
    n('Softmax', 'act', `${nW}×${h}×${T}×${T}`, {
      what: R`Chuẩn hoá từng hàng của ma trận \(${T} \times ${T}\) thành trọng số không âm, tổng bằng 1.`,
      formula: [R`A_{ij} = \frac{\exp(\tilde S_{ij} + B_{ij}${sh ? ' + M_{ij}' : ''})}{\sum_k \exp(\tilde S_{ik} + B_{ik}${sh ? ' + M_{ik}' : ''})}`],
      code: `attn = scores.softmax(dim=-1)`
    }, { size: SLAB }),
    n('A·V', 'attn', `→ ${nW}×${h}×${T}×${d}`, {
      what: 'Mỗi token nhận về trung bình có trọng số các value trong cửa sổ của nó.',
      formula: [R`O = AV`],
      code: `out = attn @ v   # (B·${nW}, ${h}, ${T}, ${d})`
    }, { size: [1.2, 1.5, 1.3] })
  ];
  const core = n('Scaled Dot-Product Attention', 'attn', `${nW}×${h}×${T}×${d} → ${nW}×${h}×${T}×${d}`, {
    what: R`Attention lõi, chạy riêng trong từng cửa sổ và từng đầu: \(${nW} \times ${h}\) bài attention nhỏ, mỗi bài ${T} token, thay vì một bài ${fmt(hw)} token.`,
    formula: [R`\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\big(QK^\top / \sqrt{d} + B${sh ? ' + M' : ''}\big)V`],
    why: 'Giống attention của Transformer, thêm bias vị trí tương đối B' + (sh ? ' và mặt nạ M của cửa sổ dịch.' : '.'),
    code: `out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias${sh ? ' + attn_mask' : ''})   # tương đương`
  }, { tag: 'Attention lõi', repeat: h, hint: `${kids.length} bước · ${h} đầu`, size: [1.5, 1.6, 1.3], children: kids, ref: [ref('swin', 'công thức (4)'), ref('vaswani', 'mục 3.2.1')] });

  const children = [
    ...(sh ? [n('Dịch vòng (cyclic shift)', 'shape', `${g(H)}×${C}`, {
      what: R`Cuộn cả lưới lên trên ${SH} hàng và sang trái ${SH} cột; phần tràn khỏi mép trên và mép trái quay vòng xuống mép dưới và mép phải. Chia cửa sổ đều như cũ trên lưới đã cuộn chính là chia cửa sổ lệch ${SH} patch trên lưới gốc.`,
      formula: [R`x'_{i,j} = x_{(i + ${SH}) \bmod ${H},\ (j + ${SH}) \bmod ${H}}`],
      why: R`Cửa sổ lệch nửa cửa sổ (\(\lfloor M/2 \rfloor = ${SH}\)) cắt ngang ranh giới các cửa sổ của khối trước, nên thông tin đi được từ cửa sổ này sang cửa sổ kia. Lệch thật thì số cửa sổ tăng từ \(${n7}^2\) lên \(${n7 + 1}^2\), vài cửa sổ nhỏ hơn 7×7. Cuộn vòng giữ đúng ${nW} cửa sổ đủ cỡ; đổi lại vài cửa sổ gom các patch vốn ở hai mép đối diện, và mặt nạ phía sau phải tách chúng ra.`,
      code: `x = torch.roll(x, shifts=(-${SH}, -${SH}), dims=(1, 2))`
    }, { tag: `Cuộn −${SH}`, size: step, ...demo({ rolled: true }) })] : []),
    ...(win ? [n('Chia cửa sổ', 'shape', `${g(H)}×${C} → ${W}×${C}`, {
      what: R`Cắt lưới \(${H} \times ${H}\) thành ${nW} cửa sổ 7×7 không chồng nhau; mỗi cửa sổ thành một chuỗi ${T} token. Từ đây các cửa sổ được xử lý như một batch: attention chỉ xảy ra giữa ${T} token trong cùng cửa sổ.`,
      formula: [R`\Omega(\mathrm{MSA}) = 4hwC^2 + 2(hw)^2C`, R`\Omega(\text{W-MSA}) = 4hwC^2 + 2M^2hwC`],
      why: R`Phần \(4hwC^2\) là các phép chiếu Q, K, V, Wᴼ, như nhau ở cả hai. Khác ở phần so khớp: toàn cục thì mỗi token so với \(hw\) token, chi phí tăng theo bình phương số token; trong cửa sổ thì chỉ so với \(M^2 = ${T}\) token, nên chi phí tăng tuyến tính theo kích thước ảnh.`,
      example: R`Ở tầng này (\(h = w = ${H}\), \(C = ${C}\)): toàn cục \(2(hw)^2C = ${sci(2 * hw * hw * C)}\), theo cửa sổ \(2M^2hwC = ${sci(2 * T * hw * C)}\), ít hơn \(hw / M^2 = ${hw / T}\) lần.`,
      code: `x = x.view(B, ${n7}, 7, ${n7}, 7, ${C}).permute(0, 1, 3, 2, 4, 5).reshape(B * ${nW}, ${T}, ${C})`
    }, { tag: 'Chia cửa sổ', size: step, ref: [ref('swin', 'mục 3.2, công thức (1) và (2)')] })] : []),
    n('Chiếu Q, K, V', 'linear', `${W}×${C} → 3 × ${W}×${C}`, {
      what: 'Một lớp Linear C → 3C tạo query, key, value cho mọi token, dùng chung một bộ trọng số cho mọi cửa sổ.',
      formula: [R`[Q, K, V] = X W_{qkv} + b, \qquad W_{qkv} \in \mathbb{R}^{${C} \times ${3 * C}}`],
      params: R`\(${C} \cdot ${3 * C} + ${3 * C} = ${fmt(3 * C * C + 3 * C)}\).`,
      code: `qkv = self.qkv(x)   # nn.Linear(${C}, ${3 * C})`
    }, { tag: 'Q, K, V', size: [1.2, 1.5, 1.3] }),
    n(`Tách ${h} đầu`, 'shape', `${W}×${C} → ${nW}×${h}×${T}×${d}`, {
      what: R`Cắt \(${C} = ${h} \times ${d}\) chiều thành ${h} đầu, mỗi đầu ${d} chiều.`,
      formula: [R`\mathbb{R}^{${T} \times ${C}} \to \mathbb{R}^{${h} \times ${T} \times ${d}}`],
      why: 'Swin giữ cố định 32 chiều mỗi đầu ở mọi tầng: số kênh gấp đôi thì số đầu cũng gấp đôi (3, 6, 12, 24).',
      code: `q, k, v = qkv.reshape(B * ${nW}, ${T}, 3, ${h}, ${d}).permute(2, 0, 3, 1, 4)`
    }, { tag: 'Tách đầu', size: step }),
    core,
    n('Nối các đầu', 'shape', `${nW}×${h}×${T}×${d} → ${W}×${C}`, {
      what: `Đặt output của ${h} đầu cạnh nhau để trở lại ${C} chiều.`,
      formula: [R`\mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_{${h}}) \in \mathbb{R}^{${T} \times ${C}}`],
      code: `out = out.transpose(1, 2).reshape(B * ${nW}, ${T}, ${C})`
    }, { tag: 'Nối đầu', size: step }),
    n('Chiếu ra Wᴼ', 'linear', `${W}×${C} → ${W}×${C}`, {
      what: 'Một lớp Linear trộn thông tin giữa các đầu.',
      formula: [R`Y = O W^O + b^O, \qquad W^O \in \mathbb{R}^{${C} \times ${C}}`],
      params: R`\(${C} \cdot ${C} + ${C} = ${fmt(C * C + C)}\).`,
      code: `out = self.proj(out)   # nn.Linear(${C}, ${C})`
    }, { tag: 'Wᴼ', size: [1.2, 1.5, 1.3] }),
    ...(win ? [n('Ghép cửa sổ', 'shape', `${W}×${C} → ${g(H)}×${C}`, {
      what: R`Ngược với bước chia: đặt ${nW} cửa sổ về đúng chỗ của chúng trên lưới \(${H} \times ${H}\).`,
      code: `x = out.view(B, ${n7}, ${n7}, 7, 7, ${C}).permute(0, 1, 3, 2, 4, 5).reshape(B, ${H}, ${H}, ${C})`
    }, { tag: 'Ghép cửa sổ', size: step })] : []),
    ...(sh ? [n('Dịch ngược', 'shape', `${g(H)}×${C}`, {
      what: `Cuộn ngược ${SH} hàng, ${SH} cột để mọi patch về lại vị trí ban đầu, sẵn sàng cộng với đường tắt residual.`,
      formula: [R`x_{i,j} = x'_{(i - ${SH}) \bmod ${H},\ (j - ${SH}) \bmod ${H}}`],
      code: `x = torch.roll(x, shifts=(${SH}, ${SH}), dims=(1, 2))`
    }, { tag: `Cuộn +${SH}`, size: step })] : [])
  ];
  return n(shifted ? 'Shifted window attention (SW-MSA)' : 'Window attention (W-MSA)', 'attn', `${g(H)}×${C} → ${g(H)}×${C}`, {
    what: !win ? R`Lưới \(${H} \times ${H}\) ở tầng này vừa đúng một cửa sổ 7×7, nên không chia cửa sổ và không dịch: đây là attention toàn cục trên ${T} token như ViT, chỉ thêm bias vị trí tương đối.`
      : sh ? `Cửa sổ lệch nửa cửa sổ (${SH} patch) so với khối trước. Cài đặt: cuộn lưới, chia cửa sổ đều như thường, chạy attention có mặt nạ, rồi cuộn ngược lại.`
      : R`Chia lưới \(${H} \times ${H}\) thành ${nW} cửa sổ 7×7 không chồng nhau, rồi chạy multi-head self-attention (${h} đầu) riêng trong từng cửa sổ ${T} token.`,
    formula: [R`Y_w = \mathrm{MSA}(X_w), \qquad w = 1, \dots, ${nW}`, R`\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\big(QK^\top / \sqrt{d} + B${sh ? ' + M' : ''}\big)V`],
    why: !win ? 'Tới tầng này mỗi token đại diện cho một vùng 32×32 pixel của ảnh gốc và cả ảnh chỉ còn 49 token, nên attention toàn cục đã rẻ.'
      : sh ? 'Xen kẽ cửa sổ thường và cửa sổ dịch thì sau hai khối thông tin đã vượt ranh giới cửa sổ; chồng nhiều cặp, vùng nhìn lan dần ra cả ảnh. Bảng 4 của bài gốc: chỉ dùng cửa sổ thường ở mọi khối thì top-1 còn 80.2% thay vì 81.3%, và kém 2.8 điểm cả ở phát hiện vật thể (box AP trên COCO) lẫn phân vùng (mIoU trên ADE20K).'
      : 'Attention trong cửa sổ thì rẻ, nhưng các cửa sổ không trao đổi gì với nhau: patch ở mép một cửa sổ không biết gì về patch sát bên ở cửa sổ kế. Khối ngay sau (SW-MSA) sửa đúng chỗ này.',
    params: R`qkv \(${fmt(3 * C * C + 3 * C)}\), Wᴼ \(${fmt(C * C + C)}\), bảng bias \(${fmt((2 * M - 1) ** 2 * h)}\): \(${fmt(4 * C * C + 4 * C + (2 * M - 1) ** 2 * h)}\).`,
    code: `x = self.attn(self.norm1(x))   # ShiftedWindowAttention(${C}, window_size=[7, 7], shift_size=[${sh ? SH : 0}, ${sh ? SH : 0}], num_heads=${h})`
  }, { tag: shifted ? 'SW-MSA' : 'W-MSA', size: [1.6, 1.8, 1.5], children, ref: [ref('swin', 'mục 3.2, hình 2 và 4')], ...demo({ shifted: sh }) });
}

function block(C, h, H, shifted, rep) {
  const s = g(H);
  return n(shifted ? 'Khối Swin (SW-MSA)' : 'Khối Swin (W-MSA)', 'block', `${s}×${C} → ${s}×${C}`, {
    what: (shifted ? 'Khối thứ hai của mỗi cặp: LayerNorm, attention trong cửa sổ dịch' : 'Khối thứ nhất của mỗi cặp: LayerNorm, attention trong cửa sổ thường')
      + ', cộng residual; LayerNorm, MLP, cộng residual.' + (rep > 1 ? ` Lặp ${rep} lần, xen kẽ với khối ${shifted ? 'W-MSA' : 'SW-MSA'}, mỗi lần trọng số riêng.` : ''),
    formula: shifted ? [R`\hat z^{l+1} = \text{SW-MSA}\big(\mathrm{LN}(z^{l})\big) + z^{l}`, R`z^{l+1} = \mathrm{MLP}\big(\mathrm{LN}(\hat z^{l+1})\big) + \hat z^{l+1}`]
      : [R`\hat z^{l} = \text{W-MSA}\big(\mathrm{LN}(z^{l-1})\big) + z^{l-1}`, R`z^{l} = \mathrm{MLP}\big(\mathrm{LN}(\hat z^{l})\big) + \hat z^{l}`],
    why: 'Ngoài phần attention, khối giống hệt một lớp encoder Pre-LN của ViT: LayerNorm trước mỗi nhánh, residual sau. Khi train, mỗi nhánh còn qua stochastic depth như EfficientNet, xác suất bỏ tăng dần từ 0 ở khối đầu tới 0.2 ở khối thứ 12.',
    params: R`\(${fmt(pBlock(C, h))}\) mỗi khối${rep > 1 ? R`, \(${fmt(rep * pBlock(C, h))}\) cho cả ${rep} khối` : ''}: attention \(${fmt(4 * C * C + 4 * C + (2 * M - 1) ** 2 * h)}\), MLP \(${fmt(8 * C * C + 5 * C)}\), 2 LayerNorm \(${fmt(4 * C)}\).`,
    code: `x = x + self.stochastic_depth(self.attn(self.norm1(x)))
x = x + self.stochastic_depth(self.mlp(self.norm2(x)))`
  }, { tag: shifted ? 'Khối SW-MSA' : 'Khối W-MSA', repeat: rep, hint: '6 phần', size: vol(C, H), ref: [ref('swin', 'công thức (3), hình 3b')],
    children: [layerNorm(s, C), windowAttention(C, h, H, shifted), addLeaf(s, C), layerNorm(s, C), ffn(s, { d: C, dff: 4 * C, act: 'gelu' }), addLeaf(s, C)],
    skips: [[0, 2, 'z'], [3, 5, 'ẑ']] });
}

const merging = (C, H) => { // C channels on an H×H grid in, 2C on H/2 × H/2 out
  const h2 = H / 2;
  return n('Patch Merging', 'linear', `${g(H)}×${C} → ${g(h2)}×${2 * C}`, {
    what: R`Gộp mỗi nhóm 2×2 patch kề nhau thành một: nối 4 vector ${C} chiều thành ${4 * C} chiều, LayerNorm, rồi Linear \(${4 * C} \to ${2 * C}\). Lưới nhỏ đi một nửa mỗi chiều (\(${H} \to ${h2}\)), số kênh gấp đôi.`,
    formula: [R`x'_{i,j} = \mathrm{LN}\big([x_{2i,2j};\ x_{2i+1,2j};\ x_{2i,2j+1};\ x_{2i+1,2j+1}]\big)\, W`, R`W \in \mathbb{R}^{${4 * C} \times ${2 * C}}`],
    why: 'Đây là thứ biến Swin thành kim tự tháp như CNN: lưới 56 → 28 → 14 → 7, đúng các độ phân giải của ResNet-50 sau stem. Nhờ có bản đồ đặc trưng nhiều tỉ lệ, Swin thay được ResNet làm xương sống cho các khung phát hiện vật thể và phân vùng. Gần giống một conv 2×2 bước 2 không bias, chỉ khác ở LayerNorm chen giữa.',
    params: R`LayerNorm \(2 \cdot ${4 * C} = ${fmt(8 * C)}\), Linear \(${4 * C} \cdot ${2 * C} = ${fmt(8 * C * C)}\) (không bias): \(${fmt(pMerge(C))}\).`,
    code: `x0, x1 = x[:, 0::2, 0::2], x[:, 1::2, 0::2]
x2, x3 = x[:, 0::2, 1::2], x[:, 1::2, 1::2]
x = torch.cat([x0, x1, x2, x3], -1)   # (B, ${h2}, ${h2}, ${4 * C})
x = self.reduction(self.norm(x))     # (B, ${h2}, ${h2}, ${2 * C})`
  }, { tag: 'Patch Merging', size: vol(2 * C, h2), ref: [ref('swin', 'mục 3.1')], children: [
    n('Gom 2×2 patch', 'shape', `${g(H)}×${C} → ${g(h2)}×${4 * C}`, {
      what: `Lấy 4 patch ở các vị trí (chẵn, chẵn), (lẻ, chẵn), (chẵn, lẻ), (lẻ, lẻ) và nối theo chiều kênh: mỗi nhóm 2×2 thành một vector ${4 * C} chiều. Không tính gì, không có tham số.`,
      formula: [R`\mathbb{R}^{${H} \times ${H} \times ${C}} \to \mathbb{R}^{${h2} \times ${h2} \times ${4 * C}}`],
      code: `x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)`
    }, { tag: 'Gom 2×2', size: vol(4 * C, h2) }),
    layerNorm(g(h2), 4 * C),
    n(`Linear ${4 * C} → ${2 * C}`, 'linear', `${g(h2)}×${4 * C} → ${g(h2)}×${2 * C}`, {
      what: `Nén ${4 * C} chiều về ${2 * C}: số kênh chỉ gấp đôi chứ không gấp bốn.`,
      formula: [R`y = x W, \qquad W \in \mathbb{R}^{${4 * C} \times ${2 * C}}`],
      params: R`\(${4 * C} \times ${2 * C} = ${fmt(8 * C * C)}\).`,
      code: `self.reduction = nn.Linear(${4 * C}, ${2 * C}, bias=False)`
    }, { tag: 'Linear ↓', size: vol(2 * C, h2) })
  ] });
};

// stage k: C channels, h heads on an H×H grid, L blocks (in pairs W-MSA / SW-MSA); stages 2–4 start with patch merging
function stage(k, C, h, H, L) {
  const Hin = k > 1 ? 2 * H : H, Cin = k > 1 ? C / 2 : C, p = (k > 1 ? pMerge(Cin) : 0) + L * pBlock(C, h);
  return n(`Tầng ${k}`, 'block', `${g(Hin)}×${Cin} → ${g(H)}×${C}`, {
    what: (k > 1 ? R`Patch Merging giảm lưới \(${Hin} \to ${H}\) và tăng kênh \(${Cin} \to ${C}\), rồi ` : 'Ngay sau patch embedding, ')
      + `${L} khối Swin xen kẽ cửa sổ thường và cửa sổ dịch, ${h} đầu attention, mỗi đầu 32 chiều. `
      + (H > M ? R`Lưới \(${H} \times ${H}\) chia thành ${(H / M) ** 2} cửa sổ 7×7.` : R`Lưới \(7 \times 7\) chỉ còn đúng một cửa sổ, nên attention ở đây là toàn cục và khối SW-MSA không dịch gì.`),
    formula: [R`\hat z^{l} = \text{W-MSA}\big(\mathrm{LN}(z^{l-1})\big) + z^{l-1}`, R`z^{l} = \mathrm{MLP}\big(\mathrm{LN}(\hat z^{l})\big) + \hat z^{l}`,
      R`\hat z^{l+1} = \text{SW-MSA}\big(\mathrm{LN}(z^{l})\big) + z^{l}`, R`z^{l+1} = \mathrm{MLP}\big(\mathrm{LN}(\hat z^{l+1})\big) + \hat z^{l+1}`],
    why: 'Số khối mỗi tầng luôn chẵn vì khối đi theo cặp: cửa sổ thường rồi cửa sổ dịch.'
      + (k === 3 ? ' Tầng này có nhiều khối nhất (6 trong 12), giống conv4_x của ResNet-50 cũng nhiều khối nhất (6 trong 16): phần lớn công việc diễn ra ở độ phân giải 14×14.' : ''),
    params: R`\(${fmt(p)}\) (khoảng ${(p / 1e6).toFixed(2)}M).`,
    code: (k > 1 ? `x = model.features[${2 * k - 2}](x)   # PatchMerging\n` : '') + `x = model.features[${2 * k - 1}](x)   # ${L} SwinTransformerBlock`
  }, { tag: `Tầng ${k}`, hint: `${L} khối · ${h} đầu`, size: vol(C, H), _p: p, ref: [ref('swin', 'mục 3.1 và 3.3')], children: [
    ...(k > 1 ? [merging(Cin, Hin)] : []), block(C, h, H, false, L / 2), block(C, h, H, true, L / 2)
  ] });
}

// Swin V2: what changed to scale Swin up (floats above the row: not a step of Swin-T's forward pass)
const swinV2 = n('Swin V2 (2022): sửa gì', 'block', 'các thay đổi so với Swin V1', {
  what: 'Swin V2 giữ nguyên khung của Swin (cửa sổ, cửa sổ dịch, patch merging) nhưng sửa ba chỗ để phóng to được lên 3 tỷ tham số và ảnh 1536×1536: đặt LayerNorm sau nhánh residual, đổi tích vô hướng thành cosine có nhiệt độ, và thay bảng bias bằng một mạng nhỏ trên toạ độ log. Thêm SimMIM để bớt phụ thuộc dữ liệu có nhãn. Khối này không nằm trong Swin-T ở hàng dưới; nó chỉ ra những gì V2 thay.',
  formula: [R`\text{V1: } x_{l+1} = x_l + F\big(\mathrm{LN}(x_l)\big)`, R`\text{V2: } x_{l+1} = x_l + \mathrm{LN}\big(F(x_l)\big)`, R`\text{V2: } \mathrm{Sim}(q_i, k_j) = \cos(q_i, k_j) / \tau + B_{ij}`],
  why: 'Ba rào cản khi phóng to: train mất ổn định (activation ở lớp sâu phình to), khoảng cách độ phân giải (train ở ảnh nhỏ, dùng ở ảnh lớn thì tụt độ chính xác), và đói dữ liệu có nhãn. Mỗi thay đổi nhắm vào một rào cản.',
  example: 'SwinV2-G có 3 tỷ tham số, train với 70 triệu ảnh có nhãn, ít hơn 40 lần so với JFT-3B mà các mô hình cùng cỡ trước đó dùng.',
  code: `from torchvision.models import swin_v2_t
model = swin_v2_t(weights="IMAGENET1K_V1").eval()`
}, { tag: 'Swin V2', float: true, hint: '4 phần', size: [1.2, 1.2, 1.0], ref: [ref('swinv2')], children: [
  n('Res-post-norm', 'norm', 'đổi chỗ LayerNorm', {
    what: 'V1 chuẩn hoá đầu vào của nhánh (pre-norm) rồi cộng thẳng output của nhánh vào dòng chính. V2 chuẩn hoá output của nhánh trước khi cộng.',
    formula: [R`\text{V1: } x_{l+1} = x_l + F\big(\mathrm{LN}(x_l)\big)`, R`\text{V2: } x_{l+1} = x_l + \mathrm{LN}\big(F(x_l)\big)`],
    why: 'Với pre-norm, output của nhánh không bị giới hạn nên biên độ của dòng chính cứ cộng dồn qua từng khối: ở lớp sâu nó lớn hơn hẳn ở lớp đầu, và chênh lệch đó càng lớn khi mô hình càng lớn, làm train mất ổn định. Chuẩn hoá output trước khi cộng thì mỗi khối chỉ thêm một lượng cỡ 1. Với mô hình rất lớn (SwinV2-H, -G), V2 còn thêm một LayerNorm trên dòng chính sau mỗi 6 khối.',
    code: `x = x + self.norm1(self.attn(x))   # V2: norm after the branch
x = x + self.norm2(self.mlp(x))`
  }, { tag: 'Post-norm', size: SLAB }),
  n('Scaled cosine attention', 'attn', 'thay QKᵀ/√d', {
    what: R`Điểm của cặp \((i, j)\) là cosine của góc giữa query và key, chia cho nhiệt độ \(\tau\) học được (riêng cho từng đầu và từng lớp, luôn lớn hơn 0.01), rồi cộng bias vị trí.`,
    formula: [R`\mathrm{Sim}(q_i, k_j) = \frac{\cos(q_i, k_j)}{\tau} + B_{ij}, \qquad \tau > 0.01`],
    why: R`Tích vô hướng tăng theo \(\|q\|\,\|k\|\): ở mô hình lớn, vài cặp có điểm rất lớn, softmax gần như dồn hết vào một token và gradient qua nó gần bằng 0. Cosine luôn nằm trong \([-1, 1]\), nên điểm chỉ phụ thuộc hướng của vector; \(\tau\) cho mỗi đầu tự chọn độ nhọn.`,
    example: R`Với \(\tau = 0.01\), điểm nằm trong \([-100, 100]\); với \(\tau = 1\), trong \([-1, 1]\): softmax gần như phẳng.`,
    code: `attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
attn = attn * torch.clamp(self.logit_scale, max=math.log(100.0)).exp() + bias   # logit_scale = log(1 / τ)`
  }, { tag: 'Cosine attn', size: [1.2, 1.3, 1.3] }),
  n('Log-CPB', 'embed', 'thay bảng bias', {
    what: R`Thay bảng \(13 \times 13\) bằng một MLP nhỏ (hai lớp, ReLU ở giữa) nhận độ lệch \((\Delta x, \Delta y)\) và trả về bias. Độ lệch được đưa qua thang log trước.`,
    formula: [R`B(\Delta x, \Delta y) = G\big(\widehat{\Delta x}, \widehat{\Delta y}\big), \qquad \widehat{\Delta x} = \mathrm{sign}(\Delta x) \cdot \log(1 + |\Delta x|)`],
    why: 'Bảng của V1 chỉ có giá trị cho đúng các độ lệch đã thấy khi train; đổi cỡ cửa sổ thì phải nội suy. MLP cho ra bias cho độ lệch bất kỳ, và thang log làm vùng phải ngoại suy hẹp lại.',
    example: R`Từ cửa sổ 8×8 sang 16×16: toạ độ thường đi từ \([-7, 7]\) lên \([-15, 15]\), phải ngoại suy thêm \(8/7 = 1.14\) lần; toạ độ log đi từ \([-2.079, 2.079]\) lên \([-2.773, 2.773]\), chỉ thêm 0.33 lần. Bảng 1 của bài: đem mô hình train ở cửa sổ 8 (ảnh 256) đi chạy ở cửa sổ 24 (ảnh 768) mà không fine-tune, bảng bias của V1 tụt từ 81.7% xuống 68.7%, Log-CPB chỉ từ 81.8% xuống 79.1%.`,
    code: `coords = torch.sign(coords) * torch.log2(coords.abs() + 1.0) / 3.0   # torchvision: log2, scaled to about [-1, 1]
bias = 16 * torch.sigmoid(self.cpb_mlp(coords))   # cpb_mlp: Linear(2, 512), ReLU, Linear(512, heads)`
  }, { tag: 'Log-CPB', size: SLAB }),
  n('SimMIM', 'io', 'train trước không cần nhãn', {
    what: 'Học tự giám sát bằng cách che ảnh: che ngẫu nhiên các mảng của ảnh, rồi bắt mạng dự đoán lại giá trị pixel ở chỗ bị che, bằng loss ℓ1 và một lớp dự đoán rất nhẹ. Sau đó mới fine-tune với nhãn.',
    formula: [R`\mathcal{L} = \frac{1}{|\Omega|} \sum_{p \in \Omega} \big| \hat x_p - x_p \big|, \qquad \Omega = \text{các pixel bị che}`],
    why: 'Mô hình càng lớn càng cần nhiều dữ liệu có nhãn. Dự đoán phần bị che không cần nhãn, nên mô hình học được biểu diễn tốt từ ảnh thô; nhờ vậy SwinV2-G chỉ cần 70 triệu ảnh có nhãn. ConvNeXt V2 dùng cùng ý tưởng cho CNN (FCMAE).',
    code: `x_rec = decoder(encoder(mask(x)))
loss = (x_rec - x).abs()[masked].mean()`
  }, { tag: 'SimMIM', size: [0.6, 1.4, 0.6], ref: [ref('simmim')] })
] });

const STAGES = [stage(1, 96, 3, 56, 2), stage(2, 192, 6, 28, 2), stage(3, 384, 12, 14, 6), stage(4, 768, 24, 7, 2)];
const pEmbed = 96 * 3 * 4 * 4 + 96 + 2 * 96, pTotal = pEmbed + STAGES.reduce((t, s) => t + s._p, 0) + 2 * 768 + 768 * 1000 + 1000;
// self-check: must reproduce torchvision's swin_t parameter count
console.assert(pTotal === 28288354, 'Swin-T param count mismatch');

MODELS.swin = n('Swin-T', 'block', '3×224×224 → 1000', {
  what: 'Transformer cho ảnh dựng theo kiểu kim tự tháp như CNN. Attention chỉ chạy trong từng cửa sổ 7×7 patch, và cứ mỗi khối thứ hai cửa sổ lệch đi nửa cửa sổ để thông tin chảy qua ranh giới. Giữa các tầng, Patch Merging gộp 2×2 patch thành một: lưới 56 → 28 → 14 → 7, kênh 96 → 768.',
  formula: [R`\Omega(\mathrm{MSA}) = 4hwC^2 + 2(hw)^2C`, R`\Omega(\text{W-MSA}) = 4hwC^2 + 2M^2hwC, \qquad M = 7`],
  why: 'ViT cho mọi token chú ý tới mọi token: chi phí tăng theo bình phương số token, nên không dùng được ô nhỏ hay ảnh lớn, và chỉ có một độ phân giải. Swin giữ lại hai giả định của CNN mà ViT bỏ: nhìn cục bộ (cửa sổ) và nhiều tỉ lệ (kim tự tháp). Chi phí tăng tuyến tính theo kích thước ảnh, và các bản đồ 56/28/14/7 lắp thẳng được vào các khung phát hiện vật thể hay phân vùng vốn làm cho ResNet. Trong ảnh y khoa, Swin-Unet và Swin UNETR dùng chính các khối này làm xương sống phân vùng.',
  example: 'Bảng 1 của bài gốc, train trên ImageNet-1K ở ảnh 224×224: Swin-T 81.3% top-1 với 4.5G FLOPs, DeiT-S (ViT cỡ tương đương) 79.8% với 4.6G. Bảng 4: bỏ cửa sổ dịch thì còn 80.2%, bỏ bias vị trí thì còn 80.1%.',
  params: R`\(${fmt(pTotal)}\) (khoảng 28.3M; Bảng 1 của bài gốc ghi 29M): patch embedding \(${fmt(pEmbed)}\), 4 tầng \(${STAGES.map(s => fmt(s._p)).join(' + ')}\), LayerNorm cuối \(1\,536\), head \(769\,000\). Cấu hình Tiny: \(C = 96\), số khối \(\{2, 2, 6, 2\}\), số đầu \(\{3, 6, 12, 24\}\).`,
  code: `from torchvision.models import swin_t, Swin_T_Weights
model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1).eval()
probs = model(x).softmax(-1)   # x: (B, 3, 224, 224) → (B, 1000)`
}, { label: 'Swin-T', noIn: true, outLabel: '1000 lớp', ref: [ref('swin', 'Bảng 1 và 4'), ref('swinunet'), ref('swinunetr')], children: [
  imageIn,
  n('Patch Embedding', 'embed', '3×224×224 → 56×56×96', {
    what: 'Cắt ảnh thành các ô 4×4 pixel không chồng nhau (56 × 56 = 3136 ô), chiếu mỗi ô (48 số) thành vector 96 chiều, rồi LayerNorm. Ô của Swin nhỏ hơn ô của ViT 16 lần về diện tích (4×4 so với 16×16).',
    formula: [R`x_p^k \in \mathbb{R}^{4 \cdot 4 \cdot 3 = 48}, \qquad e_k = \mathrm{LN}(x_p^k E + b)`, R`E \in \mathbb{R}^{48 \times 96}`],
    why: 'Ô nhỏ giữ chi tiết cần cho phát hiện vật thể và phân vùng, nhưng sinh ra rất nhiều token: 3136 thay vì 196 của ViT. Attention toàn cục trên 3136 token thì quá đắt, vì vậy Swin chỉ cho attention chạy trong từng cửa sổ nhỏ.',
    params: R`Conv \(96 \times 3 \times 4 \times 4 + 96 = 4\,704\), LayerNorm \(192\): \(${fmt(pEmbed)}\).`,
    code: `x = model.features[0](x)   # Conv2d(3, 96, 4, stride=4) → permute → LayerNorm: (B, 56, 56, 96)`
  }, { tag: 'Patch 4×4', size: vol(96, 56), ref: [ref('swin', 'mục 3.1')], children: [
    n('Conv 4×4, bước 4', 'conv', '3×224×224 → 96×56×56', {
      what: '96 bộ lọc cỡ 3×4×4, mỗi lần đặt lên đúng một ô rồi nhảy 4 pixel: chính là một lớp Linear áp riêng cho từng ô.',
      formula: CONV_F,
      example: R`\(k = 4,\ p = 0,\ s = 4\): \(\lfloor (224 - 4) / 4 \rfloor + 1 = 56\).`,
      params: R`\(96 \times 3 \times 4 \times 4 + 96 = 4\,704\).`,
      code: `nn.Conv2d(3, 96, kernel_size=4, stride=4)`
    }, { tag: 'Conv 4×4 /4', size: vol(96, 56) }),
    n('Đổi trục kênh ra sau', 'shape', '96×56×56 → 56×56×96', {
      what: 'Đưa trục kênh về cuối: mỗi vị trí trên lưới 56×56 giờ là một token 96 chiều. Swin giữ nguyên lưới 2D chứ không duỗi thành chuỗi như ViT, vì còn cần biết ô nào nằm cạnh ô nào để chia cửa sổ và gộp ô.',
      formula: [R`\mathbb{R}^{96 \times 56 \times 56} \xrightarrow{\text{permute}} \mathbb{R}^{56 \times 56 \times 96}`],
      code: `x = x.permute(0, 2, 3, 1)   # (B, 56, 56, 96)`
    }, { tag: 'Permute', size: step }),
    layerNorm('56×56', 96)
  ] }),
  ...STAGES,
  { ...layerNorm('7×7', 768), name: 'LayerNorm cuối' },
  gapNode(768, 7, 'Không có tham số. ViT lấy token [CLS] để phân loại; Swin không có [CLS] mà lấy trung bình 49 token cuối, như ResNet.'),
  fcOut(768, 'head'),
  swinV2
] });
})();
