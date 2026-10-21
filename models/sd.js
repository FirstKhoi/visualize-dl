// Stable Diffusion v1.5 (latent diffusion, Rombach et al., 2022) as in diffusers: CLIP ViT-L/14 text encoder, a U-Net that
// predicts the noise in 4×64×64 latents with cross-attention to the text, a 50-step sampler loop, and the KL-f8 VAE decoder.
(() => {
const TX = 77, TD = 768; // text tokens × width
const slab = H => [0.14, sp(H), sp(H)];
const hw = H => `${H}×${H}`;

const gn = (C, H) => n('GroupNorm', 'norm', `${C}×${hw(H)}`, {
  what: R`Chia ${C} kênh thành 32 nhóm, mỗi nhóm ${C / 32} kênh, rồi chuẩn hoá từng nhóm theo trung bình và phương sai tính trên các giá trị của riêng một ảnh. Sau đó co giãn và dịch từng kênh bằng \(\gamma, \beta\) học được.`,
  formula: [R`\mu_g = \frac{1}{|S_g|} \sum_{i \in S_g} x_i, \qquad \sigma_g^2 = \frac{1}{|S_g|} \sum_{i \in S_g} (x_i - \mu_g)^2`, R`y_i = \gamma_c\, \frac{x_i - \mu_g}{\sqrt{\sigma_g^2 + \epsilon}} + \beta_c`],
  why: 'Nằm giữa LayerNorm (một nhóm gồm mọi kênh) và InstanceNorm (mỗi kênh một nhóm). Khác BatchNorm, thống kê không phụ thuộc batch, nên train với batch nhỏ (ảnh lớn tốn bộ nhớ) vẫn ổn, và lúc train lẫn lúc suy luận chạy giống hệt nhau. U-Net khuếch tán dùng GroupNorm từ DDPM.',
  example: R`Ở đây mỗi nhóm có \(|S_g| = ${C / 32} \times ${H} \times ${H} = ${fmt(C / 32 * H * H)}\) giá trị.`,
  params: R`\(\gamma, \beta\): \(2 \times ${C} = ${fmt(2 * C)}\).`,
  code: `nn.GroupNorm(32, ${C})`
}, { tag: 'GN', size: slab(H), ref: [ref('groupnorm'), ref('ddpm', 'phụ lục B')] });

const conv3 = (name, tag, cin, cout, H, s = 1) => n(name, 'conv', `${cin}×${hw(H)} → ${cout}×${hw(H / s)}`, {
  what: s > 1 ? R`Conv 3×3 bước 2, có đệm: giảm kích thước một nửa (\(${H} \to ${H / 2}\)), giữ ${cout} kênh. Đây là mũi tên đi xuống của chữ U.`
    : R`Conv 3×3 có đệm 1: giữ nguyên kích thước \(${H} \times ${H}\), \(${cin} \to ${cout}\) kênh. Khác U-Net 2015 (không đệm), ở đây không phải cắt skip cho khớp.`,
  formula: CONV_F,
  example: R`\(k = 3,\ p = 1,\ s = ${s}\): \(\lfloor (${H} + 2 - 3) / ${s} \rfloor + 1 = ${H / s}\).`,
  params: R`\(${cin} \times ${cout} \times 3 \times 3 + ${cout} = ${fmt(9 * cin * cout + cout)}\).`,
  code: `nn.Conv2d(${cin}, ${cout}, kernel_size=3, stride=${s}, padding=1)`
}, { tag, size: vol(cout, H / s) });

const upsample = (C, H) => n('Phóng to ×2 + Conv 3×3', 'conv', `${C}×${hw(H)} → ${C}×${hw(2 * H)}`, {
  what: R`Lặp mỗi điểm thành một ô 2×2 (nearest neighbor), rồi conv 3×3: kích thước \(${H} \to ${2 * H}\), giữ ${C} kênh. Đây là mũi tên đi lên của chữ U.`,
  formula: [R`u_{c,i,j} = x_{c,\lfloor i/2 \rfloor,\lfloor j/2 \rfloor}, \qquad y = W_{3\times3} * u`],
  why: 'U-Net 2015 phóng to bằng conv chuyển vị. Phóng to cố định rồi mới conv tránh hẳn vân bàn cờ, lỗi hay gặp của conv chuyển vị khi kernel không chia hết cho bước.',
  params: R`\(${C} \times ${C} \times 3 \times 3 + ${C} = ${fmt(9 * C * C + C)}\).`,
  code: `x = F.interpolate(x, scale_factor=2.0, mode="nearest")
x = self.conv(x)   # nn.Conv2d(${C}, ${C}, 3, padding=1)`
}, { tag: 'Phóng to ×2', size: vol(C, 2 * H), ref: [ref('odena', 'resize rồi conv')] });

// ResNet-style block of the diffusion U-Net (temb: adds the timestep embedding) and of the VAE decoder (no temb).
// skipC > 0: an up-path block that first concatenates a skip from the down path.
function resBlock(cin, cout, H, temb, skipC = 0, rep = 1) {
  const c = cin + skipC, s = hw(H), proj = c !== cout;
  const p = 2 * c + 9 * c * cout + cout + (temb ? 1281 * cout : 0) + 2 * cout + 9 * cout * cout + cout + (proj ? c * cout + cout : 0);
  const kids = [
    ...(skipC ? [n('Nối skip', 'shape', `${cin}×${s} + ${skipC}×${s} → ${c}×${s}`, {
      what: R`Nối theo chiều kênh output đang đi lên (${cin} kênh) với bản đồ đặc trưng cùng độ phân giải lấy từ nhánh xuống (${skipC} kênh): \(${cin} + ${skipC} = ${c}\).`,
      formula: [R`h = [\,x;\ e\,] \in \mathbb{R}^{${c} \times ${H} \times ${H}}`],
      why: 'Như U-Net 2015: nhánh xuống giữ chi tiết vị trí, nhánh lên mang ngữ cảnh, và nối để conv phía sau tự học cách trộn. Mỗi tầng của nhánh xuống để lại 3 bản đồ (2 ResBlock và bước giảm kích thước, hoặc conv đầu vào), và mỗi tầng nhánh lên dùng đúng 3 bản đồ đó, mỗi ResBlock một bản.',
      code: `x = torch.cat([x, skips.pop()], dim=1)   # (B, ${c}, ${H}, ${H})`
    }, { tag: 'Nối skip', size: vol(c, H) })] : []),
    gn(c, H), silu(`${c}×${s}`, slab(H)), conv3('Conv 3×3', 'Conv 3×3', c, cout, H),
    ...(temb ? [n('Cộng nhúng t', 'embed', `${cout}×${s}`, {
      what: R`Vector nhúng bước thời gian (1280 chiều) qua SiLU và một Linear \(1280 \to ${cout}\), rồi cộng cùng một số vào mọi vị trí \(${H} \times ${H}\) của mỗi kênh.`,
      formula: [R`h \leftarrow h + \big(W_t\, \mathrm{SiLU}(e_t) + b_t\big)[:, \text{None}, \text{None}], \qquad W_t \in \mathbb{R}^{${cout} \times 1280}`],
      why: 'Cách rẻ nhất để mọi ResBlock biết đang ở bước nào: mỗi kênh được dịch lên hoặc xuống tuỳ t. Đủ để mạng hành xử khác nhau giữa lúc gần như toàn nhiễu và lúc chỉ còn chút nhiễu.',
      params: R`\(1280 \times ${cout} + ${cout} = ${fmt(1281 * cout)}\).`,
      code: `h = h + self.time_emb_proj(F.silu(temb))[:, :, None, None]`
    }, { tag: '+ t', size: [0.3, sp(H), sp(H)] })] : []),
    gn(cout, H), silu(`${cout}×${s}`, slab(H)), conv3('Conv 3×3', 'Conv 3×3', cout, cout, H),
    n('Cộng đường tắt', 'add', `${cout}×${s}`, {
      what: proj ? R`Cộng output nhánh chính với đầu vào đã qua conv 1×1 (\(${c} \to ${cout}\) kênh) cho khớp shape.` : R`Cộng từng phần tử output nhánh chính với chính đầu vào \(x\), như ResNet.`,
      formula: [R`y = F(x) + ${proj ? R`W_s * x` : 'x'}`],
      why: 'Như ResNet: đường tắt giữ cho gradient đi thẳng qua hàng chục khối của U-Net.',
      code: `out = h + ${proj ? 'self.conv_shortcut(x)' : 'x'}`
    }, { tag: 'Cộng', size: BALL, exp: 'res', ref: [ref('resnet', 'mục 3.1–3.2')] })
  ];
  const add = kids.length - 1, g0 = skipC ? 1 : 0;
  if (proj) kids.push(n('Đường tắt: Conv 1×1', 'conv', `${c}×${s} → ${cout}×${s}`, {
    what: R`Chiếu \(x\) từ ${c} sang ${cout} kênh để cộng được với nhánh chính.`,
    formula: [R`\mathrm{shortcut}(x) = W_s * x + b_s, \qquad W_s \in \mathbb{R}^{${cout} \times ${c} \times 1 \times 1}`],
    params: R`\(${c} \times ${cout} + ${cout} = ${fmt(c * cout + cout)}\).`,
    code: `self.conv_shortcut = nn.Conv2d(${c}, ${cout}, kernel_size=1)`
  }, { tag: 'Shortcut 1×1', float: true, size: vol(cout, H) }));
  return n('ResBlock', 'block', `${c}×${s} → ${cout}×${s}`, {
    what: (skipC ? 'Nối skip, rồi ' : '') + `GroupNorm → SiLU → conv 3×3${temb ? ' → cộng nhúng t' : ''} → GroupNorm → SiLU → conv 3×3, cộng đường tắt.`
      + (temb ? '' : ' Khác ResBlock của U-Net khuếch tán: không có nhúng t, vì VAE không biết gì về các bước khử nhiễu.') + (rep > 1 ? ` Lặp ${rep} lần, mỗi lần trọng số riêng.` : ''),
    formula: [R`y = x' + W_2 * \mathrm{SiLU}\Big(\mathrm{GN}\big(W_1 * \mathrm{SiLU}(\mathrm{GN}(x))${temb ? ' + t' : ''}\big)\Big)`],
    why: 'Thứ tự chuẩn hoá → kích hoạt → conv (pre-activation) như ResNet v2. Đây là khối có trọng số nhiều nhất của U-Net: phần lớn tham số nằm trong các conv 3×3 ở 1280 kênh.',
    params: R`\(${fmt(p)}\)${rep > 1 ? R` mỗi khối, \(${fmt(p * rep)}\) cho cả ${rep} khối` : ''}.`,
    code: `h = self.conv1(F.silu(self.norm1(x)))` + (temb ? `\nh = h + self.time_emb_proj(F.silu(temb))[:, :, None, None]` : '') + `
h = self.conv2(F.silu(self.norm2(h)))
return h + ${proj ? 'self.conv_shortcut(x)' : 'x'}`
  }, { tag: 'ResBlock', repeat: rep, size: vol(cout, H), _p: p * rep, children: kids, ref: [ref('ddpm', 'phụ lục B'), ref('resnet2')],
    ...(proj ? { extra: [{ a: { gap: g0 }, b: { i: add + 1, s: 'l' }, kind: 'short', label: 'x' }, { a: { i: add + 1, s: 'r' }, b: { i: add, s: 't' }, kind: 'short', label: 'Wₛx' }] }
      : { skips: [[g0, add, 'x']] }) });
}

// diffusers Attention: q, k, v without bias, output projection with bias; 8 heads. cross: keys and values come from the text.
function attention(cross, C, N) {
  const d = C / 8, kv = cross ? TX : N, kd = cross ? TD : C, p = 2 * C * C + 2 * kd * C + C;
  return n(cross ? 'Cross-Attention' : 'Multi-Head Self-Attention', 'attn', cross ? `Q ${N}×${C} · K,V ${TX}×${TD} → ${N}×${C}` : `${N}×${C} → ${N}×${C}`, {
    what: cross ? R`Query lấy từ ${fmt(N)} vị trí của bản đồ đặc trưng; key và value lấy từ ${TX} token văn bản (768 chiều) của CLIP. Mỗi vị trí trong ảnh tự chọn nên "nghe" từ nào trong câu mô tả. Đây là chỗ duy nhất văn bản đi vào U-Net.`
      : R`Mỗi trong ${fmt(N)} vị trí của bản đồ đặc trưng chú ý tới mọi vị trí khác: hai vùng xa nhau trong ảnh trao đổi thông tin trực tiếp, điều mà conv 3×3 phải chồng rất nhiều lớp mới làm được.`,
    formula: [R`Q = W_Q\, \varphi(z_t), \qquad K = W_K\, ${cross ? R`\tau_\theta(y)` : R`\varphi(z_t)`}, \qquad V = W_V\, ${cross ? R`\tau_\theta(y)` : R`\varphi(z_t)`}`, R`\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\big(QK^\top / \sqrt{${d}}\big)V`],
    why: cross ? 'Bài Latent Diffusion đưa điều kiện vào U-Net bằng đúng cơ chế này: τθ là bộ mã hoá điều kiện (ở Stable Diffusion là CLIP), φ(z_t) là biểu diễn giữa chừng của U-Net. Cùng khung đó dùng được cho điều kiện khác (bản đồ phân vùng, một ảnh khác) chỉ bằng cách đổi τθ. Ma trận attention chỉ cỡ ' + fmt(N).replace(/\\,/g, ' ') + ' × 77, rẻ hơn self-attention nhiều.'
      : `Ma trận attention mỗi đầu cỡ ${fmt(N).replace(/\\,/g, ' ')} × ${fmt(N).replace(/\\,/g, ' ')}${N === 4096 ? ', khoảng 16.8 triệu ô: phần tốn bộ nhớ nhất của U-Net, và là chỗ các cài đặt kiểu FlashAttention giúp nhiều nhất' : ''}.`,
    params: R`\(W_Q\): \(${C}^2\), \(W_K, W_V\): \(2 \cdot ${kd} \cdot ${C}\), \(W_O\) có bias: \(${C}^2 + ${C}\). Cộng \(${fmt(p)}\).`,
    code: `attn = Attention(query_dim=${C}${cross ? ', cross_attention_dim=768' : ''}, heads=8, dim_head=${d})
out = attn(x${cross ? ', encoder_hidden_states=text_emb' : ''})`
  }, { tag: cross ? 'Cross-Attn' : 'Self-Attn', size: [1.6, 1.8, 1.5], ref: [ref('ldm', 'mục 3.3'), ref('vaswani', 'mục 3.2')], children: [
    n('Chiếu Q, K, V', 'linear', cross ? `Q ${N}×${C} · K,V ${TX}×${TD} → ${C}` : `${N}×${C} → 3 × ${N}×${C}`, {
      what: cross ? R`Q chiếu từ ảnh (\(${C} \to ${C}\)); K và V chiếu từ văn bản (\(768 \to ${C}\)), nên văn bản và ảnh gặp nhau trong cùng không gian ${C} chiều.` : 'Ba phép chiếu tuyến tính không bias tạo query, key, value cho mọi vị trí.',
      formula: [cross ? R`Q = X W_Q, \quad K = T W_K, \quad V = T W_V, \qquad T \in \mathbb{R}^{77 \times 768}` : R`Q = X W_Q, \quad K = X W_K, \quad V = X W_V`],
      params: R`\(${C}^2 + 2 \cdot ${kd} \cdot ${C} = ${fmt(C * C + 2 * kd * C)}\).`,
      code: `q = self.to_q(x); k = self.to_k(${cross ? 'text_emb' : 'x'}); v = self.to_v(${cross ? 'text_emb' : 'x'})`
    }, { tag: 'Q, K, V', size: [1.2, 1.5, 1.3] }),
    n('Tách 8 đầu', 'shape', `${N}×${C} → 8×${N}×${d}`, {
      what: `Cắt ${C} chiều thành 8 đầu, mỗi đầu ${d} chiều. Stable Diffusion v1 giữ 8 đầu ở mọi tầng, nên số chiều mỗi đầu tăng theo số kênh (40, 80, 160).`,
      formula: [R`\mathbb{R}^{${N} \times ${C}} \to \mathbb{R}^{8 \times ${N} \times ${d}}`],
      code: `q = q.view(B, ${N}, 8, ${d}).transpose(1, 2)`
    }, { tag: 'Tách đầu', size: [0.6, 1.5, 1.3] }),
    sdpa(cross ? 'cross' : 'self', N, kv, { h: 8, dk: d, noMask: true, noLive: true }),
    n('Nối các đầu', 'shape', `8×${N}×${d} → ${N}×${C}`, {
      what: `Đặt output của 8 đầu cạnh nhau để trở lại ${C} chiều.`,
      formula: [R`\mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_8) \in \mathbb{R}^{${N} \times ${C}}`],
      code: `out = out.transpose(1, 2).reshape(B, ${N}, ${C})`
    }, { tag: 'Nối đầu', size: [0.6, 1.5, 1.3] }),
    n('Chiếu ra Wᴼ', 'linear', `${N}×${C} → ${N}×${C}`, {
      what: 'Một lớp Linear có bias trộn thông tin giữa các đầu.',
      formula: [R`Y = O W_O + b_O, \qquad W_O \in \mathbb{R}^{${C} \times ${C}}`],
      params: R`\(${C}^2 + ${C} = ${fmt(C * C + C)}\).`,
      code: `out = self.to_out[0](out)   # nn.Linear(${C}, ${C})`
    }, { tag: 'Wᴼ', size: [1.2, 1.5, 1.3] })
  ] });
}

function geglu(C, N) {
  const F = 4 * C, p = 12 * C * C + 9 * C;
  return n('Feed-Forward GEGLU', 'linear', `${N}×${C} → ${N}×${C}`, {
    what: R`FFN có cổng như SwiGLU của LLaMA, nhưng cổng dùng GELU: một Linear chiếu lên \(2 \times ${F}\) chiều rồi chia đôi, nửa sau qua GELU làm cổng cho nửa trước, rồi nén về ${C} chiều.`,
    formula: [R`[a;\ g] = xW_1 + b_1, \qquad a, g \in \mathbb{R}^{${F}}`, R`\mathrm{FFN}(x) = \big(a \odot \mathrm{GELU}(g)\big)\, W_2 + b_2`],
    why: 'Shazeer (2020) thử nhiều biến thể FFN có cổng; GEGLU và SwiGLU là hai bản tốt nhất. Lớp ẩn vẫn rộng 4C như FFN thường, nhưng Linear đầu phải ra gấp đôi (8C) cho cả giá trị lẫn cổng.',
    params: R`\(${C} \cdot ${2 * F} + ${2 * F} + ${F} \cdot ${C} + ${C} = ${fmt(p)}\).`,
    code: `a, g = self.proj(x).chunk(2, dim=-1)   # nn.Linear(${C}, ${2 * F})
y = self.out(a * F.gelu(g))            # nn.Linear(${F}, ${C})`
  }, { tag: 'FF (GEGLU)', size: [1.4, 1.7, 1.4], ref: [ref('glu')], children: [
    n(`Linear ${C} → ${2 * F}`, 'linear', `${N}×${C} → ${N}×${2 * F}`, {
      what: `Chiếu lên ${2 * F} chiều, rồi chia thành nhánh giá trị a và nhánh cổng g, mỗi nhánh ${F} chiều.`,
      formula: [R`[a;\ g] = xW_1 + b_1`],
      params: R`\(${C} \cdot ${2 * F} + ${2 * F} = ${fmt(2 * F * C + 2 * F)}\).`,
      code: `a, g = self.proj(x).chunk(2, dim=-1)`
    }, { tag: 'Linear ↑', size: [1.7, 1.7, 1.4] }),
    gelu(`${N}×${F}`, SLAB),
    n('Nhân cổng ⊙', 'act', `${N}×${F}`, {
      what: 'Nhân từng phần tử: nhánh giá trị nhân với cổng.',
      formula: [R`h = a \odot \mathrm{GELU}(g)`],
      code: `h = a * F.gelu(g)`
    }, { tag: '⊙', size: BALL }),
    n(`Linear ${F} → ${C}`, 'linear', `${N}×${F} → ${N}×${C}`, {
      what: `Nén về lại ${C} chiều để cộng vào residual.`,
      formula: [R`y = h W_2 + b_2`],
      params: R`\(${F} \cdot ${C} + ${C} = ${fmt(F * C + C)}\).`,
      code: `y = self.out(h)   # nn.Linear(${F}, ${C})`
    }, { tag: 'Linear ↓', size: [1.0, 1.5, 1.3] })
  ] });
}

// Transformer2DModel: GroupNorm, conv 1×1, flatten to H·W tokens, one BasicTransformerBlock, back to a grid, conv 1×1, residual
function spatialTransformer(C, H) {
  const N = H * H, p = 2 * C + (C * C + C) + 6 * C + (4 * C * C + C) + (2 * C * C + 2 * TD * C + C) + (12 * C * C + 9 * C) + (C * C + C);
  const block = n('Khối Transformer cơ bản', 'block', `${N}×${C} → ${N}×${C}`, {
    what: 'Ba nhánh kiểu Pre-LN, mỗi nhánh có residual riêng: self-attention giữa các vị trí, cross-attention tới văn bản, rồi FFN.',
    formula: [R`h \leftarrow h + \mathrm{SelfAttn}\big(\mathrm{LN}(h)\big)`, R`h \leftarrow h + \mathrm{CrossAttn}\big(\mathrm{LN}(h),\ \tau_\theta(y)\big)`, R`h \leftarrow h + \mathrm{FFN}\big(\mathrm{LN}(h)\big)`],
    why: 'Giống một lớp decoder của Transformer gốc (self-attention, cross-attention, FFN), chỉ khác: "câu nguồn" ở đây là câu mô tả, còn "câu đích" là các vị trí của bản đồ đặc trưng.',
    code: `h = h + self.attn1(self.norm1(h))
h = h + self.attn2(self.norm2(h), encoder_hidden_states=text_emb)
h = h + self.ff(self.norm3(h))`
  }, { tag: 'Transformer', hint: '9 phần', size: [2.0, 2.2, 1.6], ref: [ref('vaswani', 'hình 1, decoder')],
    children: [layerNorm(N, C), attention(false, C, N), addLeaf(N, C), layerNorm(N, C), attention(true, C, N), addLeaf(N, C), layerNorm(N, C), geglu(C, N), addLeaf(N, C)],
    skips: [[0, 2, 'h'], [3, 5, 'h'], [6, 8, 'h']],
    extra: [{ a: { ext: 4, s: 'b', d: 1.5 }, b: { i: 4, s: 'b' }, kind: 'cross', label: 'văn bản 77×768' }] });
  return n('Khối Transformer (ảnh ↔ văn bản)', 'attn', `${C}×${hw(H)} → ${C}×${hw(H)}`, {
    what: R`Duỗi bản đồ \(${C} \times ${H} \times ${H}\) thành chuỗi ${fmt(N)} token ${C} chiều, cho qua một khối Transformer (self-attention giữa các vị trí, cross-attention tới văn bản, FFN), gấp lại thành lưới, rồi cộng residual.`,
    formula: [R`y = x + W_{\text{out}} * \mathrm{Grid}\Big(\mathrm{Block}\big(\mathrm{Seq}(W_{\text{in}} * \mathrm{GN}(x)),\ \tau_\theta(y)\big)\Big)`],
    why: 'ResBlock chỉ nhìn cục bộ và không biết gì về câu mô tả; khối này bù cả hai. Có mặt ở mọi tầng trừ tầng 8×8 cuối của nhánh xuống và tầng 8×8 đầu của nhánh lên (khối giữa ở 8×8 thì có).',
    params: R`\(${fmt(p)}\).`,
    code: `h = self.proj_in(self.norm(x)).flatten(2).transpose(1, 2)   # (B, ${N}, ${C})
h = self.transformer_blocks[0](h, encoder_hidden_states=text_emb)
y = x + self.proj_out(h.transpose(1, 2).view(B, ${C}, ${H}, ${H}))`
  }, { tag: 'Transformer', size: vol(C, H), _p: p, ref: [ref('ldm', 'mục 3.3 và phụ lục E')], children: [
    gn(C, H),
    n('Conv 1×1 vào (proj_in)', 'conv', `${C}×${hw(H)} → ${C}×${hw(H)}`, {
      what: 'Một lớp Linear áp riêng ở từng vị trí, trộn kênh trước khi vào Transformer.',
      formula: [R`h = W_{\text{in}} * x + b, \qquad W_{\text{in}} \in \mathbb{R}^{${C} \times ${C} \times 1 \times 1}`],
      params: R`\(${C}^2 + ${C} = ${fmt(C * C + C)}\).`,
      code: `nn.Conv2d(${C}, ${C}, kernel_size=1)`
    }, { tag: 'Conv 1×1', size: vol(C, H) }),
    n('Duỗi thành chuỗi', 'shape', `${C}×${hw(H)} → ${N}×${C}`, {
      what: R`Lưới \(${H} \times ${H}\) được duỗi thành chuỗi ${fmt(N)} token, mỗi token ${C} chiều.`,
      formula: [R`\mathbb{R}^{${C} \times ${H} \times ${H}} \to \mathbb{R}^{${N} \times ${C}}`],
      code: `h = h.flatten(2).transpose(1, 2)`
    }, { tag: 'Duỗi', size: [0.6, 1.5, 1.3] }),
    block,
    n('Gấp lại thành lưới', 'shape', `${N}×${C} → ${C}×${hw(H)}`, {
      what: R`Ngược với bước duỗi: chuỗi ${fmt(N)} token trở lại lưới \(${H} \times ${H}\).`,
      code: `h = h.transpose(1, 2).view(B, ${C}, ${H}, ${H})`
    }, { tag: 'Gấp lại', size: [0.6, 1.5, 1.3] }),
    n('Conv 1×1 ra (proj_out)', 'conv', `${C}×${hw(H)} → ${C}×${hw(H)}`, {
      what: 'Lớp Linear từng vị trí, trước khi cộng residual.',
      formula: [R`y = W_{\text{out}} * h + b`],
      params: R`\(${C}^2 + ${C} = ${fmt(C * C + C)}\).`,
      code: `nn.Conv2d(${C}, ${C}, kernel_size=1)`
    }, { tag: 'Conv 1×1', size: vol(C, H) }),
    addLeaf(`${C}×${H}`, H)
  ], skips: [[0, 6, 'x']] });
}

// down / up stages of the U-Net
function down(k, cin, C, H, attn, ds) {
  const kids = [resBlock(cin, C, H, true), ...(attn ? [spatialTransformer(C, H)] : []), resBlock(C, C, H, true), ...(attn ? [spatialTransformer(C, H)] : []),
    ...(ds ? [conv3('Conv 3×3, bước 2', 'Conv 3×3 /2', C, C, H, 2)] : [])];
  const p = kids.reduce((t, x) => t + (x._p ?? 9 * C * C + C), 0);
  return n(`Nhánh xuống, tầng ${k}`, 'block', `${cin}×${hw(H)} → ${C}×${hw(ds ? H / 2 : H)}`, {
    what: (attn ? `Hai cặp ResBlock + Transformer ở ${H}×${H}, ${C} kênh` : `Hai ResBlock ở ${H}×${H}, ${C} kênh, không có attention`) + (ds ? `, rồi conv 3×3 bước 2 giảm kích thước còn ${H / 2}×${H / 2}.` : '.')
      + (cin !== C ? ` ResBlock đầu tăng kênh ${cin} → ${C}.` : ''),
    formula: [R`h_1 = ${attn ? R`\mathrm{T}_1\big(\mathrm{Res}_1(x)\big)` : R`\mathrm{Res}_1(x)`}, \qquad h_2 = ${attn ? R`\mathrm{T}_2\big(\mathrm{Res}_2(h_1)\big)` : R`\mathrm{Res}_2(h_1)`}`],
    why: `Khối này để lại ${ds ? 3 : 2} bản đồ cho nhánh lên: output sau mỗi ${attn ? 'cặp ResBlock + Transformer' : 'ResBlock'}${ds ? ', và output của bước giảm kích thước (ở độ phân giải của tầng dưới)' : ''}.` + (attn ? '' : ' Tầng 8×8 cuối bỏ attention; khối giữa ngay sau vẫn có.'),
    params: R`\(${fmt(p)}\) (khoảng ${(p / 1e6).toFixed(1)}M).`,
    code: `x = model.unet.down_blocks[${k - 1}](x, temb, encoder_hidden_states=text_emb)   # trả về x và 3 skip`
  }, { tag: `Xuống ${k}`, hint: `${kids.length} phần`, size: vol(C, ds ? H / 2 : H), _p: p, children: kids });
}

function up(k, prev, C, skips, H, attn, us) {
  const kids = [];
  skips.forEach((sk, j) => { kids.push(resBlock(j ? C : prev, C, H, true, sk)); if (attn) kids.push(spatialTransformer(C, H)); });
  if (us) kids.push(upsample(C, H));
  const p = kids.reduce((t, x) => t + (x._p ?? 9 * C * C + C), 0);
  return n(`Nhánh lên, tầng ${k}`, 'block', `${prev}×${hw(H)} → ${C}×${hw(us ? 2 * H : H)}`, {
    what: `Ba lần (nối 1 skip, ResBlock${attn ? ', Transformer' : ''}) ở ${H}×${H}, ${C} kênh` + (us ? `, rồi phóng to lên ${2 * H}×${2 * H}.` : '.')
      + R` Ba skip nối vào có \(${skips.join(', ')}\) kênh.`,
    formula: [R`h_j = ${attn ? R`\mathrm{T}_j\big(\mathrm{Res}_j([h_{j-1};\ e_j])\big)` : R`\mathrm{Res}_j([h_{j-1};\ e_j])`}, \qquad j = 1, 2, 3`],
    why: 'Nhánh lên có 3 ResBlock mỗi tầng, nhánh xuống chỉ 2: khối thứ ba dùng skip đến từ bước giảm kích thước (hoặc từ conv đầu vào). Tầng này là nơi chi tiết từ nhánh xuống được trộn lại với ngữ cảnh đi lên từ đáy.',
    params: R`\(${fmt(p)}\) (khoảng ${(p / 1e6).toFixed(1)}M).`,
    code: `x = model.unet.up_blocks[${k - 1}](x, skips[-3:], temb, encoder_hidden_states=text_emb)`
  }, { tag: `Lên ${k}`, hint: `${kids.length} phần`, size: vol(C, us ? 2 * H : H), _p: p, children: kids });
}

// ---------- U-Net ----------
const timeEmb = n('Nhúng bước thời gian t', 'embed', 't → 1280', {
  what: 'Biến số bước t (0 đến 999) thành vector 1280 chiều: mã hoá sin/cos 320 chiều giống positional encoding của Transformer, rồi một MLP nhỏ. Vector này được cộng vào mọi ResBlock của U-Net.',
  formula: [R`e_t = W_2\, \mathrm{SiLU}\big(W_1\, \mathrm{sin\text{-}cos}(t) + b_1\big) + b_2, \qquad W_1 \in \mathbb{R}^{1280 \times 320}`],
  why: 'Ở t lớn, latent gần như toàn nhiễu và mạng phải đoán bố cục thô; ở t nhỏ chỉ còn chút nhiễu và mạng chỉ sửa chi tiết. Một mạng duy nhất làm cả 1000 việc khác nhau đó, nên phải được cho biết t.',
  params: R`\(320 \cdot 1280 + 1280 + 1280^2 + 1280 = 2\,050\,560\).`,
  code: `temb = model.unet.time_embedding(model.unet.time_proj(t))   # (B, 1280)`
}, { tag: 'Nhúng t', size: [0.9, 1.3, 1.1], ref: [ref('ddpm', 'phụ lục B'), ref('vaswani', 'mục 3.5')], children: [
  n('Sinusoid của t', 'embed', 't → 320', {
    what: 'Mỗi cặp chiều là cos và sin của t ở một tần số; tần số giảm dần theo cấp số nhân từ 1 xuống 10⁻⁴. Chiều đầu đổi nhanh theo t, chiều cuối đổi rất chậm.',
    formula: [R`\omega_k = 10000^{-k/160}, \qquad k = 0, \dots, 159`, R`e_k = \cos(t\, \omega_k), \qquad e_{160 + k} = \sin(t\, \omega_k)`],
    why: 'Đúng công thức PE của Transformer, chỉ thay vị trí bằng bước thời gian. Hai bước gần nhau cho hai vector gần nhau, nên mạng nội suy được giữa các t.',
    example: R`\(t = 500\): \(e_0 = \cos 500 = -0.884\), \(e_{160} = \sin 500 = -0.468\).`,
    code: `emb = get_timestep_embedding(t, 320, flip_sin_to_cos=True, downscale_freq_shift=0)`
  }, { tag: 'sin/cos', size: [0.5, 1.3, 1.1] }),
  n('Linear 320 → 1280', 'linear', '320 → 1280', {
    what: 'Mở rộng lên 1280 chiều.',
    formula: [R`h = W_1 e + b_1`],
    params: R`\(320 \cdot 1280 + 1280 = 410\,880\).`,
    code: `nn.Linear(320, 1280)`
  }, { tag: 'Linear ↑', size: [1.0, 1.3, 1.1] }),
  silu('1280', [0.14, 1.3, 1.1]),
  n('Linear 1280 → 1280', 'linear', '1280 → 1280', {
    what: 'Lớp thứ hai của MLP nhúng t.',
    formula: [R`e_t = W_2\, \mathrm{SiLU}(h) + b_2`],
    params: R`\(1280^2 + 1280 = 1\,639\,680\).`,
    code: `nn.Linear(1280, 1280)`
  }, { tag: 'Linear', size: [1.0, 1.3, 1.1] })
] });

const D = [down(1, 320, 320, 64, true, true), down(2, 320, 640, 32, true, true), down(3, 640, 1280, 16, true, true), down(4, 1280, 1280, 8, false, false)];
const midKids = [resBlock(1280, 1280, 8, true), spatialTransformer(1280, 8), resBlock(1280, 1280, 8, true)];
const pMid = midKids.reduce((t, x) => t + x._p, 0);
const mid = n('Khối giữa (đáy chữ U)', 'block', '1280×8×8 → 1280×8×8', {
  what: 'ResBlock, Transformer, ResBlock ở độ phân giải nhỏ nhất (8×8, 1280 kênh). Mỗi vị trí ở đây đại diện cho một vùng 64×64 pixel của ảnh 512×512.',
  formula: [R`h = \mathrm{Res}_2\Big(\mathrm{T}\big(\mathrm{Res}_1(x)\big)\Big)`],
  why: 'Ở 8×8 chỉ có 64 vị trí, nên attention giữa mọi vị trí rẻ, và đây là nơi bố cục tổng thể của ảnh được quyết định.',
  params: R`\(${fmt(pMid)}\) (khoảng ${(pMid / 1e6).toFixed(1)}M).`,
  code: `x = model.unet.mid_block(x, temb, encoder_hidden_states=text_emb)`
}, { tag: 'Giữa', hint: '3 phần', size: vol(1280, 8), children: midKids });
const U = [up(1, 1280, 1280, [1280, 1280, 1280], 8, false, true), up(2, 1280, 1280, [1280, 1280, 640], 16, true, true),
  up(3, 1280, 640, [640, 640, 320], 32, true, true), up(4, 640, 320, [320, 320, 320], 64, true, false)];
const head = n('Đầu ra: GroupNorm, SiLU, Conv 3×3', 'conv', '320×64×64 → 4×64×64', {
  what: 'GroupNorm, SiLU, rồi conv 3×3 đưa 320 kênh về 4 kênh: nhiễu dự đoán ε̂, cùng shape với latent.',
  formula: [R`\hat\varepsilon = W * \mathrm{SiLU}\big(\mathrm{GN}(h)\big) + b`],
  params: R`\(640 + 320 \cdot 4 \cdot 9 + 4 = 12\,164\).`,
  code: `eps = self.conv_out(F.silu(self.conv_norm_out(h)))   # (B, 4, 64, 64)`
}, { tag: 'Đầu ra', size: vol(4, 64), children: [gn(320, 64), silu('320×64×64', slab(64)), conv3('Conv 3×3 ra', 'Conv 3×3', 320, 4, 64)] });

const pUnet = 2050560 + 11840 + [...D, ...U].reduce((t, b) => t + b._p, 0) + pMid + 12164;
// self-check: must reproduce the parameter count of diffusers' UNet2DConditionModel with the SD v1.5 config
console.assert(pUnet === 859520964, 'Stable Diffusion U-Net param count mismatch');

const Y = [5.2, 2.6, 0, -2.6, -5.2], X = [-9, -7, -5, -3];
const unet = n('U-Net dự đoán nhiễu', 'block', '4×64×64 + t + 77×768 → 4×64×64', {
  what: 'Nhận latent nhiễu z_t, số bước t và 77 vector văn bản; trả về nhiễu dự đoán ε̂ cùng cỡ 4×64×64. Khung chữ U như U-Net 2015 (nhánh xuống, đáy, nhánh lên, nối tắt ngang), nhưng mỗi tầng gồm ResBlock có cộng nhúng t và khối Transformer có cả self-attention lẫn cross-attention tới văn bản. Conv có đệm nên không phải cắt skip.',
  formula: [R`z_t = \sqrt{\bar\alpha_t}\, z_0 + \sqrt{1 - \bar\alpha_t}\, \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)`, R`\mathcal{L} = \mathbb{E}_{z_0, y, \varepsilon, t}\Big[\big\| \varepsilon - \varepsilon_\theta\big(z_t, t, \tau_\theta(y)\big) \big\|^2\Big]`],
  why: 'Khi train, mỗi bước chọn ngẫu nhiên một t từ 1 đến 1000, làm nhiễu latent của một ảnh thật đúng mức t, rồi bắt U-Net đoán lại đúng ε đã thêm vào. DDPM thấy đoán nhiễu (thay vì đoán thẳng ảnh sạch) với loss bình phương đơn giản này cho mẫu tốt nhất. Khi sinh ảnh, cùng mạng đó được gọi lặp lại từ t lớn về t nhỏ.',
  example: 'Lịch nhiễu của SD (β tăng từ 0.00085 tới 0.012, theo bình phương của một dãy tuyến tính): ᾱ = 0.674 ở t = 250, 0.276 ở t = 500, 0.0047 ở t = 999. Ở t = 999, √ᾱ = 0.068: latent gần như toàn nhiễu.',
  params: R`\(${fmt(pUnet)}\) (khoảng 860M): nhúng t \(2\,050\,560\), nhánh xuống \(${D.map(d => fmt(d._p)).join(' + ')}\), khối giữa \(${fmt(pMid)}\), nhánh lên \(${U.map(u => fmt(u._p)).join(' + ')}\), conv vào và ra \(11\,840 + 12\,164\).`,
  code: `noise_pred = unet(latents, t, encoder_hidden_states=text_emb).sample   # (B, 4, 64, 64)`
}, { tag: 'U-Net', hint: '12 phần · gọi 50 lần', layout: 'manual', size: [2.4, 2.4, 2.0], ref: [ref('ldm', 'mục 3.2–3.3, công thức (2) và (3)'), ref('ddpm', 'mục 3.2–3.4'), ref('unet', 'khung chữ U')], children: [
  { ...timeEmb, pos: [-11.8, -1.6, 0] },
  { ...conv3('Conv 3×3 vào', 'Conv 3×3', 4, 320, 64), pos: [-12.6, Y[0], 0] },
  ...D.map((d, k) => ({ ...d, pos: [X[k], Y[k], 0] })),
  { ...mid, pos: [0, Y[4], 0] },
  ...U.map((u, k) => ({ ...u, pos: [-X[3 - k], Y[3 - k], 0] })),
  { ...head, pos: [12.4, Y[0], 0] }
], links: [
  { a: { stub: 1, s: 'l' }, b: { i: 1, s: 'l' }, label: 'zₜ' },
  { a: { i: 0, s: 'r' }, b: { i: 2, s: 'd', off: [-0.3, 0, 0] }, kind: 'branch', label: 'tới mọi ResBlock', at: 0.2 },
  { a: { i: 1, s: 'r' }, b: { i: 2, s: 'l' } },
  ...[2, 3, 4].map(i => ({ a: { i, s: 'd' }, b: { i: i + 1, s: 'l' } })),
  { a: { i: 5, s: 'd' }, b: { i: 6, s: 'l' } },
  { a: { i: 6, s: 'r' }, b: { i: 7, s: 'd' } },
  ...[7, 8, 9].map(i => ({ a: { i, s: 'r' }, b: { i: i + 1, s: 'd' } })),
  { a: { i: 10, s: 'r' }, b: { i: 11, s: 'l' } },
  { a: { i: 11, s: 'r' }, b: { stub: 11, s: 'r' }, label: 'ε̂' },
  ...[0, 1, 2, 3].map(k => ({ a: { i: 2 + k, s: 'r' }, b: { i: 10 - k, s: 'l' }, kind: 'concat', label: '3 skip' }))
] });

// ---------- CLIP text encoder (ViT-L/14 text tower, causal mask, quick GELU) ----------
const CC = { d: 768, h: 12, dk: 64, dff: 3072, act: 'gelu', drop: 0, noLive: true, L: 12 };
const mlp = ffn('77', CC);
const clipMlp = { ...mlp, info: { ...mlp.info, formula: [R`\mathrm{MLP}(x) = \mathrm{QuickGELU}(x W_1 + b_1)\, W_2 + b_2`, mlp.info.formula[1]], code: `y = self.fc2(quick_gelu(self.fc1(x)))   # 768 → 3072 → 768` },
  children: [mlp.children[0], n('Quick GELU', 'act', '77×3072', {
    what: 'Xấp xỉ GELU bằng sigmoid. CLIP được train với hàm này, nên khi dùng lại trọng số phải dùng đúng nó.',
    formula: [R`\mathrm{QuickGELU}(x) = x\,\sigma(1.702\,x)`],
    example: R`\(\mathrm{QuickGELU}([-1.5,\ 0.3,\ 2]) = [-0.108,\ 0.187,\ 1.936]\); GELU thật cho \([-0.100,\ 0.185,\ 1.954]\).`,
    code: `x * torch.sigmoid(1.702 * x)`
  }, { size: SLAB }), mlp.children[2]] };
const pClipLayer = 4 * 768 + (4 * 768 * 768 + 4 * 768) + (2 * 768 * 3072 + 3072 + 768);
const pClip = 49408 * 768 + 77 * 768 + 12 * pClipLayer + 2 * 768;
console.assert(pClip === 123060480, 'CLIP text encoder param count mismatch');
const clip = n('CLIP text encoder', 'block', '77 token → 77×768', {
  what: 'Encoder văn bản của CLIP ViT-L/14, đóng băng khi train Stable Diffusion. Stable Diffusion lấy cả 77 vector ở lớp cuối (sau LayerNorm) làm điều kiện cho cross-attention, chứ không chỉ vector của token kết thúc như khi CLIP so ảnh với chữ.',
  formula: [R`\tau_\theta(y) = \mathrm{LN}\big(\mathrm{Encoder}_{12}(E_{\text{tok}}[y] + E_{\text{pos}})\big) \in \mathbb{R}^{77 \times 768}`],
  why: 'CLIP được train để vector của một câu mô tả và vector của ảnh tương ứng gần nhau (400 triệu cặp ảnh và chữ từ internet), nên biểu diễn của nó đã gắn chặt với nội dung hình ảnh: một lựa chọn tự nhiên để điều khiển việc sinh ảnh.',
  params: R`\(${fmt(pClip)}\): token embedding \(49\,408 \times 768 = 37\,945\,344\), vị trí \(77 \times 768 = 59\,136\), 12 lớp × \(${fmt(pClipLayer)}\), LayerNorm cuối \(1\,536\).`,
  code: `text_emb = text_encoder(ids).last_hidden_state   # (B, 77, 768)`
}, { tag: 'CLIP', hint: '4 phần', size: [1.8, 2.0, 1.5], ref: [ref('clip', 'mục 2.4'), ref('sd15')], children: [
  n('Token Embedding', 'embed', '77 → 77×768', {
    what: 'Tra bảng: mỗi id token (trong từ điển 49 408) lấy ra một vector 768 chiều.',
    formula: [R`x_i = E_{\text{tok}}[y_i], \qquad E_{\text{tok}} \in \mathbb{R}^{49408 \times 768}`],
    params: R`\(49\,408 \times 768 = 37\,945\,344\).`,
    code: `x = self.token_embedding(ids)   # nn.Embedding(49408, 768)`
  }, { tag: 'Token Emb', size: [1.2, 1.5, 1.3] }),
  n('Cộng vị trí (PE học được)', 'embed', '77×768', {
    what: 'Cộng một vector vị trí học được cho mỗi trong 77 vị trí, như ViT và GPT-2.',
    formula: [R`x_i \leftarrow x_i + E_{\text{pos}}[i], \qquad E_{\text{pos}} \in \mathbb{R}^{77 \times 768}`],
    params: R`\(77 \times 768 = 59\,136\).`,
    code: `x = x + self.position_embedding(torch.arange(77))`
  }, { tag: 'Vị trí (PE)', size: [0.9, 1.5, 1.3] }),
  n('Encoder', 'block', '77×768 → 77×768', {
    what: '12 lớp Pre-LN có mặt nạ nhân quả: token ở vị trí i chỉ nhìn được các token 1 đến i, như GPT. MLP dùng quick GELU.',
    formula: [R`h = x + \mathrm{MaskedMHA}\big(\mathrm{LN}(x)\big)`, R`x' = h + \mathrm{MLP}\big(\mathrm{LN}(h)\big)`],
    why: 'Mặt nạ nhân quả ở một encoder là lựa chọn của CLIP: bài gốc giữ nó để còn có thể khởi tạo từ một mô hình ngôn ngữ có sẵn hoặc thêm mục tiêu mô hình hoá ngôn ngữ. Hệ quả cho Stable Diffusion: vector của một từ chỉ mang thông tin của các từ đứng trước nó.',
    params: R`Mỗi lớp \(${fmt(pClipLayer)}\), cả 12 lớp \(${fmt(12 * pClipLayer)}\).`,
    code: `h = x + self.self_attn(self.layer_norm1(x), causal_attention_mask=causal)
x = h + self.mlp(self.layer_norm2(h))`
  }, { repeat: 12, hint: '6 phần · 12 lớp', size: [2.0, 2.2, 1.6], ref: [ref('clip', 'mục 2.4')],
    children: [layerNorm('77', 768), mha('masked', '77', '77', CC), addLeaf('77', 768), layerNorm('77', 768), clipMlp, addLeaf('77', 768)],
    skips: [[0, 2, 'x'], [3, 5, 'h']] }),
  { ...layerNorm('77', 768), name: 'LayerNorm cuối' }
] });

// ---------- VAE decoder (KL-f8) ----------
const vMid = [resBlock(512, 512, 64, false), n('Self-attention (1 đầu)', 'attn', '512×64×64 → 512×64×64', {
  what: 'GroupNorm, rồi self-attention một đầu giữa 4096 vị trí của lưới 64×64 (512 chiều), cộng residual.',
  formula: [R`y = x + W_O\, \mathrm{softmax}\big(QK^\top / \sqrt{512}\big)V, \qquad Q, K, V = \mathrm{GN}(x)\, W_{Q,K,V}`],
  why: 'Lớp attention duy nhất của decoder, đặt ở độ phân giải nhỏ nhất của nó: cho mỗi vị trí nhìn toàn latent trước khi phóng to.',
  params: R`GroupNorm \(1\,024\), 4 Linear có bias \(4 \times (512^2 + 512) = 1\,050\,624\). Cộng \(1\,051\,648\).`,
  code: `y = x + self.to_out(attention(self.group_norm(x)))   # Attention(512, heads=1)`
}, { tag: 'Attention', size: vol(512, 64), _p: 1051648 }), resBlock(512, 512, 64, false)];
const vUp = [[512, 512, 64, true], [512, 512, 128, true], [512, 256, 256, true], [256, 128, 512, false]].map(([cin, C, H, us], k) => {
  const kids = [resBlock(cin, C, H, false), resBlock(C, C, H, false, 0, 2), ...(us ? [upsample(C, H)] : [])];
  const p = kids.reduce((t, x) => t + (x._p ?? 9 * C * C + C), 0);
  return n(`Tầng lên ${k + 1}`, 'block', `${cin}×${hw(H)} → ${C}×${hw(us ? 2 * H : H)}`, {
    what: `3 ResBlock ở ${H}×${H}, ${C} kênh` + (us ? `, rồi phóng to lên ${2 * H}×${2 * H}.` : '.') + (cin !== C ? ` ResBlock đầu giảm kênh ${cin} → ${C}.` : ''),
    formula: [R`h = ${us ? R`\mathrm{Up}\big(` : ''}\mathrm{Res}_3(\mathrm{Res}_2(\mathrm{Res}_1(x)))${us ? R`\big)` : ''}`],
    params: R`\(${fmt(p)}\).`,
    code: `x = vae.decoder.up_blocks[${k}](x)`
  }, { tag: `Lên ${k + 1}`, size: vol(C, us ? 2 * H : H), _p: p, children: kids });
});
const pDec = 20 + (9 * 4 * 512 + 512) + vMid.reduce((t, x) => t + x._p, 0) + vUp.reduce((t, u) => t + u._p, 0) + (256 + 9 * 128 * 3 + 3);
// self-check: post-quant conv + decoder of diffusers' AutoencoderKL with the SD config (49 490 199 parameters)
console.assert(pDec === 49490199, 'VAE decoder param count mismatch');
const vae = n('VAE decoder', 'block', '4×64×64 → 3×512×512', {
  what: 'Giải mã latent 4×64×64 thành ảnh 512×512: ResBlock xen kẽ phóng to ×2 ba lần (64 → 128 → 256 → 512), số kênh giảm dần 512 → 128. Chỉ chạy một lần, sau khi vòng khử nhiễu kết thúc.',
  formula: [R`\hat x = \mathcal{D}\big(z_0 / 0.18215\big)`, R`z = \mathcal{E}(x), \qquad \mathcal{D}(z) \approx x, \qquad f = H / h = 8`],
  why: 'VAE được train riêng từ trước (loss tái tạo, loss cảm nhận, loss đối kháng theo từng mảng ảnh, cộng một phạt KL rất nhẹ kéo latent về gần N(0, I)), rồi đóng băng. Nén 8 lần mỗi chiều nhưng giữ được chi tiết mà mắt người để ý, nên mô hình khuếch tán khỏi phải học những chi tiết đó. Encoder của VAE (34M tham số) chỉ dùng khi train và khi sửa ảnh có sẵn.',
  example: '512 × 512 × 3 = 786 432 số → 64 × 64 × 4 = 16 384 số: latent nhỏ hơn ảnh 48 lần.',
  params: R`\(${fmt(pDec)}\) (gồm conv 1×1 post-quant 20 tham số). Cả VAE, tính encoder: \(83\,653\,863\).`,
  code: `image = vae.decode(latents / 0.18215).sample   # (B, 3, 512, 512), khoảng [-1, 1]`
}, { tag: 'VAE decoder', hint: '9 phần', size: [1.6, 2.0, 1.6], ref: [ref('ldm', 'mục 3.1'), ref('vae')], children: [
  n('Bỏ hệ số tỉ lệ', 'norm', '4×64×64', {
    what: 'Khi train, latent của VAE được nhân 0.18215 để có phương sai cỡ 1, hợp với nhiễu N(0, I) của khuếch tán. Trước khi giải mã thì chia lại.',
    formula: [R`z \leftarrow z / 0.18215`],
    code: `latents = latents / vae.config.scaling_factor   # 0.18215`
  }, { tag: '÷ 0.18215', size: slab(64) }),
  n('Conv 1×1 (post-quant)', 'conv', '4×64×64 → 4×64×64', {
    what: 'Một conv 1×1 trộn 4 kênh của latent trước khi giải mã.',
    formula: [R`z' = W * z + b, \qquad W \in \mathbb{R}^{4 \times 4 \times 1 \times 1}`],
    params: R`\(4 \times 4 + 4 = 20\).`,
    code: `z = vae.post_quant_conv(z)`
  }, { tag: 'Conv 1×1', size: vol(4, 64) }),
  conv3('Conv 3×3 vào', 'Conv 3×3', 4, 512, 64),
  n('Khối giữa', 'block', '512×64×64 → 512×64×64', {
    what: 'ResBlock, self-attention, ResBlock ở 64×64, 512 kênh.',
    formula: [R`h = \mathrm{Res}_2\big(\mathrm{Attn}(\mathrm{Res}_1(x))\big)`],
    params: R`\(${fmt(vMid.reduce((t, x) => t + x._p, 0))}\).`,
    code: `x = vae.decoder.mid_block(x)`
  }, { tag: 'Giữa', size: vol(512, 64), children: vMid }),
  ...vUp,
  n('Đầu ra: GroupNorm, SiLU, Conv 3×3', 'conv', '128×512×512 → 3×512×512', {
    what: 'GroupNorm, SiLU, rồi conv 3×3 đưa 128 kênh về 3 kênh màu.',
    formula: [R`\hat x = W * \mathrm{SiLU}\big(\mathrm{GN}(h)\big) + b`],
    params: R`\(256 + 128 \cdot 3 \cdot 9 + 3 = 3\,715\).`,
    code: `x = vae.decoder.conv_out(F.silu(vae.decoder.conv_norm_out(x)))`
  }, { tag: 'Đầu ra', size: [0.06, sp(512), sp(512)], children: [gn(128, 512), silu('128×512×512', slab(512)), conv3('Conv 3×3 ra', 'Conv 3×3', 128, 3, 512)] })
] });

// ---------- the whole pipeline ----------
MODELS.sd = n('Stable Diffusion v1.5', 'block', '"prompt" → 3×512×512', {
  what: 'Sinh ảnh từ câu mô tả bằng khuếch tán trong không gian latent. CLIP biến câu mô tả thành 77 vector. Bắt đầu từ nhiễu Gauss 4×64×64, U-Net đoán phần nhiễu trong latent hiện tại (có nhìn câu mô tả qua cross-attention), bộ lập lịch bớt đi một ít nhiễu, và lặp lại 50 lần. Cuối cùng VAE giải mã latent 64×64 thành ảnh 512×512.',
  formula: [R`z_T \sim \mathcal{N}(0, I) \;\to\; z_{981} \to z_{961} \to \dots \to z_1 \to z_0 \;\xrightarrow{\ \mathcal{D}\ }\; \hat x`, R`z_{t'} = \mathrm{Step}\big(z_t,\ \hat\varepsilon_\theta(z_t, t, \tau_\theta(y))\big)`],
  why: 'Khuếch tán trực tiếp trên pixel 512×512 thì mỗi bước U-Net rất đắt. Latent Diffusion tách việc: VAE lo nén ảnh và giữ chi tiết mắt người thấy, mô hình khuếch tán chỉ học phần "nội dung" trong một không gian nhỏ hơn 48 lần. Bài gốc thấy nén hệ số 4 đến 16 cân bằng tốt nhất: trên ImageNet có điều kiện lớp, sau cùng 2 triệu bước train, mô hình nén 8 lần đạt FID tốt hơn mô hình chạy trên pixel 38 điểm.',
  example: 'Theo model card: v1.5 tiếp tục từ v1.2, train thêm 595 000 bước ở 512×512 trên tập laion-aesthetics v2 5+, batch 2048, trên 256 GPU A100, bỏ câu mô tả 10% số lần để dùng được classifier-free guidance.',
  params: R`Khi sinh ảnh: U-Net \(${fmt(pUnet)}\), CLIP text encoder \(${fmt(pClip)}\), VAE decoder \(${fmt(pDec)}\). Cộng \(${fmt(pUnet + pClip + pDec)}\), khoảng 1.03 tỷ.`,
  code: `from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5",
                                               torch_dtype=torch.float16).to("cuda")
image = pipe("a photo of a cat", num_inference_steps=50, guidance_scale=7.5).images[0]`
}, { label: 'Stable Diffusion', layout: 'manual', ref: [ref('ldm', 'mục 3 và 4.1'), ref('sd15'), ref('ddpm'), ref('clip')], children: [
  { ...n('Câu mô tả (prompt)', 'io', '"a photo of a cat" → 77 token', {
    what: 'Câu mô tả được tách thành token bằng BPE của CLIP (từ điển 49 408 token), thêm token mở đầu và kết thúc, rồi cắt bớt hoặc đệm cho đủ 77 token.',
    formula: [R`y = [\langle\text{bos}\rangle, w_1, \dots, w_n, \langle\text{eos}\rangle, \langle\text{pad}\rangle, \dots] \in \{0, \dots, 49407\}^{77}`],
    why: 'Mọi câu dài như nhau (77) để ghép batch được. Câu dài hơn bị cắt: phần mô tả sau khoảng 75 từ con không bao giờ tới được U-Net. Câu rỗng "" cũng được mã hoá như vậy, dùng cho nhánh không điều kiện của classifier-free guidance.',
    example: '"a photo of a cat": token mở đầu, 5 từ, token kết thúc là 7 token; 70 vị trí còn lại là token đệm.',
    code: `ids = tokenizer("a photo of a cat", padding="max_length", max_length=77,
                truncation=True, return_tensors="pt").input_ids   # (1, 77)`
  }, { tag: 'Prompt', size: [0.5, 1.4, 0.5] }), pos: [-9.6, 5.0, 0] },
  { ...clip, pos: [-5.0, 5.0, 0] },
  { ...n('Latent nhiễu z_T', 'io', '4×64×64', {
    what: 'Điểm bắt đầu: một tensor nhiễu Gauss 4×64×64, chưa phải ảnh. Cùng seed, cùng câu mô tả và cùng cài đặt thì ra cùng một ảnh.',
    formula: [R`z_T \sim \mathcal{N}(0, I)`],
    why: 'Khuếch tán chạy trong không gian latent của VAE (64×64×4) chứ không trên ảnh 512×512×3, nên mỗi bước U-Net rẻ hơn nhiều.',
    code: `latents = torch.randn(1, 4, 64, 64, generator=torch.Generator().manual_seed(0))`
  }, { tag: 'z_T', size: [0.06, sp(64), sp(64)], live: 'diff', diff: { at: 'start' } }), pos: [-9.6, 0, 0] },
  { ...unet, pos: [-5.0, 0, 0] },
  { ...n('Classifier-free guidance', 'act', '2 × 4×64×64 → 4×64×64', {
    what: 'Mỗi bước, U-Net chạy hai lần (gộp thành một batch 2): một lần với câu mô tả, một lần với câu rỗng. Lấy dự đoán có điều kiện rồi đẩy nó ra xa dự đoán không điều kiện theo hệ số s.',
    formula: [R`\hat\varepsilon = \varepsilon_\theta(z_t, t, \varnothing) + s\,\big(\varepsilon_\theta(z_t, t, y) - \varepsilon_\theta(z_t, t, \varnothing)\big), \qquad s = 7.5`],
    why: 'Hiệu giữa hai dự đoán là "phần do câu mô tả đóng góp"; khuếch đại nó s lần làm ảnh bám câu mô tả chặt hơn, đổi lại kém đa dạng, và s quá lớn làm màu bị cháy. Muốn có dự đoán không điều kiện thì lúc train phải thỉnh thoảng bỏ câu mô tả (SD v1.5: 10% số lần). Gọi là classifier-free vì cách cũ (classifier guidance) cần thêm một bộ phân loại được train riêng trên ảnh nhiễu. Bài gốc viết \\((1 + w)\\varepsilon_c - w\\,\\varepsilon_u\\), tức \\(s = 1 + w\\).',
    example: R`Ở một phần tử: \(\varepsilon_u = 0.2\), \(\varepsilon_c = 0.3\), \(s = 7.5\) → \(0.2 + 7.5 \cdot 0.1 = 0.95\).`,
    code: `eps_u, eps_c = unet(torch.cat([latents] * 2), t, encoder_hidden_states=torch.cat([uncond_emb, text_emb])).sample.chunk(2)
noise_pred = eps_u + 7.5 * (eps_c - eps_u)`
  }, { tag: 'Guidance', size: [0.4, 1.5, 1.3], ref: [ref('cfg'), ref('sd15', '10% bỏ câu mô tả')] }), pos: [-0.9, 0, 0] },
  { ...n('Bước khử nhiễu (scheduler)', 'act', '4×64×64 → 4×64×64', {
    what: 'Dùng nhiễu dự đoán ε̂ để đi từ z_t sang z_{t′} ít nhiễu hơn: ước lượng latent sạch ẑ₀, rồi trộn lại với lượng nhiễu nhỏ hơn. 50 bước, t = 981, 961, …, 1.',
    formula: [R`\hat z_0 = \frac{z_t - \sqrt{1 - \bar\alpha_t}\, \hat\varepsilon}{\sqrt{\bar\alpha_t}}`, R`z_{t'} = \sqrt{\bar\alpha_{t'}}\, \hat z_0 + \sqrt{1 - \bar\alpha_{t'}}\, \hat\varepsilon \qquad \text{(DDIM, } \eta = 0)`, R`\bar\alpha_t = \prod_{s=1}^{t} (1 - \beta_s), \qquad \beta_s \in [0.00085,\ 0.012]`],
    why: 'Mô hình được train với 1000 bước, nhưng DDIM cho phép nhảy cóc: mỗi bước dùng ε̂ để đoán thẳng ẑ₀ rồi "thêm nhiễu ngược" về mức t′ nhỏ hơn, không phải đi qua từng bước một. Pipeline mặc định của SD v1.5 dùng PNDM, một bộ giải nhiều bước cùng họ; đổi bộ lập lịch (DDIM, Euler, DPM-Solver) không cần train lại mạng.',
    example: R`\(\bar\alpha_{501} = 0.275\), \(\bar\alpha_{481} = 0.302\). Một phần tử \(z_t = 0.8\), \(\hat\varepsilon = 0.5\): \(\hat z_0 = (0.8 - 0.8515 \cdot 0.5) / 0.5244 = 0.714\), rồi \(z_{481} = 0.5498 \cdot 0.714 + 0.8353 \cdot 0.5 = 0.810\).`,
    code: `scheduler.set_timesteps(50)              # t = 981, 961, …, 1
latents = scheduler.step(noise_pred, t, latents).prev_sample`
  }, { tag: 'Scheduler', size: [0.6, 1.5, 1.3], ref: [ref('ddim', 'công thức (12), σ = 0'), ref('ddpm')], live: 'diff', diff: { at: 'now' } }), pos: [2.4, 0, 0] },
  { ...vae, pos: [6.0, 0, 0] },
  { ...n('Ảnh ra', 'io', '3×512×512', {
    what: 'Output của decoder nằm trong khoảng xấp xỉ [−1, 1]; đưa về [0, 255] để lưu thành ảnh.',
    formula: [R`\text{pixel} = 255 \cdot \mathrm{clip}\Big(\frac{\hat x + 1}{2},\ 0,\ 1\Big)`],
    code: `image = ((image / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)`
  }, { tag: 'Ảnh', size: [0.06, sp(512), sp(512)], live: 'diff', diff: { at: 'end' } }), pos: [9.6, 0, 0] }
], links: [
  { a: { i: 0, s: 'r' }, b: { i: 1, s: 'l' } },
  { a: { i: 2, s: 'r' }, b: { i: 3, s: 'l' } },
  { a: { i: 1, s: 'd' }, b: { i: 3, s: 't', off: [-0.5, 0, 0] }, kind: 'cross', label: '77×768' },
  { a: { i: 3, s: 'r' }, b: { i: 4, s: 'l' }, label: 'ε̂ ×2' },
  { a: { i: 4, s: 'r' }, b: { i: 5, s: 'l' } },
  { a: { i: 5, s: 'b' }, b: { i: 3, s: 'b', off: [0.6, 0, 0] }, kind: 'loop', label: 'lặp 50 bước, t giảm dần' }, // behind the row: the scheduler's top carries the toy diffusion
  { a: { i: 5, s: 'r' }, b: { i: 6, s: 'l' }, label: 'z₀' },
  { a: { i: 6, s: 'r' }, b: { i: 7, s: 'l' } }
] });
})();
