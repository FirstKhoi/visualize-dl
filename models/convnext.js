// ConvNeXt-T (Liu et al., 2022) as in torchvision convnext_tiny: a patchify stem, stages of (3, 3, 9, 3) blocks with (96, 192, 384, 768)
// channels, blocks of depthwise 7×7 → LayerNorm → Linear to 4C → GELU → Linear back → layer scale, and LayerNorm + 2×2 conv between stages.
(() => {
const DEPTHS = [3, 3, 9, 3], DIMS = [96, 192, 384, 768], NB = DEPTHS.reduce((a, b) => a + b);
const pBlock = C => 8 * C * C + 58 * C; // depthwise 7×7 with bias, LayerNorm, Linear C → 4C, Linear 4C → C, γ
const pDown = C => 8 * C * C + 4 * C; // LayerNorm(C) + conv 2×2, C → 2C
const g = H => `${H}×${H}`;
const slab = H => [0.14, sp(H), sp(H)];
const sdp = i => +(0.1 * i / (NB - 1)).toFixed(4); // stochastic depth of block i (torchvision: 0 → 0.1 over the 18 blocks)

// `rep` identical blocks starting at block index i0
function block(C, H, i0, rep) {
  const s = g(H), F = 4 * C;
  const ln = layerNorm(s, C);
  const kids = [
    n('Depthwise 7×7', 'conv', `${C}×${s} → ${C}×${s}`, {
      what: `Mỗi kênh một bộ lọc 7×7 riêng, đệm 3: nhìn vùng 7×7 quanh mỗi điểm và không trộn kênh nào với kênh nào. Đây là phần duy nhất của khối nhìn sang lân cận, cùng vai trò với self-attention trong cửa sổ 7×7 của Swin.`,
      formula: [R`y_{c,i,j} = b_c + \sum_{u=0}^{6} \sum_{v=0}^{6} W_{c,u,v}\; x_{c,\; i + u - 3,\; j + v - 3}`],
      why: 'Hai bước trong lộ trình của bài. Dời conv depthwise lên đầu khối, để phần đắt (kernel lớn) chạy ở C kênh còn các Linear hiệu quả làm việc nặng ở 4C: tạm tụt còn 79.9%. Rồi tăng kernel 3×3 → 7×7: 80.6%. Lớn hơn nữa không giúp gì (9×9: 80.6%, 11×11: 80.5%). Depthwise rẻ nên kernel lớn vẫn vừa sức.',
      example: R`Ở đây: conv 7×7 thường \(${C} \to ${C}\) kênh cần \(49 \cdot ${C} \cdot ${C} = ${fmt(49 * C * C)}\) tham số; depthwise chỉ \(49 \cdot ${C} + ${C} = ${fmt(50 * C)}\).`,
      params: R`\(${C} \times 7 \times 7 + ${C} = ${fmt(50 * C)}\).`,
      code: `nn.Conv2d(${C}, ${C}, kernel_size=7, padding=3, groups=${C})`
    }, { tag: 'DW 7×7', size: vol(C, H), ref: [ref('mobilenet', 'mục 3.1, depthwise')] }),
    { ...ln, info: { ...ln.info, why: R`Ở ResNet, mỗi conv có một BatchNorm theo sau. ConvNeXt chỉ giữ một lớp chuẩn hoá mỗi khối, ngay trước Linear mở rộng như Transformer chuẩn hoá trước mỗi nhánh (81.4%), rồi đổi BatchNorm thành LayerNorm (81.5%). LN ở đây chuẩn hoá vector ${C} kênh của từng điểm ảnh, nên tensor được đổi sang dạng kênh sau \((N, H, W, C)\) ngay trước bước này.` } },
    n(`Linear ${C} → ${F}`, 'linear', `${s}×${C} → ${s}×${F}`, {
      what: `Conv 1×1 viết bằng một Linear trên tensor kênh sau: mở rộng ${C} kênh lên ${F} ở từng điểm ảnh.`,
      formula: [R`h = x W_1 + b_1, \qquad W_1 \in \mathbb{R}^{${C} \times ${F}}`],
      why: 'Cổ chai ngược, như MLP của Transformer (hệ số 4) và MBConv của EfficientNet: hẹp → rộng → hẹp, ngược với bottleneck của ResNet. Trong lộ trình: 80.5% → 80.6%, và còn bớt FLOPs từ 5.3G xuống 4.6G, vì conv 1×1 trên đường tắt của các khối giảm độ phân giải rẻ đi nhiều.',
      params: R`\(${C} \cdot ${F} + ${F} = ${fmt(F * C + F)}\).`,
      code: `nn.Linear(${C}, ${F})`
    }, { tag: 'Linear ↑', size: [1.7, 1.7, 1.4], ref: [ref('mobilenetv2', 'inverted residual')] }),
    gelu(`${s}×${F}`, slab(H)),
    n(`Linear ${F} → ${C}`, 'linear', `${s}×${F} → ${s}×${C}`, {
      what: `Nén ${F} kênh về lại ${C} để cộng được vào đường tắt.`,
      formula: [R`y = \mathrm{GELU}(h)\, W_2 + b_2, \qquad W_2 \in \mathbb{R}^{${F} \times ${C}}`],
      params: R`\(${F} \cdot ${C} + ${C} = ${fmt(F * C + C)}\).`,
      code: `nn.Linear(${F}, ${C})`
    }, { tag: 'Linear ↓', size: [1.0, 1.5, 1.3] }),
    n('Layer scale γ', 'norm', `${C}×${s}`, {
      what: R`Nhân output của nhánh với một vector \(\gamma \in \mathbb{R}^{${C}}\) học được, mỗi kênh một số, khởi tạo bằng \(10^{-6}\).`,
      formula: [R`F(x) \leftarrow \gamma \odot F(x), \qquad \gamma_c = 10^{-6} \text{ lúc khởi tạo}`],
      why: 'Lúc đầu nhánh gần như tắt, nên mỗi khối gần là hàm đồng nhất và mạng sâu bắt đầu train ổn định; γ lớn dần trong lúc học. Cùng ý tưởng với mẹo khởi tạo γ = 0 cho BN cuối của bottleneck ResNet. Kỹ thuật này lấy từ CaiT, một ViT rất sâu. ConvNeXt V2 bỏ nó vì GRN đã lo phần ổn định.',
      params: R`\(${C}\).`,
      code: `result = self.layer_scale * self.block(x)   # layer_scale: nn.Parameter (${C}, 1, 1), 1e-6`
    }, { tag: 'γ', size: slab(H), ref: [ref('cait', 'layer scale')] }),
    n('Stochastic depth + cộng', 'add', `${C}×${s}`, {
      what: R`Cộng đường tắt như ResNet. Khi train, với xác suất \(p\), cả nhánh của khối bị bỏ cho từng ảnh trong batch.`,
      formula: [R`y = x + \frac{b}{1 - p}\, \gamma \odot F(x), \qquad b \sim \mathrm{Bernoulli}(1 - p)`],
      why: 'Như EfficientNet: mỗi bước train là một mạng nông hơn chọn ngẫu nhiên. ConvNeXt-T cho p tăng tuyến tính từ 0 ở khối đầu tới 0.1 ở khối thứ 18.',
      example: rep > 1 ? R`Các khối ở đây là khối thứ ${i0} đến ${i0 + rep - 1} (đếm từ 0): \(p\) từ \(${sdp(i0)}\) đến \(${sdp(i0 + rep - 1)}\).` : R`\(p = ${sdp(i0)}\).`,
      code: `result = self.stochastic_depth(result)   # StochasticDepth(p, "row")
return result + x`
    }, { tag: 'Cộng + SD', size: BALL, exp: 'res', ref: [ref('stochdepth')] })
  ];
  return n('Khối ConvNeXt', 'block', `${C}×${s} → ${C}×${s}`, {
    what: R`Conv depthwise 7×7 trộn thông tin giữa các vị trí; LayerNorm; một MLP \(${C} \to ${F} \to ${C}\) với một GELU xử lý từng vị trí; layer scale; cộng đường tắt.${rep > 1 ? ` Lặp ${rep} lần, mỗi lần trọng số riêng.` : ''}`,
    formula: [R`y = x + \gamma \odot W_2\, \mathrm{GELU}\Big(W_1\, \mathrm{LN}\big(\mathrm{DW}_{7\times7}(x)\big)\Big)`],
    why: 'Cùng bố cục với một khối Swin: một phần trộn không gian (conv depthwise 7×7 thay cho attention trong cửa sổ 7×7) rồi một MLP rộng 4C. Khác ở chỗ gọn hơn hẳn: một đường tắt, một lớp chuẩn hoá, một hàm kích hoạt. Riêng việc chỉ giữ một GELU, đặt giữa hai Linear như Transformer, đưa độ chính xác từ 80.6% lên 81.3%.',
    params: R`\(${fmt(pBlock(C))}\) mỗi khối${rep > 1 ? R`, \(${fmt(rep * pBlock(C))}\) cho cả ${rep} khối` : ''}: depthwise \(${fmt(50 * C)}\), LayerNorm \(${fmt(2 * C)}\), hai Linear \(${fmt(8 * C * C + 5 * C)}\), γ \(${C}\).`,
    code: `x = x.permute(0, 2, 3, 1)   # sau dwconv: (N, C, H, W) → (N, H, W, C)
result = layer_scale * pwconv2(gelu(pwconv1(norm(dwconv(x)))))
return x + stochastic_depth(result)`
  }, { tag: 'Khối ConvNeXt', repeat: rep, hint: '7 phần', size: vol(C, H), ref: [ref('convnext', 'mục 2, hình 4')], children: kids, skips: [[0, 6, 'x']] });
}

const down = (C, H) => n('Giảm độ phân giải: LN + Conv 2×2', 'conv', `${C}×${g(H)} → ${2 * C}×${g(H / 2)}`, {
  what: R`LayerNorm rồi conv 2×2 bước 2: gộp mỗi ô 2×2 thành một điểm và gấp đôi số kênh (\(${C} \to ${2 * C}\)), lưới \(${H} \to ${H / 2}\). Gần như Patch Merging của Swin (gom 2×2 rồi Linear).`,
  formula: [R`y = W_{2\times2} * \mathrm{LN}(x), \qquad \text{bước } 2`],
  why: 'ResNet giảm kích thước ngay trong khối residual đầu mỗi tầng (conv 3×3 bước 2, thêm conv 1×1 bước 2 ở đường tắt). ConvNeXt tách việc đó ra một lớp riêng giữa các tầng: 81.5% → 82.0%. Bài báo thấy làm vậy mà không có LayerNorm thì train bị phân kỳ, nên thêm LN ở mọi chỗ đổi độ phân giải.',
  params: R`LayerNorm \(${fmt(2 * C)}\), conv \(${C} \cdot ${2 * C} \cdot 2 \cdot 2 + ${2 * C} = ${fmt(8 * C * C + 2 * C)}\). Cộng \(${fmt(pDown(C))}\).`,
  code: `nn.Sequential(LayerNorm2d(${C}), nn.Conv2d(${C}, ${2 * C}, kernel_size=2, stride=2))`
}, { tag: 'LN + Conv 2×2', size: vol(2 * C, H / 2), ref: [ref('convnext', 'mục 2.6, separate downsampling')], children: [
  layerNorm(g(H), C),
  n('Conv 2×2, bước 2', 'conv', `${C}×${g(H)} → ${2 * C}×${g(H / 2)}`, {
    what: `Mỗi bộ lọc cỡ ${C}×2×2 đặt lên đúng một ô 2×2 rồi nhảy 2 điểm: các ô không chồng nhau.`,
    formula: CONV_F,
    example: R`\(k = 2,\ p = 0,\ s = 2\): \(\lfloor (${H} - 2) / 2 \rfloor + 1 = ${H / 2}\).`,
    params: R`\(${C} \times ${2 * C} \times 2 \times 2 + ${2 * C} = ${fmt(8 * C * C + 2 * C)}\).`,
    code: `nn.Conv2d(${C}, ${2 * C}, kernel_size=2, stride=2)`
  }, { tag: 'Conv 2×2 /2', size: vol(2 * C, H / 2) })
] });

function stage(k, C, H, depth, i0) {
  const p = (k > 1 ? pDown(C / 2) : 0) + depth * pBlock(C);
  return n(`Tầng ${k}`, 'block', `${k > 1 ? `${C / 2}×${g(2 * H)}` : `${C}×${g(H)}`} → ${C}×${g(H)}`, {
    what: (k > 1 ? R`Giảm độ phân giải \(${2 * H} \to ${H}\), tăng kênh \(${C / 2} \to ${C}\), rồi ` : 'Ngay sau stem, ') + `${depth} khối ConvNeXt ở ${H}×${H}, ${C} kênh.`,
    formula: [R`x_{l+1} = x_l + \gamma_l \odot F_l(x_l), \qquad l = 1, \dots, ${depth}`],
    why: 'Số khối mỗi tầng là (3, 3, 9, 3) thay cho (3, 4, 6, 3) của ResNet-50, theo tỉ lệ 1:1:3:1 của Swin-T: bước đầu tiên của lộ trình, 78.8% → 79.4%.' + (k === 3 ? ' Tầng 14×14 này nhận một nửa số khối: đủ nhỏ để rẻ, đủ lớn để còn thông tin vị trí.' : ''),
    params: R`\(${fmt(p)}\) (khoảng ${(p / 1e6).toFixed(2)}M).`,
    code: (k > 1 ? `x = model.features[${2 * k - 2}](x)   # LayerNorm2d + Conv2d 2×2 /2\n` : '') + `x = model.features[${2 * k - 1}](x)   # ${depth} CNBlock`
  }, { tag: `Tầng ${k}`, hint: `${depth} khối`, size: vol(C, H), _p: p, ref: [ref('convnext', 'mục 2.2, stage ratio')], children: [
    ...(k > 1 ? [down(C / 2, 2 * H)] : []), block(C, H, i0, depth)
  ] });
}

const STAGES = [];
DEPTHS.reduce((i0, d, k) => { STAGES.push(stage(k + 1, DIMS[k], 56 >> k, d, i0)); return i0 + d; }, 0);
const pStem = 96 * 3 * 4 * 4 + 96 + 2 * 96, pHead = 2 * 768 + 768 * 1000 + 1000;
const pTotal = pStem + STAGES.reduce((t, s) => t + s._p, 0) + pHead;
// self-check: must reproduce torchvision's convnext_tiny parameter count
console.assert(pTotal === 28589128, 'ConvNeXt-T param count mismatch');

// ConvNeXt V2: GRN in every block and masked-autoencoder pre-training (floats above the row: not part of ConvNeXt-T)
const v2 = n('ConvNeXt V2 (2023): sửa gì', 'block', 'các thay đổi so với ConvNeXt V1', {
  what: 'ConvNeXt V2 thêm Global Response Normalization (GRN) vào mỗi khối, ngay sau GELU, và bỏ layer scale; rồi train trước bằng FCMAE: che 60% ảnh và dựng lại, không cần nhãn. Khối này không nằm trong ConvNeXt-T ở hàng dưới; nó chỉ ra những gì V2 thay.',
  formula: [R`\text{V2: } y = x + W_2\, \mathrm{GRN}\Big(\mathrm{GELU}\big(W_1\, \mathrm{LN}(\mathrm{DW}_{7\times7}(x))\big)\Big)`],
  why: 'Train trước bằng cách che ảnh rồi dựng lại (masked autoencoder) giúp ViT rất nhiều, nhưng áp thẳng lên ConvNeXt V1 thì kém: nhiều kênh ở lớp MLP chết hoặc bão hoà, gọi là feature collapse. GRN tạo cạnh tranh giữa các kênh để sửa đúng chỗ đó, nên bài báo gọi hai thay đổi là "đồng thiết kế": phải đi cùng nhau.',
  example: 'ConvNeXt V2-H (659M tham số) đạt 88.9% top-1 trên ImageNet-1K, chỉ dùng dữ liệu công khai.',
  code: `# không có trong torchvision; mã gốc: github.com/facebookresearch/ConvNeXt-V2`
}, { tag: 'ConvNeXt V2', float: true, hint: '2 phần', size: [1.2, 1.2, 1.0], ref: [ref('convnextv2')], children: [
  n('FCMAE', 'io', 'train trước không cần nhãn', {
    what: 'Fully convolutional masked autoencoder: che ngẫu nhiên 60% các ô 32×32 của ảnh. Encoder ConvNeXt chỉ tính trên phần còn thấy; một decoder nhẹ (một khối ConvNeXt, 512 kênh) dựng lại ảnh; loss là sai số bình phương trên phần bị che.',
    formula: [R`\mathcal{L} = \frac{1}{|\Omega|} \sum_{p \in \Omega} \big\| \hat x_p - x_p \big\|^2, \qquad \Omega = \text{các ô bị che}`],
    why: 'Conv thường trượt qua cả vùng che, nên thông tin từ vùng còn thấy rò sang vùng che và mô hình "ăn gian" khi dựng lại. Encoder dùng sparse conv: chỉ tính ở các điểm còn thấy, vùng che giữ nguyên bằng 0. Train trước xong, sparse conv đổi lại thành conv thường với cùng trọng số để fine-tune. Swin V2 dùng cùng ý tưởng (SimMIM).',
    code: `# phác thảo
mask = random_mask(x, ratio=0.6, patch=32)
loss = ((decoder(encoder(x, mask)) - x) ** 2)[mask].mean()`
  }, { tag: 'FCMAE', size: [0.6, 1.4, 0.6], ref: [ref('convnextv2', 'mục 3')] }),
  n('GRN', 'norm', 'H×W×4C → H×W×4C', {
    what: R`Global Response Normalization, ba bước cho mỗi kênh \(i\): gom cả bản đồ \(H \times W\) thành một số (chuẩn L2), chia cho trung bình của số đó trên mọi kênh để ra tầm quan trọng tương đối, rồi nhân lại vào kênh. Thêm \(\gamma, \beta\) học được (khởi tạo 0) và đường tắt.`,
    formula: [R`G_i = \|X_i\|_2 = \sqrt{\textstyle\sum_{h,w} X_{h,w,i}^2}, \qquad N_i = \frac{G_i}{\frac{1}{C}\sum_{j} G_j}`, R`X_i \leftarrow \gamma_i\, X_i\, N_i + \beta_i + X_i`],
    why: 'Kênh phản hồi mạnh hơn trung bình được khuếch đại, kênh yếu bị kìm lại: các kênh phải cạnh tranh nhau và khác nhau, nên không còn collapse. Giống squeeze-and-excitation ở chỗ dùng thông tin của cả ảnh để chỉnh từng kênh, nhưng không có tham số nào ngoài γ, β; với γ = β = 0 lúc đầu, GRN đúng là hàm đồng nhất. Bài báo viết N chia cho tổng các kênh rồi khi cài đặt nhân thêm số kênh, tức là chia cho trung bình như ở đây.',
    example: R`Ba kênh có chuẩn \([2,\ 1,\ 0.5]\), trung bình \(1.167\): \(N = [1.714,\ 0.857,\ 0.429]\). Với \(\gamma = 1, \beta = 0\), ba kênh được nhân \(2.714,\ 1.857,\ 1.429\) lần.`,
    params: R`\(\gamma, \beta\): \(2 \times 4C\) mỗi khối.`,
    code: `gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)        # x: (N, H, W, C)
nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
return self.gamma * (x * nx) + self.beta + x`
  }, { tag: 'GRN', size: SLAB, ref: [ref('convnextv2', 'mục 4')] })
] });

MODELS.convnext = n('ConvNeXt-T', 'block', '3×224×224 → 1000', {
  what: 'Một CNN thuần, nhưng mọi lựa chọn thiết kế đều mượn từ Swin Transformer: stem patchify 4×4, tỉ lệ số khối 1:1:3:1, conv depthwise 7×7 thay cho attention trong cửa sổ 7×7, MLP cổ chai ngược 4C, GELU, LayerNorm, layer scale, và lớp giảm độ phân giải riêng giữa các tầng. Bài báo đi từ ResNet-50 tới ConvNeXt-T từng bước một và đo độ chính xác sau mỗi bước.',
  formula: [R`y = x + \gamma \odot W_2\, \mathrm{GELU}\Big(W_1\, \mathrm{LN}\big(\mathrm{DW}_{7\times7}(x)\big)\Big)`],
  why: 'Câu hỏi của bài: Swin thắng ResNet là nhờ attention, hay nhờ hàng loạt lựa chọn thiết kế đi kèm? Áp dần các lựa chọn đó lên ResNet-50 mà không dùng attention, mạng conv thuần vượt Swin-T ở cùng FLOPs. Không bước nào là phát minh mới; giá trị của bài nằm ở chỗ đo riêng từng bước. Mở từng khối để xem bước nào tạo ra nó.',
  example: 'Bảng 10 của bài gốc (ImageNet-1K, trung bình 3 lần train): ResNet-50 76.1% → train kiểu mới (300 epoch, AdamW, tăng cường dữ liệu như DeiT và Swin) 78.8% → tỉ lệ tầng (3, 3, 9, 3) 79.4% → stem patchify 79.5% → conv depthwise và tăng độ rộng lên 96 kênh 80.5% → cổ chai ngược 80.6% → dời depthwise lên đầu khối 79.9% → kernel 7×7 80.6% → GELU 80.6% → bớt hàm kích hoạt 81.3% → bớt lớp chuẩn hoá 81.4% → BatchNorm thành LayerNorm 81.5% → lớp giảm độ phân giải riêng 82.0%. Swin-T: 81.3%. Bảng 1: ConvNeXt-T 82.1% với 29M tham số và 4.5G FLOPs.',
  params: R`\(${fmt(pTotal)}\) (khoảng 28.6M; Bảng 1 của bài gốc làm tròn thành 29M): stem \(${fmt(pStem)}\), 4 tầng \(${STAGES.map(s => fmt(s._p)).join(' + ')}\), đầu ra \(${fmt(pHead)}\). Cấu hình Tiny: kênh \((96, 192, 384, 768)\), số khối \((3, 3, 9, 3)\).`,
  code: `from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).eval()
probs = model(x).softmax(-1)   # (B, 1000)`
}, { label: 'ConvNeXt-T', noIn: true, outLabel: '1000 lớp', ref: [ref('convnext', 'hình 2, Bảng 1 và 10')], children: [
  imageIn,
  n('Stem (patchify)', 'embed', '3×224×224 → 96×56×56', {
    what: 'Một conv 4×4 bước 4: cắt ảnh thành các ô 4×4 không chồng nhau và chiếu mỗi ô thành 96 kênh, đúng như patch embedding của Swin. Rồi LayerNorm.',
    formula: [R`x_0 = \mathrm{LN}(W_{4\times4} * x), \qquad \text{bước } 4`],
    why: 'Stem của ResNet là conv 7×7 bước 2 rồi max-pool, các ô chồng lên nhau. Thay bằng patchify 4×4 kiểu ViT: 79.4% → 79.5%, với ít FLOPs hơn.',
    params: R`Conv \(96 \times 3 \times 4 \times 4 + 96 = 4\,704\), LayerNorm \(192\): \(${fmt(pStem)}\).`,
    code: `stem = model.features[0]   # Conv2d(3, 96, 4, stride=4) + LayerNorm2d(96)`
  }, { tag: 'Stem 4×4', size: vol(96, 56), ref: [ref('convnext', 'mục 2.2, patchify stem')], children: [
    n('Conv 4×4, bước 4', 'conv', '3×224×224 → 96×56×56', {
      what: '96 bộ lọc cỡ 3×4×4, mỗi lần đặt lên đúng một ô 4×4 rồi nhảy 4 pixel: không ô nào bị nhìn hai lần.',
      formula: CONV_F,
      example: R`\(k = 4,\ p = 0,\ s = 4\): \(\lfloor (224 - 4) / 4 \rfloor + 1 = 56\).`,
      params: R`\(96 \times 3 \times 4 \times 4 + 96 = 4\,704\).`,
      code: `nn.Conv2d(3, 96, kernel_size=4, stride=4)`
    }, { tag: 'Conv 4×4 /4', size: vol(96, 56) }),
    layerNorm('56×56', 96)
  ] }),
  ...STAGES,
  n('Đầu ra: GAP, LayerNorm, FC', 'block', '768×7×7 → 1000', {
    what: 'Lấy trung bình mỗi kênh trên 7×7 vị trí, LayerNorm, rồi một lớp FC ra 1000 lớp.',
    formula: [R`p = \mathrm{softmax}\Big(W\, \mathrm{LN}\big(\mathrm{GAP}(x)\big) + b\Big)`],
    why: 'Như Transformer chuẩn hoá trước đầu ra: bài báo thêm LayerNorm sau global average pooling, cùng lúc với các LayerNorm ở những chỗ đổi độ phân giải.',
    params: R`LayerNorm \(1\,536\), FC \(769\,000\): \(${fmt(pHead)}\).`,
    code: `z = model.avgpool(x)          # (B, 768, 1, 1)
logits = model.classifier(z)   # LayerNorm2d(768), Flatten, Linear(768, 1000)`
  }, { tag: 'Đầu ra', size: [ch(1000), 0.22, 0.22], children: [
    gapNode(768, 7, 'Không có tham số. Như ResNet, ConvNeXt lấy trung bình 49 vị trí cuối thay vì một token [CLS].'),
    { ...layerNorm('1×1', 768), name: 'LayerNorm cuối' },
    fcOut(768, 'classifier[2]')
  ] }),
  v2
] });
})();
