// EfficientNet-B0 (Tan & Le, 2019) as in torchvision efficientnet_b0: MBConv blocks with squeeze-and-excitation, SiLU, stochastic depth.
(() => {
const slab = H => [0.14, sp(H), sp(H)];
const vec = C => [ch(C), 0.22, 0.22];
const NB = 16, sdp = i => +(0.2 * i / NB).toFixed(4); // stochastic-depth rate of block i of 16 (torchvision: 0.2 · i / 16)
const ACT = ' EfficientNet dùng SiLU ở mọi chỗ mà ResNet dùng ReLU.';
const sq = Cin => Math.max(1, Math.floor(Cin / 4)); // SE width: a quarter of the block's input channels (torchvision)
const pBlock = (e, k, Cin, Cout) => { const C = Cin * e, S = sq(Cin); return (e > 1 ? Cin * C + 2 * C : 0) + k * k * C + 2 * C + 2 * C * S + S + C + C * Cout + 2 * Cout; };

const expand = (Cin, C, H) => n('Conv 1×1 mở rộng ×6', 'conv', `${Cin}×${H}×${H} → ${C}×${H}×${H}`, {
  what: R`Mở rộng ${Cin} kênh lên ${C} kênh (gấp 6). Ngược với conv 1×1 đầu khối bottleneck của ResNet, vốn nén kênh lại.`,
  formula: [R`y_{o,i,j} = \sum_{c=1}^{${Cin}} W_{o,c}\, x_{c,i,j}, \qquad W \in \mathbb{R}^{${C} \times ${Cin}}`],
  why: 'Conv depthwise phía sau không trộn kênh, nên một mình nó chỉ lọc được đúng các kênh có sẵn. Mở rộng trước cho nó nhiều tổ hợp kênh để lọc hơn, mà vẫn rẻ vì mỗi kênh depthwise chỉ tốn k² phép nhân.',
  params: R`\(${Cin} \times ${C} = ${fmt(Cin * C)}\).`,
  code: `nn.Conv2d(${Cin}, ${C}, kernel_size=1, bias=False)`
}, { tag: 'Conv 1×1 ↑', size: vol(C, H) });

const dw = (C, k, Hin, s) => n(`Depthwise ${k}×${k}${s > 1 ? ', bước 2' : ''}`, 'conv', `${C}×${Hin}×${Hin} → ${C}×${Hin / s}×${Hin / s}`, {
  what: `Mỗi kênh có một bộ lọc ${k}×${k} riêng và chỉ nhìn chính kênh đó: ${C} kênh vào, ${C} kênh ra, không trộn kênh nào với kênh nào. Việc trộn kênh để cho conv 1×1 phía sau.`,
  formula: [R`y_{c,i,j} = \sum_{u=0}^{${k - 1}} \sum_{v=0}^{${k - 1}} W_{c,u,v}\; x_{c,\; s i + u - p,\; s j + v - p}`, R`s = ${s}, \qquad p = ${(k - 1) / 2}`,
    R`\frac{k^2 C + C\, C_{out}}{k^2\, C\, C_{out}} = \frac{1}{C_{out}} + \frac{1}{k^2}`],
  why: R`Conv thường làm cùng lúc hai việc, nhìn lân cận và trộn kênh, tốn \(k^2 C_{in} C_{out}\) phép nhân ở mỗi vị trí. Tách làm hai bước (depthwise rồi 1×1, gọi chung là depthwise separable) chỉ tốn một phần chi phí như công thức cuối: tử số là depthwise cộng conv 1×1, mẫu số là conv thường. Tức là rẻ hơn gần \(k^2\) lần mà độ chính xác giảm rất ít. Nhờ vậy khối MBConv mới dám mở rộng lên ${C} kênh: ở phần rộng đó chỉ có conv depthwise chạy.`,
  example: R`Ở đây, conv ${k}×${k} thường ${C} → ${C} kênh cần \(${k * k} \cdot ${C} \cdot ${C} = ${fmt(k * k * C * C)}\) tham số; depthwise chỉ cần \(${k * k} \cdot ${C} = ${fmt(k * k * C)}\), ít hơn ${C} lần.`
    + (s > 1 ? R` Kích thước: \(\lfloor (${Hin} + ${k - 1} - ${k}) / 2 \rfloor + 1 = ${Hin / s}\).` : ''),
  params: R`\(${C} \times ${k} \times ${k} = ${fmt(k * k * C)}\).`,
  code: `nn.Conv2d(${C}, ${C}, kernel_size=${k}, stride=${s}, padding=${(k - 1) / 2}, groups=${C}, bias=False)   # groups = số kênh`
}, { tag: `DW ${k}×${k}${s > 1 ? ' /2' : ''}`, size: vol(C, Hin / s), ref: [ref('mobilenet', 'mục 3.1, depthwise separable'), ref('mobilenetv2')],
  ...(k === 3 ? { live: 'conv', conv: { pad: 1, stride: s } } : {}) }); // the 8×8 toy is one channel with a 3×3 filter: exactly one depthwise channel

function se(C, S, H, Cin) {
  const p = 2 * C * S + S + C;
  return n('Squeeze-and-Excitation', 'block', `${C}×${H}×${H} → ${C}×${H}×${H}`, {
    what: 'Tính cho mỗi kênh một hệ số trong khoảng (0, 1) dựa trên toàn bộ ảnh, rồi nhân cả kênh với hệ số đó. Kênh nào hữu ích cho ảnh này được giữ, kênh nào không thì bị vặn nhỏ lại.',
    formula: [R`z = \mathrm{GAP}(x) \in \mathbb{R}^{${C}}`, R`s = \sigma\big(W_2\, \mathrm{SiLU}(W_1 z + b_1) + b_2\big) \in (0, 1)^{${C}}`, R`y_c = s_c\, x_c, \qquad W_1 \in \mathbb{R}^{${S} \times ${C}},\ W_2 \in \mathbb{R}^{${C} \times ${S}}`],
    why: R`Conv chỉ nhìn một vùng nhỏ nên không biết ngữ cảnh của cả ảnh. SE "ép" mỗi kênh còn một số (squeeze), cho qua một MLP nhỏ để các kênh "bàn" với nhau, rồi dùng kết quả làm cổng cho từng kênh (excitation). Lớp giữa chỉ có \(${S} = ${Cin} / 4\) chiều (một phần tư số kênh vào của khối), buộc MLP phải tóm gọn. Rất rẻ vì chạy trên vector, không phải trên cả bản đồ \(${H} \times ${H}\). Bài SENet gắn khối này vào ResNet và các biến thể của nó; mạng dựa trên đó thắng ImageNet 2017.`,
    params: R`\(2 \cdot ${C} \cdot ${S} + ${S} + ${C} = ${fmt(p)}\).`,
    code: `s = F.adaptive_avg_pool2d(x, 1)          # (B, ${C}, 1, 1)
s = torch.sigmoid(self.fc2(F.silu(self.fc1(s))))
y = x * s`
  }, { tag: 'SE', hint: '6 phần', gap: 1.6, size: vol(C, H), ref: [ref('senet'), ref('efficientnet', 'mục 4')],
    extra: [{ a: { gap: 0 }, b: { i: 5, s: 't' }, kind: 'branch', arc: true, label: 'x' }],
    children: [
      n('Global Average Pooling', 'pool', `${C}×${H}×${H} → ${C}`, {
        what: R`Squeeze: lấy trung bình mỗi kênh trên cả \(${H} \times ${H}\) vị trí, mỗi kênh còn đúng một số.`,
        formula: [R`z_c = \frac{1}{${H * H}} \sum_{i=1}^{${H}} \sum_{j=1}^{${H}} x_{c,i,j}`],
        why: 'Số này tóm tắt kênh đó "bật" mạnh tới đâu trên cả ảnh: thông tin toàn cục mà conv không có.',
        code: `z = F.adaptive_avg_pool2d(x, 1)   # (B, ${C}, 1, 1)`
      }, { tag: 'Squeeze', size: vec(C) }),
      n('FC giảm (Conv 1×1)', 'linear', `${C} → ${S}`, {
        what: `Nén ${C} số xuống ${S}.`,
        formula: [R`h = W_1 z + b_1`],
        why: 'Viết bằng Conv2d 1×1 trên tensor 1×1 cho tiện, nhưng thực chất là một lớp Linear.',
        params: R`\(${C} \times ${S} + ${S} = ${fmt(C * S + S)}\).`,
        code: `self.fc1 = nn.Conv2d(${C}, ${S}, kernel_size=1)`
      }, { tag: 'FC ↓', size: vec(S) }),
      silu(`${S}`, [0.14, 0.4, 0.4], ACT),
      n('FC tăng (Conv 1×1)', 'linear', `${S} → ${C}`, {
        what: `Mở lại ${C} số, mỗi kênh một số.`,
        formula: [R`a = W_2\, \mathrm{SiLU}(h) + b_2`],
        params: R`\(${S} \times ${C} + ${C} = ${fmt(S * C + C)}\).`,
        code: `self.fc2 = nn.Conv2d(${S}, ${C}, kernel_size=1)`
      }, { tag: 'FC ↑', size: vec(C) }),
      n('Sigmoid', 'act', `${C}`, {
        what: 'Ép mỗi số về khoảng (0, 1) để dùng làm cổng: gần 0 là tắt kênh, gần 1 là giữ nguyên.',
        formula: [R`\sigma(a) = \frac{1}{1 + e^{-a}}`],
        why: 'Khác softmax, các kênh không phải tranh nhau: nhiều kênh có thể cùng gần 1. Đây là cùng kiểu cổng với các cổng của LSTM.',
        example: R`\(\sigma([-2,\ 0,\ 3]) = [0.119,\ 0.5,\ 0.953]\).`,
        code: `s = torch.sigmoid(a)`
      }, { size: [0.14, 0.4, 0.4] }),
      n('Nhân theo kênh ⊙', 'act', `${C}×${H}×${H}`, {
        what: R`Excitation: nhân cả bản đồ \(${H} \times ${H}\) của kênh \(c\) với cùng một số \(s_c\).`,
        formula: [R`y_{c,i,j} = s_c\, x_{c,i,j}`],
        why: 'Shape giữ nguyên, chỉ độ lớn từng kênh đổi theo nội dung ảnh. Cùng một bộ lọc conv, nhưng ảnh khác nhau thì kênh của nó được dùng nhiều ít khác nhau: một dạng attention trên các kênh.',
        example: R`\(s = [0.9,\ 0.1]\): kênh 1 giữ 90% độ lớn, kênh 2 chỉ còn 10%.`,
        code: `y = x * s   # s: (B, ${C}, 1, 1), tự broadcast lên ${H}×${H}`
      }, { tag: '⊙', size: BALL })
    ] });
}

const project = (C, Cout, H) => n('Conv 1×1 chiếu (không kích hoạt)', 'conv', `${C}×${H}×${H} → ${Cout}×${H}×${H}`, {
  what: `Nén ${C} kênh về ${Cout}. Sau đó chỉ có BN, không có SiLU.`,
  formula: [R`y = \mathrm{BN}(W * h), \qquad W \in \mathbb{R}^{${Cout} \times ${C} \times 1 \times 1}`],
  why: 'MobileNetV2 chỉ ra rằng đặt phi tuyến trong không gian ít chiều làm mất thông tin: phần bị ReLU cắt về 0 không khôi phục lại được. Vì vậy đầu ra hẹp của khối để tuyến tính (linear bottleneck), phi tuyến chỉ dùng ở phần rộng bên trong.',
  params: R`\(${C} \times ${Cout} = ${fmt(C * Cout)}\).`,
  code: `nn.Conv2d(${C}, ${Cout}, kernel_size=1, bias=False)`
}, { tag: 'Conv 1×1 ↓', size: vol(Cout, H), ref: [ref('mobilenetv2', 'mục 3.2, linear bottlenecks')] });

const sdAdd = (C, H, i, rep) => n('Stochastic depth + cộng', 'add', `${C}×${H}×${H}`, {
  what: R`Cộng đường tắt như ResNet. Khác ở chỗ khi train, với xác suất \(p\), nhánh chính của cả khối bị bỏ hẳn (output chỉ còn \(x\)), riêng cho từng ảnh trong batch. Khi suy luận thì luôn cộng đủ.`,
  formula: [R`y = x + \frac{b}{1 - p}\, F(x), \qquad b \sim \mathrm{Bernoulli}(1 - p)`, R`p = 0.2 \cdot \frac{i}{16}, \qquad i = 0, 1, \dots, 15\ \text{là thứ tự của khối}`],
  why: R`Giống dropout nhưng bỏ cả một khối thay vì từng neuron: mỗi bước train là một mạng nông hơn chọn ngẫu nhiên, gradient về các lớp đầu đi đường ngắn hơn, và cả mạng hoạt động như một tập hợp nhiều mạng có độ sâu khác nhau. Chỉ làm được khi có đường tắt, vì bỏ nhánh chính thì \(x\) vẫn đi qua. Chia cho \(1 - p\) để kỳ vọng lúc train bằng lúc suy luận. Khối càng sâu càng hay bị bỏ.`,
  example: rep > 1 ? R`Các khối ở đây là khối thứ ${i} đến ${i + rep - 1} (đếm từ 0) trong 16 khối: \(p\) từ \(${sdp(i)}\) đến \(${sdp(i + rep - 1)}\).`
    : R`Đây là khối thứ ${i} (đếm từ 0) trong 16 khối: \(p = 0.2 \cdot ${i} / 16 = ${sdp(i)}\).`,
  code: `from torchvision.ops import stochastic_depth
out = stochastic_depth(out, p=${sdp(i)}, mode="row", training=self.training)
y = x + out`
}, { tag: 'Cộng + SD', size: BALL, exp: 'res', ref: [ref('stochdepth'), ref('efficientnet', 'mục 5.2: xác suất giữ 0.8')] });

// i = index of this block among the 16 (its stochastic-depth rate); rep = identical blocks this node stands for; title = stage name when the stage has one block
function mbconv(e, k, Cin, Cout, Hin, s, i, rep, title) {
  const C = Cin * e, H = Hin / s, res = s === 1 && Cin === Cout, p = pBlock(e, k, Cin, Cout);
  const kids = [
    ...(e > 1 ? [expand(Cin, C, Hin), bn(C, Hin), silu(`${C}×${Hin}×${Hin}`, slab(Hin), ACT)] : []),
    dw(C, k, Hin, s), bn(C, H), silu(`${C}×${H}×${H}`, slab(H), ACT),
    se(C, sq(Cin), H, Cin),
    project(C, Cout, H), bn(Cout, H),
    ...(res ? [sdAdd(Cout, H, i, rep)] : [])
  ];
  const changed = [Cin !== Cout ? R`\(${Cin} \to ${Cout}\) kênh` : '', s > 1 ? R`\(${Hin} \to ${H}\) không gian` : ''].filter(Boolean).join(', ');
  return n(title || (res ? 'MBConv (identity)' : 'MBConv đầu (đổi shape)'), 'block', `${Cin}×${Hin}×${Hin} → ${Cout}×${H}×${H}`, {
    what: (title ? 'Tầng này chỉ có một khối. ' : '')
      + (e === 1 ? R`MBConv1: không mở rộng kênh. Depthwise ${k}×${k}, squeeze-and-excitation, rồi conv 1×1 chiếu về ${Cout} kênh.`
        : R`Mở rộng ×${e} (\(${Cin} \to ${C}\) kênh), conv depthwise ${k}×${k}${s > 1 ? ' bước 2' : ''}, squeeze-and-excitation, rồi conv 1×1 chiếu về ${Cout} kênh.`)
      + (res ? R` Shape vào và ra giống nhau nên có đường tắt: cộng thẳng \(x\).${rep > 1 ? ` Khối này lặp ${rep} lần, mỗi lần trọng số riêng.` : ''}`
        : ` Shape đổi (${changed}) nên không có đường tắt. Khác ResNet, MBConv không chiếu đường tắt bằng conv 1×1 cho khớp shape mà bỏ hẳn nó.`),
    formula: [
      ...(e > 1 ? [R`h = \mathrm{SiLU}\big(\mathrm{BN}(W_{\uparrow} * x)\big) \in \mathbb{R}^{${C} \times ${Hin} \times ${Hin}}`] : []),
      R`h' = \mathrm{SE}\Big(\mathrm{SiLU}\big(\mathrm{BN}(\mathrm{DW}_{${k}\times${k}}(${e > 1 ? 'h' : 'x'}))\big)\Big)`,
      R`y = ${res ? 'x + ' : ''}\mathrm{BN}(W_{\downarrow} * h') \in \mathbb{R}^{${Cout} \times ${H} \times ${H}}`
    ],
    why: e === 1
      ? 'Tầng này chạy ở 112×112, mỗi kênh thêm vào phải tính trên 12 544 vị trí. Kiến trúc tìm bằng NAS chọn hệ số mở rộng 1 ở đây và 6 ở mọi tầng sau.'
      : R`Cổ chai ngược (inverted residual): bottleneck của ResNet đi rộng → hẹp → rộng và nối đường tắt giữa hai đầu rộng; MBConv đi hẹp → rộng → hẹp (\(${Cin} \to ${C} \to ${Cout}\)), đường tắt nối hai đầu hẹp. Tensor hẹp là thứ phải lưu và truyền giữa các khối nên tốn ít bộ nhớ; phần rộng chỉ tồn tại bên trong khối, nơi chỉ có conv depthwise rẻ chạy.`,
    params: R`\(${fmt(p)}\) mỗi khối${rep > 1 ? R`, \(${fmt(p * rep)}\) cho cả ${rep} khối` : ''}.`,
    code: (e > 1 ? `h = F.silu(self.bn0(self.expand(x)))     # 1×1, ${Cin} → ${C}\n` : '') + `h = F.silu(self.bn1(self.dw(${e > 1 ? 'h' : 'x'})))         # depthwise ${k}×${k}, bước ${s}
h = self.se(h)
out = self.bn2(self.project(h))          # 1×1, ${C} → ${Cout}, không kích hoạt
` + (res ? 'return x + stochastic_depth(out, p, "row", self.training)' : 'return out                               # không có đường tắt')
  }, { tag: title ? title.split(':')[0] : res ? 'MBConv' : 'MBConv 1', ...(title ? { hint: `MBConv${e} ${k}×${k} · 1 khối` } : {}), repeat: rep, size: vol(Cout, H), children: kids, _p: p * rep,
    skips: res ? [[0, kids.length - 1, 'x']] : [], ref: [ref('mobilenetv2', 'mục 3.3, inverted residuals'), ref('efficientnet', 'mục 4, Bảng 1')] });
}

// t = stage number in Table 1 of the paper (1 = stem, 9 = head); i0 = index of the stage's first block among the 16
function stage(t, e, k, s, Cin, Cout, L, Hin, i0) {
  const H = Hin / s, title = `Tầng ${t}: MBConv${e}, ${k}×${k}`;
  if (L === 1) return mbconv(e, k, Cin, Cout, Hin, s, i0, 1, title);
  const kids = [mbconv(e, k, Cin, Cout, Hin, s, i0, 1), mbconv(e, k, Cout, Cout, H, 1, i0 + 1, L - 1)], p = kids[0]._p + kids[1]._p;
  return n(title, 'block', `${Cin}×${Hin}×${Hin} → ${Cout}×${H}×${H}`, {
    what: R`${L} khối MBConv${e} với conv depthwise ${k}×${k}. Khối đầu ${s > 1 ? R`giảm kích thước (\(${Hin} \to ${H}\)) và ` : ''}đổi số kênh (\(${Cin} \to ${Cout}\)); ${L - 1} khối sau giữ nguyên shape nên có đường tắt.`,
    formula: [R`x_{l+1} = x_l + F_l(x_l), \qquad l = 2, \dots, ${L}`],
    why: 'Kernel 3×3 hay 5×5, hệ số mở rộng và số khối mỗi tầng không do người chọn tay: chúng là kết quả tìm kiếm kiến trúc tự động (NAS), tối ưu cùng lúc độ chính xác và số FLOPs.',
    params: R`\(${fmt(p)}\) (khoảng ${(p / 1e6).toFixed(2)}M).`,
    code: `stage = model.features[${t - 1}]   # nn.Sequential gồm ${L} MBConv`
  }, { tag: `Tầng ${t}`, hint: `MBConv${e} ${k}×${k} · ${L} khối`, size: vol(Cout, H), _p: p, children: kids, ref: [ref('efficientnet', 'mục 4, Bảng 1')] });
}

const STAGES = [];
let H = 112, i = 0;
// [expand, kernel, stride, in, out, blocks] for stages 2–8 of Table 1
[[1, 3, 1, 32, 16, 1], [6, 3, 2, 16, 24, 2], [6, 5, 2, 24, 40, 2], [6, 3, 2, 40, 80, 3], [6, 5, 1, 80, 112, 3], [6, 5, 2, 112, 192, 4], [6, 3, 1, 192, 320, 1]]
  .forEach(([e, k, s, a, b, L], j) => { STAGES.push(stage(j + 2, e, k, s, a, b, L, H, i)); H /= s; i += L; });
const pTotal = 928 + STAGES.reduce((t, st) => t + st._p, 0) + 409600 + 2560 + 1281000;
// self-check: must reproduce torchvision's efficientnet_b0 parameter count
console.assert(pTotal === 5288548 && i === NB && H === 7, 'EfficientNet-B0 size mismatch');

MODELS.efficientnet = n('EfficientNet-B0', 'block', '3×224×224 → 1000', {
  what: 'Mạng tích chập nhỏ mà chính xác cho phân loại ImageNet: 5.3M tham số, 0.39 tỷ phép nhân-cộng mỗi ảnh, chính xác hơn ResNet-50 vốn lớn gấp 5 lần và tốn gấp 11 lần phép tính. Khối xây dựng là MBConv: conv 1×1 mở rộng kênh, conv depthwise, squeeze-and-excitation, conv 1×1 nén lại.',
  formula: [R`\begin{aligned} \text{sâu: } d &= \alpha^\phi \\ \text{rộng: } w &= \beta^\phi \\ \text{độ phân giải: } r &= \gamma^\phi \end{aligned}`, R`\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2, \qquad \alpha = 1.2,\ \beta = 1.1,\ \gamma = 1.15`],
  why: R`B0 là mạng gốc, tìm bằng tìm kiếm kiến trúc tự động (NAS). Đóng góp chính của bài báo là cách phóng to nó (compound scaling): thay vì chỉ thêm lớp (như ResNet-18 → ResNet-152) hay chỉ thêm kênh, tăng cả độ sâu, độ rộng và độ phân giải ảnh cùng lúc theo một tỉ lệ cố định. Ảnh to hơn cần nhiều lớp hơn để vùng nhìn đủ rộng và nhiều kênh hơn để bắt các mẫu chi tiết hơn. Chi phí conv tỉ lệ với \(d \cdot w^2 \cdot r^2\), nên ràng buộc \(\alpha \beta^2 \gamma^2 \approx 2\) nghĩa là tăng \(\phi\) thêm 1 thì chi phí gấp đôi. B1 đến B7 là B0 phóng to với \(\phi\) lớn dần.`,
  example: 'Bảng 2 của bài gốc: B0 đạt 77.1% top-1 với 5.3M tham số và 0.39B FLOPs; ResNet-50 đạt 76.0% với 26M và 4.1B. B7 đạt 84.3% với 66M tham số, ít hơn 8.4 lần GPipe (557M) ở cùng độ chính xác. Bài tổng quan ảnh y khoa (Mienye và cộng sự, 2025) nhắc EfficientNet cho công cụ chẩn đoán trên thiết bị di động, và EfficientNetV2 phân loại tổn thương da đạt 96% (Bảng 2).',
  params: R`\(${fmt(pTotal)}\) (khoảng 5.3M), chỉ bằng 1/5 ResNet-50. Riêng lớp FC cuối đã chiếm \(1\,281\,000\), gần một phần tư.`,
  code: `from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).eval()
probs = model(x).softmax(-1)   # (B, 1000)`
}, { label: 'EfficientNet-B0', noIn: true, outLabel: '1000 lớp', ref: [ref('efficientnet', 'mục 3.3, Bảng 1 và 2'), ref('mienye', 'mục 5.6, Bảng 1 và 2')], children: [
  imageIn,
  n('Tầng 1: Stem', 'conv', '3×224×224 → 32×112×112', {
    what: 'Một conv 3×3 bước 2 ra 32 kênh, rồi BN và SiLU: giảm kích thước một nửa ngay từ đầu.',
    formula: [R`\mathrm{Stem}(x) = \mathrm{SiLU}\big(\mathrm{BN}(W_{3\times3} * x)\big)`],
    why: 'Nhẹ hơn nhiều so với stem của ResNet-50 (conv 7×7, 64 kênh: 9 408 tham số, so với 864 ở đây). Phần lớn công việc để cho các khối MBConv phía sau.',
    params: R`\(864 + 64 = 928\).`,
    code: `stem = model.features[0]   # Conv2dNormActivation(3, 32, kernel_size=3, stride=2)`
  }, { tag: 'Stem', size: vol(32, 112), children: [
    n('Conv 3×3, bước 2', 'conv', '3×224×224 → 32×112×112', {
      what: '32 bộ lọc cỡ 3×3×3 trượt trên ảnh với bước nhảy 2.',
      formula: CONV_F,
      example: R`\(k = 3,\ p = 1,\ s = 2\): \(\lfloor (224 + 2 - 3) / 2 \rfloor + 1 = 112\).`,
      params: R`\(32 \times 3 \times 3 \times 3 = 864\). Không có bias vì BatchNorm ngay sau đã có \(\beta\).`,
      code: `nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)`
    }, { tag: 'Conv 3×3 /2', size: vol(32, 112), live: 'conv', conv: { pad: 1, stride: 2 } }),
    bn(32, 112), silu('32×112×112', slab(112), ACT)
  ] }),
  ...STAGES,
  n('Tầng 9: Conv 1×1, pooling, FC', 'block', '320×7×7 → 1000', {
    what: 'Đúng như tầng 9 trong Bảng 1 của bài gốc: conv 1×1 mở rộng 320 kênh lên 1280 (rồi BN, SiLU), global average pooling, dropout, và lớp FC ra 1000 lớp. Lớp 1280 kênh ngay trước pool giống lớp cuối của MobileNetV2.',
    formula: [R`h = \mathrm{SiLU}\big(\mathrm{BN}(W * x)\big), \qquad W \in \mathbb{R}^{1280 \times 320}`, R`p = \mathrm{softmax}\big(W_{fc}\, \mathrm{Dropout}(\mathrm{GAP}(h)) + b\big)`],
    why: 'Sau global average pooling, thông tin không gian mất hết; một tầng rộng ngay trước đó cho bộ phân loại nhiều đặc trưng để chọn hơn. Làm ở 7×7 nên rẻ: 409 600 tham số nhưng chỉ khoảng 20 triệu phép nhân-cộng.',
    params: R`Conv 1×1 + BN: \(320 \times 1280 + 2 \times 1280 = 412\,160\); FC: \(1\,281\,000\). Cộng \(1\,693\,160\).`,
    code: `x = model.features[8](x)                 # Conv 1×1 320 → 1280 + BN + SiLU
z = torch.flatten(model.avgpool(x), 1)    # (B, 1280)
logits = model.classifier(z)              # Dropout(0.2) + Linear(1280, 1000)`
  }, { tag: 'Tầng 9', hint: 'Conv 1×1 · pool · FC', gap: 1.6, size: [ch(1000), 0.22, 0.22], ref: [ref('efficientnet', 'Bảng 1'), ref('mobilenetv2', 'Bảng 2')], children: [
    n('Conv 1×1 → 1280', 'conv', '320×7×7 → 1280×7×7', {
      what: 'Trộn 320 kênh thành 1280 kênh ở từng vị trí.',
      formula: [R`y_{o,i,j} = \sum_{c=1}^{320} W_{o,c}\, x_{c,i,j}`],
      params: R`\(320 \times 1280 = 409\,600\).`,
      code: `nn.Conv2d(320, 1280, kernel_size=1, bias=False)`
    }, { tag: 'Conv 1×1 ↑', size: vol(1280, 7) }),
    bn(1280, 7), silu('1280×7×7', slab(7), ACT),
  gapNode(1280, 7, 'Không có tham số và nhận được ảnh vào kích thước bất kỳ. Nhờ vậy B1 đến B7 dùng ảnh lớn hơn 224×224 mà phần đầu ra không phải đổi.'),
  n('Dropout', 'act', '1280', {
    what: 'Khi train, tắt ngẫu nhiên 20% phần tử của vector 1280 chiều trước lớp FC.',
    formula: [R`\mathrm{Dropout}(z)_c = \frac{m_c\, z_c}{1 - p}`, R`m_c \sim \mathrm{Bernoulli}(1 - p), \qquad p = 0.2`],
    why: 'Mạng lớn cần điều chuẩn mạnh hơn: bài gốc tăng tỉ lệ dropout tuyến tính từ 0.2 ở B0 lên 0.5 ở B7.',
    code: `z = F.dropout(z, p=0.2, training=self.training)`
  }, { size: [0.14, 0.4, 0.4], ref: [ref('dropout'), ref('efficientnet', 'mục 5.2')] }),
  fcOut(1280, 'classifier[1]')
  ] })
] });
})();
