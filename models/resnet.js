// ResNet-50 (He et al., 2015), torchvision resnet50 (v1.5: stride on the 3×3 conv).
(() => {
const bn = (C, H) => n('BatchNorm', 'norm', `${C}×${H}×${H}`, {
  what: R`Chuẩn hoá từng kênh theo trung bình và phương sai tính trên cả batch và mọi vị trí (\(B \times H \times W\) giá trị), rồi co giãn và dịch bằng \(\gamma, \beta\) học được.`,
  formula: [R`\mu_c = \frac{1}{BHW} \sum_{b,i,j} x_{b,c,i,j}, \qquad \sigma_c^2 = \frac{1}{BHW} \sum_{b,i,j} (x_{b,c,i,j} - \mu_c)^2`, R`y = \gamma_c\, \frac{x - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} + \beta_c`],
  why: R`Giữ phân bố activation ổn định giữa các lớp, nên train được với learning rate lớn và ít nhạy với khởi tạo. Khi train dùng thống kê của batch hiện tại; khi suy luận dùng trung bình trượt đã tích luỹ, nên phải gọi model.eval(). Lúc triển khai, BN gộp được hẳn vào conv phía trước: \(W' = \gamma W / \sigma\), \(b' = \beta - \gamma \mu / \sigma\), không tốn thêm phép tính nào.`,
  params: R`\(\gamma, \beta\): \(2 \times ${C} = ${fmt(2 * C)}\) tham số học, thêm \(${fmt(2 * C)}\) số thống kê trượt (không học).`,
  code: `nn.BatchNorm2d(${C})`
}, { tag: 'BN', size: [0.14, sp(H), sp(H)], ref: [ref('bn')] });

const conv1x1 = (Cin, Cout, H, reduce) => n(reduce ? 'Conv 1×1 (nén kênh)' : 'Conv 1×1 (mở rộng kênh)', 'conv', `${Cin}×${H}×${H} → ${Cout}×${H}×${H}`, {
  what: R`Conv 1×1 chính là một lớp Linear áp dụng riêng ở từng điểm ảnh: trộn \(${Cin}\) kênh thành \(${Cout}\) kênh, không nhìn sang điểm lân cận.` + (reduce ? ' Ở đây nó nén kênh để conv 3×3 phía sau rẻ hơn.' : ' Ở đây nó mở rộng lại số kênh để cộng được với đường tắt.'),
  formula: [R`y_{o,i,j} = \sum_{c=1}^{${Cin}} W_{o,c}\, x_{c,i,j}, \qquad W \in \mathbb{R}^{${Cout} \times ${Cin}}`],
  params: R`\(${Cin} \times ${Cout} = ${fmt(Cin * Cout)}\).`,
  code: `nn.Conv2d(${Cin}, ${Cout}, kernel_size=1, bias=False)`
}, { tag: reduce ? 'Conv 1×1 ↓' : 'Conv 1×1 ↑', size: vol(Cout, H) });

const conv3x3 = (C, Hin, s) => n(s > 1 ? 'Conv 3×3, bước 2' : 'Conv 3×3', 'conv', `${C}×${Hin}×${Hin} → ${C}×${Hin / s}×${Hin / s}`, {
  what: 'Conv duy nhất trong khối nhìn sang lân cận (vùng 3×3), nên là nơi khối học các mẫu không gian.'
    + (s > 1 ? R` Bước 2 nên kích thước giảm \(${Hin} \to ${Hin / s}\). torchvision (ResNet v1.5) đặt bước 2 ở đây; bài gốc đặt ở conv 1×1 đầu, khiến conv đó bỏ qua 3/4 số điểm ảnh, nên v1.5 chính xác hơn khoảng 0.5% top-1.` : ''),
  formula: CONV_F,
  example: s > 1 ? R`\(k = 3,\ p = 1,\ s = 2\): \(\lfloor (${Hin} + 2 - 3) / 2 \rfloor + 1 = ${Hin / s}\).` : R`\(k = 3,\ p = 1,\ s = 1\): \(\lfloor (${Hin} + 2 - 3) / 1 \rfloor + 1 = ${Hin}\), giữ nguyên kích thước.`,
  params: R`\(${C} \times ${C} \times 3 \times 3 = ${fmt(9 * C * C)}\).`,
  code: `nn.Conv2d(${C}, ${C}, kernel_size=3, stride=${s}, padding=1, bias=False)`
}, { tag: s > 1 ? 'Conv 3×3 /2' : 'Conv 3×3', size: vol(C, Hin / s), live: 'conv', conv: { pad: 1, stride: s } });

const blockParams = (Cin, mid, Cout, proj) => Cin * mid + 9 * mid * mid + mid * Cout + 4 * mid + 2 * Cout + (proj ? Cin * Cout + 2 * Cout : 0);

function bottleneck(Cin, mid, Cout, Hin, s, proj, rep) {
  const H = Hin / s, p = blockParams(Cin, mid, Cout, proj);
  const kids = [
    conv1x1(Cin, mid, Hin, true), bn(mid, Hin), relu(`${mid}×${Hin}×${Hin}`, [0.14, sp(Hin), sp(Hin)]),
    conv3x3(mid, Hin, s), bn(mid, H), relu(`${mid}×${H}×${H}`, [0.14, sp(H), sp(H)]),
    conv1x1(mid, Cout, H, false), bn(Cout, H),
    n('Cộng đường tắt', 'add', `${Cout}×${H}×${H}`, {
      what: proj ? 'Cộng output nhánh chính với đường tắt đã được chiếu qua conv 1×1.' : R`Cộng từng phần tử output nhánh chính \(F(x)\) với chính đầu vào \(x\). Đây là toàn bộ ý tưởng của ResNet.`,
      formula: [R`y = F(x) + x`, R`\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y}\left(I + \frac{\partial F}{\partial x}\right)`],
      why: R`Nhánh chính chỉ cần học phần dư \(F(x) = H(x) - x\). Nếu hàm tốt nhất gần với hàm đồng nhất, đẩy \(F\) về 0 dễ hơn nhiều so với bắt cả chồng conv phi tuyến học ra đúng hàm đồng nhất. Số hạng \(I\) trong gradient giữ cho tín hiệu học không bị triệt tiêu dù mạng sâu 50, 101 hay 152 lớp.`,
      code: `out = out + identity`
    }, { tag: 'Cộng', size: BALL, exp: 'res', ref: [ref('resnet', 'mục 3.1–3.2'), ref('resnet2')] }),
    relu(`${Cout}×${H}×${H}`, [0.14, sp(H), sp(H)])
  ];
  if (proj) kids.push(n(s > 1 ? 'Đường tắt: Conv 1×1 bước 2 + BN' : 'Đường tắt: Conv 1×1 + BN', 'conv', `${Cin}×${Hin}×${Hin} → ${Cout}×${H}×${H}`, {
    what: R`Chiếu \(x\) sang đúng shape của nhánh chính (\(${Cin} \to ${Cout}\) kênh${s > 1 ? ', bước 2' : ''}) để cộng được. Chỉ khối đầu của mỗi tầng cần.`,
    formula: [R`\mathrm{shortcut}(x) = \mathrm{BN}(W_s * x), \qquad W_s \in \mathbb{R}^{${Cout} \times ${Cin} \times 1 \times 1}`],
    why: 'Bài gốc so ba cách: đệm 0 cho kênh thiếu (A), chỉ chiếu khi đổi shape (B), chiếu ở mọi khối (C). B tốt hơn A một chút; C chỉ nhỉnh hơn B không đáng kể mà tốn thêm tham số, nên ResNet dùng B.',
    params: R`\(${Cin} \times ${Cout} + 2 \times ${Cout} = ${fmt(Cin * Cout + 2 * Cout)}\).`,
    code: `self.downsample = nn.Sequential(
    nn.Conv2d(${Cin}, ${Cout}, kernel_size=1, stride=${s}, bias=False),
    nn.BatchNorm2d(${Cout}))`
  }, { tag: 'Shortcut 1×1', float: true, size: vol(Cout, H), ref: [ref('resnet', 'mục 3.3 và 4.1, phương án A/B/C')] }));
  return n(proj ? 'Bottleneck đầu (projection)' : 'Bottleneck (identity)', 'block', `${Cin}×${Hin}×${Hin} → ${Cout}×${H}×${H}`, {
    what: proj
      ? R`Khối đầu của tầng. Shape đổi (\(${Cin} \to ${Cout}\) kênh${s > 1 ? R`, \(${Hin} \to ${H}\) không gian` : ''}) nên không cộng thẳng \(x\) vào được: đường tắt phải đi qua một conv 1×1${s > 1 ? ' bước 2' : ''} và BN để khớp shape.`
      : R`Shape vào và ra giống nhau (\(${Cout} \times ${H} \times ${H}\)) nên đường tắt là identity: cộng thẳng \(x\), không có tham số. Khối này lặp ${rep} lần, mỗi lần trọng số riêng.`,
    formula: [proj ? R`y = \mathrm{ReLU}\big(F(x) + \mathrm{BN}(W_s * x)\big)` : R`y = \mathrm{ReLU}\big(F(x) + x\big)`, R`F(x) = \mathrm{BN}_3\Big(W_3 * \mathrm{ReLU}\big(\mathrm{BN}_2(W_2 * \mathrm{ReLU}(\mathrm{BN}_1(W_1 * x)))\big)\Big)`],
    why: R`Thiết kế "cổ chai": conv 1×1 nén kênh (\(${Cin} \to ${mid}\)), conv 3×3 làm phần việc đắt ở số kênh nhỏ, rồi 1×1 mở rộng lại (\(${mid} \to ${Cout}\)). So với hai conv 3×3 ở ${Cout} kênh, cách này ít hơn khoảng 17 lần phép tính. Mẹo khi train: khởi tạo \(\gamma\) của BN cuối bằng 0, khi đó \(F(x) = 0\) và lúc đầu khối đúng là hàm đồng nhất.`,
    params: R`\(${fmt(p)}\) mỗi khối${rep > 1 ? R`, \(${fmt(p * rep)}\) cho cả ${rep} khối` : ''}.`,
    code: `identity = x
out = self.relu(self.bn1(self.conv1(x)))     # 1×1, ${Cin} → ${mid}
out = self.relu(self.bn2(self.conv2(out)))   # 3×3, bước ${s}
out = self.bn3(self.conv3(out))              # 1×1, ${mid} → ${Cout}
if self.downsample is not None:
    identity = self.downsample(x)            # conv 1×1 + BN
return self.relu(out + identity)`
  }, { tag: proj ? 'Bottleneck 1' : 'Bottleneck', repeat: rep, size: vol(Cout, H), children: kids, ref: [ref('resnet', 'mục 4.1, Deeper Bottleneck'), ref('goyal', 'khởi tạo γ = 0')],
    skips: proj ? [] : [[0, 8, 'x']],
    extra: proj ? [{ a: { gap: 0 }, b: { i: 10, s: 'l' }, kind: 'short', label: 'x' }, { a: { i: 10, s: 'r' }, b: { i: 8, s: 't' }, kind: 'short', label: 'Wₛx' }] : [] });
}

function stage(k, nb, Cin, mid, Cout, Hin, s) {
  const H = Hin / s, p = blockParams(Cin, mid, Cout, true) + (nb - 1) * blockParams(Cout, mid, Cout, false);
  return n(`Tầng conv${k}_x`, 'block', `${Cin}×${Hin}×${Hin} → ${Cout}×${H}×${H}`, {
    what: s > 1
      ? R`${nb} khối bottleneck. Khối đầu giảm kích thước không gian một nửa (\(${Hin} \to ${H}\)) và tăng gấp đôi số kênh (\(${Cin} \to ${Cout}\)). Số vị trí giảm 4 lần, chi phí mỗi vị trí tăng 4 lần, nên mỗi tầng tốn xấp xỉ như nhau.`
      : R`${nb} khối bottleneck ở độ phân giải \(56 \times 56\). Khối đầu tăng số kênh \(64 \to 256\) nhưng không giảm kích thước, vì stem đã giảm rồi.`,
    formula: [R`x_{l+1} = \mathrm{ReLU}\big(x_l + F(x_l)\big), \qquad l = 1, \dots, ${nb}`, R`x_L \approx x_l + \sum_{i=l}^{L-1} F(x_i)`],
    why: 'Trải phép cộng ra: tín hiệu ở lớp sâu bằng tín hiệu ở lớp nông cộng tổng các phần dư, nên gradient luôn có một đường cộng thẳng về lớp đầu. Công thức thứ hai đúng chính xác khi bỏ ReLU sau phép cộng, như ResNet v2 (pre-activation).',
    params: R`\(${fmt(p)}\) (khoảng ${(p / 1e6).toFixed(2)}M).`,
    code: `stage = model.layer${k - 1}   # nn.Sequential gồm ${nb} Bottleneck`
  }, { tag: `conv${k}_x`, hint: `${nb} khối`, size: vol(Cout, H), _params: p, ref: [ref('resnet2', 'công thức trải phép cộng')], children: [
    bottleneck(Cin, mid, Cout, Hin, s, true, 1),
    bottleneck(Cout, mid, Cout, H, 1, false, nb - 1)
  ] });
}

const STAGES = [stage(2, 3, 64, 64, 256, 56, 1), stage(3, 4, 256, 128, 512, 56, 2), stage(4, 6, 512, 256, 1024, 28, 2), stage(5, 3, 1024, 512, 2048, 14, 2)];
// self-check: the per-block formula must reproduce torchvision's resnet50 parameter count
console.assert(9408 + 128 + STAGES.reduce((t, s) => t + s._params, 0) + 2049000 === 25557032, 'ResNet-50 param count mismatch');

MODELS.resnet = n('ResNet-50', 'block', '3×224×224 → 1000', {
  what: R`Mạng tích chập 50 lớp có trọng số cho bài toán phân loại ảnh ImageNet. Ý tưởng cốt lõi: mỗi khối học phần dư \(F(x) = H(x) - x\) thay vì học thẳng \(H(x)\), rồi cộng \(x\) trở lại qua một đường tắt.`,
  formula: [R`y = F(x, \{W_i\}) + x`],
  why: R`Trước ResNet, mạng càng sâu thì lỗi trên chính tập train lại càng cao (hiện tượng degradation), dù về lý thuyết mạng sâu có thể bắt chước mạng nông bằng cách cho các lớp thừa thành hàm đồng nhất. Bộ tối ưu không tìm ra lời giải đó. Với đường tắt, hàm đồng nhất chỉ cần \(F = 0\), rất dễ học, nên thêm lớp không còn làm mạng tệ đi. Mỗi khối 3D ở đây vẽ theo output của nó: chiều ngang là số kênh, hai chiều kia là kích thước không gian.`,
  example: 'Bảng 2 của bài gốc, lỗi top-1 trên ImageNet val: plain-18 27.94%, plain-34 28.54% (sâu hơn mà tệ hơn); ResNet-18 27.88%, ResNet-34 25.03%.',
  params: R`\(25\,557\,032\) tham số (khoảng 25.6M), khoảng \(3.8 \times 10^9\) phép nhân-cộng cho một ảnh. 50 lớp = 1 conv đầu + 16 khối × 3 conv + 1 FC.`,
  code: `from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
probs = model(x).softmax(-1)   # (B, 1000)`
}, { label: 'ResNet-50', noIn: true, outLabel: '1000 lớp', exp: 'res', ref: [ref('resnet', 'Bảng 1 và 2'), ref('resnet2')], children: [
  imageIn,
  n('Stem', 'conv', '3×224×224 → 64×56×56', {
    what: 'Phần đầu giảm kích thước thật nhanh: conv 7×7 bước 2 rồi max-pool bước 2, từ 224×224 xuống 56×56 (ít hơn 16 lần số điểm ảnh), để các tầng sau rẻ hơn nhiều.',
    formula: [R`\mathrm{Stem}(x) = \mathrm{MaxPool}\big(\mathrm{ReLU}(\mathrm{BN}(W_{7\times7} * x))\big)`],
    why: 'Ở độ phân giải đầy đủ, mỗi phép tính đều đắt. Một conv 7×7 nhìn được vùng đủ rộng ngay từ đầu, rồi kích thước được giảm sớm.',
    params: R`\(9\,408 + 128 = 9\,536\).`,
    code: `stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)`
  }, { size: vol(64, 56), children: [
    n('Conv 7×7, 64 kênh, bước 2', 'conv', '3×224×224 → 64×112×112', {
      what: 'Trượt 64 bộ lọc cỡ 3×7×7 trên ảnh với bước nhảy 2. Tại mỗi vị trí, mỗi bộ lọc tính tích vô hướng với mảng ảnh bên dưới, ra một kênh output.',
      formula: CONV_F,
      why: 'Hai tính chất khiến conv hợp với ảnh: chỉ nhìn cục bộ, và dùng chung trọng số ở mọi vị trí (một bộ phát hiện cạnh dùng được khắp ảnh). Các bộ lọc lớp đầu sau khi train thường giống bộ lọc cạnh và đốm màu kiểu Gabor.',
      example: R`\(k = 7,\ p = 3,\ s = 2\): \(H_{out} = \lfloor (224 + 2 \cdot 3 - 7) / 2 \rfloor + 1 = \lfloor 111.5 \rfloor + 1 = 112\).`,
      params: R`\(64 \times 3 \times 7 \times 7 = 9\,408\). Không có bias vì BatchNorm ngay sau đã có \(\beta\).`,
      code: `nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)`
    }, { tag: 'Conv 7×7 /2', size: vol(64, 112) }),
    bn(64, 112),
    relu('64×112×112', [0.14, sp(112), sp(112)]),
    n('MaxPool 3×3, bước 2', 'pool', '64×112×112 → 64×56×56', {
      what: 'Lấy giá trị lớn nhất trong mỗi ô 3×3 với bước nhảy 2, giảm mỗi chiều không gian một nửa.',
      formula: [R`y_{c,i,j} = \max_{0 \le u, v \lt 3}\; x_{c,\; 2i + u - 1,\; 2j + v - 1}`],
      why: 'Giữ phản hồi mạnh nhất trong mỗi vùng, nên dịch đối tượng một chút thì output gần như không đổi. Không có tham số. Khi lan truyền ngược, gradient chỉ chảy về đúng phần tử lớn nhất.',
      example: R`\(\lfloor (112 + 2 \cdot 1 - 3) / 2 \rfloor + 1 = \lfloor 55.5 \rfloor + 1 = 56\).`,
      code: `nn.MaxPool2d(kernel_size=3, stride=2, padding=1)`
    }, { tag: 'MaxPool', size: vol(64, 56) })
  ] }),
  ...STAGES,
  n('Global Average Pooling', 'pool', '2048×7×7 → 2048', {
    what: 'Lấy trung bình mỗi kênh trên toàn bộ 7×7 vị trí, ra một vector 2048 chiều mô tả cả ảnh.',
    formula: [R`z_c = \frac{1}{7 \cdot 7} \sum_{i=1}^{7} \sum_{j=1}^{7} x_{c,i,j}`],
    why: 'Không có tham số và nhận được ảnh vào kích thước bất kỳ. VGG-16 dùng các lớp FC ở vị trí này, tốn khoảng 124M tham số, gần gấp 5 lần toàn bộ ResNet-50.',
    code: `z = torch.flatten(nn.AdaptiveAvgPool2d(1)(x), 1)   # (B, 2048)`
  }, { tag: 'Avg Pool', size: [ch(2048), 0.22, 0.22], ref: [ref('nin', 'global average pooling'), ref('vgg', 'để so sánh')] }),
  n('FC 1000 + Softmax', 'linear', '2048 → 1000', {
    what: 'Một lớp Linear cho ra 1000 điểm số, mỗi lớp của ImageNet một điểm, rồi softmax thành xác suất.',
    formula: [R`\text{logits} = W z + b, \qquad W \in \mathbb{R}^{1000 \times 2048}`, R`p_k = \frac{e^{\text{logit}_k}}{\sum_{j=1}^{1000} e^{\text{logit}_j}}, \qquad \mathcal{L} = -\log p_{y}`],
    why: 'Trong PyTorch, softmax nằm sẵn trong hàm loss (cross_entropy nhận logit), nên model chỉ trả về logit. Muốn dùng ResNet cho bài toán khác, thường chỉ cần thay lớp này rồi fine-tune.',
    params: R`\(2048 \times 1000 + 1000 = 2\,049\,000\).`,
    code: `model.fc = nn.Linear(2048, num_classes)   # thay đầu ra khi fine-tune
loss = F.cross_entropy(model(x), y)`
  }, { tag: 'FC → 1000', size: [ch(1000), 0.22, 0.22] })
] });
})();
