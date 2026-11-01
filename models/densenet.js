// DenseNet-121 (Huang et al., 2017) as in torchvision densenet121: DenseNet-BC, growth rate k = 32, bottleneck 4k = 128, compression 0.5.
(() => {
const K = 32, W = 4 * K;
const PRE = ' Ở DenseNet, BN đứng trước conv (BN → ReLU → Conv, kiểu pre-activation như ResNet v2) nên không gộp vào conv phía trước được. Mỗi lớp có γ, β riêng cho cả chồng kênh nó đọc vào, nên cùng một kênh cũ được mỗi lớp sau co giãn lại theo cách riêng.';
const pLayer = c => 2 * c + c * W + 2 * W + 9 * W * K; // BN(c) + 1×1 c→128 + BN(128) + 3×3 128→32
const pTrans = c => 2 * c + c * c / 2;
const slab = H => [0.14, sp(H), sp(H)];

// layers a..b of a block (one node, ghost copies for the rest); children show layer a, whose input has c channels
function denseLayer(a, b, c, H, c0) {
  const rep = b - a + 1, cs = k => c + K * (k - a);
  let p = 0;
  for (let k = a; k <= b; k++) p += pLayer(cs(k));
  return n(rep > 1 ? `Lớp dense ${a}–${b}` : `Lớp dense ${a}`, 'conv', `${rep > 1 ? `${c}…${cs(b)}` : c}×${H}×${H} → ${K}×${H}×${H}`, {
    what: rep > 1
      ? R`Lớp ${a} đến ${b} của khối: cùng cấu trúc, mỗi lớp trọng số riêng. Lớp \(\ell\) đọc \(${c0} + 32(\ell - 1)\) kênh, từ ${c} kênh (lớp ${a}) đến ${cs(b)} kênh (lớp ${b}), và lớp nào cũng chỉ viết ra 32 kênh. Bên trong hiển thị lớp ${a}.`
      : R`Đọc cả chồng ${c} kênh hiện có${a === 1 ? ' (chính là đầu vào của khối)' : R` (\(${c0} + 32 \cdot ${a - 1}\))`}, cho qua BN → ReLU → Conv 1×1 → BN → ReLU → Conv 3×3, và chỉ viết ra 32 kênh mới.`,
    formula: [R`x_\ell = H_\ell\big([x_0, x_1, \dots, x_{\ell-1}]\big)`, R`h = \mathrm{Conv}_{1\times1}\big(\mathrm{ReLU}(\mathrm{BN}([x_0, \dots, x_{\ell-1}]))\big)`, R`x_\ell = \mathrm{Conv}_{3\times3}\big(\mathrm{ReLU}(\mathrm{BN}(h))\big)`],
    why: 'Số kênh vào tăng dần theo độ sâu trong khối, nhưng số kênh ra luôn là k = 32, và nhờ conv 1×1 cổ chai, conv 3×3 lúc nào cũng chỉ đọc 128 kênh. Vì thế mỗi lớp tốn gần như nhau dù nằm sâu đến đâu: chỉ BN đầu và conv 1×1 lớn dần theo số kênh vào.',
    params: R`\(2 \cdot ${c} + ${c} \cdot 128 + 2 \cdot 128 + 9 \cdot 128 \cdot 32 = ${fmt(pLayer(c))}\)` + (rep > 1 ? R` cho lớp ${a}; cả ${rep} lớp: \(${fmt(p)}\).` : '.'),
    code: `x = torch.cat(features, 1)                   # (B, ${c}, ${H}, ${H})
h = self.conv1(F.relu(self.norm1(x)))        # 1×1 → 128
new = self.conv2(F.relu(self.norm2(h)))      # 3×3 → 32`
  }, { tag: rep > 1 ? `Dense ${a}–${b}` : `Dense ${a}`, repeat: rep, size: vol(K, H), _p: p, ref: [ref('densenet', 'mục 3, Composite function và Bottleneck layers')], children: [
    bn(c, H, PRE), relu(`${c}×${H}×${H}`, slab(H)),
    n('Conv 1×1 → 4k = 128', 'conv', `${c}×${H}×${H} → ${W}×${H}×${H}`, {
      what: 'Lớp cổ chai (bottleneck): nén chồng kênh đang phình to về cố định 4k = 128 kênh trước conv 3×3 đắt tiền.',
      formula: [R`y_{o,i,j} = \sum_{c=1}^{${c}} W_{o,c}\, x_{c,i,j}, \qquad W \in \mathbb{R}^{128 \times ${c}}`],
      why: R`Không có bước này, conv 3×3 phải đọc thẳng cả chồng kênh: \(9 \cdot ${c} \cdot 32 = ${fmt(9 * c * K)}\) tham số ở lớp này, và chồng còn lớn tới 992 kênh ở cuối khối 3. Có cổ chai, conv 3×3 luôn chỉ đọc 128 kênh. Đây là chữ B trong DenseNet-BC.`,
      params: R`\(${c} \times 128 = ${fmt(c * W)}\).`,
      code: `nn.Conv2d(${c}, 128, kernel_size=1, bias=False)`
    }, { tag: 'Conv 1×1 ↓', size: vol(W, H) }),
    bn(W, H, PRE), relu(`${W}×${H}×${H}`, slab(H)),
    n('Conv 3×3 → k = 32', 'conv', `${W}×${H}×${H} → ${K}×${H}×${H}`, {
      what: 'Conv duy nhất trong lớp nhìn sang lân cận. Nó tạo ra đúng k = 32 kênh mới: phần đóng góp của lớp này vào "kiến thức chung" của khối.',
      formula: CONV_F,
      why: 'Lớp rất hẹp (32 kênh, trong khi mỗi conv 3×3 của ResNet-50 có 64 đến 512 kênh) mà vẫn đủ, vì nó không phải chép lại thông tin đã có: mọi kênh cũ vẫn nằm nguyên trong chồng, lớp sau đọc thẳng được. Số kênh mỗi lớp thêm vào, k, gọi là tốc độ tăng trưởng (growth rate).',
      example: R`\(k = 3,\ p = 1,\ s = 1\): \(\lfloor (${H} + 2 - 3) / 1 \rfloor + 1 = ${H}\), giữ nguyên kích thước để nối được.`,
      params: R`\(128 \times 32 \times 3 \times 3 = 36\,864\).`,
      code: `nn.Conv2d(128, 32, kernel_size=3, padding=1, bias=False)`
    }, { tag: 'Conv 3×3', size: vol(K, H), live: 'conv', conv: { pad: 1, stride: 1 } })
  ] });
}

// rep new 32-channel maps appended to a stack of c channels
const cat = (c, rep, H, c0) => n('Nối kênh', 'shape', `${c}×${H}×${H} + ${rep > 1 ? `${rep} × ` : ''}${K}×${H}×${H} → ${c + K * rep}×${H}×${H}`, {
  what: rep > 1
    ? R`Sau mỗi lớp trong nhóm, 32 kênh mới được nối vào cuối chồng. Qua ${rep} lớp, chồng dài thêm \(${rep} \times 32 = ${rep * K}\) kênh: \(${c} \to ${c + K * rep}\).`
    : R`Đặt 32 kênh mới cạnh chồng ${c} kênh cũ: \(${c} + 32 = ${c + K}\). Không tính toán gì, không có tham số.`,
  formula: [R`[x_0, x_1, \dots, x_\ell] \in \mathbb{R}^{(k_0 + \ell k) \times ${H} \times ${H}}`, R`k_0 = ${c0}, \qquad k = 32`],
  why: R`Đây là chỗ khác ResNet. ResNet cộng: \(x + F(x)\) giữ nguyên số kênh, đặc trưng mới trộn lẫn vào đặc trưng cũ. DenseNet nối: đặc trưng cũ còn nguyên ở đúng kênh của nó, lớp nào phía sau cũng đọc lại được, nên mỗi lớp không phải học lại thứ đã có mà chỉ cần thêm ít kênh mới.`,
  code: `features.append(new)            # danh sách tensor, nối khi lớp sau cần
x = torch.cat(features, 1)      # (B, ${c + K * rep}, ${H}, ${H})`
}, { tag: 'Nối', size: vol(c + K * rep, H), ref: [ref('densenet', 'mục 3, Dense connectivity')] });

function denseBlock(b, L, c0, H) {
  const cEnd = c0 + K * L;
  const kids = [denseLayer(1, 1, c0, H, c0), cat(c0, 1, H, c0), denseLayer(2, 2, c0 + K, H, c0), cat(c0 + K, 1, H, c0), denseLayer(3, L, c0 + 2 * K, H, c0), cat(c0 + 2 * K, L - 2, H, c0)];
  const p = kids.reduce((t, k) => t + (k._p || 0), 0);
  return n(`Dense block ${b}`, 'block', `${c0}×${H}×${H} → ${cEnd}×${H}×${H}`, {
    what: R`${L} lớp dense ở độ phân giải \(${H} \times ${H}\). Lớp thứ \(\ell\) đọc toàn bộ \(${c0} + 32(\ell - 1)\) kênh của đầu vào khối và của mọi lớp trước nó, rồi viết thêm 32 kênh. Ra khỏi khối: \(${c0} + ${L} \cdot 32 = ${cEnd}\) kênh.`,
    formula: [R`x_\ell = H_\ell\big([x_0, x_1, \dots, x_{\ell-1}]\big), \qquad \ell = 1, \dots, ${L}`, R`C_\ell = k_0 + k(\ell - 1) = ${c0} + 32(\ell - 1)`],
    why: R`Khối có \(L(L+1)/2 = ${L * (L + 1) / 2}\) kết nối trực tiếp thay vì ${L} (lớp \(\ell\) nhận \(\ell\) đầu vào). Mỗi lớp nối tới bộ phân loại qua đường ngắn, chỉ cách nó tối đa hai ba transition layer, nên nhận gradient gần như trực tiếp: bài gốc gọi đây là giám sát ngầm (implicit deep supervision). Cài đặt ngây thơ tốn bộ nhớ vì mỗi lớp tạo một bản nối mới; torchvision có tuỳ chọn memory_efficient=True, tính lại bước nối + BN + ReLU + conv 1×1 khi lan truyền ngược thay vì lưu lại.`,
    example: R`Số kênh vào của lớp 1, 2, 3, …, ${L}: \(${c0},\ ${c0 + K},\ ${c0 + 2 * K},\ \dots,\ ${c0 + K * (L - 1)}\).`,
    params: R`\(${fmt(p)}\) (khoảng ${(p / 1e6).toFixed(2)}M).`,
    code: `features = [x]                                      # x: (B, ${c0}, ${H}, ${H})
for layer in model.features.denseblock${b}.values():   # ${L} _DenseLayer
    features.append(layer(features))                 # mỗi lớp trả về 32 kênh mới
x = torch.cat(features, 1)                          # (B, ${cEnd}, ${H}, ${H})`
  }, { tag: `Dense ${b}`, hint: `${L} lớp`, size: vol(cEnd, H), _p: p, children: kids,
    extra: [0, 2, 4].map((g, j) => ({ a: { gap: g }, b: { i: g + 1, s: 't' }, kind: 'concat', arc: true, label: ['x₀', 'x₀, x₁', 'x₀…x₂'][j] })),
    ref: [ref('densenet', 'mục 3 và Bảng 1')] });
}

const transition = (t, c, H) => n(`Transition ${t}`, 'conv', `${c}×${H}×${H} → ${c / 2}×${H / 2}×${H / 2}`, {
  what: R`Nằm giữa hai dense block: BN → ReLU → conv 1×1 giảm một nửa số kênh (\(${c} \to ${c / 2}\)), rồi average pool 2×2 giảm một nửa kích thước (\(${H} \to ${H / 2}\)).`,
  formula: [R`x' = \mathrm{AvgPool}_{2\times2}\Big(W * \mathrm{ReLU}\big(\mathrm{BN}(x)\big)\Big)`, R`W \in \mathbb{R}^{${c / 2} \times ${c} \times 1 \times 1}`],
  why: R`Phép nối chỉ làm được khi các bản đồ đặc trưng cùng kích thước, nên việc giảm kích thước phải đặt ngoài dense block. Số kênh giảm theo hệ số nén \(\theta = 0.5\): đây là chữ C trong DenseNet-BC. Không nén thì số kênh cứ cộng dồn qua các khối và cuối mạng có 1920 kênh thay vì 1024.`,
  params: R`\(2 \cdot ${c} + ${c} \cdot ${c / 2} = ${fmt(pTrans(c))}\).`,
  code: `x = model.features.transition${t}(x)   # (B, ${c / 2}, ${H / 2}, ${H / 2})`
}, { tag: `Trans. ${t}`, size: vol(c / 2, H / 2), _p: pTrans(c), ref: [ref('densenet', 'mục 3, Pooling layers và Compression')], children: [
  bn(c, H, PRE), relu(`${c}×${H}×${H}`, slab(H)),
  n('Conv 1×1 nén ×0.5', 'conv', `${c}×${H}×${H} → ${c / 2}×${H}×${H}`, {
    what: `Trộn ${c} kênh thành ${c / 2} kênh ở từng điểm ảnh.`,
    formula: [R`y_{o,i,j} = \sum_{c=1}^{${c}} W_{o,c}\, x_{c,i,j}, \qquad W \in \mathbb{R}^{${c / 2} \times ${c}}`],
    why: 'Chồng kênh ra khỏi dense block chứa cả đặc trưng cũ lẫn mới; conv 1×1 tóm chúng lại còn một nửa trước khi sang khối sau, nơi một chồng mới lại bắt đầu lớn dần.',
    params: R`\(${c} \times ${c / 2} = ${fmt(c * c / 2)}\).`,
    code: `nn.Conv2d(${c}, ${c / 2}, kernel_size=1, bias=False)`
  }, { tag: 'Conv 1×1 ↓', size: vol(c / 2, H) }),
  n('AvgPool 2×2, bước 2', 'pool', `${c / 2}×${H}×${H} → ${c / 2}×${H / 2}×${H / 2}`, {
    what: 'Lấy trung bình mỗi ô 2×2, giảm mỗi chiều không gian một nửa.',
    formula: [R`y_{c,i,j} = \frac{1}{4} \sum_{u, v \in \{0, 1\}} x_{c,\; 2i + u,\; 2j + v}`],
    why: 'Khác MaxPool, mọi phần tử trong ô đều góp vào output, nên khi lan truyền ngược cả 4 phần tử đều nhận gradient (mỗi phần tử 1/4), chứ không chỉ phần tử lớn nhất.',
    example: R`Ô \(\begin{bmatrix} 1 & 3 \\ 2 & 6 \end{bmatrix}\): \(\tfrac{1}{4}(1 + 3 + 2 + 6) = 3\). MaxPool trên cùng ô cho 6.`,
    code: `nn.AvgPool2d(kernel_size=2, stride=2)`
  }, { tag: 'AvgPool', size: vol(c / 2, H / 2) })
] });

const body = [];
let c = 64, H = 56;
[6, 12, 24, 16].forEach((L, j) => {
  body.push(denseBlock(j + 1, L, c, H)); c += K * L;
  if (j < 3) { body.push(transition(j + 1, c, H)); c /= 2; H /= 2; }
});
const stem = MODELS.resnet.children.find(k => k.name === 'Stem'); // identical to ResNet-50's (conv 7×7 /2, BN, ReLU, max-pool /2)
const pTotal = 9408 + 128 + body.reduce((t, b) => t + b._p, 0) + 2 * 1024 + 1024 * 1000 + 1000;
// self-check: must reproduce torchvision's densenet121 parameter count
console.assert(stem && pTotal === 7978856 && c === 1024 && H === 7, 'DenseNet-121 size mismatch');

// CheXNet: this DenseNet-121 on chest X-rays, 14 labels instead of ImageNet's 1000 classes
const pBack = pTotal - (1024 * 1000 + 1000), pHead = 1024 * 14 + 14;
const ex = { P: 13, N: 987 }, wPos = ex.N / (ex.P + ex.N), wNeg = ex.P / (ex.P + ex.N);
const lossPos = -wPos * Math.log(0.2), lossNeg = -wNeg * Math.log(0.8);
// self-check: the two classes weigh the same in total, and the head is torchvision's with num_classes = 14
console.assert(Math.abs(ex.P * wPos - ex.N * wNeg) < 1e-12 && pBack + pHead === 6968206, 'CheXNet numbers mismatch');
const chexnet = n('CheXNet (2017)', 'block', '3×224×224 → 14 xác suất', {
  what: 'Chính DenseNet-121 này, dùng cho ảnh X-quang ngực (Rajpurkar và cộng sự, 2017). Lớp FC 1000 lớp của ImageNet được thay bằng một lớp ra 14 điểm số, mỗi bệnh một điểm, rồi mỗi điểm qua một sigmoid riêng, vì một ảnh có thể có nhiều bệnh cùng lúc. Dữ liệu là ChestX-ray14: 112 120 ảnh chụp thẳng của 30 805 bệnh nhân, nhãn được trích tự động từ báo cáo của bác sĩ.',
  why: `Ý chính của bài: không cần kiến trúc mới, một mạng ImageNet thay đầu ra rồi train lại trên đủ nhiều ảnh y khoa là làm được việc. Hai chỗ dễ sai nằm ngoài mạng. Thứ nhất là cách chia tập: trung bình mỗi bệnh nhân có ${(112120 / 30805).toFixed(2)} ảnh. Chia ngẫu nhiên theo ảnh thì cùng một người có ảnh ở cả tập train lẫn tập test, và mạng có thể nhận ra người (dáng lồng ngực, dây điện cực, máy móc) thay vì nhận ra bệnh, nên điểm test bị thổi phồng. CheXNet chia theo bệnh nhân: không ai có mặt ở hai tập. Thứ hai là nhãn: trích từ văn bản báo cáo bằng xử lý ngôn ngữ nên có nhiễu, và không mô hình nào đáng tin hơn chính nhãn dùng để chấm nó.`,
  example: 'Bài toán viêm phổi của bài báo: trên tập test 420 ảnh được 4 bác sĩ X-quang gán nhãn, F1 của CheXNet là 0.435, trung bình của các bác sĩ là 0.387.',
  params: R`\(${fmt(pBack + pHead)}\): phần đặc trưng của DenseNet-121 \(${fmt(pBack)}\) cộng lớp ra 14 nhãn \(${fmt(pHead)}\).`,
  code: `from torchvision.models import densenet121, DenseNet121_Weights
model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)   # khởi tạo từ ImageNet
model.classifier = nn.Linear(1024, 14)                          # 14 bệnh thay cho 1000 lớp
probs = torch.sigmoid(model(x))                                 # (B, 14): mỗi bệnh một xác suất`
}, { tag: 'CheXNet', float: true, hint: '5 phần', size: [1.2, 1.2, 1.0], ref: [ref('chexnet'), ref('chestxray14', 'bộ dữ liệu ChestX-ray14')], children: [
  n('Ảnh X-quang ngực', 'io', '3×224×224', {
    what: 'Ảnh chụp thẳng từ ChestX-ray14, thu nhỏ về 224×224 và chuẩn hoá theo trung bình, độ lệch chuẩn của ImageNet như ảnh vào của DenseNet-121. Ảnh X-quang chỉ có một kênh xám; lặp nó thành 3 kênh để dùng được trọng số ImageNet. Khi train, ảnh được lật ngang ngẫu nhiên.',
    why: 'Thu nhỏ làm mất chi tiết nhỏ như các nốt phổi, đổi lại train nhanh và dùng lại được nguyên mạng ImageNet. Lật ngang là tăng cường dữ liệu quen thuộc với ảnh thường, nhưng trên phim ngực nó đưa tim sang bên phải, điều rất hiếm ở người thật: phép tăng cường nào cũng nên hỏi xem ảnh sau khi biến đổi có còn là một ảnh có thể gặp không.',
    code: `tf = T.Compose([T.Grayscale(3), T.Resize((224, 224)), T.RandomHorizontalFlip(), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])`
  }, { tag: 'X-quang', size: [0.06, 1.5, 1.5] }),
  n('Đặc trưng DenseNet-121', 'block', '3×224×224 → 1024', {
    what: 'Toàn bộ phần đặc trưng của DenseNet-121 bên dưới, tới global average pooling. Khởi tạo bằng trọng số ImageNet rồi train lại hết (end-to-end), không đóng băng lớp nào.',
    why: 'Ảnh X-quang khác xa ảnh ImageNet, nhưng các lớp đầu nhận cạnh và kết cấu, những thứ vẫn dùng lại được. Bắt đầu từ trọng số ImageNet thay vì ngẫu nhiên là cách thường gặp khi dữ liệu y khoa có hạn.',
    params: R`\(${fmt(pTotal)} - ${fmt(1024 * 1000 + 1000)} = ${fmt(pBack)}\): cả DenseNet-121 trừ lớp FC 1000 lớp.`,
    code: `h = torch.flatten(F.adaptive_avg_pool2d(F.relu(model.features(x)), 1), 1)   # (B, 1024)`
  }, { tag: 'DenseNet-121', size: [0.9, 1.2, 1.2] }),
  n('Linear 1024 → 14', 'linear', '1024 → 14', {
    what: 'Thay lớp FC 1000 lớp. Mỗi hàng của W dò một bệnh: xẹp phổi, tim to, đông đặc, phù phổi, tràn dịch màng phổi, khí phế thũng, xơ phổi, thoát vị, thâm nhiễm, khối u, nốt, dày màng phổi, viêm phổi, tràn khí màng phổi.',
    formula: [R`z = W h + b, \qquad W \in \mathbb{R}^{14 \times 1024}`],
    params: R`\(1024 \times 14 + 14 = ${fmt(pHead)}\).`,
    code: `model.classifier = nn.Linear(1024, 14)`
  }, { tag: 'Linear 14', size: [ch(14), 0.9, 0.9] }),
  n('Sigmoid từng nhãn', 'act', '14 → 14', {
    what: 'Mỗi điểm số qua một sigmoid riêng, thành xác suất có bệnh đó. 14 xác suất không cần cộng lại bằng 1.',
    formula: [R`p_c = \sigma(z_c) = \frac{1}{1 + e^{-z_c}}, \qquad c = 1, \dots, 14`],
    why: 'Softmax buộc các lớp tranh nhau: một lớp tăng thì các lớp khác giảm. Điều đó hợp khi mỗi ảnh có đúng một nhãn, như ImageNet. Một phim X-quang có thể vừa tràn dịch vừa xẹp phổi, nên mỗi bệnh là một câu hỏi có hay không riêng.',
    example: R`\(z = [2,\ 0,\ -3]\): sigmoid cho \([0.881,\ 0.5,\ 0.047]\), tổng \(1.428\); softmax sẽ cho \([0.876,\ 0.118,\ 0.006]\), tổng đúng bằng 1.`,
    code: `probs = torch.sigmoid(logits)   # không phải softmax`
  }, { tag: 'Sigmoid', size: SLAB }),
  n('BCE có trọng số', 'act', '14 → loss', {
    what: R`Loss của CheXNet khi chỉ dò viêm phổi (một đầu ra): binary cross-entropy, nhưng lỗi trên ảnh có bệnh nhân với trọng số lớn \(w_{+}\), lỗi trên ảnh không bệnh nhân với trọng số nhỏ \(w_{-}\). \(|P|\) và \(|N|\) là số ảnh dương tính và âm tính trong tập train.`,
    formula: [R`L = -w_{+}\, y \log p - w_{-}\, (1 - y) \log (1 - p), \qquad p = p(Y{=}1 \mid X)`, R`w_{+} = \frac{|N|}{|P| + |N|}, \qquad w_{-} = \frac{|P|}{|P| + |N|}`,
      R`L_{14} = \sum_{c=1}^{14} \big[ -y_c \log p_c - (1 - y_c) \log (1 - p_c) \big]`],
    why: R`Bệnh càng hiếm thì ảnh âm tính càng áp đảo, và một mạng đoán "không bệnh" cho mọi ảnh đã được loss thấp (như tổn thương 1% ảnh ở khối đầu ra của U-Net). Trọng số bù lại: \(|P|\, w_{+} = |N|\, w_{-} = \frac{|P|\,|N|}{|P| + |N|}\), nên tổng trọng số của hai lớp bằng nhau. Khi mở rộng ra 14 bệnh, bài báo đổi sang tổng 14 BCE không trọng số (dòng thứ ba). Muốn vẫn cân bằng từng bệnh thì trong PyTorch đặt \(\texttt{pos_weight} = |N| / |P|\) cho mỗi bệnh: loss đó chỉ khác loss của bài một hằng số nhân \(w_{-}\), nên gradient cùng hướng.`,
    example: R`Giả sử ${ex.P} trên ${ex.P + ex.N} ảnh train có viêm phổi: \(w_{+} = ${ex.N}/${ex.P + ex.N} = ${wPos.toFixed(3)}\), \(w_{-} = ${ex.P}/${ex.P + ex.N} = ${wNeg.toFixed(3)}\). Tổng trọng số mỗi lớp: \(${ex.P} \times ${wPos.toFixed(3)} = ${(ex.P * wPos).toFixed(2)}\) và \(${ex.N} \times ${wNeg.toFixed(3)} = ${(ex.N * wNeg).toFixed(2)}\). Ảnh có bệnh mà mạng chỉ cho \(p = 0.2\): loss \(-${wPos.toFixed(3)} \log 0.2 = ${lossPos.toFixed(3)}\). Ảnh không bệnh mà mạng cho \(p = 0.2\): loss \(-${wNeg.toFixed(3)} \log 0.8 = ${lossNeg.toFixed(4)}\), nhỏ hơn khoảng ${Math.round(lossPos / lossNeg)} lần.`,
    code: `# một bệnh, như bài báo: w_pos = n_neg / (n_pos + n_neg), w_neg = n_pos / (n_pos + n_neg)
loss = -(w_pos * y * torch.log(p) + w_neg * (1 - y) * torch.log(1 - p)).mean()
# 14 bệnh, mỗi bệnh một trọng số (cùng hướng gradient, khác một hằng số nhân)
loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=n_neg / n_pos)   # n_neg, n_pos: (14,)`
  }, { tag: 'BCE (w)', size: SLAB })
] });

MODELS.densenet = n('DenseNet-121', 'block', '3×224×224 → 1000', {
  what: 'Mạng tích chập 121 lớp có trọng số cho phân loại ảnh ImageNet. Trong mỗi dense block, mỗi lớp nhận bản đồ đặc trưng của tất cả các lớp trước nó (nối theo chiều kênh) và chuyển output của mình cho tất cả các lớp sau. Giữa các khối là transition layer: conv 1×1 giảm kênh, average pool giảm kích thước.',
  formula: [R`\text{ResNet:}\quad x_\ell = H_\ell(x_{\ell-1}) + x_{\ell-1}`, R`\text{DenseNet:}\quad x_\ell = H_\ell\big([x_0, x_1, \dots, x_{\ell-1}]\big)`],
  why: 'So với ResNet, chỉ khác ở chỗ nối thay vì cộng, nhưng hệ quả rất khác. Cộng thì đặc trưng mới trộn vào đặc trưng cũ trong cùng các kênh; nối thì mọi đặc trưng đã học vẫn còn nguyên để lớp sau đọc thẳng, nên mỗi lớp chỉ cần thêm k = 32 kênh. Nhờ dùng lại đặc trưng, DenseNet đạt cùng độ chính xác với ít tham số hơn hẳn. Mở một dense block: các đường cong "nối (concat)" là chồng kênh cũ đi vòng qua lớp để nối với kênh mới.',
  example: 'Bảng 3 của bài gốc, lỗi top-1 trên ImageNet val (1 crop / 10 crop): DenseNet-121 25.02% / 23.61%. Hình 3: DenseNet-201 với 20M tham số có lỗi tương đương ResNet-101 với hơn 40M. Bảng 2 của bài tổng quan ảnh y khoa (Mienye và cộng sự, 2025) ghi DenseNet cho bệnh võng mạc tiểu đường (độ chính xác 86%) và DenseNet-121 cho gãy xương hông (86.5%).',
  params: R`\(${fmt(pTotal)}\) (khoảng 8M), chưa bằng 1/3 ResNet-50 (25.6M). 121 lớp = 1 conv đầu + 58 lớp dense × 2 conv + 3 transition + 1 FC.`,
  code: `from torchvision.models import densenet121, DenseNet121_Weights
model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).eval()
probs = model(x).softmax(-1)   # (B, 1000)`
}, { label: 'DenseNet-121', noIn: true, outLabel: '1000 lớp', ref: [ref('densenet', 'Bảng 1 và 3, Hình 3'), ref('mienye', 'mục 5.5, Bảng 1 và 2')], children: [
  imageIn, stem, ...body,
  n('Đầu phân loại', 'block', '1024×7×7 → 1000', {
    what: 'BN và ReLU cuối, global average pooling, rồi một lớp FC ra 1000 lớp.',
    formula: [R`p = \mathrm{softmax}\Big(W\, \mathrm{GAP}\big(\mathrm{ReLU}(\mathrm{BN}(x))\big) + b\Big)`],
    why: 'Phần duy nhất của DenseNet nhìn toàn bộ ảnh một lúc. Mọi thứ trước đó đều là conv, chỉ nhìn cục bộ.',
    params: R`\(2 \cdot 1024 + 1024 \cdot 1000 + 1000 = 1\,027\,048\).`,
    code: `x = F.relu(model.features.norm5(x))          # norm5 là BN cuối
z = torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)
logits = model.classifier(z)                   # nn.Linear(1024, 1000)`
  }, { tag: 'Đầu ra', size: [ch(1000), 0.22, 0.22], children: [
  bn(1024, 7, ' Đây là BN cuối (norm5): mỗi lớp dense kết thúc bằng conv, nên chồng kênh ra khỏi khối 4 chưa được chuẩn hoá; cần một BN và ReLU trước khi pool, giống ResNet v2.'),
  relu('1024×7×7', slab(7)),
  gapNode(1024, 7, 'Không có tham số. Chồng 1024 kênh này gồm 512 kênh từ transition 3 (bản tóm tắt các khối trước) và 16 × 32 kênh do từng lớp của khối 4 viết ra, nên bộ phân loại quyết định dựa trên đặc trưng của cả mạng chứ không chỉ của lớp cuối.'),
  fcOut(1024, 'classifier')
  ] }),
  chexnet // after DenseNet's own blocks, so links to those (#densenet.i) keep their numbers
] });
})();
