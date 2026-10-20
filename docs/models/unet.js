// U-Net (Ronneberger, Fischer & Brox, 2015): unpadded 3×3 convs, 572×572 grayscale in → 388×388 two-class map out.
(() => {
const pConv = (a, b, k = 3) => k * k * a * b + b;

// first = first conv of its level: only that one runs the live toy conv (the second would need the first's output)
const convV = (cin, cout, h, first) => n('Conv 3×3 (không đệm)', 'conv', `${cin}×${h}×${h} → ${cout}×${h - 2}×${h - 2}`, {
  what: 'Conv 3×3 không đệm (valid): bộ lọc chỉ được đặt ở những chỗ nằm trọn trong ảnh, nên output nhỏ hơn input 2 pixel mỗi chiều. ResNet thì đệm 1 để giữ nguyên kích thước.',
  formula: CONV_F,
  why: 'Không đệm nghĩa là không pixel output nào bị ảnh hưởng bởi các số 0 đệm ở viền. Nhờ vậy khi chia một ảnh lớn thành nhiều ô rồi ghép kết quả lại, chỗ nối giữa các ô không bị lộ. Đổi lại ảnh co dần qua mỗi conv, và đường tắt phải được cắt bớt cho khớp.',
  example: R`\(k = 3,\ p = 0,\ s = 1\): \(\lfloor (${h} - 3) / 1 \rfloor + 1 = ${h - 2}\).`,
  params: R`\(${cin} \times ${cout} \times 3 \times 3 + ${cout} = ${fmt(pConv(cin, cout))}\) (có bias; bản gốc chưa dùng BatchNorm).`,
  code: `nn.Conv2d(${cin}, ${cout}, kernel_size=3)   # padding=0`
}, { tag: 'Conv 3×3', size: vol(cout, h - 2), ...(first ? { live: 'conv', conv: { pad: 0, stride: 1 } } : {}) });

const pool = (c, h) => n('MaxPool 2×2, bước 2', 'pool', `${c}×${h}×${h} → ${c}×${h / 2}×${h / 2}`, {
  what: 'Lấy giá trị lớn nhất trong mỗi ô 2×2, giảm mỗi chiều một nửa. Đây là mũi tên đi xuống trong hình chữ U.',
  formula: [R`y_{c,i,j} = \max_{u, v \in \{0, 1\}} x_{c,\; 2i + u,\; 2j + v}`],
  why: 'Giảm kích thước một nửa thì mỗi conv phía sau nhìn được vùng ảnh gốc rộng gấp đôi. Bài gốc chọn kích thước input sao cho mọi lần pool đều rơi vào cạnh có độ dài chẵn, để không mất pixel nào và nhánh lên ghép lại khớp.',
  example: R`\(${h} / 2 = ${h / 2}\).`,
  code: `nn.MaxPool2d(kernel_size=2)`
}, { tag: 'MaxPool ↓', size: vol(c, h / 2) });

const upconv = (cin, h) => n('Up-conv 2×2, bước 2', 'conv', `${cin}×${h}×${h} → ${cin / 2}×${2 * h}×${2 * h}`, {
  what: `Conv chuyển vị (transposed conv): mỗi pixel input "rải" ra một ô 2×2 ở output theo bộ trọng số học được. Kích thước gấp đôi, số kênh giảm một nửa (${cin} → ${cin / 2}). Đây là mũi tên đi lên trong hình chữ U.`,
  formula: [R`y_{o,\; 2i + u,\; 2j + v} = b_o + \sum_{c} W_{c,o,u,v}\; x_{c,i,j}, \qquad u, v \in \{0, 1\}`, R`H_{out} = (H_{in} - 1)\, s - 2p + k`],
  why: 'Phóng to theo cách mạng tự học thay vì nội suy cố định (bilinear). Với kernel 2 và bước 2, các ô 2×2 không chồng lên nhau nên không bị vân bàn cờ (checkerboard), lỗi hay gặp khi kernel không chia hết cho bước.',
  example: R`\((${h} - 1) \cdot 2 - 0 + 2 = ${2 * h}\).`,
  params: R`\(${cin} \times ${cin / 2} \times 2 \times 2 + ${cin / 2} = ${fmt(cin * (cin / 2) * 4 + cin / 2)}\).`,
  code: `nn.ConvTranspose2d(${cin}, ${cin / 2}, kernel_size=2, stride=2)`
}, { tag: 'Up-conv ↑', size: vol(cin / 2, 2 * h), ref: [ref('odena')] });

const cropCat = (ce, he, cu, h) => n('Cắt và nối (copy & crop)', 'shape', `${ce}×${he}×${he} + ${cu}×${h}×${h} → ${ce + cu}×${h}×${h}`, {
  what: R`Cắt lấy phần giữa \(${h} \times ${h}\) của bản đồ đặc trưng cùng tầng bên nhánh xuống (\(${he} \times ${he}\), bỏ ${(he - h) / 2} pixel mỗi viền), rồi nối theo chiều kênh với output của up-conv: \(${ce} + ${cu} = ${ce + cu}\) kênh.`,
  formula: [R`z = \big[\, \mathrm{crop}(e)\, ;\ u \,\big] \in \mathbb{R}^{${ce + cu} \times ${h} \times ${h}}`],
  why: 'ResNet cộng đường tắt vào, U-Net thì nối. Cộng buộc hai nguồn phải cùng ý nghĩa ở từng kênh; nối giữ chúng riêng rẽ để conv phía sau tự học cách trộn. Kênh từ nhánh xuống mang vị trí chính xác (chi tiết, đường viền); kênh từ nhánh lên mang ngữ cảnh (đây là cái gì). Phải cắt vì conv không đệm làm bản đồ bên nhánh xuống to hơn.',
  code: `d = (e.shape[-1] - u.shape[-1]) // 2                       # ${(he - h) / 2}
z = torch.cat([e[..., d:d + ${h}, d:d + ${h}], u], dim=1)     # (B, ${ce + cu}, ${h}, ${h})`
}, { tag: 'Cắt + nối', size: vol(ce + cu, h), ref: [ref('unet', 'mục 2')] });

// contracting path: level k reads h×h (already pooled for k > 0), writes cout×(h − 4)
function down(k, cin, cout, hin) {
  const h = k ? hin / 2 : hin, p = pConv(cin, cout) + pConv(cout, cout), bottom = k === 4;
  return n(bottom ? 'Đáy chữ U' : `Nhánh xuống, tầng ${k + 1}`, 'conv', `${cin}×${hin}×${hin} → ${cout}×${h - 4}×${h - 4}`, {
    what: (k ? `Max-pool 2×2 giảm kích thước một nửa, rồi hai conv 3×3 + ReLU và tăng số kênh lên ${cout}.` : 'Hai conv 3×3 không đệm, mỗi conv theo sau bởi ReLU, ở độ phân giải đầy đủ: học các đặc trưng cấp thấp như cạnh và kết cấu.')
      + (bottom ? ' Đây là độ phân giải nhỏ nhất và nhiều kênh nhất của mạng; bài gốc đặt dropout ở cuối nhánh xuống, tức ở đây.' : ' Càng xuống sâu, mỗi pixel càng nhìn được vùng rộng hơn của ảnh gốc nhưng vị trí càng kém chính xác.'),
    formula: [k ? R`e_${k + 1} = \mathrm{ReLU}\big(W_2 * \mathrm{ReLU}(W_1 * \mathrm{MaxPool}(e_${k}))\big)` : R`e_1 = \mathrm{ReLU}\big(W_2 * \mathrm{ReLU}(W_1 * x)\big)`],
    example: (k ? R`\(${hin} \to ${h} \to ${h - 2} \to ${h - 4}\).` : R`\(${hin} \to ${h - 2} \to ${h - 4}\).`) + (bottom ? ' Vùng nhìn (receptive field) của mỗi pixel ở đây là 140 × 140 pixel ảnh gốc.' : ''),
    params: R`\(${fmt(pConv(cin, cout))} + ${fmt(pConv(cout, cout))} = ${fmt(p)}\).`,
    code: (k ? `x = F.max_pool2d(x, 2)\n` : '') + `x = F.relu(self.conv${k + 1}a(x))   # nn.Conv2d(${cin}, ${cout}, 3)
x = F.relu(self.conv${k + 1}b(x))   # nn.Conv2d(${cout}, ${cout}, 3)`
  }, { tag: bottom ? 'Đáy' : `Xuống ${k + 1}`, size: vol(cout, h - 4), _p: p, _out: h - 4, children: [
    ...(k ? [pool(cin, hin)] : []),
    convV(cin, cout, h, true), relu(`${cout}×${h - 2}×${h - 2}`, [0.14, sp(h - 2), sp(h - 2)]),
    convV(cout, cout, h - 2), relu(`${cout}×${h - 4}×${h - 4}`, [0.14, sp(h - 4), sp(h - 4)])
  ] });
}

// expansive path: up-conv cin → cin/2, concat with cropped skip (cin/2 channels), two convs to cin/2
function up(k, cin, hin, he) {
  const c = cin / 2, h = 2 * hin, p = cin * c * 4 + c + pConv(cin, c) + pConv(c, c);
  return n(`Nhánh lên, tầng ${k + 1}`, 'conv', `${cin}×${hin}×${hin} → ${c}×${h - 4}×${h - 4}`, {
    what: `Up-conv 2×2 phóng to gấp đôi và giảm nửa số kênh, nối với bản đồ đặc trưng cùng tầng bên nhánh xuống, rồi hai conv 3×3 + ReLU.`,
    formula: [R`d_${k + 1} = \mathrm{ReLU}\Big(W_2 * \mathrm{ReLU}\big(W_1 * [\, \mathrm{crop}(e_${k + 1})\, ;\ \mathrm{Up}(d_${k + 2}) \,]\big)\Big)`],
    example: R`\(${hin} \xrightarrow{\text{up}} ${h}\), nối với \(\mathrm{crop}_{${he} \to ${h}}(e_${k + 1})\), rồi \(${h} \to ${h - 2} \to ${h - 4}\).`,
    params: R`\(${fmt(p)}\).`,
    code: `u = self.up${k + 1}(x)                    # ConvTranspose2d(${cin}, ${c}, 2, stride=2)
x = crop_cat(e${k + 1}, u)                # (B, ${cin}, ${h}, ${h})
x = F.relu(self.conv${k + 1}c(x))         # nn.Conv2d(${cin}, ${c}, 3)
x = F.relu(self.conv${k + 1}d(x))         # nn.Conv2d(${c}, ${c}, 3)`
  }, { tag: `Lên ${k + 1}`, size: vol(c, h - 4), _p: p, _h: h, hint: '6 phần',
    extra: [{ a: { ext: 1, s: 'b', d: 1.6 }, b: { i: 1, s: 'b' }, kind: 'concat', label: `từ Xuống ${k + 1}` }],
    children: [
      upconv(cin, hin), cropCat(c, he, c, h),
      convV(cin, c, h, true), relu(`${c}×${h - 2}×${h - 2}`, [0.14, sp(h - 2), sp(h - 2)]),
      convV(c, c, h - 2), relu(`${c}×${h - 4}×${h - 4}`, [0.14, sp(h - 4), sp(h - 4)])
    ] });
}

const D = [down(0, 1, 64, 572)];
[[64, 128], [128, 256], [256, 512], [512, 1024]].forEach(([a, b], j) => D.push(down(j + 1, a, b, D[j]._out)));
const U = [];
for (let k = 3, cin = 1024, h = D[4]._out; k >= 0; k--) { U[k] = up(k, cin, h, D[k]._out); h = 2 * h - 4; cin /= 2; }
const pTotal = D.reduce((t, d) => t + d._p, 0) + U.reduce((t, u) => t + u._p, 0) + pConv(64, 2, 1);
// self-check: unpadded U-Net of the paper, no BatchNorm (31 030 658 parameters)
console.assert(pTotal === 31030658 && U[0].shape.endsWith('64×388×388'), 'U-Net size mismatch');

const Y = [5.2, 2.6, 0, -2.6, -5.2], X = [-9, -7, -5, -3, 0];
const crop = k => `cắt ${D[k]._out}→${U[k]._h}`;

MODELS.unet = n('U-Net (2015)', 'conv', '1×572×572 → 2×388×388', {
  what: 'Mạng phân vùng ảnh (segmentation): gán cho mỗi pixel một nhãn, ở đây là "tế bào" hay "đường viền" trong ảnh kính hiển vi điện tử. Nhánh xuống bên trái giảm kích thước để hiểu ảnh chứa gì; nhánh lên bên phải phóng to lại để biết chính xác ở đâu; các đường nối ngang mang chi tiết sắc nét từ nhánh xuống sang nhánh lên.',
  formula: [R`p_k(x) = \frac{e^{a_k(x)}}{\sum_{k'=1}^{2} e^{a_{k'}(x)}} \quad \text{tại mỗi pixel } x`, R`E = -\sum_{x} w(x)\, \log p_{\ell(x)}(x)`],
  why: 'Chỉ có nhánh xuống thì mạng biết "có tế bào" nhưng mất vị trí chính xác; chỉ giữ độ phân giải cao thì mỗi pixel không nhìn được đủ rộng. Đường nối ngang ghép cả hai. Bài gốc train được chỉ với 30 ảnh, nhờ tăng cường dữ liệu bằng biến dạng đàn hồi (elastic deformation). Cũng khung chữ U này là xương sống của Stable Diffusion.',
  example: R`Input \(572 \times 572\) → output \(388 \times 388\): mỗi conv 3×3 không đệm làm mất 2 pixel mỗi chiều. Để phân vùng ảnh lớn, bài gốc cắt ảnh thành nhiều ô, mỗi ô lấy thêm \((572 - 388) / 2 = 92\) pixel ngữ cảnh mỗi viền; phần nằm ngoài ảnh được lấp bằng cách phản chiếu gương.`,
  params: R`\(${fmt(pTotal)}\) (khoảng 31M). Kênh: \(64 \to 128 \to 256 \to 512 \to 1024\) rồi ngược lại.`,
  code: `# phác thảo forward; block(a, b) = 2 × (Conv2d(·, b, 3) + ReLU), không đệm
e1 = block(1, 64)(x)                          # 64×568×568
e2 = block(64, 128)(F.max_pool2d(e1, 2))      # 128×280×280
e3 = block(128, 256)(F.max_pool2d(e2, 2))     # 256×136×136
e4 = block(256, 512)(F.max_pool2d(e3, 2))     # 512×64×64
b  = block(512, 1024)(F.max_pool2d(e4, 2))    # 1024×28×28
d4 = block(1024, 512)(crop_cat(e4, up4(b)))   # 512×52×52
d3 = block(512, 256)(crop_cat(e3, up3(d4)))   # 256×100×100
d2 = block(256, 128)(crop_cat(e2, up2(d3)))   # 128×196×196
d1 = block(128, 64)(crop_cat(e1, up1(d2)))    # 64×388×388
logits = nn.Conv2d(64, 2, 1)(d1)              # 2×388×388`
}, { label: 'U-Net', layout: 'manual', ref: [ref('unet')], children: [
  { ...n('Ảnh vào', 'io', '1×572×572', {
    what: 'Ảnh kính hiển vi điện tử xám, 1 kênh (bộ dữ liệu ISBI 2012). Vì mọi conv đều không đệm, input phải lớn hơn output 184 pixel mỗi chiều.',
    why: 'Phần viền thêm vào chính là ngữ cảnh mà các pixel ở rìa vùng output cần nhìn. Ở rìa ảnh thật, ngữ cảnh được lấp bằng cách phản chiếu gương nội dung bên trong.',
    code: `x = F.pad(img, (92, 92, 92, 92), mode="reflect")   # 388 + 2·92 = 572`
  }, { tag: 'Ảnh', size: [0.06, sp(572), sp(572)] }), pos: [-12.4, Y[0], 0] },
  ...D.map((d, k) => ({ ...d, pos: [X[k], Y[k], 0] })),
  ...[3, 2, 1, 0].map(k => ({ ...U[k], pos: [-X[k], Y[k], 0] })),
  { ...n('Conv 1×1 → 2 lớp', 'conv', '64×388×388 → 2×388×388', {
    what: 'Conv 1×1 biến 64 kênh ở mỗi pixel thành 2 điểm số: tế bào hay nền (đường viền), rồi softmax theo từng pixel.',
    formula: [R`a(x) = W h(x) + b, \qquad W \in \mathbb{R}^{2 \times 64}`, R`w(x) = w_c(x) + w_0 \exp\!\left(-\frac{\big(d_1(x) + d_2(x)\big)^2}{2\sigma^2}\right), \qquad w_0 = 10,\ \sigma \approx 5`,
      R`\mathrm{DSC} = \frac{2\,|P \cap G|}{|P| + |G|}`, R`\mathrm{IoU} = \frac{|P \cap G|}{|P \cup G|} = \frac{\mathrm{DSC}}{2 - \mathrm{DSC}}`],
    why: R`Các tế bào nằm sát nhau chỉ cách bởi một đường nền rất mảnh. Bản đồ trọng số \(w(x)\) phạt nặng lỗi ở những pixel nền nằm giữa hai tế bào gần nhau (\(d_1, d_2\) là khoảng cách tới tế bào gần nhất và gần nhì), buộc mạng học tách chúng ra. \(w_c\) cân bằng tần suất hai lớp. Để chấm kết quả phân vùng, các bài y khoa thường dùng hệ số Dice (DSC) giữa vùng dự đoán \(P\) và vùng nhãn \(G\): nó chỉ đếm phần dương, nên phần nền chiếm gần hết ảnh không làm đẹp số được như độ chính xác từng pixel. Khối u chiếm 1% ảnh thì đoán toàn nền đã được 99% độ chính xác pixel, nhưng DSC = 0. V-Net dùng chính Dice (bản mềm, xác suất thay cho 0/1) làm hàm loss.`,
    example: R`Dự đoán 4 pixel là tế bào, nhãn thật có 5 pixel, trùng nhau 3: \(\mathrm{DSC} = 2 \cdot 3 / (4 + 5) = 0.667\), \(\mathrm{IoU} = 3 / (4 + 5 - 3) = 0.5\). Bảng 2 của bài tổng quan ảnh y khoa (Mienye và cộng sự, 2025): U-Net phân vùng tổn thương sụn khớp trên MRI đạt DSC 92% (Wirth và cộng sự).`,
    params: R`\(64 \times 2 + 2 = 130\).`,
    code: `logits = nn.Conv2d(64, 2, kernel_size=1)(d1)                     # (B, 2, 388, 388)
loss = (w * F.cross_entropy(logits, target, reduction="none")).sum()
pred = logits.argmax(1) == 1                                       # chấm điểm: Dice với nhãn gt
dice = 2 * (pred & gt).sum() / (pred.sum() + gt.sum())`
  }, { tag: 'Conv 1×1', size: [0.12, sp(388), sp(388)], ref: [ref('unet', 'mục 3, bản đồ trọng số'), ref('vnet', 'Dice làm hàm loss'), ref('mienye', 'Bảng 2: DSC')] }), pos: [12.2, Y[0], 0] }
], links: [
  { a: { i: 0, s: 'r' }, b: { i: 1, s: 'l' } },
  ...[1, 2, 3, 4].map(i => ({ a: { i, s: 'd' }, b: { i: i + 1, s: 'l' } })),
  { a: { i: 5, s: 'r' }, b: { i: 6, s: 'd' } },
  ...[6, 7, 8].map(i => ({ a: { i, s: 'r' }, b: { i: i + 1, s: 'd' } })),
  { a: { i: 9, s: 'r' }, b: { i: 10, s: 'l' } },
  { a: { i: 10, s: 'r' }, b: { stub: 10, s: 'r' }, label: '2 lớp' },
  ...[0, 1, 2, 3].map(k => ({ a: { i: k + 1, s: 'r' }, b: { i: 9 - k, s: 'l' }, kind: 'concat', label: crop(k) }))
] });
})();
