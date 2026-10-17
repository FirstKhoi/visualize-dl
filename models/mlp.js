// Training lab: the smallest network with every piece of deep learning, trained live in the page (TRAIN in live.js).
// live: 'W1' | 'W2' | 'W3' paint the real weight matrices over the blocks, 'boundary' the decision boundary; exp: 'train' = the lab panel.
(() => {
const linear = (name, tag, k, shape, what, why, extra = {}) => n(name, 'linear', shape, {
  what,
  formula: [R`z_${k} = a_${k - 1} W_${k} + b_${k}`, R`\frac{\partial L}{\partial W_${k}} = a_${k - 1}^\top\, \delta_${k}, \qquad \frac{\partial L}{\partial b_${k}} = \textstyle\sum_i \delta_${k}, \qquad \delta_{\text{trước}} = \delta_${k}\, W_${k}^\top`],
  why,
  code: `z${k} = a${k - 1} @ W${k} + b${k}           # forward
dW${k} = a${k - 1}.T @ d${k} / n          # backward: gradient của trọng số
d_prev = d${k} @ W${k}.T              # gradient chảy ngược sang lớp trước`
}, { tag, live: 'W' + k, size: [1.2, 1.5, 1.3], ...extra });

const activation = i => n(`Kích hoạt ${i}`, 'act', 'n×H', {
  what: 'ReLU hoặc tanh tuỳ lựa chọn trong phòng thí nghiệm, áp riêng cho từng phần tử.',
  formula: [R`a = f(z), \qquad \delta_z = \delta_a \odot f'(z)`, R`\mathrm{ReLU}'(z) = \mathbb{1}[z > 0], \qquad \tanh'(z) = 1 - \tanh^2(z)`],
  why: 'Lớp Linear trước nó chia mặt phẳng bằng các đường thẳng; phi tuyến bẻ cong và ghép các đường đó lại thành ranh giới cong. ReLU có thể "chết": nếu z luôn âm với mọi điểm, đạo hàm bằng 0 mãi và neuron không học nữa. Đẩy learning rate lên 3 để thấy cả mạng chết, loss kẹt ở ln 2 = 0.693.',
  code: `a = torch.relu(z)                    # forward
d_z = d_a * (z > 0)                  # backward qua ReLU`
}, { tag: i === 1 ? 'f(z₁)' : 'f(z₂)', size: SLAB });

MODELS.mlp = n('MLP: phòng thí nghiệm huấn luyện', 'block', '2 toạ độ → 2 lớp', {
  what: 'Mạng nhỏ nhất có đủ các thành phần của học sâu: các lớp Linear xen kẽ hàm kích hoạt, softmax và cross-entropy ở cuối, và một bộ tối ưu cập nhật trọng số theo gradient. Mạng được train ngay trong trang: forward tính dự đoán, backward tính gradient bằng quy tắc chuỗi từ lớp cuối ngược về lớp đầu, rồi bộ tối ưu bước một bước nhỏ xuống dốc của loss.',
  formula: [
    R`p = \mathrm{softmax}\big(f(f(x W_1 + b_1)\, W_2 + b_2)\, W_3 + b_3\big), \qquad L = -\frac{1}{n}\sum_i \log p_{i,\,y_i}`,
    R`\text{SGD:}\quad \theta \leftarrow \theta - \eta\, \nabla_\theta L`,
    R`\text{Momentum:}\quad v \leftarrow 0.9\, v + \nabla_\theta L, \qquad \theta \leftarrow \theta - \eta\, v`,
    R`\text{Adam:}\quad m \leftarrow \beta_1 m + (1 - \beta_1) g, \quad v \leftarrow \beta_2 v + (1 - \beta_2) g^2, \quad \theta \leftarrow \theta - \eta\, \frac{\hat m}{\sqrt{\hat v} + \epsilon}`
  ],
  why: 'Learning rate η là núm quan trọng nhất: quá nhỏ thì học rất chậm, quá lớn thì mỗi bước nhảy qua cả thung lũng của loss. Adam tự co giãn bước đi cho từng tham số theo độ lớn gradient gần đây, nên ít nhạy với η hơn SGD. Lớp ẩn và phi tuyến là thứ cho phép mạng vẽ ranh giới cong; số neuron quyết định ranh giới phức tạp được đến đâu.',
  example: 'Đo trên chính trang này (bộ "Vòng tròn", 8 neuron, ReLU): Adam với η = 0.03 đưa loss xuống dưới 0.1 sau 41 bước; SGD cùng η sau 600 bước vẫn còn 0.145; η = 3 làm ReLU chết hết, loss kẹt ở 0.693 và độ chính xác 50%. Bộ "Xoắn ốc": 8 neuron kẹt quanh 94%, 16 neuron đạt 100% sau khoảng 180 bước.',
  params: R`\(2H + H + H^2 + H + 2H + 2\): với \(H = 8\) là 114 tham số, \(H = 16\) là 338.`,
  code: `model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
opt = torch.optim.Adam(model.parameters(), lr=0.03)
for step in range(1000):
    loss = F.cross_entropy(model(X), y)   # forward + loss
    opt.zero_grad()
    loss.backward()                       # backprop: tính mọi gradient
    opt.step()                            # cập nhật trọng số`
}, { label: 'Huấn luyện', noIn: true, outLabel: 'loss', exp: 'train',
  ref: [ref('rumelhart'), ref('adam'), ref('momentum'), ref('heinit', 'khởi tạo trọng số'), ref('playground')],
  children: [
    n('Điểm vào (x, y)', 'io', 'n×2', {
      what: '200 điểm trên mặt phẳng, mỗi điểm có 2 toạ độ và một nhãn 0 hoặc 1. Mạng phải học một ranh giới tách hai lớp.',
      why: 'Cả 200 điểm đi qua mạng trong mỗi bước (full batch), nên đường loss trơn. Mạng thật thường dùng mini-batch: nhanh hơn, và nhiễu của nó giúp thoát khỏi điểm tồi.',
      code: `X = torch.tensor(points)   # (200, 2)
y = torch.tensor(labels)   # (200,), 0 hoặc 1`
    }, { tag: 'Điểm (x, y)', size: [0.5, 1.4, 0.5] }),
    linear('Linear 2 → H', 'W₁', 1, 'n×2 → n×H',
      'Mỗi neuron ẩn tính một tổ hợp tuyến tính của x và y. Về hình học, mỗi neuron là một đường thẳng chia mặt phẳng làm hai.',
      R`Heatmap trên khối là ma trận \(W_1\) thật (2 hàng × H cột), đổi màu khi mạng học: hồng là dương, xanh là âm. Khởi tạo He: \(W \sim \mathcal{N}(0, 2/\text{fan\_in})\), để độ lớn tín hiệu không teo đi hay phình ra qua các lớp ReLU.`),
    activation(1),
    linear('Linear H → H', 'W₂', 2, 'n×H → n×H',
      'Lớp ẩn thứ hai ghép các đường thẳng của lớp đầu thành hình phức tạp hơn.',
      R`Khi lan truyền ngược, gradient đi qua đúng ma trận này nhưng chuyển vị: \(\delta_1 = \delta_2 W_2^\top\). Đây là lý do độ lớn của W quyết định gradient tắt dần hay bùng nổ (xem thí nghiệm gradient ở ResNet).`),
    activation(2),
    linear('Linear H → 2', 'W₃', 3, 'n×H → n×2',
      'Lớp cuối cho 2 logit, mỗi lớp một điểm số.',
      R`Hai cột của \(W_3\) là hai "mẫu" mà trạng thái ẩn được so với; lớp nào khớp hơn thì logit lớn hơn.`),
    n('Softmax + Cross-entropy', 'act', 'n×2 → loss', {
      what: 'Đổi 2 logit thành xác suất, rồi phạt bằng −log của xác suất gán cho nhãn đúng. Heatmap trên khối là ranh giới quyết định thật của mạng.',
      formula: [R`p = \mathrm{softmax}(z_3), \qquad L = -\log p_y`, R`\delta_3 = \frac{\partial L}{\partial z_3} = p - \mathrm{onehot}(y)`],
      why: 'Gradient của softmax + cross-entropy gọn đến bất ngờ: chỉ là xác suất trừ nhãn. Đoán đúng và chắc chắn thì gradient gần 0; đoán sai thì gradient lớn, kéo mạnh các trọng số phía trước.',
      example: R`\(p = [0.3,\ 0.7]\), nhãn \(y = 1\): \(L = -\log 0.7 = 0.357\), \(\delta_3 = [0.3,\ -0.3]\).`,
      code: `loss = F.cross_entropy(logits, y)    # gộp softmax + −log
d3 = (logits.softmax(-1) - F.one_hot(y, 2)) / n`
    }, { tag: 'Softmax + CE', live: 'boundary', size: [0.4, 1.5, 1.3], exp: 'train' })
  ] });
})();
