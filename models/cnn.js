// A small CNN trained on MNIST that really runs in the page: draw a digit and every block shows its true activations.
// Weights: models/weights/mnist-cnn.js (from training/train_mnist_cnn.py); forward pass and preprocessing: CNN in live.js.
(() => {
const W = MNIST_CNN, pc = (cin, cout) => 9 * cin * cout + cout; // a 3×3 conv's weights and biases
const P = { conv1: pc(1, 8), conv2: pc(8, 16), fc: 400 * 10 + 10 };
const acc = `${(W.test_acc * 100).toFixed(2)}%`;
// self-checks: the counts below are the trained network's, and the sample digits are ones it gets right
console.assert(P.conv1 + P.conv2 + P.fc === W.params && Object.values(W.weights).reduce((t, w) => t + w.data.length, 0) === W.params, 'MNIST CNN parameter count mismatch');
console.assert(W.samples.every(s => { const p = CNN.run(CNN.decode(s.pixels)).probs; return p.indexOf(Math.max(...p)) === s.label; }), 'MNIST CNN misreads a sample digit');

const conv = (name, cin, cout, H, live, what, why) => n(name, 'conv', `${cin}×${H}×${H} → ${cout}×${H - 2}×${H - 2}`, {
  what,
  formula: CONV_F,
  why,
  example: R`\(k = 3,\ p = 0,\ s = 1\): \(\lfloor (${H} + 0 - 3) / 1 \rfloor + 1 = ${H - 2}\).`,
  params: R`\(${cout} \times ${cin} \times 3 \times 3 + ${cout} = ${fmt(pc(cin, cout))}\).`,
  code: `nn.Conv2d(${cin}, ${cout}, kernel_size=3)   # không đệm: ${H} → ${H - 2}`
}, { tag: 'Conv 3×3', size: vol(cout, H - 2), live });

const pool = (k, C, H) => n(`MaxPool 2×2 (${k})`, 'pool', `${C}×${H}×${H} → ${C}×${H >> 1}×${H >> 1}`, {
  what: 'Giữ số lớn nhất trong mỗi ô 2×2 không chồng lên nhau: bản đồ nhỏ đi một nửa mỗi chiều, số kênh giữ nguyên.',
  formula: [R`y_{c,i,j} = \max_{u,\,v \in \{0,1\}} x_{c,\,2i+u,\,2j+v}`],
  why: 'Một nét xê dịch một pixel thường vẫn nằm trong cùng ô 2×2, nên đầu ra gần như không đổi: mạng bớt nhạy với vị trí chính xác. Lớp này không có tham số. Khi train, gradient chỉ chảy về đúng ô lớn nhất của mỗi cửa sổ.',
  example: H % 2 ? R`\(\lfloor ${H} / 2 \rfloor = ${H >> 1}\): hàng và cột thứ ${H} bị bỏ, như \(\texttt{nn.MaxPool2d}\).` : R`\(${H} / 2 = ${H >> 1}\).`,
  code: 'x = F.max_pool2d(x, 2)'
}, { tag: 'Pool', size: vol(C, H >> 1), live: k === 1 ? 'cnnP1' : 'cnnP2' });

const slab = H => [0.12, sp(H), sp(H)];

MODELS.cnn = n('CNN nhỏ trên MNIST', 'conv', '1×28×28 → 10', {
  what: `Mạng tích chập nhỏ đã train thật trên 60 000 chữ số viết tay MNIST, chạy ngay trong trang: vẽ một chữ số và mọi khối hiện activation thật. Hai lớp conv 3×3 học các nét, hai lớp max-pool thu nhỏ bản đồ, một lớp Linear chấm điểm 10 chữ số. Đúng ${acc} trên 10 000 ảnh test mạng chưa từng thấy.`,
  formula: [
    R`z = W_{fc}\, \mathrm{flatten}\Big(\mathrm{pool}\big(\mathrm{ReLU}(W_2 * \mathrm{pool}(\mathrm{ReLU}(W_1 * x + b_1)) + b_2)\big)\Big) + b_{fc}`,
    R`p = \mathrm{softmax}(z), \qquad \hat y = \arg\max_d\ p_d`
  ],
  why: `Vì sao conv mà không phải MLP: cùng một bộ lọc 3×3 trượt qua mọi vị trí (dùng chung trọng số), nên một nét ngang ở góc nào cũng được nhận ra như nhau, và số tham số nhỏ hẳn: mạng này chỉ có ${fmtText(W.params)}. Max-pool làm mạng ít nhạy với xê dịch nhỏ. Đây là ý tưởng của LeNet (LeCun và cộng sự, 1998), tổ tiên của ResNet và ConvNeXt trong trang này.`,
  example: `Train bằng NumPy với backprop viết tay (đã kiểm tra bằng sai phân), Adam η = 0.001, batch 64, 6 epoch, mỗi ảnh dịch ngẫu nhiên tới 2 pixel: đúng ${acc} trên 10 000 ảnh test, mất khoảng 2 phút trên CPU. Script: training/train_mnist_cnn.py.`,
  params: R`\(${P.conv1} + ${fmt(P.conv2)} + ${fmt(P.fc)} = ${fmt(W.params)}\): conv1 \(8 \times 1 \times 9 + 8\), conv2 \(16 \times 8 \times 9 + 16\), Linear \(10 \times 400 + 10\). Ba phần tư nằm ở lớp Linear cuối.`,
  code: `class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.fc = nn.Linear(16 * 5 * 5, 10)

    def forward(self, x):                         # (B, 1, 28, 28), pixel trong [0, 1]
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # (B, 8, 13, 13)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # (B, 16, 5, 5)
        return self.fc(torch.flatten(x, 1))          # (B, 10)`
}, { label: 'CNN (MNIST)', noIn: true, outLabel: 'chữ số', live: 'cnn', ref: [ref('lecun98', 'LeNet-5 và MNIST'), ref('mnist'), ref('adam')], children: [
  n('Ảnh vào 28×28', 'io', '1×28×28', {
    what: 'Chữ số bạn vẽ, đưa về đúng dạng ảnh MNIST: lấy khung bao quanh nét, thu vào ô 20×20 giữ nguyên tỉ lệ, rồi đặt vào ảnh 28×28 sao cho trọng tâm của nét nằm ở giữa. Mỗi pixel là một số từ 0 (nền) đến 1 (mực).',
    formula: [R`x \in [0, 1]^{1 \times 28 \times 28}`],
    why: 'MNIST được làm đúng như vậy, nên mạng chỉ quen với chữ số cỡ đó, nằm giữa ảnh. Bỏ bước căn giữa thì một chữ số vẽ lệch về một góc dễ bị đoán sai. Ô "Mạng nhìn thấy" trong bảng là đúng 784 số này.',
    code: `x = torch.tensor(pixels, dtype=torch.float32).view(1, 1, 28, 28) / 255`
  }, { tag: 'Ảnh', size: [0.06, sp(28), sp(28)], live: 'cnnIn' }),
  conv('Conv 3×3, 8 kênh', 1, 8, 28, 'cnnC1',
    'Tám bộ lọc 3×3 trượt trên ảnh. Mỗi bộ lọc cho ra một bản đồ đặc trưng 26×26, sáng ở chỗ ảnh có đúng kiểu nét mà bộ lọc đã học: nét ngang, nét dọc, mép trái, mép phải…',
    'Bản đồ trong bảng là activation thật sau ReLU. Vẽ một nét ngang rồi một nét dọc: các kênh sáng lên khác nhau. Lớp đầu chỉ nhìn một ô 3×3 pixel nên chỉ học được những đoạn nét rất ngắn.'),
  relu('8×26×26', slab(26)),
  pool(1, 8, 26),
  conv('Conv 3×3, 16 kênh', 8, 16, 13, 'cnnC2',
    'Mười sáu bộ lọc, mỗi bộ lọc nhìn cả 8 kênh của lớp trước trong một ô 3×3. Vì lớp trước đã thu nhỏ một nửa, mỗi điểm ở đây ứng với một vùng 8×8 pixel của ảnh gốc: đủ để ghép các nét ngắn thành góc, đoạn cong, chỗ giao nhau.',
    R`Vùng nhìn (receptive field) lớn dần qua từng lớp: \(3 \to 4\) (sau pool) \(\to 4 + 2 \times 2 = 8\) pixel. Mạng sâu hơn nhìn được cả chữ số; ở đây lớp Linear cuối làm phần ghép đó.`),
  relu('16×11×11', slab(11)),
  pool(2, 16, 11),
  n('Duỗi phẳng', 'shape', '16×5×5 → 400', {
    what: 'Xếp 16 bản đồ 5×5 thành một vector 400 số, hết kênh này mới đến kênh kia: kênh 1 hàng 1, kênh 1 hàng 2, …, rồi kênh 2.',
    why: 'Thứ tự phải đúng như lúc train. Xếp hàng trước kênh sau thì cùng trọng số cho kết quả sai hoàn toàn; test của trang kiểm tra đúng điều này bằng cách so logit với lúc train.',
    code: 'f = torch.flatten(x, 1)   # (B, 400)'
  }, { tag: 'Flatten', size: SLAB }),
  n('Linear 400 → 10', 'linear', '400 → 10', {
    what: 'Mỗi chữ số có một hàng 400 trọng số. Điểm (logit) của chữ số d là tích vô hướng của hàng đó với vector 400 đặc trưng, cộng bias.',
    formula: [R`z_d = \sum_{k=1}^{400} W_{d,k}\, f_k + b_d, \qquad d = 0, \dots, 9`],
    why: `Chiếm ${fmtText(P.fc)} trên ${fmtText(W.params)} tham số. Hai lớp conv trích đặc trưng; lớp này chỉ còn phải cân các đặc trưng đó cho từng chữ số.`,
    params: R`\(10 \times 400 + 10 = ${fmt(P.fc)}\).`,
    code: 'z = self.fc(f)   # nn.Linear(400, 10)'
  }, { tag: 'Linear', size: [0.5, 1.4, 1.2], live: 'cnnOut' }),
  n('Softmax', 'act', '10', {
    what: 'Đổi 10 logit thành 10 xác suất cộng lại bằng 1. Chữ số có xác suất cao nhất là câu trả lời của mạng.',
    formula: [R`p_d = \frac{e^{z_d}}{\sum_{j=0}^{9} e^{z_j}}`],
    why: 'Khi train, loss là cross-entropy: −log của xác suất gán cho chữ số đúng. Xác suất cao không có nghĩa là chắc đúng: vẽ một thứ chẳng giống chữ số nào, mạng vẫn phải chọn một chữ số và đôi khi còn rất tự tin.',
    code: 'p = z.softmax(-1)'
  }, { size: SLAB })
] });
})();
